"""SampleCollector: collect training samples from TMM search expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..domain.models import Edge, Vertex
from ..tmm.multigraph import TaskMotionMultigraph
from .path_features import PathFeatureConfig, PathFeatureExtractor


@dataclass
class Sample:
    features: np.ndarray
    cost: float


class SampleCollector:
    """Collect (features, cost) training samples from a TMM graph.

    When a vertex is expanded, enumerate paths from that vertex to goal
    vertices.  For each path, extract features and sum edge costs as target.
    """

    def __init__(self, feature_config: Optional[PathFeatureConfig] = None) -> None:
        self.extractor = PathFeatureExtractor(feature_config)
        self._samples: List[Sample] = []
        self._seen: set[tuple[str, str]] = set()

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def collect(
        self,
        expanded_vertex: Vertex,
        graph: TaskMotionMultigraph,
        target_poses: dict,
        goal_ids: Optional[set[str]] = None,
        max_depth: int = 5,
    ) -> int:
        """Collect samples from expanded_vertex to all reachable goals.

        Returns number of new samples collected.
        """
        if goal_ids is None:
            goal_ids = {v.vertex_id for v in graph.get_goal_vertices()}

        before = self._sample_count
        self._dfs(expanded_vertex, graph, target_poses, goal_ids, [], 0, max_depth)
        return self._sample_count - before

    def _dfs(
        self,
        current: Vertex,
        graph: TaskMotionMultigraph,
        target_poses: dict,
        goal_ids: set[str],
        path_edges: List[Edge],
        current_cost: float,
        depth_left: int,
    ) -> None:
        if depth_left <= 0:
            return

        if current.vertex_id in goal_ids and path_edges:
            key = (path_edges[0].source, current.vertex_id)
            if key not in self._seen:
                self._seen.add(key)
                features = self.extractor.extract(current, path_edges, target_poses)
                self._samples.append(Sample(features=features, cost=current_cost))

        for edge in graph.get_outgoing_edges(current.vertex_id):
            if edge.motion_plan is None or not edge.motion_plan.success:
                continue
            next_v = self._get_vertex(graph, edge.target)
            if next_v is None:
                continue
            self._dfs(
                next_v,
                graph,
                target_poses,
                goal_ids,
                path_edges + [edge],
                current_cost + edge.cost,
                depth_left - 1,
            )

    def _get_vertex(self, graph: TaskMotionMultigraph, vid: str) -> Optional[Vertex]:
        data = graph._graph.nodes.get(vid)
        if data is None:
            return None
        return data.get("vertex")

    def get_samples(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._samples:
            return np.array([]), np.array([])
        X = np.stack([s.features for s in self._samples])
        y = np.array([s.cost for s in self._samples], dtype=np.float32)
        return X, y

    def save_npz(self, path: str) -> None:
        X, y = self.get_samples()
        np.savez_compressed(path, X=X, y=y)

    def save_csv(self, path: str) -> None:
        X, y = self.get_samples()
        if X.size == 0:
            return
        data = np.column_stack([X, y])
        np.savetxt(path, data, delimiter=",")

    def load_npz(self, path: str) -> None:
        data = np.load(path)
        X, y = data["X"], data["y"]
        self._samples = [Sample(features=X[i], cost=float(y[i])) for i in range(len(y))]
        self._seen.clear()

    @property
    def _sample_count(self) -> int:
        return len(self._samples)
