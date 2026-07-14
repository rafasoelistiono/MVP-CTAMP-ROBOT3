"""Heuristic path estimator using learned models."""

from __future__ import annotations

import heapq
from typing import Optional, Set


from ..domain.models import Edge, Vertex
from ..learning.heuristic_models import LearnedModel
from ..learning.path_features import PathFeatureConfig, PathFeatureExtractor
from ..tmm.multigraph import TaskMotionMultigraph
from .tmm_astar import TMMHeuristic


class HeuristicPathEstimator(TMMHeuristic):
    """Estimate h(v) by finding shortest unit-cost path to goal, extracting features, and predicting cost."""

    def __init__(
        self,
        model: Optional[LearnedModel] = None,
        feature_config: Optional[PathFeatureConfig] = None,
    ) -> None:
        self.model = model
        self.extractor = PathFeatureExtractor(feature_config)
        self._graph: Optional[TaskMotionMultigraph] = None
        self._untrained_baseline = 1.0

    def set_graph(self, graph: TaskMotionMultigraph) -> None:
        self._graph = graph

    def evaluate(self, vertex: Vertex, goal_ids: Set[str]) -> float:
        if vertex.vertex_id in goal_ids:
            return 0.0

        if self.model is None:
            return self._untrained_baseline

        path_edges = self._find_shortest_path(vertex, goal_ids)
        if path_edges is None:
            return float("inf")

        features = self.extractor.extract(vertex, path_edges, {})
        return self._clip(self.model.predict(features))

    def _find_shortest_path(
        self, start: Vertex, goal_ids: Set[str]
    ) -> Optional[list[Edge]]:
        if self._graph is None:
            return None

        open_set: list[tuple[float, str, list[Edge]]] = [(0.0, start.vertex_id, [])]
        visited: Set[str] = set()

        while open_set:
            cost, vid, path = heapq.heappop(open_set)

            if vid in visited:
                continue
            visited.add(vid)

            if vid in goal_ids and vid != start.vertex_id:
                return path

            for edge in self._graph.get_outgoing_edges(vid):
                if edge.target not in visited:
                    heapq.heappush(open_set, (cost + 1.0, edge.target, path + [edge]))

        return None

    @staticmethod
    def _clip(value: float) -> float:
        return max(value, 1e-6)
