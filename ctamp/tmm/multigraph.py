"""TaskMotionMultigraph: directed multigraph for CTAMP search space."""

from __future__ import annotations

from typing import List, Optional

import networkx as nx

from ..domain.models import Edge, Vertex


class TaskMotionMultigraph:
    """Directed multigraph where vertices are states and edges are actions.

    Supports multiple edges between the same source-target pair, which is
    needed when different joint-space / motion-plan options connect the
    same two task vertices.
    """

    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def add_vertex(self, vertex: Vertex) -> None:
        self._graph.add_node(
            vertex.vertex_id,
            vertex=vertex,
            is_root=vertex.is_root,
            is_goal=vertex.is_goal,
        )

    def add_edge(self, edge: Edge) -> None:
        self._graph.add_edge(
            edge.source,
            edge.target,
            key=f"{edge.action.action_id}@{edge.joint_space.name}",
            edge=edge,
        )

    def get_outgoing_edges(self, vertex_id: str) -> List[Edge]:
        if vertex_id not in self._graph:
            return []
        result: List[Edge] = []
        for _, _, data in self._graph.out_edges(vertex_id, data=True):
            result.append(data["edge"])
        return result

    def get_edges_between(self, source_id: str, target_id: str) -> List[Edge]:
        result: List[Edge] = []
        if not self._graph.has_node(source_id) or not self._graph.has_node(target_id):
            return result
        for _, data in self._graph[source_id].get(target_id, {}).items():
            result.append(data["edge"])
        return result

    def get_root(self) -> Optional[Vertex]:
        for _, data in self._graph.nodes(data=True):
            if data.get("is_root"):
                return data["vertex"]
        return None

    def get_goal_vertices(self) -> List[Vertex]:
        goals: List[Vertex] = []
        for _, data in self._graph.nodes(data=True):
            if data.get("is_goal"):
                goals.append(data["vertex"])
        return goals

    def get_path_edges(self, path_vertex_ids: List[str]) -> List[Edge]:
        """Return the first edge between each consecutive pair of vertices."""
        if len(path_vertex_ids) < 2:
            return []
        edges: List[Edge] = []
        for src, tgt in zip(path_vertex_ids, path_vertex_ids[1:]):
            between = self.get_edges_between(src, tgt)
            if between:
                edges.append(between[0])
        return edges

    @property
    def vertex_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def is_directed(self) -> bool:
        return True

    @property
    def is_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self._graph)
