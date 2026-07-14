"""A* search for TaskMotionMultigraph."""

from __future__ import annotations

import heapq
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Set

from ..domain.models import Edge, MotionPlan, Vertex
from ..tmm.multigraph import TaskMotionMultigraph


class TMMHeuristic(ABC):
    @abstractmethod
    def evaluate(self, vertex: Vertex, goal_ids: Set[str]) -> float:
        pass


class ZeroTMMHeuristic(TMMHeuristic):
    def evaluate(self, vertex: Vertex, goal_ids: Set[str]) -> float:
        return 0.0


class GoalDistanceHeuristic(TMMHeuristic):
    def __init__(self, distance_scale: float = 1.0) -> None:
        self.scale = distance_scale

    def evaluate(self, vertex: Vertex, goal_ids: Set[str]) -> float:
        return 0.0


class SearchVisitor(ABC):
    @abstractmethod
    def on_expand(self, graph: TaskMotionMultigraph, vertex: Vertex) -> None:
        pass

    @abstractmethod
    def on_edge(self, edge: Edge, motion_plan: Optional[MotionPlan]) -> None:
        pass


class MockVisitor(SearchVisitor):
    def __init__(
        self, planner: Optional[object] = None, workspace: Optional[object] = None
    ) -> None:
        self.planner = planner
        self.workspace = workspace
        self.expanded: List[str] = []
        self.edge_results: List[tuple] = []

    def on_expand(self, graph: TaskMotionMultigraph, vertex: Vertex) -> None:
        self.expanded.append(vertex.vertex_id)

    def on_edge(self, edge: Edge, motion_plan: Optional[MotionPlan]) -> None:
        self.edge_results.append((edge, motion_plan))


@dataclass(order=True)
class TMMNode:
    f: float
    g: float = field(compare=False)
    vertex_id: str = field(compare=False)
    parent: Optional[TMMNode] = field(default=None, compare=False)
    edge_from_parent: Optional[Edge] = field(default=None, compare=False)


@dataclass
class TMMSearchResult:
    success: bool
    goal_node: Optional[TMMNode] = None
    nodes_expanded: int = 0
    nodes_generated: int = 0
    time_elapsed: float = 0.0
    cost: float = float("inf")
    error: Optional[str] = None

    @property
    def path_vertex_ids(self) -> List[str]:
        if self.goal_node is None:
            return []
        path = []
        node: Optional[TMMNode] = self.goal_node
        while node is not None:
            path.append(node.vertex_id)
            node = node.parent
        return list(reversed(path))

    @property
    def path_edges(self) -> List[Edge]:
        if self.goal_node is None:
            return []
        edges = []
        node: Optional[TMMNode] = self.goal_node
        while node is not None and node.edge_from_parent is not None:
            edges.append(node.edge_from_parent)
            node = node.parent
        return list(reversed(edges))


class TMMEdgeCostCalculator:
    def compute(self, edge: Edge) -> float:
        return edge.cost


class TMMAStar:
    def __init__(
        self,
        heuristic: Optional[TMMHeuristic] = None,
        visitor: Optional[SearchVisitor] = None,
        cost_calculator: Optional[TMMEdgeCostCalculator] = None,
        max_nodes: int = 100000,
        max_time: float = 60.0,
    ) -> None:
        self.heuristic = heuristic or ZeroTMMHeuristic()
        self.visitor = visitor or MockVisitor()
        self.cost_calc = cost_calculator or TMMEdgeCostCalculator()
        self.max_nodes = max_nodes
        self.max_time = max_time

    def search(self, graph: TaskMotionMultigraph) -> TMMSearchResult:
        root = graph.get_root()
        if root is None:
            return TMMSearchResult(success=False, error="no_root")

        goal_ids = {v.vertex_id for v in graph.get_goal_vertices()}
        start_time = time.time()

        start_h = self.heuristic.evaluate(root, goal_ids)
        start_node = TMMNode(f=start_h, g=0.0, vertex_id=root.vertex_id)
        open_set = [start_node]
        closed_set: Set[str] = set()
        nodes_generated = 1
        nodes_expanded = 0

        while open_set:
            if time.time() - start_time > self.max_time:
                return TMMSearchResult(
                    success=False,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_elapsed=time.time() - start_time,
                    error="timeout",
                )

            current = heapq.heappop(open_set)
            if current.vertex_id in closed_set:
                continue
            closed_set.add(current.vertex_id)

            if current.vertex_id in goal_ids:
                return TMMSearchResult(
                    success=True,
                    goal_node=current,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_elapsed=time.time() - start_time,
                    cost=current.g,
                )

            nodes_expanded += 1
            if nodes_expanded >= self.max_nodes:
                return TMMSearchResult(
                    success=False,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_elapsed=time.time() - start_time,
                    error="max_nodes",
                )

            vertex = self._get_vertex(graph, current.vertex_id)
            if vertex is None:
                continue
            self.visitor.on_expand(graph, vertex)

            for edge in graph.get_outgoing_edges(current.vertex_id):
                if edge.target in closed_set:
                    continue

                self.visitor.on_edge(edge, edge.motion_plan)
                cost = self.cost_calc.compute(edge)
                g = current.g + cost
                h = self.heuristic.evaluate(
                    self._get_vertex(graph, edge.target) or vertex, goal_ids
                )
                child = TMMNode(
                    f=g + h,
                    g=g,
                    vertex_id=edge.target,
                    parent=current,
                    edge_from_parent=edge,
                )
                heapq.heappush(open_set, child)
                nodes_generated += 1

        return TMMSearchResult(
            success=False,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            time_elapsed=time.time() - start_time,
            error="no_solution",
        )

    def _get_vertex(self, graph: TaskMotionMultigraph, vid: str) -> Optional[Vertex]:
        data = graph._graph.nodes.get(vid)
        if data is None:
            return None
        return data.get("vertex")
