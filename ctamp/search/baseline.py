"""Baseline planner using uniform-cost search (h=0) for benchmarking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..domain.models import Edge
from ..tmm.multigraph import TaskMotionMultigraph
from .motion_visitor import MotionPlanningVisitor
from .tmm_astar import (
    TMMEdgeCostCalculator,
    TMMAStar,
    TMMSearchResult,
    ZeroTMMHeuristic,
)


@dataclass
class BaselineResult:
    success: bool
    cost: float
    nodes_expanded: int
    nodes_generated: int
    time_elapsed: float
    path_vertex_ids: list[str] = field(default_factory=list)
    path_edges: list[Edge] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "cost": self.cost,
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "time_elapsed": self.time_elapsed,
            "path_vertex_ids": self.path_vertex_ids,
            "error": self.error,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


class BaselinePlanner:
    """UCS baseline: h(v)=0, same motion planner and cost calculator."""

    def __init__(
        self,
        visitor: Optional[MotionPlanningVisitor] = None,
        cost_calculator: Optional[TMMEdgeCostCalculator] = None,
        max_nodes: int = 100000,
        max_time: float = 60.0,
    ) -> None:
        self.visitor = visitor
        self.cost_calculator = cost_calculator or TMMEdgeCostCalculator()
        self.max_nodes = max_nodes
        self.max_time = max_time

    def search(self, graph: TaskMotionMultigraph) -> BaselineResult:
        astar = TMMAStar(
            heuristic=ZeroTMMHeuristic(),
            visitor=self.visitor,
            cost_calculator=self.cost_calculator,
            max_nodes=self.max_nodes,
            max_time=self.max_time,
        )
        result: TMMSearchResult = astar.search(graph)

        path_edges = result.path_edges
        total_cost = result.cost if result.success else float("inf")

        return BaselineResult(
            success=result.success,
            cost=total_cost,
            nodes_expanded=result.nodes_expanded,
            nodes_generated=result.nodes_generated,
            time_elapsed=result.time_elapsed,
            path_vertex_ids=result.path_vertex_ids,
            path_edges=path_edges,
            error=result.error,
        )
