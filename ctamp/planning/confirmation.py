"""Confirmation step for CTAMP solution validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..domain.models import Action, Edge


@dataclass
class CompletePlan:
    actions: List[Action]
    edges: List[Edge]
    total_cost: float

    @property
    def num_steps(self) -> int:
        return len(self.actions)


@dataclass
class EmptyPlan:
    reason: str = ""
    failed_edge: Optional[Edge] = None


def confirm_solution(path_edges: List[Edge]) -> CompletePlan | EmptyPlan:
    """Validate that every edge in the path has a motion plan.

    Returns CompletePlan if all edges are valid, EmptyPlan otherwise.
    """
    total_cost = 0.0
    actions: List[Action] = []
    valid_edges: List[Edge] = []

    for edge in path_edges:
        if edge.motion_plan is None or not edge.motion_plan.success:
            return EmptyPlan(
                reason=f"edge {edge.source}->{edge.target} has no valid motion plan",
                failed_edge=edge,
            )
        actions.append(edge.action)
        valid_edges.append(edge)
        total_cost += edge.cost

    return CompletePlan(actions=actions, edges=valid_edges, total_cost=total_cost)
