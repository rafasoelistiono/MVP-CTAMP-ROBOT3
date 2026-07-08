"""Edge cost calculator for CTAMP based on result + process cost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..domain.models import Edge, MotionPlan
from ..tmm.builder import DEFAULT_EDGE_COST


@dataclass
class CostWeights:
    smoothness: float = 1.0
    length: float = 1.0
    clearance: float = 1.0
    iterations: float = 0.1
    joint_dim: float = 0.5
    planning_time: float = 0.1
    failure_penalty: float = DEFAULT_EDGE_COST


class EdgeCostCalculator:
    """Compute edge cost as result_cost + process_cost.

    On success, cost = weighted(smoothness, length, clearance) + weighted(iterations, joint_dim, time).
    On failure, cost = failure_penalty.
    """

    def __init__(self, weights: Optional[CostWeights] = None) -> None:
        self.weights = weights or CostWeights()

    def compute(self, edge: Edge, motion_plan: Optional[MotionPlan] = None) -> float:
        if motion_plan is None or not motion_plan.success:
            return self.weights.failure_penalty

        result_cost = (
            self.weights.smoothness * (1.0 - motion_plan.smoothness)
            + self.weights.length * motion_plan.length
            + self.weights.clearance * (1.0 / (1.0 + motion_plan.clearance))
        )

        process_cost = (
            self.weights.iterations * motion_plan.iterations
            + self.weights.joint_dim * len(edge.joint_space.joints)
            + self.weights.planning_time * motion_plan.planning_time
        )

        return max(0.0, result_cost + process_cost)
