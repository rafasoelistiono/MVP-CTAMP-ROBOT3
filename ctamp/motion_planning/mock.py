"""MockMotionPlanner: deterministic 2D tabletop motion planner for testing."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from ..domain.models import Edge, MotionPlan, ObjectState, Pose, WorkspaceState


@dataclass
class MockPlannerConfig:
    max_distance: float = 5.0
    failure_probability: float = 0.0
    obstacle_radius: float = 0.2
    planning_time_base: float = 0.01
    iterations_per_meter: int = 10


class MockMotionPlanner:
    """2D tabletop motion planner for testing.

    Plans straight-line paths in 2D.  Failure triggers:
    - target object overlaps with an obstacle
    - straight-line distance exceeds max_distance
    - random failure probability fires
    """

    def __init__(
        self,
        config: Optional[MockPlannerConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or MockPlannerConfig()
        self._rng = random.Random(seed)

    def plan(self, edge: Edge, workspace: WorkspaceState) -> Optional[MotionPlan]:
        src_obj = workspace.objects.get(edge.source)
        tgt_obj = workspace.objects.get(edge.target)

        if src_obj is None or tgt_obj is None:
            return None

        if self._is_obstructed(tgt_obj, workspace):
            return None

        dist = self._distance(src_obj.pose, tgt_obj.pose)
        if dist > self.config.max_distance:
            return None

        if self._rng.random() < self.config.failure_probability:
            return None

        waypoints = self._generate_waypoints(src_obj.pose, tgt_obj.pose)
        planning_time = self.config.planning_time_base + dist * 0.005
        iterations = max(1, int(dist * self.config.iterations_per_meter))

        return MotionPlan(
            waypoints=waypoints,
            length=dist,
            smoothness=self._smoothness(waypoints),
            clearance=self._clearance(tgt_obj, workspace),
            planning_time=planning_time,
            iterations=iterations,
            success=True,
        )

    def _is_obstructed(self, target: ObjectState, workspace: WorkspaceState) -> bool:
        for oid, obj in workspace.objects.items():
            if oid == target.object_id:
                continue
            if self._distance(target.pose, obj.pose) < self.config.obstacle_radius:
                return True
        return False

    def _distance(self, a: Pose, b: Pose) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _generate_waypoints(self, src: Pose, tgt: Pose) -> list[list[float]]:
        steps = max(2, int(self._distance(src, tgt) * 10))
        return [
            [
                src.x + (tgt.x - src.x) * i / (steps - 1),
                src.y + (tgt.y - src.y) * i / (steps - 1),
            ]
            for i in range(steps)
        ]

    def _smoothness(self, waypoints: list[list[float]]) -> float:
        if len(waypoints) < 3:
            return 1.0
        total_curvature = 0.0
        for i in range(1, len(waypoints) - 1):
            dx1 = waypoints[i][0] - waypoints[i - 1][0]
            dy1 = waypoints[i][1] - waypoints[i - 1][1]
            dx2 = waypoints[i + 1][0] - waypoints[i][0]
            dy2 = waypoints[i + 1][1] - waypoints[i][1]
            cross = abs(dx1 * dy2 - dy1 * dx2)
            total_curvature += cross
        return 1.0 / (1.0 + total_curvature)

    def _clearance(self, target: ObjectState, workspace: WorkspaceState) -> float:
        min_dist = float("inf")
        for oid, obj in workspace.objects.items():
            if oid == target.object_id:
                continue
            d = self._distance(target.pose, obj.pose)
            if d < min_dist:
                min_dist = d
        if min_dist == float("inf"):
            return 1.0
        return min_dist
