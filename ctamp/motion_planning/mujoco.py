"""MuJoCo scene-aware geometric motion-planner adapter.

This adapter intentionally performs a 2-D end-effector target probe, not Panda
inverse kinematics or joint-space trajectory planning.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from ..domain.models import MotionPlan
from ..simulation.scene import MotionProbe


class MuJoCoMotionPlanner:
    def __init__(self, config: dict, clearance: float = 0.055) -> None:
        self.probe = MotionProbe(config, clearance=clearance)

    def plan_xy(self, start: Sequence[float], goal: Sequence[float]) -> MotionPlan:
        started = time.perf_counter()
        result = self.probe.probe((float(start[0]), float(start[1])),
                                  (float(goal[0]), float(goal[1])))
        return MotionPlan(
            success=result.success,
            waypoints=[list(p) for p in result.waypoints],
            length=result.length,
            smoothness=1.0 if len(result.waypoints) == 2 else 0.8,
            clearance=result.clearance,
            planning_time=time.perf_counter() - started,
            iterations=max(1, len(result.waypoints) - 1),
            metadata={"route_type": result.route_type, "reason": result.reason,
                      "validation_level": "geometric_2d_probe"},
        )
