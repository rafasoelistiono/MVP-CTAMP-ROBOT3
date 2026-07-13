"""Performance-oriented CTAMP scene runner v2.

V2 keeps the v1 runner intact and applies only run-local optimizations that
should not change object ordering or success semantics.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..domain.models import MotionPlan
from ..motion_planning.mujoco import MuJoCoMotionPlanner
from ..simulation.mujoco_backend import MuJoCoBackend
from ..simulation.panda_physics import PandaPhysicsExecutor
from . import run_scene as v1


def _copy_motion_plan(plan: MotionPlan) -> MotionPlan:
    return plan.model_copy(deep=True)


@contextmanager
def _v2_runtime_patches(stats: dict[str, Any]):
    original_plan_xy = MuJoCoMotionPlanner.plan_xy
    original_step = MuJoCoBackend.step
    original_settle = PandaPhysicsExecutor.settle
    cache: dict[tuple[int, float, float, float, float], MotionPlan] = {}

    def cached_plan_xy(
        self: MuJoCoMotionPlanner,
        start: Sequence[float],
        goal: Sequence[float],
    ) -> MotionPlan:
        key = (
            id(self),
            round(float(start[0]), 4),
            round(float(start[1]), 4),
            round(float(goal[0]), 4),
            round(float(goal[1]), 4),
        )
        stats["plan_xy_calls"] += 1
        if key in cache:
            stats["plan_xy_cache_hits"] += 1
            return _copy_motion_plan(cache[key])

        stats["plan_xy_cache_misses"] += 1
        started = time.perf_counter()
        plan = original_plan_xy(self, start, goal)
        stats["xy_plan_time"] += time.perf_counter() - started
        cache[key] = _copy_motion_plan(plan)
        return plan

    def batched_step(self: MuJoCoBackend, n: int = 1) -> None:
        self._require_loaded()
        n = int(n)
        if n <= 0:
            return
        if n == 1:
            stats["mujoco_step_calls"] += 1
            original_step(self, n)
            return
        stats["mujoco_step_calls"] += 1
        stats["mujoco_step_batched_calls"] += 1
        stats["mujoco_step_batched_steps"] += n
        self._mujoco().mj_step(self.model, self.data, nstep=n)

    def batched_settle(self: PandaPhysicsExecutor, steps: int = 200) -> None:
        steps = int(steps)
        if steps <= 0:
            return
        if self.viewer is not None:
            original_settle(self, steps)
            return
        stats["physics_settle_batched_calls"] += 1
        stats["physics_settle_batched_steps"] += steps
        self.mujoco.mj_step(self.model, self.data, nstep=steps)

    MuJoCoMotionPlanner.plan_xy = cached_plan_xy
    MuJoCoBackend.step = batched_step
    PandaPhysicsExecutor.settle = batched_settle
    try:
        yield
    finally:
        MuJoCoMotionPlanner.plan_xy = original_plan_xy
        MuJoCoBackend.step = original_step
        PandaPhysicsExecutor.settle = original_settle


def run(
    config_path: Path,
    output: Path,
    max_retries: int | None = None,
    max_objects: int | None = None,
    project_root: Path | None = None,
    viewer: bool = False,
) -> dict:
    stats: dict[str, Any] = {
        "plan_xy_calls": 0,
        "plan_xy_cache_hits": 0,
        "plan_xy_cache_misses": 0,
        "xy_plan_time": 0.0,
        "mujoco_step_calls": 0,
        "mujoco_step_batched_calls": 0,
        "mujoco_step_batched_steps": 0,
        "physics_settle_batched_calls": 0,
        "physics_settle_batched_steps": 0,
    }
    with _v2_runtime_patches(stats):
        metrics = v1.run(
            config_path,
            output,
            max_retries=max_retries,
            max_objects=max_objects,
            project_root=project_root,
            viewer=viewer,
        )
    metrics["ctamp_version"] = "v2"
    metrics["performance_v2"] = stats
    v1._write_json(output / "metrics.json", metrics)
    return metrics
