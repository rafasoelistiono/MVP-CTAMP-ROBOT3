from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from backends.adaptive.hint_cache import ExecutionHints
from task_planning.types import Step


@dataclass(frozen=True)
class PrimitiveResult:
    """Diagnostic result only; physical success is decided by the verifier."""

    completed: bool
    failure_reason: str | None = None


class PrimitiveExecutor(Protocol):
    def execute(
        self,
        step: Step,
        target: tuple[float, float, float] | None,
        hints: ExecutionHints,
    ) -> PrimitiveResult: ...

    def object_pose(self, object_id: str) -> tuple[float, float, float]: ...

    def all_object_poses(self) -> dict[str, tuple[float, float, float]]: ...

    def held_object_name(self) -> str | None: ...

    def object_orientation(self, object_id: str) -> tuple[float, float, float, float]: ...

    def object_velocity(
        self,
        object_id: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]: ...

    def object_vertical_half_extent(self, object_id: str) -> float: ...

    def settle_for_verification(self, steps: int) -> None: ...


class MuJoCoExecutorPrimitives:
    """Adapter from generic task primitives to the MuJoCo IK/OMPL backend."""

    def __init__(self, executor_module: Any):
        self.executor = executor_module

    def execute(
        self,
        step: Step,
        target: tuple[float, float, float] | None,
        hints: ExecutionHints,
    ) -> PrimitiveResult:
        old_backend = getattr(self.executor, "_IK_BACKEND_NAME", None)
        old_plan_tolerance = getattr(self.executor, "IK_PLAN_POS_ERR_LIMIT", None)
        try:
            self._apply_hints(step, hints)
            if step.action == "pick":
                self.executor.pick(step.object)
            elif step.action == "place":
                if target is None:
                    return PrimitiveResult(False, "missing_place_target")
                self.executor.place(
                    target[0],
                    target[1],
                    step.object,
                    target_z=target[2],
                    release_lift=self.executor.CONFIG.grasp.place_release_lift_m,
                )
            else:
                return PrimitiveResult(False, f"unsupported_action:{step.action}")
        except RuntimeError as exc:
            return PrimitiveResult(False, str(exc) or exc.__class__.__name__)
        except Exception as exc:
            return PrimitiveResult(False, f"{exc.__class__.__name__}:{exc}")
        finally:
            if old_backend is not None:
                self.executor._IK_BACKEND_NAME = old_backend
            if old_plan_tolerance is not None:
                self.executor.IK_PLAN_POS_ERR_LIMIT = old_plan_tolerance
        return PrimitiveResult(True)

    def _apply_hints(self, step: Step, hints: ExecutionHints) -> None:
        if (
            hints.ik_backend == "mujoco_dls"
            and hasattr(self.executor, "_IK_BACKEND_NAME")
        ):
            self.executor._IK_BACKEND_NAME = hints.ik_backend
        if hasattr(self.executor, "IK_PLAN_POS_ERR_LIMIT"):
            self.executor.IK_PLAN_POS_ERR_LIMIT = min(
                float(hints.ik_position_tolerance), 0.030
            )
        if step.action == "pick" and hasattr(self.executor, "_pick_call_counts"):
            profile_index = {
                "default_cube": 0,
                "side_cylinder": 0,
                "far_reach_cube": 1,
                "retry_cube": 1,
            }.get(hints.grasp_profile)
            if (
                profile_index is not None
                and step.object not in self.executor._pick_call_counts
            ):
                self.executor._pick_call_counts[step.object] = profile_index

    def object_pose(self, object_id: str) -> tuple[float, float, float]:
        body_id = self.executor.name_to_cube[object_id]
        self.executor.mujoco.mj_forward(self.executor.model, self.executor.data)
        pose = self.executor.data.xpos[body_id]
        return float(pose[0]), float(pose[1]), float(pose[2])

    def all_object_poses(self) -> dict[str, tuple[float, float, float]]:
        return {
            object_id: self.object_pose(object_id)
            for object_id in self.executor.name_to_cube
        }

    def held_object_name(self) -> str | None:
        return getattr(self.executor, "_held_object_name", None)

    def object_orientation(self, object_id: str) -> tuple[float, float, float, float]:
        body_id = self.executor.name_to_cube[object_id]
        self.executor.mujoco.mj_forward(self.executor.model, self.executor.data)
        quat = self.executor.data.xquat[body_id]
        return tuple(float(value) for value in quat)

    def object_velocity(
        self,
        object_id: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        body_id = self.executor.name_to_cube[object_id]
        self.executor.mujoco.mj_forward(self.executor.model, self.executor.data)
        spatial = self.executor.data.cvel[body_id]
        angular = tuple(float(value) for value in spatial[:3])
        linear = tuple(float(value) for value in spatial[3:])
        return linear, angular

    def object_vertical_half_extent(self, object_id: str) -> float:
        """Return the live world-Z half extent of an object's box geometry."""
        body_id = self.executor.name_to_cube[object_id]
        self.executor.mujoco.mj_forward(self.executor.model, self.executor.data)
        best = 0.0
        for geom_id in range(self.executor.model.ngeom):
            if int(self.executor.model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(self.executor.model.geom_type[geom_id]) != int(
                self.executor.mujoco.mjtGeom.mjGEOM_BOX
            ):
                continue
            matrix = self.executor.data.geom_xmat[geom_id]
            size = self.executor.model.geom_size[geom_id]
            extent = sum(abs(float(matrix[6 + axis])) * float(size[axis]) for axis in range(3))
            best = max(best, extent)
        if best <= 0.0:
            raise RuntimeError(f"box_geometry_not_found:{object_id}")
        return best

    def settle_for_verification(self, steps: int) -> None:
        settle = getattr(self.executor, "_step_sim", None)
        if callable(settle) and steps > 0:
            settle(int(steps))
