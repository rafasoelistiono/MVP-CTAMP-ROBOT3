from __future__ import annotations

import time
from typing import Any

from task_planning.types import ProbePlanResult, ProbeResult, TaskPlan
from world.state import WorldState


def requires_motion_probe(world: WorldState) -> bool:
    """Return True if the world challenge requires mandatory motion probing."""
    challenge = world.challenge
    return bool(challenge and challenge.enabled and challenge.require_motion_probe)


class MotionProbe:
    """Probes motion feasibility without committing to full execution.

    Uses the MuJoCo backend's IK solver and OMPL planner in dry-run mode
    when available. Falls back to geometric feasibility checks when the
    backend is not initialized or dry-run is not supported.
    """

    def __init__(
        self,
        runtime: Any = None,
        primitives: Any = None,
        hint_cache: Any = None,
    ):
        self._runtime = runtime
        self._primitives = primitives
        self._hint_cache = hint_cache
        self._backend_available = runtime is not None
        self._pick_cache: dict[str, ProbeResult] = {}
        self._place_cache: dict[str, ProbeResult] = {}

    def probe_pick_feasibility(
        self,
        world: WorldState,
        object_id: str,
    ) -> ProbeResult:
        if object_id in self._pick_cache:
            return self._pick_cache[object_id]
        obj = world.object_by_id(object_id)
        if obj is None:
            return ProbeResult(
                feasible=False,
                ik_success=False,
                ompl_success=False,
                failure_reason=f"unknown object {object_id!r}",
            )
        if not obj.reachable:
            return ProbeResult(
                feasible=False,
                ik_success=False,
                ompl_success=False,
                failure_reason=f"object {object_id!r} not reachable",
            )
        if not self._backend_available:
            result = self._geometric_pick_probe(world, obj)
        else:
            result = self._backend_pick_probe(world, object_id, obj)
        self._pick_cache[object_id] = result
        return result

    def probe_place_feasibility(
        self,
        world: WorldState,
        object_id: str,
        slot_id: str,
        slots: dict[str, tuple[float, float, float]],
    ) -> ProbeResult:
        if slot_id in self._place_cache:
            return self._place_cache[slot_id]
        slot_pos = slots.get(slot_id)
        if slot_pos is None:
            return ProbeResult(
                feasible=False,
                ik_success=False,
                ompl_success=False,
                failure_reason=f"unknown slot {slot_id!r}",
            )
        if not self._backend_available:
            result = self._geometric_place_probe(world, slot_pos)
        else:
            result = self._backend_place_probe(world, object_id, slot_pos)
        self._place_cache[slot_id] = result
        return result

    def probe_align_edge(
        self,
        world: WorldState,
        object_id: str,
        slot_id: str,
        slots: dict[str, tuple[float, float, float]],
    ) -> ProbeResult:
        pick_result = self.probe_pick_feasibility(world, object_id)
        if not pick_result.feasible:
            return pick_result
        place_result = self.probe_place_feasibility(world, object_id, slot_id, slots)
        if not place_result.feasible:
            return place_result
        return ProbeResult(
            feasible=True,
            ik_success=pick_result.ik_success and place_result.ik_success,
            ompl_success=pick_result.ompl_success and place_result.ompl_success,
            planning_time=pick_result.planning_time + place_result.planning_time,
            estimated_path_length=(
                pick_result.estimated_path_length + place_result.estimated_path_length
            ),
            min_clearance=min(
                pick_result.min_clearance, place_result.min_clearance
            ),
            collision_count=pick_result.collision_count + place_result.collision_count,
            grasp_diagnostics=pick_result.grasp_diagnostics,
            attempted_orientations=pick_result.attempted_orientations,
            attempted_seeds=pick_result.attempted_seeds + place_result.attempted_seeds,
        )

    def probe_align_plan_feasibility(
        self,
        world: WorldState,
        plan: TaskPlan,
        slots: dict[str, tuple[float, float, float]],
    ) -> ProbePlanResult:
        start_time = time.perf_counter()
        edge_results: list[ProbeResult] = []
        failure_reasons: list[str] = []
        i = 0
        while i < len(plan.steps) - 1:
            pick_step = plan.steps[i]
            place_step = plan.steps[i + 1]
            if pick_step.action == "pick" and place_step.action == "place":
                edge_result = self.probe_align_edge(
                    world, pick_step.object, place_step.slot or "", slots
                )
                edge_results.append(edge_result)
                if not edge_result.feasible:
                    failure_reasons.append(
                        f"edge({pick_step.object}->{place_step.slot}): "
                        f"{edge_result.failure_reason or 'infeasible'}"
                    )
                i += 2
            else:
                i += 1
        total_time = time.perf_counter() - start_time
        all_feasible = all(r.feasible for r in edge_results)
        return ProbePlanResult(
            feasible=all_feasible,
            edge_results=tuple(edge_results),
            total_planning_time=round(total_time, 4),
            failure_reasons=tuple(failure_reasons),
        )

    def _geometric_pick_probe(
        self,
        world: WorldState,
        obj: Any,
    ) -> ProbeResult:
        import math
        dist = math.dist(obj.pose[:2], world.robot_base_xy)
        in_reach = world.robot_reach_min <= dist <= world.robot_reach_max
        near_obs = any(
            _point_near_obstacle(obj.pose[:2], obstacle, 0.08)
            for obstacle in world.obstacles
        )
        feasible = in_reach and not near_obs
        return ProbeResult(
            feasible=feasible,
            ik_success=feasible,
            ompl_success=feasible,
            planning_time=0.0,
            estimated_path_length=dist,
            min_clearance=0.15 if not near_obs else 0.05,
            collision_count=0,
            failure_reason=(
                None
                if feasible
                else ("not_in_reach" if not in_reach else "obstacle_proximity")
            ),
        )

    def _geometric_place_probe(
        self,
        world: WorldState,
        slot_pos: tuple[float, float, float],
    ) -> ProbeResult:
        import math
        dist = math.dist(slot_pos[:2], world.robot_base_xy)
        in_reach = world.robot_reach_min <= dist <= world.robot_reach_max
        near_obs = any(
            _point_near_obstacle(slot_pos[:2], obstacle, 0.08)
            for obstacle in world.obstacles
        )
        feasible = in_reach and not near_obs
        return ProbeResult(
            feasible=feasible,
            ik_success=feasible,
            ompl_success=feasible,
            planning_time=0.0,
            estimated_path_length=dist,
            min_clearance=0.15 if not near_obs else 0.05,
            collision_count=0,
            failure_reason=(
                None
                if feasible
                else ("not_in_reach" if not in_reach else "obstacle_proximity")
            ),
        )

    def _backend_pick_probe(
        self,
        world: WorldState,
        object_id: str,
        obj: Any,
    ) -> ProbeResult:
        start = time.perf_counter()
        try:
            import math

            dist = math.dist(obj.pose[:2], world.robot_base_xy)
            hints = None
            if self._hint_cache is not None:
                hints = self._hint_cache.hints_for(object_id, obj.cls, dist)
            clearance = float(self._runtime.CONFIG.grasp.approach_clearance_m)
            pregrasp_pos = (
                obj.pose[0],
                obj.pose[1],
                obj.pose[2] + clearance,
            )
            # Candidate plans change cube occupancy before later edges. Probing
            # every edge against the untouched initial clutter creates false
            # negatives for slots that will already be cleared. Keep static
            # geometry (typed wall/table) active and ignore only task cubes.
            probe_ignored_bodies = [item.id for item in world.objects]
            probe_budget = float(
                getattr(
                    getattr(self._runtime.CONFIG, "motion", None),
                    "probe_time_limit_s",
                    2.0,
                )
            )
            remaining = max(0.001, probe_budget - (time.perf_counter() - start))
            if all(
                hasattr(self._runtime, name)
                for name in ("grasp_pose_candidates", "probe_motion_to")
            ):
                report = self._probe_grasp_candidates_with_fast_ik_failover(
                    object_id,
                    obj.pose,
                    ignored_body_names=probe_ignored_bodies,
                    time_limit=remaining,
                    lateral_enabled=obj.cls == "cube",
                )
            elif hasattr(self._runtime, "probe_grasp_candidates"):
                report = self._runtime.probe_grasp_candidates(
                    object_id,
                    obj.pose,
                    ignored_body_names=probe_ignored_bodies,
                    time_limit=remaining,
                    lateral_enabled=obj.cls == "cube",
                )
            else:
                report = self._runtime.probe_motion_to(
                    pregrasp_pos,
                    label=f"pick({object_id}) pregrasp probe",
                    ignored_body_names=probe_ignored_bodies,
                    time_limit=remaining,
                )
            planning_time = time.perf_counter() - start
            return ProbeResult(
                feasible=bool(report.get("success", False)),
                ik_success=bool(report.get("ik_success", False)),
                ompl_success=bool(report.get("ompl_success", False)),
                planning_time=round(planning_time, 4),
                estimated_path_length=float(report.get("path_length", dist)),
                min_clearance=float(report.get("min_clearance", 0.0)),
                collision_count=0,
                failure_reason=report.get("failure_reason"),
                grasp_diagnostics=tuple(report.get("grasp_diagnostics", ())),
                timeout_phase=report.get("timeout_phase"),
                attempted_orientations=int(report.get("attempted_orientations", 0)),
                attempted_seeds=int(report.get("attempted_seeds", 0)),
            )
        except Exception as exc:
            planning_time = time.perf_counter() - start
            return ProbeResult(
                feasible=False,
                ik_success=False,
                ompl_success=False,
                planning_time=round(planning_time, 4),
                failure_reason=f"probe_exception:{exc.__class__.__name__}:{exc}",
            )

    @staticmethod
    def _is_fast_ik_failure(report: dict, elapsed: float) -> bool:
        if elapsed > 0.3 or bool(report.get("ik_success", False)):
            return False
        reason = str(report.get("failure_reason") or "").strip().lower()
        return (
            reason == "ik_failed"
            or reason.startswith("ik_")
            or reason.startswith("no_ik_")
        )

    def _probe_grasp_candidates_with_fast_ik_failover(
        self,
        object_id: str,
        object_pose: tuple[float, ...],
        *,
        ignored_body_names: list[str],
        time_limit: float,
        lateral_enabled: bool,
    ) -> dict:
        """Probe ordered execution poses, immediately skipping fast IK failures."""
        started = time.perf_counter()
        deadline = started + float(time_limit)
        diagnostics: list[dict] = []
        candidates = self._runtime.grasp_pose_candidates(
            object_pose,
            lateral_enabled=lateral_enabled,
        )

        for candidate in candidates:
            name = str(candidate["name"])
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return self._grasp_probe_failure(
                    diagnostics,
                    started,
                    "motion_probe_timeout",
                    timeout_phase=f"grasp_candidate:{name}",
                )

            candidate_started = time.perf_counter()
            report = self._runtime.probe_motion_to(
                candidate["pregrasp_xyz"],
                label=f"pick({object_id}) pregrasp probe {name}",
                ignored_body_names=ignored_body_names,
                time_limit=max(0.001, remaining),
                target_rotation=candidate["rotation"],
                deadline=deadline,
            )
            elapsed = time.perf_counter() - candidate_started
            diagnostic = {
                "candidate": name,
                "approach_vector": [
                    round(float(value), 6) for value in candidate["approach"]
                ],
                "pregrasp_position": [
                    round(float(value), 6) for value in candidate["pregrasp_xyz"]
                ],
                "ik_success": bool(report.get("ik_success", False)),
                "position_error_m": report.get("position_error_m"),
                "orientation_error_rad": report.get("orientation_error_rad"),
                "joint_limit_valid": report.get("joint_limit_valid"),
                "collision_valid": report.get("collision_valid"),
                "collision_reason": report.get("collision_reason"),
                "ompl_success": bool(report.get("ompl_success", False)),
                "planning_time_s": round(elapsed, 6),
                "attempted_seeds": int(report.get("attempted_seeds", 0)),
                "failure_reason": report.get("failure_reason"),
                "fast_ik_failover": self._is_fast_ik_failure(report, elapsed),
            }
            diagnostics.append(diagnostic)
            if hasattr(self._runtime, "log_event"):
                self._runtime.log_event(
                    "GRASP_PROBE_CANDIDATE",
                    "OK" if report.get("success") else "REJECT",
                    object_id=object_id,
                    phase=name,
                    failure_reason=report.get("failure_reason"),
                    grasp_diagnostic=diagnostic,
                )

            if report.get("success"):
                result = dict(report)
                result.update(
                    selected_grasp_candidate=name,
                    attempted_orientations=len(diagnostics),
                    grasp_diagnostics=diagnostics,
                    planning_time=time.perf_counter() - started,
                )
                return result

            if diagnostic["fast_ik_failover"]:
                continue

            if (
                report.get("failure_reason") == "motion_probe_timeout"
                and time.perf_counter() >= deadline
            ):
                return self._grasp_probe_failure(
                    diagnostics,
                    started,
                    "motion_probe_timeout",
                    timeout_phase=report.get(
                        "timeout_phase", f"grasp_candidate:{name}"
                    ),
                )

        return self._grasp_probe_failure(
            diagnostics,
            started,
            "no_grasp_found",
        )

    @staticmethod
    def _grasp_probe_failure(
        diagnostics: list[dict],
        started: float,
        failure_reason: str,
        *,
        timeout_phase: str | None = None,
    ) -> dict:
        return {
            "success": False,
            "ik_success": any(
                bool(item.get("ik_success", False)) for item in diagnostics
            ),
            "ompl_success": False,
            "failure_reason": failure_reason,
            "timeout_phase": timeout_phase,
            "attempted_orientations": len(diagnostics),
            "attempted_seeds": sum(
                int(item.get("attempted_seeds", 0)) for item in diagnostics
            ),
            "planning_time": time.perf_counter() - started,
            "grasp_diagnostics": diagnostics,
        }

    def _backend_place_probe(
        self,
        world: WorldState,
        object_id: str,
        slot_pos: tuple[float, float, float],
    ) -> ProbeResult:
        start = time.perf_counter()
        try:
            import math

            dist = math.dist(slot_pos[:2], world.robot_base_xy)
            clearance = float(self._runtime.CONFIG.grasp.approach_clearance_m)
            preplace_pos = (
                slot_pos[0],
                slot_pos[1],
                slot_pos[2] + clearance,
            )
            probe_ignored_bodies = [item.id for item in world.objects]
            report = self._runtime.probe_motion_to(
                preplace_pos,
                label=f"place({object_id}) preplace probe",
                ignored_body_names=probe_ignored_bodies,
                time_limit=max(
                    0.001,
                    float(
                        getattr(
                            getattr(self._runtime.CONFIG, "motion", None),
                            "probe_time_limit_s",
                            2.0,
                        )
                    )
                    - (time.perf_counter() - start),
                ),
            )
            planning_time = time.perf_counter() - start
            return ProbeResult(
                feasible=bool(report.get("success", False)),
                ik_success=bool(report.get("ik_success", False)),
                ompl_success=bool(report.get("ompl_success", False)),
                planning_time=round(planning_time, 4),
                estimated_path_length=float(report.get("path_length", dist)),
                min_clearance=float(report.get("min_clearance", 0.0)),
                collision_count=0,
                failure_reason=report.get("failure_reason"),
                timeout_phase=report.get("timeout_phase"),
                attempted_seeds=int(report.get("attempted_seeds", 0)),
            )
        except Exception as exc:
            planning_time = time.perf_counter() - start
            return ProbeResult(
                feasible=False,
                ik_success=False,
                ompl_success=False,
                planning_time=round(planning_time, 4),
                failure_reason=f"probe_exception:{exc.__class__.__name__}:{exc}",
            )


def _point_near_obstacle(
    point_xy: tuple[float, float],
    obstacle: Any,
    clearance: float,
) -> bool:
    import math

    size = getattr(obstacle, "size", None)
    if getattr(obstacle, "kind", "obstacle") == "wall" and size is None:
        raise ValueError(f"wall obstacle {obstacle.id!r} requires explicit AABB size")
    if size is None:
        return math.dist(point_xy, obstacle.pose[:2]) < obstacle.radius + clearance
    dx = max(abs(float(point_xy[0]) - obstacle.pose[0]) - size[0] / 2.0, 0.0)
    dy = max(abs(float(point_xy[1]) - obstacle.pose[1]) - size[1] / 2.0, 0.0)
    return math.hypot(dx, dy) < clearance
