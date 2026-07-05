from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

from backends.adaptive.event_log import EventLog
from backends.adaptive.hint_cache import HintCache
from configuration import RuntimeConfig, get_active_runtime_config
from task_planning.types import ConfirmationResult, ScoredPlan, Step, TaskPlan
from plugins.registry import PluginRegistry
from world.state import WorldState

from .primitives import PrimitiveExecutor
from .recovery import RecoveryAction, RecoveryPolicy
from .verifier import ObservedPredicateVerifier


_PREDICATE_RE = re.compile(r"^\s*([a-z][a-z0-9-]*)(?:\(([^()]*)\))?\s*$")


@dataclass(frozen=True)
class StepResult:
    step_id: int
    action: str
    object_id: str
    success: bool
    attempts: int
    failure_reason: str | None = None


@dataclass(frozen=True)
class RunResult:
    success: bool
    moved_count: int
    failure_reasons: tuple[str, ...]
    step_results: tuple[StepResult, ...]


@dataclass(frozen=True)
class RobustAlignTelemetry:
    candidate_count: int = 0
    ranked_costs: tuple[float, ...] = ()
    selected_candidate_id: str | None = None
    selected_candidate_strategy: str | None = None
    failed_before_success: int = 0
    probe_planning_time: float = 0.0
    ik_failure_count: int = 0
    ompl_failure_count: int = 0
    alignment_error: float = 0.0
    spacing_error: float = 0.0


class TaskRunner:
    """Generic step runner. Task-specific validation and goals live in plugins."""

    def __init__(
        self,
        plan: TaskPlan,
        world: WorldState,
        slots: dict[str, tuple[float, float, float]],
        hint_cache: HintCache,
        plugin_registry: PluginRegistry,
        event_log: EventLog,
        primitives: PrimitiveExecutor,
        runtime_config: RuntimeConfig | None = None,
        robust_align: bool = False,
        runtime_module=None,
    ):
        self.plan = plan
        self.world = world
        self.slots = slots
        self.hint_cache = hint_cache
        self.event_log = event_log
        self.primitives = primitives
        self.plugin = plugin_registry.get(plan.task)
        self.plugin.validate_plan(plan, world)
        resolved_config = runtime_config or get_active_runtime_config()
        self.runtime_config = resolved_config
        self.verifier = ObservedPredicateVerifier(
            primitives,
            config=resolved_config.verification,
        )
        self.recovery = RecoveryPolicy(world.max_retries_per_object)
        self._recovery_step_id = max(step.step_id for step in plan.steps) + 1
        self._align_recoveries = 0
        from .motion_probe import requires_motion_probe

        self._runtime_module = runtime_module
        self._motion_probe_required = requires_motion_probe(world)
        self._robust_align = (
            plan.task == "align"
            and (robust_align or self._motion_probe_required)
        )
        self._robust_align_telemetry = RobustAlignTelemetry()

    def run(self) -> RunResult:
        if self._robust_align:
            return self._run_robust_align()
        return self._run_standard()

    @property
    def robust_align_telemetry(self) -> RobustAlignTelemetry:
        return self._robust_align_telemetry

    def _run_robust_align(self) -> RunResult:
        from task_planning.candidate_generator import generate_align_candidates
        from task_planning.cost_model import rank_candidate_plans
        from .confirmation import confirm_ranked_align_candidates
        from .motion_probe import MotionProbe

        candidates = generate_align_candidates(self.world, self.slots)
        cache_cfg = self.runtime_config.align_cache
        cache_config_dict = {
            "granularity": cache_cfg.cache_key_granularity,
            "min_samples": cache_cfg.min_samples_for_cache,
            "cache_weight": cache_cfg.adaptive_cache_weight,
            "failure_penalty": cache_cfg.failure_penalty,
        }
        ranked = rank_candidate_plans(
            candidates,
            self.world,
            self.slots,
            hint_cache=self.hint_cache if cache_cfg.use_adaptive_cache else None,
            use_adaptive_cache=cache_cfg.use_adaptive_cache,
            cache_config=cache_config_dict,
        )
        self.event_log.write(
            "ROBUST_ALIGN",
            "CANDIDATES",
            task=self.plan.task,
            scene_id=self.plan.scene_id,
            candidate_count=len(candidates),
            ranked_costs=[s.estimated_cost for s in ranked],
            adaptive_cache_used=cache_cfg.use_adaptive_cache,
        )

        motion_probe = MotionProbe(
            runtime=self._runtime_module,
            primitives=self.primitives,
            hint_cache=self.hint_cache,
        )
        confirmation = confirm_ranked_align_candidates(
            self.world, ranked, self.slots, motion_probe
        )

        self._record_cache_entries(
            ranked, confirmation, motion_probe, slots=self.slots
        )

        self._robust_align_telemetry = RobustAlignTelemetry(
            candidate_count=len(candidates),
            ranked_costs=tuple(s.estimated_cost for s in ranked),
            selected_candidate_id=confirmation.selected_plan_id,
            selected_candidate_strategy=next(
                (
                    scored.generation_method
                    for scored in ranked
                    if scored.plan_id == confirmation.selected_plan_id
                ),
                None,
            ),
            failed_before_success=len(confirmation.failed_plan_ids),
            probe_planning_time=confirmation.total_planning_time,
            ik_failure_count=confirmation.total_ik_failures,
            ompl_failure_count=confirmation.total_ompl_failures,
        )

        if not confirmation.confirmed or confirmation.plan is None:
            self.event_log.write(
                "ROBUST_ALIGN",
                "ALL_FAILED",
                task=self.plan.task,
                scene_id=self.plan.scene_id,
                candidate_count=len(candidates),
                failure_reasons=confirmation.failure_reasons,
            )
            if cache_cfg.use_adaptive_cache:
                self.hint_cache.save_align_caches()
            return RunResult(
                success=False,
                moved_count=0,
                failure_reasons=(
                    "all_candidates_infeasible",
                    *confirmation.failure_reasons,
                ),
                step_results=(),
            )

        confirmed_plan = confirmation.plan
        self.event_log.write(
            "ROBUST_ALIGN",
            "CONFIRMED",
            task=self.plan.task,
            scene_id=self.plan.scene_id,
            selected_plan_id=confirmation.selected_plan_id,
            estimated_cost=next(
                (s.estimated_cost for s in ranked if s.plan_id == confirmation.selected_plan_id),
                0.0,
            ),
        )

        self.plan = confirmed_plan
        self._recovery_step_id = max(step.step_id for step in confirmed_plan.steps) + 1
        self.plugin.validate_plan(confirmed_plan, self.world)

        result = self._run_standard()

        if cache_cfg.use_adaptive_cache:
            from task_planning.adaptive_heuristic import record_plan_result_to_cache
            record_plan_result_to_cache(
                cache=self.hint_cache,
                world=self.world,
                plan=confirmed_plan,
                slots=self.slots,
                success=result.success,
                actual_cost=next(
                    (s.estimated_cost for s in ranked if s.plan_id == confirmation.selected_plan_id),
                    0.0,
                ),
                planning_time=confirmation.total_planning_time,
                ik_failures=confirmation.total_ik_failures,
                ompl_failures=confirmation.total_ompl_failures,
                failure_reason=";".join(result.failure_reasons) if not result.success else "",
                run_id=self.event_log.run_id,
                granularity=cache_cfg.cache_key_granularity,
            )
            self.hint_cache.save_align_caches()

        return result

    def _record_cache_entries(
        self,
        ranked: list,
        confirmation,
        motion_probe: "MotionProbe",
        slots: dict[str, tuple[float, float, float]],
    ) -> None:
        from task_planning.adaptive_heuristic import record_probe_result_to_cache

        cache_cfg = self.runtime_config.align_cache
        if not cache_cfg.use_adaptive_cache:
            return

        for scored in ranked:
            probe_result = motion_probe.probe_align_plan_feasibility(
                self.world, scored.plan, slots
            )
            i = 0
            step_results = list(probe_result.edge_results)
            edge_idx = 0
            while i < len(scored.plan.steps) - 1:
                pick_step = scored.plan.steps[i]
                place_step = scored.plan.steps[i + 1]
                if pick_step.action == "pick" and place_step.action == "place":
                    if edge_idx < len(step_results):
                        edge = step_results[edge_idx]
                        record_probe_result_to_cache(
                            cache=self.hint_cache,
                            world=self.world,
                            object_id=pick_step.object,
                            slot_id=place_step.slot or "",
                            slots=slots,
                            success=edge.feasible,
                            actual_cost=scored.estimated_cost / max(1, len(step_results)),
                            planning_time=edge.planning_time,
                            ik_failures=0 if edge.ik_success else 1,
                            ompl_failures=0 if edge.ompl_success or not edge.ik_success else 1,
                            collisions=edge.collision_count,
                            failure_reason=edge.failure_reason or "",
                            run_id=self.event_log.run_id,
                            granularity=cache_cfg.cache_key_granularity,
                        )
                    edge_idx += 1
                i += 2

    def _run_standard(self) -> RunResult:
        results: list[StepResult] = []
        failures: list[str] = []
        completed_objects: set[str] = set()
        completed_plan_steps = 0

        for step in self.plan.steps:
            result = self._execute_step(step)
            results.append(result)
            if not result.success:
                recovered = False
                progress_recovery_attempted = False
                if (
                    step.action == "place"
                    and (result.failure_reason or "").startswith(
                        f"{RecoveryAction.REPLAN_REQUIRED.value}:"
                    )
                ):
                    completed_objects.add(step.object)
                    progress_recovery_attempted = self.plan.task == "align"
                    recovered, recovery_results = self._ensure_stable_progress(
                        completed_objects
                    )
                    results.extend(recovery_results)
                if not recovered:
                    completed_objects.discard(step.object)
                    failures.append(
                        "align_recovery_exhausted"
                        if progress_recovery_attempted
                        else result.failure_reason or "unknown_step_failure"
                    )
                    break
                completed_plan_steps += 1
                continue
            completed_plan_steps += 1
            if step.action == "place":
                completed_objects.add(step.object)
                stable, recovery_results = self._ensure_stable_progress(
                    completed_objects
                )
                results.extend(recovery_results)
                if not stable:
                    failures.append("align_recovery_exhausted")
                    break

        goal_ok = (
            completed_plan_steps == len(self.plan.steps)
            and not failures
            and self.plugin.verify_goal(
                self.plan, self.world, self.verifier, self.slots
            )
        )
        if not goal_ok and not failures:
            failures.append("final_goal_not_observed")
        self.event_log.write(
            "RUN",
            "OK" if goal_ok else "FAILED",
            task=self.plan.task,
            scene_id=self.plan.scene_id,
            failure_reason=";".join(failures) or None,
            moved_count=len(completed_objects),
        )
        return RunResult(
            success=goal_ok,
            moved_count=len(completed_objects),
            failure_reasons=tuple(failures),
            step_results=tuple(results),
        )

    def _ensure_stable_progress(
        self,
        completed_objects: set[str],
    ) -> tuple[bool, list[StepResult]]:
        progress = self.plugin.assess_progress(
            self.plan,
            self.verifier,
            self.slots,
            completed_objects,
        )
        if progress.valid:
            return True, []
        settle = getattr(self.primitives, "settle_for_verification", None)
        if callable(settle):
            settle(self.runtime_config.recovery.verification_settle_steps)
            confirmed = self.plugin.assess_progress(
                self.plan,
                self.verifier,
                self.slots,
                completed_objects,
            )
            self.event_log.write(
                "ALIGN_STABILITY",
                "CONFIRMED" if not confirmed.valid else "TRANSIENT",
                task=self.plan.task,
                scene_id=self.plan.scene_id,
                failure_reason=confirmed.reason,
                initial_failure_reason=progress.reason,
            )
            progress = confirmed
            if progress.valid:
                return True, []
        if self.plan.task == "align":
            return self._recover_invalid_align(completed_objects, progress)
        completed_objects.intersection_update(progress.stable_objects)
        return False, []

    def _recover_invalid_align(self, completed_objects, progress):
        results: list[StepResult] = []
        limit = max(1, self.world.max_retries_per_object)
        assigned_slots = {
            step.object: step.slot
            for step in self.plan.steps
            if step.action == "place" and step.slot
        }
        recovery_attempts = 0
        while not progress.valid and recovery_attempts < limit:
            recovery_attempts += 1
            self._align_recoveries += 1
            attempt = recovery_attempts
            self.event_log.write(
                "ALIGN_RECOVERY",
                "START",
                task=self.plan.task,
                scene_id=self.plan.scene_id,
                attempt=attempt,
                invalid_objects=progress.invalid_objects,
                failure_reason=progress.reason,
            )
            attempt_ok = True
            for object_id in progress.invalid_objects:
                slot_id = assigned_slots.get(object_id)
                if slot_id is None:
                    attempt_ok = False
                    break
                if self.primitives.held_object_name() != object_id:
                    pick_result = self._execute_step(
                        self._recovery_step("pick", object_id)
                    )
                    results.append(pick_result)
                    if not pick_result.success:
                        attempt_ok = False
                        break
                place_result = self._execute_step(
                    self._recovery_step("place", object_id, slot=slot_id)
                )
                results.append(place_result)
                if not place_result.success:
                    attempt_ok = False
                    break
            progress = self.plugin.assess_progress(
                self.plan,
                self.verifier,
                self.slots,
                completed_objects,
            )
            if attempt_ok and progress.valid:
                self.event_log.write(
                    "ALIGN_RECOVERY",
                    "OK",
                    task=self.plan.task,
                    scene_id=self.plan.scene_id,
                    attempt=attempt,
                )
                return True, results
            self.event_log.write(
                "ALIGN_RECOVERY",
                "FAILED",
                task=self.plan.task,
                scene_id=self.plan.scene_id,
                attempt=attempt,
                failure_reason=progress.reason or "recovery_step_failed",
            )
        completed_objects.intersection_update(progress.stable_objects)
        return False, results

    def _recovery_step(
        self,
        action: str,
        object_id: str,
        *,
        slot: str | None = None,
    ) -> Step:
        step = Step(
            step_id=self._recovery_step_id,
            action=action,  # type: ignore[arg-type]
            object=object_id,
            slot=slot,
        )
        self._recovery_step_id += 1
        return step

    def _execute_step(self, step: Step) -> StepResult:
        obj = self.world.object_by_id(step.object)
        if obj is None:
            return StepResult(
                step.step_id, step.action, step.object, False, 0, "unknown_object"
            )
        distance = math.dist(obj.pose[:2], self.world.robot_base_xy)
        hints = self.hint_cache.hints_for(step.object, obj.cls, distance)
        target = self._resolve_target(step)
        attempt = 0

        while True:
            attempt += 1
            started = time.perf_counter()
            primitive_result = self.primitives.execute(step, target, hints)
            observed = self._effects_observed(step)
            duration_ms = int((time.perf_counter() - started) * 1000)
            # Primitive return values are diagnostics. Physical truth comes only
            # from the live-state predicate verifier.
            success = observed
            failure_reason = primitive_result.failure_reason
            if not observed and failure_reason is None:
                failure_reason = "expected_effect_not_observed"
            self.event_log.write(
                "STEP",
                "OK" if success else "FAILED",
                step_id=step.step_id,
                task=self.plan.task,
                scene_id=self.plan.scene_id,
                object_id=step.object,
                action=step.action,
                attempt=attempt,
                failure_reason=failure_reason,
                duration_ms=duration_ms,
                ik_backend=hints.ik_backend,
                ik_position_tolerance=hints.ik_position_tolerance,
                grasp_profile=hints.grasp_profile,
                target=target,
                primitive_completed=primitive_result.completed,
            )
            if success:
                return StepResult(
                    step.step_id, step.action, step.object, True, attempt
                )

            decision = self.recovery.decide(
                step,
                attempt,
                failure_reason or "unknown_step_failure",
                object_still_held=self.primitives.held_object_name() == step.object,
            )
            if decision.action == RecoveryAction.RETRY:
                continue
            return StepResult(
                step.step_id,
                step.action,
                step.object,
                False,
                attempt,
                f"{decision.action.value}:{decision.reason}",
            )

    def _resolve_target(self, step: Step) -> tuple[float, float, float] | None:
        if step.action == "place":
            return self.slots.get(step.slot or "")
        return None

    def _effects_observed(self, step: Step) -> bool:
        # LLM effects may add checks, but cannot replace the physical effects
        # required by the selected action.
        expressions = tuple(dict.fromkeys(self._default_effects(step) + step.effects))
        predicates = [self._parse_predicate(expression) for expression in expressions]
        return all(self.verifier.evaluate(predicate, self.slots) for predicate in predicates)

    @staticmethod
    def _default_effects(step: Step) -> tuple[str, ...]:
        if step.action == "pick":
            return (f"holding({step.object})",)
        if step.action == "place":
            return (f"at({step.object},{step.slot})", "handempty")
        return ()

    @staticmethod
    def _parse_predicate(expression: str) -> dict:
        match = _PREDICATE_RE.match(expression)
        if match is None:
            return {"name": "invalid", "args": []}
        raw_args = match.group(2)
        args = [] if raw_args is None else [part.strip() for part in raw_args.split(",")]
        return {"name": match.group(1), "args": args}
