from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from backends.adaptive.event_log import EventLog
from backends.adaptive.hint_cache import HintCache
from configuration import RuntimeConfig, get_active_runtime_config
from task_planning.types import Step, TaskPlan
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
            layer_height_m=plan.slot_config.layer_height_m,
            config=resolved_config.verification,
        )
        self.recovery = RecoveryPolicy(world.max_retries_per_object)
        self._recovery_step_id = max(step.step_id for step in plan.steps) + 1
        self._stack_rebuilds = 0

    def run(self) -> RunResult:
        results: list[StepResult] = []
        failures: list[str] = []
        completed_objects: set[str] = set()
        completed_plan_steps = 0

        # A satisfied stack goal is terminal.  This prevents a resumed run or
        # a stale plan from picking apart an already completed tower.
        if self._stack_goal_complete(completed_objects, allow_untracked=True):
            return RunResult(
                success=True,
                moved_count=len(self.plan.target_objects),
                failure_reasons=(),
                step_results=(),
            )

        for step in self.plan.steps:
            result = self._execute_step(step)
            results.append(result)
            if not result.success:
                recovered = False
                stack_recovery_attempted = False
                if (
                    step.action in {"place", "stack_place"}
                    and (result.failure_reason or "").startswith(
                        f"{RecoveryAction.REPLAN_REQUIRED.value}:"
                    )
                ):
                    # Retry the failed stack level now. A released cube is
                    # picked again and placed on its intended support before
                    # the runner is allowed to advance to the next cube.
                    completed_objects.add(step.object)
                    stack_recovery_attempted = self.plan.task == "stack"
                    recovered, recovery_results = self._ensure_stable_progress(
                        completed_objects
                    )
                    results.extend(recovery_results)
                if not recovered:
                    completed_objects.discard(step.object)
                    failures.append(
                        "stack_rebuild_exhausted"
                        if stack_recovery_attempted
                        else result.failure_reason or "unknown_step_failure"
                    )
                    break
                completed_plan_steps += 1
                continue
            completed_plan_steps += 1
            if step.action in {"place", "stack_place"}:
                completed_objects.add(step.object)
                if self._stack_goal_complete(completed_objects):
                    # Final verified layer: stop immediately.  In particular,
                    # do not enter suffix recovery after a completed 4/4 tower.
                    break
                stable, recovery_results = self._ensure_stable_progress(
                    completed_objects
                )
                results.extend(recovery_results)
                if not stable:
                    failures.append("stack_rebuild_exhausted")
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

    def _stack_goal_complete(
        self,
        completed_objects: set[str],
        *,
        allow_untracked: bool = False,
    ) -> bool:
        if self.plan.task != "stack":
            return False
        targets = set(self.plan.target_objects)
        if not allow_untracked and not targets.issubset(completed_objects):
            return False
        # Avoid an unnecessary settle cycle for the normal ungrouped start.
        # Only confirm an untracked/pre-existing tower when it already matches
        # the complete goal in the current observation.
        if allow_untracked and not self.plugin.verify_goal(
            self.plan, self.world, self.verifier, self.slots
        ):
            return False
        settle = getattr(self.primitives, "settle_for_verification", None)
        if callable(settle):
            settle(self.runtime_config.recovery.verification_settle_steps)
        if not self.plugin.verify_goal(
            self.plan, self.world, self.verifier, self.slots
        ):
            return False
        self.event_log.write(
            "STACK_COMPLETE",
            "TERMINAL",
            task=self.plan.task,
            scene_id=self.plan.scene_id,
            moved_count=len(self.plan.target_objects),
        )
        return True

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
                "STACK_STABILITY",
                "CONFIRMED" if not confirmed.valid else "TRANSIENT",
                task=self.plan.task,
                scene_id=self.plan.scene_id,
                failure_reason=confirmed.reason,
                initial_failure_reason=progress.reason,
            )
            progress = confirmed
            if progress.valid:
                return True, []
        return self._rebuild_invalid_suffix(completed_objects, progress)

    def _rebuild_invalid_suffix(self, completed_objects, progress):
        results: list[StepResult] = []
        limit = self.runtime_config.recovery.max_stack_rebuilds

        while not progress.valid and self._stack_rebuilds < limit:
            self._stack_rebuilds += 1
            attempt = self._stack_rebuilds
            self.event_log.write(
                "STACK_REBUILD",
                "START",
                task=self.plan.task,
                scene_id=self.plan.scene_id,
                attempt=attempt,
                failure_reason=progress.reason,
                stable_objects=progress.stable_objects,
                invalid_objects=progress.invalid_objects,
            )
            attempt_ok, attempt_results = self._execute_stack_rebuild(
                progress.invalid_objects,
                completed_objects,
                attempt,
            )
            results.extend(attempt_results)
            progress = self.plugin.assess_progress(
                self.plan,
                self.verifier,
                self.slots,
                completed_objects,
            )
            if attempt_ok and progress.valid:
                self.event_log.write(
                    "STACK_REBUILD",
                    "OK",
                    task=self.plan.task,
                    scene_id=self.plan.scene_id,
                    attempt=attempt,
                )
                return True, results
            self.event_log.write(
                "STACK_REBUILD",
                "FAILED",
                task=self.plan.task,
                scene_id=self.plan.scene_id,
                attempt=attempt,
                failure_reason=progress.reason or "rebuild_step_failed",
            )

        completed_objects.intersection_update(progress.stable_objects)
        return False, results

    def _execute_stack_rebuild(
        self,
        invalid_objects: tuple[str, ...],
        completed_objects: set[str],
        attempt: int,
    ) -> tuple[bool, list[StepResult]]:
        results: list[StepResult] = []

        # A fallen cube can leave the hand in contact at the current state.
        # OMPL must not be asked to plan from that invalid start state. Escape
        # once while ignoring only the suffix being rebuilt; subsequent
        # pick/place calls use the normal collision policy again.
        prepare = getattr(self.primitives, "prepare_stack_recovery", None)
        if callable(prepare):
            prepared = prepare(invalid_objects)
            self.event_log.write(
                "STACK_RECOVERY_ESCAPE",
                "OK" if prepared.completed else "FAILED",
                task=self.plan.task,
                scene_id=self.plan.scene_id,
                attempt=attempt,
                invalid_objects=invalid_objects,
                failure_reason=prepared.failure_reason,
            )
            if not prepared.completed:
                return False, results

        # Clear a multi-cube unstable suffix from top to bottom. A single failed
        # level is retried directly, avoiding a wasteful place-to-staging then
        # pick-again cycle for cases such as cube1 missing tower_base.
        staging_ok = True
        if len(invalid_objects) > 1:
            try:
                for object_id in reversed(invalid_objects):
                    slot_id = f"recovery_{attempt}_{object_id}"
                    target = self._allocate_staging_target(object_id)
                    if target is None:
                        staging_ok = False
                        break

                    self.slots[slot_id] = target

                    if self.primitives.held_object_name() != object_id:
                        pick = self._recovery_step("pick", object_id)
                        pick_result = self._execute_step(pick)
                        results.append(pick_result)
                        if not pick_result.success:
                            staging_ok = False
                            break

                    place = self._recovery_step("place", object_id, slot=slot_id)
                    place_result = self._execute_step(place)
                    results.append(place_result)
                    if not place_result.success:
                        staging_ok = False
                        break
            finally:
                finish_staging = getattr(
                    self.primitives,
                    "finish_stack_recovery_staging",
                    None,
                )
                if callable(finish_staging):
                    finish_staging()
        else:
            finish_staging = getattr(
                self.primitives,
                "finish_stack_recovery_staging",
                None,
            )
            if callable(finish_staging):
                finish_staging()
        if not staging_ok:
            return False, results

        # Rebuild bottom-to-top and validate the complete rebuilt prefix after
        # each placement, not only the cube that was just released.
        targets = self.plan.target_objects
        for object_id in invalid_objects:
            level = targets.index(object_id)
            if self.primitives.held_object_name() != object_id:
                pick = self._recovery_step("pick", object_id)
                pick_result = self._execute_step(pick)
                results.append(pick_result)
                if not pick_result.success:
                    return False, results

            if level == 0:
                placement = self._recovery_step(
                    "place",
                    object_id,
                    slot="tower_base",
                )
            else:
                placement = self._recovery_step(
                    "stack_place",
                    object_id,
                    on_top_of=targets[level - 1],
                )
            placement_result = self._execute_step(placement)
            results.append(placement_result)
            if not placement_result.success:
                return False, results

            rebuilt = set(targets[: level + 1]) & completed_objects
            rebuilt.add(object_id)
            progress = self.plugin.assess_progress(
                self.plan,
                self.verifier,
                self.slots,
                rebuilt,
            )
            if not progress.valid:
                return False, results

        return True, results

    def _recovery_step(
        self,
        action: str,
        object_id: str,
        *,
        slot: str | None = None,
        on_top_of: str | None = None,
    ) -> Step:
        step = Step(
            step_id=self._recovery_step_id,
            action=action,  # type: ignore[arg-type]
            object=object_id,
            slot=slot,
            on_top_of=on_top_of,
        )
        self._recovery_step_id += 1
        return step

    def _allocate_staging_target(
        self,
        object_id: str,
    ) -> tuple[float, float, float] | None:
        clearance = self.runtime_config.recovery.staging_clearance_m
        step = self.runtime_config.recovery.staging_grid_step_m
        poses = self.primitives.all_object_poses()
        source = poses[object_id]
        candidates: list[tuple[float, float, float]] = []
        x = self.world.table_x_range[0] + clearance
        while x <= self.world.table_x_range[1] - clearance:
            y = self.world.table_y_range[0] + clearance
            while y <= self.world.table_y_range[1] - clearance:
                distance = math.dist((x, y), self.world.robot_base_xy)
                away_from_tower = math.dist((x, y), self.plan.slot_config.base_xy)
                obstacle_safe = all(
                    math.dist((x, y), obstacle.pose[:2])
                    >= obstacle.radius + clearance
                    for obstacle in self.world.obstacles
                )
                object_safe = all(
                    other_id == object_id
                    or math.dist((x, y), pose[:2]) >= clearance
                    for other_id, pose in poses.items()
                )
                if (
                    self.world.robot_reach_min <= distance <= self.world.robot_reach_max
                    and away_from_tower >= clearance * 1.5
                    and obstacle_safe
                    and object_safe
                ):
                    candidates.append((x, y, self.plan.slot_config.base_z))
                y += step
            x += step
        if not candidates:
            return None
        return min(candidates, key=lambda pose: math.dist(source[:2], pose[:2]))

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
                if step.action == "stack_place":
                    target = self._resolve_target(step)
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
            target = self.slots.get(step.slot or "")
            extent = getattr(self.primitives, "object_vertical_half_extent", None)
            if target is not None and self.plan.task == "pyramid":
                target = self._resolve_pyramid_target(step, target)
            if (
                target is not None
                and self.plan.task == "stack"
                and step.slot == "tower_base"
                and callable(extent)
            ):
                target = (
                    target[0],
                    target[1],
                    self.world.table_z_top + float(extent(step.object)),
                )
                self.slots["tower_base"] = target
            return target
        if step.action == "stack_place" and step.on_top_of:
            support = self.primitives.object_pose(step.on_top_of)
            # Keep every level on the observed live tower axis. Following the
            # immediately lower cube's XY would accumulate small placement
            # errors and make the fourth level miss an increasingly leaning
            # tower. The base pose is measured, not hardcoded.
            tower_base = self.primitives.object_pose(self.plan.target_objects[0])
            layer_height = self.plan.slot_config.layer_height_m
            extent = getattr(self.primitives, "object_vertical_half_extent", None)
            if callable(extent):
                layer_height = float(extent(step.on_top_of)) + float(
                    extent(step.object)
                )
            return (
                tower_base[0],
                tower_base[1],
                support[2] + layer_height,
            )
        return None

    def _resolve_pyramid_target(
        self,
        step: Step,
        target: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if not step.slot:
            return target
        match = re.fullmatch(r"row(\d+)_col(\d+)", step.slot)
        if match is None:
            return target
        row = int(match.group(1))
        column = int(match.group(2))
        if row == 0:
            return target

        support_ids = self._pyramid_support_objects(row, column)
        if support_ids is None:
            return target
        left_support, right_support = (
            self.primitives.object_pose(support_ids[0]),
            self.primitives.object_pose(support_ids[1]),
        )
        layer_height = self.plan.slot_config.layer_height_m
        extent = getattr(self.primitives, "object_vertical_half_extent", None)
        if callable(extent):
            layer_height = max(float(extent(object_id)) for object_id in support_ids)
            layer_height += float(extent(step.object))

        resolved = (
            (left_support[0] + right_support[0]) / 2.0,
            (left_support[1] + right_support[1]) / 2.0,
            max(left_support[2], right_support[2]) + layer_height,
        )
        self.slots[step.slot] = resolved
        return resolved

    def _pyramid_support_objects(
        self,
        row: int,
        column: int,
    ) -> tuple[str, str] | None:
        slot_to_object = {
            slot_id: object_id
            for object_id, slot_id in zip(
                self.plan.target_objects,
                self._pyramid_slot_order(),
            )
        }
        left = slot_to_object.get(f"row{row - 1}_col{column}")
        right = slot_to_object.get(f"row{row - 1}_col{column + 1}")
        if left is None or right is None:
            return None
        return left, right

    def _pyramid_slot_order(self) -> tuple[str, ...]:
        config = self.plan.slot_config
        slot_ids: list[str] = []
        for row in range(config.row_count):
            row_length = config.base_row_length - row
            slot_ids.extend(f"row{row}_col{column}" for column in range(row_length))
        return tuple(slot_ids)

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
        return (f"on({step.object},{step.on_top_of})", "handempty")

    @staticmethod
    def _parse_predicate(expression: str) -> dict:
        match = _PREDICATE_RE.match(expression)
        if match is None:
            return {"name": "invalid", "args": []}
        raw_args = match.group(2)
        args = [] if raw_args is None else [part.strip() for part in raw_args.split(",")]
        return {"name": match.group(1), "args": args}
