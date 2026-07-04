from __future__ import annotations

from dataclasses import replace

from configuration import RuntimeConfig
from task_planning.types import SlotConfig, TaskPlan
from task_planning.validator import PlanValidationError
from world.state import WorldState

from .common import pick_place_pairs, validate_common_cube_plan
from .protocol import TaskProgress


class StackTaskPlugin:
    api_version = "ctamp-task/v2"
    name = "stack"
    supported_actions = {"pick", "place", "stack_place"}

    def validate_plan(self, plan: TaskPlan, world: WorldState) -> None:
        validate_common_cube_plan(
            plan,
            world,
            task=self.name,
            slot_type="tower",
            supported_actions=self.supported_actions,
        )
        expected = plan.target_objects
        for level, object_id, pick_step, place_step in pick_place_pairs(plan, self.name):
            if pick_step.action != "pick" or pick_step.object != object_id:
                raise PlanValidationError(
                    "stack plan must pick each cube once in bottom-to-top order"
                )
            expected_action = "place" if level == 0 else "stack_place"
            expected_slot = "tower_base" if level == 0 else None
            expected_support = None if level == 0 else expected[level - 1]
            if (
                place_step.action != expected_action
                or place_step.object != object_id
                or place_step.slot != expected_slot
                or place_step.on_top_of != expected_support
            ):
                raise PlanValidationError(
                    "stack plan must place each cube immediately after picking it"
                )
        stack_steps = [
            step for step in plan.steps if step.action == "stack_place"
        ]
        if len(stack_steps) != len(expected) - 1:
            raise PlanValidationError(
                "stack plan requires exactly one stack_place per upper cube"
            )
        base_places = [step for step in plan.steps if step.action == "place"]
        if (
            len(base_places) != 1
            or base_places[0].object != expected[0]
            or base_places[0].slot != "tower_base"
        ):
            raise PlanValidationError(
                "stack plan must place the bottom cube once at tower_base"
            )
        for index, step in enumerate(stack_steps, start=1):
            if index >= len(expected):
                raise PlanValidationError("stack plan contains too many stack_place steps")
            if step.object != expected[index] or step.on_top_of != expected[index - 1]:
                raise PlanValidationError(
                    "stack dependency must follow target_objects bottom-to-top"
                )

    def make_slot_config(
        self,
        plan: TaskPlan,
        world: WorldState,
    ) -> SlotConfig:
        return plan.slot_config

    def configure_runtime(
        self,
        plan: TaskPlan,
        world: WorldState,
        config: RuntimeConfig,
    ) -> RuntimeConfig:
        # Stack must retreat to HOME so the ready pose does not intersect the
        # growing tower. This is a task/model policy, not an environment knob.
        return replace(
            config,
            model=replace(config.model, grasp_ready_q=config.model.home_q),
        ).validate()

    def assess_progress(self, plan, verifier, slots, completed_objects):
        if not completed_objects:
            return TaskProgress(stable_objects=(), invalid_objects=())

        placed_count = max(
            index
            for index, object_id in enumerate(plan.target_objects)
            if object_id in completed_objects
        ) + 1
        placed = plan.target_objects[:placed_count]

        if not verifier.check_at(placed[0], slots["tower_base"]):
            return TaskProgress(
                stable_objects=(),
                invalid_objects=placed,
                first_invalid_level=0,
                reason="tower_base_displaced",
            )
        base_stability = verifier.stability_failure_reason(
            placed[0], include_velocity=True
        )
        if base_stability is not None:
            return TaskProgress(
                stable_objects=(),
                invalid_objects=placed,
                first_invalid_level=0,
                reason=f"stack_level_0_unstable:{base_stability}",
            )
        for level in range(1, len(placed)):
            if not verifier.check_on(placed[level], placed[level - 1]):
                return TaskProgress(
                    stable_objects=placed[:level],
                    invalid_objects=placed[level:],
                    first_invalid_level=level,
                    reason=f"stack_level_{level}_unstable",
                )
            stability = verifier.stability_failure_reason(
                placed[level], include_velocity=True
            )
            if stability is not None:
                return TaskProgress(
                    stable_objects=placed[:level],
                    invalid_objects=placed[level:],
                    first_invalid_level=level,
                    reason=f"stack_level_{level}_unstable:{stability}",
                )

        return TaskProgress(stable_objects=placed, invalid_objects=())

    def verify_goal(self, plan, world, verifier, slots) -> bool:
        if not verifier.check_at(plan.target_objects[0], slots["tower_base"]):
            return False
        if not verifier.check_stable(
            plan.target_objects[0], include_velocity=True
        ):
            return False
        for lower, upper in zip(plan.target_objects, plan.target_objects[1:]):
            if not verifier.check_on(upper, lower):
                return False
            if not verifier.check_stable(upper, include_velocity=True):
                return False
        return True


PLUGIN = StackTaskPlugin()
