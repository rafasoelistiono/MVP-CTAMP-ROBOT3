from __future__ import annotations

from dataclasses import replace

from configuration import RuntimeConfig
from task_planning.types import SlotConfig, TaskPlan
from task_planning.validator import PlanValidationError
from world.state import WorldState

from .protocol import TaskProgress


class PyramidTaskPlugin:
    api_version = "ctamp-task/v2"
    name = "pyramid"
    supported_actions = {"pick", "place"}

    def validate_plan(self, plan: TaskPlan, world: WorldState) -> None:
        if plan.task != self.name:
            raise PlanValidationError(
                f"pyramid plugin cannot execute task {plan.task!r}"
            )
        if plan.slot_config.type != "pyramid":
            raise PlanValidationError("pyramid task requires pyramid slot_config")
        if plan.target_objects != world.target_objects:
            raise PlanValidationError(
                "pyramid plan target_objects must exactly match context task targets"
            )
        if plan.constraints.get("preserve_obstacles", True) is not True:
            raise PlanValidationError("pyramid plan cannot disable obstacle preservation")
        build_order = plan.constraints.get("build_order")
        if build_order is not None and tuple(build_order) != plan.target_objects:
            raise PlanValidationError(
                "pyramid build_order must match target_objects base-row first"
            )

        unsupported = sorted(
            {step.action for step in plan.steps} - self.supported_actions
        )
        if unsupported:
            raise PlanValidationError(
                f"pyramid task does not support actions: {unsupported}"
            )
        missing_capabilities = sorted(
            {step.action for step in plan.steps} - set(world.robot_capabilities)
        )
        if missing_capabilities:
            raise PlanValidationError(
                f"robot lacks capabilities required by pyramid plan: {missing_capabilities}"
            )
        non_cubes = [
            object_id
            for object_id in plan.target_objects
            if world.object_by_id(object_id).cls != "cube"
        ]
        if non_cubes:
            raise PlanValidationError(
                "pyramid task accepts cube objects only: " + ", ".join(non_cubes)
            )
        unreachable = [
            object_id
            for object_id in plan.target_objects
            if not world.object_by_id(object_id).reachable
        ]
        if unreachable:
            raise PlanValidationError(
                "pyramid target objects are unreachable: " + ", ".join(unreachable)
            )

        expected_slots = _pyramid_slot_order(plan.slot_config)
        if len(expected_slots) != len(plan.target_objects):
            raise PlanValidationError(
                "pyramid slot count must match target_objects count"
            )
        if len(plan.steps) != len(plan.target_objects) * 2:
            raise PlanValidationError(
                "pyramid plan must contain exactly one pick/place pair per cube"
            )

        for index, object_id in enumerate(plan.target_objects):
            pick_step = plan.steps[index * 2]
            place_step = plan.steps[index * 2 + 1]
            expected_slot = expected_slots[index]
            if pick_step.action != "pick" or pick_step.object != object_id:
                raise PlanValidationError(
                    "pyramid plan must pick cubes in build_order"
                )
            if (
                place_step.action != "place"
                or place_step.object != object_id
                or place_step.slot != expected_slot
                or place_step.on_top_of is not None
            ):
                raise PlanValidationError(
                    "pyramid plan must place each cube immediately in its row/column slot"
                )

        at_predicates = [
            predicate
            for predicate in plan.goal_predicates
            if predicate.get("name") == "at"
        ]
        expected_goals = {
            (object_id, slot_id)
            for object_id, slot_id in zip(plan.target_objects, expected_slots)
        }
        actual_goals = {
            (str(predicate.get("args", ["", ""])[0]), str(predicate.get("args", ["", ""])[1]))
            for predicate in at_predicates
            if len(predicate.get("args", [])) == 2
        }
        if actual_goals != expected_goals:
            raise PlanValidationError(
                "pyramid goal_predicates must contain one at(cube,row_col) per target"
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
        # Pyramid uses the runtime profile as the single source of motion and
        # release tuning. The plugin only applies the same HOME ready-pose
        # policy used by stacked towers so retreats do not intersect the build.
        if world.obstacles:
            return replace(
                config,
                model=replace(config.model, grasp_ready_q=config.model.home_q),
            ).validate()
        return config.validate()

    def assess_progress(self, plan, verifier, slots, completed_objects):
        stable = tuple(
            object_id
            for object_id in plan.target_objects
            if object_id in completed_objects
        )
        return TaskProgress(stable_objects=stable, invalid_objects=())

    def verify_goal(self, plan, world, verifier, slots) -> bool:
        at_predicates = [
            predicate
            for predicate in plan.goal_predicates
            if predicate.get("name") == "at"
        ]
        if at_predicates:
            if not all(verifier.evaluate(predicate, slots) for predicate in at_predicates):
                return False
            return all(
                verifier.check_stable(object_id, include_velocity=True)
                for object_id in plan.target_objects
            )

        for object_id, slot_id in zip(plan.target_objects, _pyramid_slot_order(plan.slot_config)):
            if not verifier.check_at(object_id, slots[slot_id]):
                return False
        return all(
            verifier.check_stable(object_id, include_velocity=True)
            for object_id in plan.target_objects
        )


def _pyramid_slot_order(config: SlotConfig) -> tuple[str, ...]:
    slot_ids: list[str] = []
    for row in range(config.row_count):
        row_length = config.base_row_length - row
        slot_ids.extend(f"row{row}_col{column}" for column in range(row_length))
    return tuple(slot_ids)


PLUGIN = PyramidTaskPlugin()
