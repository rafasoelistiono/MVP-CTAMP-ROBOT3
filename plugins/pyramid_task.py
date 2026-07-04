from __future__ import annotations

from dataclasses import replace

from configuration import RuntimeConfig
from task_planning.types import SlotConfig, TaskPlan
from task_planning.validator import PlanValidationError
from world.state import WorldState

from .common import pick_place_pairs, pyramid_slot_order, validate_common_cube_plan
from .protocol import TaskProgress


class PyramidTaskPlugin:
    api_version = "ctamp-task/v2"
    name = "pyramid"
    supported_actions = {"pick", "place"}

    def validate_plan(self, plan: TaskPlan, world: WorldState) -> None:
        validate_common_cube_plan(
            plan,
            world,
            task=self.name,
            slot_type="pyramid",
            supported_actions=self.supported_actions,
        )
        build_order = plan.constraints.get("build_order")
        if build_order is not None and tuple(build_order) != plan.target_objects:
            raise PlanValidationError(
                "pyramid build_order must match target_objects base-row first"
            )

        expected_slots = pyramid_slot_order(plan.slot_config)
        if len(expected_slots) != len(plan.target_objects):
            raise PlanValidationError(
                "pyramid slot count must match target_objects count"
            )

        for index, object_id, pick_step, place_step in pick_place_pairs(plan, self.name):
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

        for object_id, slot_id in zip(plan.target_objects, pyramid_slot_order(plan.slot_config)):
            if not verifier.check_at(object_id, slots[slot_id]):
                return False
        return all(
            verifier.check_stable(object_id, include_velocity=True)
            for object_id in plan.target_objects
        )

PLUGIN = PyramidTaskPlugin()
