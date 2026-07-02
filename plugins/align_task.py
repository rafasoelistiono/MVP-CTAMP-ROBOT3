from __future__ import annotations

from dataclasses import replace

from configuration import RuntimeConfig
from task_planning.types import SlotConfig, TaskPlan
from task_planning.validator import PlanValidationError, validate_grouped_align_order
from world.slot_allocator import allocate_grouped_align_slots
from world.state import WorldState

from .protocol import TaskProgress


class AlignTaskPlugin:
    api_version = "ctamp-task/v2"
    name = "align"
    supported_actions = {"pick", "place"}

    def validate_plan(self, plan: TaskPlan, world: WorldState) -> None:
        if plan.task != self.name:
            raise PlanValidationError(
                f"align plugin cannot execute task {plan.task!r}"
            )
        if plan.slot_config.type != "line":
            raise PlanValidationError("align task requires line slot_config")
        if plan.target_objects != world.target_objects:
            raise PlanValidationError(
                "align plan target_objects must exactly match context task targets"
            )
        if plan.constraints.get("preserve_obstacles", True) is not True:
            raise PlanValidationError("align plan cannot disable obstacle preservation")
        unsupported = sorted(
            {step.action for step in plan.steps} - self.supported_actions
        )
        if unsupported:
            raise PlanValidationError(
                f"align task does not support actions: {unsupported}"
            )
        missing_capabilities = sorted(
            {step.action for step in plan.steps} - set(world.robot_capabilities)
        )
        if missing_capabilities:
            raise PlanValidationError(
                f"robot lacks capabilities required by align plan: {missing_capabilities}"
            )
        non_cubes = [
            object_id
            for object_id in plan.target_objects
            if world.object_by_id(object_id).cls != "cube"
        ]
        if non_cubes:
            raise PlanValidationError(
                "align task accepts cube objects only: " + ", ".join(non_cubes)
            )
        unreachable = [
            object_id
            for object_id in plan.target_objects
            if not world.object_by_id(object_id).reachable
        ]
        if unreachable:
            raise PlanValidationError(
                "align target objects are unreachable: " + ", ".join(unreachable)
            )
        expected = list(plan.target_objects)
        if len(plan.steps) != len(expected) * 2:
            raise PlanValidationError(
                "align plan must contain exactly one pick/place pair per cube"
            )
        flexible_order = plan.constraints.get("flexible_order", False)
        used_slots: set[str] = set()
        picked_objects: list[str] = []
        for index in range(len(expected)):
            pick_step = plan.steps[index * 2]
            place_step = plan.steps[index * 2 + 1]
            if pick_step.action != "pick":
                raise PlanValidationError(
                    "align plan must alternate pick/place starting with pick"
                )
            if place_step.action != "place" or not place_step.slot:
                raise PlanValidationError(
                    f"align plan step {place_step.step_id} must be a place with slot"
                )
            if pick_step.object != place_step.object:
                raise PlanValidationError(
                    f"align plan pick/place pair {index} must operate on same object: "
                    f"pick={pick_step.object!r}, place={place_step.object!r}"
                )
            if not flexible_order:
                if pick_step.object != expected[index]:
                    raise PlanValidationError(
                        "align plan must pick cubes in target_objects order"
                    )
            if place_step.slot in used_slots:
                raise PlanValidationError(
                    f"align plan assigns duplicate slot {place_step.slot!r}"
                )
            used_slots.add(place_step.slot)
            picked_objects.append(pick_step.object)
        if len(set(picked_objects)) != len(expected):
            raise PlanValidationError(
                "align plan must pick each target object exactly once"
            )
        not_picked = sorted(set(expected) - set(picked_objects))
        if not_picked:
            raise PlanValidationError(
                "align plan does not pick all target objects: " + ", ".join(not_picked)
            )

        # --- grouped tidy variant checks ---
        gt = world.grouped_tidy
        if gt and gt.enabled:
            if plan.target_objects != world.target_objects:
                raise PlanValidationError(
                    "grouped tidy plan target_objects must match context targets"
                )
            assigned_objects = set()
            for group in gt.groups:
                for obj in group.objects:
                    if obj in assigned_objects:
                        raise PlanValidationError(
                            f"object {obj!r} assigned to multiple tidy groups"
                        )
                    assigned_objects.add(obj)
            unassigned = sorted(set(world.target_objects) - assigned_objects)
            if unassigned:
                raise PlanValidationError(
                    "target_objects missing from tidy groups: " + ", ".join(unassigned)
                )
            for step in plan.steps:
                if step.slot and not step.slot.startswith(gt.slot_prefix):
                    raise PlanValidationError(
                        f"slot {step.slot!r} does not start with tidy prefix {gt.slot_prefix!r}"
                    )
            validate_grouped_align_order(plan, gt.slot_prefix, gt.groups)

    def make_slot_config(
        self,
        plan: TaskPlan,
        world: WorldState,
    ) -> SlotConfig:
        gt = world.grouped_tidy
        if gt and gt.enabled:
            slots = allocate_grouped_align_slots(world, gt)
            return SlotConfig(
                type="line",
                axis=gt.axis,
                spacing_m=gt.spacing,
                center_x=world.goal_center[0],
                row_y=world.goal_center[1],
                base_z=world.table_z_top + 0.033,
            )
        return plan.slot_config

    def configure_runtime(
        self,
        plan: TaskPlan,
        world: WorldState,
        config: RuntimeConfig,
    ) -> RuntimeConfig:
        gt = world.grouped_tidy
        if gt and gt.enabled:
            return replace(
                config,
                model=replace(config.model, grasp_ready_q=config.model.home_q),
                motion=replace(
                    config.motion,
                    time_limit_s=20.0,
                    waypoint_step=0.010,
                    settle_steps_per_waypoint=10,
                    final_settle_steps=20,
                ),
                grasp=replace(
                    config.grasp,
                    approach_clearance_m=0.24,
                    pick_grip_sequence=(0.024, 0.023, 0.022),
                ),
            ).validate()
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
            if not all(
                verifier.evaluate(predicate, slots) for predicate in at_predicates
            ):
                return False
        else:
            for index, object_id in enumerate(plan.target_objects):
                slot_id = f"align_slot_{index}"
                if slot_id not in slots:
                    return False
                if not verifier.check_at(object_id, slots[slot_id]):
                    return False
        if not all(
            verifier.check_stable(object_id, include_velocity=True)
            for object_id in plan.target_objects
        ):
            return False
        if not verifier.check_handempty():
            return False
        gt = world.grouped_tidy
        if gt and gt.enabled:
            for group in gt.groups:
                if not verifier.verify_group_row_alignment(
                    group.objects, gt.axis
                ):
                    return False
                if not verifier.verify_group_spacing(
                    group.objects, gt.spacing, gt.axis
                ):
                    return False
            if not verifier.verify_no_grouped_slot_overlap(slots, gt):
                return False
        return True


PLUGIN = AlignTaskPlugin()
