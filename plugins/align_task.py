from __future__ import annotations

from dataclasses import replace

from configuration import RuntimeConfig
from task_planning.types import SlotConfig, TaskPlan
from task_planning.validator import PlanValidationError, validate_grouped_align_order
from world.state import WorldState

from .common import pick_place_pairs, validate_common_cube_plan
from .protocol import TaskProgress


class AlignTaskPlugin:
    api_version = "ctamp-task/v2"
    name = "align"
    supported_actions = {"pick", "place"}

    def validate_plan(self, plan: TaskPlan, world: WorldState) -> None:
        validate_common_cube_plan(
            plan,
            world,
            task=self.name,
            slot_type="line",
            supported_actions=self.supported_actions,
        )
        expected = plan.target_objects
        flexible_order = plan.constraints.get("flexible_order", False)
        used_slots: set[str] = set()
        picked_objects: list[str] = []
        for index, _, pick_step, place_step in pick_place_pairs(plan, self.name):
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
            home_q = config.model.home_q
            if world.scene_id == "align_grouped_tidy_wall_world":
                home_q = (-0.8, -0.8, 0.0, -2.4, 0.0, 1.6, -0.7)
            motion = replace(
                config.motion,
                time_limit_s=20.0,
                waypoint_step=0.010,
                settle_steps_per_waypoint=10,
                final_settle_steps=20,
            )
            verification = config.verification
            safety = config.safety
            grasp = replace(
                config.grasp,
                approach_clearance_m=0.24,
                pick_grip_sequence=(0.024, 0.023, 0.022),
            )
            if world.scene_id == "align_grouped_tidy_wall_world":
                motion = replace(
                    motion,
                    ompl_required=True,
                    state_validity_resolution=0.0004,
                    waypoint_step=0.006,
                    settle_steps_per_waypoint=20,
                    valid_state_sampler="obstacle_based",
                    optimization_planner="BITstar",
                )
                ik = replace(
                    config.ik,
                    backend="pinocchio",
                    require_pinocchio=True,
                    use_fallback=False,
                )
                verification = replace(
                    verification,
                    at_x_m=0.020,
                    at_y_m=0.020,
                )
                grasp = replace(
                    grasp,
                    approach_clearance_m=0.20,
                    pick_grip_sequence=(0.030, 0.028, 0.026),
                    pick_offset_sequence_m=(0.105, 0.105, 0.105),
                )
                safety = replace(safety, planning_obstacle_clearance_m=0.001)
            else:
                ik = config.ik
            return replace(
                config,
                model=replace(
                    config.model,
                    base_xy=world.robot_base_xy,
                    base_z=world.robot_base_z,
                    obstacle_body_names=tuple(
                        obstacle.id for obstacle in world.obstacles
                    ),
                    home_q=home_q,
                    grasp_ready_q=home_q,
                    desired_tool_x=(
                        (1.0, 0.0, 0.0)
                        if world.scene_id == "align_grouped_tidy_wall_world"
                        else config.model.desired_tool_x
                    ),
                ),
                ik=ik,
                motion=motion,
                verification=verification,
                grasp=grasp,
                safety=safety,
            ).validate()
        if world.obstacles:
            return replace(
                config,
                model=replace(
                    config.model,
                    base_xy=world.robot_base_xy,
                    base_z=world.robot_base_z,
                    obstacle_body_names=tuple(
                        obstacle.id for obstacle in world.obstacles
                    ),
                    grasp_ready_q=config.model.home_q,
                ),
            ).validate()
        return config.validate()

    def assess_progress(self, plan, verifier, slots, completed_objects):
        stable: list[str] = []
        invalid: list[str] = []
        for object_id in plan.target_objects:
            if object_id not in completed_objects:
                continue
            at_ok = True
            stable_ok = True
            if hasattr(verifier, "check_at"):
                slot_id = None
                for step in plan.steps:
                    if step.action == "place" and step.object == object_id and step.slot:
                        slot_id = step.slot
                        break
                if slot_id and slot_id in slots:
                    at_ok = verifier.check_at(object_id, slots[slot_id])
            if hasattr(verifier, "check_stable"):
                stable_ok = verifier.check_stable(object_id, include_velocity=True)
            if at_ok and stable_ok:
                stable.append(object_id)
            else:
                invalid.append(object_id)
        return TaskProgress(
            stable_objects=tuple(stable),
            invalid_objects=tuple(invalid),
        )

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
