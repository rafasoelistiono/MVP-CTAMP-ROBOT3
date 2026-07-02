from __future__ import annotations

import math
import random
from typing import Sequence

from .types import SCHEMA_VERSION, SlotConfig, Step, TaskPlan
from .loader import parse_plan
from .validator import validate, PlanValidationError

from world.state import WorldState


def generate_align_candidates(
    world: WorldState,
    slots: dict[str, tuple[float, float, float]],
    llm_plan: TaskPlan | None = None,
) -> list[TaskPlan]:
    candidates: list[TaskPlan] = []
    if world.grouped_tidy and world.grouped_tidy.enabled:
        generators = [
            ("grouped_nearest_first", generate_grouped_tidy_nearest_first_plan),
            ("grouped_obstacle_aware", generate_grouped_tidy_obstacle_aware_plan),
            ("grouped_nearest_to_slot", generate_grouped_tidy_nearest_to_slot_plan),
            ("grouped_random", generate_grouped_tidy_random_plan),
        ]
    else:
        generators = [
            ("nearest_first", generate_nearest_first_plan),
            ("nearest_to_slot", generate_nearest_to_slot_plan),
            ("left_to_right", generate_left_to_right_plan),
            ("right_to_left", generate_right_to_left_plan),
            ("obstacle_aware", generate_obstacle_aware_plan),
            ("random_baseline", generate_random_baseline_plan),
        ]
    for method_name, generator in generators:
        try:
            plan = generator(world, slots)
            if plan is not None:
                candidates.append(plan)
        except Exception:
            continue
    if llm_plan is not None:
        if _validate_candidate(llm_plan, world):
            candidates.append(llm_plan)
    seen_keys: set[str] = set()
    unique: list[TaskPlan] = []
    for plan in candidates:
        key = _plan_ordering_key(plan)
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(plan)
    return unique


def generate_nearest_first_plan(
    world: WorldState,
    slots: dict[str, tuple[float, float, float]],
) -> TaskPlan:
    robot_xy = world.robot_base_xy
    objects = _target_objects_with_poses(world)
    objects.sort(key=lambda o: math.dist(o[1][:2], robot_xy))
    ordered_ids = [o[0] for o in objects]
    return _build_align_plan(world, ordered_ids, slots, "nearest_first")


def generate_nearest_to_slot_plan(
    world: WorldState,
    slots: dict[str, tuple[float, float, float]],
) -> TaskPlan:
    objects = _target_objects_with_poses(world)
    slot_ids = sorted(slots.keys())
    ordered_ids: list[str] = []
    remaining = list(objects)
    for slot_id in slot_ids:
        slot_pos = slots[slot_id]
        remaining.sort(key=lambda o: math.dist(o[1][:2], slot_pos[:2]))
        if remaining:
            ordered_ids.append(remaining.pop(0)[0])
    return _build_align_plan(world, ordered_ids, slots, "nearest_to_slot")


def generate_left_to_right_plan(
    world: WorldState,
    slots: dict[str, tuple[float, float, float]],
) -> TaskPlan:
    objects = _target_objects_with_poses(world)
    objects.sort(key=lambda o: o[1][0])
    ordered_ids = [o[0] for o in objects]
    return _build_align_plan(world, ordered_ids, slots, "left_to_right")


def generate_right_to_left_plan(
    world: WorldState,
    slots: dict[str, tuple[float, float, float]],
) -> TaskPlan:
    objects = _target_objects_with_poses(world)
    objects.sort(key=lambda o: o[1][0], reverse=True)
    ordered_ids = [o[0] for o in objects]
    return _build_align_plan(world, ordered_ids, slots, "right_to_left")


def generate_obstacle_aware_plan(
    world: WorldState,
    slots: dict[str, tuple[float, float, float]],
) -> TaskPlan:
    robot_xy = world.robot_base_xy
    objects = _target_objects_with_poses(world)
    obstacles = [(obs.pose[:2], obs.radius) for obs in world.obstacles]

    def obstacle_score(item: tuple[str, tuple[float, float, float]]) -> float:
        oid, pose = item
        robot_dist = math.dist(pose[:2], robot_xy)
        obstacle_risk = 0.0
        for obs_xy, obs_radius in obstacles:
            dist = math.dist(pose[:2], obs_xy)
            if dist < obs_radius + 0.15:
                obstacle_risk += (obs_radius + 0.15 - dist) * 10.0
        return robot_dist + obstacle_risk

    objects.sort(key=obstacle_score)
    ordered_ids = [o[0] for o in objects]
    return _build_align_plan(world, ordered_ids, slots, "obstacle_aware")


def generate_random_baseline_plan(
    world: WorldState,
    slots: dict[str, tuple[float, float, float]],
    seed: int | None = None,
) -> TaskPlan:
    rng = random.Random(seed)
    ordered_ids = list(world.target_objects)
    rng.shuffle(ordered_ids)
    return _build_align_plan(world, ordered_ids, slots, "random_baseline")


def generate_grouped_tidy_nearest_first_plan(
    world: WorldState,
    slots: dict[str, tuple[float, float, float]],
) -> TaskPlan:
    gt = world.grouped_tidy
    if not gt or not gt.enabled:
        return None  # type: ignore[return-value]
    robot_xy = world.robot_base_xy
    ordered_ids: list[str] = []
    for group in gt.groups:
        group_objects = list(group.objects)
        group_objects.sort(
            key=lambda oid: math.dist(
                world.object_by_id(oid).pose[:2], robot_xy
            )
        )
        ordered_ids.extend(group_objects)
    return _build_grouped_align_plan(world, ordered_ids, slots, gt, "grouped_nearest_first")


def generate_grouped_tidy_obstacle_aware_plan(
    world: WorldState,
    slots: dict[str, tuple[float, float, float]],
) -> TaskPlan:
    gt = world.grouped_tidy
    if not gt or not gt.enabled:
        return None  # type: ignore[return-value]
    robot_xy = world.robot_base_xy
    obstacles = [(obs.pose[:2], obs.radius) for obs in world.obstacles]
    ordered_ids: list[str] = []
    for group in gt.groups:
        group_objects = list(group.objects)

        def obstacle_score(item: str) -> float:
            obj = world.object_by_id(item)
            if obj is None:
                return float("inf")
            robot_dist = math.dist(obj.pose[:2], robot_xy)
            obstacle_risk = 0.0
            for obs_xy, obs_radius in obstacles:
                dist = math.dist(obj.pose[:2], obs_xy)
                if dist < obs_radius + 0.15:
                    obstacle_risk += (obs_radius + 0.15 - dist) * 10.0
            return robot_dist + obstacle_risk

        group_objects.sort(key=obstacle_score)
        ordered_ids.extend(group_objects)
    return _build_grouped_align_plan(world, ordered_ids, slots, gt, "grouped_obstacle_aware")


def generate_grouped_tidy_nearest_to_slot_plan(
    world: WorldState,
    slots: dict[str, tuple[float, float, float]],
) -> TaskPlan:
    gt = world.grouped_tidy
    if not gt or not gt.enabled:
        return None  # type: ignore[return-value]
    ordered_ids: list[str] = []
    for group in gt.groups:
        group_slot_ids = sorted(
            k for k in slots.keys() if k.startswith(f"{gt.slot_prefix}_{group.id}_")
        )
        remaining = list(group.objects)
        for slot_id in group_slot_ids:
            slot_pos = slots[slot_id]
            remaining.sort(
                key=lambda oid: math.dist(
                    world.object_by_id(oid).pose[:2], slot_pos[:2]
                )
            )
            if remaining:
                ordered_ids.append(remaining.pop(0))
    return _build_grouped_align_plan(world, ordered_ids, slots, gt, "grouped_nearest_to_slot")


def generate_grouped_tidy_random_plan(
    world: WorldState,
    slots: dict[str, tuple[float, float, float]],
    seed: int | None = None,
) -> TaskPlan:
    gt = world.grouped_tidy
    if not gt or not gt.enabled:
        return None  # type: ignore[return-value]
    rng = random.Random(seed)
    ordered_ids: list[str] = []
    for group in gt.groups:
        group_objects = list(group.objects)
        rng.shuffle(group_objects)
        ordered_ids.extend(group_objects)
    return _build_grouped_align_plan(world, ordered_ids, slots, gt, "grouped_random")


def _build_grouped_align_plan(
    world: WorldState,
    ordered_object_ids: list[str],
    slots: dict[str, tuple[float, float, float]],
    gt: "GroupedTidyConfig",
    method: str,
) -> TaskPlan:
    from world.state import GroupedTidyConfig
    object_to_slot: dict[str, str] = {}
    for group in gt.groups:
        group_slot_ids = sorted(
            k for k in slots.keys() if k.startswith(f"{gt.slot_prefix}_{group.id}_")
        )
        for i, obj_id in enumerate(group.objects):
            if i < len(group_slot_ids):
                object_to_slot[obj_id] = group_slot_ids[i]

    steps: list[Step] = []
    for idx, object_id in enumerate(ordered_object_ids):
        slot_id = object_to_slot.get(object_id, "")
        step_id = idx * 2
        steps.append(Step(step_id=step_id, action="pick", object=object_id))
        steps.append(
            Step(step_id=step_id + 1, action="place", object=object_id, slot=slot_id)
        )

    goal_predicates: list[dict] = []
    for object_id in world.target_objects:
        slot_id = object_to_slot.get(object_id, "")
        goal_predicates.append({"name": "at", "args": [object_id, slot_id]})

    slot_config = SlotConfig(
        type="line",
        axis=gt.axis,
        spacing_m=gt.spacing,
        row_y=world.goal_center[1],
        center_x=world.goal_center[0],
        base_z=world.table_z_top + 0.033,
    )
    return TaskPlan(
        schema_version=SCHEMA_VERSION,
        task="align",
        scene_id=world.scene_id,
        target_objects=world.target_objects,
        goal_predicates=tuple(goal_predicates),
        slot_config=slot_config,
        steps=tuple(steps),
        constraints={
            "preserve_obstacles": True,
            "flexible_order": True,
            "grouped_tidy": True,
            "generation_method": method,
        },
    )


def _target_objects_with_poses(
    world: WorldState,
) -> list[tuple[str, tuple[float, float, float]]]:
    result = []
    for oid in world.target_objects:
        obj = world.object_by_id(oid)
        if obj is not None:
            result.append((oid, obj.pose))
    return result


def _slot_index_map(
    slots: dict[str, tuple[float, float, float]],
) -> dict[str, int]:
    sorted_slots = sorted(slots.keys())
    return {slot_id: idx for idx, slot_id in enumerate(sorted_slots)}


def _build_align_plan(
    world: WorldState,
    ordered_object_ids: Sequence[str],
    slots: dict[str, tuple[float, float, float]],
    method: str,
) -> TaskPlan:
    slot_ids = sorted(slots.keys())
    object_to_slot: dict[str, str] = {}
    for idx, object_id in enumerate(ordered_object_ids):
        object_to_slot[object_id] = slot_ids[idx]

    steps: list[Step] = []
    for idx, object_id in enumerate(ordered_object_ids):
        slot_id = object_to_slot[object_id]
        step_id = idx * 2
        steps.append(Step(step_id=step_id, action="pick", object=object_id))
        steps.append(
            Step(step_id=step_id + 1, action="place", object=object_id, slot=slot_id)
        )

    goal_predicates: list[dict] = []
    for object_id in world.target_objects:
        slot_id = object_to_slot[object_id]
        goal_predicates.append({"name": "at", "args": [object_id, slot_id]})

    slot_config = SlotConfig(
        type="line",
        axis="x",
        spacing_m=0.10,
        row_y=-0.06,
        center_x=0.22,
        base_z=0.833,
    )
    return TaskPlan(
        schema_version=SCHEMA_VERSION,
        task="align",
        scene_id=world.scene_id,
        target_objects=world.target_objects,
        goal_predicates=tuple(goal_predicates),
        slot_config=slot_config,
        steps=tuple(steps),
        constraints={
            "preserve_obstacles": True,
            "flexible_order": True,
            "generation_method": method,
        },
    )


def _validate_candidate(plan: TaskPlan, world: WorldState) -> bool:
    try:
        validate(plan, world.all_object_ids(), world.allowed_predicates)
        return True
    except PlanValidationError:
        return False


def _plan_ordering_key(plan: TaskPlan) -> str:
    pairs = []
    for step in plan.steps:
        if step.action == "pick":
            pairs.append(("pick", step.object))
        elif step.action == "place":
            pairs.append(("place", step.slot or ""))
    return str(pairs)
