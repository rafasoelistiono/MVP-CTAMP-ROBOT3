from __future__ import annotations

import pytest

from plugins.registry import DEFAULT_REGISTRY
from task_planning.loader import parse_plan
from task_planning.types import SlotConfig
from task_planning.validator import PlanValidationError, validate
from world.builder import build_world_state
from world.slot_allocator import allocate_grouped_align_slots

CONTEXT_PATH = "contexts/examples/align_grouped_tidy_gang.md"


@pytest.fixture
def world():
    return build_world_state(CONTEXT_PATH)


@pytest.fixture
def gt(world):
    return world.grouped_tidy


@pytest.fixture
def slots(world, gt):
    return allocate_grouped_align_slots(world, gt)


def _make_plan(world, gt, slots, overrides=None):
    object_to_slot = {}
    for group in gt.groups:
        group_slot_ids = sorted(
            k for k in slots if k.startswith(f"tidy_slot_{group.id}_")
        )
        for i, obj_id in enumerate(group.objects):
            object_to_slot[obj_id] = group_slot_ids[i]

    steps = []
    for idx, obj_id in enumerate(world.target_objects):
        slot_id = object_to_slot[obj_id]
        steps.append({"step_id": idx * 2, "action": "pick", "object": obj_id})
        steps.append(
            {"step_id": idx * 2 + 1, "action": "place", "object": obj_id, "slot": slot_id}
        )

    goal_predicates = [
        {"name": "at", "args": [obj_id, object_to_slot[obj_id]]}
        for obj_id in world.target_objects
    ]

    payload = {
        "schema_version": "ctamp-plan/v1",
        "task": "align",
        "scene_id": world.scene_id,
        "target_objects": list(world.target_objects),
        "goal_predicates": goal_predicates,
        "slot_config": {
            "type": "line",
            "axis": gt.axis,
            "spacing_m": gt.spacing,
            "row_y": world.goal_center[1],
            "center_x": world.goal_center[0],
            "base_z": world.table_z_top + 0.033,
        },
        "steps": steps,
        "constraints": {"preserve_obstacles": True, "flexible_order": True},
    }
    if overrides:
        payload.update(overrides)
    return payload


def test_valid_plan_passes(world, gt, slots):
    payload = _make_plan(world, gt, slots)
    plan = parse_plan(payload)
    validate(plan, world.all_object_ids(), world.allowed_predicates)
    plugin = DEFAULT_REGISTRY.get("align")
    plugin.validate_plan(plan, world)


def test_wrong_group_assignment_rejected(world, gt, slots):
    payload = _make_plan(world, gt, slots)
    payload["steps"][0]["slot"] = "tidy_slot_green_top_0"
    payload["steps"][0]["object"] = "a"
    payload["steps"][1]["object"] = "a"
    payload["steps"][1]["slot"] = "tidy_slot_green_top_0"
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError):
        plugin = DEFAULT_REGISTRY.get("align")
        plugin.validate_plan(plan, world)


def test_duplicate_slot_rejected(world, gt, slots):
    payload = _make_plan(world, gt, slots)
    payload["steps"][1]["slot"] = "tidy_slot_green_top_0"
    payload["steps"][3]["slot"] = "tidy_slot_green_top_0"
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="duplicate slot"):
        plugin = DEFAULT_REGISTRY.get("align")
        plugin.validate_plan(plan, world)


def test_missing_object_rejected(world, gt, slots):
    payload = _make_plan(world, gt, slots)
    payload["target_objects"] = payload["target_objects"][:-1]
    payload["steps"] = payload["steps"][:-2]
    payload["goal_predicates"] = payload["goal_predicates"][:-1]
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError):
        plugin = DEFAULT_REGISTRY.get("align")
        plugin.validate_plan(plan, world)


def test_unknown_object_rejected(world, gt, slots):
    payload = _make_plan(world, gt, slots)
    payload["steps"][0]["object"] = "unknown_obj"
    payload["steps"][1]["object"] = "unknown_obj"
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError):
        plugin = DEFAULT_REGISTRY.get("align")
        plugin.validate_plan(plan, world)


def test_stack_place_action_rejected(world, gt, slots):
    payload = _make_plan(world, gt, slots)
    payload["steps"][1]["action"] = "stack_place"
    payload["steps"][1].pop("slot", None)
    payload["steps"][1]["on_top_of"] = "b"
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="does not support"):
        plugin = DEFAULT_REGISTRY.get("align")
        plugin.validate_plan(plan, world)


def test_pick_while_holding_rejected(world, gt, slots):
    payload = _make_plan(world, gt, slots)
    payload["steps"] = [
        {"step_id": 0, "action": "pick", "object": "a"},
        {"step_id": 1, "action": "pick", "object": "b"},
    ]
    payload["target_objects"] = ["a", "b"]
    payload["goal_predicates"] = []
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="gate 4"):
        validate(plan, world.all_object_ids(), world.allowed_predicates)


def test_place_without_pick_rejected(world, gt, slots):
    payload = _make_plan(world, gt, slots)
    payload["steps"] = [
        {"step_id": 0, "action": "place", "object": "a", "slot": "tidy_slot_green_top_0"},
    ]
    payload["target_objects"] = ["a"]
    payload["goal_predicates"] = []
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="gate 4"):
        validate(plan, world.all_object_ids(), world.allowed_predicates)


def test_plan_ends_while_holding_rejected(world, gt, slots):
    payload = _make_plan(world, gt, slots)
    payload["steps"] = [
        {"step_id": 0, "action": "pick", "object": "a"},
    ]
    payload["target_objects"] = ["a"]
    payload["goal_predicates"] = []
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="gate 4"):
        validate(plan, world.all_object_ids(), world.allowed_predicates)
