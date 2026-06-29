from __future__ import annotations

import pytest

from task_planning.loader import PlanLoadError, parse_plan
from task_planning.validator import PlanValidationError, validate


def valid_stack_payload():
    return {
        "schema_version": "ctamp-plan/v1",
        "task": "stack",
        "scene_id": "group_no_obs",
        "target_objects": ["cube1", "cube2"],
        "goal_predicates": [
            {"name": "at", "args": ["cube1", "tower_base"]},
            {"name": "on", "args": ["cube2", "cube1"]},
        ],
        "slot_config": {
            "type": "tower",
            "base_xy": [0.22, -0.06],
            "base_z": 0.83,
            "layer_height_m": 0.06,
        },
        "steps": [
            {"step_id": 0, "action": "pick", "object": "cube1"},
            {
                "step_id": 1,
                "action": "place",
                "object": "cube1",
                "slot": "tower_base",
            },
            {"step_id": 2, "action": "pick", "object": "cube2"},
            {
                "step_id": 3,
                "action": "stack_place",
                "object": "cube2",
                "on_top_of": "cube1",
            },
        ],
        "constraints": {"preserve_obstacles": True},
    }


def test_valid_stack_plan_passes_all_gates():
    plan = parse_plan(valid_stack_payload())
    validate(plan, {"cube1", "cube2"})


def test_unknown_schema_field_is_rejected():
    payload = valid_stack_payload()
    payload["joint_trajectory"] = [0, 1, 2]
    with pytest.raises(PlanLoadError, match="unsupported fields"):
        parse_plan(payload)


def test_fictional_object_is_rejected_at_gate_2():
    payload = valid_stack_payload()
    payload["steps"][0]["object"] = "cube404"
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="gate 2"):
        validate(plan, {"cube1", "cube2"})


def test_fictional_object_in_goal_is_rejected_at_gate_2():
    payload = valid_stack_payload()
    payload["goal_predicates"][0] = {
        "name": "at",
        "args": ["invented_cube", "tower_base"],
    }
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="invented_cube"):
        validate(plan, {"cube1", "cube2"})


def test_unknown_predicate_is_rejected_at_gate_3():
    payload = valid_stack_payload()
    payload["goal_predicates"][0]["name"] = "teleported"
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="gate 3"):
        validate(plan, {"cube1", "cube2"})


def test_pick_without_place_is_rejected_at_gate_4():
    payload = valid_stack_payload()
    payload["target_objects"] = ["cube1"]
    payload["goal_predicates"] = [{"name": "holding", "args": ["cube1"]}]
    payload["steps"] = [{"step_id": 0, "action": "pick", "object": "cube1"}]
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="gate 4"):
        validate(plan, {"cube1", "cube2"})


def test_stack_dependency_must_already_be_placed():
    payload = valid_stack_payload()
    payload["steps"][0] = {"step_id": 0, "action": "pick", "object": "cube2"}
    payload["steps"][1] = {
        "step_id": 1,
        "action": "stack_place",
        "object": "cube2",
        "on_top_of": "cube1",
    }
    payload["steps"][2] = {"step_id": 2, "action": "pick", "object": "cube1"}
    payload["steps"][3] = {
        "step_id": 3,
        "action": "place",
        "object": "cube1",
        "slot": "tower_base",
    }
    with pytest.raises(PlanValidationError, match="support object"):
        validate(parse_plan(payload), {"cube1", "cube2"})
