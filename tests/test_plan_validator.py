from __future__ import annotations

import pytest

from task_planning.loader import PlanLoadError, parse_plan
from task_planning.validator import PlanValidationError, validate


def valid_align_payload():
    return {
        "schema_version": "ctamp-plan/v1",
        "task": "align",
        "scene_id": "group_no_obs",
        "target_objects": ["cube1", "cube2"],
        "goal_predicates": [
            {"name": "at", "args": ["cube1", "align_slot_0"]},
            {"name": "at", "args": ["cube2", "align_slot_1"]},
        ],
        "slot_config": {
            "type": "line",
            "axis": "x",
            "spacing_m": 0.125,
            "row_y": -0.06,
            "center_x": 0.22,
            "base_z": 0.83,
        },
        "steps": [
            {"step_id": 0, "action": "pick", "object": "cube1"},
            {
                "step_id": 1,
                "action": "place",
                "object": "cube1",
                "slot": "align_slot_0",
            },
            {"step_id": 2, "action": "pick", "object": "cube2"},
            {
                "step_id": 3,
                "action": "place",
                "object": "cube2",
                "slot": "align_slot_1",
            },
        ],
        "constraints": {"preserve_obstacles": True},
    }


def test_valid_align_plan_passes_all_gates():
    plan = parse_plan(valid_align_payload())
    validate(plan, {"cube1", "cube2"})


def test_unknown_schema_field_is_rejected():
    payload = valid_align_payload()
    payload["joint_trajectory"] = [0, 1, 2]
    with pytest.raises(PlanLoadError, match="unsupported fields"):
        parse_plan(payload)


def test_fictional_object_is_rejected_at_gate_2():
    payload = valid_align_payload()
    payload["steps"][0]["object"] = "cube404"
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="gate 2"):
        validate(plan, {"cube1", "cube2"})


def test_fictional_object_in_goal_is_rejected_at_gate_2():
    payload = valid_align_payload()
    payload["goal_predicates"][0] = {
        "name": "at",
        "args": ["invented_cube", "align_slot_0"],
    }
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="invented_cube"):
        validate(plan, {"cube1", "cube2"})


def test_unknown_predicate_is_rejected_at_gate_3():
    payload = valid_align_payload()
    payload["goal_predicates"][0]["name"] = "teleported"
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="gate 3"):
        validate(plan, {"cube1", "cube2"})


def test_pick_without_place_is_rejected_at_gate_4():
    payload = valid_align_payload()
    payload["target_objects"] = ["cube1"]
    payload["goal_predicates"] = [{"name": "holding", "args": ["cube1"]}]
    payload["steps"] = [{"step_id": 0, "action": "pick", "object": "cube1"}]
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="gate 4"):
        validate(plan, {"cube1", "cube2"})


def test_unsupported_action_is_rejected_at_gate_1():
    payload = valid_align_payload()
    payload["steps"][1]["action"] = "teleport"
    plan = parse_plan(payload)
    with pytest.raises(PlanValidationError, match="unsupported action"):
        validate(plan, {"cube1", "cube2"})
