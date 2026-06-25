from __future__ import annotations

import json
from pathlib import Path

import pytest

from backends.adaptive.event_log import EventLog
from backends.adaptive.hint_cache import HintCache
from configuration import load_runtime_config
from execution.primitives import PrimitiveResult
from execution.runner import TaskRunner
from plugins.registry import DEFAULT_REGISTRY
from task_planning.loader import load_plan, parse_plan
from task_planning.validator import PlanValidationError, validate
from world.builder import build_world_state
from world.slot_allocator import allocate_slots, validate_slots


ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "contexts/examples/ungroup_obs_pyramid_cubes.md"
PLAN = ROOT / "task_plans/examples/ungroup_obs_pyramid_cubes.json"


class DeterministicPrimitives:
    def __init__(self, poses):
        self.poses = dict(poses)
        self.held = None

    def execute(self, step, target, hints):
        if step.action == "pick":
            x, y, _ = self.poses[step.object]
            self.poses[step.object] = (x, y, 0.96)
            self.held = step.object
        else:
            assert target is not None
            self.poses[step.object] = target
            self.held = None
        return PrimitiveResult(True)

    def object_pose(self, object_id):
        return self.poses[object_id]

    def all_object_poses(self):
        return dict(self.poses)

    def held_object_name(self):
        return self.held


def _slot_order(row_count: int, base_row_length: int) -> list[str]:
    return [
        f"row{row}_col{column}"
        for row in range(row_count)
        for column in range(base_row_length - row)
    ]


def test_pyramid_plan_validator_accepts_reference_plan():
    world = build_world_state(CONTEXT)
    plan = load_plan(PLAN)

    assert world.task_name == plan.task == "pyramid"
    assert len(world.target_objects) == len(plan.target_objects) == 6
    validate(plan, world.all_object_ids(), world.allowed_predicates)
    DEFAULT_REGISTRY.get("pyramid").validate_plan(plan, world)

    slots = allocate_slots(plan.slot_config, len(plan.target_objects))
    validate_slots(
        slots,
        world,
        obstacle_buffer_m=load_runtime_config("obstacle").safety.target_obstacle_buffer_m,
    )


def test_pyramid_rejects_stack_place():
    world = build_world_state(CONTEXT)
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["steps"][3] = {
        "step_id": 3,
        "action": "stack_place",
        "object": "cube2",
        "on_top_of": "cube1",
    }
    plan = parse_plan(payload)

    validate(plan, world.all_object_ids(), world.allowed_predicates)
    with pytest.raises(PlanValidationError, match="does not support actions"):
        DEFAULT_REGISTRY.get("pyramid").validate_plan(plan, world)


def test_pyramid_6_6(tmp_path):
    world = build_world_state(CONTEXT)
    plan = load_plan(PLAN)
    validate(plan, world.all_object_ids(), world.allowed_predicates)
    plugin = DEFAULT_REGISTRY.get("pyramid")
    plugin.validate_plan(plan, world)
    slots = allocate_slots(plugin.make_slot_config(plan, world), len(plan.target_objects))

    primitives = DeterministicPrimitives({obj.id: obj.pose for obj in world.objects})
    result = TaskRunner(
        plan=plan,
        world=world,
        slots=slots,
        hint_cache=HintCache(tmp_path / "history"),
        plugin_registry=DEFAULT_REGISTRY,
        event_log=EventLog(tmp_path / "events.csv", "pyramid-6-6"),
        primitives=primitives,
        runtime_config=load_runtime_config("obstacle"),
    ).run()

    assert result.success
    assert result.moved_count == 6
    assert result.failure_reasons == ()

    rows = []
    for object_id, slot_id in zip(
        plan.target_objects,
        _slot_order(plan.slot_config.row_count, plan.slot_config.base_row_length),
    ):
        target = slots[slot_id]
        actual = primitives.object_pose(object_id)
        ok = (
            abs(actual[0] - target[0]) <= 0.015
            and abs(actual[1] - target[1]) <= 0.015
            and abs(actual[2] - target[2]) <= 0.015
        )
        rows.append((object_id, slot_id, target, actual, ok))
        assert ok, f"{object_id} missed {slot_id}: target={target}, actual={actual}"

    for object_id, slot_id, target, actual, ok in rows:
        marker = "✓" if ok else "✗"
        print(
            f"{object_id:6} {slot_id:9} "
            f"target=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) "
            f"actual=({actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f}) {marker}"
        )
    print(f"RESULT: {sum(row[-1] for row in rows)}/6 ✓")
