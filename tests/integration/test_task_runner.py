from __future__ import annotations

import pytest

from backends.adaptive.event_log import EventLog
from backends.adaptive.hint_cache import HintCache
from execution.primitives import PrimitiveResult
from execution.runner import TaskRunner
from task_planning.loader import parse_plan
from task_planning.validator import validate
from plugins.registry import DEFAULT_REGISTRY
from world.slot_allocator import allocate_slots
from world.state import ObjectState, WorldState


class FakePrimitives:
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


def make_world(objects, *, task, target_objects):
    return WorldState(
        scene_id="unit_scene",
        variant="group_no_obs",
        objects=tuple(objects),
        obstacles=(),
        table_x_range=(-0.55, 0.55),
        table_y_range=(-0.75, 0.75),
        table_z_top=0.80,
        goal_center=(0.22, -0.06, 0.806),
        robot_id="panda_left",
        robot_base_xy=(-0.4, 0.0),
        robot_reach_min=0.30,
        robot_reach_max=0.82,
        robot_capabilities=("pick", "place", "stack_place"),
        task_name=task,
        target_objects=tuple(target_objects),
        task_description="integration test",
        preserve_obstacles=True,
        max_retries_per_object=1,
        allowed_predicates=(
            "at", "on", "clear", "handempty", "holding", "aligned-row"
        ),
    )


def run_plan(tmp_path, payload, world):
    plan = parse_plan(payload)
    validate(plan, world.all_object_ids(), world.allowed_predicates)
    plugin = DEFAULT_REGISTRY.get(plan.task)
    plugin.validate_plan(plan, world)
    slots = allocate_slots(plugin.make_slot_config(plan, world), len(plan.target_objects))
    primitives = FakePrimitives({obj.id: obj.pose for obj in world.objects})
    runner = TaskRunner(
        plan,
        world,
        slots,
        HintCache(tmp_path / "history"),
        DEFAULT_REGISTRY,
        EventLog(tmp_path / "events.csv", "test-run"),
        primitives,
    )
    return runner.run(), primitives


def test_align_one_object_end_to_end_without_mujoco(tmp_path):
    world = make_world(
        [ObjectState("cube1", "cube", (0.10, -0.30, 0.83), True, False)],
        task="align",
        target_objects=["cube1"],
    )
    payload = {
        "schema_version": "ctamp-plan/v1",
        "task": "align",
        "scene_id": "unit_scene",
        "target_objects": ["cube1"],
        "goal_predicates": [{"name": "at", "args": ["cube1", "slot_0"]}],
        "slot_config": {"type": "line", "center_x": 0.22, "row_y": -0.06},
        "steps": [
            {"step_id": 0, "action": "pick", "object": "cube1"},
            {"step_id": 1, "action": "place", "object": "cube1", "slot": "slot_0"},
        ],
    }
    result, primitives = run_plan(tmp_path, payload, world)
    assert result.success
    assert result.moved_count == 1
    assert primitives.object_pose("cube1") == (0.22, -0.06, 0.83)


def test_stack_two_cubes_uses_live_support_pose(tmp_path):
    world = make_world(
        [
            ObjectState("cube1", "cube", (0.00, -0.30, 0.83), True, False),
            ObjectState("cube2", "cube", (0.10, -0.30, 0.83), True, False),
        ],
        task="stack",
        target_objects=["cube1", "cube2"],
    )
    payload = {
        "schema_version": "ctamp-plan/v1",
        "task": "stack",
        "scene_id": "unit_scene",
        "target_objects": ["cube1", "cube2"],
        "goal_predicates": [{"name": "on", "args": ["cube2", "cube1"]}],
        "slot_config": {
            "type": "tower",
            "base_xy": [0.22, -0.06],
            "base_z": 0.83,
            "layer_height_m": 0.06,
        },
        "steps": [
            {"step_id": 0, "action": "pick", "object": "cube1"},
            {"step_id": 1, "action": "place", "object": "cube1", "slot": "tower_base"},
            {"step_id": 2, "action": "pick", "object": "cube2"},
            {"step_id": 3, "action": "stack_place", "object": "cube2", "on_top_of": "cube1"},
        ],
    }
    result, primitives = run_plan(tmp_path, payload, world)
    assert result.success
    assert result.moved_count == 2
    assert primitives.object_pose("cube2") == pytest.approx((0.22, -0.06, 0.89))


def test_stack_upper_levels_use_observed_base_axis_without_xy_drift(tmp_path):
    world = make_world(
        [
            ObjectState("cube1", "cube", (0.00, -0.30, 0.83), True, False),
            ObjectState("cube2", "cube", (0.10, -0.30, 0.83), True, False),
            ObjectState("cube3", "cube", (0.20, -0.30, 0.83), True, False),
        ],
        task="stack",
        target_objects=["cube1", "cube2", "cube3"],
    )
    payload = {
        "schema_version": "ctamp-plan/v1",
        "task": "stack",
        "scene_id": "unit_scene",
        "target_objects": ["cube1", "cube2", "cube3"],
        "goal_predicates": [
            {"name": "at", "args": ["cube1", "tower_base"]},
            {"name": "on", "args": ["cube2", "cube1"]},
            {"name": "on", "args": ["cube3", "cube2"]},
        ],
        "slot_config": {
            "type": "tower",
            "base_xy": [0.22, -0.06],
            "base_z": 0.83,
            "layer_height_m": 0.06,
        },
        "steps": [
            {"step_id": 0, "action": "pick", "object": "cube1"},
            {"step_id": 1, "action": "place", "object": "cube1", "slot": "tower_base"},
            {"step_id": 2, "action": "pick", "object": "cube2"},
            {"step_id": 3, "action": "stack_place", "object": "cube2", "on_top_of": "cube1"},
            {"step_id": 4, "action": "pick", "object": "cube3"},
            {"step_id": 5, "action": "stack_place", "object": "cube3", "on_top_of": "cube2"},
        ],
    }
    plan = parse_plan(payload)
    plugin = DEFAULT_REGISTRY.get("stack")
    slots = allocate_slots(plugin.make_slot_config(plan, world), 3)
    primitives = FakePrimitives({obj.id: obj.pose for obj in world.objects})
    runner = TaskRunner(
        plan,
        world,
        slots,
        HintCache(tmp_path / "history"),
        DEFAULT_REGISTRY,
        EventLog(tmp_path / "events.csv", "axis-test"),
        primitives,
    )

    primitives.poses["cube1"] = (0.215, -0.057, 0.83)
    primitives.poses["cube2"] = (0.225, -0.050, 0.89)
    target = runner._resolve_target(plan.steps[-1])

    assert target == pytest.approx((0.215, -0.057, 0.95))
