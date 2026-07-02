from __future__ import annotations

import math
from pathlib import Path

import pytest

from backends.adaptive.event_log import EventLog
from backends.adaptive.hint_cache import HintCache
from execution.primitives import PrimitiveResult
from execution.runner import TaskRunner
from plugins.registry import DEFAULT_REGISTRY
from task_planning.candidate_generator import generate_align_candidates
from task_planning.validator import validate
from world.builder import build_world_state
from world.slot_allocator import allocate_grouped_align_slots

CONTEXT_PATH = "contexts/examples/align_grouped_tidy_gang.md"


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


@pytest.fixture
def world():
    return build_world_state(CONTEXT_PATH)


@pytest.fixture
def gt(world):
    return world.grouped_tidy


@pytest.fixture
def slots(world, gt):
    return allocate_grouped_align_slots(world, gt)


def test_context_loads_and_world_valid(world, gt):
    assert world.scene_id == "align_grouped_tidy_gang"
    assert gt is not None
    assert gt.enabled
    assert len(gt.groups) == 4
    assert len(world.target_objects) == 12


def test_candidate_generation_and_execution(tmp_path, world, gt, slots):
    candidates = generate_align_candidates(world, slots)
    assert len(candidates) > 0

    plan = candidates[0]
    validate(plan, world.all_object_ids(), world.allowed_predicates)
    plugin = DEFAULT_REGISTRY.get("align")
    plugin.validate_plan(plan, world)

    primitives = DeterministicPrimitives(
        {obj.id: obj.pose for obj in world.objects}
    )
    result = TaskRunner(
        plan=plan,
        world=world,
        slots=slots,
        hint_cache=HintCache(tmp_path / "history"),
        plugin_registry=DEFAULT_REGISTRY,
        event_log=EventLog(tmp_path / "events.csv", "grouped-tidy-test"),
        primitives=primitives,
    ).run()

    assert result.success
    assert result.moved_count == 12
    assert not result.failure_reasons


def test_grouped_tidy_goal_verified(tmp_path, world, gt, slots):
    candidates = generate_align_candidates(world, slots)
    plan = candidates[0]

    primitives = DeterministicPrimitives(
        {obj.id: obj.pose for obj in world.objects}
    )
    result = TaskRunner(
        plan=plan,
        world=world,
        slots=slots,
        hint_cache=HintCache(tmp_path / "history"),
        plugin_registry=DEFAULT_REGISTRY,
        event_log=EventLog(tmp_path / "events.csv", "grouped-tidy-goal"),
        primitives=primitives,
    ).run()

    assert result.success

    object_to_slot = {}
    for group in gt.groups:
        group_slot_ids = sorted(
            k for k in slots if k.startswith(f"tidy_slot_{group.id}_")
        )
        for i, obj_id in enumerate(group.objects):
            object_to_slot[obj_id] = group_slot_ids[i]

    for obj_id in world.target_objects:
        pose = primitives.object_pose(obj_id)
        slot_pose = slots[object_to_slot[obj_id]]
        assert math.dist(pose[:2], slot_pose[:2]) < 0.06

    assert primitives.held_object_name() is None


def test_all_objects_stable_after_execution(tmp_path, world, gt, slots):
    candidates = generate_align_candidates(world, slots)
    plan = candidates[0]

    primitives = DeterministicPrimitives(
        {obj.id: obj.pose for obj in world.objects}
    )

    result = TaskRunner(
        plan=plan,
        world=world,
        slots=slots,
        hint_cache=HintCache(tmp_path / "history"),
        plugin_registry=DEFAULT_REGISTRY,
        event_log=EventLog(tmp_path / "events.csv", "stability-test"),
        primitives=primitives,
    ).run()

    assert result.success
    for obj in world.objects:
        pose = primitives.object_pose(obj.id)
        assert pose[2] > world.table_z_top - 0.01


def test_baseline_align_context_unchanged():
    baseline = build_world_state("contexts/examples/ungroup_obs_align_cubes.md")
    assert baseline.grouped_tidy is None or not baseline.grouped_tidy.enabled
    assert baseline.task_name == "align"
    assert len(baseline.target_objects) == 4
