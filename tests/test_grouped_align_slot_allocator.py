from __future__ import annotations

import math

import pytest

from world.builder import build_world_state
from world.slot_allocator import allocate_grouped_align_slots, allocate_slots, validate_slots
from world.state import GroupedTidyConfig, ObstacleState, TidyGroup, WorldState


CONTEXT_PATH = "contexts/examples/align_grouped_tidy_wall_world.md"


@pytest.fixture
def world():
    return build_world_state(CONTEXT_PATH)


@pytest.fixture
def gt(world):
    return world.grouped_tidy


@pytest.fixture
def slots(world, gt):
    return allocate_grouped_align_slots(world, gt)


def test_grouped_slots_generated_for_all_12_objects(slots):
    assert len(slots) == 12


def test_exactly_2_groups(slots, gt):
    group_ids = {group.id for group in gt.groups}
    assert group_ids == {"blue_lane", "red_lane"}


def test_every_group_has_6_slots(slots, gt):
    for group in gt.groups:
        group_slots = [
            k for k in slots if k.startswith(f"tidy_slot_{group.id}_")
        ]
        assert len(group_slots) == 6


def test_slots_inside_table_bounds(slots, world):
    for slot_id, (x, y, z) in slots.items():
        assert world.table_x_range[0] < x < world.table_x_range[1], slot_id
        assert world.table_y_range[0] < y < world.table_y_range[1], slot_id


def test_slots_are_reachable(slots, world):
    for slot_id, (x, y, z) in slots.items():
        dist = math.dist((x, y), world.robot_base_xy)
        assert world.robot_reach_min <= dist <= world.robot_reach_max, slot_id


def test_slots_do_not_overlap(slots):
    poses = list(slots.values())
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            distance = math.dist(poses[i][:2], poses[j][:2])
            assert distance >= 0.066, f"physical overlap at distance {distance}"


def test_slots_not_inside_obstacle_region(slots, world):
    validate_slots(slots, world, obstacle_buffer_m=0.13)


def test_group_order_correct(slots, gt):
    group_ids = [group.id for group in gt.groups]
    assert group_ids == ["blue_lane", "red_lane"]


def test_tidy_groups_stay_in_two_right_side_y_lanes(slots, world, gt):
    wall = world.obstacles[0]
    right_edge = wall.pose[0] + wall.size[0] / 2.0 + 0.13
    lane_x = {"blue_lane": 0.22, "red_lane": 0.36}
    for group in gt.groups:
        xs = [
            pose[0]
            for slot_id, pose in slots.items()
            if slot_id.startswith(f"{gt.slot_prefix}_{group.id}_")
        ]
        ys = [
            pose[1]
            for slot_id, pose in slots.items()
            if slot_id.startswith(f"{gt.slot_prefix}_{group.id}_")
        ]
        assert min(xs) > right_edge
        assert set(round(x, 2) for x in xs) == {lane_x[group.id]}
        assert max(ys) - min(ys) == pytest.approx(0.34)


def test_baseline_align_slots_unchanged():
    from task_planning.types import SlotConfig

    config = SlotConfig(
        type="line",
        spacing_m=0.125,
        center_x=0.22,
        row_y=-0.06,
        base_z=0.83,
    )
    slots = allocate_slots(config, 4)
    assert list(slots.keys()) == [
        "align_slot_0",
        "align_slot_1",
        "align_slot_2",
        "align_slot_3",
    ]
