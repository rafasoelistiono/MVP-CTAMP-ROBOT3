from __future__ import annotations

import math

import pytest

from world.builder import build_world_state
from world.slot_allocator import allocate_grouped_align_slots, allocate_slots
from world.state import GroupedTidyConfig, ObstacleState, TidyGroup, WorldState


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


def test_grouped_slots_generated_for_all_12_objects(slots):
    assert len(slots) == 12


def test_exactly_4_groups(slots, gt):
    group_ids = {group.id for group in gt.groups}
    assert len(group_ids) == 4


def test_every_group_has_3_slots(slots, gt):
    for group in gt.groups:
        group_slots = [
            k for k in slots if k.startswith(f"tidy_slot_{group.id}_")
        ]
        assert len(group_slots) == 3


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
    obstacle_buffer = 0.13
    for slot_id, (x, y, z) in slots.items():
        for obstacle in world.obstacles:
            clearance = math.dist((x, y), obstacle.pose[:2])
            assert clearance >= obstacle.radius + obstacle_buffer, (
                f"{slot_id} violates obstacle {obstacle.id} region"
            )


def test_group_order_correct(slots, gt):
    group_ids = [group.id for group in gt.groups]
    assert "green_top" in group_ids
    assert "red_bottom" in group_ids
    assert "yellow_top" in group_ids
    assert "blue_bottom" in group_ids


def test_tidy_groups_stay_on_opposite_wall_sides(slots, world, gt):
    left_wall_edge = min(obs.pose[0] - obs.radius for obs in world.obstacles)
    right_wall_edge = max(obs.pose[0] + obs.radius for obs in world.obstacles)
    left_groups = {"green_top", "red_bottom"}
    right_groups = {"yellow_top", "blue_bottom"}
    for group in gt.groups:
        xs = [
            pose[0]
            for slot_id, pose in slots.items()
            if slot_id.startswith(f"{gt.slot_prefix}_{group.id}_")
        ]
        if group.id in left_groups:
            assert max(xs) <= left_wall_edge + 0.02
        elif group.id in right_groups:
            assert min(xs) >= right_wall_edge
        else:
            pytest.fail(f"unexpected group {group.id}")


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
