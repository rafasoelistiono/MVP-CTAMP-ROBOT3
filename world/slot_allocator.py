from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .state import GroupedTidyConfig, TidyGroup, WorldState


WALL_SLOT_CLEARANCE_M = 0.20


@dataclass(frozen=True)
class SlotConfig:
    type: Literal["line"]
    axis: str = "x"
    spacing_m: float = 0.125
    row_y: float = -0.06
    center_x: float = 0.22
    base_z: float = 0.83


class SlotAllocationError(ValueError):
    pass


def allocate_slots(
    config: SlotConfig,
    n: int,
) -> dict[str, tuple[float, float, float]]:
    if n <= 0:
        raise SlotAllocationError("slot count must be positive")
    if config.type == "line":
        return _allocate_line(config, n)
    raise SlotAllocationError(f"unknown slot type: {config.type}")


def _allocate_line(
    config: SlotConfig,
    n: int,
) -> dict[str, tuple[float, float, float]]:
    if config.axis != "x":
        raise SlotAllocationError("only line axis 'x' is currently supported")
    total_width = (n - 1) * config.spacing_m
    start_x = config.center_x - total_width / 2.0
    return {
        f"align_slot_{index}": (
            start_x + index * config.spacing_m,
            config.row_y,
            config.base_z,
        )
        for index in range(n)
    }


def validate_slots(
    slots: dict[str, tuple[float, float, float]],
    world: WorldState,
    obstacle_buffer_m: float = 0.13,
) -> None:
    for slot_id, pose in slots.items():
        x, y, z = pose
        if not (world.table_x_range[0] < x < world.table_x_range[1]):
            raise SlotAllocationError(f"{slot_id} x={x:.4f} is outside table bounds")
        if not (world.table_y_range[0] < y < world.table_y_range[1]):
            raise SlotAllocationError(f"{slot_id} y={y:.4f} is outside table bounds")
        distance = math.dist((x, y), world.robot_base_xy)
        if not world.robot_reach_min <= distance <= world.robot_reach_max:
            raise SlotAllocationError(
                f"{slot_id} is outside robot reach: distance={distance:.4f}"
            )
        if z < world.table_z_top:
            raise SlotAllocationError(
                f"{slot_id} z={z:.4f} is below table top {world.table_z_top:.4f}"
            )
        goal_half_x = world.goal_area_size_xy[0] / 2.0
        goal_half_y = world.goal_area_size_xy[1] / 2.0
        goal_x, goal_y, _ = world.goal_center
        if not (goal_x - goal_half_x <= x <= goal_x + goal_half_x):
            raise SlotAllocationError(
                f"{slot_id} is outside goal area x bounds: coordinate=({x:.4f}, {y:.4f}, {z:.4f})"
            )
        if not (goal_y - goal_half_y <= y <= goal_y + goal_half_y):
            raise SlotAllocationError(
                f"{slot_id} is outside goal area y bounds: coordinate=({x:.4f}, {y:.4f}, {z:.4f})"
            )
        for obstacle in world.obstacles:
            if _inside_obstacle_clearance((x, y), obstacle, obstacle_buffer_m):
                raise SlotAllocationError(
                    f"{slot_id} violates inflated obstacle region for {obstacle.id}: "
                    f"coordinate=({x:.4f}, {y:.4f}, {z:.4f})"
                )


def allocate_grouped_align_slots(
    world: WorldState,
    config: GroupedTidyConfig,
) -> dict[str, tuple[float, float, float]]:
    """Allocate slots for grouped tidy align variant with obstacle-aware Y offsets."""
    obstacle_buffer = 0.13
    y_offsets = [
        0.0,
        0.04,
        -0.04,
        0.08,
        -0.08,
        0.12,
        -0.12,
        0.16,
        -0.16,
        0.20,
        -0.20,
        0.24,
        -0.24,
        0.28,
        -0.28,
    ]
    object_height = 0.066
    minimum_slot_distance = 0.066
    z = world.table_z_top + object_height / 2.0

    slots: dict[str, tuple[float, float, float]] = {}
    for group in config.groups:
        group_slots = _allocate_group_with_offset(
            group,
            config,
            world,
            obstacle_buffer,
            y_offsets,
            z,
            slots,
            minimum_slot_distance,
        )
        slots.update(group_slots)

    _check_slot_overlap(slots, minimum_slot_distance)
    return slots


def _allocate_group_with_offset(
    group: TidyGroup,
    config: GroupedTidyConfig,
    world: WorldState,
    obstacle_buffer: float,
    y_offsets: list[float],
    z: float,
    existing_slots: dict[str, tuple[float, float, float]],
    minimum_slot_distance: float,
) -> dict[str, tuple[float, float, float]]:
    existing_poses = tuple(existing_slots.values())
    for y_offset in y_offsets:
        candidate = allocate_group_row_slots(
            group.id,
            group.objects,
            group.center,
            config.axis,
            config.spacing,
            y_offset,
            z,
            config.slot_prefix,
        )
        separated = all(
            math.dist(pose[:2], existing[:2]) >= minimum_slot_distance
            for pose in candidate.values()
            for existing in existing_poses
        )
        if _all_slots_valid(candidate, world, obstacle_buffer) and separated:
            return candidate
    raise SlotAllocationError(
        f"No valid grouped align slot found for group {group.id!r} "
        f"due to dual tall obstacle inflated region."
    )


def allocate_group_row_slots(
    group_id: str,
    objects: tuple[str, ...],
    center: tuple[float, float, float],
    axis: str,
    spacing: float,
    y_offset: float,
    z: float,
    slot_prefix: str = "tidy",
) -> dict[str, tuple[float, float, float]]:
    """Allocate slots for one group as a horizontal row along axis."""
    n = len(objects)
    cx, cy, _ = center
    slots: dict[str, tuple[float, float, float]] = {}
    for i, obj_id in enumerate(objects):
        if axis == "x":
            x = cx + (i - (n - 1) / 2.0) * spacing
            y = cy + y_offset
        else:
            x = cx + y_offset
            y = cy + (i - (n - 1) / 2.0) * spacing
        slots[f"{slot_prefix}_{group_id}_{i}"] = (x, y, z)
    return slots


def _all_slots_valid(
    slots: dict[str, tuple[float, float, float]],
    world: WorldState,
    obstacle_buffer: float,
) -> bool:
    for slot_id, (x, y, z) in slots.items():
        if not (world.table_x_range[0] < x < world.table_x_range[1]):
            return False
        if not (world.table_y_range[0] < y < world.table_y_range[1]):
            return False
        distance = math.dist((x, y), world.robot_base_xy)
        if not world.robot_reach_min <= distance <= world.robot_reach_max:
            return False
        if z < world.table_z_top:
            return False
        goal_half_x = world.goal_area_size_xy[0] / 2.0
        goal_half_y = world.goal_area_size_xy[1] / 2.0
        goal_x, goal_y, _ = world.goal_center
        if not (goal_x - goal_half_x <= x <= goal_x + goal_half_x):
            return False
        if not (goal_y - goal_half_y <= y <= goal_y + goal_half_y):
            return False
        for obstacle in world.obstacles:
            if _inside_obstacle_clearance((x, y), obstacle, obstacle_buffer):
                return False
    return True


def _inside_obstacle_clearance(
    xy: tuple[float, float],
    obstacle,
    buffer: float,
) -> bool:
    if getattr(obstacle, "kind", "obstacle") == "wall":
        if obstacle.size is None:
            raise SlotAllocationError(
                f"wall obstacle {obstacle.id!r} requires explicit AABB size"
            )
        half_x, half_y, _ = (value / 2.0 for value in obstacle.size)
        wall_buffer = max(buffer, WALL_SLOT_CLEARANCE_M)
        return (
            abs(xy[0] - obstacle.pose[0]) < half_x + wall_buffer
            and abs(xy[1] - obstacle.pose[1]) < half_y + wall_buffer
        )
    if obstacle.size:
        half_x, half_y, _ = (value / 2.0 for value in obstacle.size)
        return (
            abs(xy[0] - obstacle.pose[0]) < half_x + buffer
            and abs(xy[1] - obstacle.pose[1]) < half_y + buffer
        )
    return math.dist(xy, obstacle.pose[:2]) < obstacle.radius + buffer


def _check_slot_overlap(
    slots: dict[str, tuple[float, float, float]],
    minimum_distance: float,
) -> None:
    items = list(slots.items())
    for index, (slot_id, pose) in enumerate(items):
        for other_id, other_pose in items[index + 1 :]:
            distance = math.dist(pose[:2], other_pose[:2])
            if distance < minimum_distance:
                raise SlotAllocationError(
                    f"slot {slot_id!r} overlaps with {other_id!r}: "
                    f"distance={distance:.4f}"
                )
