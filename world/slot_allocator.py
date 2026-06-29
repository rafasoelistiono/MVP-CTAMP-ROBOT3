from __future__ import annotations

import math
from collections.abc import Sequence

from task_planning.types import SlotConfig

from .state import WorldState


class SlotAllocationError(ValueError):
    pass


def allocate_slots(
    config: SlotConfig,
    n: int,
) -> dict[str, tuple[float, float, float]]:
    if n <= 0:
        raise SlotAllocationError("slot count must be positive")
    if config.type == "tower":
        return _allocate_tower(config, n)
    if config.type == "pyramid":
        return resolve_pyramid_slots(config, tuple(f"cube{index + 1}" for index in range(n)))
    raise SlotAllocationError(f"unknown slot type: {config.type}")


def _allocate_tower(
    config: SlotConfig,
    n: int,
) -> dict[str, tuple[float, float, float]]:
    bx, by = config.base_xy
    slots: dict[str, tuple[float, float, float]] = {}
    for index in range(n):
        label = "tower_base" if index == 0 else f"level_{index}"
        slots[label] = (
            bx,
            by,
            config.base_z + index * config.layer_height_m,
        )
    return slots


def resolve_pyramid_slots(
    config: SlotConfig,
    target_objects: Sequence[str],
) -> dict[str, tuple[float, float, float]]:
    if config.axis != "x":
        raise SlotAllocationError("only pyramid row axis 'x' is currently supported")
    if config.row_count <= 0:
        raise SlotAllocationError("pyramid row_count must be positive")
    if config.base_row_length <= 0:
        raise SlotAllocationError("pyramid base_row_length must be positive")
    expected = config.row_count * (config.row_count + 1) // 2
    if len(target_objects) != expected:
        raise SlotAllocationError(
            "pyramid slot count must equal row_count*(row_count+1)/2: "
            f"expected {expected}, got {len(target_objects)}"
        )

    slots: dict[str, tuple[float, float, float]] = {}
    assigned = 0
    for row in range(config.row_count):
        row_length = config.base_row_length - row
        if row_length <= 0:
            raise SlotAllocationError(
                f"pyramid row {row} has non-positive length {row_length}"
            )
        start_x = config.center_x - ((row_length - 1) / 2.0) * config.spacing_m
        y = config.base_y
        z = config.base_z + row * config.layer_height_m
        for column in range(row_length):
            if assigned >= len(target_objects):
                raise SlotAllocationError("pyramid target_objects ended before slots")
            slots[f"row{row}_col{column}"] = (
                start_x + column * config.spacing_m,
                y,
                z,
            )
            assigned += 1
    return slots


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
            clearance = math.dist((x, y), obstacle.pose[:2])
            if clearance < obstacle.radius + obstacle_buffer_m:
                raise SlotAllocationError(
                    f"{slot_id} violates inflated obstacle region for {obstacle.id}: "
                    f"clearance={clearance:.4f}"
                )
