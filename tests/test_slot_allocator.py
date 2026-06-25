from __future__ import annotations

import pytest

from task_planning.types import SlotConfig
from world.slot_allocator import allocate_slots, resolve_pyramid_slots


def test_align_four_cube_coordinates():
    slots = allocate_slots(
        SlotConfig(
            type="line",
            spacing_m=0.125,
            center_x=0.22,
            row_y=-0.06,
            base_z=0.83,
        ),
        4,
    )
    assert [slots[f"slot_{i}"][0] for i in range(4)] == pytest.approx(
        [0.0325, 0.1575, 0.2825, 0.4075]
    )


def test_stack_four_cube_coordinates():
    slots = allocate_slots(
        SlotConfig(
            type="tower",
            base_xy=(0.22, -0.06),
            base_z=0.83,
            layer_height_m=0.06,
        ),
        4,
    )
    assert [slots[label][2] for label in ("tower_base", "level_1", "level_2", "level_3")] == pytest.approx(
        [0.83, 0.89, 0.95, 1.01]
    )


def test_pyramid_six_cube_coordinates():
    config = SlotConfig(
        type="pyramid",
        row_count=3,
        base_row_length=3,
        spacing_m=0.066,
        center_x=0.22,
        base_y=-0.06,
        base_z=0.833,
        layer_height_m=0.066,
    )
    slots = resolve_pyramid_slots(
        config,
        [f"cube{index}" for index in range(1, 7)],
    )

    assert list(slots) == [
        "row0_col0",
        "row0_col1",
        "row0_col2",
        "row1_col0",
        "row1_col1",
        "row2_col0",
    ]
    assert slots["row0_col0"] == pytest.approx((0.154, -0.06, 0.833))
    assert slots["row0_col1"] == pytest.approx((0.22, -0.06, 0.833))
    assert slots["row0_col2"] == pytest.approx((0.286, -0.06, 0.833))
    assert slots["row1_col0"] == pytest.approx((0.187, -0.06, 0.899))
    assert slots["row1_col1"] == pytest.approx((0.253, -0.06, 0.899))
    assert slots["row2_col0"] == pytest.approx((0.22, -0.06, 0.965))

    allocated = allocate_slots(config, 6)
    assert allocated == pytest.approx(slots)
