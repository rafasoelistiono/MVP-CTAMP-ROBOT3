from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path

import numpy as np
import pytest

from configuration import activate_runtime_config, load_runtime_config
from plugins.registry import DEFAULT_REGISTRY
from scene import prepare_scene_variant
from task_planning.loader import load_plan
from world.builder import build_world_state
from world.slot_allocator import allocate_grouped_align_slots


pytestmark = pytest.mark.simulation

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "contexts/examples/align_grouped_tidy_wall_world.md"
PLAN = ROOT / "task_plans/examples/align_grouped_tidy_wall_world.json"


def test_wall_probe_uses_right_gateway_without_mutating_simulation():
    world = build_world_state(CONTEXT)
    plan = load_plan(PLAN)
    config = DEFAULT_REGISTRY.get("align").configure_runtime(
        plan,
        world,
        load_runtime_config("obstacle", enable_viewer=False),
    )
    scene_path = prepare_scene_variant(
        world.variant,
        base_model_file=config.model.xml_path,
        object_states=world.objects,
        obstacle_states=world.obstacles,
        goal_center=world.goal_center,
        goal_area_size_xy=world.goal_area_size_xy,
        table_size_xy=(
            world.table_x_range[1] - world.table_x_range[0],
            world.table_y_range[1] - world.table_y_range[0],
        ),
        base_xy=world.robot_base_xy,
        base_z=world.robot_base_z,
    )
    config = replace(
        config,
        model=replace(config.model, xml_path=scene_path),
        enable_viewer=False,
    ).validate()
    activate_runtime_config(config)

    runtime = importlib.import_module("backends.mujoco.runtime")
    slots = allocate_grouped_align_slots(world, world.grouped_tidy)
    slot = slots["tidy_slot_blue_lane_0"]
    target = (
        slot[0],
        slot[1],
        slot[2] + config.grasp.approach_clearance_m,
    )
    qpos_before = runtime.data.qpos.copy()
    qvel_before = runtime.data.qvel.copy()
    ctrl_before = runtime.data.ctrl.copy()

    try:
        report = runtime.probe_motion_to(
            target,
            label="place(simulation_probe) preplace probe",
            time_limit=15.0,
        )
        assert report["success"], report
        assert report["route"] == "wall_right"
        assert report["segment_count"] == 4
        assert report["waypoint_count"] > 4
        assert np.array_equal(runtime.data.qpos, qpos_before)
        assert np.array_equal(runtime.data.qvel, qvel_before)
        assert np.array_equal(runtime.data.ctrl, ctrl_before)

        runtime.pick("g")
        assert runtime._held_object_name == "g"
        runtime.place(slot[0], slot[1], obj="g", target_z=slot[2])
        final_position = np.asarray(runtime._object_xyz("g"), dtype=float)
        assert runtime._held_object_name is None
        assert np.linalg.norm(final_position[:2] - np.asarray(slot[:2])) <= 0.020
        equality_id = runtime.mujoco.mj_name2id(
            runtime.model,
            runtime.mujoco.mjtObj.mjOBJ_EQUALITY,
            "carry_g",
        )
        assert equality_id >= 0
        assert not runtime.data.eq_active[equality_id]
    finally:
        runtime.shutdown_runtime()
