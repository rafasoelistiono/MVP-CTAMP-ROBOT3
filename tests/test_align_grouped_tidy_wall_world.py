from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest
import mujoco

from configuration import load_runtime_config
from plugins.registry import DEFAULT_REGISTRY
from scene import prepare_scene_variant
from task_planning.loader import load_plan
from task_planning.validator import validate
from world.builder import build_world_state
from world.slot_allocator import allocate_grouped_align_slots, validate_slots


CONTEXT_PATH = "contexts/examples/align_grouped_tidy_wall_world.md"
PLAN_PATH = "task_plans/examples/align_grouped_tidy_wall_world.json"


@pytest.fixture
def world():
    return build_world_state(CONTEXT_PATH)


@pytest.fixture
def slots(world):
    return allocate_grouped_align_slots(world, world.grouped_tidy)


def _scene_root(world):
    path = prepare_scene_variant(
        world.variant,
        object_states=world.objects,
        obstacle_states=world.obstacles,
        goal_center=world.goal_center,
        goal_area_size_xy=world.goal_area_size_xy,
        table_size_xy=(
            world.table_x_range[1] - world.table_x_range[0],
            world.table_y_range[1] - world.table_y_range[0],
        ),
    )
    return ET.parse(path).getroot()


def _inside_inflated_wall(pose, wall, buffer=0.13):
    half_x, half_y, _ = (value / 2.0 for value in wall.size)
    return (
        abs(pose[0] - wall.pose[0]) < half_x + buffer
        and abs(pose[1] - wall.pose[1]) < half_y + buffer
    )


def test_context_layout_invariants(world):
    wall = world.obstacles[0]
    half_x, half_y, _ = (value / 2.0 for value in wall.size)
    group_y = min(group.center[1] for group in world.grouped_tidy.groups)
    assert world.robot_base_xy[1] < wall.pose[1] < group_y
    assert world.robot_base_xy[1] == pytest.approx(-0.86)
    assert world.robot_reach_max == pytest.approx(1.50)
    assert abs(world.robot_base_xy[0] - wall.pose[0]) <= 0.10
    assert wall.id == "frontal_tall_wall"
    assert wall.size == pytest.approx((0.08, 0.20, 1.60))
    assert wall.pose[2] + wall.size[2] / 2.0 >= 2.30
    assert world.grouped_tidy.axis == "y"
    assert world.grouped_tidy.spacing == pytest.approx(0.075)
    assert world.goal_center[0] > wall.pose[0] + half_x + 0.13
    assert world.goal_center[1] > wall.pose[1] + half_y
    assert world.challenge.type == "frontal_tall_wall"
    assert world.challenge.wall_blocks_direct_path
    assert world.challenge.side_corridors_required
    assert {group.color for group in world.grouped_tidy.groups} == {"blue", "red"}
    assert {group.center[1] for group in world.grouped_tidy.groups} == {0.34}
    assert {group.center[0] for group in world.grouped_tidy.groups} == {0.20, 0.32}
    assert all(group.center[0] > wall.pose[0] + half_x + 0.13 for group in world.grouped_tidy.groups)


def test_table_is_wide_and_deep(world):
    assert world.table_x_range[1] - world.table_x_range[0] >= 1.50
    assert world.table_y_range[1] - world.table_y_range[0] >= 2.00


def test_objects_are_scattered_behind_and_around_wall(world):
    wall = world.obstacles[0]
    half_x, half_y, _ = (value / 2.0 for value in wall.size)
    colors = {}
    for obj in world.objects:
        colors.setdefault(obj.rgba, 0)
        colors[obj.rgba] += 1
    left = [obj for obj in world.objects if obj.pose[0] < wall.pose[0] + half_x + 0.13]
    right = [obj for obj in world.objects if obj.pose[0] > wall.pose[0] + half_x + 0.13]
    y_values = [obj.pose[1] for obj in world.objects]
    assert len(world.objects) == 12
    assert sorted(colors.values()) == [6, 6]
    assert len(left) == 0
    assert len(right) == len(world.objects)
    assert min(y_values) <= world.table_y_range[0] + 0.15
    assert max(y_values) >= 0.55
    assert max(y_values) - min(y_values) >= 1.35
    assert len({round(y, 2) for y in y_values}) == len(world.objects)
    for obj in world.objects:
        assert obj.reachable
        assert obj.pose[0] > wall.pose[0] + half_x + 0.13
        assert not _inside_inflated_wall(obj.pose, wall, buffer=0.13)
    for index, obj in enumerate(world.objects):
        for other in world.objects[index + 1:]:
            assert math.dist(obj.pose[:2], other.pose[:2]) >= 0.066


def test_tidy_slots_are_behind_wall_and_safe(world, slots):
    wall = world.obstacles[0]
    half_x, half_y, _ = (value / 2.0 for value in wall.size)
    expected = {
        "tidy_slot_blue_lane_0",
        "tidy_slot_blue_lane_1",
        "tidy_slot_blue_lane_2",
        "tidy_slot_blue_lane_3",
        "tidy_slot_blue_lane_4",
        "tidy_slot_blue_lane_5",
        "tidy_slot_red_lane_0",
        "tidy_slot_red_lane_1",
        "tidy_slot_red_lane_2",
        "tidy_slot_red_lane_3",
        "tidy_slot_red_lane_4",
        "tidy_slot_red_lane_5",
    }
    assert set(slots) == expected
    validate_slots(slots, world, obstacle_buffer_m=0.13)
    assert {round(pose[0], 2) for pose in slots.values()} == {0.20, 0.32}
    assert max(pose[1] for pose in slots.values()) - min(pose[1] for pose in slots.values()) == pytest.approx(0.375)
    for pose in slots.values():
        assert pose[1] > wall.pose[1] + half_y
        assert pose[0] > wall.pose[0] + half_x + 0.13
        assert not _inside_inflated_wall(pose, wall)


def test_plan_loads_and_validates(world):
    plan = load_plan(PLAN_PATH)
    validate(plan, world.all_object_ids(), world.allowed_predicates)
    DEFAULT_REGISTRY.get("align").validate_plan(plan, world)


def test_scene_wall_is_static_box_collision_and_tall(world):
    root = _scene_root(world)
    body = root.find(".//body[@name='frontal_tall_wall']")
    assert body is not None
    assert body.find("joint") is None
    geom = body.find("geom")
    assert geom is not None
    assert geom.get("type") == "box"
    assert geom.get("contype") == "1"
    assert geom.get("conaffinity") == "1"
    assert [float(value) for value in geom.get("size").split()] == pytest.approx(
        [0.04, 0.10, 0.80]
    )


def test_scene_arm_is_moved_back_for_wall_world(world):
    root = _scene_root(world)
    link0 = root.find("./worldbody/body[@name='link0']")
    assert link0 is not None
    assert [float(value) for value in link0.get("pos").split()] == pytest.approx(
        [-0.4, -0.18, 0.8]
    )


def test_initial_arm_pose_does_not_touch_wall(world):
    plan = load_plan(PLAN_PATH)
    config = DEFAULT_REGISTRY.get("align").configure_runtime(
        plan,
        world,
        load_runtime_config("obstacle"),
    )
    root_path = prepare_scene_variant(
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
    )
    model = mujoco.MjModel.from_xml_path(str(root_path))
    data = mujoco.MjData(model)
    arm_qpos = [model.joint(f"joint{idx}").qposadr[0] for idx in range(1, 8)]
    data.qpos[arm_qpos] = config.model.home_q
    mujoco.mj_forward(model, data)
    wall_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "frontal_tall_wall")
    wall_geoms = {
        geom_id for geom_id in range(model.ngeom) if model.geom_bodyid[geom_id] == wall_body
    }
    assert all(
        data.contact[index].geom1 not in wall_geoms
        and data.contact[index].geom2 not in wall_geoms
        for index in range(data.ncon)
    )


def test_scene_table_visual_uses_context_size(world):
    root = _scene_root(world)
    table = root.find(".//geom[@name='table_top']")
    assert table is not None
    assert [float(value) for value in table.get("size").split()[:2]] == pytest.approx(
        [0.85, 1.00]
    )


def test_goal_visual_zones_are_hidden_for_wall_world(world):
    root = _scene_root(world)
    green = root.find(".//geom[@name='goal_left_zone']")
    blue = root.find(".//geom[@name='goal_right_zone']")
    assert green is None
    assert blue is None
