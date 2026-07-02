from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from world.builder import build_world_state
from scene import (
    GROUPED_TIDY_WALL_HALF_DEPTH,
    GROUPED_TIDY_WALL_HALF_HEIGHT,
    prepare_scene_variant,
)

CONTEXT_PATH = "contexts/examples/align_grouped_tidy_gang.md"


@pytest.fixture
def world():
    return build_world_state(CONTEXT_PATH)


def _prepare(world):
    return prepare_scene_variant(
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


def test_context_loads_successfully(world):
    assert world.scene_id == "align_grouped_tidy_gang"
    assert world.variant == "align_grouped_tidy_gang"
    assert world.task_name == "align"
    assert len(world.target_objects) == 12


def test_scene_contains_12_movable_objects(world):
    assert len(world.objects) == 12
    for obj in world.objects:
        assert obj.cls == "cube"


def test_scene_contains_tall_obstacles(world):
    obstacle_ids = {obs.id for obs in world.obstacles}
    assert "tall_obs_left" in obstacle_ids
    assert "tall_obs_right" in obstacle_ids


def test_obstacles_are_static(world):
    for obs in world.obstacles:
        assert obs.height == "long"

    path = _prepare(world)
    root = ET.parse(path).getroot()
    for obs in world.obstacles:
        body = root.find(f".//body[@name='{obs.id}']")
        assert body is not None
        assert body.find("joint") is None


def test_obstacles_have_collision_geoms(world):
    for obs in world.obstacles:
        assert obs.radius > 0


def test_no_initial_object_obstacle_overlap(world):
    for obj in world.objects:
        for obs in world.obstacles:
            dist = math.dist(obj.pose[:2], obs.pose[:2])
            assert dist > obs.radius + 0.05, (
                f"object {obj.id} overlaps obstacle {obs.id}"
            )


def test_objects_start_randomized_on_both_wall_sides(world):
    left_ids = {"b", "d", "f", "h", "j", "l"}
    right_ids = {"a", "c", "e", "g", "i", "k"}
    left_wall_edge = min(obs.pose[0] - obs.radius for obs in world.obstacles)
    right_wall_edge = max(obs.pose[0] + obs.radius for obs in world.obstacles)

    assert all(world.object_by_id(oid).pose[0] < left_wall_edge for oid in left_ids)
    assert all(world.object_by_id(oid).pose[0] > right_wall_edge for oid in right_ids)
    assert len({obj.pose for obj in world.objects}) == 12
    assert all(obj.pose[1] < -GROUPED_TIDY_WALL_HALF_DEPTH for obj in world.objects)
    for index, obj in enumerate(world.objects):
        for other in world.objects[index + 1:]:
            assert math.dist(obj.pose[:2], other.pose[:2]) >= 0.16


def test_obstacles_not_treated_as_movable(world):
    object_ids = {obj.id for obj in world.objects}
    assert "tall_obs_left" not in object_ids
    assert "tall_obs_right" not in object_ids


def test_scene_xml_generation(world):
    path = _prepare(world)
    root = ET.parse(path).getroot()
    body_names = {body.get("name") for body in root.findall(".//body")}
    assert "tall_obs_left" in body_names
    assert "tall_obs_right" in body_names
    for obj in world.objects:
        assert obj.id in body_names
        geom = root.find(f".//body[@name='{obj.id}']/geom")
        assert geom is not None
        assert [float(value) for value in geom.get("friction").split()] == pytest.approx(
            [3.0, 1.5, 0.8]
        )
        assert float(geom.get("mass")) == pytest.approx(0.06)
    finger_actuator = root.find("./actuator/position[@joint='finger_joint1']")
    assert finger_actuator is not None
    assert float(finger_actuator.get("kp")) == pytest.approx(600.0)


def test_tall_obstacles_have_correct_height(world):
    path = _prepare(world)
    root = ET.parse(path).getroot()
    for obs in world.obstacles:
        body = root.find(f".//body[@name='{obs.id}']")
        assert body is not None
        geom = body.find("geom")
        assert geom is not None
        size = [float(value) for value in geom.get("size").split()]
        assert geom.get("type") == "box"
        assert size == pytest.approx(
            [obs.radius, GROUPED_TIDY_WALL_HALF_DEPTH, GROUPED_TIDY_WALL_HALF_HEIGHT]
        )
        assert obs.pose[2] + size[2] > 0.80 + world.robot_reach_max


def test_center_gap_is_too_narrow_for_cube_or_arm(world):
    ordered = sorted(world.obstacles, key=lambda obs: obs.pose[0])
    gap = (
        ordered[1].pose[0]
        - ordered[1].radius
        - (ordered[0].pose[0] + ordered[0].radius)
    )
    assert gap == pytest.approx(world.challenge.min_gap_width)
    assert gap < 0.066


def test_table_and_goal_visual_match_context(world):
    path = _prepare(world)
    root = ET.parse(path).getroot()
    table = root.find(".//geom[@name='table_top']")
    goal = root.find(".//body[@name='goal_area']")
    goal_geom = root.find(".//geom[@name='goal_area_base']")
    assert table is not None
    assert goal is not None
    assert goal_geom is not None
    table_size = [float(value) for value in table.get("size").split()]
    assert table_size[:2] == pytest.approx([0.90, 0.95])
    goal_pos = [float(value) for value in goal.get("pos").split()]
    assert goal_pos[:2] == pytest.approx(world.goal_center[:2])
    goal_size = [float(value) for value in goal_geom.get("size").split()]
    assert goal_size[:2] == pytest.approx([0.425, 0.20])
