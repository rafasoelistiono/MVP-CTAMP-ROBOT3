from __future__ import annotations

from pathlib import Path

import pytest

from world.builder import ContextValidationError, build_world_state


def context_text(variant: str = "group_obs", include_obstacles: bool = True) -> str:
    obstacle_block = """
## obstacles
- id: obstacle1
  pose: [0.11, -0.30, 0.89]
  fragile: true
  radius: 0.035
  height: short
""" if include_obstacles else ""
    return f"""# CONTEXT.MD — CTAMP Tabletop Scene

## scene
- scene_id: test_scene
- variant: {variant}

## table
- x_range: [-0.55, 0.55]
- y_range: [-0.75, 0.75]
- z_top: 0.80
- goal_center: [0.22, -0.06, 0.806]

## robot
- id: panda_left
- reach_min_xy: 0.30
- reach_max_xy: 0.82
- base_xy: [-0.4, 0.0]
- capabilities: [pick, place, stack_place]

## objects
- id: cube1
  class: cube
  pose: [-0.02, -0.46, 0.83]
  reachable: true
  near_obstacle: false
- id: cube_far
  class: cube
  pose: [0.54, 0.70, 0.83]
  reachable: true
  near_obstacle: false

{obstacle_block}
## task
- name: stack
- target_objects: [cube1]
- description: stack cube

## constraints
- preserve_obstacles: true
- max_retries_per_object: 3
- allowed_predicates: [at, on, clear, handempty, holding]
"""


def write_context(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "context.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_build_context_with_obstacle(tmp_path):
    world = build_world_state(write_context(tmp_path, context_text()))
    assert world.scene_id == "test_scene"
    assert world.object_by_id("cube1").reachable
    assert world.obstacles[0].fragile


def test_build_context_without_obstacle(tmp_path):
    world = build_world_state(
        write_context(
            tmp_path,
            context_text("group_no_obs", include_obstacles=False),
        )
    )
    assert world.obstacles == ()


def test_unreachable_is_computed_not_trusted(tmp_path):
    world = build_world_state(write_context(tmp_path, context_text()))
    assert not world.object_by_id("cube_far").reachable


def test_missing_required_context_field_is_descriptive(tmp_path):
    broken = context_text().replace("- base_xy: [-0.4, 0.0]\n", "")
    with pytest.raises(ContextValidationError, match="base_xy"):
        build_world_state(write_context(tmp_path, broken))

