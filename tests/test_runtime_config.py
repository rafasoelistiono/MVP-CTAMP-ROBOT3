from __future__ import annotations

from dataclasses import replace

import pytest

from configuration import (
    DEFAULT_PROFILE_REGISTRY,
    RuntimeConfigError,
    RuntimeProfileRegistry,
    activate_runtime_config,
    clear_runtime_config,
    get_active_runtime_config,
    load_runtime_config,
)


def test_code_profiles_select_scene_specific_tuning():
    normal = load_runtime_config("conservative", enable_viewer=False)
    obstacle = load_runtime_config("obstacle", enable_viewer=False)
    assert normal.motion.time_limit_s == 6.0
    assert obstacle.motion.time_limit_s == 12.0
    assert obstacle.motion.sampler_range == 0.04
    assert obstacle.motion.valid_state_sampler == "obstacle_based"
    assert obstacle.ik.max_valid_candidates == 8
    assert obstacle.safety.min_pick_obstacle_clearance_m == 0.10
    assert obstacle.grasp.open_grip_m == 0.05
    assert obstacle.grasp.pick_grip_sequence == (0.026, 0.025, 0.024)
    assert obstacle.grasp.obstacle_cube_grip == 0.026
    assert obstacle.grasp.release_guide_clearance_m == 0.008
    assert obstacle.recovery.verification_settle_steps == 60


def test_environment_does_not_mutate_motion_tuning(monkeypatch):
    monkeypatch.setenv("OMPL_TIME_LIMIT", "999")
    monkeypatch.setenv("IK_PLAN_POS_ERR_LIMIT", "0.999")
    config = load_runtime_config("conservative")
    assert config.motion.time_limit_s == 6.0
    assert config.ik.plan_position_error_m == 0.020


def test_strict_toml_overlay_changes_only_declared_fields(tmp_path):
    path = tmp_path / "tuned.toml"
    path.write_text(
        'extends = "conservative"\nname = "experiment_a"\n'
        "[motion]\ntime_limit_s = 7.5\n"
        "[grasp]\npick_grip_sequence = [0.020, 0.017, 0.014]\n",
        encoding="utf-8",
    )
    config = load_runtime_config(config_file=path)
    assert config.name == "experiment_a"
    assert config.motion.time_limit_s == 7.5
    assert config.grasp.pick_grip_sequence == (0.020, 0.017, 0.014)
    assert config.safety.max_pick_object_xy_m == 0.92


def test_unknown_tuning_field_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        'extends = "conservative"\n[motion]\nmagic_speed = 42\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeConfigError, match="unsupported fields"):
        load_runtime_config(config_file=path)


def test_custom_profile_can_be_registered_without_core_change():
    registry = RuntimeProfileRegistry()
    base = DEFAULT_PROFILE_REGISTRY.get("conservative")
    registry.register(replace(base, name="research_profile"))
    assert registry.get("research_profile").name == "research_profile"


def test_active_runtime_config_is_explicit_and_immutable():
    clear_runtime_config()
    selected = load_runtime_config("obstacle", enable_viewer=False)
    activate_runtime_config(selected)
    assert get_active_runtime_config() is selected
    clear_runtime_config()
