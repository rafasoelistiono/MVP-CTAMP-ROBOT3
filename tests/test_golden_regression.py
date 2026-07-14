from pathlib import Path

import pytest

from ctamp.experiments.run_scene import run as run_scene_v1
from ctamp.experiments.run_scene_v2 import run as run_scene_v2
from ctamp.experiments.run_stacking_v2 import run as run_stacking_v2


TIDY_CONFIG = Path("configs/scenes/align_grouped_tidy_wall_world.yaml")
STACKING_CONFIG = Path("configs/scenes/stacking_wall_world_v2.yaml")


def _assert_tidy_one_object_golden(metrics: dict, version: str | None = None) -> None:
    first = metrics["per_object_result"][0]

    assert metrics["scene_id"] == "align_grouped_tidy_wall_world"
    assert metrics["solution_found"] is True
    assert metrics["completed_objects"] == 1
    assert metrics["completion_policy"] == "best_effort"
    assert metrics["failed_objects"] == []
    assert first["object_id"] == "j"
    assert first["slot"] == "tidy_slot_red_lane_0"
    assert first["route_type"] == "direct"
    assert first["transit_route_type"] == "direct"
    assert first["ik_success"] is True
    assert first["grasp_style"] == "top"
    if version is not None:
        assert metrics["ctamp_version"] == version


@pytest.mark.simulation
def test_v1_and_v2_tidy_one_object_golden(tmp_path):
    v1_metrics = run_scene_v1(TIDY_CONFIG, tmp_path / "v1", max_objects=1)
    v2_metrics = run_scene_v2(TIDY_CONFIG, tmp_path / "v2", max_objects=1)

    _assert_tidy_one_object_golden(v1_metrics)
    _assert_tidy_one_object_golden(v2_metrics, version="v2")
    assert v2_metrics["performance_v2"]["plan_xy_cache_hits"] >= 1


@pytest.mark.simulation
def test_stacking_dry_run_golden(tmp_path):
    metrics = run_stacking_v2(STACKING_CONFIG, tmp_path / "stacking", dry_run=True)

    assert metrics["ctamp_version"] == "v2"
    assert metrics["task"] == "stack"
    assert metrics["dry_run"] is True
    assert metrics["largest_to_smallest_order"] == ["c6", "c5", "c4", "c3", "c2", "c1"]
    assert metrics["safe_zone_order_right_first"] == [
        "c6",
        "c5",
        "c4",
        "c3",
        "c2",
        "c1",
    ]
    assert metrics["final_order_bottom_to_top"] == ["c6", "c5", "c4", "c3", "c2", "c1"]
    assert metrics["safe_zone_positions"]["c6"] == [0.08, -0.5, 0.8490000000000001]
    assert metrics["final_stack_positions"]["c6"] == [-0.3, -0.75, 0.8490000000000001]
