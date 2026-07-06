from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from backends.mujoco.adaptive_hints import HintCache
from execution.motion_probe import MotionProbe
from world.builder import build_world_state


def _write_history(path, fingerprint: str) -> None:
    fieldnames = ["stage", "status", "failure_reason", "extra_json"]
    rows = [
        {
            "stage": "RUNTIME_FINGERPRINT",
            "status": "OK",
            "failure_reason": "",
            "extra_json": json.dumps({"config_fingerprint": fingerprint}),
        },
        {
            "stage": "IK_SOLVE",
            "status": "BACKEND_FALLBACK",
            "failure_reason": "pinocchio_fk_validation_failed",
            "extra_json": "",
        },
        {
            "stage": "IK_CANDIDATE",
            "status": "REJECT",
            "failure_reason": "ik_error_above_limit",
            "extra_json": "",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_adaptive_history_requires_matching_configuration_fingerprint(tmp_path):
    _write_history(tmp_path / "motion_scene_events.csv", "old")
    cache = HintCache(
        log_dir=str(tmp_path),
        config_fingerprint="new",
        min_samples=1,
    )
    assert cache.preferred_backend() is None
    assert cache.summary()["fingerprint_files_ignored"] == 1


def test_matching_fingerprint_can_supply_adaptive_hint(tmp_path):
    _write_history(tmp_path / "motion_scene_events.csv", "same")
    cache = HintCache(
        log_dir=str(tmp_path),
        config_fingerprint="same",
        min_samples=1,
        pinocchio_skip_rate=0.5,
    )
    assert cache.preferred_backend() == "mujoco_dls"
    assert cache.summary()["fingerprint_files_accepted"] == 1


def test_motion_probe_propagates_lateral_success_and_diagnostics():
    world = build_world_state("contexts/examples/align_grouped_tidy_wall_world.md")
    diagnostics = (
        {"candidate": "top_down", "failure_reason": "ik_unreachable"},
        {"candidate": "side_pos_x", "failure_reason": None},
    )

    class Runtime:
        CONFIG = SimpleNamespace(
            grasp=SimpleNamespace(approach_clearance_m=0.20)
        )

        @staticmethod
        def probe_grasp_candidates(*args, **kwargs):
            return {
                "success": True,
                "ik_success": True,
                "ompl_success": True,
                "selected_grasp_candidate": "side_pos_x",
                "attempted_orientations": 2,
                "attempted_seeds": 7,
                "grasp_diagnostics": diagnostics,
                "path_length": 1.0,
            }

    result = MotionProbe(runtime=Runtime()).probe_pick_feasibility(world, "a")
    assert result.feasible
    assert result.attempted_orientations == 2
    assert result.grasp_diagnostics == diagnostics
    assert result.grasp_diagnostics[0]["candidate"] == "top_down"
    assert result.grasp_diagnostics[1]["candidate"] == "side_pos_x"


def test_motion_probe_propagates_structured_no_grasp_found():
    world = build_world_state("contexts/examples/align_grouped_tidy_wall_world.md")
    diagnostics = tuple(
        {"candidate": name, "failure_reason": "ik_unreachable"}
        for name in ("top_down", "side_pos_x", "side_neg_x", "side_pos_y")
    )

    class Runtime:
        CONFIG = SimpleNamespace(
            grasp=SimpleNamespace(approach_clearance_m=0.20)
        )

        @staticmethod
        def probe_grasp_candidates(*args, **kwargs):
            return {
                "success": False,
                "ik_success": False,
                "ompl_success": False,
                "failure_reason": "no_grasp_found",
                "attempted_orientations": 4,
                "attempted_seeds": 16,
                "grasp_diagnostics": diagnostics,
            }

    result = MotionProbe(runtime=Runtime()).probe_pick_feasibility(world, "a")
    assert not result.feasible
    assert result.failure_reason == "no_grasp_found"
    assert len(result.grasp_diagnostics) == 4
