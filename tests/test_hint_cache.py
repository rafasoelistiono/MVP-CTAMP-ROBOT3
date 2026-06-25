from __future__ import annotations

import csv

from backends.adaptive.hint_cache import HintCache
from execution.verifier import ObservedPredicateVerifier


def test_empty_log_returns_safe_defaults(tmp_path):
    cache = HintCache(tmp_path)
    hints = cache.hints_for("cube1", "cube", 0.5)
    assert hints.ik_backend == "pinocchio"
    assert hints.ik_position_tolerance == 0.020
    assert hints.grasp_profile == "default_cube"


def test_high_fallback_rate_skips_pinocchio(tmp_path):
    path = tmp_path / "sample_events.csv"
    rows = [
        {
            "stage": "IK_SOLVE",
            "status": "BACKEND_FALLBACK",
            "object_id": "cube1",
            "failure_reason": "pinocchio_fk_validation_failed",
        }
        for _ in range(3)
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    cache = HintCache(tmp_path)
    assert cache.get_ik_backend("cube1", "normal") == "mujoco_dls"


def test_hint_cache_cannot_change_verifier_tolerances(tmp_path):
    before = dict(ObservedPredicateVerifier.TOLERANCES)
    cache = HintCache(tmp_path)
    cache.get_ik_tolerance("cube1", "normal")
    assert ObservedPredicateVerifier.TOLERANCES == before

