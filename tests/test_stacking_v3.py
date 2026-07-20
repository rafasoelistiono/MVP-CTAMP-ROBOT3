from pathlib import Path

from ctamp.experiments.run_stacking_v3 import run as run_stacking_v3


STACKING_CONFIG = Path("configs/scenes/stacking_wall_world_v2.yaml")


def test_stacking_v3_dry_run_uses_search_and_confirmation(tmp_path):
    metrics = run_stacking_v3(
        STACKING_CONFIG,
        tmp_path / "stacking_v3",
        max_objects=2,
        dry_run=True,
    )

    assert metrics["ctamp_version"] == "v3"
    assert metrics["dry_run"] is True
    assert metrics["solution_found"] is True
    assert metrics["confirmed_order"] == ["c6", "c5"]
    assert metrics["ctamp_v3"]["search_success"] is True
    assert metrics["ctamp_v3"]["confirmation_success"] is True
    assert metrics["ctamp_v3"]["algorithm1"]["search_then_confirmation"] is True
    assert metrics["ctamp_v3"]["heuristic"]["online_updates"] > 0
    assert metrics["ctamp_v3"]["tmm_is_dag"] is True
    assert (tmp_path / "stacking_v3" / "continuous_stack_v3.yaml").exists()
