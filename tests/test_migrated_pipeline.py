from pathlib import Path

import yaml

from cli.run_simulation import _context_config
from ctamp.cost import EdgeCostCalculator
from ctamp.experiments.run_stacking_v2 import build_phase_configs
from ctamp.learning.heuristic_models import OnlineSGDModel
from ctamp.planning.symbolic import PlanningProblem, SymbolicTaskPlanner
from ctamp.search.tmm_astar import TMMAStar
from ctamp.simulation.scene import generate_tidy_slots
from ctamp.tmm.builder import TMMGraphBuilder


def test_context_adapter_feeds_migrated_scene_pipeline():
    config = _context_config(Path("contexts/examples/align_grouped_tidy_wall_world.md"))

    assert config["scene"]["scene_id"] == "align_grouped_tidy_wall_world"
    assert config["grouped_tidy"]["axis"] == "y"
    assert len(config["objects"]) == 12
    assert len(generate_tidy_slots(config)) == 12


def test_source_learning_planning_cost_search_tmm_are_imported():
    assert OnlineSGDModel is not None
    assert EdgeCostCalculator is not None
    assert SymbolicTaskPlanner(PlanningProblem(objects={}, target_poses={})).solve() is not None
    assert TMMGraphBuilder is not None
    assert TMMAStar is not None


def test_stacking_v2_builds_placeholder_then_large_to_small_stack():
    config = yaml.safe_load(Path("configs/scenes/stacking_wall_world_v2.yaml").read_text())

    phase1, phase2, summary = build_phase_configs(config)
    sizes = {obj["id"]: obj["size_xyz"][0] for obj in config["objects"]}
    grip_widths = {obj["id"]: obj["grip_target_width"] for obj in config["objects"]}

    assert phase1["task"]["target_objects"] == ["c6", "c5", "c4", "c3", "c2", "c1"]
    assert phase2["task"]["target_objects"] == ["c6", "c5", "c4", "c3", "c2", "c1"]
    assert phase2["task"]["preserve_order"] is True
    assert summary["largest_to_smallest_order"] == ["c6", "c5", "c4", "c3", "c2", "c1"]
    assert sizes["c6"] - sizes["c1"] == 0.04
    assert grip_widths["c1"] < grip_widths["c6"]
    assert summary["safe_zone_positions"]["c6"][0] < summary["safe_zone_positions"]["c1"][0]
    assert summary["safe_zone_positions"]["c1"][2] < 0.84
    assert summary["safe_zone_positions"]["c6"][2] < 0.85
    assert summary["final_stack_positions"]["c6"][:2] == [-0.30, -0.75]
    assert summary["final_stack_positions"]["c6"][2] < summary["final_stack_positions"]["c1"][2]
    assert len(generate_tidy_slots(phase1)) == 6
    assert len(generate_tidy_slots(phase2)) == 6
