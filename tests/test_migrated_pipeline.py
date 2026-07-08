from pathlib import Path

from cli.run_simulation import _context_config
from ctamp.cost import EdgeCostCalculator
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
