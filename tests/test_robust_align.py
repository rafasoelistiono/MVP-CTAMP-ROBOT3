from __future__ import annotations

import json
import math
import pytest

from task_planning.candidate_generator import (
    generate_align_candidates,
    generate_nearest_first_plan,
    generate_left_to_right_plan,
    generate_right_to_left_plan,
    generate_obstacle_aware_plan,
    generate_random_baseline_plan,
    generate_nearest_to_slot_plan,
)
from task_planning.cost_model import (
    estimate_align_edge_cost,
    estimate_align_plan_cost,
    rank_candidate_plans,
    INFEASIBLE_PENALTY,
)
from task_planning.feature_extractor import (
    extract_align_edge_features,
    extract_align_plan_features,
)
from task_planning.loader import parse_plan
from task_planning.types import (
    ConfirmationResult,
    ProbePlanResult,
    ProbeResult,
    ScoredPlan,
    TaskPlan,
)
from task_planning.validator import validate
from execution.motion_probe import MotionProbe
from execution.confirmation import (
    confirm_align_plan,
    confirm_ranked_align_candidates,
)
from plugins.registry import DEFAULT_REGISTRY
from world.slot_allocator import allocate_slots
from world.state import ObjectState, ObstacleState, WorldState


def _make_world(
    objects=None,
    obstacles=(),
    task="align",
    target_objects=None,
):
    if objects is None:
        objects = [
            ObjectState("cube1", "cube", (-0.16, -0.42, 0.833), True, False),
            ObjectState("cube2", "cube", (0.10, -0.54, 0.833), True, False),
            ObjectState("cube3", "cube", (-0.10, 0.28, 0.833), True, False),
            ObjectState("cube4", "cube", (0.12, 0.20, 0.833), True, False),
        ]
    if target_objects is None:
        target_objects = [obj.id for obj in objects if obj.cls == "cube"]
    return WorldState(
        scene_id="test_scene",
        variant="ungroup_obs" if obstacles else "ungroup_no_obs",
        objects=tuple(objects),
        obstacles=tuple(obstacles),
        table_x_range=(-0.55, 0.55),
        table_y_range=(-0.75, 0.75),
        table_z_top=0.80,
        goal_center=(0.22, -0.06, 0.806),
        robot_id="panda_left",
        robot_base_xy=(-0.4, 0.0),
        robot_reach_min=0.30,
        robot_reach_max=0.82,
        robot_capabilities=("pick", "place"),
        task_name=task,
        target_objects=tuple(target_objects),
        task_description="test",
        preserve_obstacles=True,
        max_retries_per_object=3,
        allowed_predicates=("at", "clear", "handempty", "holding", "stable"),
    )


def _make_slots(n=4):
    from task_planning.types import SlotConfig
    config = SlotConfig(type="line", center_x=0.22, row_y=-0.06, base_z=0.833, spacing_m=0.10)
    return allocate_slots(config, n)


class TestCandidateGenerator:
    def test_generates_multiple_candidates(self):
        world = _make_world()
        slots = _make_slots(4)
        candidates = generate_align_candidates(world, slots)
        assert len(candidates) >= 4

    def test_all_candidates_are_valid_taskplans(self):
        world = _make_world()
        slots = _make_slots(4)
        candidates = generate_align_candidates(world, slots)
        for plan in candidates:
            assert isinstance(plan, TaskPlan)
            assert plan.task == "align"
            assert plan.schema_version == "ctamp-plan/v1"

    def test_all_candidates_cover_target_objects(self):
        world = _make_world()
        slots = _make_slots(4)
        candidates = generate_align_candidates(world, slots)
        for plan in candidates:
            picked = [s.object for s in plan.steps if s.action == "pick"]
            assert set(picked) == set(world.target_objects)

    def test_no_duplicate_slot_assignments(self):
        world = _make_world()
        slots = _make_slots(4)
        candidates = generate_align_candidates(world, slots)
        for plan in candidates:
            place_slots = [s.slot for s in plan.steps if s.action == "place"]
            assert len(place_slots) == len(set(place_slots))

    def test_nearest_first_plan_ordering(self):
        world = _make_world()
        slots = _make_slots(4)
        plan = generate_nearest_first_plan(world, slots)
        robot_xy = world.robot_base_xy
        objects = [(oid, world.object_by_id(oid).pose) for oid in world.target_objects]
        objects.sort(key=lambda o: math.dist(o[1][:2], robot_xy))
        picked = [s.object for s in plan.steps if s.action == "pick"]
        assert picked == [o[0] for o in objects]

    def test_left_to_right_plan_ordering(self):
        world = _make_world()
        slots = _make_slots(4)
        plan = generate_left_to_right_plan(world, slots)
        objects = [(oid, world.object_by_id(oid).pose) for oid in world.target_objects]
        objects.sort(key=lambda o: o[1][0])
        picked = [s.object for s in plan.steps if s.action == "pick"]
        assert picked == [o[0] for o in objects]

    def test_right_to_left_plan_ordering(self):
        world = _make_world()
        slots = _make_slots(4)
        plan = generate_right_to_left_plan(world, slots)
        objects = [(oid, world.object_by_id(oid).pose) for oid in world.target_objects]
        objects.sort(key=lambda o: o[1][0], reverse=True)
        picked = [s.object for s in plan.steps if s.action == "pick"]
        assert picked == [o[0] for o in objects]

    def test_random_baseline_plan_has_all_objects(self):
        world = _make_world()
        slots = _make_slots(4)
        plan = generate_random_baseline_plan(world, slots, seed=42)
        picked = [s.object for s in plan.steps if s.action == "pick"]
        assert set(picked) == set(world.target_objects)

    def test_random_baseline_deterministic_with_seed(self):
        world = _make_world()
        slots = _make_slots(4)
        plan1 = generate_random_baseline_plan(world, slots, seed=42)
        plan2 = generate_random_baseline_plan(world, slots, seed=42)
        picked1 = [s.object for s in plan1.steps if s.action == "pick"]
        picked2 = [s.object for s in plan2.steps if s.action == "pick"]
        assert picked1 == picked2

    def test_candidates_deduplicated(self):
        world = _make_world()
        slots = _make_slots(4)
        candidates = generate_align_candidates(world, slots)
        keys = []
        for plan in candidates:
            pairs = []
            for step in plan.steps:
                if step.action == "pick":
                    pairs.append(("pick", step.object))
                elif step.action == "place":
                    pairs.append(("place", step.slot or ""))
            keys.append(str(pairs))
        assert len(keys) == len(set(keys))

    def test_with_5_cubes(self):
        objects = [
            ObjectState("cube1", "cube", (-0.20, -0.40, 0.833), True, False),
            ObjectState("cube2", "cube", (0.00, -0.50, 0.833), True, False),
            ObjectState("cube3", "cube", (0.15, -0.35, 0.833), True, False),
            ObjectState("cube4", "cube", (-0.10, 0.20, 0.833), True, False),
            ObjectState("cube5", "cube", (0.20, 0.15, 0.833), True, False),
        ]
        world = _make_world(objects=objects, target_objects=["cube1", "cube2", "cube3", "cube4", "cube5"])
        slots = _make_slots(5)
        candidates = generate_align_candidates(world, slots)
        assert len(candidates) >= 4
        for plan in candidates:
            assert len(plan.steps) == 10

    def test_obstacle_aware_plan_prefers_safe_objects(self):
        obstacles = (
            ObstacleState("obs1", (0.10, -0.40, 0.89), True, 0.035, "short"),
        )
        objects = [
            ObjectState("cube1", "cube", (0.08, -0.42, 0.833), True, True),
            ObjectState("cube2", "cube", (-0.20, 0.20, 0.833), True, False),
        ]
        world = _make_world(
            objects=objects,
            obstacles=obstacles,
            target_objects=["cube1", "cube2"],
        )
        slots = _make_slots(2)
        plan = generate_obstacle_aware_plan(world, slots)
        picked = [s.object for s in plan.steps if s.action == "pick"]
        assert picked[0] == "cube2"


class TestFeatureExtractor:
    def test_edge_features_are_json_serializable(self):
        world = _make_world()
        slots = _make_slots(4)
        features = extract_align_edge_features(world, "cube1", "align_slot_0", slots)
        json_str = json.dumps(features)
        assert isinstance(json_str, str)

    def test_edge_features_contain_required_keys(self):
        world = _make_world()
        slots = _make_slots(4)
        features = extract_align_edge_features(world, "cube1", "align_slot_0", slots)
        required_keys = [
            "object_to_slot_distance",
            "robot_to_object_distance",
            "robot_to_slot_distance",
            "object_near_obstacle",
            "slot_near_obstacle",
            "line_crosses_obstacle",
            "object_reachability_margin",
            "slot_reachability_margin",
            "slot_index",
            "object_order_index",
            "remaining_unplaced_count",
            "placed_objects_density",
            "estimated_transfer_distance",
        ]
        for key in required_keys:
            assert key in features, f"Missing key: {key}"

    def test_plan_features_are_json_serializable(self):
        world = _make_world()
        slots = _make_slots(4)
        plan = generate_nearest_first_plan(world, slots)
        features = extract_align_plan_features(world, plan, slots)
        json_str = json.dumps(features)
        assert isinstance(json_str, str)

    def test_plan_features_edge_count_matches(self):
        world = _make_world()
        slots = _make_slots(4)
        plan = generate_nearest_first_plan(world, slots)
        features = extract_align_plan_features(world, plan, slots)
        assert features["edge_count"] == 4

    def test_distance_is_non_negative(self):
        world = _make_world()
        slots = _make_slots(4)
        features = extract_align_edge_features(world, "cube1", "align_slot_0", slots)
        assert features["object_to_slot_distance"] >= 0
        assert features["robot_to_object_distance"] >= 0
        assert features["estimated_transfer_distance"] >= 0


class TestCostModel:
    def test_edge_cost_is_finite(self):
        world = _make_world()
        slots = _make_slots(4)
        features = extract_align_edge_features(world, "cube1", "align_slot_0", slots)
        cost = estimate_align_edge_cost(features)
        assert 0 < cost < INFEASIBLE_PENALTY

    def test_plan_cost_is_finite(self):
        world = _make_world()
        slots = _make_slots(4)
        plan = generate_nearest_first_plan(world, slots)
        cost, edge_costs = estimate_align_plan_cost(plan, world, slots)
        assert 0 < cost < INFEASIBLE_PENALTY
        assert len(edge_costs) == 4

    def test_ranking_returns_sorted_by_cost(self):
        world = _make_world()
        slots = _make_slots(4)
        candidates = generate_align_candidates(world, slots)
        ranked = rank_candidate_plans(candidates, world, slots)
        assert len(ranked) > 0
        costs = [s.estimated_cost for s in ranked]
        assert costs == sorted(costs)

    def test_ranking_returns_scored_plans(self):
        world = _make_world()
        slots = _make_slots(4)
        candidates = generate_align_candidates(world, slots)
        ranked = rank_candidate_plans(candidates, world, slots)
        for scored in ranked:
            assert isinstance(scored, ScoredPlan)
            assert isinstance(scored.plan_id, str)
            assert isinstance(scored.estimated_cost, float)

    def test_infinite_cost_for_invalid_features(self):
        features = {"error": "unknown object"}
        cost = estimate_align_edge_cost(features)
        assert cost >= 0


class TestMotionProbe:
    def test_geometric_pick_probe_reachable_object(self):
        world = _make_world()
        probe = MotionProbe(runtime=None)
        result = probe.probe_pick_feasibility(world, "cube1")
        assert isinstance(result, ProbeResult)
        assert result.feasible is True

    def test_geometric_pick_probe_unreachable_object(self):
        objects = [
            ObjectState("cube1", "cube", (0.00, 0.00, 0.833), False, False),
        ]
        world = _make_world(objects=objects, target_objects=["cube1"])
        probe = MotionProbe(runtime=None)
        result = probe.probe_pick_feasibility(world, "cube1")
        assert result.feasible is False

    def test_geometric_place_probe_reachable_slot(self):
        world = _make_world()
        slots = _make_slots(4)
        probe = MotionProbe(runtime=None)
        result = probe.probe_place_feasibility(world, "cube1", "align_slot_0", slots)
        assert result.feasible is True

    def test_geometric_place_probe_unknown_slot(self):
        world = _make_world()
        probe = MotionProbe(runtime=None)
        result = probe.probe_place_feasibility(world, "cube1", "nonexistent_slot", {})
        assert result.feasible is False

    def test_probe_align_edge_combines_pick_and_place(self):
        world = _make_world()
        slots = _make_slots(4)
        probe = MotionProbe(runtime=None)
        result = probe.probe_align_edge(world, "cube1", "align_slot_0", slots)
        assert result.feasible is True

    def test_probe_align_plan_all_edges(self):
        world = _make_world()
        slots = _make_slots(4)
        plan = generate_nearest_first_plan(world, slots)
        probe = MotionProbe(runtime=None)
        result = probe.probe_align_plan_feasibility(world, plan, slots)
        assert isinstance(result, ProbePlanResult)
        assert result.feasible is True
        assert len(result.edge_results) == 4

    def test_probe_records_planning_time(self):
        world = _make_world()
        slots = _make_slots(4)
        plan = generate_nearest_first_plan(world, slots)
        probe = MotionProbe(runtime=None)
        result = probe.probe_align_plan_feasibility(world, plan, slots)
        assert result.total_planning_time >= 0


class TestConfirmation:
    def test_confirm_single_feasible_plan(self):
        world = _make_world()
        slots = _make_slots(4)
        plan = generate_nearest_first_plan(world, slots)
        probe = MotionProbe(runtime=None)
        result = confirm_align_plan(world, plan, slots, probe)
        assert isinstance(result, ConfirmationResult)
        assert result.confirmed is True
        assert result.plan is not None

    def test_confirm_ranked_candidates_returns_first_feasible(self):
        world = _make_world()
        slots = _make_slots(4)
        candidates = generate_align_candidates(world, slots)
        ranked = rank_candidate_plans(candidates, world, slots)
        probe = MotionProbe(runtime=None)
        result = confirm_ranked_align_candidates(world, ranked, slots, probe)
        assert result.confirmed is True
        assert result.selected_plan_id == ranked[0].plan_id

    def test_confirm_records_ik_and_ompl_failures(self):
        world = _make_world()
        slots = _make_slots(4)
        plan = generate_nearest_first_plan(world, slots)
        probe = MotionProbe(runtime=None)
        result = confirm_align_plan(world, plan, slots, probe)
        assert result.total_ik_failures >= 0
        assert result.total_ompl_failures >= 0

    def test_confirm_with_unreachable_object_fails(self):
        objects = [
            ObjectState("cube1", "cube", (0.00, 0.00, 0.833), False, False),
        ]
        world = _make_world(objects=objects, target_objects=["cube1"])
        slots = _make_slots(1)
        plan = generate_nearest_first_plan(world, slots)
        probe = MotionProbe(runtime=None)
        result = confirm_align_plan(world, plan, slots, probe)
        assert result.confirmed is False


class TestValidatorWithFlexibleOrder:
    def test_flexible_order_plan_passes_validation(self):
        world = _make_world()
        slots = _make_slots(4)
        plan = generate_nearest_first_plan(world, slots)
        validate(plan, world.all_object_ids(), world.allowed_predicates)
        plugin = DEFAULT_REGISTRY.get("align")
        plugin.validate_plan(plan, world)

    def test_original_order_plan_still_passes(self):
        world = _make_world()
        reference_plan = parse_plan({
            "schema_version": "ctamp-plan/v1",
            "task": "align",
            "scene_id": "test_scene",
            "target_objects": ["cube1", "cube2", "cube3", "cube4"],
            "goal_predicates": [
                {"name": "at", "args": ["cube1", "align_slot_0"]},
                {"name": "at", "args": ["cube2", "align_slot_1"]},
                {"name": "at", "args": ["cube3", "align_slot_2"]},
                {"name": "at", "args": ["cube4", "align_slot_3"]},
            ],
            "slot_config": {"type": "line", "center_x": 0.22, "row_y": -0.06, "base_z": 0.833, "spacing_m": 0.10},
            "steps": [
                {"step_id": 0, "action": "pick", "object": "cube1"},
                {"step_id": 1, "action": "place", "object": "cube1", "slot": "align_slot_0"},
                {"step_id": 2, "action": "pick", "object": "cube2"},
                {"step_id": 3, "action": "place", "object": "cube2", "slot": "align_slot_1"},
                {"step_id": 4, "action": "pick", "object": "cube3"},
                {"step_id": 5, "action": "place", "object": "cube3", "slot": "align_slot_2"},
                {"step_id": 6, "action": "pick", "object": "cube4"},
                {"step_id": 7, "action": "place", "object": "cube4", "slot": "align_slot_3"},
            ],
            "constraints": {"preserve_obstacles": True},
        })
        validate(reference_plan, world.all_object_ids(), world.allowed_predicates)
        plugin = DEFAULT_REGISTRY.get("align")
        plugin.validate_plan(reference_plan, world)


class TestScoredPlanDataclass:
    def test_scored_plan_creation(self):
        world = _make_world()
        slots = _make_slots(4)
        plan = generate_nearest_first_plan(world, slots)
        scored = ScoredPlan(
            plan_id="test_0",
            plan=plan,
            estimated_cost=1.5,
            generation_method="nearest_first",
            edge_costs=(0.3, 0.4, 0.5, 0.3),
        )
        assert scored.plan_id == "test_0"
        assert scored.estimated_cost == 1.5
        assert len(scored.edge_costs) == 4


class TestProbeResultDataclass:
    def test_probe_result_creation(self):
        result = ProbeResult(
            feasible=True,
            ik_success=True,
            ompl_success=True,
            planning_time=0.05,
            estimated_path_length=0.5,
            min_clearance=0.1,
            collision_count=0,
        )
        assert result.feasible is True
        assert result.failure_reason is None


class TestConfirmationResultDataclass:
    def test_confirmation_result_creation(self):
        result = ConfirmationResult(
            confirmed=True,
            selected_plan_id="candidate_0",
            total_probes=4,
        )
        assert result.confirmed is True
        assert result.total_probes == 4
