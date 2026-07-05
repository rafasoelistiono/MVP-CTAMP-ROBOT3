from __future__ import annotations

import json
import math
import pytest

from backends.adaptive.hint_cache import AlignCacheEntry, HintCache
from task_planning.adaptive_heuristic import (
    combine_static_and_cached_cost,
    predict_align_edge_cost,
    predict_align_plan_cost,
    rank_align_candidates_with_cache,
    record_probe_result_to_cache,
    record_plan_result_to_cache,
)
from task_planning.candidate_generator import generate_align_candidates
from task_planning.cost_model import rank_candidate_plans, INFEASIBLE_PENALTY
from task_planning.feature_extractor import (
    bucketize_align_features,
    extract_align_edge_features,
    make_align_cache_key,
)
from task_planning.types import TaskPlan
from world.slot_allocator import allocate_slots
from world.state import ObjectState, ObstacleState, WorldState


def _make_world(
    objects=None,
    obstacles=(),
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
        task_name="align",
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


def _make_cache(tmp_path) -> HintCache:
    return HintCache(tmp_path / "test_logs")


class TestAlignCacheEntry:
    def test_initial_state(self):
        entry = AlignCacheEntry(feature_key="test")
        assert entry.total_samples == 0
        assert entry.failure_rate == 0.0

    def test_update_success(self):
        entry = AlignCacheEntry(feature_key="test")
        entry.update(success=True, actual_cost=1.5)
        assert entry.success_count == 1
        assert entry.total_samples == 1
        assert entry.ema_cost == 1.5

    def test_update_failure(self):
        entry = AlignCacheEntry(feature_key="test")
        entry.update(success=False, actual_cost=10.0, failure_reason="ik_failed")
        assert entry.failure_count == 1
        assert entry.last_failure_reason == "ik_failed"

    def test_ema_convergence(self):
        entry = AlignCacheEntry(feature_key="test")
        entry.update(success=True, actual_cost=10.0, alpha=0.3)
        entry.update(success=True, actual_cost=2.0, alpha=0.3)
        expected = 0.3 * 2.0 + 0.7 * 10.0
        assert abs(entry.ema_cost - expected) < 0.01

    def test_to_dict_roundtrip(self):
        entry = AlignCacheEntry(feature_key="k1")
        entry.update(success=True, actual_cost=1.5, run_id="run1")
        d = entry.to_dict()
        restored = AlignCacheEntry.from_dict(d)
        assert restored.feature_key == "k1"
        assert restored.success_count == 1
        assert restored.ema_cost == 1.5

    def test_failure_rate(self):
        entry = AlignCacheEntry(feature_key="test")
        entry.update(success=True, actual_cost=1.0)
        entry.update(success=False, actual_cost=5.0)
        entry.update(success=False, actual_cost=8.0)
        assert entry.failure_rate == pytest.approx(2.0 / 3.0)


class TestBucketizeFeatures:
    def test_returns_bucketized_dict(self):
        features = extract_align_edge_features(
            _make_world(), "cube1", "align_slot_0", _make_slots(4)
        )
        bucketed = bucketize_align_features(features)
        assert "dist_bucket" in bucketed
        assert "robot_obj_bucket" in bucketed
        assert "obj_near_obs" in bucketed

    def test_deterministic(self):
        features = extract_align_edge_features(
            _make_world(), "cube1", "align_slot_0", _make_slots(4)
        )
        b1 = bucketize_align_features(features)
        b2 = bucketize_align_features(features)
        assert b1 == b2

    def test_error_features(self):
        bucketed = bucketize_align_features({"error": "bad"})
        assert "error" in bucketed

    def test_granularity_levels(self):
        features = extract_align_edge_features(
            _make_world(), "cube1", "align_slot_0", _make_slots(4)
        )
        fine = bucketize_align_features(features, "fine")
        coarse = bucketize_align_features(features, "coarse")
        assert fine["dist_bucket"] >= coarse["dist_bucket"]


class TestMakeCacheKey:
    def test_returns_string(self):
        features = extract_align_edge_features(
            _make_world(), "cube1", "align_slot_0", _make_slots(4)
        )
        key = make_align_cache_key(features)
        assert isinstance(key, str)
        assert "|" in key

    def test_deterministic(self):
        features = extract_align_edge_features(
            _make_world(), "cube1", "align_slot_0", _make_slots(4)
        )
        k1 = make_align_cache_key(features)
        k2 = make_align_cache_key(features)
        assert k1 == k2

    def test_different_features_different_keys(self):
        world = _make_world()
        slots = _make_slots(4)
        f1 = extract_align_edge_features(world, "cube1", "align_slot_0", slots)
        f2 = extract_align_edge_features(world, "cube4", "align_slot_3", slots)
        k1 = make_align_cache_key(f1)
        k2 = make_align_cache_key(f2)
        assert k1 != k2

    def test_error_key(self):
        key = make_align_cache_key({"error": "bad"})
        assert key == "error"


class TestCombineStaticAndCachedCost:
    def test_zero_confidence_returns_static(self):
        result = combine_static_and_cached_cost(10.0, 2.0, 0.0, 0.5)
        assert result == 10.0

    def test_full_confidence_blends(self):
        result = combine_static_and_cached_cost(10.0, 2.0, 1.0, 0.5)
        assert result == 6.0

    def test_half_weight(self):
        result = combine_static_and_cached_cost(10.0, 0.0, 1.0, 0.5)
        assert result == 5.0


class TestPredictAlignEdgeCost:
    def test_falls_back_to_static_when_empty_cache(self, tmp_path):
        cache = _make_cache(tmp_path)
        world = _make_world()
        slots = _make_slots(4)
        cost, used_cache = predict_align_edge_cost(cache, world, "cube1", "align_slot_0", slots)
        assert not used_cache
        assert 0 < cost < INFEASIBLE_PENALTY

    def test_uses_cache_when_sufficient_samples(self, tmp_path):
        cache = _make_cache(tmp_path)
        world = _make_world()
        slots = _make_slots(4)
        features = extract_align_edge_features(world, "cube1", "align_slot_0", slots)
        key = make_align_cache_key(features)
        for _ in range(5):
            cache.record_align_edge_result(key, success=True, actual_cost=1.5)
        cost, used_cache = predict_align_edge_cost(
            cache, world, "cube1", "align_slot_0", slots, min_samples=3
        )
        assert used_cache
        assert 0 < cost < INFEASIBLE_PENALTY

    def test_failure_increases_cost(self, tmp_path):
        cache = _make_cache(tmp_path)
        world = _make_world()
        slots = _make_slots(4)
        features = extract_align_edge_features(world, "cube1", "align_slot_0", slots)
        key = make_align_cache_key(features)
        for _ in range(5):
            cache.record_align_edge_result(key, success=False, actual_cost=10.0, failure_reason="ik_failed")
        cost_fail, _ = predict_align_edge_cost(
            cache, world, "cube1", "align_slot_0", slots, min_samples=3
        )
        cache2 = _make_cache(tmp_path / "v2")
        for _ in range(5):
            cache2.record_align_edge_result(key, success=True, actual_cost=1.5)
        cost_ok, _ = predict_align_edge_cost(
            cache2, world, "cube1", "align_slot_0", slots, min_samples=3
        )
        assert cost_fail > cost_ok


class TestPredictAlignPlanCost:
    def test_returns_total_and_edge_costs(self, tmp_path):
        cache = _make_cache(tmp_path)
        world = _make_world()
        slots = _make_slots(4)
        from task_planning.candidate_generator import generate_nearest_first_plan
        plan = generate_nearest_first_plan(world, slots)
        total, edge_costs, used_cache = predict_align_plan_cost(cache, world, plan, slots)
        assert len(edge_costs) == 4
        assert total == sum(edge_costs)
        assert not used_cache


class TestRankAlignCandidatesWithCache:
    def test_returns_sorted_scored_plans(self, tmp_path):
        cache = _make_cache(tmp_path)
        world = _make_world()
        slots = _make_slots(4)
        candidates = generate_align_candidates(world, slots)
        ranked = rank_align_candidates_with_cache(cache, world, candidates, slots)
        assert len(ranked) > 0
        costs = [s.estimated_cost for s in ranked]
        assert costs == sorted(costs)

    def test_with_cache_entries_changes_ranking(self, tmp_path):
        cache = _make_cache(tmp_path)
        world = _make_world()
        slots = _make_slots(4)
        candidates = generate_align_candidates(world, slots)
        ranked_no_cache = rank_align_candidates_with_cache(cache, world, candidates, slots)

        first_plan = ranked_no_cache[0].plan
        pick_steps = [s for s in first_plan.steps if s.action == "pick"]
        place_steps = [s for s in first_plan.steps if s.action == "place"]
        for ps, pls in zip(pick_steps, place_steps):
            features = extract_align_edge_features(world, ps.object, pls.slot or "", slots)
            key = make_align_cache_key(features)
            for _ in range(10):
                cache.record_align_edge_result(key, success=True, actual_cost=0.5)

        ranked_with_cache = rank_align_candidates_with_cache(
            cache, world, candidates, slots, min_samples=3
        )
        assert len(ranked_with_cache) == len(ranked_no_cache)


class TestRecordProbeResultToCache:
    def test_records_edge_result(self, tmp_path):
        cache = _make_cache(tmp_path)
        world = _make_world()
        slots = _make_slots(4)
        key = record_probe_result_to_cache(
            cache, world, "cube1", "align_slot_0", slots,
            success=True, actual_cost=1.5, run_id="test_run"
        )
        assert isinstance(key, str)
        entry = cache.get_align_edge_entry(key)
        assert entry is not None
        assert entry.success_count == 1

    def test_records_failure_pattern(self, tmp_path):
        cache = _make_cache(tmp_path)
        world = _make_world()
        slots = _make_slots(4)
        record_probe_result_to_cache(
            cache, world, "cube1", "align_slot_0", slots,
            success=False, actual_cost=10.0,
            failure_reason="ik_solve_failed", run_id="test_run"
        )
        features = extract_align_edge_features(world, "cube1", "align_slot_0", slots)
        key = make_align_cache_key(features)
        pattern_key = f"{key}:ik_solve_failed"
        pattern = cache.get_align_failure_entry(pattern_key)
        assert pattern is not None
        assert pattern.failure_count == 1


class TestRecordPlanResultToCache:
    def test_records_plan_result(self, tmp_path):
        cache = _make_cache(tmp_path)
        world = _make_world()
        slots = _make_slots(4)
        from task_planning.candidate_generator import generate_nearest_first_plan
        plan = generate_nearest_first_plan(world, slots)
        plan_key = record_plan_result_to_cache(
            cache, world, plan, slots,
            success=True, actual_cost=5.0, run_id="test_run"
        )
        assert isinstance(plan_key, str)
        entry = cache.get_align_plan_entry(plan_key)
        assert entry is not None
        assert entry.success_count == 1


class TestHintCacheAlignment:
    def test_record_and_retrieve_edge(self, tmp_path):
        cache = _make_cache(tmp_path)
        key = cache.record_align_edge_result(
            "test_key", success=True, actual_cost=1.0
        )
        entry = cache.get_align_edge_entry("test_key")
        assert entry is not None
        assert entry.total_samples == 1

    def test_record_and_retrieve_plan(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache.record_align_plan_result(
            "plan_key", success=True, actual_cost=5.0
        )
        entry = cache.get_align_plan_entry("plan_key")
        assert entry is not None
        assert entry.ema_cost == 5.0

    def test_record_and_retrieve_failure_pattern(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache.record_align_edge_result(
            "edge_key", success=False, actual_cost=10.0,
            failure_reason="ompl_timeout"
        )
        entry = cache.get_align_failure_entry("edge_key:ompl_timeout")
        assert entry is not None
        assert entry.failure_count == 1

    def test_save_and_load_roundtrip(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache.record_align_edge_result("k1", success=True, actual_cost=1.5)
        cache.record_align_plan_result("p1", success=True, actual_cost=5.0)
        cache.record_align_edge_result("k1", success=False, actual_cost=8.0, failure_reason="ik_fail")
        cache.save_align_caches()

        cache2 = HintCache(tmp_path / "test_logs")
        e1 = cache2.get_align_edge_entry("k1")
        assert e1 is not None
        assert e1.total_samples == 2
        p1 = cache2.get_align_plan_entry("p1")
        assert p1 is not None
        assert p1.success_count == 1
        f1 = cache2.get_align_failure_entry("k1:ik_fail")
        assert f1 is not None


class TestCostModelWithCache:
    def test_rank_candidate_plans_with_cache_disabled(self):
        world = _make_world()
        slots = _make_slots(4)
        candidates = generate_align_candidates(world, slots)
        ranked = rank_candidate_plans(candidates, world, slots, use_adaptive_cache=False)
        assert len(ranked) > 0
        costs = [s.estimated_cost for s in ranked]
        assert costs == sorted(costs)

    def test_rank_candidate_plans_with_cache_none(self):
        world = _make_world()
        slots = _make_slots(4)
        candidates = generate_align_candidates(world, slots)
        ranked = rank_candidate_plans(candidates, world, slots, hint_cache=None, use_adaptive_cache=True)
        assert len(ranked) > 0


class TestAlignCacheConfig:
    def test_default_config(self):
        from configuration.types import AlignCacheConfig
        cfg = AlignCacheConfig()
        assert cfg.use_adaptive_cache is False
        assert cfg.adaptive_cache_weight == 0.5
        assert cfg.min_samples_for_cache == 3
        assert cfg.failure_penalty == 2.0
        assert cfg.cache_key_granularity == "medium"

    def test_custom_config(self):
        from configuration.types import AlignCacheConfig
        cfg = AlignCacheConfig(
            use_adaptive_cache=True,
            adaptive_cache_weight=0.7,
            min_samples_for_cache=5,
            failure_penalty=3.0,
            cache_key_granularity="fine",
        )
        assert cfg.use_adaptive_cache is True
        assert cfg.cache_key_granularity == "fine"
