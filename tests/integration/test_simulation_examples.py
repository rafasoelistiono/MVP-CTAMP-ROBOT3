from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import pytest

from backends.adaptive.event_log import EventLog
from backends.adaptive.hint_cache import HintCache
from configuration import load_runtime_config
from execution.primitives import PrimitiveResult
from execution.runner import TaskRunner
from plugins.registry import DEFAULT_REGISTRY
from scene import prepare_scene_variant
from task_planning.loader import load_plan
from task_planning.validator import validate
from world.builder import build_world_state
from world.slot_allocator import (
    allocate_grouped_align_slots,
    allocate_slots,
    validate_slots,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = (
    (
        "align",
        ROOT / "contexts/examples/align_grouped_tidy_wall_world.md",
        ROOT / "task_plans/examples/align_grouped_tidy_wall_world.json",
    ),
    (
        "stack",
        ROOT / "contexts/examples/ungroup_obs_stack_cubes.md",
        ROOT / "task_plans/examples/ungroup_obs_stack_cubes.json",
    ),
    (
        "pyramid",
        ROOT / "contexts/examples/ungroup_obs_pyramid_cubes.md",
        ROOT / "task_plans/examples/ungroup_obs_pyramid_cubes.json",
    ),
)


class DeterministicPrimitives:
    def __init__(self, poses):
        self.poses = dict(poses)
        self.held = None
        self.settle_calls = []

    def execute(self, step, target, hints):
        if step.action == "pick":
            x, y, _ = self.poses[step.object]
            self.poses[step.object] = (x, y, 0.96)
            self.held = step.object
        else:
            assert target is not None
            self.poses[step.object] = target
            self.held = None
        return PrimitiveResult(True)

    def object_pose(self, object_id):
        return self.poses[object_id]

    def all_object_poses(self):
        return dict(self.poses)

    def held_object_name(self):
        return self.held

    def settle_for_verification(self, steps):
        self.settle_calls.append(steps)


class FallingStackPrimitives(DeterministicPrimitives):
    """Inject one lower-layer displacement while the final cube is carried."""

    def __init__(self, poses):
        super().__init__(poses)
        self.disturbance_injected = False

    def execute(self, step, target, hints):
        result = super().execute(step, target, hints)
        if (
            step.action == "pick"
            and step.object == "cube4"
            and not self.disturbance_injected
        ):
            self.poses["cube2"] = (0.42, 0.42, 0.833)
            self.disturbance_injected = True
        return result


class EarlyFallingStackPrimitives(DeterministicPrimitives):
    """Drop a lower layer early and record recovery before the next cube."""

    def __init__(self, poses):
        super().__init__(poses)
        self.executed_step_ids = []
        self.disturbance_injected = False

    def execute(self, step, target, hints):
        self.executed_step_ids.append(step.step_id)
        result = super().execute(step, target, hints)
        if (
            step.action == "stack_place"
            and step.object == "cube2"
            and not self.disturbance_injected
        ):
            self.poses["cube2"] = (0.42, 0.42, 0.833)
            self.disturbance_injected = True
        return result


class FailedBaseOncePrimitives(DeterministicPrimitives):
    """Make the first red-cube base placement miss once."""

    def __init__(self, poses):
        super().__init__(poses)
        self.executed = []
        self.failed_once = False

    def execute(self, step, target, hints):
        self.executed.append((step.action, step.object))
        result = super().execute(step, target, hints)
        if (
            step.action == "place"
            and step.object == "cube1"
            and not self.failed_once
        ):
            self.poses["cube1"] = (0.42, 0.42, 0.833)
            self.failed_once = True
        return result


class AlwaysFallingStackPrimitives(DeterministicPrimitives):
    """Keep the second layer invalid so the rebuild bound is exercised."""

    def execute(self, step, target, hints):
        result = super().execute(step, target, hints)
        if step.action == "stack_place" and step.object == "cube2":
            self.poses["cube2"] = (0.42, 0.42, 0.833)
        return result


class AlreadyStackedPrimitives(DeterministicPrimitives):
    def __init__(self, poses):
        super().__init__(poses)
        self.execute_calls = 0

    def execute(self, step, target, hints):
        self.execute_calls += 1
        return super().execute(step, target, hints)


@pytest.mark.parametrize(("task", "context_path", "plan_path"), EXAMPLES)
def test_obstacle_example_passes_complete_deterministic_pipeline(
    tmp_path,
    task,
    context_path,
    plan_path,
):
    world = build_world_state(context_path)
    plan = load_plan(plan_path)
    assert world.task_name == task == plan.task
    assert plan.scene_id == world.scene_id

    validate(plan, world.all_object_ids(), world.allowed_predicates)
    plugin = DEFAULT_REGISTRY.get(task)
    plugin.validate_plan(plan, world)

    config = plugin.configure_runtime(
        plan,
        world,
        load_runtime_config("obstacle"),
    )
    if world.grouped_tidy and world.grouped_tidy.enabled:
        slots = allocate_grouped_align_slots(world, world.grouped_tidy)
    else:
        slots = allocate_slots(
            plugin.make_slot_config(plan, world),
            len(plan.target_objects),
        )
    validate_slots(
        slots,
        world,
        obstacle_buffer_m=config.safety.target_obstacle_buffer_m,
    )

    scene_path = prepare_scene_variant(
        world.variant,
        base_model_file=config.model.xml_path,
        object_states=world.objects,
        obstacle_states=world.obstacles,
        goal_center=world.goal_center,
        goal_area_size_xy=world.goal_area_size_xy,
            table_size_xy=(
                world.table_x_range[1] - world.table_x_range[0],
                world.table_y_range[1] - world.table_y_range[0],
            ),
            base_xy=world.robot_base_xy,
            base_z=world.robot_base_z,
        )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.nbody > 0

    primitives = DeterministicPrimitives(
        {obj.id: obj.pose for obj in world.objects}
    )
    result = TaskRunner(
        plan=plan,
        world=world,
        slots=slots,
        hint_cache=HintCache(tmp_path / "history"),
        plugin_registry=DEFAULT_REGISTRY,
        event_log=EventLog(tmp_path / "events.csv", f"example-{task}"),
        primitives=primitives,
        runtime_config=config,
    ).run()

    assert result.success
    assert result.moved_count == len(plan.target_objects)
    assert not result.failure_reasons


def test_stack_fall_rebuilds_only_invalid_suffix(tmp_path):
    context_path = ROOT / "contexts/examples/ungroup_obs_stack_cubes.md"
    plan_path = ROOT / "task_plans/examples/ungroup_obs_stack_cubes.json"
    world = build_world_state(context_path)
    plan = load_plan(plan_path)
    validate(plan, world.all_object_ids(), world.allowed_predicates)
    plugin = DEFAULT_REGISTRY.get("stack")
    plugin.validate_plan(plan, world)
    config = plugin.configure_runtime(plan, world, load_runtime_config("obstacle"))
    slots = allocate_slots(plugin.make_slot_config(plan, world), 4)
    primitives = FallingStackPrimitives(
        {obj.id: obj.pose for obj in world.objects}
    )
    event_path = tmp_path / "events.csv"

    result = TaskRunner(
        plan=plan,
        world=world,
        slots=slots,
        hint_cache=HintCache(tmp_path / "history"),
        plugin_registry=DEFAULT_REGISTRY,
        event_log=EventLog(event_path, "stack-fall"),
        primitives=primitives,
        runtime_config=config,
    ).run()

    assert primitives.disturbance_injected
    assert result.success
    assert result.moved_count == 4
    assert len(result.step_results) > len(plan.steps)
    assert primitives.object_pose("cube1") == pytest.approx(slots["tower_base"])
    assert "STACK_REBUILD" in event_path.read_text(encoding="utf-8")
    assert ",OK," in event_path.read_text(encoding="utf-8")


def test_stack_recovers_current_cube_before_picking_the_next_cube(tmp_path):
    context_path = ROOT / "contexts/examples/ungroup_obs_stack_cubes.md"
    plan = load_plan(ROOT / "task_plans/examples/ungroup_obs_stack_cubes.json")
    world = build_world_state(context_path)
    plugin = DEFAULT_REGISTRY.get("stack")
    config = plugin.configure_runtime(plan, world, load_runtime_config("obstacle"))
    slots = allocate_slots(plugin.make_slot_config(plan, world), 4)
    primitives = EarlyFallingStackPrimitives(
        {obj.id: obj.pose for obj in world.objects}
    )

    result = TaskRunner(
        plan=plan,
        world=world,
        slots=slots,
        hint_cache=HintCache(tmp_path / "history"),
        plugin_registry=DEFAULT_REGISTRY,
        event_log=EventLog(tmp_path / "events.csv", "stack-level-check"),
        primitives=primitives,
        runtime_config=config,
    ).run()

    assert result.success
    assert primitives.disturbance_injected
    cube2_place_index = primitives.executed_step_ids.index(3)
    cube3_pick_index = primitives.executed_step_ids.index(4)
    recovery_steps = primitives.executed_step_ids[
        cube2_place_index + 1 : cube3_pick_index
    ]
    assert len(recovery_steps) == 2
    assert all(step_id >= len(plan.steps) for step_id in recovery_steps)
    assert primitives.settle_calls == [
        config.recovery.verification_settle_steps,
        config.recovery.verification_settle_steps,
    ]


def test_already_completed_stack_is_terminal_without_repeating_actions(tmp_path):
    context_path = ROOT / "contexts/examples/ungroup_obs_stack_cubes.md"
    plan = load_plan(ROOT / "task_plans/examples/ungroup_obs_stack_cubes.json")
    world = build_world_state(context_path)
    plugin = DEFAULT_REGISTRY.get("stack")
    config = plugin.configure_runtime(plan, world, load_runtime_config("obstacle"))
    slots = allocate_slots(plugin.make_slot_config(plan, world), 4)
    base = slots["tower_base"]
    primitives = AlreadyStackedPrimitives(
        {
            object_id: (base[0], base[1], base[2] + index * 0.066)
            for index, object_id in enumerate(plan.target_objects)
        }
    )
    event_path = tmp_path / "events.csv"

    result = TaskRunner(
        plan=plan,
        world=world,
        slots=slots,
        hint_cache=HintCache(tmp_path / "history"),
        plugin_registry=DEFAULT_REGISTRY,
        event_log=EventLog(event_path, "already-stacked"),
        primitives=primitives,
        runtime_config=config,
    ).run()

    assert result.success
    assert result.moved_count == 4
    assert primitives.execute_calls == 0
    assert "STACK_COMPLETE,TERMINAL" in event_path.read_text(encoding="utf-8")


def test_failed_red_cube_is_picked_again_and_reaches_base_before_cube2(tmp_path):
    context_path = ROOT / "contexts/examples/ungroup_obs_stack_cubes.md"
    plan = load_plan(ROOT / "task_plans/examples/ungroup_obs_stack_cubes.json")
    world = build_world_state(context_path)
    plugin = DEFAULT_REGISTRY.get("stack")
    config = plugin.configure_runtime(plan, world, load_runtime_config("obstacle"))
    slots = allocate_slots(plugin.make_slot_config(plan, world), 4)
    primitives = FailedBaseOncePrimitives(
        {obj.id: obj.pose for obj in world.objects}
    )

    result = TaskRunner(
        plan=plan,
        world=world,
        slots=slots,
        hint_cache=HintCache(tmp_path / "history"),
        plugin_registry=DEFAULT_REGISTRY,
        event_log=EventLog(tmp_path / "events.csv", "red-base-retry"),
        primitives=primitives,
        runtime_config=config,
    ).run()

    assert result.success
    assert primitives.executed[:5] == [
        ("pick", "cube1"),
        ("place", "cube1"),
        ("pick", "cube1"),
        ("place", "cube1"),
        ("pick", "cube2"),
    ]
    assert primitives.object_pose("cube1") == pytest.approx(slots["tower_base"])


def test_stack_stops_after_three_failed_rebuilds(tmp_path):
    context_path = ROOT / "contexts/examples/ungroup_obs_stack_cubes.md"
    plan = load_plan(ROOT / "task_plans/examples/ungroup_obs_stack_cubes.json")
    world = build_world_state(context_path)
    plugin = DEFAULT_REGISTRY.get("stack")
    config = plugin.configure_runtime(plan, world, load_runtime_config("obstacle"))
    slots = allocate_slots(plugin.make_slot_config(plan, world), 4)
    event_path = tmp_path / "events.csv"

    result = TaskRunner(
        plan=plan,
        world=world,
        slots=slots,
        hint_cache=HintCache(tmp_path / "history"),
        plugin_registry=DEFAULT_REGISTRY,
        event_log=EventLog(event_path, "stack-rebuild-limit"),
        primitives=AlwaysFallingStackPrimitives(
            {obj.id: obj.pose for obj in world.objects}
        ),
        runtime_config=config,
    ).run()

    rows = event_path.read_text(encoding="utf-8").splitlines()
    starts = [row for row in rows if "STACK_REBUILD" in row and ",START," in row]
    assert not result.success
    assert result.failure_reasons == ("stack_rebuild_exhausted",)
    assert len(starts) == 3


def test_stack_plan_rejects_repeated_pick_place_pairs():
    plan = load_plan(ROOT / "task_plans/examples/ungroup_obs_stack_cubes.json")
    world = build_world_state(
        ROOT / "contexts/examples/ungroup_obs_stack_cubes.md"
    )
    repeated = replace(
        plan,
        steps=plan.steps
        + (
            replace(plan.steps[-2], step_id=8),
            replace(plan.steps[-1], step_id=9),
        ),
    )

    with pytest.raises(ValueError, match="exactly one pick/place pair per cube"):
        DEFAULT_REGISTRY.get("stack").validate_plan(repeated, world)
