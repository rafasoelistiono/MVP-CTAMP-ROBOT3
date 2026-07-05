from __future__ import annotations

from pathlib import Path

import mujoco

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
from world.slot_allocator import allocate_grouped_align_slots, validate_slots


ROOT = Path(__file__).resolve().parents[2]
CONTEXT_PATH = ROOT / "contexts/examples/align_grouped_tidy_wall_world.md"
PLAN_PATH = ROOT / "task_plans/examples/align_grouped_tidy_wall_world.json"


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


def test_align_wall_example_passes_complete_deterministic_pipeline(tmp_path):
    world = build_world_state(CONTEXT_PATH)
    plan = load_plan(PLAN_PATH)
    assert world.task_name == plan.task == "align"
    assert plan.scene_id == world.scene_id

    validate(plan, world.all_object_ids(), world.allowed_predicates)
    plugin = DEFAULT_REGISTRY.get("align")
    plugin.validate_plan(plan, world)

    config = plugin.configure_runtime(plan, world, load_runtime_config("obstacle"))
    slots = allocate_grouped_align_slots(world, world.grouped_tidy)
    validate_slots(slots, world, obstacle_buffer_m=config.safety.target_obstacle_buffer_m)

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

    result = TaskRunner(
        plan=plan,
        world=world,
        slots=slots,
        hint_cache=HintCache(tmp_path / "history"),
        plugin_registry=DEFAULT_REGISTRY,
        event_log=EventLog(tmp_path / "events.csv", "example-align"),
        primitives=DeterministicPrimitives({obj.id: obj.pose for obj in world.objects}),
        runtime_config=config,
    ).run()

    assert result.success
    assert result.moved_count == len(plan.target_objects)
    assert not result.failure_reasons
