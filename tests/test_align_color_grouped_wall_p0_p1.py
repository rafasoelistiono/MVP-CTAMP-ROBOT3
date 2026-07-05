"""Fast acceptance tests for the align_color_grouped_wall P0/P1 task.

Run from the repository root:

    python -m pytest -q /path/to/test_align_color_grouped_wall_p0_p1.py

Most tests are intentionally red on the pre-P0/P1 implementation.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import pytest


REPO = Path(
    os.environ.get("CTAMP_REPO", "/home/rafasoelistiono/MVP-CTAMP-ROBOT2")
).resolve()
sys.path.insert(0, str(REPO))

from configuration import load_runtime_config
from backends.adaptive.event_log import EventLog
from backends.adaptive.hint_cache import HintCache
from backends.mujoco import collision
from cli import generate_plan
from execution import motion_probe
from execution.primitives import PrimitiveResult
from execution.runner import TaskRunner
from plugins.registry import DEFAULT_REGISTRY
from scene import prepare_scene_variant
from task_planning import generator
from task_planning.candidate_generator import generate_align_candidates
from task_planning.cost_model import estimate_align_plan_cost
from task_planning.loader import load_plan
from world.builder import ContextValidationError, build_world_state
from world.slot_allocator import allocate_grouped_align_slots


CONTEXT = REPO / "contexts/examples/align_grouped_tidy_wall_world.md"
PLAN = REPO / "task_plans/examples/align_grouped_tidy_wall_world.json"


@pytest.fixture(scope="module")
def world():
    return build_world_state(CONTEXT)


@pytest.fixture(scope="module")
def plan():
    return load_plan(PLAN)


@pytest.fixture(scope="module")
def slots(world):
    return allocate_grouped_align_slots(world, world.grouped_tidy)


@pytest.fixture(scope="module")
def variant_config(world, plan):
    return DEFAULT_REGISTRY.get("align").configure_runtime(
        plan,
        world,
        load_runtime_config("obstacle", enable_viewer=False),
    )


@pytest.fixture(scope="module")
def scene_path(world, variant_config):
    return prepare_scene_variant(
        world.variant,
        base_model_file=variant_config.model.xml_path,
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


def _xml_body_xyz(scene_path: Path, body_name: str) -> tuple[float, float, float]:
    root = ET.parse(scene_path).getroot()
    body = root.find(f"./worldbody/body[@name='{body_name}']")
    assert body is not None, body_name
    values = [float(value) for value in body.get("pos", "").split()]
    return values[0], values[1], values[2]


def test_p0_wall_contract_is_immutable(world, scene_path):
    wall = world.obstacles[0]
    assert wall.id == "frontal_tall_wall"
    assert wall.pose == pytest.approx((0.00, -0.08, 1.60))
    assert wall.size == pytest.approx((0.08, 0.20, 1.60))
    assert wall.fragile is True

    root = ET.parse(scene_path).getroot()
    body = root.find("./worldbody/body[@name='frontal_tall_wall']")
    assert body is not None
    assert body.find("joint") is None
    assert [float(v) for v in body.get("pos", "").split()] == pytest.approx(
        wall.pose
    )
    geom = body.find("geom")
    assert geom is not None
    assert geom.get("type") == "box"
    assert [float(v) for v in geom.get("size", "").split()] == pytest.approx(
        (0.04, 0.10, 0.80)
    )
    assert geom.get("contype") == "1"
    assert geom.get("conaffinity") == "1"

    carry_welds = root.findall("./equality/weld")
    assert {weld.get("name") for weld in carry_welds} == {
        f"carry_{object_id}" for object_id in world.target_objects
    }
    assert all(weld.get("active") == "false" for weld in carry_welds)
    assert all(weld.get("body1") == "hand" for weld in carry_welds)


def test_p0_unchanged_wall_has_a_right_side_workspace_corridor(world):
    wall = world.obstacles[0]
    half_x = wall.size[0] / 2.0
    cube_half_width = 0.033
    planning_clearance = 0.13
    corridor_min_x = wall.pose[0] + half_x + cube_half_width + planning_clearance
    corridor_width = world.table_x_range[1] - corridor_min_x

    # This only proves workspace room exists. IK plus OMPL tests must prove the
    # corresponding configuration-space corridor.
    assert corridor_width >= 0.50
    assert world.challenge.wall_blocks_direct_path is True
    assert world.challenge.side_corridors_required is True


def test_p0_world_config_and_mujoco_share_one_base(
    world, variant_config, scene_path
):
    model_base = _xml_body_xyz(scene_path, "link0")
    assert world.robot_base_xy == pytest.approx(variant_config.model.base_xy, abs=1e-3)
    assert world.robot_base_xy == pytest.approx(model_base[:2], abs=1e-3)
    assert world.robot_base_z == pytest.approx(variant_config.model.base_z, abs=1e-3)
    assert world.robot_base_z == pytest.approx(model_base[2], abs=1e-3)


def test_p0_all_objects_pass_runtime_reach_precheck(
    world, variant_config, scene_path
):
    model_base = _xml_body_xyz(scene_path, "link0")[:2]
    limit = variant_config.safety.max_pick_object_xy_m
    failures = {
        obj.id: round(math.dist(obj.pose[:2], model_base), 4)
        for obj in world.objects
        if math.dist(obj.pose[:2], model_base) > limit
    }
    assert not failures, f"objects outside runtime reach {limit}: {failures}"


def test_p0_all_slots_pass_runtime_reach_precheck(
    slots, variant_config, scene_path
):
    model_base = _xml_body_xyz(scene_path, "link0")[:2]
    limit = variant_config.safety.max_pick_object_xy_m
    failures = {
        slot_id: round(math.dist(pose[:2], model_base), 4)
        for slot_id, pose in slots.items()
        if math.dist(pose[:2], model_base) > limit
    }
    assert not failures, f"slots outside runtime reach {limit}: {failures}"


def test_p0_challenge_enforces_motion_probe_without_optional_cli_flag(world):
    required = getattr(motion_probe, "requires_motion_probe", None)
    assert callable(required), "add execution.motion_probe.requires_motion_probe(world)"
    assert required(world) is True


def test_p0_collision_policy_has_semantic_obstacle_registry():
    resolver = getattr(collision, "resolve_obstacle_body_ids", None)
    assert callable(resolver), "resolve obstacles from WorldState IDs, not name tokens"

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='wall'/></worldbody></mujoco>"
    )
    assert resolver(model, ["wall"]) == {"wall": model.body("wall").id}
    with pytest.raises(ValueError, match="missing"):
        resolver(model, ["wrong_wall_id"])


def test_p0_collision_policy_supports_attached_object_geometry():
    attach = getattr(collision.CollisionPolicy, "attach_body", None)
    detach = getattr(collision.CollisionPolicy, "detach_body", None)
    assert callable(attach) and callable(detach)

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="hand" pos="0 0 1"><geom type="sphere" size="0.02"/></body>
            <body name="cube" pos="0.50 0 1">
              <freejoint/><geom type="box" size="0.04 0.04 0.04"/>
            </body>
            <body name="wall" pos="0.53 0 1">
              <geom type="box" size="0.04 0.20 0.40"/>
            </body>
            <body name="table" pos="0.50 0 0.91">
              <geom type="box" size="0.20 0.20 0.05"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    policy = collision.CollisionPolicy(
        model,
        robot_body_names=("hand",),
        obstacle_body_names=("wall",),
        obstacle_penetration_tolerance=0.0,
    )
    assert policy.check_contacts(data).valid
    policy.attach_body("cube")
    report = policy.check_contacts(data)
    assert not report.valid
    assert {report.body1, report.body2} == {"cube", "wall"}
    policy.detach_body("cube")
    assert policy.check_contacts(data).valid

    policy.set_ignored_bodies(("wall",))
    policy.attach_body("cube")
    assert policy.check_contacts(data).valid
    cube_body_id = model.body("cube").id
    cube_joint_id = int(model.body_jntadr[cube_body_id])
    cube_qpos_address = int(model.jnt_qposadr[cube_joint_id])
    data.qpos[cube_qpos_address + 2] -= 0.02
    mujoco.mj_forward(model, data)
    deep_support = policy.check_contacts(data)
    assert not deep_support.valid
    assert {deep_support.body1, deep_support.body2} == {"cube", "table"}


def test_p0_panda_mesh_clearance_does_not_report_false_zero(scene_path):
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    policy = collision.CollisionPolicy(
        model,
        obstacle_body_names=("frontal_tall_wall",),
    )
    assert any(
        int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        for geom_id in policy.robot_geom_ids
    )
    assert policy.minimum_obstacle_geom_clearance(data) > 0.0


def test_p1_context_rejects_object_assigned_to_wrong_color_group(tmp_path):
    text = CONTEXT.read_text(encoding="utf-8")
    text = text.replace(
        "objects: [a, c, e, g, i, k]",
        "objects: [b, c, e, g, i, k]",
    ).replace(
        "objects: [b, d, f, h, j, l]",
        "objects: [a, d, f, h, j, l]",
    )
    bad_context = tmp_path / "wrong_color_group.md"
    bad_context.write_text(text, encoding="utf-8")

    with pytest.raises(ContextValidationError, match="color"):
        build_world_state(bad_context)


def test_p1_prompt_builder_contains_grounded_json_contract(world, slots):
    build_prompt = getattr(generator, "build_task_prompt", None)
    assert callable(build_prompt), "add task_planning.generator.build_task_prompt"
    prompt = build_prompt(CONTEXT.read_text(encoding="utf-8"), world, slots)

    required_fragments = {
        "schema_version",
        "ctamp-plan/v1",
        "tidy_slot_blue_lane_0",
        "tidy_slot_red_lane_5",
        "frontal_tall_wall",
        '"pick"',
        '"place"',
        "24",
    }
    missing = sorted(fragment for fragment in required_fragments if fragment not in prompt)
    assert not missing, f"prompt is missing grounded contract fragments: {missing}"
    for object_id in world.target_objects:
        assert object_id in prompt


def test_p1_generate_plan_cli_sends_grounded_prompt(
    world, tmp_path, monkeypatch
):
    captured = {}
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        generate_plan,
        "_arguments",
        lambda: SimpleNamespace(
            context=CONTEXT,
            task=world.task_name,
            output=tmp_path,
            response_file=None,
            experiment_label="",
        ),
    )

    def fake_request(prompt, settings):
        captured["prompt"] = prompt
        return payload

    monkeypatch.setattr(
        generate_plan.LLMSettings,
        "from_env",
        classmethod(lambda cls: SimpleNamespace()),
    )
    monkeypatch.setattr(generate_plan, "request_task_plan", fake_request)
    assert generate_plan.main() == 0
    assert "frontal_tall_wall" in captured["prompt"]
    assert "tidy_slot_blue_lane_0" in captured["prompt"]
    assert '"schema_version"' in captured["prompt"]


def test_p1_grouped_candidate_cost_is_order_sensitive(world, slots):
    candidates = generate_align_candidates(world, slots)
    assert len(candidates) >= 2
    costs = [estimate_align_plan_cost(candidate, world, slots)[0] for candidate in candidates]
    spread = max(costs) - min(costs)
    assert spread >= 0.01, f"candidate cost spread is not meaningful: {costs}"


def test_p1_align_progress_uses_observed_pose(world, plan, slots):
    class RejectingVerifier:
        def check_at(self, object_id, target):
            return False

        def check_stable(self, object_id, include_velocity=False):
            return False

    progress = DEFAULT_REGISTRY.get("align").assess_progress(
        plan,
        RejectingVerifier(),
        slots,
        {"a"},
    )
    assert progress.valid is False
    assert "a" in progress.invalid_objects


def test_p1_align_recovery_repeats_pick_place_only(
    world, plan, slots, variant_config, tmp_path
):
    class MissFirstPlacePrimitives:
        def __init__(self):
            self.poses = {obj.id: obj.pose for obj in world.objects}
            self.held = None
            self.actions = []
            self.failed_object = None

        def execute(self, step, target, hints):
            self.actions.append((step.action, step.object))
            if step.action == "pick":
                x, y, _ = self.poses[step.object]
                self.poses[step.object] = (x, y, 0.98)
                self.held = step.object
            else:
                assert target is not None
                self.poses[step.object] = target
                self.held = None
                if self.failed_object is None:
                    self.failed_object = step.object
                    self.poses[step.object] = (0.70, 0.55, target[2])
            return PrimitiveResult(True)

        def object_pose(self, object_id):
            return self.poses[object_id]

        def all_object_poses(self):
            return dict(self.poses)

        def held_object_name(self):
            return self.held

        def settle_for_verification(self, steps):
            return None

    primitives = MissFirstPlacePrimitives()
    result = TaskRunner(
        plan=plan,
        world=world,
        slots=slots,
        hint_cache=HintCache(tmp_path / "history"),
        plugin_registry=DEFAULT_REGISTRY,
        event_log=EventLog(tmp_path / "events.csv", "align-recovery"),
        primitives=primitives,
        runtime_config=variant_config,
    ).run()

    assert result.success
    assert {action for action, _ in primitives.actions} == {"pick", "place"}
    failed = primitives.failed_object
    assert failed is not None
    failed_actions = [entry for entry in primitives.actions if entry[1] == failed]
    assert failed_actions[:4] == [
        ("pick", failed),
        ("place", failed),
        ("pick", failed),
        ("place", failed),
    ]


def test_p1_variant_uses_strict_lane_verification(variant_config):
    assert variant_config.verification.at_x_m <= 0.020
    assert variant_config.verification.at_y_m <= 0.020


def test_p1_variant_requires_first_solution_ompl_policy(variant_config):
    motion = variant_config.motion
    assert motion.ompl_required is True
    assert motion.planner == "RRTConnect"
    assert getattr(motion, "valid_state_sampler", None) == "obstacle_based"
    assert getattr(motion, "optimization_planner", None) == "BITstar"
    assert variant_config.ik.backend == "pinocchio"
    assert variant_config.ik.require_pinocchio is True
    assert variant_config.model.desired_tool_x == pytest.approx((1.0, 0.0, 0.0))
    assert variant_config.grasp.approach_clearance_m == pytest.approx(0.20)
    assert variant_config.grasp.pick_grip_sequence == pytest.approx(
        (0.030, 0.028, 0.026)
    )
