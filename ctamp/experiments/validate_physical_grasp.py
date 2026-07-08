"""Validate a saved Panda pick with actuator dynamics, finger contacts, and lift."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..simulation import (
    MuJoCoBackend,
    MuJoCoSceneBuilder,
    PandaIKSolver,
    PandaPhysicsExecutor,
    load_scene_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--object", default="a")
    args = parser.parse_args()

    config = load_scene_config(args.config)
    objects = {obj["id"]: obj for obj in config["objects"]}
    if args.object not in objects:
        parser.error(f"unknown object: {args.object}")
    builder = MuJoCoSceneBuilder(config, args.config.resolve().parents[2])
    xml = builder.build_xml()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    action = next(
        (item for item in plan["actions"] if item["object_id"] == args.object), None,
    )
    if action is None:
        parser.error(f"object {args.object!r} is absent from plan")

    # Compute close approach and lift targets on an independent kinematic data set.
    ik_backend = MuJoCoBackend()
    ik_backend.load_model(xml_string=xml)
    ik = PandaIKSolver(ik_backend)
    grasp_ready = np.asarray(action["transit_joint_waypoints"][0], dtype=float)
    ik.set_qpos(grasp_ready)
    obj = objects[args.object]
    physical_grasp = ik.plan_physical_grasp(
        args.object, tuple(obj["pose"]), start_qpos=grasp_ready,
    )
    if not physical_grasp.success:
        raise RuntimeError(f"physical grasp planning failed: {physical_grasp.reason}")
    approach_qpos = physical_grasp.joint_waypoints[-1]
    ik.set_qpos(approach_qpos)
    approach_rotation = ik.data.site_xmat[ik.site_id].reshape(3, 3).copy()
    lift_target = np.asarray(ik.site_position()) + np.array([0.0, 0.0, 0.14])
    lift = ik.solve(lift_target, seed=approach_qpos, orientation=approach_rotation)
    if not lift.success:
        lift = ik.solve(lift_target, seed=approach_qpos)
    if not lift.success:
        raise RuntimeError(f"lift IK failed: {lift.reason}")

    physics_backend = MuJoCoBackend()
    physics_backend.load_model(xml_string=xml)
    executor = PandaPhysicsExecutor(physics_backend)
    executor.initialize_arm(grasp_ready)
    executor.open_gripper(steps=100)
    tracked, tracking_error = executor.follow_joint_path(
        physical_grasp.joint_waypoints[1:-1],
    )
    if not tracked:
        raise RuntimeError(
            f"pre-grasp actuator tracking failed: {tracking_error}; "
            f"contacts={executor.ik.robot_collision_pairs()}"
        )
    result = executor.validate_grasp_and_lift(
        args.object, approach_qpos, lift.qpos,
        grasp_site_target=np.asarray(obj["pose"], dtype=float) + np.array([0.0, 0.0, 0.02]),
    )
    payload = {
        "object_id": args.object,
        "grasp_style": physical_grasp.grasp_style,
        **result.__dict__,
    }
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
