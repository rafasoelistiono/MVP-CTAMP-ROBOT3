"""Replay a saved real-Panda joint plan in the interactive MuJoCo viewer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..simulation import (
    MuJoCoBackend,
    MuJoCoSceneBuilder,
    PandaPhysicsExecutor,
    load_scene_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--interpolation-steps", type=int, default=4)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    if args.fps <= 0 or args.speed <= 0:
        parser.error("--fps and --speed must be positive")
    if args.interpolation_steps < 1:
        parser.error("--interpolation-steps must be at least 1")

    config = load_scene_config(args.config)
    builder = MuJoCoSceneBuilder(
        config, args.project_root or args.config.resolve().parents[2],
    )
    if builder.panda_asset.status != "real_panda_asset":
        raise RuntimeError("real Panda asset is required for joint-plan replay")
    backend = MuJoCoBackend()
    backend.load_model(xml_string=builder.build_xml())
    actions = json.loads(args.plan.read_text(encoding="utf-8"))["actions"]
    objects = {obj["id"]: obj for obj in config["objects"]}
    safe_q = np.asarray(config["robot"].get("physical_start_qpos"), dtype=float)
    if safe_q.shape != (7,):
        raise RuntimeError("robot.physical_start_qpos must contain seven joint values")

    import mujoco.viewer

    frame_time = 1.0 / (args.fps * args.speed)
    with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
        viewer.cam.lookat[:] = [0.0, -0.10, 1.05]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -35.0
        while viewer.is_running():
            backend.reset()
            executor = PandaPhysicsExecutor(
                backend, viewer=viewer, realtime_factor=args.speed,
            )
            executor.initialize_arm(safe_q)
            for action in actions:
                object_id = action["object_id"]
                if object_id not in objects:
                    continue
                transit = action.get("transit_joint_waypoints", [])
                transfer = action.get("transfer_joint_waypoints", [])
                if len(transit) < 2 or len(transfer) < 1:
                    continue
                executor.open_gripper(steps=120)
                tracked, _ = executor.follow_joint_path(
                    transit[1:-1], max_joint_step=0.025,
                )
                if not tracked:
                    continue
                grasp = executor.validate_grasp_and_lift(
                    object_id,
                    transit[-1],
                    transfer[0],
                    grasp_site_target=np.asarray(objects[object_id]["pose"]) + np.array([0.0, 0.0, 0.02]),
                    grip_width=float(action.get("grasp_width", 0.052)),
                )
                if not grasp.success:
                    executor.open_gripper(steps=120)
                    continue
                tracked, _ = executor.follow_joint_path(
                    transfer[1:], max_joint_step=0.025,
                )
                executor.set_carry_constraint(object_id, False)
                executor.open_gripper(steps=220)
                if len(transfer) >= 2:
                    executor.follow_joint_path([transfer[-2]], max_joint_step=0.025)
                executor.settle(steps=120)
                if not tracked:
                    continue
                started = time.perf_counter()
                while time.perf_counter() - started < frame_time * args.fps:
                    if not viewer.is_running():
                        return 0
                    viewer.sync()
                    time.sleep(0.02)
            if not args.loop:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
