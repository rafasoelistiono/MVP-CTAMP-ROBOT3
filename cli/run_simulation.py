from __future__ import annotations

import argparse
import json
import sys
import time
import tempfile
from pathlib import Path

import yaml

from ctamp.experiments.run_scene import run as run_scene_pipeline
from world.builder import build_world_state

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "configs/scenes/align_grouped_tidy_wall_world.yaml"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run migrated CTAMP scene pipeline without TaskPlan/OMPL/cache gates."
    )
    parser.add_argument("--config", type=Path, help="Scene YAML. Overrides --context.")
    parser.add_argument("--context", type=Path, help="CONTEXT.MD-style scene context.")
    parser.add_argument("--output", type=Path, help="Run artifact directory.")
    parser.add_argument("--log-dir", default=ROOT_DIR / "runs", type=Path)
    parser.add_argument("--max-retries-per-object", type=int)

    parser.add_argument("--max-objects", type=int)

    # Legacy args accepted so old commands keep invoking the replacement pipeline.
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--scene")
    parser.add_argument("--runtime-profile")
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plan-source")
    parser.add_argument("--benchmark-role")
    parser.add_argument("--benchmark-label")
    parser.add_argument("--experiment-label")
    parser.add_argument("--robust-align", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-adaptive-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--motion-planner", default="mujoco")
    parser.add_argument("--robot", default="panda_left")
    parser.add_argument("--learning-mode", default="online")
    return parser.parse_args()


def _context_config(context_path: Path) -> dict:
    world = build_world_state(context_path)
    if world.grouped_tidy is None:
        raise ValueError("migrated pipeline requires grouped_tidy context")
    return {
        "scene": {"scene_id": world.scene_id, "variant": world.variant},
        "table": {
            "x_range": list(world.table_x_range),
            "y_range": list(world.table_y_range),
            "z_top": world.table_z_top,
            "goal_center": list(world.goal_center),
            "goal_area_size_xy": list(world.goal_area_size_xy),
        },
        "geometry": {"cube_size_xyz": [0.066, 0.066, 0.066]},
        "robot": {
            "id": world.robot_id,
            "reach_min_xy": world.robot_reach_min,
            "reach_max_xy": world.robot_reach_max,
            "base_xy": list(world.robot_base_xy),
            "base_z": world.robot_base_z,
            "capabilities": list(world.robot_capabilities),
        },
        "objects": [
            {
                "id": obj.id,
                "class": obj.cls,
                "color": obj.color,
                "pose": list(obj.pose),
                "reachable": obj.reachable,
                "near_obstacle": obj.near_obstacle,
            }
            for obj in world.objects
        ],
        "obstacles": [
            {
                "id": obstacle.id,
                "kind": obstacle.kind,
                "pose": list(obstacle.pose),
                "fragile": obstacle.fragile,
                "radius": obstacle.radius,
                "height": obstacle.height,
                "size": list(obstacle.size or (obstacle.radius * 2, obstacle.radius * 2, 1.6)),
            }
            for obstacle in world.obstacles
        ],
        "task": {
            "name": world.task_name,
            "target_objects": list(world.target_objects),
            "description": world.task_description,
        },
        "constraints": {
            "preserve_obstacles": world.preserve_obstacles,
            "max_retries_per_object": world.max_retries_per_object,
            "allowed_predicates": list(world.allowed_predicates),
        },
        "grouped_tidy": {
            "enabled": world.grouped_tidy.enabled,
            "require_ordered": world.grouped_tidy.require_ordered,
            "slot_prefix": world.grouped_tidy.slot_prefix,
            "axis": world.grouped_tidy.axis,
            "spacing": world.grouped_tidy.spacing,
            "row_spacing": world.grouped_tidy.row_spacing,
        },
        "tidy_groups": [
            {
                "id": group.id,
                "color": group.color,
                "objects": list(group.objects),
                "center": list(group.center),
            }
            for group in world.grouped_tidy.groups
        ],
        "physical_execution": {
            "enabled": True,
            "prioritize_side_access": True,
            "require_force_closure": False,
            "completion_policy": "best_effort",
            "minimum_completion_ratio": 0.80,
            "allow_adaptive_grasp_orientation": True,
            "cube_mass": 0.10,
            "cube_friction": [2.0, 1.0, 0.5],
            "table_friction": [1.0, 0.01, 0.001],
            "grip_target_width": 0.052,
        },
        "challenge": None if world.challenge is None else {
            "type": world.challenge.type,
            "enabled": world.challenge.enabled,
            "obstacle_ids": list(world.challenge.obstacle_ids),
            "require_obstacle_aware_slots": world.challenge.require_obstacle_aware_slots,
            "require_motion_probe": world.challenge.require_motion_probe,
            "inflated_clearance_required": world.challenge.inflated_clearance_required,
            "wall_blocks_direct_path": world.challenge.wall_blocks_direct_path,
            "side_corridors_required": world.challenge.side_corridors_required,
        },
    }


def _materialize_context_config(context_path: Path, config_dir: Path) -> Path:
    config_path = config_dir / f"{context_path.stem}_from_context.yaml"
    config_path.write_text(
        yaml.safe_dump(_context_config(context_path), sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _run_config(config_path: Path, args: argparse.Namespace) -> int:
    if not config_path.exists():
        raise FileNotFoundError(f"scene config not found: {config_path}")
    scene_id = yaml.safe_load(config_path.read_text(encoding="utf-8"))["scene"]["scene_id"]
    output = args.output or args.log_dir / f"{scene_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    metrics = run_scene_pipeline(
        config_path,
        output,
        max_retries=args.max_retries_per_object,
        max_objects=args.max_objects,
        project_root=ROOT_DIR,
        viewer=bool(args.viewer),
    )
    sys.stdout.write(json.dumps(metrics, indent=2) + "\n")
    return 0 if metrics["solution_found"] else 2


def main() -> int:
    args = _arguments()
    config_path = args.config
    if config_path is not None:
        return _run_config(config_path, args)
    if args.context:
        with tempfile.TemporaryDirectory(prefix="ctamp_context_") as temp_dir:
            return _run_config(_materialize_context_config(args.context, Path(temp_dir)), args)
    return _run_config(DEFAULT_CONFIG, args)


def cli() -> None:
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
