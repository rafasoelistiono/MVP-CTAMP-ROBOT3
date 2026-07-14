"""Continuous stacking scenario for CTAMP v2."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from .run_scene_v2 import run as run_scene_v2

STACKING_STRATEGY = "continuous_single_viewer_stack_with_safe_zone_preview"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _object_sizes(config: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    default_size = tuple(float(v) for v in config["geometry"]["cube_size_xyz"])
    return {
        obj["id"]: tuple(float(v) for v in obj.get("size_xyz", default_size))
        for obj in config["objects"]
    }


def _stack_positions(
    order_bottom_to_top: list[str],
    sizes: dict[str, tuple[float, float, float]],
    xy: list[float],
    table_z: float,
) -> dict[str, list[float]]:
    z = table_z
    positions: dict[str, list[float]] = {}
    for object_id in order_bottom_to_top:
        height = sizes[object_id][2]
        positions[object_id] = [float(xy[0]), float(xy[1]), z + height / 2.0]
        z += height
    return positions


def _safe_zone_positions(
    order: list[str],
    sizes: dict[str, tuple[float, float, float]],
    origin: list[float],
    axis: str,
    spacing: float,
    table_z: float,
) -> dict[str, list[float]]:
    positions: dict[str, list[float]] = {}
    for index, object_id in enumerate(order):
        x, y = float(origin[0]), float(origin[1])
        if axis == "x":
            x += index * spacing
        elif axis == "y":
            y += index * spacing
        else:
            raise ValueError("safe_zone_axis must be x or y")
        positions[object_id] = [x, y, table_z + sizes[object_id][2] / 2.0]
    return positions


def _limited_orders(
    stack: dict[str, Any], max_objects: int | None
) -> tuple[list[str], list[str]]:
    safe_zone_order = list(
        stack.get(
            "safe_zone_order_right_first",
            stack["placeholder_order_small_to_large"],
        )
    )
    final_order = list(stack["final_order_bottom_to_top"])
    if max_objects is None:
        return safe_zone_order, final_order

    safe_zone_order = safe_zone_order[:max_objects]
    selected = set(safe_zone_order)
    return safe_zone_order, [
        object_id for object_id in final_order if object_id in selected
    ]


def _phase_config(
    base: dict[str, Any],
    phase_name: str,
    target_order: list[str],
    target_positions: dict[str, list[float]],
    object_poses: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["scene"]["scene_id"] = f"{base['scene']['scene_id']}_{phase_name}"
    config["task"]["target_objects"] = target_order
    config["grouped_tidy"]["slot_prefix"] = phase_name
    config["grouped_tidy"]["axis"] = "z"
    config["tidy_groups"] = [
        {
            "id": phase_name,
            "color": "mixed",
            "objects": target_order,
            "center": target_positions[target_order[0]],
            "positions": {
                object_id: target_positions[object_id] for object_id in target_order
            },
        }
    ]
    if object_poses is not None:
        for obj in config["objects"]:
            if obj["id"] in object_poses:
                obj["pose"] = object_poses[obj["id"]]
    return config


def build_phase_configs(
    config: dict[str, Any], max_objects: int | None = None
) -> tuple[dict, dict, dict]:
    stack = config["stacking_v2"]
    sizes = _object_sizes(config)
    safe_zone_order, final_order = _limited_orders(stack, max_objects)
    table_z = float(config["table"]["z_top"])
    safe_zone_positions = _safe_zone_positions(
        safe_zone_order,
        sizes,
        stack["safe_zone_origin"],
        stack.get("safe_zone_axis", "x"),
        float(stack["safe_zone_spacing"]),
        table_z,
    )
    final_positions = _stack_positions(
        final_order, sizes, stack["final_stack_xy"], table_z
    )
    phase1 = _phase_config(config, "safe_zone", safe_zone_order, safe_zone_positions)
    phase2 = _phase_config(config, "continuous_stack", final_order, final_positions)
    summary = {
        "largest_to_smallest_order": final_order,
        "safe_zone_order_right_first": safe_zone_order,
        "final_order_bottom_to_top": final_order,
        "safe_zone_positions": safe_zone_positions,
        "placeholder_positions": safe_zone_positions,
        "final_stack_positions": final_positions,
    }
    return phase1, phase2, summary


def _write_preview_configs(
    output: Path, safe_zone_config: dict, stack_config: dict
) -> Path:
    _write_yaml(output / "safe_zone_preview.yaml", safe_zone_config)
    stack_path = output / "continuous_stack.yaml"
    _write_yaml(stack_path, stack_config)
    return stack_path


def _dry_run_metrics(summary: dict) -> dict:
    return {
        "ctamp_version": "v2",
        "task": "stack",
        "strategy": STACKING_STRATEGY,
        "dry_run": True,
        **summary,
    }


def _run_metrics(continuous: dict, stack_config: dict, summary: dict) -> dict:
    return {
        "ctamp_version": "v2",
        "task": "stack",
        "strategy": STACKING_STRATEGY,
        "solution_found": continuous["solution_found"],
        "completed_objects": continuous["completed_objects"],
        "target_objects": len(stack_config["task"]["target_objects"]),
        "continuous_stack": continuous,
        **summary,
    }


def run(
    config_path: Path,
    output: Path,
    max_retries: int | None = None,
    max_objects: int | None = None,
    project_root: Path | None = None,
    viewer: bool = False,
    dry_run: bool = False,
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    safe_zone_config, stack_config, summary = build_phase_configs(
        config, max_objects=max_objects
    )
    stack_path = _write_preview_configs(output, safe_zone_config, stack_config)
    _write_json(output / "stacking_plan.json", summary)

    if dry_run:
        metrics = _dry_run_metrics(summary)
        _write_json(output / "metrics.json", metrics)
        return metrics

    continuous = run_scene_v2(
        stack_path,
        output / "continuous_stack",
        max_retries=max_retries,
        max_objects=max_objects,
        project_root=project_root,
        viewer=viewer,
    )
    metrics = _run_metrics(continuous, stack_config, summary)
    _write_json(output / "metrics.json", metrics)
    return metrics
