from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

from configuration.defaults import ROOT_DIR
from scene import obstacle_mode_for_scene
from .naming import normalize_experiment_label, with_experiment_label


def write_summary_csv(
    task_name: str,
    scene_key: str,
    summary: dict,
    log_dir: str | Path = "logs",
) -> Path:
    out_dir = Path(log_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = str(summary.get("run_id") or time.strftime("%Y%m%d_%H%M%S"))
    experiment_label = normalize_experiment_label(
        summary.get("experiment_label")
    )
    output_stem = with_experiment_label(
        f"{task_name}_{scene_key}_{timestamp}",
        experiment_label,
    )
    out_path = out_dir / f"{output_stem}.csv"
    failed = list(summary.get("failed", []))
    objects_moved = int(summary.get("objects_moved") or 0)
    objects_total = int(summary.get("objects_total") or objects_moved + len(failed))
    success_rate = objects_moved / objects_total if objects_total else 0.0
    completion_percent = round(success_rate * 100.0, 2)
    overall_success = bool(summary.get("success"))
    benchmark_role = str(summary.get("benchmark_role") or "candidate")
    reference_100_percent = (
        benchmark_role == "reference"
        and overall_success
        and completion_percent == 100.0
    )
    obstacle_mode = summary.get("obstacle_mode") or obstacle_mode_for_scene(scene_key)
    scenario_type = summary.get("scenario_type") or "static"
    failure_counts = Counter(_failure_reason(item) for item in failed)
    row = {
        "task": task_name,
        "scene": scene_key,
        "run_id": with_experiment_label(timestamp, experiment_label),
        "experiment_label": experiment_label,
        "scenario_type": scenario_type,
        "obstacle_mode": obstacle_mode,
        "success": overall_success,
        "success_count": objects_moved,
        "failure_count": len(failed),
        "objects_moved": objects_moved,
        "objects_total": objects_total,
        "object_success_rate": round(success_rate, 4),
        "completion_percent": completion_percent,
        "plan_source": summary.get("plan_source", "unspecified"),
        "benchmark_role": benchmark_role,
        "benchmark_label": summary.get("benchmark_label", ""),
        "reference_100_percent": str(reference_100_percent).lower(),
        "failed_json": json.dumps(failed, ensure_ascii=False),
        "failure_reason_counts_json": json.dumps(
            dict(sorted(failure_counts.items())), ensure_ascii=False
        ),
        "duration_ms": summary.get("duration_ms"),
        "llm_used": str(bool(summary.get("llm_used", False))).lower(),
        "plan_file": summary.get("plan_file", ""),
        "runtime_profile": summary.get("runtime_profile", ""),
        "runtime_config_file": summary.get("runtime_config_file", ""),
        "run_manifest": summary.get("run_manifest", ""),
        "robust_align_candidate_count": summary.get("robust_align_candidate_count", ""),
        "robust_align_ranked_costs_json": json.dumps(
            summary.get("robust_align_ranked_costs", []), ensure_ascii=False
        ),
        "robust_align_selected_plan_id": summary.get("robust_align_selected_plan_id", ""),
        "robust_align_failed_before_success": summary.get(
            "robust_align_failed_before_success", ""
        ),
        "robust_align_probe_planning_time": summary.get(
            "robust_align_probe_planning_time", ""
        ),
        "robust_align_ik_failure_count": summary.get("robust_align_ik_failure_count", ""),
        "robust_align_ompl_failure_count": summary.get(
            "robust_align_ompl_failure_count", ""
        ),
        "robust_align_alignment_error": summary.get("robust_align_alignment_error", ""),
        "robust_align_spacing_error": summary.get("robust_align_spacing_error", ""),
        "task_variant": summary.get("task_variant", ""),
        "challenge_type": summary.get("challenge_type", ""),
        "num_objects": summary.get("num_objects", ""),
        "num_groups": summary.get("num_groups", ""),
        "num_obstacles": summary.get("num_obstacles", ""),
        "planner_name": summary.get("planner_name", ""),
        "collision_count": summary.get("collision_count", ""),
        "plan_steps": summary.get("plan_steps", ""),
        "executed_steps": summary.get("executed_steps", ""),
        "retry_count": summary.get("retry_count", ""),
        "replan_count": summary.get("replan_count", ""),
        "execution_time": summary.get("execution_time", ""),
        "alignment_error_mean": summary.get("alignment_error_mean", ""),
        "alignment_error_max": summary.get("alignment_error_max", ""),
        "spacing_error_mean": summary.get("spacing_error_mean", ""),
        "spacing_error_max": summary.get("spacing_error_max", ""),
        "selected_candidate_strategy": summary.get("selected_candidate_strategy", ""),
        "failed_candidate_count": summary.get("failed_candidate_count", ""),
        "motion_probe_failure_count": summary.get("motion_probe_failure_count", ""),
    }
    with out_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return out_path


def _failure_reason(item) -> str:
    if isinstance(item, dict):
        return str(item.get("failure_reason") or item.get("stage") or "unknown")
    return "unknown"
