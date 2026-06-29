"""Generate benchmark figures from telemetry CSV files.

The newest run for every task/model pair is selected.  Execution telemetry is
kept separate from symbolic-plan quality so a summary counter cannot hide an
invalid place target or a recovery action.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_MODELS: tuple[tuple[str, str], ...] = (
    ("deepseekv4flash", "DeepSeek\nV4 Flash"),
    ("qwen3coder", "Qwen3\nCoder"),
    ("minimaxm3", "MiniMax\nM3"),
    ("gptoss", "GPT-OSS"),
    ("sonnet46", "Sonnet\n4.6"),
    ("gpt55", "GPT-5.5"),
)
TASK_MODELS: dict[str, tuple[tuple[str, str], ...]] = {
    "stack": BASE_MODELS,
    "pyramid": (
        ("deepseekv4flash", "DeepSeek\nV4 Flash"),
        ("qwen3coder", "Qwen3\nCoder"),
        ("minimaxm27", "MiniMax\nM2.7"),
        ("gptoss", "GPT-OSS"),
        ("sonnet46", "Sonnet\n4.6"),
        ("gpt55", "GPT-5.5"),
    ),
}

MODEL_COLORS = ("#2563EB", "#7C3AED", "#EA580C", "#0F766E", "#9333EA", "#0891B2")
EXECUTION_COLORS = ("#94A3B8", "#2563EB")


@dataclass(frozen=True)
class RunMetrics:
    model_id: str
    model_name: str
    run_id: str
    target_count: int
    completion_percent: float
    verified_placement_percent: float
    final_goal_percent: float
    planned_step_success_percent: float
    plan_structure_percent: float
    goal_predicate_coverage_percent: float
    duration_s: float
    failed_attempts: int
    missing_place_target_attempts: int
    stack_rebuilds: int
    geometry_errors_mm: dict[str, float]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _canonical_predicate(predicate: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return predicate["name"], tuple(str(arg) for arg in predicate.get("args", []))


def _step_signature(step: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(step.get("action", "")),
        str(step.get("object", "")),
        str(step.get("slot", "")),
        str(step.get("on_top_of", "")),
    )


def _plan_structure_score(plan: dict[str, Any], reference: dict[str, Any]) -> float:
    plan_steps = plan.get("steps", [])
    reference_steps = reference.get("steps", [])
    denominator = max(len(plan_steps), len(reference_steps), 1)
    matches = sum(
        _step_signature(candidate) == _step_signature(expected)
        for candidate, expected in zip(plan_steps, reference_steps)
    )
    return 100.0 * matches / denominator


def _goal_match_score(plan: dict[str, Any], reference: dict[str, Any]) -> float:
    expected = {
        _canonical_predicate(predicate)
        for predicate in reference.get("goal_predicates", [])
    }
    actual = {
        _canonical_predicate(predicate)
        for predicate in plan.get("goal_predicates", [])
    }
    return 100.0 * len(expected & actual) / max(len(expected | actual), 1)


def _geometry_errors_mm(
    task: str, plan: dict[str, Any], reference: dict[str, Any]
) -> dict[str, float]:
    actual = plan["slot_config"]
    expected = reference["slot_config"]
    if task == "pyramid":
        return {
            "spacing": 1000.0 * abs(float(actual["spacing_m"]) - float(expected["spacing_m"])),
            "base Z": 1000.0 * abs(float(actual["base_z"]) - float(expected["base_z"])),
        }
    return {
        "base Z": 1000.0 * abs(float(actual["base_z"]) - float(expected["base_z"])),
        "layer height": 1000.0
        * abs(float(actual["layer_height_m"]) - float(expected["layer_height_m"])),
    }


def _latest_summary(logs_dir: Path, task: str, model_id: str) -> Path:
    candidates = sorted(
        logs_dir.glob(f"task_plan_{task}_ungroup_obs_*_{model_id}.csv")
    )
    if not candidates:
        raise FileNotFoundError(f"No {task} summary found for {model_id} in {logs_dir}")
    return candidates[-1]


def _resolve_plan(repo_root: Path, summary: dict[str, str], task: str, model_id: str) -> Path:
    recorded = Path(summary.get("plan_file", ""))
    candidates = (
        recorded if recorded.is_absolute() else repo_root / recorded,
        repo_root
        / "task_plans"
        / "generated"
        / f"ungroup_obs_{task}_cubes_{task}_{model_id}.json",
        repo_root
        / "task_plans"
        / "examples"
        / f"ungroup_obs_{task}_cubes_{model_id}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"TaskPlan for {task}/{model_id} was not found")


def collect_task_metrics(repo_root: Path, task: str) -> list[RunMetrics]:
    logs_dir = repo_root / "logs"
    reference = _load_json(
        repo_root / "task_plans" / "examples" / f"ungroup_obs_{task}_cubes.json"
    )
    results: list[RunMetrics] = []

    for model_id, model_name in TASK_MODELS[task]:
        summary_path = _latest_summary(logs_dir, task, model_id)
        summary_rows = _read_csv(summary_path)
        if len(summary_rows) != 1:
            raise ValueError(f"Expected one summary row in {summary_path}")
        summary = summary_rows[0]
        run_id = summary["run_id"]
        event_path = logs_dir / f"events_{task}_ungroup_obs_{run_id}.csv"
        if not event_path.is_file():
            raise FileNotFoundError(f"Event log declared by {summary_path} was not found: {event_path}")

        plan = _load_json(_resolve_plan(repo_root, summary, task, model_id))
        events = _read_csv(event_path)
        step_events = [event for event in events if event["stage"] == "STEP"]
        planned_ids = {str(step["step_id"]) for step in plan.get("steps", [])}
        successful_planned_ids = {
            event["step_id"]
            for event in step_events
            if event["status"] == "OK" and event["step_id"] in planned_ids
        }
        verified_objects = {
            event["object_id"]
            for event in step_events
            if event["status"] == "OK"
            and event["action"] in {"place", "stack_place"}
            and event["object_id"]
        }
        failed_attempts = sum(event["status"] == "FAILED" for event in step_events)
        missing_place_target_attempts = sum(
            event["failure_reason"] == "missing_place_target" for event in step_events
        )
        rebuilds = sum(
            event["stage"] == "STACK_REBUILD" and event["status"] == "START"
            for event in events
        )
        target_count = max(len(plan.get("target_objects", [])), 1)

        results.append(
            RunMetrics(
                model_id=model_id,
                model_name=model_name,
                run_id=run_id,
                target_count=target_count,
                completion_percent=float(summary["completion_percent"]),
                verified_placement_percent=100.0 * len(verified_objects) / target_count,
                final_goal_percent=100.0 if _as_bool(summary["success"]) else 0.0,
                planned_step_success_percent=100.0
                * len(successful_planned_ids)
                / max(len(planned_ids), 1),
                plan_structure_percent=_plan_structure_score(plan, reference),
                goal_predicate_coverage_percent=_goal_match_score(plan, reference),
                duration_s=float(summary["duration_ms"]) / 1000.0,
                failed_attempts=failed_attempts,
                missing_place_target_attempts=missing_place_target_attempts,
                stack_rebuilds=rebuilds,
                geometry_errors_mm=_geometry_errors_mm(task, plan, reference),
            )
        )
    return results


def _label_bars(axis: Any, bars: Any, suffix: str = "%") -> None:
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            f"{value:.0f}{suffix}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def plot_task(metrics: list[RunMetrics], task: str, output_path: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ctamp-matplotlib")
    )
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment-dependent message
        raise SystemExit('Install visualization dependencies with: pip install -e ".[viz]"') from exc

    names = [item.model_name for item in metrics]
    x = np.arange(len(metrics))
    figure, axes = plt.subplots(2, 2, figsize=(18, 9))
    figure.subplots_adjust(
        left=0.13,
        right=0.93,
        bottom=0.14,
        top=0.90,
        wspace=0.34,
        hspace=0.38,
    )
    figure.suptitle(
        f"{task.upper()} benchmark — {len(metrics)} LLM TaskPlans",
        fontsize=20,
        fontweight="bold",
    )

    # Execution progress: the summary counter and physically verified placements.
    axis = axes[0, 0]
    width = 0.34
    completion = [item.completion_percent for item in metrics]
    verified = [item.verified_placement_percent for item in metrics]
    bars_a = axis.bar(x - width / 2, completion, width, color=EXECUTION_COLORS[0], label="Summary completion")
    bars_b = axis.bar(x + width / 2, verified, width, color=EXECUTION_COLORS[1], label="Verified placements")
    axis.axhline(100, color="#16A34A", linestyle="--", linewidth=1.4, label="Goal target")
    axis.set_title("Execution progress")
    axis.set_ylabel(f"Percent of {metrics[0].target_count} target cubes")
    axis.set_ylim(0, 118)
    axis.set_xticks(x, names)
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    _label_bars(axis, bars_a)
    _label_bars(axis, bars_b)

    # Runtime and failed attempts share an x-axis but retain their native units.
    axis = axes[0, 1]
    durations = [item.duration_s for item in metrics]
    bars = axis.bar(x, durations, color=MODEL_COLORS, width=0.62)
    axis.set_title("Runtime and failed STEP attempts")
    axis.set_ylabel("Runtime (seconds)")
    axis.set_xticks(x, names)
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, durations):
        axis.annotate(
            f"{value:.1f}s",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
    failures_axis = axis.twinx()
    failures = [item.failed_attempts for item in metrics]
    failures_axis.plot(x, failures, color="#DC2626", marker="o", linewidth=2.2, label="Failed attempts")
    failures_axis.set_ylabel("Failed STEP attempts", color="#DC2626")
    failures_axis.tick_params(axis="y", colors="#DC2626")
    failures_axis.set_ylim(0, max(failures + [1]) * 1.35)
    for position, value in zip(x, failures):
        failures_axis.annotate(
            str(value),
            (position, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color="#B91C1C",
            fontweight="bold",
        )

    # Percent metrics allow direct comparison of final goal, event execution,
    # symbolic action structure, and goal predicate coverage.
    axis = axes[1, 0]
    heatmap_rows = (
        ("Final goal reached", [item.final_goal_percent for item in metrics]),
        ("Planned steps OK", [item.planned_step_success_percent for item in metrics]),
        ("Plan structure match", [item.plan_structure_percent for item in metrics]),
        ("Goal predicate match", [item.goal_predicate_coverage_percent for item in metrics]),
    )
    matrix = np.array([values for _, values in heatmap_rows])
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    axis.set_title("Goal state and TaskPlan fidelity")
    axis.set_xticks(x, names)
    axis.set_yticks(np.arange(len(heatmap_rows)), [label for label, _ in heatmap_rows])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.0f}%",
                ha="center",
                va="center",
                color="white" if value < 35 else "#111827",
                fontweight="bold",
            )
    figure.colorbar(image, ax=axis, shrink=0.75, label="Score (%)")

    # Only fields with explanatory value are plotted.  Other slot fields are
    # identical to the reference and are documented in README.
    axis = axes[1, 1]
    geometry_labels = list(metrics[0].geometry_errors_mm)
    geometry_width = min(0.8 / max(len(geometry_labels), 1), 0.28)
    offsets = [
        (index - (len(geometry_labels) - 1) / 2) * geometry_width
        for index in range(len(geometry_labels))
    ]
    geometry_colors = ("#F59E0B", "#DB2777", "#10B981")
    for index, label in enumerate(geometry_labels):
        values = [item.geometry_errors_mm[label] for item in metrics]
        bars = axis.bar(
            x + offsets[index],
            values,
            geometry_width,
            color=geometry_colors[index % len(geometry_colors)],
            label=f"|Δ {label}|",
        )
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.set_title("Slot geometry deviation from reference plan")
    axis.set_ylabel("Absolute deviation (mm)")
    axis.set_xticks(x, names)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    maximum_geometry = max(
        value for item in metrics for value in item.geometry_errors_mm.values()
    )
    axis.set_ylim(0, max(5.0, maximum_geometry * 1.28))

    if task == "stack":
        rebuilds = ", ".join(
            f"{item.model_name.replace(chr(10), ' ')}={item.stack_rebuilds}"
            for item in metrics
        )
        note = f"STACK_REBUILD count: {rebuilds}"
    else:
        verified = ", ".join(
            f"{item.model_name.replace(chr(10), ' ')}={item.verified_placement_percent:.0f}%"
            for item in metrics
        )
        note = f"verified pyramid placements: {verified}"
    figure.text(
        0.5,
        0.008,
        "Latest run per LLM • reference: task_plans/examples/ungroup_obs_"
        f"{task}_cubes.json\n{note}",
        ha="center",
        fontsize=9,
        color="#475569",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _print_summary(task: str, metrics: list[RunMetrics]) -> None:
    print(f"{task.upper()} (latest run per LLM)")
    for item in metrics:
        print(
            f"  {item.model_name.replace(chr(10), ' '):18} "
            f"run={item.run_id} final={item.final_goal_percent:.0f}% "
            f"verified={item.verified_placement_percent:.0f}% "
            f"runtime={item.duration_s:.3f}s failures={item.failed_attempts}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing logs/ and task_plans/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="PNG output directory (default: <repo>/docs/images)",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir or repo_root / "docs" / "images").resolve()

    for task in ("stack", "pyramid"):
        try:
            metrics = collect_task_metrics(repo_root, task)
        except FileNotFoundError as exc:
            print(f"SKIP {task.upper()}: {exc}")
            continue
        output_path = output_dir / f"llm_{task}_benchmark.png"
        plot_task(metrics, task, output_path)
        _print_summary(task, metrics)
        print(f"  wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
