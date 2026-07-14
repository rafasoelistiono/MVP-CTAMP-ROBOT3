"""Generate benchmark plots."""

from __future__ import annotations

from importlib.util import find_spec
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np

from .episode_runner import EpisodeMetrics, EpisodeResult


def generate_plots(results: List[EpisodeResult], output_dir: str) -> None:
    """Generate benchmark plots for expanded vertices, runtime, and solution cost ratio."""
    if find_spec("matplotlib.pyplot") is None:
        print("matplotlib not installed, skipping plots")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_objects: dict[int, dict[str, list[EpisodeMetrics]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in results:
        for planner_type, metrics in result.metrics.items():
            by_objects[result.num_objects][planner_type].append(metrics)

    object_counts = sorted(by_objects.keys())
    planner_types = ["baseline", "offline", "online"]

    _plot_expanded(object_counts, by_objects, planner_types, out)
    _plot_runtime(object_counts, by_objects, planner_types, out)
    _plot_cost_ratio(object_counts, by_objects, planner_types, out)


def _plot_expanded(
    object_counts: list[int],
    by_objects: dict[int, dict[str, list[EpisodeMetrics]]],
    planner_types: list[str],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(object_counts))
    width = 0.25

    for i, pt in enumerate(planner_types):
        vals = []
        for oc in object_counts:
            metrics = by_objects[oc].get(pt, [])
            successful = [m for m in metrics if m.success]
            vals.append(
                np.mean([m.nodes_expanded for m in successful]) if successful else 0
            )
        ax.bar(x + i * width, vals, width, label=pt)

    ax.set_xlabel("Number of Objects")
    ax.set_ylabel("Expanded Vertices")
    ax.set_title("Vertices Expanded by Problem Size")
    ax.set_xticks(x + width)
    ax.set_xticklabels(object_counts)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "expanded_vertices.png", dpi=150)
    plt.close(fig)


def _plot_runtime(
    object_counts: list[int],
    by_objects: dict[int, dict[str, list[EpisodeMetrics]]],
    planner_types: list[str],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(object_counts))
    width = 0.25

    for i, pt in enumerate(planner_types):
        vals = []
        for oc in object_counts:
            metrics = by_objects[oc].get(pt, [])
            vals.append(np.mean([m.time_elapsed for m in metrics]) if metrics else 0)
        ax.bar(x + i * width, vals, width, label=pt)

    ax.set_xlabel("Number of Objects")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Runtime by Problem Size")
    ax.set_xticks(x + width)
    ax.set_xticklabels(object_counts)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "runtime.png", dpi=150)
    plt.close(fig)


def _plot_cost_ratio(
    object_counts: list[int],
    by_objects: dict[int, dict[str, list[EpisodeMetrics]]],
    planner_types: list[str],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(object_counts))
    width = 0.25

    for i, pt in enumerate(planner_types):
        if pt == "baseline":
            continue
        ratios = []
        for oc in object_counts:
            learned = [m for m in by_objects[oc].get(pt, []) if m.success]
            baseline = [m for m in by_objects[oc].get("baseline", []) if m.success]
            if learned and baseline:
                ratio = np.mean([m.cost for m in learned]) / np.mean(
                    [m.cost for m in baseline]
                )
                ratios.append(ratio)
            else:
                ratios.append(np.nan)
        ax.bar(x + i * width, ratios, width, label=f"{pt}/baseline")

    ax.set_xlabel("Number of Objects")
    ax.set_ylabel("Cost Ratio")
    ax.set_title("Solution Cost Ratio vs Baseline")
    ax.set_xticks(x + width)
    ax.set_xticklabels(object_counts)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cost_ratio.png", dpi=150)
    plt.close(fig)
