"""CLI commands for CTAMP planner."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from ..benchmark.episode_runner import EpisodeRunner
from ..benchmark.plots import generate_plots
from ..learning.heuristic_models import OfflineSVRModel
from ..learning.sample_collector import SampleCollector
from ..planning.symbolic import PlanningProblem, SymbolicTaskPlanner
from ..tmm.builder import TMMGraphBuilder


def cmd_run(args: argparse.Namespace) -> int:
    """Run planning experiment from config."""
    import yaml

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    with open(config_path) as f:
        config = yaml.safe_load(f)

    num_episodes = config.get("num_episodes", 10)
    object_counts = config.get("object_counts", [1, 2, 3])
    seed = config.get("seed", 42)
    output_dir = config.get("output_dir", "results")

    runner = EpisodeRunner(
        num_episodes=num_episodes,
        object_counts=object_counts,
        seed=seed,
        output_dir=output_dir,
    )

    print(f"Running {num_episodes} episodes with objects={object_counts}...")
    results = runner.run()
    runner.save_csv(results)
    runner.save_json(results)
    generate_plots(results, output_dir)
    print(f"Results saved to {output_dir}/")
    return 0


def cmd_train_offline(args: argparse.Namespace) -> int:
    """Train offline model from samples."""
    samples_path = Path(args.samples)
    if not samples_path.exists():
        print(f"Error: samples file not found: {samples_path}", file=sys.stderr)
        return 1

    collector = SampleCollector()
    collector.load_npz(str(samples_path))
    samples = collector._samples
    print(f"Loaded {len(samples)} samples")

    X = np.array([s.features for s in samples])
    y = np.array([s.cost for s in samples])

    model = OfflineSVRModel()
    model.fit(X, y)
    print(f"Model fitted on {len(X)} samples")

    output_path = args.output or str(samples_path.with_suffix(".model.pkl"))
    model.save(output_path)
    print(f"Model saved to {output_path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate results from CSV."""
    csv_path = Path(args.results)
    if not csv_path.exists():
        print(f"Error: results file not found: {csv_path}", file=sys.stderr)
        return 1

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No results found in CSV")
        return 0

    by_type: dict[str, list[dict]] = {}
    for row in rows:
        pt = row.get("planner_type", "unknown")
        by_type.setdefault(pt, []).append(row)

    print(f"{'Planner':<12} {'Success':>8} {'Cost':>10} {'Expanded':>10} {'Time':>8}")
    print("-" * 52)
    for pt, metrics in by_type.items():
        success = sum(1 for m in metrics if m.get("success") == "True")
        costs = [float(m["cost"]) for m in metrics if m.get("success") == "True"]
        expanded = [
            int(m["nodes_expanded"]) for m in metrics if m.get("success") == "True"
        ]
        times = [float(m["time_elapsed"]) for m in metrics]

        avg_cost = sum(costs) / len(costs) if costs else 0
        avg_exp = sum(expanded) / len(expanded) if expanded else 0
        avg_time = sum(times) / len(times) if times else 0

        print(
            f"{pt:<12} {success:>4}/{len(metrics):<4} {avg_cost:>10.2f} {avg_exp:>10.0f} {avg_time:>8.3f}"
        )

    if args.output:
        summary = {}
        for pt, metrics in by_type.items():
            success = sum(1 for m in metrics if m.get("success") == "True")
            costs = [float(m["cost"]) for m in metrics if m.get("success") == "True"]
            summary[pt] = {
                "success_rate": success / len(metrics) if metrics else 0,
                "avg_cost": sum(costs) / len(costs) if costs else 0,
                "total_episodes": len(metrics),
            }
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"Summary saved to {args.output}")

    return 0


def cmd_visualize_tmm(args: argparse.Namespace) -> int:
    """Visualize TMM graph from problem config."""
    import yaml

    problem_path = Path(args.problem)
    if not problem_path.exists():
        print(f"Error: problem file not found: {problem_path}", file=sys.stderr)
        return 1

    with open(problem_path) as f:
        config = yaml.safe_load(f)

    objects = {}
    for oid, obj_data in config.get("objects", {}).items():
        from ..domain.models import ObjectState, Pose, Shape

        objects[oid] = ObjectState(
            object_id=oid,
            pose=Pose(x=obj_data.get("x", 0), y=obj_data.get("y", 0)),
            shape=Shape(type=obj_data.get("shape", "box")),
        )

    target_poses = {}
    for oid, pose_data in config.get("target_poses", {}).items():
        from ..domain.models import Pose

        target_poses[oid] = Pose(x=pose_data.get("x", 0), y=pose_data.get("y", 0))

    problem = PlanningProblem(objects=objects, target_poses=target_poses)
    planner = SymbolicTaskPlanner(problem)
    symbolic_graph = planner.solve()
    builder = TMMGraphBuilder()
    tmm_graph = builder.build(symbolic_graph)

    print(
        f"TMM graph: {tmm_graph._graph.number_of_nodes()} nodes, {tmm_graph._graph.number_of_edges()} edges"
    )

    if args.output:
        try:
            import matplotlib.pyplot as plt
            import networkx as nx

            fig, ax = plt.subplots(figsize=(12, 8))
            pos = nx.spring_layout(tmm_graph._graph, seed=42)
            nx.draw(
                tmm_graph._graph,
                pos,
                ax=ax,
                with_labels=True,
                node_size=200,
                font_size=8,
            )
            fig.tight_layout()
            fig.savefig(args.output, dpi=150)
            plt.close(fig)
            print(f"Graph saved to {args.output}")
        except ImportError:
            print("matplotlib not installed, skipping visualization")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ctamp",
        description="CTAMP - Combined Task and Motion Planning with learned heuristics",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser(
        "run", help="Run planning experiment from config"
    )
    run_parser.add_argument(
        "--config", required=True, help="Path to experiment config YAML"
    )
    run_parser.set_defaults(func=cmd_run)

    # train-offline command
    train_parser = subparsers.add_parser(
        "train-offline", help="Train offline model from samples"
    )
    train_parser.add_argument(
        "--samples", required=True, help="Path to samples NPZ file"
    )
    train_parser.add_argument("--output", help="Output model path")
    train_parser.set_defaults(func=cmd_train_offline)

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate results from CSV")
    eval_parser.add_argument("--results", required=True, help="Path to results CSV")
    eval_parser.add_argument("--output", help="Output summary JSON path")
    eval_parser.set_defaults(func=cmd_evaluate)

    # visualize-tmm command
    viz_parser = subparsers.add_parser("visualize-tmm", help="Visualize TMM graph")
    viz_parser.add_argument(
        "--problem", required=True, help="Path to problem config YAML"
    )
    viz_parser.add_argument("--output", help="Output image path")
    viz_parser.set_defaults(func=cmd_visualize_tmm)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
