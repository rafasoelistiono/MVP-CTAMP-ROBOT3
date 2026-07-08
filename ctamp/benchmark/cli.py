"""CLI entry point for running benchmark episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .episode_runner import EpisodeRunner
from .plots import generate_plots


def main() -> int:
    parser = argparse.ArgumentParser(description="CTAMP benchmark runner")
    parser.add_argument("-n", "--num-episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("-o", "--objects", type=str, default="1,2,3,4,5", help="Comma-separated object counts")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-d", "--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    object_counts = [int(x) for x in args.objects.split(",")]

    runner = EpisodeRunner(
        num_episodes=args.num_episodes,
        object_counts=object_counts,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"Running {args.num_episodes} episodes with objects={object_counts}...")
    results = runner.run()

    runner.save_csv(results)
    runner.save_json(results)
    print(f"Results saved to {args.output_dir}/")

    if not args.no_plots:
        generate_plots(results, args.output_dir)
        print(f"Plots saved to {args.output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
