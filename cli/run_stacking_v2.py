from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ctamp.experiments.run_stacking_v2 import run as run_stacking_v2

from .common import exit_with_errors
from .run_simulation import ROOT_DIR

DEFAULT_CONFIG = ROOT_DIR / "configs/scenes/stacking_wall_world_v2.yaml"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CTAMP v2 continuous cube stacking."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-dir", default=ROOT_DIR / "runs", type=Path)
    parser.add_argument("--max-retries-per-object", type=int)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write phase configs without MuJoCo execution",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not args.config.exists():
        raise FileNotFoundError(f"scene config not found: {args.config}")
    output = (
        args.output
        or args.log_dir / f"stacking_wall_world_v2_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    metrics = run_stacking_v2(
        args.config,
        output,
        max_retries=args.max_retries_per_object,
        max_objects=args.max_objects,
        project_root=ROOT_DIR,
        viewer=bool(args.viewer),
        dry_run=args.dry_run,
    )
    sys.stdout.write(json.dumps(metrics, indent=2) + "\n")
    return 0 if metrics.get("solution_found", False) or args.dry_run else 2


def cli() -> None:
    exit_with_errors(main)


if __name__ == "__main__":
    cli()
