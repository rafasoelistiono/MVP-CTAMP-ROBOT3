from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from ctamp.experiments.run_scene import run as run_scene_pipeline

from .run_simulation import ROOT_DIR, _materialize_context_config


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate migrated CTAMP final_plan.json via source planner pipeline."
    )
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--task", default="align")
    parser.add_argument(
        "--output", default=ROOT_DIR / "task_plans/generated", type=Path
    )
    parser.add_argument("--max-retries-per-object", type=int)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument(
        "--response-file", type=Path, help="Ignored legacy TaskPlan arg."
    )
    parser.add_argument("--experiment-label", default="")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ctamp_plan_") as temp_dir:
        config_path = _materialize_context_config(args.context, Path(temp_dir))
        metrics = run_scene_pipeline(
            config_path,
            args.output,
            max_retries=args.max_retries_per_object,
            max_objects=args.max_objects,
            project_root=ROOT_DIR,
        )
    plan_path = args.output / "final_plan.json"
    print(json.dumps({"plan": str(plan_path), "success": metrics["solution_found"]}))
    return 0 if metrics["solution_found"] else 2


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
