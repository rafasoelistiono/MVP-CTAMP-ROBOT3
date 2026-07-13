from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

from ctamp.experiments.run_scene_v2 import run as run_scene_pipeline_v2

from .run_simulation import DEFAULT_CONFIG, ROOT_DIR, _arguments, _materialize_context_config


def _run_config(config_path: Path, args) -> int:
    import yaml

    if not config_path.exists():
        raise FileNotFoundError(f"scene config not found: {config_path}")
    scene_id = yaml.safe_load(config_path.read_text(encoding="utf-8"))["scene"]["scene_id"]
    output = args.output or args.log_dir / f"{scene_id}_v2_{time.strftime('%Y%m%d_%H%M%S')}"
    metrics = run_scene_pipeline_v2(
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
    if args.config is not None:
        return _run_config(args.config, args)
    if args.context:
        with tempfile.TemporaryDirectory(prefix="ctamp_context_v2_") as temp_dir:
            return _run_config(_materialize_context_config(args.context, Path(temp_dir)), args)
    return _run_config(DEFAULT_CONFIG, args)


def cli() -> None:
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
