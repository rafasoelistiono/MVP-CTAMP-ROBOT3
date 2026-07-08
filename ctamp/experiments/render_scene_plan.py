"""Render a previously planned real-Panda scene without rerunning IK/search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..simulation import MuJoCoBackend, MuJoCoSceneBuilder, load_scene_config
from .run_scene import _render_video


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    config = load_scene_config(args.config)
    builder = MuJoCoSceneBuilder(config, args.config.resolve().parents[2])
    if builder.panda_asset.status != "real_panda_asset":
        raise RuntimeError("real Panda asset is required to render the joint plan")
    backend = MuJoCoBackend()
    backend.load_model(xml_string=builder.build_xml())
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    video = _render_video(backend, plan["actions"], args.output, args.fps, True)
    sys.stdout.write(f"{video}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
