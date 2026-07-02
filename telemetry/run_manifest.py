from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from configuration import RuntimeConfig


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_config_dict(config: RuntimeConfig) -> dict[str, Any]:
    return _json_safe(asdict(config))


def write_run_manifest(
    path: str | Path,
    *,
    run_id: str,
    config: RuntimeConfig,
    plan_file: str | Path,
    context_file: str | Path,
    scene_id: str,
    scene_variant: str,
    task: str,
    plugin_package: str,
    plan_source: str = "unspecified",
    benchmark_role: str = "candidate",
    benchmark_label: str = "",
    task_variant: str = "",
    challenge_type: str = "",
    num_objects: int | None = None,
    num_groups: int | None = None,
    num_obstacles: int | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plan_path = Path(plan_file).resolve()
    context_path = Path(context_file).resolve()
    payload = {
        "schema_version": "ctamp-run-manifest/v1",
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scene_id": scene_id,
        "scene_variant": scene_variant,
        "task": task,
        "plugin_package": plugin_package,
        "benchmark": {
            "plan_source": plan_source,
            "role": benchmark_role,
            "label": benchmark_label,
        },
        "challenge": {
            "task_variant": task_variant,
            "challenge_type": challenge_type,
            "num_objects": num_objects,
            "num_groups": num_groups,
            "num_obstacles": num_obstacles,
        },
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "context": {
            "path": str(context_path),
            "sha256": sha256_file(context_path),
        },
        "runtime_config": runtime_config_dict(config),
        "platform": {
            "python": sys.version.split()[0],
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
