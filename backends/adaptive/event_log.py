from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


EVENT_FIELDS = (
    "timestamp",
    "run_id",
    "step_id",
    "stage",
    "status",
    "task",
    "scene_id",
    "object_id",
    "action",
    "attempt",
    "failure_reason",
    "duration_ms",
    "extra_json",
)


class EventLog:
    """Append-only structured events for the new task layer."""

    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        stage: str,
        status: str,
        *,
        step_id: int | None = None,
        task: str = "",
        scene_id: str = "",
        object_id: str = "",
        action: str = "",
        attempt: int | None = None,
        failure_reason: str | None = None,
        duration_ms: int | None = None,
        **extra: Any,
    ) -> None:
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": self.run_id,
            "step_id": "" if step_id is None else step_id,
            "stage": stage,
            "status": status,
            "task": task,
            "scene_id": scene_id,
            "object_id": object_id,
            "action": action,
            "attempt": "" if attempt is None else attempt,
            "failure_reason": failure_reason or "",
            "duration_ms": "" if duration_ms is None else duration_ms,
            "extra_json": json.dumps(extra, ensure_ascii=False, sort_keys=True),
        }
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

