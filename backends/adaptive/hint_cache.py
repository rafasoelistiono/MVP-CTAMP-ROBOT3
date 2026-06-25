from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionHints:
    ik_backend: str
    ik_position_tolerance: float
    grasp_profile: str


class HintCache:
    """
    Read historical events without changing goals, verifier tolerances, or safety.

    Hints are efficiency preferences only. The hard cap of 0.030 m prevents
    historical data from progressively weakening IK acceptance.
    """

    DEFAULT_IK_TOLERANCE = 0.020
    MAX_IK_TOLERANCE = 0.030

    def __init__(
        self,
        log_dir: str | Path,
        *,
        min_samples: int = 3,
        fallback_threshold: float = 0.70,
    ):
        self.log_dir = Path(log_dir)
        self.min_samples = int(min_samples)
        self.fallback_threshold = float(fallback_threshold)
        self._rows = self._load(self.log_dir)

    @staticmethod
    def reach_bucket(distance: float) -> str:
        if distance > 0.78:
            return "borderline"
        if distance > 0.70:
            return "far"
        return "normal"

    def hints_for(self, obj_id: str, cls: str, reach_distance: float) -> ExecutionHints:
        bucket = self.reach_bucket(reach_distance)
        return ExecutionHints(
            ik_backend=self.get_ik_backend(obj_id, bucket),
            ik_position_tolerance=self.get_ik_tolerance(obj_id, bucket),
            grasp_profile=self.get_grasp_profile(obj_id, cls),
        )

    def get_ik_backend(self, obj_id: str, reach_bucket: str) -> str:
        relevant = [
            row
            for row in self._rows
            if row.get("stage") in {"IK_SOLVE", "IK_CANDIDATE"}
            and self._row_matches_object(row, obj_id)
        ]
        fallback = [
            row
            for row in relevant
            if row.get("status") == "BACKEND_FALLBACK"
            or row.get("failure_reason") == "pinocchio_fk_validation_failed"
        ]
        candidate_count = sum(
            1 for row in relevant if row.get("stage") == "IK_CANDIDATE"
        )
        denominator = candidate_count or len(relevant)
        if denominator >= self.min_samples and len(fallback) / denominator >= self.fallback_threshold:
            return "mujoco_dls"
        return "pinocchio"

    def get_ik_tolerance(self, obj_id: str, reach_bucket: str) -> float:
        near_misses: list[float] = []
        for row in self._rows:
            if row.get("stage") != "IK_CANDIDATE":
                continue
            if row.get("failure_reason") != "ik_error_above_limit":
                continue
            if not self._row_matches_object(row, obj_id):
                continue
            try:
                error = float(row.get("pos_err", "nan"))
            except (TypeError, ValueError):
                continue
            if self.DEFAULT_IK_TOLERANCE < error <= self.MAX_IK_TOLERANCE:
                near_misses.append(error)
        if len(near_misses) < self.min_samples:
            return self.DEFAULT_IK_TOLERANCE
        near_misses.sort()
        median = near_misses[len(near_misses) // 2]
        return min(max(median, self.DEFAULT_IK_TOLERANCE), self.MAX_IK_TOLERANCE)

    def get_grasp_profile(self, obj_id: str, cls: str) -> str:
        scores: dict[str, list[bool]] = {}
        for row in self._rows:
            if row.get("stage") != "GRASP_PROFILE_RESULT":
                continue
            if row.get("object_id") != obj_id:
                continue
            profile = row.get("grasp_profile", "")
            if not profile:
                try:
                    profile = json.loads(row.get("extra_json", "{}")).get(
                        "grasp_profile", ""
                    )
                except (TypeError, json.JSONDecodeError):
                    profile = ""
            if profile:
                scores.setdefault(profile, []).append(row.get("status") == "OK")
        ranked = [
            (sum(outcomes) / len(outcomes), profile)
            for profile, outcomes in scores.items()
            if len(outcomes) >= self.min_samples
        ]
        if ranked:
            return max(ranked)[1]
        return "side_cylinder" if cls == "cylinder" else "default_cube"

    @staticmethod
    def _row_matches_object(row: dict[str, str], obj_id: str) -> bool:
        direct = row.get("object_id")
        if direct:
            return direct == obj_id
        return obj_id in row.get("phase", "")

    @staticmethod
    def _load(log_dir: Path) -> list[dict[str, str]]:
        if not log_dir.exists():
            return []
        rows: list[dict[str, str]] = []
        for path in sorted(log_dir.glob("*_events.csv")) + sorted(
            log_dir.glob("events_*.csv")
        ):
            try:
                with path.open(newline="", encoding="utf-8") as stream:
                    rows.extend(dict(row) for row in csv.DictReader(stream))
            except (OSError, csv.Error):
                continue
        return rows

