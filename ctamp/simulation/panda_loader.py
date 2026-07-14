"""Locate an optional Franka Emika Panda MJCF asset tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PandaAsset:
    status: str
    path: Path | None


def find_panda_asset(project_root: str | Path) -> PandaAsset:
    root = Path(project_root)
    candidates = (
        root,
        root / "assets/mujoco/franka_panda",
        root / "assets/mujoco/franka_emika_panda",
        root / "third_party/mujoco_menagerie/franka_emika_panda",
    )
    for directory in candidates:
        if directory.is_dir() and (
            (directory / "panda.xml").is_file()
            or (directory / "models/panda.xml").is_file()
        ):
            return PandaAsset("real_panda_asset", directory)
    return PandaAsset("fallback_panda_proxy", None)
