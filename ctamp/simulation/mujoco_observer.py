"""Read stable named-body state from a MuJoCo backend."""

from __future__ import annotations

from .mujoco_backend import MuJoCoBackend


class MuJoCoObserver:
    def __init__(self, backend: MuJoCoBackend) -> None:
        self.backend = backend

    def object_poses(self, object_ids: list[str]) -> dict[str, list[float]]:
        return {oid: self.backend.get_body_pose(f"cube_{oid}") for oid in object_ids}

