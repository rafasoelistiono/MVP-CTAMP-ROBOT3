"""Small optional MuJoCo runtime wrapper with no import-time dependency."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class MuJoCoBackend:
    def __init__(self) -> None:
        self.model = None
        self.data = None
        self._renderer = None
        self._renderer_size: tuple[int, int] | None = None

    @staticmethod
    def _mujoco():
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("MuJoCo backend requires `pip install mujoco`") from exc
        return mujoco

    def load_model(
        self, xml_path: str | Path | None = None, xml_string: str | None = None
    ) -> None:
        if (xml_path is None) == (xml_string is None):
            raise ValueError("provide exactly one of xml_path or xml_string")
        mujoco = self._mujoco()
        self.model = (
            mujoco.MjModel.from_xml_path(str(xml_path))
            if xml_path is not None
            else mujoco.MjModel.from_xml_string(xml_string)
        )
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._renderer = None
        self._renderer_size = None

    def reset(self) -> None:
        self._require_loaded()
        self._mujoco().mj_resetData(self.model, self.data)

    def step(self, n: int = 1) -> None:
        self._require_loaded()
        for _ in range(n):
            self._mujoco().mj_step(self.model, self.data)

    def get_body_pose(self, name: str) -> list[float]:
        self._require_loaded()
        body_id = self._body_id(name)
        return [*self.data.xpos[body_id].tolist(), *self.data.xquat[body_id].tolist()]

    def set_body_pose(self, name: str, pose: Sequence[float]) -> None:
        self._require_loaded()
        mujoco = self._mujoco()
        body_id = self._body_id(name)
        joint_id = int(self.model.body_jntadr[body_id])
        if joint_id < 0 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"body {name!r} does not have a free joint")
        qadr = int(self.model.jnt_qposadr[joint_id])
        values = list(pose)
        if len(values) not in (3, 7):
            raise ValueError("pose must be xyz or xyz+wxyz")
        self.data.qpos[qadr : qadr + 3] = values[:3]
        if len(values) == 7:
            self.data.qpos[qadr + 3 : qadr + 7] = values[3:]
        mujoco.mj_forward(self.model, self.data)

    def check_collision(self) -> bool:
        self._require_loaded()
        return bool(self.data.ncon)

    def render_offscreen(
        self,
        width: int = 640,
        height: int = 480,
        camera: str | int | None = None,
    ):
        self._require_loaded()
        try:
            if self._renderer is None or self._renderer_size != (width, height):
                if self._renderer is not None:
                    self._renderer.close()
                self._renderer = self._mujoco().Renderer(
                    self.model,
                    height=height,
                    width=width,
                )
                self._renderer_size = (width, height)
            self._renderer.update_scene(self.data, camera=camera)
            return self._renderer.render().copy()
        except Exception as exc:  # EGL/OSMesa availability is environment-specific
            raise RuntimeError(
                "offscreen rendering unavailable; try MUJOCO_GL=egl or osmesa"
            ) from exc

    def _body_id(self, name: str) -> int:
        body_id = self._mujoco().mj_name2id(
            self.model, self._mujoco().mjtObj.mjOBJ_BODY, name
        )
        if body_id < 0:
            raise KeyError(name)
        return int(body_id)

    def _require_loaded(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("no MuJoCo model loaded")
