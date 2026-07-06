from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .types import ModelConfig, RuntimeConfig


ROOT_DIR = Path(__file__).resolve().parents[1]


class RuntimeProfileRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, RuntimeConfig] = {}

    def register(self, profile: RuntimeConfig, *, replace_existing: bool = False) -> None:
        if profile.name in self._profiles and not replace_existing:
            raise ValueError(f"runtime profile {profile.name!r} is already registered")
        self._profiles[profile.name] = profile.validate()

    def get(self, name: str) -> RuntimeConfig:
        try:
            return self._profiles[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown runtime profile {name!r}; available: {sorted(self._profiles)}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))


PANDA_MODEL = ModelConfig(name="franka_panda", xml_path=ROOT_DIR / "models" / "panda.xml")
CONSERVATIVE_PROFILE = RuntimeConfig(name="conservative", model=PANDA_MODEL)
OBSTACLE_PROFILE = replace(
    CONSERVATIVE_PROFILE,
    name="obstacle",
    model=replace(
        CONSERVATIVE_PROFILE.model,
        wall_right_q=(
            0.305267,
            0.563609,
            -0.024401,
            -1.841814,
            0.019420,
            2.405206,
            1.055646,
        ),
    ),
    ik=replace(CONSERVATIVE_PROFILE.ik, max_valid_candidates=8),
    motion=replace(
        CONSERVATIVE_PROFILE.motion,
        planner="RRTConnect",
        time_limit_s=12.0,
        sampler_range=0.03,
        valid_state_sampler="gaussian",
        probe_time_limit_s=10.0,
        probe_segment_time_limit_s=3.0,
    ),
    safety=replace(
        CONSERVATIVE_PROFILE.safety,
        min_pick_obstacle_clearance_m=0.10,
        cautious_obstacle_clearance_m=0.22,
    ),
)

DEFAULT_PROFILE_REGISTRY = RuntimeProfileRegistry()
DEFAULT_PROFILE_REGISTRY.register(CONSERVATIVE_PROFILE)
DEFAULT_PROFILE_REGISTRY.register(OBSTACLE_PROFILE)
