from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SCHEMA_VERSION = "ctamp-plan/v1"
ALLOWED_ACTIONS = frozenset({"pick", "place", "stack_place"})
ALLOWED_PREDICATES = frozenset(
    {"at", "on", "clear", "handempty", "holding"}
)


@dataclass(frozen=True)
class SlotConfig:
    type: Literal["tower", "pyramid"]
    axis: str = "x"
    spacing_m: float = 0.125
    row_spacing_m: float = 0.08
    row_count: int = 0
    base_row_length: int = 0
    center_x: float = 0.22
    base_y: float = -0.06
    base_z: float = 0.83
    base_xy: tuple[float, float] = (0.22, -0.06)
    layer_height_m: float = 0.06


@dataclass(frozen=True)
class Step:
    step_id: int
    action: Literal["pick", "place", "stack_place"]
    object: str
    slot: str | None = None
    on_top_of: str | None = None
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskPlan:
    schema_version: str
    task: str
    scene_id: str
    target_objects: tuple[str, ...]
    goal_predicates: tuple[dict, ...]
    slot_config: SlotConfig
    steps: tuple[Step, ...]
    constraints: dict = field(default_factory=dict)
