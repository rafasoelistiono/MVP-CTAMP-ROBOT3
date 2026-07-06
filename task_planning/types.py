from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SCHEMA_VERSION = "ctamp-plan/v1"
ALLOWED_ACTIONS = frozenset({"pick", "place"})
ALLOWED_PREDICATES = frozenset({"at", "clear", "handempty", "holding", "stable"})


@dataclass(frozen=True)
class SlotConfig:
    type: Literal["line"]
    axis: str = "x"
    spacing_m: float = 0.125
    row_y: float = -0.06
    center_x: float = 0.22
    base_z: float = 0.83


@dataclass(frozen=True)
class Step:
    step_id: int
    action: Literal["pick", "place"]
    object: str
    slot: str | None = None
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


@dataclass(frozen=True)
class ScoredPlan:
    plan_id: str
    plan: TaskPlan
    estimated_cost: float
    generation_method: str
    edge_costs: tuple[float, ...] = ()


@dataclass(frozen=True)
class ProbeResult:
    feasible: bool
    ik_success: bool
    ompl_success: bool
    planning_time: float = 0.0
    estimated_path_length: float = 0.0
    min_clearance: float = 0.0
    collision_count: int = 0
    failure_reason: str | None = None
    grasp_diagnostics: tuple[dict, ...] = ()
    timeout_phase: str | None = None
    attempted_orientations: int = 0
    attempted_seeds: int = 0


@dataclass(frozen=True)
class ProbePlanResult:
    feasible: bool
    edge_results: tuple[ProbeResult, ...] = ()
    total_planning_time: float = 0.0
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfirmationResult:
    confirmed: bool
    selected_plan_id: str | None = None
    plan: TaskPlan | None = None
    total_probes: int = 0
    total_ik_failures: int = 0
    total_ompl_failures: int = 0
    total_planning_time: float = 0.0
    failed_plan_ids: tuple[str, ...] = ()
    failure_reasons: tuple[str, ...] = ()
