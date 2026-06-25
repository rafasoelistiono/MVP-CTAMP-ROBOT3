from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SceneVariant = Literal[
    "group_no_obs",
    "ungroup_no_obs",
    "group_obs",
    "ungroup_obs",
    "group_long_obs",
    "ungroup_long_obs",
]


@dataclass(frozen=True)
class ObjectState:
    id: str
    cls: Literal["cube", "cylinder"]
    pose: tuple[float, float, float]
    reachable: bool
    near_obstacle: bool
    rgba: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ObstacleState:
    id: str
    pose: tuple[float, float, float]
    fragile: bool
    radius: float
    height: Literal["short", "long"]


@dataclass(frozen=True)
class WorldState:
    scene_id: str
    variant: str
    objects: tuple[ObjectState, ...]
    obstacles: tuple[ObstacleState, ...]
    table_x_range: tuple[float, float]
    table_y_range: tuple[float, float]
    table_z_top: float
    goal_center: tuple[float, float, float]
    robot_id: str
    robot_base_xy: tuple[float, float]
    robot_reach_min: float
    robot_reach_max: float
    robot_capabilities: tuple[str, ...]
    task_name: str
    target_objects: tuple[str, ...]
    task_description: str
    preserve_obstacles: bool
    max_retries_per_object: int
    allowed_predicates: tuple[str, ...]
    goal_area_size_xy: tuple[float, float] = (0.52, 0.40)

    def object_by_id(self, oid: str) -> ObjectState | None:
        return next((obj for obj in self.objects if obj.id == oid), None)

    def all_object_ids(self) -> set[str]:
        return {obj.id for obj in self.objects}
