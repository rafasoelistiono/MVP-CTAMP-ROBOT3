"""Core domain models for CTAMP: object, robot, workspace, action, motion, graph."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Pose(BaseModel):
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


class Shape(BaseModel):
    type: str = "box"
    width: float = 0.1
    height: float = 0.1
    radius: float = 0.0


class ObjectState(BaseModel):
    object_id: str
    pose: Pose = Field(default_factory=Pose)
    shape: Shape = Field(default_factory=Shape)
    movable: bool = False


class RobotState(BaseModel):
    joint_values: dict[str, float] = Field(default_factory=dict)
    active_arm: Optional[str] = None
    holding_object_id: Optional[str] = None


class WorkspaceState(BaseModel):
    objects: dict[str, ObjectState] = Field(default_factory=dict)


class Action(BaseModel):
    action_id: str
    action_type: Literal["transit", "transfer"]
    object_id: str
    arm: Literal["left", "right"]


class JointSpace(BaseModel):
    name: str
    joints: list[str] = Field(default_factory=list)

    @property
    def dimension(self) -> int:
        return len(self.joints)


class MotionPlan(BaseModel):
    waypoints: list[list[float]] = Field(default_factory=list)
    length: float = 0.0
    smoothness: float = 0.0
    clearance: float = 0.0
    planning_time: float = 0.0
    iterations: int = 0
    success: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Vertex(BaseModel):
    vertex_id: str
    robot_state: RobotState
    workspace_state: WorkspaceState
    is_root: bool = False
    is_goal: bool = False


class Edge(BaseModel):
    source: str
    target: str
    action: Action
    joint_space: JointSpace
    motion_plan: Optional[MotionPlan] = None
    cost: float = float("inf")
    flag_motion_planned: bool = False
