"""Pydantic models for API requests and responses."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# Health
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


# Planning
class ObjectPose(BaseModel):
    x: float = 0.0
    y: float = 0.0


class TargetPose(BaseModel):
    x: float = 0.0
    y: float = 0.0


class PlanningObject(BaseModel):
    object_id: str
    pose: ObjectPose = Field(default_factory=ObjectPose)
    shape: str = "box"


class PlanningProblemRequest(BaseModel):
    objects: list[PlanningObject]
    target_poses: dict[str, TargetPose]
    available_arms: list[str] = Field(default_factory=lambda: ["left", "right"])
    max_time: float = 30.0


class ActionStep(BaseModel):
    action_type: str
    object_id: str
    arm: str


class PlanningResult(BaseModel):
    success: bool
    actions: list[ActionStep] = Field(default_factory=list)
    cost: float = 0.0
    vertices_expanded: int = 0
    time_elapsed: float = 0.0
    error: Optional[str] = None


# Learning
class SampleData(BaseModel):
    features: list[float]
    cost: float


class TrainingRequest(BaseModel):
    samples: list[SampleData]
    model_type: str = "offline"


class TrainingResult(BaseModel):
    success: bool
    num_samples: int = 0
    model_path: Optional[str] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = None
