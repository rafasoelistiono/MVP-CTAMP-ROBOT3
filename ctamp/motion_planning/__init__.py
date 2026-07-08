"""Motion planning: trajectory generation and kinematics."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class TrajectoryPoint:
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    time: float = 0.0


@dataclass
class Trajectory:
    points: List[TrajectoryPoint] = field(default_factory=list)
    cost: float = 0.0

    def __len__(self) -> int:
        return len(self.points)

    def __post_init__(self):
        if not self.points:
            self.points = []


@dataclass
class MotionPlanResult:
    success: bool
    trajectory: Optional[Trajectory] = None
    cost: float = 0.0
    time_elapsed: float = 0.0
    error: Optional[str] = None


class MotionPlanner(ABC):
    @abstractmethod
    def plan(self, start: np.ndarray, goal: np.ndarray, **kwargs) -> MotionPlanResult:
        pass


class RRTPlanner(MotionPlanner):
    def __init__(self, max_iterations: int = 1000, step_size: float = 0.1):
        self.max_iterations = max_iterations
        self.step_size = step_size

    def plan(self, start: np.ndarray, goal: np.ndarray, **kwargs) -> MotionPlanResult:
        trajectory = Trajectory(
            points=[
                TrajectoryPoint(position=start, time=0.0),
                TrajectoryPoint(position=goal, time=1.0),
            ]
        )
        return MotionPlanResult(success=True, trajectory=trajectory)


class PRMPlanner(MotionPlanner):
    def __init__(self, num_samples: int = 100, k_neighbors: int = 5):
        self.num_samples = num_samples
        self.k_neighbors = k_neighbors

    def plan(self, start: np.ndarray, goal: np.ndarray, **kwargs) -> MotionPlanResult:
        trajectory = Trajectory(
            points=[
                TrajectoryPoint(position=start, time=0.0),
                TrajectoryPoint(position=goal, time=1.0),
            ]
        )
        return MotionPlanResult(success=True, trajectory=trajectory)


@dataclass
class CollisionObject:
    position: np.ndarray
    radius: float = 1.0


@dataclass
class CollisionWorld:
    obstacles: List[CollisionObject] = field(default_factory=list)

    def is_collision_free(self, trajectory: Trajectory) -> bool:
        return True


class KinematicModel:
    def __init__(self, joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.joint_limits = joint_limits

    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        return joint_positions

    def inverse_kinematics(self, end_effector_pos: np.ndarray) -> Optional[np.ndarray]:
        return end_effector_pos


from .mock import MockMotionPlanner, MockPlannerConfig
from .mujoco import MuJoCoMotionPlanner

__all__ = [
    "TrajectoryPoint",
    "Trajectory",
    "MotionPlanResult",
    "MotionPlanner",
    "RRTPlanner",
    "PRMPlanner",
    "CollisionObject",
    "CollisionWorld",
    "KinematicModel",
    "MockMotionPlanner",
    "MockPlannerConfig",
    "MuJoCoMotionPlanner",
]
