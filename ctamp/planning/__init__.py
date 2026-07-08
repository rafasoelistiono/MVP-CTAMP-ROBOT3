"""Unified planning interface combining task and motion planning."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple
from abc import ABC, abstractmethod

from ..domain import Domain, Problem, State, GroundAction
from ..task_planning import TaskPlanner, PlanResult, PlannerConfig, Heuristic


@dataclass
class PlanStep:
    task_action: Optional[GroundAction] = None
    motion_trajectory: Any = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FullPlan:
    steps: List[PlanStep] = field(default_factory=list)
    total_cost: float = 0.0
    total_duration: float = 0.0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.steps)


@dataclass
class PlannerConfig:
    task_config: PlannerConfig = field(default_factory=PlannerConfig)
    motion_config: Any = None
    integration_mode: str = "sequential"
    max_iterations: int = 100
    timeout: float = 60.0


class TaskMotionPlanner(ABC):
    @abstractmethod
    def solve(self, problem: Problem, **kwargs) -> FullPlan:
        pass


class SequentialPlanner(TaskMotionPlanner):
    def __init__(self, task_planner: Optional[TaskPlanner] = None):
        self.task_planner = task_planner or TaskPlanner()

    def solve(self, problem: Problem, **kwargs) -> FullPlan:
        result = self.task_planner.plan(problem)
        if not result.success:
            return FullPlan(success=False)

        steps = []
        for action in result.actions:
            step = PlanStep(task_action=action)
            steps.append(step)

        return FullPlan(steps=steps, total_cost=result.cost, success=True)


class IntegratedPlanner(TaskMotionPlanner):
    def __init__(self, task_planner: Optional[TaskPlanner] = None):
        self.task_planner = task_planner or TaskPlanner()

    def solve(self, problem: Problem, **kwargs) -> FullPlan:
        return SequentialPlanner(self.task_planner).solve(problem, **kwargs)


class LazyPlanner(TaskMotionPlanner):
    def __init__(self, task_planner: Optional[TaskPlanner] = None):
        self.task_planner = task_planner or TaskPlanner()

    def solve(self, problem: Problem, **kwargs) -> FullPlan:
        return SequentialPlanner(self.task_planner).solve(problem, **kwargs)


class CTAMPPlanner(TaskMotionPlanner):
    def __init__(
        self,
        task_planner: Optional[TaskPlanner] = None,
        heuristic: Optional[Heuristic] = None,
        config: Optional[PlannerConfig] = None,
    ):
        self.task_planner = task_planner or TaskPlanner(heuristic=heuristic)
        self.config = config or PlannerConfig()

    def solve(self, problem: Problem, **kwargs) -> FullPlan:
        task_result = self.task_planner.plan(problem)
        if not task_result.success:
            return FullPlan(success=False, metadata={"error": task_result.error})

        steps = [PlanStep(task_action=a) for a in task_result.actions]
        return FullPlan(steps=steps, total_cost=task_result.cost, success=True)


from .symbolic import PlanningProblem, SymbolicTaskPlanner
from .confirmation import confirm_solution, CompletePlan, EmptyPlan

__all__ = [
    "PlanStep",
    "FullPlan",
    "PlannerConfig",
    "TaskMotionPlanner",
    "SequentialPlanner",
    "IntegratedPlanner",
    "LazyPlanner",
    "CTAMPPlanner",
    "PlanningProblem",
    "SymbolicTaskPlanner",
    "confirm_solution",
    "CompletePlan",
    "EmptyPlan",
]
