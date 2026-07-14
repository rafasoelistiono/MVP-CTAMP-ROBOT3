"""Task planner with A* search."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from abc import ABC, abstractmethod

from ..domain import Problem, State, GroundAction
from .search import AStarSearch


@dataclass
class PlannerConfig:
    max_nodes: int = 10000
    timeout: float = 30.0
    heuristic_weight: float = 1.0


@dataclass
class PlanResult:
    success: bool
    actions: List[GroundAction] = field(default_factory=list)
    states: List[State] = field(default_factory=list)
    cost: float = 0.0
    nodes_expanded: int = 0
    error: Optional[str] = None


class Heuristic(ABC):
    @abstractmethod
    def __call__(self, state: State, problem: Problem) -> float:
        pass


class AdditiveHeuristic(Heuristic):
    def __call__(self, state: State, problem: Problem) -> float:
        return float(len(problem.goal - state.predicates))


class RelaxedPlanHeuristic(Heuristic):
    def __call__(self, state: State, problem: Problem) -> float:
        return float(len(problem.goal - state.predicates))


class TaskPlanner:
    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        heuristic: Optional[Heuristic] = None,
    ):
        self.config = config or PlannerConfig()
        self.heuristic = heuristic or AdditiveHeuristic()

    def plan(self, problem: Problem) -> PlanResult:
        search = AStarSearch(
            heuristic=self.heuristic,
            max_nodes=self.config.max_nodes,
            weight=self.config.heuristic_weight,
        )
        return search.search(problem)
