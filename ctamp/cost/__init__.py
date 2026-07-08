"""Cost computation for task-motion planning."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class CostComponent:
    name: str
    weight: float = 1.0
    cost: float = 0.0

    @property
    def weighted_cost(self) -> float:
        return self.weight * self.cost


@dataclass
class PlanCost:
    components: List[CostComponent] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(c.weighted_cost for c in self.components)

    def add(self, name: str, weight: float, cost: float) -> None:
        self.components.append(CostComponent(name, weight, cost))


class CostFunction(ABC):
    @abstractmethod
    def compute(self, plan: Any, state: Any) -> float:
        pass


class TaskCost(CostFunction):
    def __init__(self, action_costs: Optional[dict] = None):
        self.action_costs = action_costs or {}

    def compute(self, plan: Any, state: Any = None) -> float:
        return float(len(plan)) if hasattr(plan, '__len__') else 0.0


class MotionCost(CostFunction):
    def __init__(self, distance_weight: float = 1.0, time_weight: float = 0.1):
        self.distance_weight = distance_weight
        self.time_weight = time_weight

    def compute(self, trajectory: Any, state: Any = None) -> float:
        if trajectory is None:
            return 0.0
        return float(len(trajectory)) if hasattr(trajectory, '__len__') else 0.0


class CompositeCost(CostFunction):
    def __init__(self, functions: List[CostFunction], weights: Optional[List[float]] = None):
        self.functions = functions
        self.weights = weights or [1.0] * len(functions)

    def compute(self, plan: Any, state: Any = None) -> float:
        total = 0.0
        for fn, w in zip(self.functions, self.weights):
            total += w * fn.compute(plan, state)
        return total


from .edge_cost import EdgeCostCalculator, CostWeights

__all__ = [
    "CostComponent",
    "PlanCost",
    "CostFunction",
    "TaskCost",
    "MotionCost",
    "CompositeCost",
    "EdgeCostCalculator",
    "CostWeights",
]
