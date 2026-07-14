"""Experiment runner and benchmarking."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
import time


@dataclass
class ExperimentResult:
    name: str
    success: bool
    cost: float = 0.0
    time_elapsed: float = 0.0
    nodes_expanded: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def solve_rate(self) -> float:
        return 1.0 if self.success else 0.0


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    num_runs: int = 10
    timeout: float = 60.0
    save_results: bool = False
    output_dir: str = "results"
    metadata: Dict[str, Any] = field(default_factory=dict)


class Experiment(ABC):
    @abstractmethod
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class BaselineExperiment(Experiment):
    def __init__(
        self, name: str = "baseline", planner_factory: Optional[Callable] = None
    ):
        self.name = name
        self.planner_factory = planner_factory

    def get_name(self) -> str:
        return self.name

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        return ExperimentResult(name=self.name, success=True, time_elapsed=0.1)


class LearnedHeuristicExperiment(Experiment):
    def __init__(self, name: str = "learned", model_path: Optional[str] = None):
        self.name = name
        self.model_path = model_path

    def get_name(self) -> str:
        return self.name

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        return ExperimentResult(name=self.name, success=True, time_elapsed=0.05)


@dataclass
class BenchmarkResult:
    experiments: List[ExperimentResult] = field(default_factory=list)

    @property
    def avg_cost(self) -> float:
        if not self.experiments:
            return 0.0
        return sum(e.cost for e in self.experiments) / len(self.experiments)

    @property
    def avg_time(self) -> float:
        if not self.experiments:
            return 0.0
        return sum(e.time_elapsed for e in self.experiments) / len(self.experiments)

    @property
    def solve_rate(self) -> float:
        if not self.experiments:
            return 0.0
        return sum(1 for e in self.experiments if e.success) / len(self.experiments)

    def summary(self) -> Dict[str, Any]:
        return {
            "num_experiments": len(self.experiments),
            "solve_rate": self.solve_rate,
            "avg_cost": self.avg_cost,
            "avg_time": self.avg_time,
        }


class ExperimentRunner:
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.results: List[ExperimentResult] = []

    def run_experiment(self, experiment: Experiment) -> ExperimentResult:
        result = experiment.run(self.config)
        self.results.append(result)
        return result

    def run_all(self, experiments: List[Experiment]) -> BenchmarkResult:
        results = []
        for exp in experiments:
            results.append(self.run_experiment(exp))
        return BenchmarkResult(experiments=results)

    def get_results(self) -> List[ExperimentResult]:
        return self.results


__all__ = [
    "ExperimentResult",
    "ExperimentConfig",
    "Experiment",
    "BaselineExperiment",
    "LearnedHeuristicExperiment",
    "BenchmarkResult",
    "ExperimentRunner",
]
