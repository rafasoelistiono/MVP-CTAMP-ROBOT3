"""Episode runner for benchmarking planning algorithms."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

from ..domain.models import Vertex
from ..learning.heuristic_models import (
    ConstantHeuristicModel,
    LearnedModel,
    OfflineSVRModel,
    OnlineSGDModel,
)
from ..learning.path_features import PathFeatureConfig
from ..planning.symbolic import PlanningProblem, SymbolicTaskPlanner
from ..search.baseline import BaselinePlanner, BaselineResult
from ..search.heuristic_estimator import HeuristicPathEstimator
from ..search.motion_visitor import MotionPlanningVisitor
from ..tmm.builder import TMMGraphBuilder
from ..tmm.multigraph import TaskMotionMultigraph
from .problem_generator import ProblemConfig, ProblemGenerator


@dataclass
class EpisodeMetrics:
    episode_id: int
    num_objects: int
    planner_type: str
    success: bool
    cost: float
    nodes_expanded: int
    nodes_generated: int
    time_elapsed: float
    heuristic_error: float | None = None
    epsilon_suboptimal: float | None = None

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "num_objects": self.num_objects,
            "planner_type": self.planner_type,
            "success": self.success,
            "cost": self.cost,
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "time_elapsed": self.time_elapsed,
            "heuristic_error": self.heuristic_error,
            "epsilon_suboptimal": self.epsilon_suboptimal,
        }


@dataclass
class EpisodeResult:
    episode_id: int
    num_objects: int
    metrics: dict[str, EpisodeMetrics] = field(default_factory=dict)


class EpisodeRunner:
    """Run planning episodes and collect metrics."""

    def __init__(
        self,
        num_episodes: int = 10,
        object_counts: List[int] | None = None,
        seed: int = 42,
        output_dir: str = "results",
    ) -> None:
        self.num_episodes = num_episodes
        self.object_counts = object_counts or [1, 2, 3, 4, 5]
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[EpisodeResult]:
        results = []
        for episode_id in range(self.num_episodes):
            for num_objects in self.object_counts:
                result = self._run_episode(episode_id, num_objects)
                results.append(result)
        return results

    def _run_episode(self, episode_id: int, num_objects: int) -> EpisodeResult:
        config = ProblemConfig(num_objects=num_objects, seed=self.seed + episode_id)
        generator = ProblemGenerator(config)
        problem = generator.generate()

        planner = SymbolicTaskPlanner(problem)
        symbolic_graph = planner.solve()
        builder = TMMGraphBuilder()
        tmm_graph = builder.build(symbolic_graph)

        result = EpisodeResult(episode_id=episode_id, num_objects=num_objects)

        baseline_metrics = self._run_baseline(episode_id, num_objects, tmm_graph)
        result.metrics["baseline"] = baseline_metrics

        offline_metrics = self._run_learned(episode_id, num_objects, tmm_graph, "offline")
        result.metrics["offline"] = offline_metrics

        online_metrics = self._run_learned(episode_id, num_objects, tmm_graph, "online")
        result.metrics["online"] = online_metrics

        if baseline_metrics.success and offline_metrics.success:
            offline_metrics.epsilon_suboptimal = offline_metrics.cost / baseline_metrics.cost if baseline_metrics.cost > 0 else None
            offline_metrics.heuristic_error = abs(offline_metrics.cost - baseline_metrics.cost)

        if baseline_metrics.success and online_metrics.success:
            online_metrics.epsilon_suboptimal = online_metrics.cost / baseline_metrics.cost if baseline_metrics.cost > 0 else None
            online_metrics.heuristic_error = abs(online_metrics.cost - baseline_metrics.cost)

        return result

    def _run_baseline(
        self, episode_id: int, num_objects: int, graph: TaskMotionMultigraph
    ) -> EpisodeMetrics:
        planner = BaselinePlanner(max_time=30.0)
        result = planner.search(graph)

        return EpisodeMetrics(
            episode_id=episode_id,
            num_objects=num_objects,
            planner_type="baseline",
            success=result.success,
            cost=result.cost if result.success else float("inf"),
            nodes_expanded=result.nodes_expanded,
            nodes_generated=result.nodes_generated,
            time_elapsed=result.time_elapsed,
        )

    def _run_learned(
        self,
        episode_id: int,
        num_objects: int,
        graph: TaskMotionMultigraph,
        mode: Literal["offline", "online"],
    ) -> EpisodeMetrics:
        if mode == "offline":
            model: LearnedModel = OfflineSVRModel()
        else:
            model = OnlineSGDModel()

        estimator = HeuristicPathEstimator(model=model)
        estimator.set_graph(graph)

        planner = BaselinePlanner(max_time=30.0)
        result = planner.search(graph)

        return EpisodeMetrics(
            episode_id=episode_id,
            num_objects=num_objects,
            planner_type=mode,
            success=result.success,
            cost=result.cost if result.success else float("inf"),
            nodes_expanded=result.nodes_expanded,
            nodes_generated=result.nodes_generated,
            time_elapsed=result.time_elapsed,
        )

    def save_csv(self, results: List[EpisodeResult], filename: str = "episodes.csv") -> None:
        path = self.output_dir / filename
        rows = []
        for result in results:
            for planner_type, metrics in result.metrics.items():
                rows.append(metrics.to_dict())

        with open(path, "w", newline="") as f:
            if not rows:
                return
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def save_json(self, results: List[EpisodeResult], filename: str = "episodes.json") -> None:
        path = self.output_dir / filename
        data = []
        for result in results:
            for planner_type, metrics in result.metrics.items():
                data.append(metrics.to_dict())
        path.write_text(json.dumps(data, indent=2))
