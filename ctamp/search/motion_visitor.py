"""MotionPlanningVisitor: Algorithm 2 from the paper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from ..domain.models import Edge, MotionPlan, Vertex
from ..motion_planning.mock import MockMotionPlanner
from ..cost.edge_cost import EdgeCostCalculator
from ..tmm.multigraph import TaskMotionMultigraph


FAIL_PENALTY = 1e6


@dataclass
class LearningSample:
    source_id: str
    target_id: str
    joint_space_dim: int
    success: bool
    cost: float


class LearningMode(ABC):
    @abstractmethod
    def add_sample(self, sample: LearningSample) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass


class OfflineLearning(LearningMode):
    def __init__(self) -> None:
        self.samples: List[LearningSample] = []

    def add_sample(self, sample: LearningSample) -> None:
        self.samples.append(sample)

    def train(self) -> None:
        pass


class OnlineLearning(LearningMode):
    def __init__(self, model: Optional[object] = None) -> None:
        self.model = model
        self.samples: List[LearningSample] = []

    def add_sample(self, sample: LearningSample) -> None:
        self.samples.append(sample)
        if self.model is not None and hasattr(self.model, "partial_fit"):
            self.model.partial_fit(sample)

    def train(self) -> None:
        pass


class MotionPlanningVisitor:
    """Algorithm 2: motion planning visitor for TMM A*.

    On expand(v):
      for each v' adjacent to v:
        edges = edges(v, v') sorted by joint_dim asc
        for edge in edges:
          if edge already planned → skip remaining alternatives
          run motion planner
          if plan found → store, update cost, break
          else → mark planned, assign penalty
    """

    def __init__(
        self,
        planner: Optional[MockMotionPlanner] = None,
        cost_calculator: Optional[EdgeCostCalculator] = None,
        learning_mode: Optional[LearningMode] = None,
    ) -> None:
        self.planner = planner or MockMotionPlanner(seed=42)
        self.cost_calc = cost_calculator or EdgeCostCalculator()
        self.learning = learning_mode or OfflineLearning()
        self._planned_edges: set[str] = set()

    def on_expand(self, graph: TaskMotionMultigraph, vertex: Vertex) -> None:
        outgoing = graph.get_outgoing_edges(vertex.vertex_id)
        by_target: dict[str, List[Edge]] = {}
        for e in outgoing:
            by_target.setdefault(e.target, []).append(e)

        for target_id, edges in by_target.items():
            edges.sort(key=lambda e: len(e.joint_space.joints))
            found_plan = False
            for edge in edges:
                if edge.flag_motion_planned:
                    found_plan = True
                    break
                ws = vertex.workspace_state
                plan = self.planner.plan(edge, ws)
                edge.flag_motion_planned = True
                self._planned_edges.add(edge.action.action_id)
                if plan is not None:
                    edge.motion_plan = plan
                    edge.cost = self.cost_calc.compute(edge, plan)
                    self._add_sample(edge, True)
                    found_plan = True
                    break
                else:
                    edge.cost = FAIL_PENALTY
                    self._add_sample(edge, False)
            if not found_plan:
                pass

    def _add_sample(self, edge: Edge, success: bool) -> None:
        sample = LearningSample(
            source_id=edge.source,
            target_id=edge.target,
            joint_space_dim=len(edge.joint_space.joints),
            success=success,
            cost=edge.cost,
        )
        self.learning.add_sample(sample)

    def on_edge(self, edge: Edge, motion_plan: Optional[MotionPlan]) -> None:
        pass
