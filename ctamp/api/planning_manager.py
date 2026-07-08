"""Planning manager for API service."""

from __future__ import annotations

from typing import Optional

from ..domain.models import ObjectState, Pose, Shape
from ..planning.symbolic import PlanningProblem, SymbolicTaskPlanner
from ..search.baseline import BaselinePlanner
from ..tmm.builder import TMMGraphBuilder
from .models import (
    ActionStep,
    PlanningProblemRequest,
    PlanningResult,
)


class PlanningManager:
    """Manage planning requests."""

    def __init__(self, max_time: float = 30.0) -> None:
        self.max_time = max_time

    def run(self, request: PlanningProblemRequest) -> PlanningResult:
        try:
            problem = self._build_problem(request)
            planner = SymbolicTaskPlanner(problem)
            symbolic_graph = planner.solve()
            builder = TMMGraphBuilder()
            tmm_graph = builder.build(symbolic_graph)

            if not request.objects:
                return PlanningResult(
                    success=True,
                    actions=[],
                    cost=0.0,
                    vertices_expanded=0,
                    time_elapsed=0.0,
                )

            baseline = BaselinePlanner(max_time=request.max_time)
            result = baseline.search(tmm_graph)

            if not result.success:
                return PlanningResult(
                    success=False,
                    error=result.error or "no_solution",
                )

            actions = [
                ActionStep(
                    action_type=e.action.action_type,
                    object_id=e.action.object_id,
                    arm=e.action.arm,
                )
                for e in result.path_edges
            ]

            return PlanningResult(
                success=True,
                actions=actions,
                cost=result.cost,
                vertices_expanded=result.nodes_expanded,
                time_elapsed=result.time_elapsed,
            )
        except Exception as e:
            return PlanningResult(success=False, error=str(e))

    def _build_problem(self, request: PlanningProblemRequest) -> PlanningProblem:
        objects = {}
        for obj in request.objects:
            objects[obj.object_id] = ObjectState(
                object_id=obj.object_id,
                pose=Pose(x=obj.pose.x, y=obj.pose.y),
                shape=Shape(type=obj.shape),
            )

        target_poses = {}
        for oid, tp in request.target_poses.items():
            target_poses[oid] = Pose(x=tp.x, y=tp.y)

        return PlanningProblem(
            objects=objects,
            target_poses=target_poses,
            available_arms=request.available_arms,
        )
