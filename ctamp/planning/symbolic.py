"""Symbolic task planner for tabletop pick-and-place."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Literal

from ..domain.models import (
    Action,
    Edge,
    JointSpace,
    ObjectState,
    Pose,
    RobotState,
    Vertex,
    WorkspaceState,
)
from ..tmm.multigraph import TaskMotionMultigraph


@dataclass
class PlanningProblem:
    objects: dict[str, ObjectState]
    target_poses: dict[str, Pose]
    available_arms: List[Literal["left", "right"]] = field(default_factory=lambda: ["left", "right"])


class SymbolicTaskPlanner:
    """Generate a task graph for tabletop pick-and-place.

    For each object ordering and arm assignment, builds a linear chain of
    transit→pick then transfer→place actions.  All orderings and arm combos
    are enumerated as separate branches in a single multigraph.
    """

    def __init__(self, problem: PlanningProblem) -> None:
        self.problem = problem
        self._joint_space = JointSpace(name="manipulator", joints=["j1", "j2", "j3", "j4", "j5", "j6", "j7"])
        self._next_id = 0

    def _fresh_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def solve(self) -> TaskMotionMultigraph:
        graph = TaskMotionMultigraph()
        obj_ids = list(self.problem.objects.keys())
        root = self._make_vertex("root", placed=frozenset(), is_root=True)
        graph.add_vertex(root)

        if not obj_ids:
            return graph

        for perm in itertools.permutations(obj_ids):
            for arm_combo in itertools.product(self.problem.available_arms, repeat=len(perm)):
                self._build_chain(graph, root, list(perm), list(arm_combo))

        return graph

    def _build_chain(
        self,
        graph: TaskMotionMultigraph,
        root: Vertex,
        obj_order: List[str],
        arm_assignments: List[str],
    ) -> None:
        placed: frozenset[str] = frozenset()
        prev_id = root.vertex_id

        for obj_id, arm in zip(obj_order, arm_assignments):
            pick_vertex = self._make_vertex(
                f"pick_{obj_id}_{arm}_{self._fresh_id()}",
                placed=placed,
            )
            graph.add_vertex(pick_vertex)
            graph.add_edge(self._make_edge(prev_id, pick_vertex.vertex_id, "transit", obj_id, arm))

            placed_after = placed | {obj_id}
            place_vertex = self._make_vertex(
                f"place_{obj_id}_{arm}_{self._fresh_id()}",
                placed=placed_after,
            )
            graph.add_vertex(place_vertex)
            graph.add_edge(self._make_edge(pick_vertex.vertex_id, place_vertex.vertex_id, "transfer", obj_id, arm))

            prev_id = place_vertex.vertex_id
            placed = placed_after

        goal = self._make_vertex(
            f"goal_{self._fresh_id()}",
            placed=placed,
            is_goal=True,
        )
        graph.add_vertex(goal)
        graph.add_edge(self._make_edge(prev_id, goal.vertex_id, "done", "", ""))

    def _make_vertex(
        self,
        vid: str,
        placed: frozenset[str],
        is_root: bool = False,
        is_goal: bool = False,
    ) -> Vertex:
        holding_id = None
        robot = RobotState(holding_object_id=holding_id)
        objects = {
            oid: obj for oid, obj in self.problem.objects.items()
            if oid not in placed
        }
        ws = WorkspaceState(objects=objects)
        return Vertex(vertex_id=vid, robot_state=robot, workspace_state=ws, is_root=is_root, is_goal=is_goal)

    def _make_edge(
        self,
        src: str,
        tgt: str,
        action_type: str,
        obj_id: str,
        arm: str,
    ) -> Edge:
        a_type: Literal["transit", "transfer"] = (
            "transit" if action_type in ("transit", "done") else "transfer"
        )
        action = Action(
            action_id=f"{action_type}_{obj_id}_{arm}_{src}_{tgt}",
            action_type=a_type,
            object_id=obj_id,
            arm=arm if arm else "left",  # default for 'done'
        )
        return Edge(source=src, target=tgt, action=action, joint_space=self._joint_space)
