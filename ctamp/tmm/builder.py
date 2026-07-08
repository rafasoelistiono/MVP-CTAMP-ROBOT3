"""TMMGraphBuilder: expand symbolic task graph into TMM multigraph."""

from __future__ import annotations

from typing import Dict, List

from ..domain.models import Edge, JointSpace, MotionPlan, Vertex
from .multigraph import TaskMotionMultigraph


JOINT_SPACE_ALTERNATIVES: Dict[str, List[JointSpace]] = {
    "left": [
        JointSpace(name="left_arm", joints=["l_j1", "l_j2", "l_j3", "l_j4", "l_j5", "l_j6", "l_j7"]),
        JointSpace(name="left_arm_chest", joints=["l_j1", "l_j2", "l_j3", "l_j4", "l_j5", "l_j6", "l_j7", "chest_j1"]),
    ],
    "right": [
        JointSpace(name="right_arm", joints=["r_j1", "r_j2", "r_j3", "r_j4", "r_j5", "r_j6", "r_j7"]),
        JointSpace(name="right_arm_chest", joints=["r_j1", "r_j2", "r_j3", "r_j4", "r_j5", "r_j6", "r_j7", "chest_j1"]),
    ],
}

DEFAULT_EDGE_COST = 1e6


class TMMGraphBuilder:
    """Expand a symbolic task graph into a TMM multigraph.

    Each symbolic edge becomes N TMM edges, one per compatible joint-space
    alternative.  Vertices are copied 1-to-1.
    """

    def build(self, symbolic_graph: TaskMotionMultigraph) -> TaskMotionMultigraph:
        tmm = TaskMotionMultigraph()

        for _, data in symbolic_graph._graph.nodes(data=True):
            tmm.add_vertex(data["vertex"])

        for src, tgt, edge_data in symbolic_graph._graph.edges(data=True):
            sym_edge: Edge = edge_data["edge"]
            arm = sym_edge.action.arm
            alternatives = JOINT_SPACE_ALTERNATIVES.get(arm, [])
            for js in alternatives:
                tmm_edge = Edge(
                    source=sym_edge.source,
                    target=sym_edge.target,
                    action=sym_edge.action,
                    joint_space=js,
                    motion_plan=None,
                    cost=DEFAULT_EDGE_COST,
                    flag_motion_planned=False,
                )
                tmm.add_edge(tmm_edge)

        return tmm
