"""FeatureExtractor for path cost prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..domain.models import Edge, Pose, Vertex


@dataclass
class PathFeatureConfig:
    max_joints: int = 7
    max_objects: int = 10


class PathFeatureExtractor:
    """Extract fixed-size feature vector from a candidate path.

    Combines geometric features (robot state, object poses, distances)
    and symbolic features (action ratios, position indices).
    """

    def __init__(self, config: Optional[PathFeatureConfig] = None) -> None:
        self.config = config or PathFeatureConfig()

    @property
    def feature_dim(self) -> int:
        c = self.config
        geometric = (
            c.max_joints + 1 + c.max_objects * 3 + c.max_objects * 4 + c.max_objects
        )
        symbolic = 7
        return geometric + symbolic

    def extract(
        self,
        root_vertex: Vertex,
        path_edges: List[Edge],
        target_poses: dict[str, Pose],
    ) -> np.ndarray:
        features: List[float] = []

        robot = root_vertex.robot_state
        joints = [
            robot.joint_values.get(f"j{i}", 0.0) for i in range(self.config.max_joints)
        ]
        features.extend(joints)

        arm_val = 0.5
        if robot.active_arm == "left":
            arm_val = 0.0
        elif robot.active_arm == "right":
            arm_val = 1.0
        features.append(arm_val)

        ws = root_vertex.workspace_state
        obj_ids = sorted(ws.objects.keys())[: self.config.max_objects]
        for oid in obj_ids:
            obj = ws.objects[oid]
            features.extend([obj.pose.x, obj.pose.y, obj.pose.theta])
        for _ in range(self.config.max_objects - len(obj_ids)):
            features.extend([0.0, 0.0, 0.0])

        for oid in obj_ids:
            obj = ws.objects[oid]
            features.extend(
                [
                    _hash_type(obj.shape.type),
                    obj.shape.width,
                    obj.shape.height,
                    obj.shape.radius,
                ]
            )
        for _ in range(self.config.max_objects - len(obj_ids)):
            features.extend([0.0, 0.0, 0.0, 0.0])

        for oid in obj_ids:
            obj = ws.objects[oid]
            tgt = target_poses.get(oid, Pose())
            dist = np.hypot(obj.pose.x - tgt.x, obj.pose.y - tgt.y)
            features.append(dist)
        for _ in range(self.config.max_objects - len(obj_ids)):
            features.append(0.0)

        n = len(path_edges)
        features.append(float(n))

        if n > 0:
            transit_count = sum(
                1 for e in path_edges if e.action.action_type == "transit"
            )
            transfer_count = n - transit_count
            left_count = sum(1 for e in path_edges if e.action.arm == "left")
            right_count = n - left_count

            features.append(transit_count / n)
            features.append(transfer_count / n)
            features.append(right_count / n)
            features.append(left_count / n)

            transit_idx = [
                i for i, e in enumerate(path_edges) if e.action.action_type == "transit"
            ]
            transfer_idx = [
                i
                for i, e in enumerate(path_edges)
                if e.action.action_type == "transfer"
            ]
            features.append(np.mean(transit_idx) / n if transit_idx else 0.0)
            features.append(np.mean(transfer_idx) / n if transfer_idx else 0.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)


def _hash_type(t: str) -> float:
    h = hash(t) % 1000
    return float(h) / 1000.0
