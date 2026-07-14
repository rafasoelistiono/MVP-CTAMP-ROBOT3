"""Generate tabletop planning problems for benchmarking."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from ..domain.models import ObjectState, Pose, Shape
from ..planning.symbolic import PlanningProblem


@dataclass
class ProblemConfig:
    num_objects: int = 3
    workspace_size: float = 1.0
    object_size_range: tuple[float, float] = (0.05, 0.15)
    seed: int = 42


class ProblemGenerator:
    """Generate random tabletop pick-and-place problems."""

    def __init__(self, config: ProblemConfig | None = None) -> None:
        self.config = config or ProblemConfig()
        self._rng = random.Random(self.config.seed)

    def generate(self) -> PlanningProblem:
        objects = {}
        target_poses = {}

        for i in range(self.config.num_objects):
            oid = f"obj{i}"
            x = self._rng.uniform(0, self.config.workspace_size)
            y = self._rng.uniform(0, self.config.workspace_size)
            size = self._rng.uniform(*self.config.object_size_range)

            objects[oid] = ObjectState(
                object_id=oid,
                pose=Pose(x=x, y=y),
                shape=Shape(type="box", width=size, height=size),
            )

            tx = self._rng.uniform(0, self.config.workspace_size)
            ty = self._rng.uniform(0, self.config.workspace_size)
            target_poses[oid] = Pose(x=tx, y=ty)

        return PlanningProblem(
            objects=objects,
            target_poses=target_poses,
            available_arms=["left", "right"],
        )

    def generate_batch(self, count: int) -> List[PlanningProblem]:
        problems = []
        for i in range(count):
            gen = ProblemGenerator(
                ProblemConfig(
                    num_objects=self.config.num_objects,
                    workspace_size=self.config.workspace_size,
                    object_size_range=self.config.object_size_range,
                    seed=self.config.seed + i,
                )
            )
            problems.append(gen.generate())
        return problems
