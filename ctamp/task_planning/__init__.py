"""Task planner with A* search."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, FrozenSet
from abc import ABC, abstractmethod
import time
import heapq

from ..domain import (
    Domain,
    Problem,
    State,
    GroundPredicate,
    GroundAction,
    Operator,
    Object,
    Predicate,
    PredicateType,
)


class Heuristic(ABC):
    @abstractmethod
    def evaluate(self, state: State, goal: FrozenSet[GroundPredicate]) -> float:
        pass

    def __call__(self, state: State, goal: FrozenSet[GroundPredicate]) -> float:
        return self.evaluate(state, goal)


class ZeroHeuristic(Heuristic):
    def evaluate(self, state: State, goal: FrozenSet[GroundPredicate]) -> float:
        return 0.0


class GoalCountHeuristic(Heuristic):
    def evaluate(self, state: State, goal: FrozenSet[GroundPredicate]) -> float:
        return float(len(goal - state.predicates))


class FFHeuristic(Heuristic):
    def evaluate(self, state: State, goal: FrozenSet[GroundPredicate]) -> float:
        return float(len(goal - state.predicates))


class LearnedHeuristicWrapper(Heuristic):
    def __init__(self, model):
        self.model = model

    def evaluate(self, state: State, goal: FrozenSet[GroundPredicate]) -> float:
        return self.model.predict(state, goal)


@dataclass(order=True)
class SearchNode:
    f_cost: float
    state: State = field(compare=False)
    g_cost: float = field(compare=False)
    action: Optional[GroundAction] = field(default=None, compare=False)
    parent: Optional[SearchNode] = field(default=None, compare=False)

    @property
    def f(self) -> float:
        return self.f_cost

    @property
    def g(self) -> float:
        return self.g_cost

    def path_actions(self) -> List[GroundAction]:
        actions = []
        node = self
        while node.parent is not None:
            if node.action is not None:
                actions.append(node.action)
            node = node.parent
        return list(reversed(actions))


@dataclass
class PlanResult:
    success: bool
    actions: List[GroundAction] = field(default_factory=list)
    states: List[State] = field(default_factory=list)
    cost: float = 0.0
    nodes_expanded: int = 0
    time_elapsed: float = 0.0
    error: Optional[str] = None


@dataclass
class PlannerConfig:
    max_nodes: int = 100000
    max_time: float = 30.0
    heuristic_weight: float = 1.0
    use_goal_count: bool = True


class TaskPlanner:
    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        heuristic: Optional[Heuristic] = None,
    ):
        self.config = config or PlannerConfig()
        self.heuristic = heuristic or GoalCountHeuristic()

    def plan(self, problem: Problem) -> PlanResult:
        start_time = time.time()
        open_set: List[SearchNode] = []
        closed_set: set = set()

        h0 = self.heuristic(problem.init, problem.goal)
        start_node = SearchNode(f_cost=h0, state=problem.init, g_cost=0.0)
        heapq.heappush(open_set, start_node)

        nodes_expanded = 0

        while open_set:
            if time.time() - start_time > self.config.max_time:
                return PlanResult(
                    success=False,
                    nodes_expanded=nodes_expanded,
                    time_elapsed=time.time() - start_time,
                    error="timeout",
                )

            current = heapq.heappop(open_set)
            state_key = frozenset(current.state.predicates)

            if state_key in closed_set:
                continue
            closed_set.add(state_key)

            if problem.is_goal(current.state):
                return PlanResult(
                    success=True,
                    actions=current.path_actions(),
                    cost=current.g_cost,
                    nodes_expanded=nodes_expanded,
                    time_elapsed=time.time() - start_time,
                )

            nodes_expanded += 1

            if nodes_expanded >= self.config.max_nodes:
                return PlanResult(
                    success=False,
                    nodes_expanded=nodes_expanded,
                    time_elapsed=time.time() - start_time,
                    error="max_nodes_reached",
                )

            for action in self._get_applicable_actions(current.state, problem):
                try:
                    new_state = action(current.state)
                except ValueError:
                    continue

                new_state_key = frozenset(new_state.predicates)
                if new_state_key in closed_set:
                    continue

                g = current.g_cost + action.cost
                h = self.heuristic(new_state, problem.goal)
                f = g + self.config.heuristic_weight * h

                child = SearchNode(
                    f_cost=f,
                    state=new_state,
                    g_cost=g,
                    action=action,
                    parent=current,
                )
                heapq.heappush(open_set, child)

        return PlanResult(
            success=False,
            nodes_expanded=nodes_expanded,
            time_elapsed=time.time() - start_time,
            error="no_solution",
        )

    def _get_applicable_actions(
        self, state: State, problem: Problem
    ) -> List[GroundAction]:
        actions = []
        for op in problem.domain.operators.values():
            for combo in self._get_groundings(op, problem):
                ga = op.ground(*combo)
                if ga.is_applicable(state):
                    actions.append(ga)
        return actions

    def _get_groundings(self, op: Operator, problem: Problem) -> List[tuple]:
        from itertools import product

        param_types = [p.type for p in op.schema.parameters]
        candidates = []
        for pt in param_types:
            candidates.append([o for o in problem.objects if o.type.is_subtype(pt)])
        return list(product(*candidates))
