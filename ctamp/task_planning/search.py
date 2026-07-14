"""A* search implementation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set, Callable, Generic, TypeVar
import heapq
import time

from ..domain import GroundAction, Operator, Problem, State

T = TypeVar("T")


@dataclass(order=True)
class SearchNode(Generic[T]):
    f_cost: float
    state: T = field(compare=False)
    g_cost: float = field(compare=False)
    action: Optional[GroundAction] = field(default=None, compare=False)
    parent: Optional["SearchNode[T]"] = field(default=None, compare=False)

    def path(self) -> List[GroundAction]:
        actions = []
        node = self
        while node.parent:
            if node.action:
                actions.append(node.action)
            node = node.parent
        return list(reversed(actions))

    def states(self) -> List[T]:
        states = []
        node = self
        while node:
            states.append(node.state)
            node = node.parent
        return list(reversed(states))


@dataclass
class SearchResult:
    success: bool
    node: Optional[SearchNode[State]] = None
    nodes_expanded: int = 0
    error: Optional[str] = None


class AStarSearch:
    def __init__(
        self,
        heuristic: Callable[[State, Problem], float],
        max_nodes: int = 10000,
        weight: float = 1.0,
    ):
        self.heuristic = heuristic
        self.max_nodes = max_nodes
        self.weight = weight

    def search(self, problem: Problem) -> SearchResult:
        start_time = time.time()
        open_set: List[SearchNode[State]] = []
        closed_set: Set[State] = set()

        start_node = SearchNode(
            f_cost=self.weight * self.heuristic(problem.init, problem),
            state=problem.init,
            g_cost=0.0,
        )
        heapq.heappush(open_set, start_node)
        nodes_expanded = 0

        while open_set and nodes_expanded < self.max_nodes:
            if time.time() - start_time > 30.0:
                return SearchResult(
                    False, nodes_expanded=nodes_expanded, error="timeout"
                )

            current = heapq.heappop(open_set)

            if current.state in closed_set:
                continue

            if problem.is_goal(current.state):
                return SearchResult(True, node=current, nodes_expanded=nodes_expanded)

            closed_set.add(current.state)
            nodes_expanded += 1

            for action in self._get_applicable_actions(current.state, problem):
                next_state = action(current.state)
                if next_state in closed_set:
                    continue

                g = current.g_cost + action.cost
                h = self.heuristic(next_state, problem)
                f = g + self.weight * h

                child = SearchNode(
                    f_cost=f,
                    state=next_state,
                    g_cost=g,
                    action=action,
                    parent=current,
                )
                heapq.heappush(open_set, child)

        return SearchResult(
            False, nodes_expanded=nodes_expanded, error="max nodes reached"
        )

    def _get_applicable_actions(
        self, state: State, problem: Problem
    ) -> List[GroundAction]:
        actions = []
        for op in problem.domain.operators.values():
            for objects in self._get_groundings(op, problem):
                action = op.ground(*objects)
                if action.is_applicable(state):
                    actions.append(action)
        return actions

    def _get_groundings(self, op: Operator, problem: Problem) -> List[tuple]:
        from itertools import product

        param_types = [t for _, t in op.schema.parameters]
        candidates = []
        for t in param_types:
            candidates.append(
                [obj for obj in problem.objects if obj.type.is_subtype(t)]
            )
        return list(product(*candidates))
