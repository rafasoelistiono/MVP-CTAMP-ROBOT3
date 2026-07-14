"""Search algorithms for combined task-motion planning."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Set
from abc import ABC, abstractmethod
import heapq
import time


@dataclass(order=True)
class SearchNode:
    f: float
    g: float = field(compare=False)
    state: Any = field(compare=False)
    parent: Optional[SearchNode] = field(default=None, compare=False)
    action: Any = field(default=None, compare=False)

    def path_actions(self) -> List[Any]:
        actions = []
        node = self
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))

    def path_states(self) -> List[Any]:
        states = []
        node = self
        while node is not None:
            states.append(node.state)
            node = node.parent
        return list(reversed(states))


@dataclass
class SearchResult:
    success: bool
    goal_node: Optional[SearchNode] = None
    nodes_expanded: int = 0
    nodes_generated: int = 0
    time_elapsed: float = 0.0
    error: Optional[str] = None

    @property
    def actions(self) -> List[Any]:
        if self.goal_node:
            return self.goal_node.path_actions()
        return []


class Heuristic(ABC):
    @abstractmethod
    def evaluate(self, state: Any) -> float:
        pass


class ZeroHeuristic(Heuristic):
    def evaluate(self, state: Any) -> float:
        return 0.0


class AdmissibleHeuristic(Heuristic):
    def __init__(self, h: Callable[[Any], float]):
        self._h = h

    def evaluate(self, state: Any) -> float:
        return self._h(state)


@dataclass
class SearchConfig:
    max_nodes: int = 100000
    max_time: float = 60.0
    weight: float = 1.0


class SearchAlgorithm(ABC):
    @abstractmethod
    def search(
        self,
        initial: Any,
        is_goal: Callable[[Any], bool],
        get_successors: Callable[[Any], List[tuple]],
        heuristic: Optional[Heuristic] = None,
        config: Optional[SearchConfig] = None,
    ) -> SearchResult:
        pass


class AStarSearch(SearchAlgorithm):
    def search(
        self,
        initial: Any,
        is_goal: Callable[[Any], bool],
        get_successors: Callable[[Any], List[tuple]],
        heuristic: Optional[Heuristic] = None,
        config: Optional[SearchConfig] = None,
    ) -> SearchResult:
        config = config or SearchConfig()
        h = heuristic or ZeroHeuristic()

        start_time = time.time()
        start_node = SearchNode(f=h.evaluate(initial), g=0.0, state=initial)
        open_set = [start_node]
        closed_set: Set[int] = set()
        nodes_generated = 1
        nodes_expanded = 0

        while open_set:
            if time.time() - start_time > config.max_time:
                return SearchResult(
                    success=False,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_elapsed=time.time() - start_time,
                    error="timeout",
                )

            current = heapq.heappop(open_set)
            state_id = id(current.state)

            if state_id in closed_set:
                continue
            closed_set.add(state_id)

            if is_goal(current.state):
                return SearchResult(
                    success=True,
                    goal_node=current,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_elapsed=time.time() - start_time,
                )

            nodes_expanded += 1
            if nodes_expanded >= config.max_nodes:
                return SearchResult(
                    success=False,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_elapsed=time.time() - start_time,
                    error="max_nodes",
                )

            for action, next_state, cost in get_successors(current.state):
                next_id = id(next_state)
                if next_id in closed_set:
                    continue
                g = current.g + cost
                f = g + config.weight * h.evaluate(next_state)
                child = SearchNode(
                    f=f, g=g, state=next_state, parent=current, action=action
                )
                heapq.heappush(open_set, child)
                nodes_generated += 1

        return SearchResult(
            success=False,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            time_elapsed=time.time() - start_time,
            error="no_solution",
        )


class BeamSearch(SearchAlgorithm):
    def __init__(self, beam_width: int = 10):
        self.beam_width = beam_width

    def search(
        self,
        initial: Any,
        is_goal: Callable[[Any], bool],
        get_successors: Callable[[Any], List[tuple]],
        heuristic: Optional[Heuristic] = None,
        config: Optional[SearchConfig] = None,
    ) -> SearchResult:
        config = config or SearchConfig()
        h = heuristic or ZeroHeuristic()
        start_time = time.time()

        current_beam = [SearchNode(f=h.evaluate(initial), g=0.0, state=initial)]
        nodes_generated = 1
        nodes_expanded = 0

        while current_beam:
            if time.time() - start_time > config.max_time:
                return SearchResult(
                    success=False,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_elapsed=time.time() - start_time,
                    error="timeout",
                )

            next_beam = []
            for node in current_beam:
                if is_goal(node.state):
                    return SearchResult(
                        success=True,
                        goal_node=node,
                        nodes_expanded=nodes_expanded,
                        nodes_generated=nodes_generated,
                        time_elapsed=time.time() - start_time,
                    )

                nodes_expanded += 1
                for action, next_state, cost in get_successors(node.state):
                    g = node.g + cost
                    f = g + h.evaluate(next_state)
                    child = SearchNode(
                        f=f, g=g, state=next_state, parent=node, action=action
                    )
                    next_beam.append(child)
                    nodes_generated += 1

            next_beam.sort(key=lambda n: n.f)
            current_beam = next_beam[: self.beam_width]

        return SearchResult(
            success=False,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            time_elapsed=time.time() - start_time,
            error="no_solution",
        )


class GreedySearch(SearchAlgorithm):
    def search(
        self,
        initial: Any,
        is_goal: Callable[[Any], bool],
        get_successors: Callable[[Any], List[tuple]],
        heuristic: Optional[Heuristic] = None,
        config: Optional[SearchConfig] = None,
    ) -> SearchResult:
        config = config or SearchConfig()
        h = heuristic or ZeroHeuristic()
        start_time = time.time()

        open_set = [SearchNode(f=h.evaluate(initial), g=0.0, state=initial)]
        closed_set: Set[int] = set()
        nodes_generated = 1
        nodes_expanded = 0

        while open_set:
            if time.time() - start_time > config.max_time:
                return SearchResult(
                    success=False,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_elapsed=time.time() - start_time,
                    error="timeout",
                )

            current = heapq.heappop(open_set)
            state_id = id(current.state)
            if state_id in closed_set:
                continue
            closed_set.add(state_id)

            if is_goal(current.state):
                return SearchResult(
                    success=True,
                    goal_node=current,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_elapsed=time.time() - start_time,
                )

            nodes_expanded += 1
            for action, next_state, cost in get_successors(current.state):
                if id(next_state) in closed_set:
                    continue
                child = SearchNode(
                    f=h.evaluate(next_state),
                    g=current.g + cost,
                    state=next_state,
                    parent=current,
                    action=action,
                )
                heapq.heappush(open_set, child)
                nodes_generated += 1

        return SearchResult(
            success=False,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            time_elapsed=time.time() - start_time,
            error="no_solution",
        )


from .heuristic_estimator import HeuristicPathEstimator
from .baseline import BaselinePlanner, BaselineResult
from .tmm_astar import (
    TMMHeuristic,
    ZeroTMMHeuristic,
    GoalDistanceHeuristic,
    SearchVisitor,
    MockVisitor,
    TMMAStar,
    TMMSearchResult,
    TMMEdgeCostCalculator,
)
from .motion_visitor import (
    MotionPlanningVisitor,
    LearningSample,
    LearningMode,
    OfflineLearning,
    OnlineLearning,
)

__all__ = [
    "SearchNode",
    "SearchResult",
    "Heuristic",
    "ZeroHeuristic",
    "AdmissibleHeuristic",
    "SearchConfig",
    "SearchAlgorithm",
    "AStarSearch",
    "BeamSearch",
    "GreedySearch",
    "HeuristicPathEstimator",
    "BaselinePlanner",
    "BaselineResult",
    "TMMHeuristic",
    "ZeroTMMHeuristic",
    "GoalDistanceHeuristic",
    "SearchVisitor",
    "MockVisitor",
    "TMMAStar",
    "TMMSearchResult",
    "TMMEdgeCostCalculator",
    "MotionPlanningVisitor",
    "LearningSample",
    "LearningMode",
    "OfflineLearning",
    "OnlineLearning",
]
