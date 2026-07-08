"""Task-Motion integration: TMM graph and state space."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple, Any
from abc import ABC, abstractmethod
from enum import Enum
import networkx as nx

from ..domain import State, GroundAction


@dataclass
class TMMState:
    task_state: State
    config: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(frozenset(self.task_state.predicates))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TMMState):
            return False
        return self.task_state.predicates == other.task_state.predicates


@dataclass
class TMMEdge:
    action: Optional[GroundAction] = None
    cost: float = 1.0
    motion_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TMM:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.states: Dict[int, TMMState] = {}
        self._next_id = 0

    def add_state(self, state: TMMState) -> int:
        sid = self._next_id
        self._next_id += 1
        self.states[sid] = state
        self.graph.add_node(sid)
        return sid

    def add_transition(self, from_id: int, to_id: int, edge: TMMEdge) -> None:
        self.graph.add_edge(from_id, to_id, edge=edge)

    def get_state(self, state_id: int) -> Optional[TMMState]:
        return self.states.get(state_id)


class TMMBuilder(ABC):
    @abstractmethod
    def build(self, initial: TMMState) -> TMM:
        pass


class SimpleTMMBuilder(TMMBuilder):
    def build(self, initial: TMMState) -> TMM:
        tmm = TMM()
        tmm.add_state(initial)
        return tmm


class TMMPlanner:
    def __init__(self, tmm: Optional[TMM] = None):
        self.tmm = tmm or TMM()

    def search(self, initial: TMMState, goal_check: Any = None) -> Optional[List[int]]:
        if not self.tmm.states:
            return None
        return [0]


from .multigraph import TaskMotionMultigraph
from .builder import TMMGraphBuilder

__all__ = [
    "TMMState",
    "TMMEdge",
    "TMM",
    "TMMBuilder",
    "SimpleTMMBuilder",
    "TMMPlanner",
    "TaskMotionMultigraph",
    "TMMGraphBuilder",
]
