"""Domain and problem definitions."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, FrozenSet, Callable, Any

from .types import Type, TypeHierarchy, Object, Constant
from .predicates import Predicate, GroundPredicate, State, make_state
from .operators import Operator, OperatorSchema, GroundAction, Action, ContinuousAction


@dataclass
class Domain:
    name: str
    type_hierarchy: TypeHierarchy = field(default_factory=TypeHierarchy)
    predicates: Dict[str, Predicate] = field(default_factory=dict)
    operators: Dict[str, Operator] = field(default_factory=dict)
    constants: Dict[str, Constant] = field(default_factory=dict)

    def add_type(self, name: str, parent: str = "object") -> Type:
        return self.type_hierarchy.add_type(name, parent)

    def add_predicate(self, pred: Predicate) -> Predicate:
        self.predicates[pred.name] = pred
        return pred

    def add_operator(self, op: Operator) -> Operator:
        self.operators[op.schema.name] = op
        return op

    def add_constant(self, const: Constant) -> Constant:
        self.constants[const.name] = const
        return const

    def get_predicate(self, name: str) -> Optional[Predicate]:
        return self.predicates.get(name)

    def get_operator(self, name: str) -> Optional[Operator]:
        return self.operators.get(name)

    def get_type(self, name: str) -> Optional[Type]:
        return self.type_hierarchy.get_type(name)


@dataclass
class Problem:
    domain: Domain
    name: str
    objects: Tuple[Object, ...] = ()
    init: State = field(default_factory=State)
    goal: FrozenSet[GroundPredicate] = field(default_factory=frozenset)
    metric: Optional[Callable[[State, State], float]] = None

    def __post_init__(self):
        obj_set = set(self.objects)
        for pred in self.init:
            for obj in pred.objects:
                if obj not in obj_set:
                    raise ValueError(f"Object {obj} in init not in objects")
        for pred in self.goal:
            for obj in pred.objects:
                if obj not in obj_set:
                    raise ValueError(f"Object {obj} in goal not in objects")

    def is_goal(self, state: State) -> bool:
        return all(state.holds(gp) for gp in self.goal)

    def __repr__(self) -> str:
        return f"Problem({self.name}, objects={self.objects}, goal={self.goal})"


def create_domain(name: str) -> Domain:
    return Domain(name=name)