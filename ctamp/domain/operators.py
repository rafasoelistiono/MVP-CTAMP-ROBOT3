"""Operator and action definitions."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, FrozenSet, Callable, Any
from abc import ABC, abstractmethod

from .types import Type, TypeHierarchy, Object
from .predicates import Predicate, GroundPredicate, State


@dataclass(frozen=True)
class OperatorSchema:
    name: str
    parameters: Tuple[Tuple[str, Type], ...]
    preconditions: Tuple[Predicate, ...]
    add_effects: Tuple[Predicate, ...]
    del_effects: Tuple[Predicate, ...]
    cost: float = 1.0

    def __repr__(self) -> str:
        params = ", ".join(f"{n}: {t.name}" for n, t in self.parameters)
        return f"OperatorSchema({self.name}({params}))"

    def ground(self, objects: Tuple[Object, ...]) -> "GroundAction":
        if len(objects) != len(self.parameters):
            raise ValueError(f"Operator {self.name} expects {len(self.parameters)} args")
        for obj, (_, expected_type) in zip(objects, self.parameters):
            if not obj.type.is_subtype(expected_type):
                raise TypeError(f"Object {obj} has type {obj.type}, expected {expected_type}")
        return GroundAction(self, objects)

    def is_applicable(self, state: State, objects: Tuple[Object, ...]) -> bool:
        for pred in self.preconditions:
            gp = pred(*objects)
            if not state.holds(gp):
                return False
        return True

    def apply(self, state: State, objects: Tuple[Object, ...]) -> State:
        new_state = state
        for pred in self.del_effects:
            new_state = new_state.remove(pred(*objects))
        for pred in self.add_effects:
            new_state = new_state.add(pred(*objects))
        return new_state


@dataclass(frozen=True)
class GroundAction:
    schema: OperatorSchema
    objects: Tuple[Object, ...]

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def cost(self) -> float:
        return self.schema.cost

    def __call__(self, state: State) -> State:
        return self.schema.apply(state, self.objects)

    def is_applicable(self, state: State) -> bool:
        return self.schema.is_applicable(state, self.objects)

    def __repr__(self) -> str:
        objs = ", ".join(o.name for o in self.objects)
        return f"{self.name}({objs})"

    def __hash__(self) -> int:
        return hash((self.schema, self.objects))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GroundAction) and self.schema == other.schema and self.objects == other.objects


class Operator:
    def __init__(
        self,
        name: str,
        parameters: List[Tuple[str, Type]],
        preconditions: List[Predicate],
        add_effects: List[Predicate],
        del_effects: List[Predicate],
        cost: float = 1.0,
    ):
        self.schema = OperatorSchema(
            name=name,
            parameters=tuple(parameters),
            preconditions=tuple(preconditions),
            add_effects=tuple(add_effects),
            del_effects=tuple(del_effects),
            cost=cost,
        )

    def ground(self, *objects: Object) -> GroundAction:
        return self.schema.ground(objects)

    def __repr__(self) -> str:
        return str(self.schema)

    def __call__(self, state: State, *objects: Object) -> State:
        action = self.ground(*objects)
        return action(state)


class Action(ABC):
    @abstractmethod
    def is_applicable(self, state: State) -> bool:
        pass

    @abstractmethod
    def apply(self, state: State) -> State:
        pass

    @abstractmethod
    def cost(self) -> float:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class ContinuousAction(Action):
    def __init__(self, name: str, parameters: Tuple[float, ...], cost: float = 0.0):
        self.name = name
        self.parameters = parameters
        self._cost = cost

    def is_applicable(self, state: State) -> bool:
        return True

    def apply(self, state: State) -> State:
        return state

    def cost(self) -> float:
        return self._cost

    def __repr__(self) -> str:
        return f"{self.name}{self.parameters}"