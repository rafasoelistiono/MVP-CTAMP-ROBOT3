"""Predicate definitions and state representation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, FrozenSet

from .types import Type, Object


@dataclass(frozen=True)
class PredicateSignature:
    name: str
    param_types: Tuple[Type, ...]

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(t.name for t in self.param_types)})"

    def __hash__(self) -> int:
        return hash((self.name, self.param_types))


class Predicate:
    def __init__(
        self,
        name: str,
        param_types: Tuple[Type, ...],
        is_static: bool = False,
        is_derived: bool = False,
    ):
        self.signature = PredicateSignature(name, param_types)
        self.is_static = is_static
        self.is_derived = is_derived

    def __call__(self, *objects: Object) -> GroundPredicate:
        if len(objects) != len(self.signature.param_types):
            raise ValueError(
                f"Predicate {self.name} expects {len(self.signature.param_types)} args"
            )
        for obj, expected_type in zip(objects, self.signature.param_types):
            if not obj.type.is_subtype(expected_type):
                raise TypeError(
                    f"Object {obj} has type {obj.type}, expected {expected_type}"
                )
        return GroundPredicate(self, objects)

    @property
    def name(self) -> str:
        return self.signature.name

    @property
    def param_types(self) -> Tuple[Type, ...]:
        return self.signature.param_types

    def __repr__(self) -> str:
        return f"Predicate({self.signature})"

    def __hash__(self) -> int:
        return hash(self.signature)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Predicate) and self.signature == other.signature


@dataclass(frozen=True)
class GroundPredicate:
    predicate: Predicate
    objects: Tuple[Object, ...]

    @property
    def name(self) -> str:
        return self.predicate.name

    def __repr__(self) -> str:
        return f"{self.predicate.name}({', '.join(o.name for o in self.objects)})"

    def __hash__(self) -> int:
        return hash((self.predicate, self.objects))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GroundPredicate)
            and self.predicate == other.predicate
            and self.objects == other.objects
        )


class State:
    def __init__(
        self,
        predicates: Optional[FrozenSet[GroundPredicate]] = None,
        static_predicates: Optional[FrozenSet[GroundPredicate]] = None,
        objects: Optional[Tuple[Object, ...]] = None,
    ):
        self._predicates = predicates or frozenset()
        self._static_predicates = static_predicates or frozenset()
        self._objects = objects or tuple()

    @property
    def predicates(self) -> FrozenSet[GroundPredicate]:
        return self._predicates

    @property
    def static_predicates(self) -> FrozenSet[GroundPredicate]:
        return self._static_predicates

    @property
    def objects(self) -> Tuple[Object, ...]:
        return self._objects

    def holds(self, pred: GroundPredicate) -> bool:
        return pred in self._predicates or pred in self._static_predicates

    def add(self, pred: GroundPredicate) -> "State":
        if pred in self._static_predicates:
            return self
        return State(self._predicates | {pred}, self._static_predicates, self._objects)

    def remove(self, pred: GroundPredicate) -> "State":
        if pred in self._static_predicates:
            return self
        return State(self._predicates - {pred}, self._static_predicates, self._objects)

    def with_objects(self, objects: Tuple[Object, ...]) -> "State":
        return State(self._predicates, self._static_predicates, objects)

    def with_static(self, static: FrozenSet[GroundPredicate]) -> "State":
        return State(self._predicates, static, self._objects)

    def __contains__(self, pred: object) -> bool:
        if isinstance(pred, str):
            return any(p.name == pred for p in self._predicates)
        return pred in self._predicates or pred in self._static_predicates

    def __repr__(self) -> str:
        preds = sorted(self._predicates, key=str)
        statics = sorted(self._static_predicates, key=str)
        return f"State(predicates={preds}, static={statics})"

    def __hash__(self) -> int:
        return hash((self._predicates, self._static_predicates, self._objects))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return False
        return (
            self._predicates == other._predicates
            and self._static_predicates == other._static_predicates
            and self._objects == other._objects
        )

    def __len__(self) -> int:
        return len(self._predicates) + len(self._static_predicates)

    def __iter__(self):
        yield from self._predicates
        yield from self._static_predicates


def make_state(
    predicates: List[GroundPredicate],
    static_predicates: Optional[List[GroundPredicate]] = None,
    objects: Optional[List[Object]] = None,
) -> State:
    return State(
        predicates=frozenset(predicates),
        static_predicates=frozenset(static_predicates or []),
        objects=tuple(objects or []),
    )
