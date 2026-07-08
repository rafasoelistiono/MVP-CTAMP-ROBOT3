"""Domain definition for task and motion planning."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, FrozenSet, Callable
from abc import ABC, abstractmethod
from enum import Enum


class PredicateType(Enum):
    SYMBOLIC = "symbolic"
    GEOMETRIC = "geometric"
    KINEMATIC = "kinematic"


@dataclass(frozen=True)
class Type:
    name: str
    parent: Optional["Type"] = None

    def is_subtype(self, other: "Type") -> bool:
        if self == other:
            return True
        if self.parent:
            return self.parent.is_subtype(other)
        return False


OBJECT = Type("object")
ROBOT = Type("robot", OBJECT)
LOCATION = Type("location", OBJECT)
OBJECT_TYPE = Type("physical_object", OBJECT)
GRASP = Type("grasp", OBJECT)


@dataclass(frozen=True)
class Predicate:
    name: str
    param_types: Tuple[Type, ...]
    pred_type: PredicateType = PredicateType.SYMBOLIC

    def __call__(self, *args) -> "GroundPredicate":
        return GroundPredicate(self, tuple(args))

    def matches(self, gp: "GroundPredicate") -> bool:
        return gp.predicate == self


@dataclass(frozen=True)
class GroundPredicate:
    predicate: Predicate
    args: Tuple[Any, ...]

    def __str__(self) -> str:
        return f"{self.predicate.name}({', '.join(str(a) for a in self.args)})"

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class Object:
    name: str
    type: Type

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.name}:{self.type.name}"


@dataclass(frozen=True)
class State:
    predicates: FrozenSet[GroundPredicate] = field(default_factory=frozenset)

    def __contains__(self, pred: GroundPredicate) -> bool:
        return pred in self.predicates

    def __add__(self, other: "State") -> "State":
        return State(self.predicates | other.predicates)

    def add(self, *preds: GroundPredicate) -> "State":
        return State(self.predicates | frozenset(preds))

    def remove(self, *preds: GroundPredicate) -> "State":
        return State(self.predicates - frozenset(preds))

    def __str__(self) -> str:
        return "{" + ", ".join(str(p) for p in self.predicates) + "}"

    def __repr__(self) -> str:
        return f"State({{ {', '.join(str(p) for p in self.predicates)} }})"


@dataclass
class Parameter:
    name: str
    type: Type


@dataclass
class Operator:
    name: str
    schema: "OperatorSchema"
    preconditions: List[GroundPredicate] = field(default_factory=list)
    add_effects: List[GroundPredicate] = field(default_factory=list)
    del_effects: List[GroundPredicate] = field(default_factory=list)
    cost: float = 1.0

    def ground(self, *objects: Object) -> "GroundAction":
        if len(objects) != len(self.schema.parameters):
            raise ValueError(f"Wrong number of objects for {self.name}")
        subs = dict(zip([p.name for p in self.schema.parameters], objects))

        def _ground_pred(gp: GroundPredicate) -> GroundPredicate:
            new_args = tuple(subs[a] if isinstance(a, str) and a in subs else a for a in gp.args)
            return GroundPredicate(gp.predicate, new_args)

        pre = [_ground_pred(p) for p in self.preconditions]
        add = [_ground_pred(p) for p in self.add_effects]
        delete = [_ground_pred(p) for p in self.del_effects]
        return GroundAction(self.name, pre, add, delete, self.cost, subs)

    def __str__(self) -> str:
        return f"Operator({self.name}, params={self.schema.parameters})"


@dataclass
class OperatorSchema:
    parameters: List[Parameter]


@dataclass
class GroundAction:
    name: str
    preconditions: List[GroundPredicate]
    add_effects: List[GroundPredicate]
    del_effects: List[GroundPredicate]
    cost: float
    substitution: Dict[str, Object]

    def is_applicable(self, state: State) -> bool:
        return all(p in state for p in self.preconditions)

    def __call__(self, state: State) -> State:
        if not self.is_applicable(state):
            raise ValueError("Action not applicable")
        return state.add(*self.add_effects).remove(*self.del_effects)

    def __str__(self) -> str:
        return f"{self.name}({', '.join(str(v) for v in self.substitution.values())})"


class Domain:
    def __init__(self, name: str):
        self.name = name
        self.types: Dict[str, Type] = {"object": OBJECT}
        self.predicates: Dict[str, Predicate] = {}
        self.operators: Dict[str, Operator] = {}

    def add_type(self, name: str, parent: Optional[str] = None) -> Type:
        if parent is None:
            parent = "object"
        parent_type = self.types.get(parent)
        t = Type(name, parent_type)
        self.types[name] = t
        return t

    def add_predicate(self, name: str, param_types: List[Type], pred_type: PredicateType = PredicateType.SYMBOLIC) -> Predicate:
        pred = Predicate(name, tuple(param_types), pred_type)
        self.predicates[name] = pred
        return pred

    def add_operator(self, op: Operator) -> None:
        self.operators[op.name] = op


@dataclass
class Problem:
    domain: Domain
    objects: List[Object] = field(default_factory=list)
    init: State = field(default_factory=State)
    goal: FrozenSet[GroundPredicate] = field(default_factory=frozenset)

    def is_goal(self, state: State) -> bool:
        return self.goal.issubset(state.predicates)


from .models import (
    Pose,
    Shape,
    ObjectState,
    RobotState,
    WorkspaceState,
    Action,
    JointSpace,
    MotionPlan,
    Vertex,
    Edge,
)

domain = Domain("ctamp")


def create_blocks_domain() -> Domain:
    d = Domain("blocks")
    d.add_type("block")
    d.add_type("hand")

    d.add_predicate("on", [d.types["block"], d.types["block"]])
    d.add_predicate("ontable", [d.types["block"]])
    d.add_predicate("clear", [d.types["block"]])
    d.add_predicate("holding", [d.types["hand"], d.types["block"]])
    d.add_predicate("handempty", [d.types["hand"]])

    on = d.predicates["on"]
    ontable = d.predicates["ontable"]
    clear = d.predicates["clear"]
    holding = d.predicates["holding"]
    handempty = d.predicates["handempty"]

    d.add_operator(Operator(
        name="pickup",
        schema=OperatorSchema([
            Parameter("h", d.types["hand"]),
            Parameter("b", d.types["block"]),
        ]),
        preconditions=[handempty("h"), clear("b"), ontable("b")],
        add_effects=[holding("h", "b")],
        del_effects=[handempty("h"), clear("b"), ontable("b")],
    ))

    d.add_operator(Operator(
        name="putdown",
        schema=OperatorSchema([
            Parameter("h", d.types["hand"]),
            Parameter("b", d.types["block"]),
        ]),
        preconditions=[holding("h", "b")],
        add_effects=[handempty("h"), clear("b"), ontable("b")],
        del_effects=[holding("h", "b")],
    ))

    d.add_operator(Operator(
        name="stack",
        schema=OperatorSchema([
            Parameter("h", d.types["hand"]),
            Parameter("b1", d.types["block"]),
            Parameter("b2", d.types["block"]),
        ]),
        preconditions=[holding("h", "b1"), clear("b2")],
        add_effects=[handempty("h"), on("b1", "b2"), clear("b1")],
        del_effects=[holding("h", "b1"), clear("b2")],
    ))

    d.add_operator(Operator(
        name="unstack",
        schema=OperatorSchema([
            Parameter("h", d.types["hand"]),
            Parameter("b1", d.types["block"]),
            Parameter("b2", d.types["block"]),
        ]),
        preconditions=[on("b1", "b2"), clear("b1"), handempty("h")],
        add_effects=[holding("h", "b1"), clear("b2")],
        del_effects=[on("b1", "b2"), handempty("h"), clear("b1")],
    ))

    return d


blocks_domain = create_blocks_domain()
