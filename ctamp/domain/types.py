"""Type hierarchy and object definitions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


class Type:
    def __init__(self, name: str, parent: Optional["Type"] = None):
        self.name = name
        self.parent = parent
        self.children: List[Type] = []
        if parent:
            parent.children.append(self)

    def is_subtype(self, other: "Type") -> bool:
        current = self
        while current:
            if current == other:
                return True
            current = current.parent
        return False

    def __repr__(self) -> str:
        return f"Type({self.name})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Type) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class TypeHierarchy:
    def __init__(self):
        self.types: Dict[str, Type] = {}
        self.root = Type("object")
        self.types["object"] = self.root

    def add_type(self, name: str, parent_name: str = "object") -> Type:
        parent = self.types.get(parent_name, self.root)
        t = Type(name, parent)
        self.types[name] = t
        return t

    def get_type(self, name: str) -> Optional[Type]:
        return self.types.get(name)

    def is_subtype(self, child: str, parent: str) -> bool:
        child_t = self.get_type(child)
        parent_t = self.get_type(parent)
        if not child_t or not parent_t:
            return False
        return child_t.is_subtype(parent_t)


@dataclass
class Object:
    name: str
    type: Type
    parent: Optional["Object"] = None

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Object) and self.name == other.name


@dataclass
class Constant:
    name: str
    type: Type
    value: object


OBJECT = Type("object")
