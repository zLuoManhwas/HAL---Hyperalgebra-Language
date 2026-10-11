from __future__ import annotations

from dataclasses import dataclass

from .types import Type


class Expr:
    def __str__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Morphism(Expr):
    name: str
    dom: Type
    cod: Type

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Identity(Expr):
    obj: Type

    def __str__(self) -> str:
        return f"id({self.obj})"


@dataclass(frozen=True)
class Sequential(Expr):
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"({self.left} # {self.right})"


@dataclass(frozen=True)
class Parallel(Expr):
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"({self.left} @ {self.right})"


@dataclass(frozen=True)
class Swap(Expr):
    left_type: Type
    right_type: Type

    def __str__(self) -> str:
        return f"swap({self.left_type}, {self.right_type})"


@dataclass(frozen=True)
class Split(Expr):
    obj: Type

    def __str__(self) -> str:
        return f"split({self.obj})"


@dataclass(frozen=True)
class Merge(Expr):
    obj: Type

    def __str__(self) -> str:
        return f"merge({self.obj})"


@dataclass(frozen=True)
class Cup(Expr):
    obj: Type

    def __str__(self) -> str:
        return f"cup({self.obj})"


@dataclass(frozen=True)
class Cap(Expr):
    obj: Type

    def __str__(self) -> str:
        return f"cap({self.obj})"
