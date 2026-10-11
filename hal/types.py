from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeAlias


@dataclass(frozen=True)
class BaseType:
    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A base type must have a non-empty name")

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TypeVariable:
    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A type variable must have a non-empty name")

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TensorType:
    factors: tuple[Type, ...]

    def __post_init__(self) -> None:
        flattened: list[Type] = []
        for factor in self.factors:
            if isinstance(factor, UnitType):
                continue
            if isinstance(factor, TensorType):
                flattened.extend(factor.factors)
            else:
                flattened.append(factor)
        if len(flattened) < 2:
            raise ValueError("TensorType must contain at least two factors")
        object.__setattr__(self, "factors", tuple(flattened))

    def __str__(self) -> str:
        return "(" + " @ ".join(map(str, self.factors)) + ")"


@dataclass(frozen=True)
class UnitType:
    def __str__(self) -> str:
        return "I"


Type: TypeAlias = BaseType | TypeVariable | TensorType | UnitType
I = UnitType()


def tensor(*types: Type) -> Type:
    factors: list[Type] = []
    for item in types:
        if isinstance(item, UnitType):
            continue
        if isinstance(item, TensorType):
            factors.extend(item.factors)
        else:
            factors.append(item)
    if not factors:
        return I
    if len(factors) == 1:
        return factors[0]
    return TensorType(tuple(factors))


def type_variables(type_: Type) -> set[TypeVariable]:
    if isinstance(type_, TypeVariable):
        return {type_}
    if isinstance(type_, TensorType):
        return set().union(*(type_variables(factor) for factor in type_.factors))
    return set()


def substitute(type_: Type, mapping: dict[TypeVariable, Type]) -> Type:
    if isinstance(type_, TypeVariable):
        replacement = mapping.get(type_, type_)
        return substitute(replacement, mapping) if replacement != type_ else type_
    if isinstance(type_, TensorType):
        return tensor(*(substitute(factor, mapping) for factor in type_.factors))
    return type_


def make_tensor(factors: Iterable[Type]) -> Type:
    return tensor(*tuple(factors))
