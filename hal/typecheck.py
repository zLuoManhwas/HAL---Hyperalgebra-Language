from __future__ import annotations

from dataclasses import dataclass, field

from .ast import (
    Cap,
    Cup,
    Expr,
    Identity,
    Merge,
    Morphism,
    Parallel,
    Sequential,
    Split,
    Swap,
)
from .types import I, TensorType, Type, TypeVariable, substitute, tensor, type_variables


class TypeCheckError(TypeError):
    pass


@dataclass
class TypeChecker:
    substitutions: dict[TypeVariable, Type] = field(default_factory=dict)
    _fresh_index: int = 0

    def resolve(self, type_: Type) -> Type:
        if isinstance(type_, TypeVariable):
            replacement = self.substitutions.get(type_)
            if replacement is None:
                return type_
            resolved = self.resolve(replacement)
            self.substitutions[type_] = resolved
            return resolved
        if isinstance(type_, TensorType):
            return tensor(*(self.resolve(factor) for factor in type_.factors))
        return type_

    def _occurs(self, variable: TypeVariable, type_: Type) -> bool:
        type_ = self.resolve(type_)
        if variable == type_:
            return True
        if isinstance(type_, TensorType):
            return any(self._occurs(variable, factor) for factor in type_.factors)
        return False

    def unify(self, left: Type, right: Type) -> None:
        left, right = self.resolve(left), self.resolve(right)
        if left == right:
            return
        if isinstance(left, TypeVariable):
            if self._occurs(left, right):
                raise TypeCheckError(f"Recursive type constraint: {left} = {right}")
            self.substitutions[left] = right
            return
        if isinstance(right, TypeVariable):
            self.unify(right, left)
            return
        if isinstance(left, TensorType) and isinstance(right, TensorType):
            if len(left.factors) != len(right.factors):
                raise TypeCheckError(f"Tensor arity mismatch: {left} != {right}")
            for left_factor, right_factor in zip(left.factors, right.factors):
                self.unify(left_factor, right_factor)
            return
        raise TypeCheckError(f"Incompatible types: {left} != {right}")

    def infer(self, expr: Expr) -> tuple[Type, Type]:
        if isinstance(expr, Morphism):
            variables = type_variables(expr.dom) | type_variables(expr.cod)
            replacements = {
                variable: TypeVariable(f"{variable.name}${self._fresh_index}")
                for variable in variables
            }
            self._fresh_index += 1
            dom = substitute(expr.dom, replacements)
            cod = substitute(expr.cod, replacements)
            return dom, cod
        if isinstance(expr, Identity):
            return expr.obj, expr.obj
        if isinstance(expr, Split):
            return expr.obj, tensor(expr.obj, expr.obj)
        if isinstance(expr, Merge):
            return tensor(expr.obj, expr.obj), expr.obj
        if isinstance(expr, Cup):
            return I, expr.obj
        if isinstance(expr, Cap):
            return expr.obj, I
        if isinstance(expr, Swap):
            return tensor(expr.left_type, expr.right_type), tensor(
                expr.right_type, expr.left_type
            )
        if isinstance(expr, Parallel):
            left_dom, left_cod = self.infer(expr.left)
            right_dom, right_cod = self.infer(expr.right)
            return tensor(left_dom, right_dom), tensor(left_cod, right_cod)
        if isinstance(expr, Sequential):
            left_dom, left_cod = self.infer(expr.left)
            right_dom, right_cod = self.infer(expr.right)
            self.unify(left_cod, right_dom)
            return left_dom, right_cod
        raise TypeCheckError(f"Unsupported expression node: {type(expr).__name__}")

    def infer_resolved(self, expr: Expr) -> tuple[Type, Type]:
        dom, cod = self.infer(expr)
        return self.resolve(dom), self.resolve(cod)


def infer_type(expr: Expr) -> tuple[Type, Type]:
    return TypeChecker().infer_resolved(expr)


def validate(expr: Expr) -> tuple[Type, Type]:
    return infer_type(expr)
