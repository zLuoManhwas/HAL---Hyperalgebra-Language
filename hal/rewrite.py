from __future__ import annotations

from dataclasses import dataclass

from .ast import (
    Cap,
    Cup,
    Expr,
    Identity,
    Merge,
    Parallel,
    Sequential,
    Split,
    Swap,
)
from .typecheck import infer_type
from .types import Type, tensor


@dataclass(frozen=True)
class RewriteRule:
    name: str
    description: str


RULES = (
    RewriteRule("sequential_identity", "id # f = f and f # id = f"),
    RewriteRule("parallel_identity", "id(A) @ id(B) = id(A @ B)"),
    RewriteRule("associativity", "Sequential and tensor products are associative"),
    RewriteRule("commutativity", "merge # swap = merge and swap # split = split"),
    RewriteRule("symmetry_involution", "swap(A, B) # swap(B, A) = id(A @ B)"),
    RewriteRule("frobenius", "Both Frobenius composites reduce to merge # split"),
    RewriteRule("special", "split # merge = id"),
    RewriteRule("unit", "cup @ id followed by merge is id; symmetric variant too"),
    RewriteRule("counit", "split followed by cap @ id or id @ cap is id"),
    RewriteRule("coherence", "Nested tensor products use canonical flattened types"),
)


def _sequence_parts(expr: Expr) -> list[Expr]:
    if isinstance(expr, Sequential):
        return _sequence_parts(expr.left) + _sequence_parts(expr.right)
    return [expr]


def _parallel_parts(expr: Expr) -> list[Expr]:
    if isinstance(expr, Parallel):
        return _parallel_parts(expr.left) + _parallel_parts(expr.right)
    return [expr]


def _seq(parts: list[Expr]) -> Expr:
    if not parts:
        raise ValueError("Cannot form a sequence without an identity type")
    result = parts[0]
    for item in parts[1:]:
        result = Sequential(result, item)
    return result


def _par(parts: list[Expr], identity_type: Type) -> Expr:
    if not parts:
        return Identity(identity_type)
    if len(parts) == 1:
        return parts[0]
    result = parts[0]
    for item in parts[1:]:
        result = Parallel(result, item)
    return result


def _same_type(left: Type, right: Type) -> bool:
    return left == right


def _reduce_sequence(parts: list[Expr]) -> list[Expr]:
    changed = True
    while changed:
        changed = False

        # Composition unit, specialness, commutativity, Frobenius, associativity,
        # and the Frobenius algebra's unit/counit laws.
        index = 0
        while index < len(parts):
            item = parts[index]
            if isinstance(item, Identity):
                parts.pop(index)
                changed = True
                continue

            if index + 1 < len(parts):
                following = parts[index + 1]
                replacement: Expr | None = None

                if isinstance(item, Split) and isinstance(following, Merge):
                    if _same_type(item.obj, following.obj):
                        replacement = Identity(item.obj)

                if isinstance(item, Swap) and isinstance(following, Merge):
                    if _same_type(item.left_type, item.right_type) and _same_type(
                        following.obj, item.left_type
                    ):
                        replacement = following

                if isinstance(item, Swap) and isinstance(following, Swap):
                    if (
                        item.left_type == following.right_type
                        and item.right_type == following.left_type
                    ):
                        replacement = Identity(tensor(item.left_type, item.right_type))

                if isinstance(item, Split) and isinstance(following, Swap):
                    if _same_type(item.obj, following.left_type) and _same_type(
                        item.obj, following.right_type
                    ):
                        replacement = item

                # (id @ split);(merge @ id) and its mirror are the two
                # non-central Frobenius composites.
                if isinstance(item, Parallel) and isinstance(following, Parallel):
                    first = _parallel_parts(item)
                    second = _parallel_parts(following)
                    if len(first) == len(second) == 2:
                        if (
                            isinstance(first[0], Identity)
                            and isinstance(first[1], Split)
                            and isinstance(second[0], Merge)
                            and isinstance(second[1], Identity)
                        ):
                            x = first[0].obj
                            if (
                                first[1].obj == x
                                and second[0].obj == x
                                and second[1].obj == x
                            ):
                                replacement = Sequential(Merge(x), Split(x))
                        elif (
                            isinstance(first[0], Split)
                            and isinstance(first[1], Identity)
                            and isinstance(second[0], Identity)
                            and isinstance(second[1], Merge)
                        ):
                            x = first[0].obj
                            if (
                                first[1].obj == x
                                and second[0].obj == x
                                and second[1].obj == x
                            ):
                                replacement = Sequential(Merge(x), Split(x))

                if replacement is not None:
                    parts[index : index + 2] = [replacement]
                    changed = True
                    continue

            # Associativity/coassociativity map both parenthesizations to one
            # left-associated canonical spelling.
            if index + 1 < len(parts):
                following = parts[index + 1]
                first = _parallel_parts(item) if isinstance(item, Parallel) else []
                if isinstance(item, Parallel) and isinstance(following, Merge) and len(first) == 2:
                    left, right = first
                    if isinstance(left, Identity) and isinstance(right, Merge):
                        if left.obj == right.obj == following.obj:
                            parts[index : index + 2] = [
                                Parallel(Merge(following.obj), Identity(following.obj)),
                                following,
                            ]
                            changed = True
                            continue

                if isinstance(item, Split) and isinstance(following, Parallel):
                    second = _parallel_parts(following)
                    if len(second) == 2:
                        left, right = second
                        if isinstance(left, Identity) and isinstance(right, Split):
                            if item.obj == left.obj == right.obj:
                                parts[index : index + 2] = [
                                    Split(item.obj),
                                    Parallel(Split(item.obj), Identity(item.obj)),
                                ]
                                changed = True
                                continue

            # Frobenius unit: eta tensor id then multiplication, or mirror.
            if index + 1 < len(parts) and isinstance(item, Parallel):
                following = parts[index + 1]
                operands = _parallel_parts(item)
                if len(operands) == 2 and isinstance(following, Merge):
                    left, right = operands
                    if (
                        isinstance(left, Cup)
                        and isinstance(right, Identity)
                        and left.obj == right.obj == following.obj
                    ) or (
                        isinstance(left, Identity)
                        and isinstance(right, Cup)
                        and left.obj == right.obj == following.obj
                    ):
                        parts[index : index + 2] = [Identity(following.obj)]
                        changed = True
                        continue

            # Frobenius counit: split followed by cap tensor id, or its mirror.
            if index + 1 < len(parts) and isinstance(item, Split):
                following = parts[index + 1]
                if isinstance(following, Parallel):
                    operands = _parallel_parts(following)
                    if len(operands) == 2:
                        left, right = operands
                        if (
                            isinstance(left, Cap)
                            and isinstance(right, Identity)
                            and left.obj == right.obj == item.obj
                        ) or (
                            isinstance(left, Identity)
                            and isinstance(right, Cap)
                            and left.obj == right.obj == item.obj
                        ):
                            parts[index : index + 2] = [Identity(item.obj)]
                            changed = True
                            continue

            index += 1

    return parts


def rewrite(expr: Expr, *, max_steps: int = 10_000) -> Expr:
    """Normalize by a terminating oriented subset of the Frobenius equations.

    Equations are oriented to a fixed spelling. This does not claim a complete
    decision procedure for equality in the free hypergraph category.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    domain, _ = infer_type(expr)
    remaining_steps = [max_steps]

    def normalize(current: Expr) -> Expr:
        remaining_steps[0] -= 1
        if remaining_steps[0] < 0:
            raise RuntimeError(
                f"Rewrite did not reach a fixed point within {max_steps} steps"
            )
        if isinstance(current, Sequential):
            normalized_parts = [normalize(part) for part in _sequence_parts(current)]
            reduced = _reduce_sequence(normalized_parts)
            if not reduced:
                return Identity(domain)
            # Frobenius rewrites may produce a sequence in a single slot.
            flattened: list[Expr] = []
            for part in reduced:
                flattened.extend(_sequence_parts(part))
            if flattened != reduced:
                reduced = _reduce_sequence(flattened)
            if not reduced:
                return Identity(domain)
            result = _seq(reduced)
            return result
        if isinstance(current, Parallel):
            parts = [normalize(part) for part in _parallel_parts(current)]
            if all(isinstance(part, Identity) for part in parts):
                return Identity(tensor(*(part.obj for part in parts if isinstance(part, Identity))))
            return _par(parts, domain)
        return current

    return normalize(expr)


def simplify(expr: Expr) -> Expr:
    return rewrite(expr)
