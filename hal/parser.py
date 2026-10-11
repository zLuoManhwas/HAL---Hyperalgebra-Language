from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .ast import Cap, Cup, Expr, Identity, Merge, Morphism, Parallel, Sequential, Split, Swap
from .types import BaseType, I, Type, TypeVariable, tensor


class HALSyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class Token:
    value: str
    offset: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    while i < len(source):
        char = source[i]
        if char.isspace():
            i += 1
            continue
        if source.startswith("//", i) or source.startswith("--", i):
            end = source.find("\n", i)
            i = len(source) if end < 0 else end + 1
            continue
        if source.startswith("->", i):
            tokens.append(Token("->", i))
            i += 2
            continue
        if char in "#@():,":
            tokens.append(Token(char, i))
            i += 1
            continue
        if char == "⊗":
            tokens.append(Token("@", i))
            i += 1
            continue
        if char.isalpha() or char == "_":
            start = i
            i += 1
            while i < len(source) and (source[i].isalnum() or source[i] == "_"):
                i += 1
            tokens.append(Token(source[start:i], start))
            continue
        raise HALSyntaxError(f"Unexpected character {char!r} at offset {i}")
    return tokens


class Parser:
    def __init__(
        self,
        source: str,
        signatures: Mapping[str, tuple[Type, Type]] | None = None,
    ) -> None:
        self.source = source
        self.tokens = tokenize(source)
        self.position = 0
        self.signatures = dict(signatures or {})

    def _peek(self) -> str | None:
        return self.tokens[self.position].value if self.position < len(self.tokens) else None

    def _take(self, expected: str | None = None) -> str:
        if self.position >= len(self.tokens):
            raise HALSyntaxError("Unexpected end of HAL expression")
        token = self.tokens[self.position]
        if expected is not None and token.value != expected:
            raise HALSyntaxError(
                f"Expected {expected!r} at offset {token.offset}, got {token.value!r}"
            )
        self.position += 1
        return token.value

    def _type(self) -> Type:
        if self._peek() == "(":
            self._take("(")
            factors = [self._type()]
            while self._peek() == "@":
                self._take("@")
                factors.append(self._type())
            self._take(")")
            return tensor(*factors)
        token = self._peek()
        if token is None:
            raise HALSyntaxError("Expected a type")
        self._take()
        if token == "I":
            return I
        if token[0].islower():
            return TypeVariable(token)
        return BaseType(token)

    def _atom(self) -> Expr:
        token = self._peek()
        if token is None:
            raise HALSyntaxError("Expected an expression")
        if token == "(":
            self._take("(")
            expression = self._expression()
            self._take(")")
            return expression
        if token in {"id", "split", "merge", "cup", "cap"}:
            operator = self._take()
            self._take("(")
            first_type = self._type()
            if operator == "swap":
                self._take(",")
                result = Swap(first_type, self._type())
            else:
                result = {
                    "id": Identity,
                    "split": Split,
                    "merge": Merge,
                    "cup": Cup,
                    "cap": Cap,
                }[operator](first_type)
            self._take(")")
            return result
        if token == "swap":
            self._take("swap")
            self._take("(")
            left = self._type()
            self._take(",")
            right = self._type()
            self._take(")")
            return Swap(left, right)

        name = self._take()
        if self._peek() == ":":
            self._take(":")
            dom = self._type()
            self._take("->")
            cod = self._type()
            self.signatures[name] = (dom, cod)
            return Morphism(name, dom, cod)
        if name not in self.signatures:
            raise HALSyntaxError(
                f"Unknown morphism {name!r}; declare it as 'name: A -> B' "
                "or provide it in signatures"
            )
        dom, cod = self.signatures[name]
        return Morphism(name, dom, cod)

    def _parallel(self) -> Expr:
        expression = self._atom()
        while self._peek() == "@":
            self._take("@")
            expression = Parallel(expression, self._atom())
        return expression

    def _expression(self) -> Expr:
        expression = self._parallel()
        while self._peek() == "#":
            self._take("#")
            expression = Sequential(expression, self._parallel())
        return expression

    def parse(self) -> Expr:
        if not self.tokens:
            raise HALSyntaxError("HAL expression is empty")
        expression = self._expression()
        if self.position != len(self.tokens):
            token = self.tokens[self.position]
            raise HALSyntaxError(f"Unexpected token {token.value!r} at offset {token.offset}")
        return expression


def parse_hal(
    source: str,
    signatures: Mapping[str, tuple[Type, Type]] | None = None,
) -> Expr:
    return Parser(source, signatures).parse()
