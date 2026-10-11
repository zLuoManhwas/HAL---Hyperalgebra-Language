from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import factorial, prod
from typing import Literal

import numpy as np

try:
    from scipy import sparse
except ImportError:  # SciPy is an optional sparse backend.
    sparse = None


MatrixFormat = Literal["dense", "csr", "auto"]


@dataclass(frozen=True)
class Hyperedge:
    name: str
    vertices: tuple[str, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices", tuple(self.vertices))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Hyperedge name must be non-empty")
        if any(not isinstance(vertex, str) or not vertex for vertex in self.vertices):
            raise ValueError("Hyperedge vertices must be non-empty strings")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError(f"Hyperedge {self.name!r} repeats a vertex")
        try:
            weight = float(self.weight)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Hyperedge {self.name!r} weight must be a real number") from error
        if not np.isfinite(weight):
            raise ValueError(f"Hyperedge {self.name!r} has a non-finite weight")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class CliqueProjection:
    gram: object
    adjacency: object
    degree: object


@dataclass(frozen=True)
class SparseTensorCOO:
    coordinates: np.ndarray
    values: np.ndarray
    shape: tuple[int, ...]

    def to_dense(self, *, max_elements: int = 10_000_000) -> np.ndarray:
        size = prod(self.shape)
        if size > max_elements:
            raise MemoryError(
                f"Dense tensor would contain {size:,} elements; limit is {max_elements:,}"
            )
        if not self.shape:
            return np.asarray(self.values.sum())
        result = np.zeros(self.shape, dtype=self.values.dtype)
        if self.coordinates.size:
            np.add.at(result, tuple(self.coordinates.T), self.values)
        return result


@dataclass(frozen=True)
class Hypergraph:
    vertices: tuple[str, ...]
    edges: tuple[Hyperedge, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices", tuple(self.vertices))
        object.__setattr__(self, "edges", tuple(self.edges))
        if any(not isinstance(vertex, str) or not vertex for vertex in self.vertices):
            raise ValueError("Vertex names must be non-empty strings")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("Vertex names must be unique")
        if any(not isinstance(edge, Hyperedge) for edge in self.edges):
            raise TypeError("edges must contain Hyperedge instances")
        vertex_set = set(self.vertices)
        edge_names = [edge.name for edge in self.edges]
        if len(set(edge_names)) != len(edge_names):
            raise ValueError("Hyperedge names must be unique")
        for edge in self.edges:
            unknown = set(edge.vertices) - vertex_set
            if unknown:
                raise ValueError(f"Hyperedge {edge.name!r} references unknown vertices: {sorted(unknown)}")

    @property
    def is_uniform(self) -> bool:
        return not self.edges or len({len(edge.vertices) for edge in self.edges}) == 1

    @property
    def uniform_order(self) -> int | None:
        if not self.is_uniform or not self.edges:
            return None
        return len(self.edges[0].vertices)

    @property
    def vertex_index(self) -> dict[str, int]:
        return {vertex: index for index, vertex in enumerate(self.vertices)}

    def incidence_matrix(self, format: MatrixFormat = "dense"):
        if format not in {"dense", "csr", "auto"}:
            raise ValueError(f"Unsupported incidence format: {format!r}")
        rows: list[int] = []
        cols: list[int] = []
        vertex_index = self.vertex_index
        for edge_index, edge in enumerate(self.edges):
            rows.extend(vertex_index[vertex] for vertex in edge.vertices)
            cols.extend([edge_index] * len(edge.vertices))
        values = np.ones(len(rows), dtype=np.float64)
        if format == "auto":
            format = "csr" if sparse is not None and len(rows) < len(self.vertices) * len(self.edges) // 2 else "dense"
        shape = (len(self.vertices), len(self.edges))
        if format == "csr":
            if sparse is None:
                raise ImportError("SciPy is required for CSR incidence matrices; install scipy")
            return sparse.coo_matrix((values, (rows, cols)), shape=shape).tocsr()
        result = np.zeros(shape, dtype=np.float64)
        if rows:
            result[rows, cols] = values
        return result

    def clique_projection(
        self,
        *,
        format: MatrixFormat = "dense",
        normalize: bool = False,
    ) -> CliqueProjection:
        incidence = self.incidence_matrix(format)
        weights = np.asarray([edge.weight for edge in self.edges], dtype=np.float64)
        if sparse is not None and sparse.issparse(incidence):
            weighted_incidence = incidence @ sparse.diags(weights)
            gram = weighted_incidence @ incidence.T
            degree = sparse.diags(gram.diagonal(), format="csr")
            adjacency = gram - degree
            if normalize:
                inv = np.zeros_like(gram.diagonal())
                nonzero = gram.diagonal() > 0
                if np.any(gram.diagonal() < 0):
                    raise ValueError("Degree normalization requires non-negative edge weights")
                inv[nonzero] = 1.0 / np.sqrt(gram.diagonal()[nonzero])
                scale = sparse.diags(inv)
                adjacency = scale @ adjacency @ scale
            return CliqueProjection(gram.tocsr(), adjacency.tocsr(), degree)

        gram = (incidence * weights[np.newaxis, :]) @ incidence.T
        degree = np.diag(np.diag(gram))
        adjacency = gram - degree
        if normalize:
            diagonal = np.diag(gram)
            if np.any(diagonal < 0):
                raise ValueError("Degree normalization requires non-negative edge weights")
            inv = np.zeros_like(diagonal)
            nonzero = diagonal > 0
            inv[nonzero] = 1.0 / np.sqrt(diagonal[nonzero])
            adjacency = inv[:, None] * adjacency * inv[None, :]
        return CliqueProjection(gram, adjacency, degree)

    def adjacency_tensor(
        self,
        *,
        format: Literal["dense", "coo"] = "dense",
        max_elements: int = 10_000_000,
        max_entries: int = 1_000_000,
    ) -> np.ndarray | SparseTensorCOO:
        if not self.is_uniform:
            raise ValueError("Adjacency tensor requires a uniform hypergraph")
        if max_elements < 0 or max_entries < 0:
            raise ValueError("Tensor allocation limits must be non-negative")
        order = self.uniform_order if self.edges else 0
        assert order is not None
        shape = (len(self.vertices),) * order
        size = prod(shape)
        if format == "dense" and size > max_elements:
            raise MemoryError(
                f"Dense tensor would contain {size:,} elements; limit is {max_elements:,}; use format='coo'"
            )
        if format not in {"dense", "coo"}:
            raise ValueError(f"Unsupported tensor format: {format!r}")
        entry_count = sum(factorial(len(edge.vertices)) for edge in self.edges)
        if entry_count > max_entries:
            raise MemoryError(
                f"Symmetric COO expansion would contain {entry_count:,} entries; "
                f"limit is {max_entries:,}"
            )

        coordinates: list[tuple[int, ...]] = []
        values: list[float] = []
        index = self.vertex_index
        for edge in self.edges:
            edge_indices = tuple(index[vertex] for vertex in edge.vertices)
            for coordinate in sorted(set(permutations(edge_indices))):
                coordinates.append(coordinate)
                values.append(edge.weight)
        coordinate_array = np.asarray(coordinates, dtype=np.int64).reshape(
            (len(coordinates), order)
        )
        value_array = np.asarray(values, dtype=np.float64)
        coo = SparseTensorCOO(coordinate_array, value_array, shape)
        return coo if format == "coo" else coo.to_dense(max_elements=max_elements)

    def tensor_contraction(self, signals: np.ndarray) -> np.ndarray:
        """Compute a symmetric-hyperedge message: each vertex sums products of the other endpoints."""
        if not self.is_uniform:
            raise ValueError("Tensor contraction requires a uniform hypergraph")
        if signals.shape != (len(self.vertices),):
            raise ValueError(f"signals must have shape ({len(self.vertices)},)")
        output = np.zeros_like(signals, dtype=np.result_type(signals.dtype, np.float64))
        index = self.vertex_index
        for edge in self.edges:
            endpoint_indices = [index[vertex] for vertex in edge.vertices]
            for position, vertex_index in enumerate(endpoint_indices):
                contribution = edge.weight
                for other_position, other_index in enumerate(endpoint_indices):
                    if position != other_position:
                        contribution *= signals[other_index]
                output[vertex_index] += contribution
        return output
