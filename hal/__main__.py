from __future__ import annotations

import numpy as np

from .hypergraph import Hyperedge, Hypergraph
from .kernels import (
    emit_cpu_aggregate_source,
    emit_cuda_aggregate_source,
    jit_cpu_aggregate,
    numba_cuda_available,
)
from .parser import parse_hal
from .typecheck import infer_type
from .types import BaseType, tensor


def main() -> None:
    signatures = {
        "f": (BaseType("A"), tensor(BaseType("B"), BaseType("C"))),
        "g": (BaseType("B"), BaseType("D")),
        "h": (BaseType("C"), BaseType("E")),
    }
    expression = parse_hal(
        "f # (split(B) @ id(C)) # (g @ id(B) @ id(C))",
        signatures,
    )
    print(f"HAL expression: {expression}")
    print(f"Inferred type: {infer_type(expression)}")

    graph = Hypergraph(
        ("v0", "v1", "v2"),
        (
            Hyperedge("e0", ("v0", "v1", "v2"), 1.0),
            Hyperedge("e1", ("v1", "v2"), 2.0),
        ),
    )
    print("Incidence matrix B:")
    print(graph.incidence_matrix())
    projection = graph.clique_projection()
    print("B W B^T:")
    print(projection.gram)
    print("Projected adjacency (off-diagonal):")
    print(projection.adjacency)

    csr = graph.incidence_matrix("csr")
    aggregate = jit_cpu_aggregate()
    result = aggregate(
        csr.indptr,
        csr.indices,
        csr.data,
        np.asarray([edge.weight for edge in graph.edges]),
        np.asarray([1.0, 3.0]),
    )
    print("CPU CSR message aggregation:", result)
    print(f"Numba CUDA device available: {numba_cuda_available()}")
    print("Generated CPU kernel:", emit_cpu_aggregate_source().name)
    print("Generated CUDA kernel:", emit_cuda_aggregate_source().name)


if __name__ == "__main__":
    main()
