from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from numba import njit
    from numba import cuda
except ImportError:
    njit = None
    cuda = None


@dataclass(frozen=True)
class KernelSource:
    name: str
    language: str
    source: str


def aggregate_incidence_numpy(
    indptr: np.ndarray,
    edge_indices: np.ndarray,
    incidence_values: np.ndarray,
    edge_weights: np.ndarray,
    edge_values: np.ndarray,
) -> np.ndarray:
    """CSR vertex aggregation: y[v] = sum_e B[v,e] w[e] x[e]."""
    _validate_csr_inputs(indptr, edge_indices, incidence_values, edge_weights, edge_values)
    n_vertices = len(indptr) - 1
    output = np.zeros(n_vertices, dtype=np.result_type(edge_weights, edge_values, np.float64))
    for vertex in range(n_vertices):
        for offset in range(indptr[vertex], indptr[vertex + 1]):
            edge = edge_indices[offset]
            output[vertex] += incidence_values[offset] * edge_weights[edge] * edge_values[edge]
    return output


def _validate_csr_inputs(indptr, edge_indices, incidence_values, edge_weights, edge_values):
    arrays = (indptr, edge_indices, incidence_values, edge_weights, edge_values)
    if any(not isinstance(array, np.ndarray) or array.ndim != 1 for array in arrays):
        raise ValueError("All CSR kernel inputs must be one-dimensional NumPy arrays")
    if len(indptr) == 0 or indptr[0] != 0 or indptr[-1] != len(edge_indices):
        raise ValueError("Invalid CSR row pointer")
    if not np.issubdtype(indptr.dtype, np.integer):
        raise ValueError("CSR row pointers must have an integer dtype")
    if not np.issubdtype(edge_indices.dtype, np.integer):
        raise ValueError("CSR edge indices must have an integer dtype")
    if np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("CSR row pointer must be monotonic")
    if edge_indices.shape != incidence_values.shape:
        raise ValueError("edge_indices and incidence_values must have matching shapes")
    if edge_weights.shape != edge_values.shape:
        raise ValueError("edge_weights and edge_values must have matching shapes")
    if np.any(edge_indices < 0) or np.any(edge_indices >= len(edge_values)):
        raise ValueError("CSR edge index is out of bounds")
    if not all(
        np.issubdtype(array.dtype, np.number)
        for array in (incidence_values, edge_weights, edge_values)
    ):
        raise ValueError("Incidence values and edge data must be numeric")
    if any(
        np.iscomplexobj(array)
        for array in (incidence_values, edge_weights, edge_values)
    ):
        raise ValueError("Incidence aggregation currently supports real-valued data only")


def _cpu_aggregate_impl(indptr, edge_indices, incidence_values, edge_weights, edge_values):
    output = np.zeros(len(indptr) - 1, dtype=np.float64)
    for vertex in range(len(indptr) - 1):
        for offset in range(indptr[vertex], indptr[vertex + 1]):
            edge = edge_indices[offset]
            output[vertex] += incidence_values[offset] * edge_weights[edge] * edge_values[edge]
    return output


def jit_cpu_aggregate():
    if njit is None:
        return aggregate_incidence_numpy
    compiled = njit(cache=True)(_cpu_aggregate_impl)

    def validated_kernel(indptr, edge_indices, incidence_values, edge_weights, edge_values):
        _validate_csr_inputs(indptr, edge_indices, incidence_values, edge_weights, edge_values)
        return compiled(indptr, edge_indices, incidence_values, edge_weights, edge_values)

    validated_kernel.__name__ = "hal_aggregate_csr_jit"
    validated_kernel.numba_kernel = compiled
    return validated_kernel


def numba_cuda_available() -> bool:
    return bool(cuda is not None and cuda.is_available())


def get_cuda_aggregate_kernel():
    if cuda is None:
        raise RuntimeError("Numba is not installed; install the optional 'jit' dependencies")

    @cuda.jit
    def aggregate(indptr, edge_indices, incidence_values, edge_weights, edge_values, output):
        vertex = cuda.grid(1)
        if vertex < output.size:
            total = 0.0
            for offset in range(indptr[vertex], indptr[vertex + 1]):
                edge = edge_indices[offset]
                total += incidence_values[offset] * edge_weights[edge] * edge_values[edge]
            output[vertex] = total

    return aggregate


def run_cuda_aggregate(
    indptr: np.ndarray,
    edge_indices: np.ndarray,
    incidence_values: np.ndarray,
    edge_weights: np.ndarray,
    edge_values: np.ndarray,
) -> np.ndarray:
    _validate_csr_inputs(indptr, edge_indices, incidence_values, edge_weights, edge_values)
    if not numba_cuda_available():
        raise RuntimeError("No CUDA device is available to execute the HAL kernel")
    kernel = get_cuda_aggregate_kernel()
    device_inputs = [
        cuda.to_device(np.ascontiguousarray(array))
        for array in (indptr, edge_indices, incidence_values, edge_weights, edge_values)
    ]
    output = np.empty(len(indptr) - 1, dtype=np.float64)
    device_output = cuda.device_array_like(output)
    threads_per_block = 128
    blocks = (len(output) + threads_per_block - 1) // threads_per_block
    if blocks:
        kernel[blocks, threads_per_block](*device_inputs, device_output)
    return device_output.copy_to_host()


def emit_cpu_aggregate_source(name: str = "hal_aggregate_csr") -> KernelSource:
    if not name.isidentifier():
        raise ValueError(f"Invalid Python kernel name: {name!r}")
    source = f'''import numpy as np

def {name}(indptr, edge_indices, incidence_values, edge_weights, edge_values):
    output = np.zeros(len(indptr) - 1, dtype=np.float64)
    for vertex in range(len(indptr) - 1):
        for offset in range(indptr[vertex], indptr[vertex + 1]):
            edge = edge_indices[offset]
            output[vertex] += incidence_values[offset] * edge_weights[edge] * edge_values[edge]
    return output
'''
    return KernelSource(name, "python", source)


def emit_cuda_aggregate_source(name: str = "hal_aggregate_csr") -> KernelSource:
    if not name.isidentifier():
        raise ValueError(f"Invalid CUDA kernel name: {name!r}")
    source = f"""extern "C" __global__
void {name}(
    const int* indptr,
    const int* edge_indices,
    const double* incidence_values,
    const double* edge_weights,
    const double* edge_values,
    double* output,
    const int n_vertices
) {{
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < n_vertices) {{
        double total = 0.0;
        for (int offset = indptr[vertex]; offset < indptr[vertex + 1]; ++offset) {{
            const int edge = edge_indices[offset];
            total += incidence_values[offset] * edge_weights[edge] * edge_values[edge];
        }}
        output[vertex] = total;
    }}
}}
"""
    return KernelSource(name, "cuda", source)
