import numpy as np
import pytest

from hal import (
    BaseType,
    Hyperedge,
    Hypergraph,
    HALSyntaxError,
    Identity,
    Merge,
    Sequential,
    Split,
    TypeCheckError,
    emit_cpu_aggregate_source,
    emit_cuda_aggregate_source,
    infer_type,
    jit_cpu_aggregate,
    numba_cuda_available,
    parse_hal,
    rewrite,
    run_cuda_aggregate,
    tensor,
)


def test_parser_precedence_types_and_declarations():
    expression = parse_hal(
        "f: A -> (B @ C) # g: (B @ C) -> D"
    )
    assert isinstance(expression, Sequential)
    assert infer_type(expression) == (BaseType("A"), BaseType("D"))


def test_parser_uses_registry_and_rejects_unknown_morphism():
    expression = parse_hal("f # g", {"f": (BaseType("A"), BaseType("B")), "g": (BaseType("B"), BaseType("C"))})
    assert infer_type(expression) == (BaseType("A"), BaseType("C"))
    with pytest.raises(HALSyntaxError, match="Unknown morphism"):
        parse_hal("missing")


def test_parser_supports_unicode_tensor_and_line_comments():
    signatures = {"f": (BaseType("A"), BaseType("B"))}
    expression = parse_hal("f ⊗ f", signatures)
    assert parse_hal("f: A -> B // trailing comment", signatures)
    assert infer_type(expression) == (
        tensor(BaseType("A"), BaseType("A")),
        tensor(BaseType("B"), BaseType("B")),
    )


@pytest.mark.parametrize("source", ["", "f #", "id(A", "f @ @ g", "f ? g"])
def test_parser_rejects_malformed_input(source):
    with pytest.raises(HALSyntaxError):
        parse_hal(source)


def test_polymorphic_signature_is_freshened_per_use():
    expression = parse_hal("identity: a -> a # convert: B -> C # identity")
    assert infer_type(expression) == (BaseType("B"), BaseType("C"))


def test_composition_type_mismatch_is_reported():
    expression = parse_hal("f: A -> B # g: C -> D")
    with pytest.raises(TypeCheckError, match="Incompatible types"):
        infer_type(expression)


def test_rewriter_speciality_has_hal_composition_order():
    reduced = rewrite(parse_hal("split(X) # merge(X)"))
    assert reduced == Identity(BaseType("X"))
    # The reverse order is merge then split and is not the special axiom.
    assert rewrite(parse_hal("merge(X) # split(X)")) != Identity(BaseType("X"))


@pytest.mark.parametrize(
    "source, expected",
    [
        ("swap(X, X) # merge(X)", Merge),
        ("split(X) # swap(X, X)", Split),
        ("cup(X) @ id(X) # merge(X)", Identity),
        ("id(X) @ cup(X) # merge(X)", Identity),
        ("split(X) # cap(X) @ id(X)", Identity),
        ("split(X) # id(X) @ cap(X)", Identity),
    ],
)
def test_frobenius_commutativity_unit_and_counit_rules(source, expected):
    assert isinstance(rewrite(parse_hal(source)), expected)


@pytest.mark.parametrize(
    "left, right",
    [
        (
            "id(X) @ split(X) # merge(X) @ id(X)",
            "split(X) @ id(X) # id(X) @ merge(X)",
        ),
    ],
)
def test_both_frobenius_composites_normalize_to_same_term(left, right):
    left_normal = rewrite(parse_hal(left))
    right_normal = rewrite(parse_hal(right))
    assert left_normal == right_normal
    assert infer_type(left_normal) == (tensor(BaseType("X"), BaseType("X")), tensor(BaseType("X"), BaseType("X")))


def test_associativity_and_coassociativity_have_canonical_orientations():
    merge_left = rewrite(parse_hal("merge(X) @ id(X) # merge(X)"))
    merge_right = rewrite(parse_hal("id(X) @ merge(X) # merge(X)"))
    assert merge_left == merge_right
    split_left = rewrite(parse_hal("split(X) # split(X) @ id(X)"))
    split_right = rewrite(parse_hal("split(X) # id(X) @ split(X)"))
    assert split_left == split_right


def test_parallel_identity_and_type_normalization():
    result = rewrite(parse_hal("id(A) @ id(B)"))
    assert result == Identity(tensor(BaseType("A"), BaseType("B")))


def test_symmetric_swap_is_involutive():
    result = rewrite(parse_hal("swap(A, B) # swap(B, A)"))
    assert result == Identity(tensor(BaseType("A"), BaseType("B")))


def test_incidence_projection_matches_gram_identity():
    graph = Hypergraph(
        ("u", "v", "w"),
        (Hyperedge("e0", ("u", "v"), 2.0), Hyperedge("e1", ("v", "w"), 3.0)),
    )
    incidence = graph.incidence_matrix()
    projection = graph.clique_projection()
    expected = incidence @ np.diag([2.0, 3.0]) @ incidence.T
    np.testing.assert_allclose(projection.gram, expected)
    np.testing.assert_allclose(projection.gram, projection.adjacency + projection.degree)
    np.testing.assert_allclose(np.diag(projection.adjacency), 0)


def test_csr_incidence_backend_matches_dense_and_projection():
    pytest.importorskip("scipy")
    graph = Hypergraph(
        ("u", "v", "w"),
        (Hyperedge("e0", ("u", "v")), Hyperedge("e1", ("v", "w"))),
    )
    dense = graph.incidence_matrix("dense")
    csr = graph.incidence_matrix("csr")
    np.testing.assert_allclose(csr.toarray(), dense)
    np.testing.assert_allclose(
        graph.clique_projection(format="csr").gram.toarray(),
        graph.clique_projection(format="dense").gram,
    )


def test_uniform_tensor_is_symmetric_and_exposes_sparse_coo():
    graph = Hypergraph(
        ("u", "v", "w"),
        (Hyperedge("triangle", ("u", "v", "w"), 2.5),),
    )
    tensor = graph.adjacency_tensor()
    assert tensor.shape == (3, 3, 3)
    assert tensor[0, 1, 2] == tensor[2, 1, 0] == 2.5
    sparse_tensor = graph.adjacency_tensor(format="coo")
    np.testing.assert_allclose(sparse_tensor.to_dense(), tensor)


def test_zero_order_tensor_for_empty_hyperedges():
    graph = Hypergraph(("u",), (Hyperedge("empty", (), 2.0),))
    assert graph.adjacency_tensor().shape == ()
    assert graph.adjacency_tensor().item() == 2.0


def test_heterogeneous_tensor_and_dense_blowup_are_rejected():
    graph = Hypergraph(
        ("u", "v", "w"),
        (Hyperedge("pair", ("u", "v")), Hyperedge("triple", ("u", "v", "w"))),
    )
    with pytest.raises(ValueError, match="uniform"):
        graph.adjacency_tensor()
    uniform = Hypergraph(("u", "v", "w"), (Hyperedge("edge", ("u", "v", "w")),))
    with pytest.raises(MemoryError, match="format='coo'"):
        uniform.adjacency_tensor(max_elements=4)


def test_sparse_tensor_symmetry_expansion_has_a_separate_safety_limit():
    graph = Hypergraph(
        tuple(f"v{i}" for i in range(7)),
        (Hyperedge("large", tuple(f"v{i}" for i in range(7))),),
    )
    with pytest.raises(MemoryError, match="COO expansion"):
        graph.adjacency_tensor(format="coo", max_entries=100)


def test_hypergraph_validates_references_duplicates_and_weights():
    with pytest.raises(ValueError, match="unknown vertices"):
        Hypergraph(("u",), (Hyperedge("bad", ("missing",)),))
    with pytest.raises(ValueError, match="repeats"):
        Hyperedge("bad", ("u", "u"))
    with pytest.raises(ValueError, match="unique"):
        Hypergraph(("u", "u"), ())


def test_cpu_csr_kernel_matches_numpy_and_emitted_source_compiles():
    pytest.importorskip("scipy")
    graph = Hypergraph(
        ("u", "v", "w"),
        (Hyperedge("e0", ("u", "v"), 2), Hyperedge("e1", ("v", "w"), 3)),
    )
    csr = graph.incidence_matrix("csr")
    weights = np.asarray([2.0, 3.0])
    values = np.asarray([5.0, 7.0])
    expected = graph.incidence_matrix() @ (weights * values)
    kernel = jit_cpu_aggregate()
    result = kernel(
        csr.indptr, csr.indices, csr.data, weights, values
    )
    np.testing.assert_allclose(result, expected)
    if hasattr(kernel, "numba_kernel"):
        assert kernel.numba_kernel.signatures

    source = emit_cpu_aggregate_source()
    namespace = {}
    exec(compile(source.source, "<generated-hal-kernel>", "exec"), namespace)
    np.testing.assert_allclose(
        namespace[source.name](csr.indptr, csr.indices, csr.data, weights, values),
        expected,
    )


def test_cuda_emitter_generates_real_kernel_source():
    source = emit_cuda_aggregate_source()
    assert source.language == "cuda"
    assert "__global__" in source.source
    assert "indptr[vertex + 1]" in source.source


def test_cuda_runtime_reports_hardware_requirement():
    graph = Hypergraph(
        ("u", "v"),
        (Hyperedge("e", ("u", "v"), 2.0),),
    )
    csr = graph.incidence_matrix("csr")
    weights = np.asarray([2.0])
    values = np.asarray([3.0])
    if not numba_cuda_available():
        with pytest.raises(RuntimeError, match="No CUDA device"):
            run_cuda_aggregate(csr.indptr, csr.indices, csr.data, weights, values)
        return
    np.testing.assert_allclose(
        run_cuda_aggregate(csr.indptr, csr.indices, csr.data, weights, values),
        graph.incidence_matrix() @ (weights * values),
    )
