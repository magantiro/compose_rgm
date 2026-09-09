"""Law parity, honest rejection and bounded production-executor work."""

from collections import Counter

import numpy as np
import pytest

from compose_v4.control.continuation import ContinuationBudgetExceeded
from compose_v4.control.lazy_reference import LazyReferenceBranch, LazyReferenceRow


def test_mixture_components_are_conditioned_before_mixing():
    # Most weighted mass is invalid. Redrawing the mixture after rejection
    # would grossly overweight its uniform component.
    branch = LazyReferenceBranch(
        (0, 1, 2), (1000.0, 1.0, 9.0), 0.15, lambda i: None if i == 0 else i
    )
    rng = np.random.default_rng(142)
    counts = Counter(branch.sample(rng) for _ in range(20000))
    assert counts[1] / 20000 == pytest.approx(0.85 * 0.1 + 0.15 * 0.5, abs=0.012)
    assert counts[None] == counts[0] == 0


def test_branches_are_uniform_not_weighted_by_clean_mark_count():
    row = LazyReferenceRow(
        {
            "empty": LazyReferenceBranch((0,), (1.0,), 0.15, lambda _: None),
            "one": LazyReferenceBranch((0,), (1.0,), 0.15, lambda _: "one"),
            "many": LazyReferenceBranch(tuple(range(9)), (1.0,) * 9, 0.15, lambda _: "many"),
        }
    )
    rng = np.random.default_rng(231)
    counts = Counter(row.sample(rng) for _ in range(10000))
    assert counts["one"] / 10000 == pytest.approx(0.5, abs=0.02)
    assert counts[None] == 0


def test_interruption_is_not_cached_as_invalid_and_zero_mass_fallback_matches():
    def stop(_):
        raise ContinuationBudgetExceeded("fixture stop")

    branch = LazyReferenceBranch((0,), (1.0,), 0.15, stop)
    with pytest.raises(ContinuationBudgetExceeded, match="fixture stop"):
        branch.has_product()
    assert not branch.products
    branch = LazyReferenceBranch((0, 1), (1.0, 0.0), 0.0, lambda i: i if i else None)
    assert branch.sample(np.random.default_rng(1)) == 1
    assert LazyReferenceRow({}).sample(np.random.default_rng(1)) is None


@pytest.mark.parametrize("option", ["generic", "grow", "scaffold_extend", "cyclize"])
def test_real_executor_lazy_support_and_draws_match_eager(option):
    from test_option_continuation import fixture_law, kernel, source

    node = source(option, 1)
    eager, lazy = kernel(), kernel()

    def nonuniform_law(graph):
        families, actions, _ = fixture_law(graph)
        weights = np.arange(1, len(actions) + 1, dtype=float) ** 2
        return families, actions, tuple(weights / weights.sum())

    eager.enumerate_law = lazy.enumerate_law = nonuniform_law
    exact = eager.row(node)
    sampled = lazy.lazy_row(node)
    assert sampled.has_product() == bool(exact.successors)
    if option == "grow":
        assert lazy.work.executor_applications < eager.work.executor_applications
    rng = np.random.default_rng(14)
    counts = Counter()
    for _ in range(4000):
        successor = sampled.sample(rng)
        counts[successor.key() if successor else None] += 1
    if not exact.successors:
        assert counts == {None: 4000}
    else:
        expected = Counter()
        for s, p in zip(exact.successors, exact.probabilities, strict=True):
            expected[s.key()] += p
        assert set(counts) == set(expected)
        for key, probability in expected.items():
            assert counts[key] / 4000 == pytest.approx(probability, abs=0.035)
        before = lazy.work.executor_applications
        replay = lazy.row(node)
        assert replay.probabilities == exact.probabilities
        assert tuple(s.key() for s in replay.successors) == tuple(s.key() for s in exact.successors)
        assert lazy.work.executor_applications == before


@pytest.mark.parametrize("topology", ["pendant", "fused"])
def test_ring_program_lazy_steps_match_full_augmented_rows(topology):
    from test_ring_program import fixture, uniform_law

    from compose_v4.control.option_continuation import OptionContinuationKernel
    from compose_v4.control.ring_program import RingSpec, completed_construction
    from compose_v4.rewrite.kernel import editing_v2_rewrite_system

    spec = RingSpec(topology, 5, (4, 1, 0), "aromatic")
    node = fixture(spec)
    kernel = OptionContinuationKernel(
        uniform_law, editing_v2_rewrite_system(), max_executor_applications=256
    )
    rng = np.random.default_rng(76)
    while node.remaining:
        lazy = kernel.lazy_row(node)
        exact = kernel.row(node)
        expected = Counter()
        for s, p in zip(exact.successors, exact.probabilities, strict=True):
            expected[s.key()] += p
        counts = Counter(lazy.sample(rng).key() for _ in range(1000))
        assert set(counts) <= set(expected)
        for key, probability in expected.items():
            assert counts[key] / 1000 == pytest.approx(probability, abs=0.055)
        node = lazy.sample(rng)
    assert completed_construction(node.origin, node.graph, node.ring_progress, spec)


def test_lazy_what_prior_matches_eager_with_fewer_product_checks():
    from test_option_continuation import kernel, source

    from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState

    root = MolecularSearchState.start(source().graph, budget=2, root_id="fixture")
    eager, lazy = (
        MolecularHierarchy(kernel()),
        MolecularHierarchy(kernel(), lazy_applicability=True),
    )
    region = max(eager.row(root).successors, key=lambda s: s.region.size)
    a, b = eager.row(region), lazy.row(region)
    assert a.labels == b.labels
    assert a.reference == pytest.approx(b.reference, abs=1e-14)
    assert tuple(s.key() for s in a.successors) == tuple(s.key() for s in b.successors)
    assert lazy.kernel.work.executor_applications <= eager.kernel.work.executor_applications


def test_reference_rollouts_supply_delayed_value_without_materializing_rows():
    from test_task_search import binary_planner

    for preferred in (0, 1):
        planner = binary_planner((preferred,) * 3)
        planner.reference_draw = lambda state, rng: state + (int(rng.integers(2)),)
        assert planner.plan((), 1000) == 1000
        assert planner.work.rows == 0
        for level in range(3):
            decision = planner.decision((preferred,) * level)
            assert decision["probabilities"][preferred] > 0.65
            assert decision["kl"] <= 1 + 1e-10
            assert min(decision["probabilities"]) >= 0.05
        assert planner.receipt()["estimator"] == "reference_rollouts_with_shrinkage"
