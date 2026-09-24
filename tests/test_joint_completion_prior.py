from itertools import product

import numpy as np
import pytest

from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior, JointCompletionSampler


def entry(key, atoms, rings=0, occurrence=1, name=None):
    return {
        "contexts": [key],
        "rooted_smiles": name or f"{key}_{atoms}_{rings}",
        "heavy_atoms": atoms,
        "ring_count": rings,
        "occurrences": occurrence,
        "source_rows": [1],
    }


def test_cell_law_does_not_multiply_by_number_of_allocations():
    # Two ways to reach 33 atoms, one each to reach 32 and 34. All cells equally likely.
    entries = [entry("x", 1), entry("x", 2)]
    prior = JointCompletionPrior(((32, 0, 10), (33, 0, 10), (34, 0, 10)))
    sampler = JointCompletionSampler(entries, prior)
    table = sampler.plan_table(("x", "x"), 30, 0)
    assert table.cells == ((32, 0), (33, 0), (34, 0))
    np.testing.assert_allclose(table.probabilities, [1 / 3] * 3)
    np.testing.assert_allclose(table.suffix_mass[0][[2, 3, 4], 0], [0.25, 0.5, 0.25])


def test_dp_equals_exhaustive_tiny_fiber():
    entries = [entry("a", 1), entry("a", 3, 1, 4), entry("b", 2), entry("b", 3, 1, 9)]
    sampler = JointCompletionSampler(entries, JointCompletionPrior(((5, 0, 1),)))
    table = sampler.plan_table(("a", "b"), 2, 0)
    expected = np.zeros_like(table.suffix_mass[0])
    for a, b in product(entries[:2], entries[2:]):
        expected[a["heavy_atoms"] + b["heavy_atoms"], a["ring_count"] + b["ring_count"]] += (
            np.sqrt(a["occurrences"]) / 3 * np.sqrt(b["occurrences"]) / 4
        )
    np.testing.assert_allclose(table.suffix_mass[0], expected)
    reverse = sampler.plan_table(("b", "a"), 2, 0)
    np.testing.assert_allclose(table.probabilities, reverse.probabilities)
    assert table.cells == reverse.cells


def test_sample_probability_matches_cell_conditioning_and_is_reproducible():
    entries = [entry("a", 1), entry("a", 2, 0, 9)]
    sampler = JointCompletionSampler(entries, JointCompletionPrior(((4, 0, 100),)))
    observed = set()
    for seed in range(40):
        first, receipt = sampler.sample(("a", "a"), 1, 0, np.random.default_rng(seed))
        second, again = sampler.sample(("a", "a"), 1, 0, np.random.default_rng(seed))
        assert first == second and receipt == again
        sizes = tuple(e["heavy_atoms"] for e in first)
        observed.add(sizes)
        if receipt["planned_heavy_atoms"] == 4:
            assert receipt["conditional_content_probability"] == pytest.approx(0.5)
        assert sum(sizes) + 1 == receipt["planned_heavy_atoms"]
    assert (1, 2) in observed and (2, 1) in observed


def test_joint_allocation_supports_six_interfaces_and_rings():
    sampler = JointCompletionSampler(
        [entry("a", 2), entry("a", 6, 1)], JointCompletionPrior(((30, 1, 1),))
    )
    selected, receipt = sampler.sample(("a",) * 6, 14, 0, np.random.default_rng(0))
    assert len(selected) == 6
    assert sum(e["heavy_atoms"] for e in selected) + 14 <= 40
    assert any(e["ring_count"] for e in selected)
    assert receipt["planned_rings"] == sum(e["ring_count"] for e in selected)


def test_positive_density_for_unobserved_reachable_cells():
    prior = JointCompletionPrior(((2, 0, 1000),))
    p = prior.cell_probabilities(((2, 0), (40, 8)))
    assert p[1] > 0 and p.sum() == pytest.approx(1)


def test_impossible_context_or_capacity_abstains():
    sampler = JointCompletionSampler([entry("a", 6, 1)], JointCompletionPrior(((6, 1, 1),)))
    with pytest.raises(ValueError, match="boundary context"):
        sampler.sample(("missing",), 1, 0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="feasible joint"):
        sampler.sample(("a",) * 6, 10, 0, np.random.default_rng(0))


@pytest.mark.parametrize(
    "counts", [(), ((0, 0, 1),), ((5, 0, 0),), ((5, -1, 1),), ((5, 0, 1), (5, 0, 2))]
)
def test_malformed_prior_is_rejected(counts):
    with pytest.raises(ValueError):
        JointCompletionPrior(counts)


def test_duplicate_regions_are_not_extra_mass():
    with pytest.raises(ValueError, match="duplicate"):
        JointCompletionSampler([entry("a", 2)] * 2, JointCompletionPrior(((5, 0, 1),)))
