"""Exact capacity conditioning and positive support for pendant completions."""

import numpy as np
import pytest
from rdkit import Chem

from compose_v4.benchmark.pendant_completion_policy import PendantCompletionSampler


def entry(smiles, weight=1.0, context="C:0:0"):
    molecule = Chem.MolFromSmiles(smiles)
    return {
        "context": context,
        "rooted_smiles": smiles,
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "ring_count": molecule.GetRingInfo().NumRings(),
        "source_balanced_weight": weight,
        "occurrences": 1,
        "source_rows": [1],
    }


def catalog(*entries):
    return {
        "schema": "split_first_training_pendant_catalog_v1",
        "split": {"partition": "train"},
        "selection_uses_qed_sa": False,
        "benchmark_prompts_used_for_selection": False,
        "entries": list(entries),
    }


class FixedRng:
    def __init__(self):
        self.probabilities = []

    def choice(self, n, *, p):
        self.probabilities.append(tuple(float(value) for value in p))
        return 0


def test_joint_capacity_conditions_without_rejection_or_lost_positive_mass():
    sampler = PendantCompletionSampler(catalog(entry("[1*]C"), entry("[1*]CC")))
    rng = FixedRng()
    chosen, receipt = sampler.sample(("C:0:0", "C:0:0"), 3, rng)
    assert rng.probabilities[0] == pytest.approx((2 / 3, 1 / 3))
    assert [item["heavy_atoms"] for item in chosen] == [1, 1]
    assert receipt["joint_capacity_probability"] == pytest.approx(0.75)
    assert receipt["unused_atom_capacity"] == 1
    for seed in range(30):
        draw, _ = sampler.sample(("C:0:0", "C:0:0"), 2, np.random.default_rng(seed))
        assert [item["heavy_atoms"] for item in draw] == [1, 1]


def test_ring_content_stays_available_and_invalid_support_fails_closed():
    sampler = PendantCompletionSampler(
        catalog(entry("[1*]C"), entry("[1*]c1ccccc1", context="C:1:1"))
    )
    chosen, _ = sampler.sample(("C:1:1",), 6, np.random.default_rng(0))
    assert chosen[0]["ring_count"] == 1
    with pytest.raises(ValueError, match="no observed training pendant"):
        sampler.sample(("N:0:0",), 10, np.random.default_rng(0))
    with pytest.raises(ValueError, match="exceed atom capacity"):
        sampler.sample(("C:1:1", "C:1:1"), 10, np.random.default_rng(0))
    bad = entry("[1*]C")
    bad["heavy_atoms"] = 2
    with pytest.raises(ValueError, match="metadata changed"):
        PendantCompletionSampler(catalog(bad))


def test_uniform_within_cell_preserves_support_and_group_mass():
    training = catalog(entry("[1*]C", 1.0), entry("[1*]N", 9.0))
    frozen = PendantCompletionSampler(training)
    broader = PendantCompletionSampler(training, content_allocation="uniform_within_cell")
    frozen_group = frozen.pools["C:0:0"][0]
    broader_group = broader.pools["C:0:0"][0]
    assert frozen_group.mass == broader_group.mass == 1.0
    assert frozen_group.content_probabilities == pytest.approx((0.25, 0.75))
    assert broader_group.content_probabilities == pytest.approx((0.5, 0.5))
    assert tuple(item["rooted_smiles"] for item in broader_group.entries) == (
        "[1*]C",
        "[1*]N",
    )
    with pytest.raises(ValueError, match="unknown pendant content allocation"):
        PendantCompletionSampler(training, content_allocation="other")
