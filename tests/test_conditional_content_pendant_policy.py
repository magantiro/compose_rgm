"""Focused tests for train-conditioned scaffold-decoration content."""

import copy

import numpy as np
import pytest

from compose_v4.benchmark.conditional_content_pendant_policy import (
    ConditionalContentPendantSampler,
)


def _inputs():
    split = {"partition": "train"}
    catalog = {
        "schema": "split_first_training_pendant_catalog_v1",
        "source_sha256": "training-source",
        "split": split,
        "training_molecules": 2,
        "accepted_source_rows": [1, 2],
        "selection_uses_qed_sa": False,
        "benchmark_prompts_used_for_selection": False,
        "entries": [
            {
                "context": "C:0:0",
                "rooted_smiles": "[1*]C",
                "heavy_atoms": 1,
                "ring_count": 0,
                "source_balanced_weight": 1.0,
                "occurrences": 1,
                "source_rows": [1],
            },
            {
                "context": "C:0:0",
                "rooted_smiles": "[1*]N",
                "heavy_atoms": 1,
                "ring_count": 0,
                "source_balanced_weight": 1.0,
                "occurrences": 1,
                "source_rows": [2],
            },
        ],
    }
    mass = {
        "schema": "split_first_training_decoration_mass_prior_v1",
        "source_sha256": "training-source",
        "split": split,
        "quality_labels_used": False,
        "benchmark_prompts_used_to_fit": False,
        "rows_with_murcko_decorations": 2,
        "counts": [[5, 1, 1, 1], [20, 1, 1, 1]],
    }
    content = {
        "schema": "split_first_training_decoration_content_context_v1",
        "source_sha256": "training-source",
        "catalog_sha256": "catalog-hash",
        "split": split,
        "quality_labels_used": False,
        "benchmark_prompts_used_to_fit": False,
        "training_rows": 2,
        "row_contexts": [[1, 5, 1], [2, 20, 1]],
    }
    return catalog, mass, content


def test_conditional_content_preserves_global_support_and_stochastic_draws():
    sampler = ConditionalContentPendantSampler(*_inputs(), catalog_sha256="catalog-hash")
    nearby, observed_mass = sampler.content_weights(5, 1, "C:0:0", 1, 0)
    assert observed_mass == 1.0
    assert 0 < nearby[1] < nearby[0] < 1
    assert np.isclose(nearby.sum(), 1)
    fallback, observed_mass = sampler.content_weights(30, 1, "C:0:0", 1, 0)
    assert observed_mass == 0.0
    assert np.allclose(fallback, (0.5, 0.5))
    entries, receipt = sampler.sample(("C:0:0",), 35, np.random.default_rng(7))
    assert len(entries) == 1
    assert receipt["schema"] == "conditional_content_pendant_plan_v1"
    assert receipt["planned_decoration_mass"] == 1
    assert receipt["draws"][0]["rooted_smiles"] == entries[0]["rooted_smiles"]
    assert receipt["selection_uses_qed_sa"] is False


def test_conditional_content_rejects_nontraining_or_corrupted_provenance():
    catalog, mass, content = _inputs()
    corrupted = copy.deepcopy(content)
    corrupted["quality_labels_used"] = True
    with pytest.raises(ValueError, match="train-only provenance"):
        ConditionalContentPendantSampler(catalog, mass, corrupted, catalog_sha256="catalog-hash")
    corrupted = copy.deepcopy(catalog)
    corrupted["entries"][0]["source_balanced_weight"] = 2.0
    with pytest.raises(ValueError, match="cannot be reconstructed"):
        ConditionalContentPendantSampler(corrupted, mass, content, catalog_sha256="catalog-hash")
