"""Source-coupled decoration content retains train lineage and original support."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from compose_v4.benchmark.source_coupled_pendant_policy import SourceCoupledPendantSampler


def _inputs():
    split = {"partition": "train"}
    entries = []
    for context, contents in (
        ("C:0:0", (("[1*]C", 1), ("[1*]N", 2))),
        ("N:0:0", (("[1*]O", 1), ("[1*]F", 2))),
    ):
        for smiles, row in contents:
            entries.append(
                {
                    "context": context,
                    "rooted_smiles": smiles,
                    "heavy_atoms": 1,
                    "ring_count": 0,
                    "source_balanced_weight": 0.5,
                    "occurrences": 1,
                    "source_rows": [row],
                }
            )
    catalog = {
        "schema": "split_first_training_pendant_catalog_v1",
        "source_sha256": "training-source",
        "split": split,
        "training_molecules": 2,
        "accepted_source_rows": [1, 2],
        "selection_uses_qed_sa": False,
        "benchmark_prompts_used_for_selection": False,
        "entries": entries,
    }
    mass = {
        "schema": "split_first_training_decoration_mass_prior_v1",
        "source_sha256": "training-source",
        "split": split,
        "quality_labels_used": False,
        "benchmark_prompts_used_to_fit": False,
        "rows_with_murcko_decorations": 2,
        "counts": [[38, 2, 2, 2]],
    }
    return catalog, mass


def test_source_coupling_is_stochastic_and_preserves_independent_lane():
    sampler = SourceCoupledPendantSampler(*_inputs())
    rng = np.random.default_rng(7)
    modes = set()
    for _ in range(100):
        entries, receipt = sampler.sample(("C:0:0", "N:0:0"), 2, rng)
        modes.add(receipt["source_coupling_used"])
        assert receipt["planned_decoration_mass"] == 2
        assert receipt["shared_training_rows"] == 2
        assert receipt["selection_uses_qed_sa"] is False
        assert receipt["observed_decoration_bundle_claim"] is False
        if receipt["source_coupling_used"]:
            assert receipt["shared_training_source_row"] in set.intersection(
                *(set(entry["source_rows"]) for entry in entries)
            )
        else:
            assert receipt["shared_training_source_row"] is None
    assert modes == {False, True}


def test_no_shared_training_source_falls_back_without_abstaining():
    catalog, mass = _inputs()
    catalog["entries"] = [catalog["entries"][0], catalog["entries"][3]]
    catalog["entries"][0]["source_balanced_weight"] = 1.0
    catalog["entries"][1]["source_balanced_weight"] = 1.0
    sampler = SourceCoupledPendantSampler(catalog, mass)
    entries, receipt = sampler.sample(("C:0:0", "N:0:0"), 2, np.random.default_rng(1))
    assert len(entries) == 2
    assert receipt["shared_training_rows"] == 0
    assert receipt["source_coupling_used"] is False


def test_rejects_corrupt_or_nontraining_source_provenance():
    catalog, mass = _inputs()
    wrong_weight = copy.deepcopy(catalog)
    wrong_weight["entries"][0]["source_balanced_weight"] = 0.6
    with pytest.raises(ValueError, match="balanced pendant mass changed"):
        SourceCoupledPendantSampler(wrong_weight, mass)
    wrong_split = copy.deepcopy(catalog)
    wrong_split["entries"][0]["source_rows"] = [3]
    with pytest.raises(ValueError, match="outside the training split"):
        SourceCoupledPendantSampler(wrong_split, mass)
    with pytest.raises(ValueError, match="both proposal lanes"):
        SourceCoupledPendantSampler(catalog, mass, coupled_probability=1.0)
