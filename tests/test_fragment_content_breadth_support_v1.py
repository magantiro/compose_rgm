"""Deterministic provenance and failure checks for the unscored breadth census."""

import json
from pathlib import Path

import pytest

from tools.audit_fragment_content_breadth_support_v1 import _plan_contexts, audit


def _inputs(root: Path):
    catalog = {
        "schema": "split_first_training_pendant_catalog_v1",
        "source_sha256": "fixture",
        "split": {"partition": "train"},
        "selection_uses_qed_sa": False,
        "benchmark_prompts_used_for_selection": False,
        "entries": [
            {
                "context": "C:0:0",
                "rooted_smiles": smiles,
                "heavy_atoms": 1,
                "ring_count": 0,
                "occurrences": 1,
                "source_rows": [index],
                "source_balanced_weight": weight,
            }
            for index, (smiles, weight) in enumerate((("[1*]C", 1.0), ("[1*]N", 9.0)))
        ],
    }
    prior = {
        "schema": "split_first_training_decoration_mass_prior_v1",
        "source_sha256": "fixture",
        "split": {"partition": "train"},
        "quality_labels_used": False,
        "benchmark_prompts_used_to_fit": False,
        "rows_with_murcko_decorations": 1,
        "counts": [[20, 1, 1, 1]],
    }
    catalog_path = root / "catalog.json"
    prior_path = root / "prior.json"
    attempts = root / "attempts"
    attempts.mkdir()
    catalog_path.write_text(json.dumps(catalog))
    prior_path.write_text(json.dumps(prior))
    for drug_index in range(10):
        for attempt_index in range(20):
            (attempts / f"D{drug_index}_{attempt_index:03d}.json").write_text(
                json.dumps(
                    {
                        "drug": f"D{drug_index}",
                        "panel": {
                            "offered": [
                                {
                                    "status": "model_supported",
                                    "provenance": {
                                        "pendant_plan": {
                                            "core_heavy_atoms": 20,
                                            "draws": [{"context": "C:0:0"}],
                                        }
                                    },
                                }
                            ]
                        },
                    }
                )
            )
    return catalog_path, prior_path, attempts


def test_support_audit_is_deterministic_and_hashes_every_attempt(tmp_path):
    inputs = _inputs(tmp_path)
    first = audit(*inputs, draws=12)
    second = audit(*inputs, draws=12)
    assert first == second
    assert first["schema"] == "fragment_content_breadth_support_v1"
    assert first["quality_labels_used"] is False
    assert len(first["input_sha256"]["attempt_files"]) == 200
    assert len(first["prompts"]) == 10
    assert all(
        row["arms"]["uniform_within_cell"]["distinct_content_plans"] <= 12
        for row in first["prompts"]
    )
    (inputs[2] / "D0_000.json").unlink()
    with pytest.raises(ValueError, match="exactly 200 attempts"):
        audit(*inputs, draws=12)


def test_context_multiset_ignores_per_offer_interface_permutation():
    forward = {"draws": [{"context": "N:0:1"}, {"context": "C:0:1"}]}
    reverse = {"draws": list(reversed(forward["draws"]))}
    assert _plan_contexts(forward) == _plan_contexts(reverse) == ("C:0:1", "N:0:1")
