from __future__ import annotations

import pytest

from compose_v4.experiments.tracelet_conditional import build_tracelet_path_records
from scripts.train_tracelet_cnof_gate import (
    _legacy_evaluation_signature_matches,
    _path_cache_signature_mismatch,
    _validate_legacy_evaluation_partition,
)


def _signature() -> dict[str, object]:
    return {
        "format_version": 4,
        "seed": 7,
        "max_atoms": 12,
        "ring_proposals": "typed_catalog",
        "max_cycle_templates": 32,
        "max_attach_templates": 32,
        "max_ear_templates": 32,
        "source_prior": "carbon_tree",
        "tree_size_prior": "empirical",
        "tree_couplings_per_target": 2,
        "tree_transport": "size_matched_graft",
        "path_checkpoint_interval": 8,
        "path_shard_size": 2048,
        "train_smiles": ("CCO", "CCN"),
        "validation_smiles": ("CCC",),
        "test_smiles": ("CCF",),
    }


def test_legacy_signature_ignores_storage_only_fields_but_not_science() -> None:
    expected = _signature()
    legacy = {
        key: value
        for key, value in expected.items()
        if key
        not in {
            "format_version",
            "path_checkpoint_interval",
            "path_shard_size",
            "tree_size_prior",
        }
    }
    assert _legacy_evaluation_signature_matches(legacy, expected)

    legacy["tree_transport"] = "primitive"
    assert not _legacy_evaluation_signature_matches(legacy, expected)


def test_legacy_signature_defers_split_payloads_to_endpoint_audit() -> None:
    expected = _signature()
    legacy = dict(expected)
    legacy["train_smiles"] = list(reversed(expected["train_smiles"]))
    legacy["validation_smiles"] = []
    assert _legacy_evaluation_signature_matches(legacy, expected)

    legacy["tree_size_prior"] = "uniform"
    assert not _legacy_evaluation_signature_matches(legacy, expected)


def test_signature_mismatch_summary_excludes_corpus_payloads() -> None:
    expected = _signature()
    candidate = dict(expected)
    candidate["train_smiles"] = ("not", "logged")
    candidate["tree_transport"] = "primitive"
    mismatch = _path_cache_signature_mismatch(candidate, expected)
    assert mismatch == {
        "tree_transport": {
            "candidate": "primitive",
            "expected": "size_matched_graft",
        }
    }


def test_legacy_partition_audits_endpoint_multiplicity() -> None:
    smiles = ("CCO", "CCN")
    records = build_tracelet_path_records(
        smiles,
        n_slots=12,
        typed_ring_payloads=True,
    )
    coupled = (records[0], records[0], records[1], records[1])
    _validate_legacy_evaluation_partition(
        coupled,
        smiles,
        n_slots=12,
        couplings_per_target=2,
        partition="train",
    )

    with pytest.raises(ValueError, match="endpoint audit failed"):
        _validate_legacy_evaluation_partition(
            records,
            smiles,
            n_slots=12,
            couplings_per_target=2,
            partition="train",
        )
