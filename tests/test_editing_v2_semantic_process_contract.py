"""Frozen prospective process contract for Editing V2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


CONTRACT = Path("configs/editing_v2_semantic_process_v1.json")
CYCLE_CLOSE_RESULT = Path(
    "diagnostics/coherence/editing_cycle_close_global_equivalence_v1/"
    "result.b0a7751ac7f8246abf2db8dd6337ee4fc083efa42d1408427de954beda6e3022.json"
)


def _semantic_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_is_self_hashed_non_authorizing_and_complete() -> None:
    value = json.loads(CONTRACT.read_text())
    claimed = value.pop("contract_sha256")
    assert claimed == "f928f6adaf22ba7520dd28839655c93bc523ce77317b32050d3dd1b02bbcf288"
    assert _semantic_sha256(value) == claimed
    assert value["training_authorized"] is False
    assert value["status"] == (
        "DESIGN_FROZEN_IMPLEMENTATION_AND_EVIDENCE_GATES_NOT_TRAINING_AUTHORIZED"
    )
    assert value["scope"]["active_families"] == [
        "atom_insert",
        "atom_delete",
        "atom_restate",
        "bond_reorder",
        "bond_reroute",
        "cycle_insert",
        "cycle_attach",
        "ring_system_restate",
    ]
    assert value["action_codec"]["raw_public_rules_forbidden"] == [
        "atom_restate",
        "bond_insert",
        "bond_delete",
    ]


def test_cycle_close_evidence_binding_matches_authoritative_artifact() -> None:
    contract = json.loads(CONTRACT.read_text())
    binding = contract["source_evidence"]["cycle_close_global_equivalence"]
    result = json.loads(CYCLE_CLOSE_RESULT.read_text())
    assert _file_sha256(CYCLE_CLOSE_RESULT) == binding["result_file_sha256"]
    assert result["result_sha256"] == binding["result_sha256"]
    assert result["equivalence_gate"]["passed"] is True
    assert result["unique_source_count"] == binding["unique_validation_sources"]
    assert result["totals"]["candidate_count"] == binding["candidate_count"]
    assert result["totals"]["admitted_count"] == binding["admitted_count"]
    assert (
        result["totals"]["ambiguous_rejection_count"]
        == binding["ambiguous_rejection_count"]
    )
    assert result["totals"]["mismatch_count"] == binding["mismatch_count"]
    assert (
        result["oracle_overflow_source_count"]
        == binding["oracle_overflow_source_count"]
    )


def test_development_diagnostics_are_not_promoted_to_authoritative_evidence() -> None:
    value = json.loads(CONTRACT.read_text())
    evidence = value["source_evidence"]
    assert evidence["atom_restate_development_diagnostic"][
        "authoritative_corpus_evidence"
    ] is False
    assert evidence["ring_system_restate_development_diagnostic"][
        "authoritative_corpus_evidence"
    ] is False
