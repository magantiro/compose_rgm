import json
from pathlib import Path

from compose_v4.experiments.t4_stale_braf_v4_reconciliation import (
    EXPECTED_RECEIPTS,
    canonical_smiles_sha256,
    validate_reconciliation_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "diagnostics/t4_shared_controller_completion_v1/stale_braf_v4_reconciliation.json"
)


def _walk_keys(value) -> set[str]:
    if isinstance(value, dict):
        result = set(map(str, value))
        for item in value.values():
            result.update(_walk_keys(item))
        return result
    if isinstance(value, list):
        result = set()
        for item in value:
            result.update(_walk_keys(item))
        return result
    return set()


def test_canonical_query_identity_is_deterministic_and_representation_invariant():
    first = canonical_smiles_sha256("CCO")
    assert first == canonical_smiles_sha256("OCC")
    assert first == canonical_smiles_sha256("CCO")
    assert len(first) == 64


def test_checked_in_reconciliation_is_self_hashed_complete_and_score_free():
    validated = validate_reconciliation_artifact(ARTIFACT)
    payload = validated["payload"]
    assert payload["charged_complete_receipts"] == 43
    assert payload["unresolved_locked_queries"] == 0
    exclusions = payload["excluded_canonical_smiles_sha256"]
    assert {
        cell: len(identities) for cell, identities in exclusions.items()
    } == EXPECTED_RECEIPTS
    assert all(values == sorted(set(values)) for values in exclusions.values())
    assert not {
        "answer",
        "best",
        "best_so_far",
        "final_best",
        "parent_score",
        "round_best",
        "score",
        "scores",
        "smiles",
        "winner",
    }.intersection(_walk_keys(payload))

    serialized = json.dumps(payload, sort_keys=True)
    assert "-9." not in serialized
    assert "-10." not in serialized
    assert payload["counts"]["json_envelopes"] == 184
    assert payload["counts"]["locked_queries"] == 43
    assert payload["counts"]["canonical_denylist_entries"] == 43
    assert payload["content_policy"] == {
        "raw_smiles_materialized": False,
        "scores_materialized": False,
        "terminal_boundary": (
            "round_004_proposals_exist_without_any_round_004_query_lock"
        ),
    }
