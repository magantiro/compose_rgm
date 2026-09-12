"""Tiny orchestration receipts, no network, no docking or model checkpoints."""

import copy

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.t4_frontier_compare import (
    ARMS,
    dock_locked_pair,
    normalize_warm_metadata,
)
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state


def inputs():
    warm = {"round": 4, "oracle_attempts": 51, "archive": [{"smiles": "C", "ds": None}]}
    locks = {
        arm: {
            "schema_version": "t4_frontier_oracle_lock_v1",
            "round": 5,
            "prior_oracle_attempts": 51,
            "config": {"guidance": arm},
            "take": [{"smiles": "CC", "option": "generic", "allocated_bundle_id": arm}],
        }
        for arm in ARMS
    }
    return locks, warm


def test_both_locks_precede_oracles_and_completed_calls_are_not_repeated(tmp_path):
    locks, warm = inputs()
    calls = []

    def dock(smiles, tag):
        assert set(unseal(tmp_path / "oracle_barrier.json")["lock_sha256"]) == set(ARMS)
        calls.append((smiles, tag))
        return -9.2 if len(calls) == 1 else None

    result = dock_locked_pair(locks, warm, tmp_path, dock)
    assert result["new_oracle_attempts"] == 2
    assert result["arms"]["in_loop"]["failed_dockings"] == 1
    assert result["arms"]["post_hoc"]["total_oracle_attempts"] == 52
    repeated = dock_locked_pair(locks, warm, tmp_path, dock)
    assert repeated == result and len(calls) == 2
    changed = copy.deepcopy(locks)
    changed["in_loop"]["take"][0]["smiles"] = "CCC"
    with pytest.raises(ValueError, match="barrier"):
        dock_locked_pair(changed, warm, tmp_path, dock)


def test_unknown_oracle_attempt_blocks_automatic_retry(tmp_path):
    locks, warm = inputs()
    calls = []

    def interrupted(*args):
        calls.append(args)
        raise OSError("simulated worker disconnect")

    with pytest.raises(OSError):
        dock_locked_pair(locks, warm, tmp_path, interrupted)
    with pytest.raises(RuntimeError, match="unknown interrupted oracle"):
        dock_locked_pair(locks, warm, tmp_path, interrupted)
    assert len(calls) == 1
    assert unseal(tmp_path / "post_hoc/oracle/000/started.json")["charged_attempts"] == 1


def test_missing_arm_and_exceeded_allowance_fail_before_any_oracle(tmp_path):
    locks, warm = inputs()

    def forbidden(*args):
        raise AssertionError("must reject before docking")

    with pytest.raises(ValueError, match="both audited"):
        dock_locked_pair({"post_hoc": locks["post_hoc"]}, warm, tmp_path, forbidden)
    locks["in_loop"]["take"] *= 21
    with pytest.raises(ValueError, match="budget"):
        dock_locked_pair(locks, warm, tmp_path, forbidden)


def test_warm_metadata_migration_changes_only_proven_equivalent_spelling():
    graph = pad_molecular_graph(smiles_to_molecular_graph("CO"), 48)
    row = {"smiles": "OC", "state": encode_state(graph), "ds": None, "round": 0}
    normalized = normalize_warm_metadata({"archive": [row]}, "a" * 64)
    assert normalized["archive"][0]["smiles"] == canonical_state_key(graph)
    assert normalized["archive"][0]["state"] == row["state"]
    assert normalized["source_metadata_normalization"]["corrections"] == [
        {
            "index": 0,
            "original_smiles": "OC",
            "canonical_smiles": canonical_state_key(graph),
            "exact_state_sha256": normalized["source_metadata_normalization"]["corrections"][0][
                "exact_state_sha256"
            ],
        }
    ]

    bad = copy.deepcopy({"archive": [row]})
    bad["archive"][0]["smiles"] = "CN"
    with pytest.raises(ValueError, match="differs from exact saved state"):
        normalize_warm_metadata(bad, "a" * 64)
