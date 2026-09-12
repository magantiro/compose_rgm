"""Tiny orchestration receipts, no network, no docking or model checkpoints."""

import copy
import json

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments import t4_frontier_audit
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_frontier_audit import matching_replayed_successor
from compose_v4.experiments.t4_frontier_compare import (
    ARMS,
    dock_locked_pair,
    load_reused_preparations,
    normalize_warm_metadata,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal
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
    assert (
        unseal(tmp_path / "post_hoc/oracle/000/started.json")["charged_attempts"] == 1
    )


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
            "exact_state_sha256": normalized["source_metadata_normalization"][
                "corrections"
            ][0]["exact_state_sha256"],
        }
    ]

    bad = copy.deepcopy({"archive": [row]})
    bad["archive"][0]["smiles"] = "CN"
    with pytest.raises(ValueError, match="differs from exact saved state"):
        normalize_warm_metadata(bad, "a" * 64)


def test_pre_oracle_reuse_authenticates_every_partition_and_blocks_crossed_barrier(
    tmp_path,
):
    run_id = "1" * 64
    revision = "2" * 40
    source_contract_file = "3" * 64
    expected_inputs = {"seed_manifest": "4" * 64, "r_theta_checkpoint": "5" * 64}
    source_sha256 = "6" * 64
    source_root = tmp_path / "t4_frontier_compare" / run_id
    source_root.mkdir(parents=True)
    launch = {
        "run_id": run_id,
        "contract_sha256": source_contract_file,
        "image_revision": {
            "commit": revision,
            "serialized_sources": {
                "configs/t4_frontier_compare.json": source_contract_file
            },
        },
    }
    failure = {
        "run_id": run_id,
        "phase": "lineage_reduction",
        "error_type": "ValueError",
        "preparation_calls": {
            f"{arm}:{i}": f"call-{arm}-{i}" for arm in ARMS for i in range(8)
        },
    }
    launch_path, failure_path = (
        source_root / "launch.json",
        source_root / "failure.json",
    )
    launch_path.write_text(json.dumps(launch, sort_keys=True))
    failure_path.write_text(json.dumps(failure, sort_keys=True))
    configs = {
        arm: {
            "lineages": 8,
            "primitive_budget": 16,
            "planning_transitions": 0 if arm == "post_hoc" else 2,
            "oracle_batch": 20,
            "seed": 1000,
            "guidance": arm,
        }
        for arm in ARMS
    }
    registered = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for lineage in range(8):
            path = (
                tmp_path
                / "t4_frontier_compare"
                / arm
                / f"lineage_{lineage:02d}"
                / run_id
                / "preparation.json"
            )
            seal(
                path,
                {
                    "checkpoint": {
                        "code_revision": revision,
                        "source_sha256": source_sha256,
                        "config": configs[arm],
                        "partition_lineages": [lineage],
                        "input_sha256": {**expected_inputs, "implementation": "7" * 64},
                        "checkpoint_sha256": f"checkpoint-{arm}-{lineage}",
                    },
                    "worker": {"arm": arm, "lineage_index": lineage},
                },
            )
            registered[arm].append(
                {
                    "path": str(path.relative_to(tmp_path)),
                    "sha256": sha256_file(path),
                }
            )
    contract = {
        "source": {"sha256": source_sha256},
        "expected_input_sha256": expected_inputs,
        "arms": configs,
        "preparation_reuse": {
            "schema_version": "t4_frontier_preparation_reuse_v1",
            "source_run_id": run_id,
            "source_code_revision": revision,
            "source_contract_file_sha256": source_contract_file,
            "source_failure_phase": "lineage_reduction",
            "source_launch": {
                "path": str(launch_path.relative_to(tmp_path)),
                "sha256": sha256_file(launch_path),
            },
            "source_failure": {
                "path": str(failure_path.relative_to(tmp_path)),
                "sha256": sha256_file(failure_path),
            },
            "preparations": registered,
        },
    }
    paths, audit = load_reused_preparations(tmp_path, contract)
    assert all(len(paths[arm]) == 8 for arm in ARMS)
    assert audit["source_oracle_barrier_absent"] is True
    assert audit["source_oracle_attempt_artifacts"] == 0

    (source_root / "oracle_barrier.json").write_text("{}")
    with pytest.raises(ValueError, match="crossed the oracle barrier"):
        load_reused_preparations(tmp_path, contract)


def test_replay_uses_exact_augmented_product_to_disambiguate_one_physical_mark(
    monkeypatch,
):
    monkeypatch.setattr(t4_frontier_audit, "state_payload", lambda value: value)
    expected = {"branch": "right", "graph": "same physical molecule"}
    successors = [
        {"branch": "left", "graph": "same physical molecule"},
        expected,
    ]
    assert matching_replayed_successor(successors, expected) is expected
    with pytest.raises(ValueError, match="uniquely recover"):
        matching_replayed_successor([successors[0]], expected)
    with pytest.raises(ValueError, match="uniquely recover"):
        matching_replayed_successor([expected, copy.deepcopy(expected)], expected)
