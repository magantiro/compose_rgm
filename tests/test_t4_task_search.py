"""Prepare-only integration with synthetic labels and the production executor."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.t4_matched_pilot import canonical_bytes, seal
from compose_v4.experiments.t4_task_search import PreparationConfig, prepare
from compose_v4.experiments.t4_task_search_audit import run_audit, summarize, verify_lock
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.trace_shard import decode_state, encode_state


@pytest.mark.parametrize("planning_policy", ["adaptive_full_rows", "lazy_reference"])
@pytest.mark.parametrize("uncapped", [False, True])
def test_preparation_is_round_frozen_bounded_and_never_docks(planning_policy, uncapped):
    rows = []
    for i in range(1, 18):
        graph = pad_molecular_graph(smiles_to_molecular_graph("C" * i), 48)
        rows.append(
            {
                "smiles": canonical_state_key(graph),
                "state": encode_state(graph),
                "ds": None if i == 1 else -float(i),
                "round": 0 if i == 1 else 1,
            }
        )

    def fixture_law(graph):
        # Synthetic one-action reference. Labels above are artificial as well.
        real = graph.n_real_atoms
        return (("atom_insert",), (AtomInsert(real, ELEMENT_TO_IDX["C"], 0, 3, ((0, 1),)),), (1.0,))

    cfg = PreparationConfig(
        lineages=1,
        primitive_budget=2,
        executor_per_parent=None if uncapped else 128,
        planning_executor_per_parent=None if uncapped else 16,
        max_rows=None if uncapped else 32,
        max_rollouts=2,
        initial_rollouts=1,
        rollouts_per_decision=1,
        planning_policy=planning_policy,
        compute_policy="metered_uncapped_v1" if uncapped else "bounded_v1",
    )
    warm = {
        "schema_version": "t4_exact_archive_v1",
        "round": 1,
        "archive": rows,
        "oracle_attempts": 16,
    }
    checkpoints = []
    result = prepare(
        warm,
        source_sha256="a" * 64,
        enumerate_law=fixture_law,
        system=editing_v2_rewrite_system(),
        input_sha256={},
        config=cfg,
        progress=checkpoints.append,
    )
    assert result["schema_version"] == "t4_hierarchical_candidate_lock_v2"
    assert result["new_oracle_calls"] == 0 and not result["automatic_docking"]
    assert result["prior_oracle_attempts"] == 16
    assert result["executor_calls"] <= 128
    assert result["executor_calls"] > 0
    assert result["work"][0]["planning_budget"]["executor_calls"] > 0
    assert sum(c["schema_version"] == "t4_task_search_parent_v2" for c in checkpoints) == 1
    assert result["value_snapshot"]["before_round"] == 2
    assert all(r["round"] < 2 for r in result["value_snapshot"]["training_rows"])
    for unit in result["work"]:
        assert unit["path"] and unit["completed_options"] >= 1
        for event in unit["path"]:
            assert event["decision"]["kl"] <= 1 + 1e-10
            assert canonical_state_key(decode_state(event["product"]))
    for candidate in result["take"]:
        assert candidate["program_complete"] and candidate["v"] == 0
        assert canonical_state_key(decode_state(candidate["state"])) == candidate["smiles"]
    json.dumps(result, allow_nan=False)
    verification = verify_lock(result, warm, editing_v2_rewrite_system())
    assert verification["replayed_steps"] >= 1
    assert summarize(result, verification)["decision"] == "diagnose_before_docking"
    assert result["pool"]
    if planning_policy == "lazy_reference":
        assert result["work"][0]["planner"]["reference_draws"] > 0
        assert result["work"][0]["planner"]["rollouts_completed"] > 0
    tampered = copy.deepcopy(result)
    tampered["pool"][0]["r_coherent"] += 0.1
    with pytest.raises(ValueError, match="structural metadata"):
        verify_lock(tampered, warm, editing_v2_rewrite_system())
    if uncapped:
        tampered = copy.deepcopy(result)
        tampered["work"][0]["planning_budget"]["executor_calls"] = result["executor_calls"] + 1
        with pytest.raises(ValueError, match="planning exceeded"):
            verify_lock(tampered, warm, editing_v2_rewrite_system())

    def forbidden_law(graph):
        raise AssertionError("complete parents must not be regenerated")

    resumed = prepare(
        warm,
        source_sha256="a" * 64,
        enumerate_law=forbidden_law,
        system=editing_v2_rewrite_system(),
        input_sha256={},
        config=cfg,
        parent_cache={0: result["work"][0]},
    )
    assert resumed["new_executor_calls"] == 0
    for key in ("pool", "take", "work", "executor_calls", "value_snapshot"):
        assert resumed[key] == result[key]


def test_unfinished_parent_is_not_automatically_replayed(tmp_path):
    seal(tmp_path / "started/00.json", {"parent_index": 0})
    with pytest.raises(RuntimeError, match="unfinished started parent"):
        run_audit({}, {}, None, None, tmp_path)


def test_planning_policy_extension_preserves_legacy_config_identity():
    cfg = PreparationConfig()
    assert "planning_policy" not in cfg.payload()
    assert "compute_policy" not in cfg.payload()
    assert PreparationConfig(**cfg.payload()) == cfg
    assert (
        PreparationConfig(planning_policy="lazy_reference").payload()["planning_policy"]
        == "lazy_reference"
    )
    with pytest.raises(ValueError, match="planning policy"):
        PreparationConfig(planning_policy="unknown")


def test_uncapped_config_is_explicit_and_preserves_legacy_guards():
    cfg = PreparationConfig(
        compute_policy="metered_uncapped_v1",
        executor_per_parent=None,
        planning_executor_per_parent=None,
        max_rows=None,
    )
    assert PreparationConfig(**cfg.payload()) == cfg
    with pytest.raises(ValueError, match="uncapped preparation"):
        PreparationConfig(compute_policy="metered_uncapped_v1")
    with pytest.raises(ValueError, match="invalid preparation field"):
        PreparationConfig(planning_executor_per_parent=None)
    with pytest.raises(ValueError, match="bounded development"):
        PreparationConfig(executor_per_parent=2501)


def test_uncapped_contract_only_changes_compute_policy_and_run_boundary():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "configs/t4_lazy_reference_probe.json").read_text())
    new = json.loads((root / "configs/t4_uncapped_lookahead_probe.json").read_text())
    digest = new.pop("contract_sha256")
    assert hashlib.sha256(canonical_bytes(new)).hexdigest() == digest
    for field in ("source", "value_check", "expected_input_sha256", "task", "required_rdkit"):
        assert new[field] == old[field]
    assert new["compute"]["oracle_call_limit"] == 0
    assert new["compute"]["automatic_retries"] == 0
    assert new["compute"]["timeout_seconds"] == 7200
    cfg = PreparationConfig(**new["preparation"])
    assert cfg.compute_policy == "metered_uncapped_v1"
    equivalent = cfg.payload()
    equivalent.pop("compute_policy")
    for field in ("executor_per_parent", "planning_executor_per_parent", "max_rows"):
        assert equivalent[field] is None
        equivalent[field] = old["preparation"][field]
    assert equivalent == old["preparation"]
