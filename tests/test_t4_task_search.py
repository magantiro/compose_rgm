"""Prepare-only integration with synthetic labels and the production executor."""

import copy
import json

import pytest

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.t4_matched_pilot import seal
from compose_v4.experiments.t4_task_search import PreparationConfig, prepare
from compose_v4.experiments.t4_task_search_audit import run_audit, summarize, verify_lock
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.trace_shard import decode_state, encode_state


def test_preparation_is_round_frozen_bounded_and_never_docks():
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
        executor_per_parent=128,
        planning_executor_per_parent=16,
        max_rows=32,
        max_rollouts=2,
        initial_rollouts=1,
        rollouts_per_decision=1,
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
    tampered = copy.deepcopy(result)
    tampered["pool"][0]["r_coherent"] += 0.1
    with pytest.raises(ValueError, match="structural metadata"):
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
