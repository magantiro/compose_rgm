import json

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.pmo_option_controller_bank import (
    PREPARED,
    candidate_lock,
    load_contract,
    prepare,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state


def test_prepare_selects_top_exact_parents_and_denies_remote_launch(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "docs/PMO_OPTION_CONTROLLER_BANK.md").write_text("frozen\n")
    online = {
        "schema_version": "fixture",
        "expected_input_sha256": {"r_theta_checkpoint": "a", "r_theta_run_paths": "b"},
    }
    online["contract_sha256"] = identity(online)
    publish_json(tmp_path / "configs/pmo_online_policy.json", online)

    source_rows = []
    for index in range(9):
        graph = pad_molecular_graph(smiles_to_molecular_graph("C" * (index + 1)), 16)
        smiles = canonical_state_key(graph)
        trace = tmp_path / f"trace-{index}.json"
        publish_json(trace, {"states": [encode_state(graph)]})
        source_rows.append(
            {
                "candidate_id": f"candidate-{index}",
                "source": smiles,
                "parent_score": index / 10,
                "path": str(trace),
                "sha256": sha256_file(trace),
            }
        )
    rows = [dict(source_rows[index % len(source_rows)]) for index in range(95)]
    for index, row in enumerate(rows):
        row["candidate_id"] = f"candidate-{index}"
    report = {
        "schema_version": "pmo_plan_pool_prevalence_result_v1",
        "task": "perindopril_mpo",
        "new_oracle_calls": 95,
        "analysis_commit": "0" * 40,
        "scored": rows,
    }
    report["content_sha256"] = identity(report)
    report_path = tmp_path / "diagnostics/report.json"
    publish_json(report_path, report)

    contract = prepare(tmp_path, report_path)
    prepared = json.loads((tmp_path / PREPARED).read_text())
    assert [row["parent_score"] for row in prepared["parents"]] == pytest.approx(
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    )
    assert all(
        identity(row["exact_state"]) == row["exact_state_id"]
        for row in prepared["parents"]
    )
    assert contract["oracle_authorized"] is False
    with pytest.raises(PermissionError, match="not authorized"):
        load_contract(tmp_path, require_proposal_authority=True)


def _candidate(candidate_id, stream_id, boundary, exact_id):
    return {
        "candidate_id": candidate_id,
        "canonical_smiles": "CC",
        "exact_state": {"variant": exact_id},
        "exact_state_id": exact_id,
        "source_id": "source",
        "source_smiles": "C",
        "parent_score": 0.5,
        "stream_index": 0,
        "stream_id": stream_id,
        "option_boundary": boundary,
        "particle_id": stream_id,
        "history": [{"option": "generic"}],
    }


def test_candidate_lock_deduplicates_only_for_scoring_and_keeps_origins():
    workers = [
        {
            "status": "complete",
            "new_oracle_calls": 0,
            "stream_id": "stream-b",
            "candidates": [_candidate("b", "stream-b", 1, "exact-b")],
        },
        {
            "status": "complete",
            "new_oracle_calls": 0,
            "stream_id": "stream-a",
            "candidates": [_candidate("a", "stream-a", 2, "exact-a")],
        },
    ]
    locked = candidate_lock(workers)
    assert locked["trajectory_boundary_candidates"] == 2
    assert locked["unique_canonical_candidates"] == 1
    candidate = locked["candidates"][0]
    assert candidate["exact_state_ids"] == ["exact-a", "exact-b"]
    assert [row["candidate_id"] for row in candidate["origins"]] == ["a", "b"]
    assert candidate["oracle_status"] == "unrequested"


def test_zero_actor_preserves_balanced_option_reference():
    from compose_v4.experiments.pmo_option_controller_bank import _zero_controller

    controller = _zero_controller()
    reference = np.asarray([0.2, 0.3, 0.5])
    choice = controller.decide(
        ("generic", "grow", "cyclize"),
        reference,
        np.zeros(controller.actor.state_dim, dtype=np.float32),
        np.random.default_rng(2),
    )
    assert choice.distribution.proposal == pytest.approx(reference)
    assert choice.distribution.kl == pytest.approx(0.0, abs=1e-12)
