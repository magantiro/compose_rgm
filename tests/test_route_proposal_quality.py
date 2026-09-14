from __future__ import annotations

import json

import pytest

from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from compose_v4.experiments.route_proposal_quality import (
    evaluate,
    transformation_equivalent,
)
from compose_v4.rewrite.trace_shard import encode_state
from tools.t4_route_proposal_quality import run


def _state(smiles):
    return encode_state(production_state_from_smiles(smiles, max_atoms=48))


def _payload():
    attempts = [
        {
            "attempt_id": "proposal-1",
            "rank": 1,
            "status": "complete",
            "endpoint_state": _state("CCC"),
        },
        {
            "attempt_id": "proposal-2",
            "rank": 2,
            "status": "complete",
            "endpoint_state": _state("CCN"),
        },
        {
            "attempt_id": "proposal-3",
            "rank": 3,
            "status": "rejected",
            "endpoint_state": None,
        },
    ]
    return {
        "schema_version": "route_proposal_quality_input_v1",
        "oracle_calls": 0,
        "split": {
            "evaluation_role": "test",
            "train_sources": ["train-source"],
            "calibration_sources": ["calibration-source"],
            "test_sources": ["test-source"],
        },
        "cases": [
            {
                "source_id": "test-source",
                "source_state": _state("CCO"),
                "teachers": [
                    {"teacher_id": "teacher-1", "endpoint_state": _state("CCN")}
                ],
                "policy_pools": [
                    {
                        "policy_id": "generic_marginal",
                        "proposal_seconds": 2.0,
                        "attempts": attempts,
                    }
                ],
            }
        ],
    }


def test_source_balanced_endpoint_recall_and_compute_metrics():
    result = evaluate(_payload())
    aggregate = result["policies"]["generic_marginal"]["aggregate"]
    assert aggregate["cutoffs"]["1"]["exact_recall"] == 0
    assert aggregate["cutoffs"]["5"]["exact_recall"] == 1
    assert aggregate["cutoffs"]["5"]["transformation_recall"] == 1
    assert aggregate["execution_precision"] == pytest.approx(2 / 3)
    assert aggregate["unique_endpoint_yield"] == pytest.approx(2 / 3)
    assert aggregate["source_balanced_exact_mrr"] == pytest.approx(0.5)
    assert result["oracle_calls"] == 0


def test_transformation_equivalence_is_not_an_edit_count_shortcut():
    source = production_state_from_smiles("CCCO", max_atoms=48)
    terminal_restate = production_state_from_smiles("NCCO", max_atoms=48)
    internal_restate = production_state_from_smiles("CNCO", max_atoms=48)
    assert transformation_equivalent(source, terminal_restate, terminal_restate)
    assert not transformation_equivalent(source, terminal_restate, internal_restate)


def test_split_leakage_and_nonzero_oracle_calls_fail_closed():
    payload = _payload()
    payload["split"]["train_sources"].append("test-source")
    with pytest.raises(ValueError, match="partitions"):
        evaluate(payload)
    payload = _payload()
    payload["oracle_calls"] = 1
    with pytest.raises(ValueError, match="zero-oracle"):
        evaluate(payload)


def test_standalone_runner_seals_and_refuses_overwrite(tmp_path):
    source = tmp_path / "proposal_pools.json"
    output = tmp_path / "result.json"
    source.write_text(json.dumps(_payload()))
    result = run(source, output)
    saved = json.loads(output.read_text())
    assert saved["payload"]["report_sha256"] == result["report_sha256"]
    assert saved["payload"]["oracle_calls"] == 0
    with pytest.raises(ValueError, match="refusing to overwrite"):
        run(source, output)
