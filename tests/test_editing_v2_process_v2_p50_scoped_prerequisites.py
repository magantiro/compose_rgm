"""Current per-family T1 evidence boundary for the bounded P50 pilot."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.data.editing_v2_process_v2_schema import (
    canonical_bytes,
    canonical_sha256,
)
from compose_v4.experiments.editing_v2_process_v2_p50_prerequisites import (
    ProcessV2P50PrerequisiteError,
    load_process_v2_p50_scoped_prerequisites,
    validate_process_v2_p50_scoped_prerequisite_relationships,
)
from compose_v4.experiments.editing_v2_process_v2_t1_scoped_result import (
    build_process_v2_t1_scoped_capacity_decision,
    validate_process_v2_t1_scoped_capacity_result,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/process_v2_t1_scoped_capacity_result.json"
GATE_ZERO_FIXTURE = (
    ROOT / "tests/fixtures/process_v2_gate_zero_t1_bound_decision.json"
)


def _policy(name: str) -> dict[str, object]:
    return json.loads((ROOT / "configs" / name).read_text())


def _inputs() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    result = validate_process_v2_t1_scoped_capacity_result(
        json.loads(FIXTURE.read_text())
    )
    gate = json.loads(GATE_ZERO_FIXTURE.read_text())
    assert gate["decision_sha256"] == result["gate_zero_decision_sha256"]
    decision = build_process_v2_t1_scoped_capacity_decision(
        result, result_file_sha256="e" * 64
    )
    return (
        _policy("editing_v2_process_v2_p50_recipe_policy.json"),
        _policy("editing_v2_process_v2_t1_capacity_policy.json"),
        gate,
        result,
        decision,
    )


def test_scoped_prerequisites_refuse_superseded_family_receipt_chain() -> None:
    p50_policy, capacity_policy, gate, result, decision = _inputs()

    current_gate_sha256 = _policy(
        "editing_v2_process_v2_gate_zero_structural.json"
    )["contract_sha256"]
    assert gate["gate_zero_structural_contract_sha256"] != current_gate_sha256
    with pytest.raises(
        ProcessV2P50PrerequisiteError, match="identity disagrees"
    ):
        validate_process_v2_p50_scoped_prerequisite_relationships(
            p50_policy=p50_policy,
            capacity_policy=capacity_policy,
            gate_zero_decision=gate,
            t1_result=result,
            t1_decision=decision,
            t1_result_file_sha256="e" * 64,
        )


def test_scoped_prerequisites_preserve_a_recomputed_no_go() -> None:
    p50_policy, capacity_policy, gate, result, _decision = _inputs()
    failed = copy.deepcopy(result)
    failed["family_receipts"][0]["threshold_checks"]["every_entry_top1"] = False
    failed["all_required_families_passed"] = False
    body = {key: value for key, value in failed.items() if key != "result_sha256"}
    failed["result_sha256"] = canonical_sha256(body)
    decision = build_process_v2_t1_scoped_capacity_decision(
        failed, result_file_sha256="e" * 64
    )

    assert decision["bounded_p50_authorized"] is False
    with pytest.raises(ProcessV2P50PrerequisiteError, match="not a bounded-P50 GO"):
        validate_process_v2_p50_scoped_prerequisite_relationships(
            p50_policy=p50_policy,
            capacity_policy=capacity_policy,
            gate_zero_decision=gate,
            t1_result=failed,
            t1_decision=decision,
            t1_result_file_sha256="e" * 64,
        )


def test_scoped_prerequisites_refuse_cross_run_gate_zero() -> None:
    p50_policy, capacity_policy, gate, result, decision = _inputs()
    changed = {**gate, "active8_completion_sha256": "1" * 64}
    changed["decision_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "decision_sha256"}
    )

    with pytest.raises(ProcessV2P50PrerequisiteError, match="identity disagrees"):
        validate_process_v2_p50_scoped_prerequisite_relationships(
            p50_policy=p50_policy,
            capacity_policy=capacity_policy,
            gate_zero_decision=changed,
            t1_result=result,
            t1_decision=decision,
            t1_result_file_sha256="e" * 64,
        )


def test_scoped_prerequisites_refuse_superseded_artifact_bytes(tmp_path: Path) -> None:
    _p50, _capacity, gate, result, _decision = _inputs()
    gate_path = tmp_path / "gate-zero.json"
    result_path = tmp_path / "t1-scoped-result.json"
    decision_path = tmp_path / "t1-scoped-decision.json"
    gate_path.write_bytes(canonical_bytes(gate) + b"\n")
    result_bytes = canonical_bytes(result) + b"\n"
    result_path.write_bytes(result_bytes)
    decision = build_process_v2_t1_scoped_capacity_decision(
        result, result_file_sha256=hashlib.sha256(result_bytes).hexdigest()
    )
    decision_path.write_bytes(canonical_bytes(decision) + b"\n")

    with pytest.raises(
        ProcessV2P50PrerequisiteError, match="identity disagrees"
    ):
        load_process_v2_p50_scoped_prerequisites(
            gate_zero_decision_path=gate_path,
            t1_result_path=result_path,
            t1_decision_path=decision_path,
            repo_root=ROOT,
        )
