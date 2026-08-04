"""Focused refusal tests for the Process-V2 T1 structural resolver."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    GATE_ZERO_DECISION_SCHEMA,
    GATE_ZERO_DECISION_SCHEMA_VERSION,
    PIPELINE_STATUS_NO_AUTHORITY,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    canonical_sha256,
)
from compose_v4.experiments.editing_v2_process_v2_t1_panel import ProcessV2T1PanelError
from compose_v4.experiments.editing_v2_process_v2_t1_resolve import (
    _validate_gate_zero_pass,
)


def _sha(character: str) -> str:
    return character * 64


def _inputs() -> tuple[dict[str, object], dict[str, object], SimpleNamespace]:
    completion = {
        "completion_sha256": _sha("1"),
        "sentinel": {"sentinel_sha256": _sha("2")},
    }
    contracts = SimpleNamespace(
        binding_sha256=_sha("3"),
        contract_sha256=_sha("4"),
        process_identity_sha256=_sha("5"),
    )
    body = {
        "schema": GATE_ZERO_DECISION_SCHEMA,
        "schema_version": GATE_ZERO_DECISION_SCHEMA_VERSION,
        "status": PIPELINE_STATUS_NO_AUTHORITY,
        **authority_false_block(),
        "decision": "PASS",
        "active8_completion_sha256": completion["completion_sha256"],
        "active8_sentinel_sha256": completion["sentinel"]["sentinel_sha256"],
        "contracts_binding_sha256": contracts.binding_sha256,
        "gate_zero_structural_contract_sha256": contracts.contract_sha256,
        "process_identity_sha256": contracts.process_identity_sha256,
        "source_index_sha256": _sha("6"),
        "checks": {"every_required_cell_present": True},
        "total_violations": 0,
        "missing_active8_families": [],
        "missing_required_cells": [],
    }
    return {**body, "decision_sha256": canonical_sha256(body)}, completion, contracts


def test_gate_zero_pass_must_bind_the_current_source_index() -> None:
    decision, completion, contracts = _inputs()

    with pytest.raises(ProcessV2T1PanelError, match="source_index_sha256"):
        _validate_gate_zero_pass(
            decision,
            completion=completion,
            source_index_sha256=_sha("7"),
            contracts=contracts,
        )


def test_gate_zero_pass_accepts_the_exact_bound_evidence() -> None:
    decision, completion, contracts = _inputs()

    assert _validate_gate_zero_pass(
        decision,
        completion=completion,
        source_index_sha256=_sha("6"),
        contracts=contracts,
    ) == decision


def test_gate_zero_pass_cannot_contradict_its_checks() -> None:
    decision, completion, contracts = _inputs()
    body = {key: value for key, value in decision.items() if key != "decision_sha256"}
    body["checks"] = {"every_required_cell_present": False}
    decision = {**body, "decision_sha256": canonical_sha256(body)}

    with pytest.raises(ProcessV2T1PanelError, match="contradicts"):
        _validate_gate_zero_pass(
            decision,
            completion=completion,
            source_index_sha256=_sha("6"),
            contracts=contracts,
        )


def test_gate_zero_fail_never_authorizes_t1() -> None:
    decision, completion, contracts = _inputs()
    body = {key: value for key, value in decision.items() if key != "decision_sha256"}
    body["decision"] = "FAIL"
    decision = {**body, "decision_sha256": canonical_sha256(body)}

    with pytest.raises(ProcessV2T1PanelError, match="requires an exact"):
        _validate_gate_zero_pass(
            decision,
            completion=completion,
            source_index_sha256=_sha("6"),
            contracts=contracts,
        )
