"""Focused tests for the zero-oracle progressive support policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.control.progressive_early_stop_v1 import (
    DEFAULT_POLICY,
    ProgressiveEarlyStopPolicy,
    census_candidates,
    make_plan_receipt,
    next_stage,
    verify_plan_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


def _row(index: int, *, eligible: bool = True, exact: bool = True) -> dict[str, object]:
    return {
        "endpoint_key": f"C{index:02d}",
        "eligible": eligible,
        "exact_execution_verified": exact,
        "macro_plan_identity": f"plan-{index % 3}",
        "macro_mode": f"mode-{index % 3}",
        "structural_signature": f"sig-{index % 3}",
    }


def _payload_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_policy_has_small_initial_budget_and_strict_expansion() -> None:
    assert DEFAULT_POLICY.initial_attempts_per_plan == 8
    assert DEFAULT_POLICY.budget_for_stage(0) == 8
    assert DEFAULT_POLICY.budget_for_stage(1) == 16
    with pytest.raises(ValueError, match="strictly increasing"):
        ProgressiveEarlyStopPolicy(expansion_attempts_per_plan=(8, 8))
    with pytest.raises(ValueError, match="between 4 and 8"):
        ProgressiveEarlyStopPolicy(initial_attempts_per_plan=9)


def test_census_deduplicates_and_stops_only_on_diverse_exact_support() -> None:
    rows = [_row(index) for index in range(20)]
    rows.extend([_row(0), _row(1, eligible=False), _row(2, exact=False)])
    census = census_candidates(rows)
    assert census["unique_eligible"] == 20
    assert census["locked_eligible"] == 20
    assert census["diversity"] == {
        "macro_plan_identities": 3,
        "macro_modes": 3,
        "structural_signatures": 3,
    }
    assert census["stop"] is True

    low_diversity = [_row(index) | {"macro_mode": "one"} for index in range(20)]
    census = census_candidates(low_diversity)
    assert census["unique_eligible"] == 20
    assert census["diversity_floors_met"] is False
    assert census["stop"] is False
    assert (
        next_stage(stage=0, receipt_complete=True, census=census)["action"] == "expand"
    )


def test_progression_stops_or_fails_without_incomplete_receipt() -> None:
    sufficient = census_candidates([_row(index) for index in range(16)])
    assert next_stage(stage=0, receipt_complete=True, census=sufficient) == {
        "action": "stop",
        "reason": "diverse_support_lock",
        "stage": 0,
    }
    with pytest.raises(ValueError, match="complete durable receipt"):
        next_stage(stage=0, receipt_complete=False, census={"stop": False})
    insufficient = {"stop": False}
    terminal = next_stage(
        stage=len(DEFAULT_POLICY.expansion_attempts_per_plan) - 1,
        receipt_complete=True,
        census=insufficient,
    )
    assert terminal == {
        "action": "fail",
        "reason": "support_shortfall_at_max_budget",
        "stage": 4,
    }


def test_receipt_round_trip_and_tamper_detection() -> None:
    contract = "a" * 64
    census = census_candidates([_row(index) for index in range(16)])
    receipt = make_plan_receipt(
        contract_payload_sha256=contract,
        plan_id="plan-0",
        stage=0,
        attempts_per_plan=8,
        source_key_sha256="b" * 64,
        candidates=[_row(index) for index in range(3)],
        census=census,
    )
    verified = verify_plan_receipt(
        receipt, contract_payload_sha256=contract, plan_id="plan-0"
    )
    assert verified["status"] == "complete"
    tampered = {
        "payload": dict(receipt["payload"]),
        "payload_sha256": receipt["payload_sha256"],
    }
    tampered["payload"]["stage"] = 1
    with pytest.raises(ValueError, match="payload hash"):
        verify_plan_receipt(
            tampered, contract_payload_sha256=contract, plan_id="plan-0"
        )
    with pytest.raises(ValueError, match="forbidden"):
        make_plan_receipt(
            contract_payload_sha256=contract,
            plan_id="plan-0",
            stage=0,
            attempts_per_plan=8,
            source_key_sha256="b" * 64,
            candidates=[_row(0) | {"target": "forbidden"}],
            census=census,
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/t4_nodistill_progressive_early_stop_v1.json",
        "configs/t4_pre_v4_baseline_fa7_0_5ht1b_0_dedup_v1.json",
    ],
)
def test_frozen_config_payload_hashes_are_bound(relative_path: str) -> None:
    config = json.loads((ROOT / relative_path).read_text())
    expected = config["contract_payload_sha256"]
    payload = dict(config)
    payload.pop("contract_payload_sha256")
    assert expected == _payload_sha256(payload)
    assert len(expected) == 64
