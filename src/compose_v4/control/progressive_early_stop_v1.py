"""Zero-oracle progressive support scheduling with durable plan receipts.

This module only decides how much generic source-only realization work remains.
It does not score candidates, call an oracle, apply target-specific rules, or
perform production endpoint admission.  A caller may use the returned census
to stop a generation batch once a small diverse support lock exists, or to
expand the next independent plan shard when it does not.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "t4_nodistill_progressive_early_stop_v1"
_RECEIPT_SCHEMA = "t4_nodistill_progressive_plan_receipt_v1"
_FORBIDDEN_KEYS = frozenset(
    {
        "target",
        "cell",
        "cell_key",
        "objective",
        "docking_score",
        "teacher",
        "teacher_endpoint",
        "teacher_action",
        "route",
        "route_id",
        "template",
        "template_id",
    }
)


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_no_forbidden_keys(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"forbidden runtime identity at {path}.{key}")
            _assert_no_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(child, path=f"{path}[{index}]")


@dataclass(frozen=True)
class ProgressiveEarlyStopPolicy:
    """Frozen generic work and support-lock policy.

    Eight attempts per plan is the initial budget.  Expansion is permitted only
    after the prior stage's durable receipt is complete and the census still
    lacks the diverse support lock.  The emitted lock is capped at 32 unique
    eligible endpoints, while the minimum stop threshold is 16.
    """

    initial_attempts_per_plan: int = 8
    expansion_attempts_per_plan: tuple[int, ...] = (8, 16, 32, 64, 128)
    minimum_unique_eligible: int = 16
    maximum_unique_eligible: int = 32
    minimum_macro_plan_identities: int = 3
    minimum_macro_modes: int = 3
    minimum_structural_signatures: int = 3

    def __post_init__(self) -> None:
        if self.initial_attempts_per_plan not in (4, 5, 6, 7, 8):
            raise ValueError("initial per-plan budget must be between 4 and 8")
        if not self.expansion_attempts_per_plan:
            raise ValueError("progressive expansion schedule cannot be empty")
        if self.expansion_attempts_per_plan[0] != self.initial_attempts_per_plan:
            raise ValueError("expansion schedule must begin at the initial budget")
        if any(
            type(value) is not int or value < 1
            for value in self.expansion_attempts_per_plan
        ):
            raise ValueError("expansion budgets must be positive integers")
        if (
            tuple(sorted(set(self.expansion_attempts_per_plan)))
            != self.expansion_attempts_per_plan
        ):
            raise ValueError("expansion budgets must be strictly increasing")
        if not 1 <= self.minimum_unique_eligible <= self.maximum_unique_eligible:
            raise ValueError("support stop bounds are inconsistent")
        if any(
            type(value) is not int or value < 1
            for value in (
                self.minimum_macro_plan_identities,
                self.minimum_macro_modes,
                self.minimum_structural_signatures,
            )
        ):
            raise ValueError("diversity floors must be positive integers")

    def budget_for_stage(self, stage: int) -> int:
        if type(stage) is not int or stage < 0:
            raise ValueError("progressive stage must be a non-negative integer")
        if stage >= len(self.expansion_attempts_per_plan):
            raise ValueError("no progressive budget remains")
        return self.expansion_attempts_per_plan[stage]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "initial_attempts_per_plan": self.initial_attempts_per_plan,
            "expansion_attempts_per_plan": list(self.expansion_attempts_per_plan),
            "minimum_unique_eligible": self.minimum_unique_eligible,
            "maximum_unique_eligible": self.maximum_unique_eligible,
            "diversity_floors": {
                "macro_plan_identities": self.minimum_macro_plan_identities,
                "macro_modes": self.minimum_macro_modes,
                "structural_signatures": self.minimum_structural_signatures,
            },
            "stop_rule": (
                "stop at the first complete stage with 16-32 unique eligible "
                "endpoints and all frozen diversity floors; expand only when "
                "the prior receipt is complete and the lock is insufficient"
            ),
        }


DEFAULT_POLICY = ProgressiveEarlyStopPolicy()


def _row_key(row: Mapping[str, Any]) -> str:
    key = row.get("endpoint_key") or row.get("canonical_smiles") or row.get("smiles")
    if not isinstance(key, str) or not key:
        raise ValueError("candidate row lacks a canonical endpoint key")
    return key


def _field(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate row lacks {name}")
    return value


def census_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: ProgressiveEarlyStopPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Return a deterministic zero-oracle support census and stop decision."""

    _assert_no_forbidden_keys(rows, path="candidates")
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("candidate rows must be mappings")
        if (
            row.get("eligible") is not True
            or row.get("exact_execution_verified") is not True
        ):
            continue
        unique.setdefault(_row_key(row), row)
    ordered = sorted(unique.items(), key=lambda item: item[0])
    capped = [row for _, row in ordered[: policy.maximum_unique_eligible]]
    plan_ids = {_field(row, "macro_plan_identity") for row in capped}
    modes = {_field(row, "macro_mode") for row in capped}
    signatures = {_field(row, "structural_signature") for row in capped}
    floors = {
        "macro_plan_identities": len(plan_ids),
        "macro_modes": len(modes),
        "structural_signatures": len(signatures),
    }
    diversity_ok = (
        floors["macro_plan_identities"] >= policy.minimum_macro_plan_identities
        and floors["macro_modes"] >= policy.minimum_macro_modes
        and floors["structural_signatures"] >= policy.minimum_structural_signatures
    )
    unique_count = len(unique)
    stop = unique_count >= policy.minimum_unique_eligible and diversity_ok
    return {
        "schema_version": SCHEMA_VERSION,
        "unique_eligible": unique_count,
        "locked_eligible": len(capped),
        "diversity": floors,
        "diversity_floors_met": diversity_ok,
        "stop": stop,
        "shortfall": max(0, policy.minimum_unique_eligible - unique_count),
        "locked_endpoint_keys": [_row_key(row) for row in capped],
    }


def next_stage(
    *,
    stage: int,
    receipt_complete: bool,
    census: Mapping[str, Any],
    policy: ProgressiveEarlyStopPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Plan one deterministic expansion, without silently backfilling."""

    if not receipt_complete:
        raise ValueError("adaptive expansion requires a complete durable receipt")
    if census.get("stop") is True:
        return {"action": "stop", "reason": "diverse_support_lock", "stage": stage}
    try:
        budget = policy.budget_for_stage(stage + 1)
    except ValueError:
        return {
            "action": "fail",
            "reason": "support_shortfall_at_max_budget",
            "stage": stage,
        }
    return {
        "action": "expand",
        "reason": "diverse_support_shortfall",
        "stage": stage + 1,
        "attempts_per_plan": budget,
    }


def make_plan_receipt(
    *,
    contract_payload_sha256: str,
    plan_id: str,
    stage: int,
    attempts_per_plan: int,
    source_key_sha256: str,
    candidates: Sequence[Mapping[str, Any]],
    census: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an atomically writable, self-hashed plan receipt payload."""

    if (
        not isinstance(contract_payload_sha256, str)
        or len(contract_payload_sha256) != 64
        or any(char not in "0123456789abcdef" for char in contract_payload_sha256)
    ):
        raise ValueError("contract payload identity must be a SHA-256 hex string")
    if not isinstance(plan_id, str) or not plan_id:
        raise ValueError("plan id is required")
    if type(stage) is not int or stage < 0:
        raise ValueError("receipt stage must be non-negative")
    if type(attempts_per_plan) is not int or attempts_per_plan < 1:
        raise ValueError("receipt attempt budget must be positive")
    if (
        not isinstance(source_key_sha256, str)
        or len(source_key_sha256) != 64
        or any(char not in "0123456789abcdef" for char in source_key_sha256)
    ):
        raise ValueError("source key identity must be a SHA-256 hex string")
    body = {
        "schema_version": _RECEIPT_SCHEMA,
        "contract_payload_sha256": contract_payload_sha256,
        "plan_id": plan_id,
        "stage": stage,
        "attempts_per_plan": attempts_per_plan,
        "source_key_sha256": source_key_sha256,
        "candidates": [dict(row) for row in candidates],
        "census": dict(census),
        "status": "complete",
    }
    _assert_no_forbidden_keys(body)
    return {"payload": body, "payload_sha256": _identity(body)}


def verify_plan_receipt(
    receipt: Mapping[str, Any],
    *,
    contract_payload_sha256: str,
    plan_id: str,
) -> dict[str, Any]:
    """Validate a receipt before resume; malformed receipts are never skipped."""

    if not isinstance(receipt, Mapping) or set(receipt) != {
        "payload",
        "payload_sha256",
    }:
        raise ValueError("receipt envelope is malformed")
    body = receipt["payload"]
    if not isinstance(body, Mapping):
        raise TypeError("receipt payload is not an object")
    if body.get("schema_version") != _RECEIPT_SCHEMA:
        raise ValueError("receipt schema mismatch")
    if body.get("contract_payload_sha256") != contract_payload_sha256:
        raise ValueError("receipt contract identity mismatch")
    if body.get("plan_id") != plan_id or body.get("status") != "complete":
        raise ValueError("receipt plan or completion state mismatch")
    if receipt.get("payload_sha256") != _identity(body):
        raise ValueError("receipt payload hash mismatch")
    _assert_no_forbidden_keys(body)
    return dict(body)


__all__ = [
    "DEFAULT_POLICY",
    "SCHEMA_VERSION",
    "ProgressiveEarlyStopPolicy",
    "census_candidates",
    "make_plan_receipt",
    "next_stage",
    "verify_plan_receipt",
]
