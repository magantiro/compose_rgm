"""Pure recovery for an already locked shared-controller scoring round.

This module deliberately starts after proposal generation and selection.  It
validates one immutable round plan, query lock, dispatch intent, reservations,
and the available query receipts, then applies the frozen missing-query policy
exactly once.  It never proposes, selects, docks, retries, or replaces a
candidate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from compose_v4.experiments.t4_shared_controller_cell_runtime import (
    settle_query_lock,
)
from compose_v4.experiments.t4_shared_controller_checkpoint import (
    apply_settled_observations,
)
from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
)


def _require_identity(
    value: Mapping[str, Any], expected: str, *, label: str
) -> dict[str, Any]:
    payload = dict(value)
    observed = payload_identity(payload)
    if observed != expected:
        raise ValueError(f"{label} identity drift: expected {expected}, observed {observed}")
    return payload


def recover_locked_round(
    *,
    checkpoint: Mapping[str, Any],
    round_plan: Mapping[str, Any],
    query_lock: Mapping[str, Any],
    dispatch_intent: Mapping[str, Any],
    reservations: Mapping[str, Mapping[str, Any]],
    receipts: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Any],
    cell: Mapping[str, Any],
    controller_config: Mapping[str, Any],
    budget_ceiling: int,
    now: float,
) -> dict[str, Any]:
    """Settle one immutable lock after its original driver became terminal."""

    frozen_checkpoint = _require_identity(
        checkpoint, str(expected["checkpoint_payload_sha256"]), label="checkpoint"
    )
    plan = _require_identity(
        round_plan, str(expected["round_plan_payload_sha256"]), label="round plan"
    )
    lock = _require_identity(
        query_lock, str(expected["query_lock_payload_sha256"]), label="query lock"
    )
    dispatch = _require_identity(
        dispatch_intent,
        str(expected["query_dispatch_payload_sha256"]),
        label="query dispatch",
    )
    if plan.get("query_lock") != lock:
        raise ValueError("round plan does not contain the immutable query lock")
    if dispatch.get("query_lock_payload_sha256") != payload_identity(lock):
        raise ValueError("query dispatch does not bind the immutable query lock")
    if dispatch.get("automatic_retries") != 0:
        raise ValueError("query dispatch retry policy drift")

    queries = lock.get("queries")
    if not isinstance(queries, list):
        raise TypeError("query lock omitted its query rows")
    query_ids = [str(row.get("query_id")) for row in queries]
    if query_ids != list(expected["query_ids"]):
        raise ValueError("query-lock query census drift")
    if sorted(reservations) != sorted(query_ids):
        raise ValueError("reservation census differs from the immutable lock")

    expected_reservations = expected["reservation_payload_sha256"]
    expected_receipts = expected["receipt_payload_sha256"]
    if not isinstance(expected_reservations, Mapping) or not isinstance(
        expected_receipts, Mapping
    ):
        raise TypeError("recovery contract omitted reservation or receipt identities")
    if set(expected_reservations) != set(query_ids) or set(expected_receipts) != set(
        query_ids
    ):
        raise ValueError("recovery contract query identities differ from the lock")

    for query_id in query_ids:
        reservation = _require_identity(
            reservations[query_id],
            str(expected_reservations[query_id]),
            label=f"reservation {query_id}",
        )
        if reservation.get("status") != "reserved" or reservation.get(
            "query_lock_payload_sha256"
        ) != payload_identity(lock):
            raise ValueError(f"reservation {query_id} does not bind the lock")
        expected_receipt = expected_receipts[query_id]
        observed_receipt = receipts.get(query_id)
        if expected_receipt is None:
            if observed_receipt is not None:
                raise ValueError(f"query {query_id} unexpectedly acquired a receipt")
        else:
            if observed_receipt is None:
                raise FileNotFoundError(f"query {query_id} lost its complete receipt")
            _require_identity(
                observed_receipt,
                str(expected_receipt),
                label=f"query receipt {query_id}",
            )

    if float(now) < float(lock["query_deadline"]):
        raise RuntimeError("immutable query deadline has not elapsed")
    settlement = settle_query_lock(
        lock, receipts, now=float(now), charged_call_ceiling=budget_ceiling
    )
    if settlement.get("action") != expected["settlement_action"]:
        raise ValueError("settlement action differs from the frozen recovery contract")
    next_checkpoint = apply_settled_observations(
        frozen_checkpoint,
        plan,
        settlement,
        cell_key=str(cell["cell_key"]),
        source_smiles=str(cell["source_smiles"]),
        delta=float(cell["delta"]),
        budget_ceiling=budget_ceiling,
        controller_config=controller_config,
    )
    if next_checkpoint["charged_count"] != expected["charged_after"] or next_checkpoint[
        "rounds_completed"
    ] != expected["round_after"]:
        raise ValueError("recovered checkpoint progress differs from the frozen contract")
    return {
        "settlement": settlement,
        "checkpoint": next_checkpoint,
    }


__all__ = ["recover_locked_round"]
