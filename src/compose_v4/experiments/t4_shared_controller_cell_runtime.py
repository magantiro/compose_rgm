"""Pure receipt and continuation state machines for shared T4 cell runners.

The functions here perform no oracle, network, Modal, or volume operation.  A
future authorized app may use them around atomic durable publication.  Query
locks are irreversible: every locked row consumes one unit of budget exactly
once, including unresolved rows after their frozen deadline.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.experiments.t4_campaign_driver import campaign_driver_action
from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
)

QUERY_LOCK_SCHEMA = "t4_shared_controller_query_lock_v1"
DRIVER_STATE_SCHEMA = "t4_shared_controller_driver_state_v1"
QUERY_RECEIPT_SCHEMA = "t4_shared_controller_query_receipt_v1"


def _safe_relative_path(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe receipt path: {relative!r}")
    return path


class ReceiptStore:
    """Atomic self-hashed JSON publication with an injected durability flush."""

    def __init__(self, root: Path, flush: Callable[[], None] | None = None):
        self.root = root.resolve()
        self.flush = flush or (lambda: None)

    def _path(self, relative: str) -> Path:
        path = _safe_relative_path(relative)
        return self.root.joinpath(*path.parts)

    def read(self, relative: str) -> dict[str, Any]:
        path = self._path(relative)
        envelope = json.loads(path.read_text())
        payload = envelope.get("payload")
        if not isinstance(payload, dict) or payload_identity(payload) != envelope.get(
            "payload_sha256"
        ):
            raise ValueError(f"durable payload hash mismatch: {relative}")
        return payload

    def publish_once(self, relative: str, payload: dict[str, Any]) -> str:
        """Publish an immutable lock or receipt, refusing every overwrite."""

        path = self._path(relative)
        return self._publish(path, payload, replace=False)

    def replace(self, relative: str, payload: dict[str, Any]) -> str:
        """Atomically replace mutable checkpoint or driver state."""

        return self._publish(self._path(relative), payload, replace=True)

    def _publish(self, path: Path, payload: dict[str, Any], *, replace: bool) -> str:
        payload_sha256 = payload_identity(payload)
        encoded = (
            json.dumps(
                {"payload": payload, "payload_sha256": payload_sha256},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            if replace:
                os.replace(temporary_name, path)
                temporary_name = None
            else:
                try:
                    os.link(temporary_name, path)
                except FileExistsError as error:
                    raise FileExistsError(
                        f"durable receipt already exists: {path.relative_to(self.root)}"
                    ) from error
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        self.flush()
        return payload_sha256


def make_query_lock(
    *,
    cell_key: str,
    round_index: int,
    charged_before: int,
    selected_rows: Sequence[Mapping[str, Any]],
    query_deadline: float,
    charged_call_ceiling: int = 49,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Create one immutable, bounded query lock from an already selected batch."""

    if not cell_key or round_index < 0 or charged_before < 0:
        raise ValueError("invalid query lock cell, round, or charged count")
    if charged_call_ceiling != 49 or batch_size != 8:
        raise ValueError("completion campaign budget or batch drift")
    if not selected_rows or len(selected_rows) > batch_size:
        raise ValueError("query lock must contain between one and eight rows")
    if charged_before + len(selected_rows) > charged_call_ceiling:
        raise ValueError("query lock exceeds the 49-call cell ceiling")
    queries = []
    for index, row in enumerate(selected_rows):
        smiles = row.get("smiles")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("selected query has no molecule")
        query_id = f"{cell_key}_r{round_index:03d}_q{index:02d}"
        queries.append(
            {
                "query_id": query_id,
                "smiles": smiles,
                "selection_kind": row.get("selection_kind"),
                "proposal_experts": list(row.get("proposal_experts", ())),
                "parent": row.get("parent"),
                "parent_score": row.get("parent_score"),
            }
        )
    if len({row["query_id"] for row in queries}) != len(queries):
        raise AssertionError("query identifiers are not unique")
    if len({row["smiles"] for row in queries}) != len(queries):
        raise ValueError("one immutable query lock may not contain duplicate molecules")
    return {
        "schema_version": QUERY_LOCK_SCHEMA,
        "cell_key": cell_key,
        "round": round_index,
        "charged_before": charged_before,
        "query_deadline": float(query_deadline),
        "queries": queries,
    }


def make_query_reservation(lock: Mapping[str, Any], query_id: str) -> dict[str, Any]:
    """Create the receipt that must be durable before an evaluator is called."""

    validate_query_lock(lock)
    query = next((row for row in lock["queries"] if row["query_id"] == query_id), None)
    if query is None:
        raise ValueError(f"query is not present in its immutable lock: {query_id}")
    return {
        "schema_version": QUERY_RECEIPT_SCHEMA,
        "status": "reserved",
        "cell_key": lock["cell_key"],
        "round": lock["round"],
        "query_id": query_id,
        "smiles": query["smiles"],
        "query_lock_payload_sha256": payload_identity(dict(lock)),
    }


def execute_reserved_query_once(
    store: ReceiptStore,
    *,
    lock_path: str,
    reservation_path: str,
    receipt_path: str,
    query_id: str,
    evaluate: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Durably reserve before one injected evaluation and never retry ambiguity.

    A crash or evaluator exception after reservation deliberately leaves no
    completion receipt.  Calling this function again then stops at the immutable
    reservation instead of invoking the evaluator a second time.
    """

    lock = store.read(lock_path)
    reservation = make_query_reservation(lock, query_id)
    store.publish_once(reservation_path, reservation)
    answer = evaluate(reservation["smiles"])
    if not isinstance(answer, Mapping):
        raise TypeError("query evaluator answer must be a mapping")
    complete = {
        **reservation,
        "status": "complete",
        "answer": dict(answer),
    }
    store.publish_once(receipt_path, complete)
    return complete


def validate_query_lock(
    lock: Mapping[str, Any], *, charged_call_ceiling: int = 49
) -> None:
    if lock.get("schema_version") != QUERY_LOCK_SCHEMA:
        raise ValueError("query lock schema drift")
    queries = lock.get("queries")
    if not isinstance(queries, list) or not 1 <= len(queries) <= 8:
        raise ValueError("query lock row count is outside one to eight")
    charged_before = lock.get("charged_before")
    if not isinstance(charged_before, int) or charged_before < 0:
        raise ValueError("query lock charged count is invalid")
    if charged_before + len(queries) > charged_call_ceiling:
        raise ValueError("query lock exceeds its charged-call ceiling")
    ids = [row.get("query_id") for row in queries]
    smiles = [row.get("smiles") for row in queries]
    if any(not isinstance(value, str) or not value for value in ids + smiles):
        raise ValueError("query lock contains an invalid identity")
    if len(set(ids)) != len(ids) or len(set(smiles)) != len(smiles):
        raise ValueError("query lock contains duplicate identities")
    if not isinstance(lock.get("query_deadline"), (float, int)):
        raise TypeError("query lock has no frozen deadline")


def settle_query_lock(
    lock: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    *,
    now: float,
    charged_call_ceiling: int = 49,
) -> dict[str, Any]:
    """Recover complete receipts or charge unresolved locks without resubmission."""

    validate_query_lock(lock, charged_call_ceiling=charged_call_ceiling)
    query_ids = [row["query_id"] for row in lock["queries"]]
    unknown = sorted(set(receipts) - set(query_ids))
    if unknown:
        raise ValueError(f"receipts do not belong to the immutable lock: {unknown}")
    statuses = [
        str(receipts.get(query_id, {}).get("status", "missing"))
        for query_id in query_ids
    ]
    if statuses and all(status == "complete" for status in statuses):
        action = "recover"
    elif float(now) < float(lock["query_deadline"]):
        action = "wait"
    else:
        action = "recover_with_unresolved"
    if action == "wait":
        return {
            "action": "wait",
            "charged_after": int(lock["charged_before"]),
            "observations": [],
            "receipt_statuses": statuses,
        }
    observations = []
    for query in lock["queries"]:
        receipt = receipts.get(query["query_id"], {})
        if receipt.get("status") == "complete":
            if (
                receipt.get("schema_version") != QUERY_RECEIPT_SCHEMA
                or receipt.get("query_id") != query["query_id"]
                or receipt.get("smiles") != query["smiles"]
                or not isinstance(receipt.get("answer"), dict)
            ):
                raise ValueError("complete query receipt does not match its lock")
            observations.append({**query, **receipt["answer"]})
        else:
            observations.append(
                {
                    **query,
                    "score": None,
                    "failure": f"unresolved_locked_query_{receipt.get('status', 'missing')}",
                }
            )
    charged_after = int(lock["charged_before"]) + len(lock["queries"])
    if charged_after > charged_call_ceiling:
        raise AssertionError("settled query lock exceeded the frozen ceiling")
    return {
        "action": action,
        "charged_after": charged_after,
        "observations": observations,
        "receipt_statuses": statuses,
    }


def reserve_driver_generation(
    *,
    phase_status: str,
    existing_state: Mapping[str, Any] | None,
    confirmed_prior_call_terminal: bool = False,
) -> dict[str, Any]:
    """Reserve one continuation generation without spawning it."""

    if existing_state is None:
        generation = 0
        continuation_state = "none"
    else:
        if existing_state.get("schema_version") != DRIVER_STATE_SCHEMA:
            raise ValueError("driver state schema drift")
        continuation_state = str(existing_state.get("state"))
        generation = int(existing_state.get("generation", -1)) + 1
        if continuation_state in {"reserved", "running"}:
            return {"action": "wait", "state": dict(existing_state)}
        if continuation_state != "terminal" or not confirmed_prior_call_terminal:
            raise RuntimeError(
                "a prior driver generation requires authoritative terminal confirmation"
            )
    action = campaign_driver_action(
        [phase_status], continuation_state=continuation_state  # type: ignore[arg-type]
    )
    if action == "finish":
        return {"action": "finish", "state": existing_state}
    if action != "reserve_and_spawn":
        return {"action": "wait", "state": existing_state}
    return {
        "action": "spawn",
        "state": {
            "schema_version": DRIVER_STATE_SCHEMA,
            "state": "reserved",
            "generation": generation,
            "function_call_id": None,
        },
    }


def mark_driver_running(
    reserved_state: Mapping[str, Any], function_call_id: str
) -> dict[str, Any]:
    if (
        reserved_state.get("schema_version") != DRIVER_STATE_SCHEMA
        or reserved_state.get("state") != "reserved"
        or not function_call_id
    ):
        raise ValueError("only a reserved generation can become running")
    return {**reserved_state, "state": "running", "function_call_id": function_call_id}


def mark_driver_terminal(state: Mapping[str, Any]) -> dict[str, Any]:
    if state.get("schema_version") != DRIVER_STATE_SCHEMA or state.get("state") not in {
        "reserved",
        "running",
    }:
        raise ValueError("only a reserved or running generation can become terminal")
    return {**state, "state": "terminal"}


__all__ = [
    "DRIVER_STATE_SCHEMA",
    "QUERY_LOCK_SCHEMA",
    "QUERY_RECEIPT_SCHEMA",
    "ReceiptStore",
    "execute_reserved_query_once",
    "make_query_lock",
    "make_query_reservation",
    "mark_driver_running",
    "mark_driver_terminal",
    "reserve_driver_generation",
    "settle_query_lock",
    "validate_query_lock",
]
