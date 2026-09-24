"""The authorized arms must reconstruct to the bytes the owner actually approved.

Authorizing an arm changes fields that live inside the payload, so the payload
hash necessarily moves off the value that was authorized. Without a checkable
link, an authorized contract is indistinguishable from one re-sealed onto a hash
nobody approved.

These tests rebuild the pre-authorization payload from the shipped contract and
require it to hash to the recorded `authorized_payload_sha256`, for every arm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.control.docking_value import identity

sys.path.insert(0, str(ROOT / "scripts"))
from authorize_t4_replicate_arms import (
    AUTHORIZED,
    PENDING_STATUS,
    RECEIPT,
    reconstruct_unauthorized,
)

ARMS = sorted(AUTHORIZED)


def _envelope(arm: str) -> dict:
    path = ROOT / f"configs/t4_unified_controller_{arm}_r23_v1.json"
    return json.loads(path.read_text())


def _authorized(arm: str) -> bool:
    return _envelope(arm)["payload"].get("scored_launch_authorized") is True


@pytest.mark.parametrize("arm", ARMS)
def test_self_hash_verifies(arm: str) -> None:
    envelope = _envelope(arm)
    assert identity(envelope["payload"]) == envelope["payload_sha256"], arm


@pytest.mark.parametrize("arm", ARMS)
def test_reconstructs_to_the_authorized_bytes(arm: str) -> None:
    """The load-bearing one: rebuild what the owner saw and check the hash."""

    payload = _envelope(arm)["payload"]
    if not _authorized(arm):
        pytest.skip(f"{arm} is not authorized yet")
    recorded = payload["authorization"]["authorized_payload_sha256"]
    assert recorded.startswith(AUTHORIZED[arm]), (
        f"{arm} records {recorded[:8]} but the owner authorized "
        f"{AUTHORIZED[arm]}"
    )
    assert identity(reconstruct_unauthorized(payload)) == recorded, (
        f"{arm} does not reconstruct to the payload that was authorized; the "
        "contract has changed in a way the authorization did not cover"
    )


@pytest.mark.parametrize("arm", ARMS)
def test_authorization_changes_exactly_three_fields(arm: str) -> None:
    """A fourth changed field would be scope the authorization never covered."""

    payload = _envelope(arm)["payload"]
    if not _authorized(arm):
        pytest.skip(f"{arm} is not authorized yet")
    rebuilt = reconstruct_unauthorized(payload)
    differing = {
        key
        for key in set(payload) | set(rebuilt)
        if json.dumps(payload.get(key), sort_keys=True)
        != json.dumps(rebuilt.get(key), sort_keys=True)
    }
    assert differing == {
        "authorization",
        "scored_launch_authorized",
        "modal_launch_authorized",
        "status",
    }, f"{arm}: authorization touched {sorted(differing)}"


@pytest.mark.parametrize("arm", ARMS)
def test_budget_is_unchanged_by_authorization(arm: str) -> None:
    """Authorizing must not move the ceiling it authorizes."""

    payload = _envelope(arm)["payload"]
    assert payload["charged_calls_per_cell"] == 250, arm
    assert payload["total_charged_call_ceiling"] == sum(
        250 for _ in payload["cells"]
    ), arm


def test_the_panel_ceiling_is_the_authorized_budget() -> None:
    total = sum(_envelope(arm)["payload"]["total_charged_call_ceiling"] for arm in ARMS)
    cells = sum(len(_envelope(arm)["payload"]["cells"]) for arm in ARMS)
    assert cells == 60, cells
    assert total == 15_000, total


def test_receipt_matches_the_contracts_when_present() -> None:
    path = ROOT / RECEIPT
    if not path.exists():
        pytest.skip("not authorized yet")
    receipt = json.loads(path.read_text())
    assert receipt["budget"]["total_charged_call_ceiling"] == 15_000
    assert receipt["budget"]["cells"] == 60
    # The receipt must not overstate how the authorization arrived.
    assert "RELAYED" in receipt["provenance"]["reached_this_agent"]
    for arm in ARMS:
        recorded = receipt["authorized_arms"][arm]["authorized_payload_sha256"]
        assert recorded.startswith(AUTHORIZED[arm]), arm
        payload = _envelope(arm)["payload"]
        if _authorized(arm):
            assert payload["authorization"]["authorized_payload_sha256"] == recorded


def test_the_reconstruction_can_actually_fail() -> None:
    """Negative control: a drifted contract must not reconstruct."""

    arm = ARMS[0]
    payload = json.loads(json.dumps(_envelope(arm)["payload"]))
    payload.setdefault("authorization", {"authorized_payload_sha256": "x" * 64})
    payload["batch"] = payload["batch"] + 1
    rebuilt = reconstruct_unauthorized(payload)
    assert rebuilt["status"] == PENDING_STATUS
    assert identity(rebuilt) != payload["authorization"]["authorized_payload_sha256"]
