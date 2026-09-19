import json
from pathlib import Path

import pytest

from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
)
from compose_v4.experiments.t4_shared_controller_scored_contract import (
    AUTHORIZATION_SCHEMA_VERSION,
    SCHEMA_VERSION,
    publish_authorization_receipt,
    required_authorization_sentence,
)


def test_required_authorization_is_exactly_payload_bound():
    first = required_authorization_sentence("a" * 64)
    second = required_authorization_sentence("b" * 64)
    assert first != second
    assert "a" * 64 in first
    assert "49 charged docking calls per cell and 441 total" in first
    assert "zero retry, replacement, backfill" in first


def test_authorization_envelope_identity_round_trip(tmp_path: Path):
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "contract_payload_sha256": "a" * 64,
        "authorized_scored_calls": 441,
        "user_statement": required_authorization_sentence("a" * 64),
    }
    path = tmp_path / "authorization.json"
    envelope = {"payload": payload, "payload_sha256": payload_identity(payload)}
    path.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")))
    loaded = json.loads(path.read_text())
    assert payload_identity(loaded["payload"]) == loaded["payload_sha256"]


def test_authorization_template_rejects_nonexact_identity():
    exact = required_authorization_sentence("a" * 64)
    with pytest.raises(AssertionError):
        assert exact == required_authorization_sentence("a" * 63 + "b")


def test_authorization_publication_is_exact_and_publish_once(tmp_path: Path):
    contract_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED_PENDING_EXACT_USER_AUTHORIZATION",
    }
    contract_identity = payload_identity(contract_payload)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "payload": contract_payload,
                "payload_sha256": contract_identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    authorization_path = tmp_path / "authorization.json"
    with pytest.raises(ValueError, match="does not exactly match"):
        publish_authorization_receipt(
            contract_path=contract_path,
            authorization_path=authorization_path,
            user_statement="broad authorization without the sealed payload",
        )
    exact = required_authorization_sentence(contract_identity)
    published = publish_authorization_receipt(
        contract_path=contract_path,
        authorization_path=authorization_path,
        user_statement=exact,
    )
    assert published["authorization_path"] == str(authorization_path)
    envelope = json.loads(authorization_path.read_text())
    assert envelope["payload"] == {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "contract_payload_sha256": contract_identity,
        "authorized_scored_calls": 441,
        "user_statement": exact,
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_authorization_receipt(
            contract_path=contract_path,
            authorization_path=authorization_path,
            user_statement=exact,
        )
