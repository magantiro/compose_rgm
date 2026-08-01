from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.experiments.editing_gate_zero_semantic_contract import (
    EXPECTED_MARK_DIM,
    EXPECTED_OPERATOR_CAPABILITY_FINGERPRINT,
    LEGACY_OPERATOR_CAPABILITY_FINGERPRINT,
    GateZeroSemanticContractError,
    build_gate_zero_semantic_contract,
    gate_zero_semantic_contract_sha256,
    load_gate_zero_semantic_contract,
    validate_gate_zero_semantic_contract,
)
from compose_v4.experiments.editing_v2_scientific_identity import (
    semantic_model_identity,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_v2_process_identity,
)

CONTRACT_PATH = Path("configs/editing_gate_zero_semantic_model_process_v1.json")


def _rehash(value: dict[str, object]) -> dict[str, object]:
    value["contract_sha256"] = gate_zero_semantic_contract_sha256(value)
    return value


def test_frozen_contract_binds_exact_current_semantic_model_and_process() -> None:
    loaded = load_gate_zero_semantic_contract(CONTRACT_PATH)

    assert loaded.payload == build_gate_zero_semantic_contract()
    assert loaded.payload["training_authorized"] is False
    assert loaded.payload["active_families"] == list(ACTIVE8_FAMILIES)
    assert loaded.payload["model_identity"] == semantic_model_identity(
        mark_dim=EXPECTED_MARK_DIM,
        operator_capability_fingerprint=EXPECTED_OPERATOR_CAPABILITY_FINGERPRINT,
    )
    assert loaded.payload["model_identity"]["mark_dim"] == 32
    assert loaded.payload["model_identity"]["use_aromatic_bond_view"] is True
    assert loaded.payload["model_identity"]["operator_capability_fingerprint"] == "d246bc88d8440d31"
    assert (
        loaded.payload["process_identity_sha256"]
        == editing_v2_process_identity()["process_identity_sha256"]
    )
    assert len(loaded.file_sha256) == 64


def test_contract_rejects_missing_semantic_mode_after_valid_rehash() -> None:
    payload = build_gate_zero_semantic_contract()
    del payload["model_identity"]["cycle_open_action_semantics"]
    _rehash(payload)

    with pytest.raises(GateZeroSemanticContractError, match="missing or mixes"):
        validate_gate_zero_semantic_contract(payload)


def test_contract_rejects_legacy_capability_fingerprint_after_valid_rehash() -> None:
    payload = build_gate_zero_semantic_contract()
    payload["model_identity"]["operator_capability_fingerprint"] = (
        LEGACY_OPERATOR_CAPABILITY_FINGERPRINT
    )
    _rehash(payload)

    with pytest.raises(GateZeroSemanticContractError, match="legacy.*forbidden"):
        validate_gate_zero_semantic_contract(payload)


def test_contract_rejects_mixed_legacy_and_semantic_modes_after_valid_rehash() -> None:
    payload = build_gate_zero_semantic_contract()
    payload["model_identity"]["cycle_open_action_semantics"] = "legacy_raw_actions_v1"
    _rehash(payload)

    with pytest.raises(GateZeroSemanticContractError, match="missing or mixes"):
        validate_gate_zero_semantic_contract(payload)


def test_contract_rejects_missing_model_identity_after_valid_rehash() -> None:
    payload = build_gate_zero_semantic_contract()
    del payload["model_identity"]
    _rehash(payload)

    with pytest.raises(GateZeroSemanticContractError, match="fields disagree"):
        validate_gate_zero_semantic_contract(payload)


def test_contract_rejects_another_process_identity_after_valid_rehash() -> None:
    payload = build_gate_zero_semantic_contract()
    payload["process_identity_sha256"] = "0" * 64
    _rehash(payload)

    with pytest.raises(GateZeroSemanticContractError, match="another production process"):
        validate_gate_zero_semantic_contract(payload)


def test_contract_rejects_content_change_without_rehash() -> None:
    payload = deepcopy(build_gate_zero_semantic_contract())
    payload["contract_id"] = "changed"

    with pytest.raises(GateZeroSemanticContractError):
        validate_gate_zero_semantic_contract(payload)
