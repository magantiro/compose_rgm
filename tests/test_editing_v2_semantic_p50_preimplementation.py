"""Semantic P50 refuses legacy lineage and unfrozen recipe values."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.data.editing_v2_semantic_active8_decision_source import (
    INDEX_ENCODING,
    INDEX_SCHEMA,
    INDEX_SCHEMA_VERSION,
    INDEX_STATUS,
)
from compose_v4.experiments.editing_v2_semantic_p50_preimplementation import (
    CONTRACT_RELATIVE_PATH,
    SemanticP50PreimplementationError,
    assert_semantic_p50_recipe_frozen,
    load_semantic_p50_preimplementation_contract,
    validate_semantic_p50_preimplementation_contract,
    validate_semantic_p50_source_inventory,
)
from compose_v4.experiments.editing_v2_semantic_p50_source_inventory import (
    SEMANTIC_P50_SOURCE_INVENTORY_FILENAME,
)

ROOT = Path(__file__).resolve().parents[1]


def _bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_bytes(value).rstrip(b"\n")).hexdigest()


def _source_identity() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": INDEX_SCHEMA,
        "schema_version": INDEX_SCHEMA_VERSION,
        "status": INDEX_STATUS,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "encoding": INDEX_ENCODING,
        "migration_completion_file_sha256": "1" * 64,
        "migration_completion_sha256": "2" * 64,
        "chunk_cache_plan_file_sha256": "3" * 64,
        "chunk_cache_plan_sha256": "4" * 64,
        "chunk_cache_global_pointer_file_sha256": "5" * 64,
        "chunk_cache_global_pointer_sha256": "6" * 64,
        "chunk_cache_global_completion_file_sha256": "7" * 64,
        "chunk_cache_global_completion_sha256": "8" * 64,
        "decision_plan_file_sha256": "9" * 64,
        "decision_plan_sha256": "a" * 64,
        "decision_run_identity_sha256": "b" * 64,
        "decision_completion_file_sha256": "c" * 64,
        "decision_completion_sha256": "d" * 64,
        "decision_result_inventory_sha256": "e" * 64,
        "process_identity_sha256": "f" * 64,
        "policy_sha256": "0" * 64,
        "model_runtime_identity_sha256": "1" * 64,
        "source_identity_sha256": "2" * 64,
        "accepted_trace_inventory_sha256": "3" * 64,
        "accepted_progress_inventory_sha256": "4" * 64,
        "excluded_trace_inventory_sha256": "5" * 64,
        "trace_decision_inventory_sha256": "6" * 64,
        "decision_lookup_inventory_sha256": "7" * 64,
        "exclusion_lookup_inventory_sha256": "8" * 64,
        "decision_source_implementation_sha256": "9" * 64,
        "counts": {"accepted_traces": 8},
        "action_family_histogram": {"atom_insert": 1},
        "active8_exclusion_reason_histogram": {},
        "migration_rejection_histogram": {},
    }
    return {**body, "inventory_sha256": _sha(body)}


def test_preimplementation_contract_is_self_hashed_and_nonauthorizing() -> None:
    contract = load_semantic_p50_preimplementation_contract(ROOT / CONTRACT_RELATIVE_PATH)
    assert contract["bounded_p50_authorized"] is False
    assert contract["proposed_recipe_values"]["initialization_regime"] == "scratch"
    assert contract["proposed_recipe_values"]["scheduled_nonterminal_examples"] == 3200
    assert (
        contract["proposed_recipe_values"]["cache_coverage_mode"]
        == "complete_trace_closure_of_planned_address_union"
    )
    assert contract["unfrozen_recipe_values"]["initialization_regime"] is None
    assert contract["unfrozen_stream_values"]["ordered_training_stream_sha256"] is None
    with pytest.raises(SemanticP50PreimplementationError, match="recipe is not frozen"):
        assert_semantic_p50_recipe_frozen(contract)


def test_preimplementation_cannot_be_edited_into_authority() -> None:
    contract = load_semantic_p50_preimplementation_contract(ROOT / CONTRACT_RELATIVE_PATH)
    for section, field, value in (
        (None, "bounded_p50_authorized", True),
        ("proposed_recipe_values", "initialization_regime", "compatible_warm_start"),
        ("unfrozen_recipe_values", "initialization_regime", "scratch"),
        (
            "unfrozen_stream_values",
            "ordered_training_stream_sha256",
            "a" * 64,
        ),
    ):
        broken = copy.deepcopy(contract)
        target = broken if section is None else broken[section]
        target[field] = value
        body = {key: item for key, item in broken.items() if key != "contract_sha256"}
        broken["contract_sha256"] = _sha(body)
        with pytest.raises(SemanticP50PreimplementationError):
            validate_semantic_p50_preimplementation_contract(broken)


def test_semantic_source_must_equal_exact_gate_zero_identity(tmp_path: Path) -> None:
    identity = _source_identity()
    path = tmp_path / SEMANTIC_P50_SOURCE_INVENTORY_FILENAME
    path.write_bytes(_bytes(identity))
    gate_zero = {
        "decision_source_identity": identity,
        "decision_source_inventory_sha256": identity["inventory_sha256"],
    }
    observed = validate_semantic_p50_source_inventory(
        identity,
        source_path=path,
        gate_zero_evidence=gate_zero,
        expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert observed["migration_completion_sha256"] == "2" * 64
    assert observed["decision_completion_sha256"] == "d" * 64

    other = copy.deepcopy(identity)
    other["decision_completion_sha256"] = "f" * 64
    body = {key: item for key, item in other.items() if key != "inventory_sha256"}
    other["inventory_sha256"] = _sha(body)
    path.write_bytes(_bytes(other))
    with pytest.raises(SemanticP50PreimplementationError, match="differs from Gate0"):
        validate_semantic_p50_source_inventory(
            other,
            source_path=path,
            gate_zero_evidence=gate_zero,
            expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_legacy_active8_inventory_schema_is_rejected(tmp_path: Path) -> None:
    legacy = {
        "schema": "compose.active8_trace_inventory",
        "inventory_sha256": "a" * 64,
    }
    path = tmp_path / SEMANTIC_P50_SOURCE_INVENTORY_FILENAME
    path.write_bytes(_bytes(legacy))
    with pytest.raises(SemanticP50PreimplementationError, match="differs from Gate0"):
        validate_semantic_p50_source_inventory(
            legacy,
            source_path=path,
            gate_zero_evidence={
                "decision_source_identity": _source_identity(),
                "decision_source_inventory_sha256": "b" * 64,
            },
            expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
