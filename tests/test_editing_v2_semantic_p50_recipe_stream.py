from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.successor_fiber_cache import SuccessorFiberCacheAddress
from compose_v4.experiments.editing_v2_semantic_p50_recipe_stream import (
    BINDING_SCHEMA,
    BINDING_STATUS,
    BOUND_PAYLOAD_SCHEMA,
    REQUIRED_BINDING_PURPOSES,
    SCHEDULED_EXAMPLES,
    SemanticP50Candidate,
    SemanticP50CandidateInventory,
    SemanticP50Prerequisites,
    SemanticP50RecipeStreamError,
    authorize_semantic_p50_recipe,
    compile_semantic_p50_prepared_recipe,
    load_semantic_p50_recipe_policy,
)


def _bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return encoded + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _prerequisites() -> SemanticP50Prerequisites:
    values = {
        field: hashlib.sha256(field.encode()).hexdigest()
        for field in SemanticP50Prerequisites.__dataclass_fields__
        if field != "operator_capability_fingerprint"
    }
    values["operator_capability_fingerprint"] = "0123456789abcdef"
    return SemanticP50Prerequisites(**values)


def _address(family_index: int, *, progress_index: int = 0) -> SuccessorFiberCacheAddress:
    return SuccessorFiberCacheAddress(
        packed_shard_content_sha256=hashlib.sha256(f"shard-{family_index}".encode()).hexdigest(),
        packed_shard_name=f"train-{family_index}.jsonl",
        entry_index=family_index,
        layer="reversible_synthetic_walk",
        partition="train",
        trace_id=f"trace-{family_index}",
        trace_source_key=f"source-{family_index}",
        trace_target_key=f"target-{family_index}",
        progress_index=progress_index,
        path_length=2,
    )


def _inventory() -> SemanticP50CandidateInventory:
    candidates = tuple(
        SemanticP50Candidate(
            address=_address(index),
            family=family,
            semantic_cell_id=f"editing_v2_active8_v1:{family}:fixture_context",
            data_lane="reversible_synthetic_walk",
            assignment_sha256=hashlib.sha256(f"assignment-{family}".encode()).hexdigest(),
        )
        for index, family in enumerate(ACTIVE8_FAMILIES)
    )
    return SemanticP50CandidateInventory(
        prerequisites=_prerequisites(),
        candidates=candidates,
    )


def test_exact_stream_is_deterministic_balanced_and_closure_safe() -> None:
    first = compile_semantic_p50_prepared_recipe(_inventory())
    second = compile_semantic_p50_prepared_recipe(_inventory())
    inventory = _inventory()
    reversed_inventory = SemanticP50CandidateInventory(
        prerequisites=inventory.prerequisites,
        candidates=tuple(reversed(inventory.candidates)),
    )

    assert first == second
    assert first == compile_semantic_p50_prepared_recipe(reversed_inventory)
    assert len(first["ordered_stream_rows"]) == SCHEDULED_EXAMPLES
    assert all(0.0 < float.fromhex(row["time_hex"]) < 1.0 for row in first["ordered_stream_rows"])
    assert {row["scheduled_draw_count"] for row in first["planned_family_exposure"]} == {400}
    assert set(first["minimum_nonzero_gradient_updates_by_family"].values()) == {40}
    assert set(first["minimum_nonzero_gradient_updates_by_semantic_cell"].values()) == {40}
    closure = first["complete_trace_closure_inventory"]
    assert len(closure) == len(ACTIVE8_FAMILIES) * 3
    assert sum(row["terminal"] for row in closure) == len(ACTIVE8_FAMILIES)
    assert all(not row["schedulable"] for row in closure if row["closure_only"])
    assert all(not row["schedulable"] for row in closure if row["terminal"])
    assert first["cache_contract"]["closure_only_rows_schedulable"] is False
    assert (
        first["validation_contract"]["maximum_cell_final_minus_baseline_successor_nll_nats"] == 0.25
    )
    assert first["bounded_p50_authorized"] is False


def test_stream_fails_closed_when_an_active_family_has_no_opportunity() -> None:
    inventory = _inventory()
    missing = SemanticP50CandidateInventory(
        prerequisites=inventory.prerequisites,
        candidates=inventory.candidates[:-1],
    )
    with pytest.raises(
        SemanticP50RecipeStreamError,
        match="zero planned optimization exposure",
    ):
        compile_semantic_p50_prepared_recipe(missing)


def test_policy_rejects_a_self_hashed_null_threshold(tmp_path: Path) -> None:
    policy, _ = load_semantic_p50_recipe_policy()
    policy["thresholds"]["maximum_cell_final_minus_baseline_successor_nll_nats"] = None
    body = dict(policy)
    body.pop("policy_sha256")
    policy["policy_sha256"] = _sha(body)
    path = tmp_path / "tampered-policy.json"
    path.write_bytes(_bytes(policy, newline=True))
    with pytest.raises(SemanticP50RecipeStreamError, match="projection disagrees"):
        load_semantic_p50_recipe_policy(path)


def _binding_details(purpose: str, prepared: dict[str, object]) -> dict[str, object]:
    if purpose == "stream_union_successor_cache":
        return {
            "coverage_mode": "complete_trace_closure_of_planned_address_union",
            "closure_only_rows_schedulable": False,
            "terminal_rows_schedulable": False,
            "requested_unique_nonterminal_address_count": prepared["cache_contract"][
                "requested_unique_nonterminal_address_count"
            ],
            "closure_record_count": prepared["cache_contract"]["closure_record_count"],
            "cache_file_sha256": "2" * 64,
            "cache_content_sha256": "3" * 64,
            "coverage_receipt_sha256": "4" * 64,
        }
    if purpose == "validation_baseline":
        return {
            "partition_role": "validation",
            "baseline_successor_nll_by_family": {family: 1.0 for family in ACTIVE8_FAMILIES},
            "baseline_successor_nll_by_semantic_cell": {
                cell: 1.0 for cell in prepared["declared_nonempty_semantic_cells"]
            },
            "entry_count_by_semantic_cell": {
                cell: 1 for cell in prepared["declared_nonempty_semantic_cells"]
            },
            "evaluated_entry_count": len(prepared["declared_nonempty_semantic_cells"]),
            "scratch_initial_model_state_sha256": prepared["prerequisites"][
                "scratch_initial_model_state_sha256"
            ],
            "validation_candidate_inventory_sha256": "5" * 64,
            "validation_stream_sha256": "6" * 64,
            "maximum_family_nll_regression_nats": 0.25,
            "maximum_cell_nll_regression_nats": 0.25,
        }
    if purpose == "trainer_runtime":
        return {
            "scratch_initial_model_state_sha256": prepared["prerequisites"][
                "scratch_initial_model_state_sha256"
            ],
            "model_runtime_identity_sha256": prepared["prerequisites"][
                "model_runtime_identity_sha256"
            ],
            "objective_name": "balanced_semantic_cell_productive_identity",
            "implementation_sha256": "1" * 64,
        }
    if purpose == "execution_environment":
        return {
            "deterministic_algorithms_required": True,
            "mixed_precision": False,
            "dtype": "float32",
            "worker_count": 0,
            "ordered_indexed_loading": True,
            "seed": 31,
        }
    if purpose == "launch_projection":
        return {
            "optimizer_steps": 50,
            "batch_size": 64,
            "resume": False,
            "ordered_training_stream_sha256": prepared["ordered_training_stream_sha256"],
        }
    raise AssertionError(purpose)


def _write_binding(
    root: Path,
    *,
    purpose: str,
    prepared: dict[str, object],
    details: dict[str, object] | None = None,
) -> Path:
    payload_path = root / f"{purpose}.payload"
    selected_details = details if details is not None else _binding_details(purpose, prepared)
    payload_body = {
        "schema": BOUND_PAYLOAD_SCHEMA,
        "schema_version": 1,
        "status": BINDING_STATUS,
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "p500_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "purpose": purpose,
        "prepared_recipe_sha256": prepared["prepared_recipe_sha256"],
        "details": selected_details,
    }
    payload = {**payload_body, "payload_sha256": _sha(payload_body)}
    payload_path.write_bytes(_bytes(payload, newline=True))
    body = {
        "schema": BINDING_SCHEMA,
        "schema_version": 1,
        "status": BINDING_STATUS,
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "p500_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "purpose": purpose,
        "prepared_recipe_sha256": prepared["prepared_recipe_sha256"],
        "source_inventory_sha256": prepared["prerequisites"]["source_inventory_sha256"],
        "ordered_address_stream_sha256": prepared["ordered_address_stream_sha256"],
        "ordered_training_stream_sha256": prepared["ordered_training_stream_sha256"],
        "requested_address_union_sha256": prepared["requested_address_union_sha256"],
        "complete_trace_closure_inventory_sha256": prepared[
            "complete_trace_closure_inventory_sha256"
        ],
        "physical_payload_relative_path": payload_path.name,
        "physical_payload_file_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        "details": selected_details,
    }
    binding = {**body, "binding_sha256": _sha(body)}
    binding_path = root / f"{purpose}.binding.json"
    binding_path.write_bytes(_bytes(binding, newline=True))
    return binding_path


def test_authority_requires_every_exact_physical_binding(tmp_path: Path) -> None:
    prepared = compile_semantic_p50_prepared_recipe(_inventory())
    bindings = [
        _write_binding(tmp_path, purpose=purpose, prepared=prepared)
        for purpose in REQUIRED_BINDING_PURPOSES
    ]
    authorized = authorize_semantic_p50_recipe(prepared=prepared, binding_paths=bindings)
    assert authorized["bounded_p50_authorized"] is True
    assert authorized["training_authorized"] is True
    assert authorized["p500_authorized"] is False

    with pytest.raises(SemanticP50RecipeStreamError, match="exactly one binding"):
        authorize_semantic_p50_recipe(prepared=prepared, binding_paths=bindings[:-1])


def test_authority_rejects_null_baseline_and_identity_mismatch(tmp_path: Path) -> None:
    prepared = compile_semantic_p50_prepared_recipe(_inventory())
    bindings = []
    for purpose in REQUIRED_BINDING_PURPOSES:
        details = _binding_details(purpose, prepared)
        if purpose == "validation_baseline":
            cell = prepared["declared_nonempty_semantic_cells"][0]
            details["baseline_successor_nll_by_semantic_cell"][cell] = None
        bindings.append(
            _write_binding(tmp_path, purpose=purpose, prepared=prepared, details=details)
        )
    with pytest.raises(SemanticP50RecipeStreamError, match="baseline"):
        authorize_semantic_p50_recipe(prepared=prepared, binding_paths=bindings)

    mismatch = replace(
        _inventory().prerequisites,
        source_inventory_sha256="f" * 64,
    )
    mismatched_inventory = SemanticP50CandidateInventory(
        prerequisites=mismatch,
        candidates=_inventory().candidates,
    )
    mismatched = compile_semantic_p50_prepared_recipe(mismatched_inventory)
    with pytest.raises(SemanticP50RecipeStreamError, match="mismatched"):
        authorize_semantic_p50_recipe(prepared=mismatched, binding_paths=bindings)
