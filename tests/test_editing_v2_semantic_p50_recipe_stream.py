from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.successor_fiber_cache import SuccessorFiberCacheAddress
from compose_v4.experiments.editing_v2_semantic_p50_recipe_stream import (
    SCHEDULED_EXAMPLES,
    SemanticP50Candidate,
    SemanticP50CandidateInventory,
    SemanticP50Prerequisites,
    SemanticP50RecipeStreamError,
    authorize_semantic_p50_recipe,
    compile_semantic_p50_prepared_recipe,
    load_semantic_p50_recipe_policy,
    semantic_p50_time_hex,
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
    assert (
        first["validation_contract"]["maximum_family_final_minus_baseline_for_p50_nonincrease_nats"]
        == 1e-7
    )
    assert (
        first["validation_contract"]["maximum_cell_final_minus_baseline_for_p50_nonincrease_nats"]
        == 1e-7
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


def test_time_derivation_rejects_invalid_stream_indices() -> None:
    address = _address(0)
    with pytest.raises(SemanticP50RecipeStreamError, match="nonnegative integer"):
        semantic_p50_time_hex(stream_index=-1, address=address)
    with pytest.raises(SemanticP50RecipeStreamError, match="nonnegative integer"):
        semantic_p50_time_hex(stream_index=True, address=address)


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


def test_authority_rejects_in_memory_recipe_and_generic_binding_paths(
    tmp_path: Path,
) -> None:
    prepared = compile_semantic_p50_prepared_recipe(_inventory())
    generic_paths = [
        tmp_path / "self-asserted-cache.json",
        tmp_path / "self-asserted-validation.json",
        tmp_path / "self-asserted-runtime.json",
        tmp_path / "self-asserted-environment.json",
        tmp_path / "self-asserted-launch.json",
    ]

    with pytest.raises(
        SemanticP50RecipeStreamError,
        match="generic binding paths are forbidden",
    ):
        authorize_semantic_p50_recipe(
            prepared=prepared,
            binding_paths=generic_paths,
        )
