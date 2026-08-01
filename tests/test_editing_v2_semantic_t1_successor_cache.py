"""Scientific boundaries for the semantic-lineage unique-state T1 cache."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from compose_v4.data.immutable_artifact import ImmutableArtifactError
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheRecord,
    canonical_successor_fiber_records_for_shard,
    successor_fiber_cache_record_from_payload,
    successor_fiber_cache_record_payload,
)
from compose_v4.experiments import editing_v2_semantic_t1_successor_cache as cache
from compose_v4.experiments.editing_v2_semantic_t1_artifact_contracts import (
    CACHE_COMPLETION_SCHEMA,
    CACHE_COMPLETION_SCHEMA_VERSION,
    CACHE_COMPLETION_STATUS,
    CACHE_LEAF_SCHEMA,
    CACHE_LEAF_SCHEMA_VERSION,
    CACHE_LEAF_STATUS,
    NO_DOWNSTREAM_AUTHORITY,
)
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)

SHARD = "a" * 64
SOURCE = "b" * 64
TARGET = "c" * 64
PANEL_ENTRY = "d" * 64


def _records() -> tuple[SuccessorFiberCacheRecord, ...]:
    source_support = StateProductiveSupport(
        source_key="C",
        source_state_sha256=SOURCE,
        virtual_aliases=(),
    )
    teacher = TeacherSuccessorFiber(
        source_key="C",
        target_key="CC",
        target_state_sha256=TARGET,
        aliases=(
            TeacherSuccessorAlias(
                family_name="atom_insert",
                table_name="atom_insert_bond_order",
                coordinate=(0, 1, 0),
            ),
        ),
        state_support=source_support,
    )
    address = {
        "packed_shard_content_sha256": SHARD,
        "packed_shard_name": "semantic_traces.jsonl.gz",
        "entry_index": 7,
        "layer": "observed_local_analogue",
        "partition": "train",
        "trace_id": "trace-7",
        "trace_source_key": "C",
        "trace_target_key": "CC",
        "path_length": 1,
    }
    return (
        SuccessorFiberCacheRecord(
            address=SuccessorFiberCacheAddress(
                **address,
                progress_index=0,
            ),
            state_support=source_support,
            teacher_fiber=teacher,
        ),
        SuccessorFiberCacheRecord(
            address=SuccessorFiberCacheAddress(
                **address,
                progress_index=1,
            ),
            state_support=StateProductiveSupport(
                source_key="CC",
                source_state_sha256=TARGET,
                virtual_aliases=(),
            ),
            teacher_fiber=None,
        ),
    )


def _leaf() -> dict[str, object]:
    records = _records()
    record_payloads = [successor_fiber_cache_record_payload(item) for item in records]
    address = cache._record_address_payload(records[0].address)
    binding_body = {
        "panel_entry_sha256": PANEL_ENTRY,
        "cache_address": address,
        "cache_address_sha256": cache._sha(address),
        "source_state_sha256": SOURCE,
        "target_state_sha256": TARGET,
        "successor_canonical_key": "CC",
    }
    bindings = [{**binding_body, "binding_sha256": cache._sha(binding_body)}]
    cache_input_body = {
        "packed_shard_content_sha256": SHARD,
        "packed_shard_name": "semantic_traces.jsonl.gz",
        "packed_entry_index": 7,
        "trace_id": "trace-7",
        "data_lane": "observed_local_analogue",
        "partition_role": "train",
        "trace_source_key": "C",
        "trace_target_key": "CC",
        "path_length": 1,
        "trace_address_sha256": "e" * 64,
        "decision_sha256": "f" * 64,
        "selected_progress_indices": [0],
        "panel_entry_sha256s": [PANEL_ENTRY],
    }
    cache_inputs = [
        {
            **cache_input_body,
            "cache_input_sha256": cache._sha(cache_input_body),
        }
    ]
    body: dict[str, object] = {
        "schema": CACHE_LEAF_SCHEMA,
        "schema_version": CACHE_LEAF_SCHEMA_VERSION,
        "status": CACHE_LEAF_STATUS,
        **NO_DOWNSTREAM_AUTHORITY,
        "run_identity_sha256": "1" * 64,
        "build_identity_sha256": "2" * 64,
        "task_identity_sha256": "3" * 64,
        "panel_artifact_sha256": "4" * 64,
        "decision_source_inventory_sha256": "5" * 64,
        "semantic_source": {
            "semantic_shard_sha256": SHARD,
            "data_lane": "observed_local_analogue",
        },
        "compiler_runtime": {
            "device": "cpu",
            "dtype": "torch.float32",
            "torch_version": "fixture",
            "cuda_version": "fixture-cuda-build",
            "rdkit_version": "fixture",
            "implementation_sha256": "9" * 64,
            "source_revision_sha256": "7" * 64,
        },
        "support_time_hex": "0x1.0000000000000p-1",
        "cache_trace_inputs": cache_inputs,
        "cache_input_inventory_sha256": cache._sha(cache_inputs),
        "record_count": len(record_payloads),
        "record_inventory_sha256": cache._sha(record_payloads),
        "records": record_payloads,
        "panel_entry_binding_count": 1,
        "panel_entry_binding_inventory_sha256": cache._sha(bindings),
        "panel_entry_bindings": bindings,
        "model_scores_or_probabilities_stored": False,
        "hazard_coordinates_included": False,
    }
    return {**body, "leaf_sha256": cache._sha(body)}


def _rehash_leaf(leaf: dict[str, object]) -> None:
    leaf["record_inventory_sha256"] = cache._sha(leaf["records"])
    leaf["panel_entry_binding_inventory_sha256"] = cache._sha(
        leaf["panel_entry_bindings"]
    )
    body = dict(leaf)
    body.pop("leaf_sha256", None)
    leaf["leaf_sha256"] = cache._sha(body)


def test_repository_policy_is_self_hashed_cpu_only_and_nonauthorizing() -> None:
    policy = cache.load_semantic_t1_successor_cache_policy(
        repo_root=Path(__file__).resolve().parents[1]
    )
    assert policy["compile_device_role"] == "cpu_once"
    assert policy["compiler_device"] == "cpu"
    assert policy["model_scores_or_probabilities_stored"] is False
    assert policy["hazard_coordinates_included"] is False
    assert policy["repeated_state_panel_included"] is False
    assert policy["maximum_leaf_count"] == 5
    assert policy["maximum_selected_trace_count"] == 1024
    assert policy["maximum_concurrent_volume_writers"] == 5
    assert all(policy[name] is False for name in NO_DOWNSTREAM_AUTHORITY)


def test_panel_request_loader_requires_the_launcher_indented_encoding(
    tmp_path: Path,
) -> None:
    payload = {"b": 2, "a": 1}
    path = tmp_path / "request.json"
    path.write_bytes(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    assert (
        cache._load_pretty_object(
            path,
            field_name="request",
            maximum_bytes=1024,
        )[0]
        == payload
    )

    path.write_bytes(json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n")
    with pytest.raises(cache.SemanticT1SuccessorCacheError, match="indented"):
        cache._load_pretty_object(
            path,
            field_name="request",
            maximum_bytes=1024,
        )


def test_public_record_codec_is_deterministic_without_legacy_provenance() -> None:
    first, second = _records()
    payload = successor_fiber_cache_record_payload(first)
    assert successor_fiber_cache_record_from_payload(payload) == first
    assert canonical_successor_fiber_records_for_shard(
        (second, first),
        expected_packed_shard_content_sha256=SHARD,
    ) == (first, second)


def test_train_source_digest_collision_fails_instead_of_overwriting_lane() -> None:
    sources = (
        SimpleNamespace(
            partition_role="train",
            semantic_shard_sha256=SHARD,
            data_lane="observed_local_analogue",
        ),
        SimpleNamespace(
            partition_role="train",
            semantic_shard_sha256=SHARD,
            data_lane="reversible_synthetic_walk",
        ),
    )
    with pytest.raises(cache.SemanticT1SuccessorCacheError, match="reused"):
        cache._unique_train_sources_by_digest(sources)


def test_leaf_validates_complete_coordinates_and_panel_binding() -> None:
    leaf = _leaf()
    assert cache.validate_semantic_t1_successor_cache_leaf(leaf) == leaf

    tampered = dict(leaf)
    tampered["model_scores_or_probabilities_stored"] = True
    body = dict(tampered)
    body.pop("leaf_sha256")
    tampered["leaf_sha256"] = cache._sha(body)
    with pytest.raises(cache.SemanticT1SuccessorCacheError, match="schema"):
        cache.validate_semantic_t1_successor_cache_leaf(tampered)

    tampered = dict(leaf)
    tampered["legacy_cache_identity"] = "forbidden"
    body = dict(tampered)
    body.pop("leaf_sha256")
    tampered["leaf_sha256"] = cache._sha(body)
    with pytest.raises(cache.SemanticT1SuccessorCacheError, match="schema"):
        cache.validate_semantic_t1_successor_cache_leaf(tampered)


def test_completion_is_nonauthorizing_and_excludes_repeated_state_law() -> None:
    body: dict[str, object] = {
        "schema": CACHE_COMPLETION_SCHEMA,
        "schema_version": CACHE_COMPLETION_SCHEMA_VERSION,
        "status": CACHE_COMPLETION_STATUS,
        **NO_DOWNSTREAM_AUTHORITY,
        "run_identity_sha256": "1" * 64,
        "build_identity_sha256": "2" * 64,
        "source_revision_sha256": "3" * 64,
        "panel_completion_sha256": "4" * 64,
        "panel_artifact_sha256": "5" * 64,
        "decision_source_inventory_sha256": "6" * 64,
        "initial_model_state_sha256": "7" * 64,
        "plan_artifact_path": "/artifacts/cache/run/SEMANTIC_T1_SUCCESSOR_CACHE_PLAN.json",
        "plan_file_sha256": "8" * 64,
        "plan_sha256": "9" * 64,
        "manifest_artifact_path": "/artifacts/cache/run/SEMANTIC_T1_SUCCESSOR_CACHE_MANIFEST.json",
        "manifest_file_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "leaf_count": 5,
        "record_count": 300,
        "panel_entry_binding_count": 64,
        "unique_state_successor_cache_compiled": True,
        "repeated_state_empirical_law_compiled": False,
        "training_launched": False,
        "next_stage_authorized": None,
    }
    completion = {**body, "completion_sha256": cache._sha(body)}
    assert (
        cache.validate_semantic_t1_successor_cache_completion(completion) == completion
    )

    invalid = dict(completion)
    invalid["t1_authorized"] = True
    rehashed = dict(invalid)
    rehashed.pop("completion_sha256")
    invalid["completion_sha256"] = cache._sha(rehashed)
    with pytest.raises(cache.SemanticT1SuccessorCacheError, match="authority"):
        cache.validate_semantic_t1_successor_cache_completion(invalid)


def test_loader_requires_explicit_full_semantic_identities(tmp_path: Path) -> None:
    with pytest.raises(cache.SemanticT1SuccessorCacheError, match="lowercase SHA"):
        cache.open_semantic_t1_successor_cache(
            tmp_path / cache.CACHE_COMPLETION_FILENAME,
            artifact_root=tmp_path,
            repo_root=Path(__file__).resolve().parents[1],
            expected_panel_completion_sha256="missing",
            expected_panel_artifact_sha256="1" * 64,
            expected_decision_source_inventory_sha256="2" * 64,
            expected_initial_model_state_sha256="3" * 64,
        )


def test_reduce_complete_and_open_exact_unique_state_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    run_root = artifact_root / "cache" / "run"
    task_root = run_root / "tasks" / ("3" * 64)
    task_root.mkdir(parents=True)
    leaf = _leaf()
    task = {
        "task_identity_sha256": "3" * 64,
        "semantic_source": leaf["semantic_source"],
        "cache_trace_inputs": leaf["cache_trace_inputs"],
        "cache_input_inventory_sha256": leaf["cache_input_inventory_sha256"],
        "complete_progress_record_count": leaf["record_count"],
        "selected_panel_entry_count": leaf["panel_entry_binding_count"],
    }
    plan = {
        "run_identity_sha256": leaf["run_identity_sha256"],
        "build_identity_sha256": leaf["build_identity_sha256"],
        "source_revision": {"source_revision_sha256": "7" * 64},
        "policy": {"policy_sha256": "8" * 64},
        "implementation_sha256": "9" * 64,
        "plan_sha256": "a" * 64,
        "panel_binding": {
            "completion_sha256": "b" * 64,
            "panel_artifact_sha256": leaf["panel_artifact_sha256"],
            "decision_source_inventory_sha256": leaf[
                "decision_source_inventory_sha256"
            ],
            "panel_entry_sha256s": [PANEL_ENTRY],
            "panel_entry_inventory_sha256": cache._sha([PANEL_ENTRY]),
            "support_time_hex": leaf["support_time_hex"],
        },
        "model_runtime_identity": {
            "initial_model_state_sha256": "c" * 64,
        },
        "semantic_model_process_contract": {"contract_sha256": "d" * 64},
        "semantic_source_inventory": {"source_inventory_sha256": "e" * 64},
        "task_count": 1,
        "selected_trace_count": 1,
        "complete_progress_record_count": leaf["record_count"],
        "selected_panel_entry_count": leaf["panel_entry_binding_count"],
        "tasks": [task],
        "run_artifact_root": "/artifacts/cache/run",
    }
    plan_path = run_root / cache.CACHE_PLAN_FILENAME
    leaf_path = task_root / cache.CACHE_LEAF_FILENAME
    cache.write_semantic_t1_successor_cache_artifact(plan_path, plan)
    cache.write_semantic_t1_successor_cache_leaf(leaf_path, leaf)
    monkeypatch.setattr(
        cache,
        "validate_semantic_t1_successor_cache_plan",
        lambda value, **kwargs: dict(plan),
    )

    substituted_trace = copy.deepcopy(leaf)
    for record in substituted_trace["records"]:
        record["trace_id"] = "substituted-trace"
    for binding in substituted_trace["panel_entry_bindings"]:
        binding["cache_address"]["trace_id"] = "substituted-trace"
        binding["cache_address_sha256"] = cache._sha(binding["cache_address"])
        binding_body = dict(binding)
        binding_body.pop("binding_sha256")
        binding["binding_sha256"] = cache._sha(binding_body)
    _rehash_leaf(substituted_trace)
    assert (
        cache.validate_semantic_t1_successor_cache_leaf(substituted_trace)
        == substituted_trace
    )
    with pytest.raises(cache.SemanticT1SuccessorCacheError, match="address union"):
        cache.validate_semantic_t1_successor_cache_leaf_for_plan(
            substituted_trace,
            plan=plan,
        )

    substituted_binding = copy.deepcopy(leaf)
    binding = substituted_binding["panel_entry_bindings"][0]
    binding["panel_entry_sha256"] = "0" * 64
    binding_body = dict(binding)
    binding_body.pop("binding_sha256")
    binding["binding_sha256"] = cache._sha(binding_body)
    _rehash_leaf(substituted_binding)
    with pytest.raises(cache.SemanticT1SuccessorCacheError, match="address union"):
        cache.validate_semantic_t1_successor_cache_leaf_for_plan(
            substituted_binding,
            plan=plan,
        )

    duplicate_binding = copy.deepcopy(leaf)
    duplicate_binding["panel_entry_bindings"].append(
        copy.deepcopy(duplicate_binding["panel_entry_bindings"][0])
    )
    duplicate_binding["panel_entry_binding_count"] = 2
    _rehash_leaf(duplicate_binding)
    with pytest.raises(cache.SemanticT1SuccessorCacheError, match="binding"):
        cache.validate_semantic_t1_successor_cache_leaf(duplicate_binding)

    gpu_leaf = copy.deepcopy(leaf)
    gpu_leaf["compiler_runtime"]["device"] = "cuda"
    _rehash_leaf(gpu_leaf)
    with pytest.raises(cache.SemanticT1SuccessorCacheError, match="frozen task"):
        cache.validate_semantic_t1_successor_cache_leaf_for_plan(
            gpu_leaf,
            plan=plan,
        )

    incomplete_leaf = copy.deepcopy(leaf)
    incomplete_leaf["records"].pop()
    incomplete_leaf["record_count"] = len(incomplete_leaf["records"])
    _rehash_leaf(incomplete_leaf)
    with pytest.raises(cache.SemanticT1SuccessorCacheError, match="coordinate"):
        cache.validate_semantic_t1_successor_cache_leaf(incomplete_leaf)

    wrong_role_leaf = copy.deepcopy(leaf)
    for record in wrong_role_leaf["records"]:
        record["partition"] = "validation"
    for binding in wrong_role_leaf["panel_entry_bindings"]:
        binding["cache_address"]["partition"] = "validation"
        binding["cache_address_sha256"] = cache._sha(binding["cache_address"])
        binding_body = dict(binding)
        binding_body.pop("binding_sha256")
        binding["binding_sha256"] = cache._sha(binding_body)
    _rehash_leaf(wrong_role_leaf)
    with pytest.raises(cache.SemanticT1SuccessorCacheError, match="role or lane"):
        cache.validate_semantic_t1_successor_cache_leaf(wrong_role_leaf)

    manifest = cache.build_semantic_t1_successor_cache_manifest(
        plan,
        plan_path=plan_path,
        artifact_root=artifact_root,
        repo_root=Path(__file__).resolve().parents[1],
    )
    manifest_path = run_root / cache.CACHE_MANIFEST_FILENAME
    cache.write_semantic_t1_successor_cache_artifact(manifest_path, manifest)
    completion = cache.build_semantic_t1_successor_cache_completion(
        plan,
        manifest,
        plan_path=plan_path,
        manifest_path=manifest_path,
        artifact_root=artifact_root,
    )
    completion_path = run_root / cache.CACHE_COMPLETION_FILENAME
    cache.write_semantic_t1_successor_cache_artifact(completion_path, completion)

    loaded = cache.open_semantic_t1_successor_cache(
        completion_path,
        artifact_root=artifact_root,
        repo_root=Path(__file__).resolve().parents[1],
        expected_panel_completion_sha256=plan["panel_binding"]["completion_sha256"],
        expected_panel_artifact_sha256=plan["panel_binding"]["panel_artifact_sha256"],
        expected_decision_source_inventory_sha256=plan["panel_binding"][
            "decision_source_inventory_sha256"
        ],
        expected_initial_model_state_sha256=plan["model_runtime_identity"][
            "initial_model_state_sha256"
        ],
    )
    assert loaded.records == _records()
    assert loaded.record_for_panel_entry_sha256(PANEL_ENTRY) == _records()[0]
    assert loaded.completion["training_authorized"] is False
    assert loaded.completion["repeated_state_empirical_law_compiled"] is False
    assert cache.write_semantic_t1_successor_cache_leaf(leaf_path, leaf) is False
    with pytest.raises(ImmutableArtifactError, match="different bytes"):
        cache.write_semantic_t1_successor_cache_leaf(leaf_path, substituted_trace)
