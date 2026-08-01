"""Strict 20-source orchestration tests for the semantic chunk cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
    ACTIVE8_ADMISSION_STATUS,
    SOURCE_INVENTORY_STATUS,
    EditingV2SemanticActive8SourceInventory,
    SemanticActive8CountFlow,
    SemanticActive8Source,
    SemanticActive8SourceBinding,
)
from compose_v4.data.semantic_active8_chunk_cache import (
    semantic_active8_chunk_cache_builder_identity,
)
from compose_v4.data.semantic_active8_chunk_cache_mapreduce import (
    GLOBAL_COMPLETION_FILENAME,
    SemanticActive8ChunkCacheIncomplete,
    SemanticActive8ChunkCacheMapReduceError,
    completed_semantic_active8_chunk_cache_task_ids,
    execute_semantic_active8_chunk_cache_task,
    plan_semantic_active8_chunk_cache,
    reduce_semantic_active8_chunk_caches,
    write_semantic_active8_chunk_cache_plan,
)
from compose_v4.data.semantic_packed_trace_store import (
    COMPLETION_FILENAME,
    MANIFEST_FILENAME,
    SHARD_FILENAME,
    write_semantic_packed_artifact,
)
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import CycleCloseEdge
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard_v3 import encode_semantic_trace_record


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(*, trace_id: str, lane: str, role: str) -> dict[str, object]:
    source = pad_molecular_graph(smiles_to_molecular_graph("CCCCCC"), 16)
    step = RewriteStep("cycle_close", CycleCloseEdge(0, 5, 1))
    target = editing_v2_semantic_rewrite_system().apply(
        source,
        step.rule_name,
        step.action,
    )
    return encode_semantic_trace_record(
        RewriteTrace(source, target, (step,), {"fixture": trace_id}),
        trace_id=trace_id,
        data_lane=lane,
        split=role,
        source_address={"source_address_sha256": hashlib.sha256(trace_id.encode()).hexdigest()},
        lineage={"semantic_migration": "fixture"},
    )


def _source_revision() -> dict[str, object]:
    body = {
        "schema": "compose.data.semantic_active8_chunk_cache_modal_revision",
        "schema_version": 1,
        "commit": "a" * 40,
        "tree": "b" * 40,
        "worktree_clean": True,
        "serialized_sources": {"fixture.py": "c" * 64},
        "chunk_builder_identity": semantic_active8_chunk_cache_builder_identity(),
    }
    return {**body, "source_revision_sha256": _sha256_value(body)}


def _inventory(artifact_root: Path) -> EditingV2SemanticActive8SourceInventory:
    sources: list[SemanticActive8Source] = []
    family_total = 0
    for index, (lane, role) in enumerate(
        (lane, role) for lane in REQUIRED_DATA_LANES for role in REQUIRED_PARTITION_ROLES
    ):
        task_root = artifact_root / "sources" / lane / role
        semantic = task_root / "semantic"
        source_binding = SemanticActive8SourceBinding(
            source_shard_name="legacy.jsonl.gz",
            source_shard_sha256=hashlib.sha256(f"source-{index}".encode()).hexdigest(),
            source_manifest_sha256=hashlib.sha256(f"manifest-{index}".encode()).hexdigest(),
            source_overlay_sha256=None,
            source_unified_manifest_sha256="d" * 64,
            source_entry_count=1,
        )
        write_semantic_packed_artifact(
            semantic,
            [_record(trace_id=f"trace-{index}", lane=lane, role=role)],
            data_lane=lane,
            split=role,
            source_binding=source_binding.as_mapping(),
            decision_binding={
                "decision_ledger_sha256": hashlib.sha256(f"decision-{index}".encode()).hexdigest(),
                "source_count": 1,
                "admitted_count": 1,
                "rejected_count": 0,
            },
        )
        manifest = json.loads((semantic / MANIFEST_FILENAME).read_text())
        semantic_completion = json.loads((semantic / COMPLETION_FILENAME).read_text())
        task_receipt = task_root / "RECEIPT.json"
        task_receipt.parent.mkdir(parents=True, exist_ok=True)
        task_receipt.write_text('{"fixture":true}\n')
        sources.append(
            SemanticActive8Source(
                data_lane=lane,
                partition_role=role,
                task_identity_sha256=hashlib.sha256(f"task-{index}".encode()).hexdigest(),
                task_output_artifact_path=f"/artifacts/sources/{lane}/{role}",
                task_output_directory=task_root,
                task_receipt_path=task_receipt,
                task_receipt_file_sha256=_sha256_file(task_receipt),
                task_receipt_sha256="e" * 64,
                semantic_artifact_directory=semantic,
                semantic_shard_path=semantic / SHARD_FILENAME,
                semantic_shard_sha256=_sha256_file(semantic / SHARD_FILENAME),
                semantic_manifest_path=semantic / MANIFEST_FILENAME,
                semantic_manifest_file_sha256=_sha256_file(semantic / MANIFEST_FILENAME),
                semantic_manifest_sha256=manifest["manifest_sha256"],
                semantic_completion_path=semantic / COMPLETION_FILENAME,
                semantic_completion_file_sha256=_sha256_file(semantic / COMPLETION_FILENAME),
                semantic_completion_sha256=semantic_completion["completion_sha256"],
                semantic_record_stream_sha256=manifest["record_stream_sha256"],
                process_identity_sha256=manifest["process_identity_sha256"],
                action_codec_schema_version=manifest["action_codec_schema_version"],
                action_codec_implementation_hash=manifest["action_codec_implementation_hash"],
                source_binding=source_binding,
                family_histogram=tuple(manifest["family_histogram"].items()),
                counts=SemanticActive8CountFlow(
                    candidate_traces=1,
                    migration_source_traces=1,
                    migration_admitted_traces=1,
                    migration_rejected_traces=0,
                    semantic_entries=1,
                    semantic_states=2,
                    semantic_actions=1,
                ),
            )
        )
        family_total += 1
    return EditingV2SemanticActive8SourceInventory(
        status=SOURCE_INVENTORY_STATUS,
        training_authorized=False,
        active8_admission_status=ACTIVE8_ADMISSION_STATUS,
        migration_completion_path=artifact_root / "SEMANTIC_MIGRATION_COMPLETE.json",
        migration_completion_file_sha256="0" * 64,
        migration_completion_sha256="1" * 64,
        migration_plan_path=artifact_root / "PLAN.json",
        migration_plan_file_sha256="2" * 64,
        migration_plan_sha256="3" * 64,
        run_identity_sha256="4" * 64,
        source_revision_sha256="5" * 64,
        source_inventory_sha256="6" * 64,
        task_inventory_sha256="7" * 64,
        result_inventory_sha256="8" * 64,
        process_identity_sha256=sources[0].process_identity_sha256,
        builder_identity_sha256="9" * 64,
        candidate_materialization_manifest_sha256="a" * 64,
        candidate_provenance_source_stream_sha256="b" * 64,
        split_assignment_sha256="c" * 64,
        lane_registry_sha256="d" * 64,
        membership_receipt_file_sha256="e" * 64,
        membership_receipt_sha256="f" * 64,
        family_histogram=(("cycle_insert", family_total),),
        rejection_histogram=(),
        counts=SemanticActive8CountFlow(
            candidate_traces=20,
            migration_source_traces=20,
            migration_admitted_traces=20,
            migration_rejected_traces=0,
            semantic_entries=20,
            semantic_states=40,
            semantic_actions=20,
        ),
        sources=tuple(sources),
    )


@pytest.fixture()
def planned(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    inventory = _inventory(artifact_root)
    plan = plan_semantic_active8_chunk_cache(
        inventory,
        source_revision=_source_revision(),
        output_artifact_root="/artifacts/cache",
        target_rows_per_chunk=1,
    )
    write_semantic_active8_chunk_cache_plan(plan, artifact_root=artifact_root)
    return artifact_root, plan


def test_plan_freezes_exact_twenty_ordered_non_authorizing_tasks(planned) -> None:
    _, plan = planned
    assert len(plan["tasks"]) == 20
    assert tuple((task["data_lane"], task["partition_role"]) for task in plan["tasks"]) == tuple(
        (lane, role) for lane in REQUIRED_DATA_LANES for role in REQUIRED_PARTITION_ROLES
    )
    assert plan["training_authorized"] is False
    assert plan["active8_admission_authorized"] is False
    assert plan["gate_zero_authorized"] is False


def test_reducer_refuses_partial_sources_then_reloads_exact_twenty(planned) -> None:
    artifact_root, plan = planned
    first = execute_semantic_active8_chunk_cache_task(
        plan,
        plan["tasks"][0]["task_identity_sha256"],
        artifact_root=artifact_root,
    )
    assert (
        len(
            completed_semantic_active8_chunk_cache_task_ids(
                plan,
                artifact_root=artifact_root,
            )
        )
        == 1
    )
    with pytest.raises(SemanticActive8ChunkCacheIncomplete, match="1/20"):
        reduce_semantic_active8_chunk_caches(plan, artifact_root=artifact_root)

    receipts = [first]
    for task in plan["tasks"][1:]:
        receipts.append(
            execute_semantic_active8_chunk_cache_task(
                plan,
                task["task_identity_sha256"],
                artifact_root=artifact_root,
            )
        )
    assert len({receipt["task_identity_sha256"] for receipt in receipts}) == 20
    complete = reduce_semantic_active8_chunk_caches(
        plan,
        artifact_root=artifact_root,
    )
    assert complete["completion"]["source_task_count"] == 20
    assert complete["completion"]["counts"] == {
        "semantic_entries": 20,
        "semantic_states": 40,
        "semantic_actions": 20,
    }
    assert complete["completion"]["active8_admission_authorized"] is False
    global_object = artifact_root / "cache" / complete["global_completion_object_path"]
    assert _sha256_file(global_object) == complete["global_completion_file_sha256"]
    pointer = (
        artifact_root
        / "cache"
        / "global_runs"
        / plan["run_identity_sha256"]
        / GLOBAL_COMPLETION_FILENAME
    )
    assert pointer.is_file()

    restarted_receipt = execute_semantic_active8_chunk_cache_task(
        plan,
        plan["tasks"][0]["task_identity_sha256"],
        artifact_root=artifact_root,
    )
    restarted_complete = reduce_semantic_active8_chunk_caches(
        plan,
        artifact_root=artifact_root,
    )
    assert restarted_receipt == first
    assert restarted_complete == complete


def test_completed_task_scan_rejects_unexpected_receipt(planned) -> None:
    artifact_root, plan = planned
    receipt_root = (
        artifact_root / "cache" / "global_runs" / plan["run_identity_sha256"] / "source_receipts"
    )
    receipt_root.mkdir(parents=True, exist_ok=True)
    (receipt_root / "unexpected.json").write_text("{}\n")
    with pytest.raises(
        SemanticActive8ChunkCacheMapReduceError,
        match="unexpected objects",
    ):
        completed_semantic_active8_chunk_cache_task_ids(
            plan,
            artifact_root=artifact_root,
        )
