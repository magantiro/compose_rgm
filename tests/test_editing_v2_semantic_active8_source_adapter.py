"""Exact-source tests for the semantic Editing-V2 Active8 adapter."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.active8_trace_inventory import Active8SourceShard
from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.editing_v2_active8_source_adapter import (
    ResolvedActive8SourceBinding,
    ResolvedEditingV2Active8Sources,
)
from compose_v4.data.editing_v2_candidate_provenance_bridge import (
    SOURCE_STREAM_SCHEMA,
    SOURCE_STREAM_SCHEMA_VERSION,
)
from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
    ACTIVE8_ADMISSION_STATUS,
    SOURCE_INVENTORY_STATUS,
    EditingV2SemanticActive8SourceError,
    resolve_editing_v2_semantic_active8_sources,
)
from compose_v4.data.editing_v2_split_census import canonical_sha256
from compose_v4.data.packed_trace_store import (
    build_packed_entry,
    manifest_path_for,
    write_packed_shard,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    COMPLETION_FILENAME,
    PLAN_FILENAME,
    build_semantic_migration_source_revision,
    execute_semantic_trace_migration_task,
    plan_semantic_trace_migration,
    reduce_semantic_trace_migration,
    write_semantic_trace_migration_plan,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    RECEIPT_FILENAME,
    SEMANTIC_ARTIFACT_DIRNAME,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.operators import BondInsert
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard import encode_trace_record

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _artifact_path(path: Path, *, artifact_root: Path) -> str:
    return f"/artifacts/{path.relative_to(artifact_root).as_posix()}"


def _candidate_source_stream(row_count: int) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": SOURCE_STREAM_SCHEMA,
        "schema_version": SOURCE_STREAM_SCHEMA_VERSION,
        "nonempty_jsonl_rows": row_count,
        "candidate_materialization": {
            "manifest_file_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "rows_file_sha256": "3" * 64,
            "rows_semantic_sha256": "4" * 64,
            "address_stream_sha256": "5" * 64,
        },
        "provenance_registry": {
            "file_sha256": "6" * 64,
            "registry_sha256": "7" * 64,
        },
        "candidate_audit_ledger": {
            "file_sha256": "8" * 64,
            "semantic_sha256": "9" * 64,
            "rows_file_sha256": "0" * 64,
            "rows_sha256": "a" * 64,
        },
        "split_candidates": {
            "file_sha256": "b" * 64,
            "semantic_sha256": "c" * 64,
        },
    }
    return {**body, "source_stream_sha256": canonical_sha256(body)}


def _source_shard(
    artifact_root: Path,
    *,
    lane: str,
    role: str,
    index: int,
) -> Path:
    source_state = pad_molecular_graph(smiles_to_molecular_graph("CCCCCC"), 16)
    action = BondInsert(0, 5, 1)
    target_state = de_novo_rewrite_system().apply(source_state, "bond_insert", action)
    trace = RewriteTrace(
        source_state,
        target_state,
        (RewriteStep("bond_insert", action),),
        {"fixture_index": index},
    )
    envelope = encode_trace_record(
        trace,
        n_slots=16,
        seed=100 + index,
        trace_id=f"source-{index}",
        partition=role,
        layer=lane,
        extra=dict(trace.metadata),
    )
    path = artifact_root / "sources" / lane / role / "shard_0000.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_packed_shard(
        path,
        [build_packed_entry(envelope, TraceProgressCTMC(trace))],
        provenance={"fixture_index": index},
        deterministic_gzip=True,
    )
    return path


def _resolved_sources(artifact_root: Path) -> ResolvedEditingV2Active8Sources:
    bindings: list[ResolvedActive8SourceBinding] = []
    for index, (lane, role) in enumerate(
        (lane, role) for lane in REQUIRED_DATA_LANES for role in REQUIRED_PARTITION_ROLES
    ):
        source = _source_shard(
            artifact_root,
            lane=lane,
            role=role,
            index=index,
        )
        manifest = manifest_path_for(source)
        bindings.append(
            ResolvedActive8SourceBinding(
                source=Active8SourceShard(
                    manifest_layer=lane,
                    envelope_layer=lane,
                    partition=role,
                    relative_path=f"{lane}/{role}/{source.name}",
                    path=source,
                ),
                artifact_path=_artifact_path(source, artifact_root=artifact_root),
                packed_shard_file_sha256=_sha256(source),
                packed_manifest_artifact_path=_artifact_path(
                    manifest,
                    artifact_root=artifact_root,
                ),
                packed_manifest_file_sha256=_sha256(manifest),
                packed_provenance_overlay_artifact_path=None,
                packed_provenance_overlay_file_sha256=None,
                candidate_ids=(f"candidate-{index}",),
                original_address_sha256s=(hashlib.sha256(f"address-{index}".encode()).hexdigest(),),
            )
        )
    membership = artifact_root / "sources" / "RESOLVED_PACKED_MEMBERSHIP.json"
    membership.write_bytes(b'{"fixture":true}\n')
    source_stream = _candidate_source_stream(len(bindings))
    return ResolvedEditingV2Active8Sources(
        membership_receipt_path=membership,
        membership_receipt_file_sha256=_sha256(membership),
        membership_receipt_sha256="1" * 64,
        candidate_materialization_manifest_sha256="2" * 64,
        candidate_provenance_source_stream=source_stream,
        split_assignment_sha256="3" * 64,
        lane_registry_sha256="4" * 64,
        bindings=tuple(bindings),
        source_manifest={"fixture": True},
    )


@pytest.fixture(scope="module")
def completed_migration(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    artifact_root = tmp_path_factory.mktemp("semantic_active8") / "artifacts"
    artifact_root.mkdir()
    resolved = _resolved_sources(artifact_root)
    revision = build_semantic_migration_source_revision(
        commit="a" * 40,
        tree="b" * 40,
        repo_root=ROOT,
        worktree_clean=True,
    )
    plan = plan_semantic_trace_migration(
        resolved,
        source_revision=revision,
        repo_root=ROOT,
        output_artifact_prefix="/artifacts/semantic_fixture",
    )
    plan_path = write_semantic_trace_migration_plan(
        plan,
        artifact_root=artifact_root,
        repo_root=ROOT,
    )
    for task in plan["tasks"]:
        execute_semantic_trace_migration_task(
            plan,
            task["task_identity_sha256"],
            artifact_root=artifact_root,
            repo_root=ROOT,
        )
    reduce_semantic_trace_migration(
        plan,
        artifact_root=artifact_root,
        repo_root=ROOT,
    )
    return artifact_root, plan_path.parent / COMPLETION_FILENAME


def _copy_migration(
    completed_migration: tuple[Path, Path],
    tmp_path: Path,
) -> tuple[Path, Path]:
    source_root, source_completion = completed_migration
    artifact_root = tmp_path / "artifacts"
    shutil.copytree(source_root, artifact_root)
    relative_completion = source_completion.relative_to(source_root)
    return artifact_root, artifact_root / relative_completion


def _rewrite_completion(path: Path, mutator: object) -> None:
    completion = json.loads(path.read_text())
    assert callable(mutator)
    mutator(completion)
    completion["completion_sha256"] = canonical_sha256(
        {key: value for key, value in completion.items() if key != "completion_sha256"}
    )
    path.write_bytes(_canonical_bytes(completion))


def test_resolves_exact_twenty_cell_immutable_inventory(
    completed_migration: tuple[Path, Path],
) -> None:
    artifact_root, completion = completed_migration
    inventory = resolve_editing_v2_semantic_active8_sources(
        completion,
        artifact_root=artifact_root,
        repo_root=ROOT,
    )

    assert inventory.status == SOURCE_INVENTORY_STATUS
    assert inventory.training_authorized is False
    assert inventory.active8_admission_status == ACTIVE8_ADMISSION_STATUS
    assert len(inventory.sources) == 20
    assert tuple(
        (source.data_lane, source.partition_role) for source in inventory.sources
    ) == tuple((lane, role) for lane in REQUIRED_DATA_LANES for role in REQUIRED_PARTITION_ROLES)
    assert inventory.counts.candidate_traces == 20
    assert inventory.counts.migration_source_traces == 20
    assert inventory.counts.migration_admitted_traces == 20
    assert inventory.counts.migration_rejected_traces == 0
    assert inventory.counts.semantic_entries == 20
    assert all(source.entry_count == 1 for source in inventory.sources)
    assert all(
        source.source_binding.source_entry_count == source.counts.migration_source_traces
        for source in inventory.sources
    )
    assert all(len(source.semantic_record_stream_sha256) == 64 for source in inventory.sources)
    assert inventory.sources[0].source_binding.as_mapping()["source_entry_count"] == 1
    with pytest.raises(FrozenInstanceError):
        inventory.training_authorized = True  # type: ignore[misc]


def test_rejects_extra_and_missing_published_objects(
    completed_migration: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    artifact_root, completion = _copy_migration(completed_migration, tmp_path)
    (completion.parent / "unexpected.json").write_text("{}\n")
    with pytest.raises(
        EditingV2SemanticActive8SourceError,
        match="run root inventory disagrees",
    ):
        resolve_editing_v2_semantic_active8_sources(
            completion,
            artifact_root=artifact_root,
            repo_root=ROOT,
        )

    (completion.parent / "unexpected.json").unlink()
    plan = json.loads((completion.parent / PLAN_FILENAME).read_text())
    first_task = plan["tasks"][0]["task_identity_sha256"]
    (completion.parent / "tasks" / first_task / RECEIPT_FILENAME).unlink()
    with pytest.raises(
        EditingV2SemanticActive8SourceError,
        match="inventory disagrees",
    ):
        resolve_editing_v2_semantic_active8_sources(
            completion,
            artifact_root=artifact_root,
            repo_root=ROOT,
        )


def test_rejects_semantic_shard_tampering(
    completed_migration: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    artifact_root, completion = _copy_migration(completed_migration, tmp_path)
    plan = json.loads((completion.parent / PLAN_FILENAME).read_text())
    first_task = plan["tasks"][0]["task_identity_sha256"]
    shard = completion.parent / "tasks" / first_task / SEMANTIC_ARTIFACT_DIRNAME / "traces.jsonl.gz"
    shard.write_bytes(shard.read_bytes() + b"tamper")

    with pytest.raises(
        EditingV2SemanticActive8SourceError,
        match="task receipt is invalid",
    ):
        resolve_editing_v2_semantic_active8_sources(
            completion,
            artifact_root=artifact_root,
            repo_root=ROOT,
        )


def test_rejects_rehashed_aggregate_count_tampering(
    completed_migration: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    artifact_root, completion = _copy_migration(completed_migration, tmp_path)

    def mutate(value: dict[str, object]) -> None:
        counts = value["counts"]
        assert isinstance(counts, dict)
        counts["source"] = 21
        counts["admitted"] = 21

    _rewrite_completion(completion, mutate)
    with pytest.raises(
        EditingV2SemanticActive8SourceError,
        match="aggregate counts disagree",
    ):
        resolve_editing_v2_semantic_active8_sources(
            completion,
            artifact_root=artifact_root,
            repo_root=ROOT,
        )


def test_rejects_duplicate_result_even_when_completion_is_rehashed(
    completed_migration: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    artifact_root, completion = _copy_migration(completed_migration, tmp_path)

    def mutate(value: dict[str, object]) -> None:
        results = value["result_inventory"]
        assert isinstance(results, list)
        results[1] = dict(results[0])
        value["result_inventory_sha256"] = canonical_sha256(results)

    _rewrite_completion(completion, mutate)
    with pytest.raises(
        EditingV2SemanticActive8SourceError,
        match="does not match its planned task|duplicate",
    ):
        resolve_editing_v2_semantic_active8_sources(
            completion,
            artifact_root=artifact_root,
            repo_root=ROOT,
        )


def test_rejects_task_symlink_escape(
    completed_migration: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    artifact_root, completion = _copy_migration(completed_migration, tmp_path)
    plan = json.loads((completion.parent / PLAN_FILENAME).read_text())
    first_task = plan["tasks"][0]["task_identity_sha256"]
    task = completion.parent / "tasks" / first_task
    outside = tmp_path / "outside-task"
    shutil.copytree(task, outside)
    shutil.rmtree(task)
    os.symlink(outside, task)

    with pytest.raises(
        EditingV2SemanticActive8SourceError,
        match="resolves outside the artifact root",
    ):
        resolve_editing_v2_semantic_active8_sources(
            completion,
            artifact_root=artifact_root,
            repo_root=ROOT,
        )


def test_rejects_plan_byte_tampering(
    completed_migration: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    artifact_root, completion = _copy_migration(completed_migration, tmp_path)
    plan_path = completion.parent / PLAN_FILENAME
    plan_path.write_bytes(plan_path.read_bytes() + b" ")

    with pytest.raises(
        EditingV2SemanticActive8SourceError,
        match="plan bytes are not canonical",
    ):
        resolve_editing_v2_semantic_active8_sources(
            completion,
            artifact_root=artifact_root,
            repo_root=ROOT,
        )
