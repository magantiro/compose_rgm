"""Task-planning, restart, and strict-reduction tests for semantic migration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path, PurePosixPath

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
from compose_v4.data.editing_v2_split_census import canonical_sha256
from compose_v4.data.packed_trace_store import (
    build_packed_entry,
    manifest_path_for,
    write_packed_shard,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    COMPLETION_FILENAME,
    EXPECTED_TASK_COUNT,
    SemanticTraceMigrationIncomplete,
    SemanticTraceMigrationMapReduceError,
    build_semantic_migration_source_revision,
    completed_semantic_trace_migration_task_ids,
    execute_semantic_trace_migration_task,
    plan_semantic_trace_migration,
    reduce_semantic_trace_migration,
    write_semantic_trace_migration_plan,
)
from compose_v4.data.semantic_trace_migration_materializer import RECEIPT_FILENAME
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.operators import BondInsert
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard import encode_trace_record

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path(path: Path, *, artifact_root: Path) -> str:
    return f"/artifacts/{path.relative_to(artifact_root).as_posix()}"


def _source(
    artifact_root: Path,
    *,
    lane: str,
    role: str,
    index: int,
) -> tuple[Path, int]:
    source_state = pad_molecular_graph(smiles_to_molecular_graph("CCCCCC"), 16)
    action = BondInsert(0, 5, 1)
    target = de_novo_rewrite_system().apply(source_state, "bond_insert", action)
    trace = RewriteTrace(
        source_state,
        target,
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
    source = artifact_root / "role_lane" / lane / role / "shard_0000.jsonl.gz"
    source.parent.mkdir(parents=True, exist_ok=True)
    write_packed_shard(
        source,
        [build_packed_entry(envelope, TraceProgressCTMC(trace))],
        provenance={"fixture_index": index},
        deterministic_gzip=True,
    )
    return source, 1


def _resolved(tmp_path: Path) -> tuple[Path, ResolvedEditingV2Active8Sources]:
    artifact_root = tmp_path / "artifacts"
    bindings = []
    index = 0
    for lane in REQUIRED_DATA_LANES:
        for role in REQUIRED_PARTITION_ROLES:
            source, entries = _source(
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
                    candidate_ids=tuple(f"candidate-{index}-{entry}" for entry in range(entries)),
                    original_address_sha256s=tuple(
                        hashlib.sha256(f"address-{index}-{entry}".encode()).hexdigest()
                        for entry in range(entries)
                    ),
                )
            )
            index += 1
    membership = artifact_root / "role_lane" / "RESOLVED_PACKED_MEMBERSHIP.json"
    membership.write_text('{"fixture":true}\n')
    source_stream_body = {
        "schema": SOURCE_STREAM_SCHEMA,
        "schema_version": SOURCE_STREAM_SCHEMA_VERSION,
        "nonempty_jsonl_rows": len(bindings),
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
    source_stream = {
        **source_stream_body,
        "source_stream_sha256": canonical_sha256(source_stream_body),
    }
    return artifact_root, ResolvedEditingV2Active8Sources(
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


def _revision() -> dict:
    return build_semantic_migration_source_revision(
        commit="a" * 40,
        tree="b" * 40,
        repo_root=ROOT,
        worktree_clean=True,
    )


def test_planner_requires_exact_ordered_five_by_four_grid(tmp_path: Path) -> None:
    _, resolved = _resolved(tmp_path)
    incomplete = replace(resolved, bindings=resolved.bindings[:-1])
    with pytest.raises(
        SemanticTraceMigrationMapReduceError,
        match="exactly the ordered five-lane by four-role",
    ):
        plan_semantic_trace_migration(
            incomplete,
            source_revision=_revision(),
            repo_root=ROOT,
        )


def test_planner_rejects_candidate_source_stream_materialization_mismatch(
    tmp_path: Path,
) -> None:
    _, resolved = _resolved(tmp_path)
    source_stream = json.loads(json.dumps(resolved.candidate_provenance_source_stream))
    source_stream["candidate_materialization"]["manifest_sha256"] = "d" * 64
    source_body = {
        key: value for key, value in source_stream.items() if key != "source_stream_sha256"
    }
    source_stream["source_stream_sha256"] = canonical_sha256(source_body)

    with pytest.raises(
        SemanticTraceMigrationMapReduceError,
        match="candidate materialization and source stream disagree",
    ):
        plan_semantic_trace_migration(
            replace(
                resolved,
                candidate_provenance_source_stream=source_stream,
            ),
            source_revision=_revision(),
            repo_root=ROOT,
        )


def test_restart_safe_map_and_strict_reducer(tmp_path: Path) -> None:
    artifact_root, resolved = _resolved(tmp_path)
    revision = _revision()
    plan = plan_semantic_trace_migration(
        resolved,
        source_revision=revision,
        repo_root=ROOT,
        output_artifact_prefix="/artifacts/semantic_fixture",
    )
    assert plan["expected_task_count"] == EXPECTED_TASK_COUNT == 20
    assert len({task["source_artifact_path"] for task in plan["tasks"]}) == 20
    assert len({task["output_artifact_path"] for task in plan["tasks"]}) == 20
    assert (
        plan["upstream"]["candidate_provenance_source_stream"]
        == resolved.candidate_provenance_source_stream
    )
    first_plan_path = write_semantic_trace_migration_plan(
        plan,
        artifact_root=artifact_root,
        repo_root=ROOT,
    )
    assert (
        write_semantic_trace_migration_plan(
            plan,
            artifact_root=artifact_root,
            repo_root=ROOT,
        )
        == first_plan_path
    )

    for task in plan["tasks"][:-1]:
        result = execute_semantic_trace_migration_task(
            plan,
            task["task_identity_sha256"],
            artifact_root=artifact_root,
            repo_root=ROOT,
        )
        assert result["reused"] is False
    task_root = (
        artifact_root / PurePosixPath(plan["run_artifact_root"]).relative_to("/artifacts") / "tasks"
    )
    private_staging = task_root / (
        f".{plan['tasks'][-1]['task_identity_sha256']}.interrupted.staging"
    )
    private_staging.mkdir()
    assert (
        len(
            completed_semantic_trace_migration_task_ids(
                plan,
                artifact_root=artifact_root,
                repo_root=ROOT,
            )
        )
        == 19
    )
    with pytest.raises(SemanticTraceMigrationIncomplete, match="missing 1"):
        reduce_semantic_trace_migration(
            plan,
            artifact_root=artifact_root,
            repo_root=ROOT,
        )

    last = plan["tasks"][-1]
    first = execute_semantic_trace_migration_task(
        plan,
        last["task_identity_sha256"],
        artifact_root=artifact_root,
        repo_root=ROOT,
    )
    reused = execute_semantic_trace_migration_task(
        plan,
        last["task_identity_sha256"],
        artifact_root=artifact_root,
        repo_root=ROOT,
    )
    assert first["reused"] is False
    assert reused["reused"] is True
    assert first["receipt_sha256"] == reused["receipt_sha256"]

    completion = reduce_semantic_trace_migration(
        plan,
        artifact_root=artifact_root,
        repo_root=ROOT,
    )
    assert completion["task_count"] == 20
    assert completion["counts"] == {"source": 20, "admitted": 20, "rejected": 0}
    assert completion["training_authorized"] is False
    assert completion["plan_file_sha256"] == _sha256(first_plan_path)
    completion_path = (
        artifact_root
        / PurePosixPath(plan["run_artifact_root"]).relative_to("/artifacts")
        / COMPLETION_FILENAME
    )
    assert json.loads(completion_path.read_text()) == completion
    assert (
        reduce_semantic_trace_migration(
            plan,
            artifact_root=artifact_root,
            repo_root=ROOT,
        )
        == completion
    )

    task_root = completion_path.parent / "tasks"
    unexpected = task_root / ("f" * 64)
    unexpected.mkdir()
    with pytest.raises(
        SemanticTraceMigrationMapReduceError,
        match="unexpected objects",
    ):
        completed_semantic_trace_migration_task_ids(
            plan,
            artifact_root=artifact_root,
            repo_root=ROOT,
        )
    unexpected.rmdir()

    receipt_path = task_root / plan["tasks"][0]["task_identity_sha256"] / RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text())
    receipt["data_lane"] = "tampered_lane"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    with pytest.raises(
        SemanticTraceMigrationMapReduceError,
        match="task result is invalid|mismatches task",
    ):
        completed_semantic_trace_migration_task_ids(
            plan,
            artifact_root=artifact_root,
            repo_root=ROOT,
        )


def test_plan_rejects_stale_serialized_source_revision(tmp_path: Path) -> None:
    _, resolved = _resolved(tmp_path)
    revision = _revision()
    changed = dict(revision)
    changed_files = dict(changed["implementation_files"])
    changed_files["src/compose_v4/data/semantic_trace_migration_materializer.py"] = "0" * 64
    changed["implementation_files"] = changed_files
    with pytest.raises(
        SemanticTraceMigrationMapReduceError,
        match="stale or malformed",
    ):
        plan_semantic_trace_migration(
            resolved,
            source_revision=changed,
            repo_root=ROOT,
        )
