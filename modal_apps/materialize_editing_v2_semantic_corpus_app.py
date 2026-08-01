"""Materialize the exact Editing V2 semantic corpus on Modal.

This is a CPU-only, restart-safe corpus-derivative job.  It resolves the
already frozen candidate, split, lane-registry, and physical-membership
artifacts; plans exactly one task for each of the 20 lane/role packed shards;
and commits every task result before the strict reducer runs.

The command must be invoked from a clean committed worktree.  It creates no
training or experiment authority.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
MAX_MAP_CONTAINERS = 20

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    .add_local_dir(
        ROOT / "src",
        str(REMOTE_ROOT / "src"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
    .add_local_dir(
        ROOT / "configs",
        str(REMOTE_ROOT / "configs"),
        copy=True,
    )
    .add_local_file(
        ROOT / "modal_apps" / "materialize_editing_v2_semantic_corpus_app.py",
        str(REMOTE_ROOT / "modal_apps" / "materialize_editing_v2_semantic_corpus_app.py"),
        copy=True,
    )
)

app = modal.App("compose-v4-editing-v2-semantic-corpus")
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts",
    create_if_missing=False,
)


def _local_source_revision() -> dict[str, object]:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from compose_v4.data.semantic_trace_migration_mapreduce import (
        repository_semantic_migration_source_revision,
    )

    return repository_semantic_migration_source_revision(repo_root=ROOT)


def _imports() -> dict[str, object]:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.data.editing_v2_active8_source_adapter import (
        resolve_editing_v2_active8_sources,
    )
    from compose_v4.data.semantic_trace_migration_mapreduce import (
        completed_semantic_trace_migration_task_ids,
        execute_semantic_trace_migration_task,
        plan_semantic_trace_migration,
        reduce_semantic_trace_migration,
        write_semantic_trace_migration_plan,
    )

    return {
        "completed_semantic_trace_migration_task_ids": (
            completed_semantic_trace_migration_task_ids
        ),
        "execute_semantic_trace_migration_task": execute_semantic_trace_migration_task,
        "plan_semantic_trace_migration": plan_semantic_trace_migration,
        "reduce_semantic_trace_migration": reduce_semantic_trace_migration,
        "resolve_editing_v2_active8_sources": resolve_editing_v2_active8_sources,
        "write_semantic_trace_migration_plan": write_semantic_trace_migration_plan,
    }


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=8 * 3600,
    max_containers=MAX_MAP_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def migrate_one_shard(
    plan: dict[str, object],
    task_identity_sha256: str,
) -> dict[str, object]:
    """Execute or verify one exact whole-physical-shard task."""

    loaded = _imports()
    artifact_volume.reload()
    result = loaded["execute_semantic_trace_migration_task"](
        plan,
        task_identity_sha256,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    artifact_volume.commit()
    print(
        json.dumps(
            {
                "phase": "semantic_migration_map_complete",
                **result,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=2 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def reduce_complete(plan: dict[str, object]) -> dict[str, object]:
    """Reduce only the exact durable result set declared by the plan."""

    loaded = _imports()
    artifact_volume.reload()
    completion = loaded["reduce_semantic_trace_migration"](
        plan,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    artifact_volume.commit()
    print(
        json.dumps(
            {
                "phase": "semantic_migration_reduce_complete",
                "run_identity_sha256": completion["run_identity_sha256"],
                "completion_sha256": completion["completion_sha256"],
                "task_count": completion["task_count"],
                "counts": completion["counts"],
                "training_authorized": completion["training_authorized"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return completion


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=24 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def driver(
    candidate_materialization_dir: str,
    split_assignment_path: str,
    editing_corpus_contract_path: str,
    lane_registry_path: str,
    membership_receipt_path: str,
    expected_membership_receipt_sha256: str,
    output_artifact_prefix: str,
    source_revision: dict[str, object],
) -> dict[str, object]:
    """Resolve, plan, durably map missing shards, then strictly reduce."""

    loaded = _imports()
    artifact_volume.reload()
    resolved = loaded["resolve_editing_v2_active8_sources"](
        candidate_materialization_dir=candidate_materialization_dir,
        split_assignment_path=split_assignment_path,
        editing_corpus_contract_path=editing_corpus_contract_path,
        lane_registry_path=lane_registry_path,
        artifact_root=ARTIFACT_ROOT,
        membership_receipt_path=membership_receipt_path,
        expected_receipt_sha256=expected_membership_receipt_sha256,
    )
    plan = loaded["plan_semantic_trace_migration"](
        resolved,
        source_revision=source_revision,
        repo_root=REMOTE_ROOT,
        output_artifact_prefix=output_artifact_prefix,
    )
    plan_path = loaded["write_semantic_trace_migration_plan"](
        plan,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    artifact_volume.commit()
    artifact_volume.reload()
    complete = loaded["completed_semantic_trace_migration_task_ids"](
        plan,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    missing = [
        task["task_identity_sha256"]
        for task in plan["tasks"]
        if task["task_identity_sha256"] not in complete
    ]
    print(
        json.dumps(
            {
                "phase": "semantic_migration_plan_complete",
                "run_identity_sha256": plan["run_identity_sha256"],
                "plan_sha256": plan["plan_sha256"],
                "plan_path": str(plan_path),
                "expected_tasks": plan["expected_task_count"],
                "completed_tasks": len(complete),
                "missing_tasks": len(missing),
                "max_map_containers": MAX_MAP_CONTAINERS,
                "cpu_only": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if missing:
        results = list(
            migrate_one_shard.starmap(
                [(plan, task_identity_sha256) for task_identity_sha256 in missing]
            )
        )
        if len(results) != len(missing):
            raise RuntimeError("semantic migration map lost task results")
    # The return values above are diagnostics only.  The reducer reloads the
    # volume and independently verifies the exact planned task namespace.
    return reduce_complete.remote(plan)


@app.local_entrypoint()
def main(
    candidate_materialization_dir: str,
    split_assignment_path: str,
    lane_registry_path: str,
    membership_receipt_path: str,
    expected_membership_receipt_sha256: str,
    editing_corpus_contract_path: str = ("/root/compose/configs/editing_corpus_v2_contract.json"),
    output_artifact_prefix: str = "/artifacts/editing_v2/semantic_migration",
) -> None:
    source_revision = _local_source_revision()
    completion = driver.remote(
        candidate_materialization_dir,
        split_assignment_path,
        editing_corpus_contract_path,
        lane_registry_path,
        membership_receipt_path,
        expected_membership_receipt_sha256,
        output_artifact_prefix,
        source_revision,
    )
    print(
        json.dumps(
            {
                "phase": "semantic_migration_complete",
                "source_revision": source_revision,
                "completion": completion,
                "training_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
