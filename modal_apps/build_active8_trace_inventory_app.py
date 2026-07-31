"""Bounded, resumable Modal map/reduce for the active-8 trace inventory.

Each map invocation owns one immutable half-open entry range from a
unified-manifest-declared packed shard. Successful range decisions are
immutable content objects with immutable task receipts. Retries reuse verified
receipts. The reducer is invoked only after the bounded map completes, refuses
gaps, overlaps, or extra receipts, and reconstructs one logical decision stream
per original physical shard.

This app builds data evidence only.  It has no training function.

    modal run --detach modal_apps/build_active8_trace_inventory_app.py \
      --support-contract /root/compose/configs/editing_gate_zero_runtime_v2.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = Path("/root/compose")
_MAX_MAP_CONTAINERS_LIMIT = 64
_MAX_MAP_CONTAINERS = int(os.environ.get("COMPOSE_ACTIVE8_MAX_MAP_CONTAINERS", "64"))
if not 1 <= _MAX_MAP_CONTAINERS <= _MAX_MAP_CONTAINERS_LIMIT:
    raise ValueError("COMPOSE_ACTIVE8_MAX_MAP_CONTAINERS must lie in [1, 64]")
_MAP_CPU = 2.0
_MAP_MEMORY_MB = 8192
_MAP_OMP_NUM_THREADS = 1
_DEFAULT_TARGET_ENTRIES_PER_RANGE = 500
_LEGACY_PARTITIONS = ("train", "validation", "test")

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
            "PYTHONPATH": os.pathsep.join((str(REMOTE_ROOT / "src"), str(REMOTE_ROOT / "scripts"))),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(_MAP_OMP_NUM_THREADS),
        }
    )
    .add_local_dir(
        ROOT / "src",
        str(REMOTE_ROOT / "src"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
    .add_local_dir(
        ROOT / "scripts",
        str(REMOTE_ROOT / "scripts"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
    .add_local_dir(
        ROOT / "configs",
        str(REMOTE_ROOT / "configs"),
        copy=True,
    )
    .add_local_file(
        ROOT / "modal_apps" / "build_active8_trace_inventory_app.py",
        str(REMOTE_ROOT / "modal_apps" / "build_active8_trace_inventory_app.py"),
        copy=True,
    )
)

app = modal.App("compose-v4-active8-trace-inventory")
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts",
    create_if_missing=False,
)


def _local_source_revision() -> dict[str, object]:
    """Bind the serialized Modal image to one clean local Git snapshot."""

    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from compose_v4.data.active8_inventory_mapreduce import (
        repository_source_revision,
    )

    return repository_source_revision(repo_root=ROOT)


def _imports() -> dict[str, object]:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.data.active8_inventory_mapreduce import (
        map_active8_source_decisions,
        plan_active8_mapreduce,
        reduce_active8_mapreduce,
        verified_completed_task_identities,
    )
    from compose_v4.data.active8_trace_inventory import (
        Active8SourceShard,
        ProductionExactCandidateChecker,
    )
    from compose_v4.data.packed_charge_policy_audit import (
        resolve_unified_manifest_shards,
    )
    from compose_v4.experiments.editing_gate_zero_runtime import (
        build_scratch_ringcore_model,
        load_gate_zero_runtime_contract,
    )

    return {
        "Active8SourceShard": Active8SourceShard,
        "ProductionExactCandidateChecker": ProductionExactCandidateChecker,
        "map_active8_source_decisions": map_active8_source_decisions,
        "plan_active8_mapreduce": plan_active8_mapreduce,
        "reduce_active8_mapreduce": reduce_active8_mapreduce,
        "verified_completed_task_identities": (verified_completed_task_identities),
        "resolve_unified_manifest_shards": resolve_unified_manifest_shards,
        "build_scratch_ringcore_model": build_scratch_ringcore_model,
        "load_gate_zero_runtime_contract": load_gate_zero_runtime_contract,
    }


@app.function(
    image=image,
    cpu=_MAP_CPU,
    memory=_MAP_MEMORY_MB,
    timeout=4 * 3600,
    max_containers=_MAX_MAP_CONTAINERS,
    volumes={"/artifacts": artifact_volume},
)
def map_source(
    task: dict[str, object],
    support_contract_path: str,
    output_root: str,
    candidate_cache_size: int,
) -> dict[str, object]:
    """Build or verify one deterministic source-entry-range decision object."""

    loaded = _imports()
    artifact_volume.reload()
    contract = loaded["load_gate_zero_runtime_contract"](Path(support_contract_path))
    model, _ = loaded["build_scratch_ringcore_model"](contract)
    checker = loaded["ProductionExactCandidateChecker"](
        model,
        cache_size=candidate_cache_size,
    )
    result = loaded["map_active8_source_decisions"](
        task,
        exact_candidate_checker=checker,
        output_root=Path(output_root),
    )
    artifact_volume.commit()
    print(
        json.dumps(
            {
                "phase": "active8_map_source",
                "task_identity_sha256": result["task_identity_sha256"],
                "range_binding": result["range_binding"],
                "counts": result["counts"],
                "reused": result["reused"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=4 * 3600,
    volumes={"/artifacts": artifact_volume},
)
def reduce_inventory(
    plan: dict[str, object],
    output_root: str,
) -> dict[str, object]:
    """Reduce only an exact complete set of immutable map receipts."""

    loaded = _imports()
    artifact_volume.reload()
    result = loaded["reduce_active8_mapreduce"](
        plan,
        output_root=Path(output_root),
    )
    artifact_volume.commit()
    print(
        json.dumps(
            {
                "phase": "active8_reduce_complete",
                "run_identity_sha256": result["run_identity_sha256"],
                "inventory_sha256": result["inventory_sha256"],
                "counts": result["counts"],
                "reused": result["reused"],
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
    timeout=24 * 3600,
    volumes={"/artifacts": artifact_volume},
)
def driver(
    unified_manifest_path: str,
    audit_root: str,
    mmp_root: str,
    support_contract_path: str,
    output_root: str,
    candidate_cache_size: int,
    partitions: tuple[str, ...],
    target_entries_per_range: int,
    source_revision: dict[str, object],
) -> dict[str, object]:
    """Plan once, map only absent tasks, then invoke the strict reducer."""

    loaded = _imports()
    artifact_volume.reload()
    contract = loaded["load_gate_zero_runtime_contract"](Path(support_contract_path))
    source_manifest, declared = loaded["resolve_unified_manifest_shards"](
        Path(unified_manifest_path),
        audit_root=Path(audit_root),
        mmp_root=Path(mmp_root),
    )
    if (
        not partitions
        or len(partitions) != len(set(partitions))
        or any(partition not in _LEGACY_PARTITIONS for partition in partitions)
    ):
        raise ValueError("Active8 partitions must be a unique nonempty legacy-partition subset")
    declared = tuple(shard for shard in declared if shard.partition in partitions)
    if not declared:
        raise RuntimeError("Active8 partition filter selected no packed shards")
    if target_entries_per_range <= 0:
        raise ValueError("target_entries_per_range must be positive")
    shards = tuple(
        loaded["Active8SourceShard"](
            manifest_layer=shard.manifest_layer,
            envelope_layer=shard.envelope_layer,
            partition=shard.partition,
            relative_path=shard.relative_path,
            path=shard.path,
        )
        for shard in declared
    )
    plan = loaded["plan_active8_mapreduce"](
        shards,
        source_manifest_path=Path(unified_manifest_path),
        source_manifest=source_manifest,
        support_contract_sha256=contract.sha256,
        source_revision=source_revision,
        repo_root=REMOTE_ROOT,
        target_entries_per_range=target_entries_per_range,
        worker_resources={
            "cpu": _MAP_CPU,
            "memory_mb": _MAP_MEMORY_MB,
            "max_containers": _MAX_MAP_CONTAINERS,
            "omp_num_threads": _MAP_OMP_NUM_THREADS,
        },
    )
    completed = loaded["verified_completed_task_identities"](
        plan,
        output_root=Path(output_root),
    )
    missing_tasks = [
        task for task in plan["tasks"] if task["task_identity_sha256"] not in completed
    ]
    print(
        json.dumps(
            {
                "phase": "active8_map_plan",
                "run_identity_sha256": plan["run_identity_sha256"],
                "source_revision": plan["source_revision"],
                "expected_source_decisions": plan["expected_source_decisions"],
                "expected_map_tasks": plan["expected_map_tasks"],
                "missing_map_tasks": len(missing_tasks),
                "max_map_containers": _MAX_MAP_CONTAINERS,
                "target_entries_per_range": target_entries_per_range,
                "worker_resources": plan["worker_resources"],
                "partitions": list(partitions),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if missing_tasks:
        results = list(
            map_source.starmap(
                [
                    (
                        task,
                        support_contract_path,
                        output_root,
                        candidate_cache_size,
                    )
                    for task in missing_tasks
                ]
            )
        )
        if len(results) != len(missing_tasks):
            raise RuntimeError("active-8 map lost task results")
    # The reducer repeats the exact receipt census; driver ordering alone is
    # never treated as completeness evidence.
    return reduce_inventory.remote(plan, output_root)


@app.local_entrypoint()
def main(
    unified_manifest_path: str = "/artifacts/UNIFIED_PACKED_MANIFEST.json",
    audit_root: str = "/artifacts/edit_packed_v1",
    mmp_root: str = "/artifacts/mmp_packed_v1",
    support_contract_path: str = ("/root/compose/configs/editing_gate_zero_runtime_v2.json"),
    output_root: str = "/artifacts/active8_trace_inventory_v1",
    candidate_cache_size: int = 4096,
    partitions: str = "train,validation,test",
    target_entries_per_range: int = _DEFAULT_TARGET_ENTRIES_PER_RANGE,
):
    if candidate_cache_size <= 0:
        raise ValueError("candidate_cache_size must be positive")
    if target_entries_per_range <= 0:
        raise ValueError("target_entries_per_range must be positive")
    selected_partitions = tuple(
        partition.strip() for partition in partitions.split(",") if partition.strip()
    )
    if (
        not selected_partitions
        or len(selected_partitions) != len(set(selected_partitions))
        or any(partition not in _LEGACY_PARTITIONS for partition in selected_partitions)
    ):
        raise ValueError(
            "partitions must be a comma-separated unique nonempty subset of train,validation,test"
        )
    source_revision = _local_source_revision()
    result = driver.remote(
        unified_manifest_path,
        audit_root,
        mmp_root,
        support_contract_path,
        output_root,
        candidate_cache_size,
        selected_partitions,
        target_entries_per_range,
        source_revision,
    )
    print(
        json.dumps(
            {
                "phase": "active8_inventory_complete",
                "unified_manifest_path": unified_manifest_path,
                "output_root": output_root,
                "source_revision": source_revision,
                "max_map_containers": _MAX_MAP_CONTAINERS,
                "target_entries_per_range": target_entries_per_range,
                "worker_resources": {
                    "cpu": _MAP_CPU,
                    "memory_mb": _MAP_MEMORY_MB,
                    "max_containers": _MAX_MAP_CONTAINERS,
                    "omp_num_threads": _MAP_OMP_NUM_THREADS,
                },
                "partitions": list(selected_partitions),
                "training_launched": False,
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
