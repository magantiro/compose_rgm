"""Run the train-only atom-restatement orbit audit as resumable Modal map/reduce.

Each CPU map invocation owns one exact frozen Active8 train shard. Successful
rows and their self-hashed receipt are immutable content-addressed objects on
the existing ``compose-v4-artifacts`` volume. Retries reuse only byte-verified
receipts. The reducer refuses missing, corrupt, or unexpected task objects and
atomically publishes the final row stream and aggregate summary.

This app builds diagnostic evidence only. It has no training function and this
module does not launch a job when imported.

    modal run --detach modal_apps/run_atom_restate_neural_orbit_full_corpus.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = Path("/root/compose")
_MAX_MAP_CONTAINERS_LIMIT = 64
_MAX_MAP_CONTAINERS = int(os.environ.get("COMPOSE_RESTATE_ORBIT_MAX_MAP_CONTAINERS", "64"))
if not 1 <= _MAX_MAP_CONTAINERS <= _MAX_MAP_CONTAINERS_LIMIT:
    raise ValueError("COMPOSE_RESTATE_ORBIT_MAX_MAP_CONTAINERS must lie in [1, 64]")
_MAP_CPU = 2.0
_MAP_MEMORY_MB = 8192
_MAP_OMP_NUM_THREADS = 1

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
        ROOT / "modal_apps" / "run_atom_restate_neural_orbit_full_corpus.py",
        str(REMOTE_ROOT / "modal_apps" / "run_atom_restate_neural_orbit_full_corpus.py"),
        copy=True,
    )
)

app = modal.App("compose-v4-atom-restate-neural-orbit-full-corpus")
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts",
    create_if_missing=False,
)


def _imports() -> dict[str, object]:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.atom_restate_neural_orbit_full_corpus import (
        FullCorpusAuditInputs,
        implementation_identity,
        load_full_corpus_driver_contract,
        map_resumable_full_corpus_shard,
        plan_resumable_full_corpus_audit,
        reduce_resumable_full_corpus_audit,
        repository_source_revision,
        verified_completed_map_tasks,
    )

    return {
        "FullCorpusAuditInputs": FullCorpusAuditInputs,
        "implementation_identity": implementation_identity,
        "load_full_corpus_driver_contract": load_full_corpus_driver_contract,
        "map_resumable_full_corpus_shard": map_resumable_full_corpus_shard,
        "plan_resumable_full_corpus_audit": plan_resumable_full_corpus_audit,
        "reduce_resumable_full_corpus_audit": reduce_resumable_full_corpus_audit,
        "repository_source_revision": repository_source_revision,
        "verified_completed_map_tasks": verified_completed_map_tasks,
    }


def _inputs(
    loaded: dict[str, object],
    *,
    driver_contract_path: str,
    orbit_contract_path: str,
    active8_inventory_path: str,
    unified_manifest_path: str,
    audit_root: str,
    mmp_root: str,
    output_root: str,
    max_problem_address_examples: int,
):
    return loaded["FullCorpusAuditInputs"](
        driver_contract_path=Path(driver_contract_path),
        orbit_contract_path=Path(orbit_contract_path),
        active8_inventory_path=Path(active8_inventory_path),
        unified_manifest_path=Path(unified_manifest_path),
        audit_root=Path(audit_root),
        mmp_root=Path(mmp_root),
        output_dir=Path(output_root),
        repository_root=REMOTE_ROOT,
        max_problem_address_examples=max_problem_address_examples,
    )


@app.function(
    image=image,
    cpu=_MAP_CPU,
    memory=_MAP_MEMORY_MB,
    timeout=6 * 3600,
    max_containers=_MAX_MAP_CONTAINERS,
    volumes={"/artifacts": artifact_volume},
)
def map_shard(
    task_identity_sha256: str,
    paths: dict[str, str],
    source_revision: dict[str, object],
    max_problem_address_examples: int,
) -> dict[str, object]:
    """Build or verify one immutable exact-shard receipt."""

    loaded = _imports()
    artifact_volume.reload()
    inputs = _inputs(
        loaded,
        **paths,
        max_problem_address_examples=max_problem_address_examples,
    )
    contract = loaded["load_full_corpus_driver_contract"](Path(paths["driver_contract_path"]))
    result = loaded["map_resumable_full_corpus_shard"](
        inputs,
        driver_contract=contract,
        source_revision=source_revision,
        output_root=Path(paths["output_root"]),
        task_identity_sha256=task_identity_sha256,
    )
    artifact_volume.commit()
    print(
        json.dumps(
            {
                "phase": "atom_restate_orbit_map_shard",
                "task_identity_sha256": task_identity_sha256,
                "row_count": result["row_evidence"]["row_count"],
                "summary_sha256": result["summary_sha256"],
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
    timeout=6 * 3600,
    volumes={"/artifacts": artifact_volume},
)
def reduce_audit(
    paths: dict[str, str],
    source_revision: dict[str, object],
    max_problem_address_examples: int,
) -> dict[str, object]:
    """Reduce only the exact complete set of byte-verified map receipts."""

    loaded = _imports()
    artifact_volume.reload()
    inputs = _inputs(
        loaded,
        **paths,
        max_problem_address_examples=max_problem_address_examples,
    )
    contract = loaded["load_full_corpus_driver_contract"](Path(paths["driver_contract_path"]))
    result = loaded["reduce_resumable_full_corpus_audit"](
        inputs,
        driver_contract=contract,
        source_revision=source_revision,
        output_root=Path(paths["output_root"]),
    )
    artifact_volume.commit()
    print(
        json.dumps(
            {
                "phase": "atom_restate_orbit_reduce_complete",
                "run_identity_sha256": result["run_identity_sha256"],
                "row_count": result["row_evidence"]["row_count"],
                "summary_sha256": result["summary_sha256"],
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
    paths: dict[str, str],
    source_revision: dict[str, object],
    max_problem_address_examples: int,
) -> dict[str, object]:
    """Plan once, map only missing verified shards, then strictly reduce."""

    loaded = _imports()
    artifact_volume.reload()
    inputs = _inputs(
        loaded,
        **paths,
        max_problem_address_examples=max_problem_address_examples,
    )
    contract = loaded["load_full_corpus_driver_contract"](Path(paths["driver_contract_path"]))
    plan = loaded["plan_resumable_full_corpus_audit"](
        inputs,
        driver_contract=contract,
        source_revision=source_revision,
    )
    completed = loaded["verified_completed_map_tasks"](
        plan,
        output_root=Path(paths["output_root"]),
    )
    missing = [
        task["task_identity_sha256"]
        for task in plan["tasks"]
        if task["task_identity_sha256"] not in completed
    ]
    print(
        json.dumps(
            {
                "phase": "atom_restate_orbit_map_plan",
                "run_identity_sha256": plan["run_identity_sha256"],
                "expected_map_tasks": plan["expected_map_tasks"],
                "verified_completed_map_tasks": len(completed),
                "missing_map_tasks": len(missing),
                "max_map_containers": _MAX_MAP_CONTAINERS,
                "worker_cpu": _MAP_CPU,
                "worker_memory_mb": _MAP_MEMORY_MB,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if missing:
        results = list(
            map_shard.starmap(
                [
                    (
                        task_identity,
                        paths,
                        source_revision,
                        max_problem_address_examples,
                    )
                    for task_identity in missing
                ]
            )
        )
        if len(results) != len(missing):
            raise RuntimeError("atom-restatement orbit map lost task results")
    return reduce_audit.remote(
        paths,
        source_revision,
        max_problem_address_examples,
    )


@app.local_entrypoint()
def main(
    driver_contract_path: str = (
        "/root/compose/configs/editing_atom_restate_neural_orbit_full_corpus_v1.json"
    ),
    orbit_contract_path: str = (
        "/root/compose/configs/editing_atom_restate_neural_orbit_audit_v1.json"
    ),
    active8_inventory_path: str = "",
    unified_manifest_path: str = "/artifacts/UNIFIED_PACKED_MANIFEST.json",
    audit_root: str = "/artifacts/edit_packed_v1",
    mmp_root: str = "/artifacts/mmp_packed_v1",
    output_root: str = (
        "/artifacts/_frozen_corpus_audits/atom_restate_neural_orbit_full_corpus_v1"
    ),
    max_problem_address_examples: int = 256,
):
    if max_problem_address_examples < 0:
        raise ValueError("max_problem_address_examples must be nonnegative")
    if not active8_inventory_path:
        raise ValueError(
            "--active8-inventory-path is required after a train-only Active8 "
            "inventory has been frozen"
        )
    loaded = _imports()
    local_contract = loaded["load_full_corpus_driver_contract"](
        ROOT / "configs/editing_atom_restate_neural_orbit_full_corpus_v1.json"
    )
    if local_contract.get("authorizes_audit_execution") is not True:
        raise RuntimeError(
            "full-corpus execution is blocked until the contract binds a frozen "
            "train-only Active8 parent"
        )
    source_revision = loaded["repository_source_revision"](
        ROOT,
        required_core_revision=local_contract["required_core_revision"],
    )
    source_revision["implementation_sha256"] = loaded["implementation_identity"](
        repository_root=ROOT
    )["implementation_sha256"]
    paths = {
        "driver_contract_path": driver_contract_path,
        "orbit_contract_path": orbit_contract_path,
        "active8_inventory_path": active8_inventory_path,
        "unified_manifest_path": unified_manifest_path,
        "audit_root": audit_root,
        "mmp_root": mmp_root,
        "output_root": output_root,
    }
    result = driver.remote(
        paths,
        source_revision,
        max_problem_address_examples,
    )
    print(
        json.dumps(
            {
                "phase": "atom_restate_orbit_full_corpus_complete",
                "source_revision": source_revision,
                "output_root": output_root,
                "training_launched": False,
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
