"""Run the reusable validation-only cycle-open equivalence audit on Modal.

Each CPU worker publishes one immutable physical-shard receipt before it
returns. A driver or reducer failure therefore does not discard completed
chemistry work. The app reads only the pinned validation inventory and never
launches training or mutates the production process.

Run only from a clean committed tree:

    MODAL_PROFILE=nitya modal run --detach \
      modal_apps/audit_editing_cycle_open_component_equivalence.py
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
ALLOWED_OUTPUT_PARENT = ARTIFACT_ROOT / "_frozen_corpus_audits"
DEFAULT_OUTPUT_ROOT = ALLOWED_OUTPUT_PARENT / "cycle_open_component_equivalence_v1"
DEFAULT_CONTRACT = REMOTE_ROOT / "configs/editing_cycle_open_component_equivalence_v1.json"
DEFAULT_INVENTORY = (
    ARTIFACT_ROOT / "active8_trace_inventory_legacy3_validation_1ef8941/objects/manifests/"
    "bbf44483c4d14bfd6c4bde7cbd2dc4fc708c46a14acff5eadb0b6f82902832ab.json"
)
DEFAULT_UNIFIED_MANIFEST = ARTIFACT_ROOT / "UNIFIED_PACKED_MANIFEST.json"
DEFAULT_AUDIT_ROOT = ARTIFACT_ROOT / "edit_packed_v1"
DEFAULT_MMP_ROOT = ARTIFACT_ROOT / "mmp_packed_v1"

_MAX_MAP_CONTAINERS_LIMIT = 64
_MAX_MAP_CONTAINERS = int(os.environ.get("COMPOSE_CYCLE_OPEN_EQUIVALENCE_MAX_CONTAINERS", "32"))
if not 1 <= _MAX_MAP_CONTAINERS <= _MAX_MAP_CONTAINERS_LIMIT:
    raise ValueError("COMPOSE_CYCLE_OPEN_EQUIVALENCE_MAX_CONTAINERS must lie in [1, 64]")
_MAP_CPU = 2.0
_MAP_MEMORY_MB = 8192
_OMP_NUM_THREADS = 1

AUDIT_RUNTIME_REQUIREMENTS = (
    "torch==2.4.0",
    "numpy==1.26.4",
    "scipy==1.13.1",
    "networkx==3.3",
    "rdkit==2024.3.5",
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*AUDIT_RUNTIME_REQUIREMENTS)
    .env(
        {
            "PYTHONPATH": os.pathsep.join((str(REMOTE_ROOT / "src"), str(REMOTE_ROOT / "scripts"))),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(_OMP_NUM_THREADS),
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
    .add_local_dir(ROOT / "configs", str(REMOTE_ROOT / "configs"), copy=True)
    .add_local_file(
        ROOT / "modal_apps/audit_editing_cycle_open_component_equivalence.py",
        str(REMOTE_ROOT / "modal_apps/audit_editing_cycle_open_component_equivalence.py"),
        copy=True,
    )
)

app = modal.App("compose-v4-cycle-open-component-equivalence")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


def _imports() -> dict[str, object]:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.data.active8_trace_inventory import load_active8_trace_admission
    from compose_v4.data.immutable_artifact import write_bytes_if_absent
    from compose_v4.data.packed_charge_policy_audit import resolve_unified_manifest_shards
    from compose_v4.data.packed_trace_store import read_frozen_source_addressed_packed_shard
    from compose_v4.experiments.editing_cycle_open_component_equivalence import (
        audit_one_addressed_shard,
        file_sha256,
        load_contract,
        reduce_shard_receipts,
        validate_shard_receipt,
    )
    from compose_v4.experiments.editing_cycle_open_component_equivalence_runtime import (
        active8_partition_census,
        build_plan,
        implementation_identity,
        load_json_artifact,
        pretty_json_bytes,
        validate_parent_admission,
        validate_plan,
    )

    return {
        "active8_partition_census": active8_partition_census,
        "audit_one_addressed_shard": audit_one_addressed_shard,
        "build_plan": build_plan,
        "file_sha256": file_sha256,
        "implementation_identity": implementation_identity,
        "load_active8_trace_admission": load_active8_trace_admission,
        "load_contract": load_contract,
        "load_json_artifact": load_json_artifact,
        "pretty_json_bytes": pretty_json_bytes,
        "read_frozen_source_addressed_packed_shard": read_frozen_source_addressed_packed_shard,
        "reduce_shard_receipts": reduce_shard_receipts,
        "resolve_unified_manifest_shards": resolve_unified_manifest_shards,
        "validate_parent_admission": validate_parent_admission,
        "validate_plan": validate_plan,
        "validate_shard_receipt": validate_shard_receipt,
        "write_bytes_if_absent": write_bytes_if_absent,
    }


def _validate_output_root(value: str | Path) -> Path:
    output = Path(value)
    try:
        output.relative_to(ALLOWED_OUTPUT_PARENT)
    except ValueError as error:
        raise ValueError(f"output_root must lie beneath {ALLOWED_OUTPUT_PARENT}") from error
    if output == ALLOWED_OUTPUT_PARENT:
        raise ValueError("output_root must name a task-specific child directory")
    return output


def _load_contract_and_admission(
    loaded: dict[str, object], *, contract_path: str, inventory_path: str
):
    contract = loaded["load_contract"](Path(contract_path))
    parent = contract["parent_active8_identity"]
    admission = loaded["load_active8_trace_admission"](
        Path(inventory_path),
        expected_manifest_file_sha256=parent["inventory_manifest_file_sha256"],
        expected_inventory_sha256=parent["inventory_sha256"],
        expected_effective_source_corpus_cache_sha256=parent[
            "effective_source_corpus_cache_sha256"
        ],
        expected_support_contract_sha256=parent["support_contract_sha256"],
    )
    loaded["validate_parent_admission"](contract, admission)
    return contract, admission


def _artifact_paths(output: Path, plan_sha256: str) -> dict[str, Path]:
    return {
        "plan": output / f"plan.{plan_sha256}.json",
        "receipt_dir": output / "receipts" / plan_sha256,
        "result": output / f"result.{plan_sha256}.json",
    }


def _matching_shard(declared: tuple[object, ...], task: dict[str, Any]) -> object:
    matching = tuple(
        shard
        for shard in declared
        if (
            shard.envelope_layer == task["layer"]
            and shard.partition == task["partition"]
            and shard.relative_path == task["relative_path"]
            and shard.path.name == task["packed_shard_name"]
        )
    )
    if len(matching) != 1:
        raise RuntimeError("planned task does not resolve to exactly one physical shard")
    return matching[0]


def _load_verified_plan(loaded, *, path: str, contract, admission):
    plan = loaded["validate_plan"](
        loaded["load_json_artifact"](Path(path), description="equivalence plan"),
        expected_contract_sha256=contract["contract_sha256"],
    )
    if plan["implementation"] != loaded["implementation_identity"](repo_root=REMOTE_ROOT) or plan[
        "active8_partition_census"
    ] != loaded["active8_partition_census"](admission):
        raise RuntimeError("remote implementation or Active8 census differs from plan")
    return plan


@app.function(
    image=image,
    cpu=_MAP_CPU,
    memory=_MAP_MEMORY_MB,
    timeout=4 * 3600,
    max_containers=_MAX_MAP_CONTAINERS,
    volumes={"/artifacts": artifact_volume},
)
def audit_physical_shard(
    task_index: int,
    plan_path: str,
    contract_path: str,
    inventory_path: str,
    unified_manifest_path: str,
    audit_root: str,
    mmp_root: str,
    output_root: str,
) -> dict[str, object]:
    loaded = _imports()
    artifact_volume.reload()
    output = _validate_output_root(output_root)
    contract, admission = _load_contract_and_admission(
        loaded, contract_path=contract_path, inventory_path=inventory_path
    )
    plan = _load_verified_plan(loaded, path=plan_path, contract=contract, admission=admission)
    if task_index < 0 or task_index >= plan["task_count"]:
        raise RuntimeError("task index lies outside the plan")
    task = plan["tasks"][task_index]
    destination = (
        _artifact_paths(output, plan["plan_sha256"])["receipt_dir"] / task["receipt_filename"]
    )
    if destination.is_file():
        receipt = loaded["validate_shard_receipt"](
            loaded["load_json_artifact"](destination, description="shard receipt"),
            expected_plan_sha256=plan["plan_sha256"],
            expected_implementation_sha256=plan["implementation"]["implementation_sha256"],
            expected_task=task,
        )
        reused = True
    else:
        _manifest, declared = loaded["resolve_unified_manifest_shards"](
            Path(unified_manifest_path),
            audit_root=Path(audit_root),
            mmp_root=Path(mmp_root),
        )
        shard = _matching_shard(tuple(declared), task)
        rows = loaded["read_frozen_source_addressed_packed_shard"](
            shard.path,
            expected_shard_sha256=task["packed_shard_content_sha256"],
            expected_manifest_sha256=task["packed_manifest_sha256"],
            expected_overlay_sha256=task["packed_provenance_overlay_sha256"],
            verify_fraction=0.0,
        )
        receipt = loaded["audit_one_addressed_shard"](
            task,
            rows,
            admission,
            plan_sha256=plan["plan_sha256"],
            implementation_sha256=plan["implementation"]["implementation_sha256"],
            maximum_oracle_structures=contract["comparison_policy"]["maximum_oracle_structures"],
        )
        loaded["write_bytes_if_absent"](destination, loaded["pretty_json_bytes"](receipt))
        artifact_volume.commit()
        reused = False
    return {
        "task_index": task_index,
        "receipt_sha256": receipt["receipt_sha256"],
        "unique_sources": len(receipt["source_records"]),
        "reused": reused,
    }


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=2 * 3600,
    volumes={"/artifacts": artifact_volume},
)
def reduce_audit(
    plan_path: str, contract_path: str, inventory_path: str, output_root: str
) -> dict[str, object]:
    loaded = _imports()
    artifact_volume.reload()
    output = _validate_output_root(output_root)
    contract, admission = _load_contract_and_admission(
        loaded, contract_path=contract_path, inventory_path=inventory_path
    )
    plan = _load_verified_plan(loaded, path=plan_path, contract=contract, admission=admission)
    paths = _artifact_paths(output, plan["plan_sha256"])
    expected_names = {task["receipt_filename"] for task in plan["tasks"]}
    observed_names = (
        {path.name for path in paths["receipt_dir"].iterdir() if path.suffix == ".json"}
        if paths["receipt_dir"].is_dir()
        else set()
    )
    if observed_names != expected_names:
        raise RuntimeError(
            "receipt directory differs from exact plan: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unexpected={sorted(observed_names - expected_names)}"
        )
    receipts = [
        loaded["load_json_artifact"](
            paths["receipt_dir"] / task["receipt_filename"],
            description=f"receipt {task['task_index']}",
        )
        for task in plan["tasks"]
    ]
    evidence_identity = {
        "contract_sha256": contract["contract_sha256"],
        "parent_active8_identity": contract["parent_active8_identity"],
        "partitions": ["validation"],
        "excluded_partitions": ["train", "controller_validation", "test"],
        "source_scope": contract["source_scope"],
        "edge_scope": contract["edge_scope"],
    }
    result = loaded["reduce_shard_receipts"](
        receipts,
        expected_tasks=plan["tasks"],
        plan_sha256=plan["plan_sha256"],
        implementation_sha256=plan["implementation"]["implementation_sha256"],
        evidence_identity=evidence_identity,
    )
    loaded["write_bytes_if_absent"](paths["result"], loaded["pretty_json_bytes"](result))
    artifact_volume.commit()
    return {
        "plan_sha256": plan["plan_sha256"],
        "result_path": str(paths["result"]),
        "result_sha256": result["result_sha256"],
        "passed": result["equivalence_gate"]["passed"],
        "unique_sources": result["unique_source_count"],
        "semantic_edges": result["totals"]["semantic_aromatic_edge_count"],
        "training_launched": False,
    }


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
    contract_path: str,
    inventory_path: str,
    output_root: str,
    code_revision: dict[str, object],
) -> dict[str, object]:
    loaded = _imports()
    artifact_volume.reload()
    output = _validate_output_root(output_root)
    contract, admission = _load_contract_and_admission(
        loaded, contract_path=contract_path, inventory_path=inventory_path
    )
    _manifest, declared = loaded["resolve_unified_manifest_shards"](
        Path(unified_manifest_path),
        audit_root=Path(audit_root),
        mmp_root=Path(mmp_root),
    )
    plan = loaded["build_plan"](
        contract,
        admission,
        tuple(declared),
        unified_manifest_file_sha256=loaded["file_sha256"](Path(unified_manifest_path)),
        inputs={
            "contract_path": contract_path,
            "active8_inventory_path": inventory_path,
            "unified_manifest_path": unified_manifest_path,
            "audit_root": audit_root,
            "mmp_root": mmp_root,
        },
        code_revision=code_revision,
        implementation=loaded["implementation_identity"](repo_root=REMOTE_ROOT),
    )
    paths = _artifact_paths(output, plan["plan_sha256"])
    loaded["write_bytes_if_absent"](paths["plan"], loaded["pretty_json_bytes"](plan))
    artifact_volume.commit()

    missing = []
    for task in plan["tasks"]:
        receipt_path = paths["receipt_dir"] / task["receipt_filename"]
        if not receipt_path.is_file():
            missing.append(task["task_index"])
        else:
            loaded["validate_shard_receipt"](
                loaded["load_json_artifact"](receipt_path, description="existing receipt"),
                expected_plan_sha256=plan["plan_sha256"],
                expected_implementation_sha256=plan["implementation"]["implementation_sha256"],
                expected_task=task,
            )
    print(
        json.dumps(
            {
                "phase": "cycle_open_component_equivalence_plan_complete",
                "plan_sha256": plan["plan_sha256"],
                "task_count": plan["task_count"],
                "missing_tasks": len(missing),
                "zero_support_shards": len(plan["zero_cycle_attach_support_shards"]),
                "training_launched": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if missing:
        returned = list(
            audit_physical_shard.starmap(
                [
                    (
                        index,
                        str(paths["plan"]),
                        contract_path,
                        inventory_path,
                        unified_manifest_path,
                        audit_root,
                        mmp_root,
                        str(output),
                    )
                    for index in missing
                ],
                return_exceptions=True,
            )
        )
        failures = [value for value in returned if isinstance(value, BaseException)]
        if failures:
            raise RuntimeError(
                f"{len(failures)} shard task(s) failed; completed receipts remain durable"
            ) from failures[0]
        if len(returned) != len(missing):
            raise RuntimeError("Modal map returned an incomplete task result set")
    return reduce_audit.remote(str(paths["plan"]), contract_path, inventory_path, str(output))


def _local_git_revision() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot bind Modal equivalence audit to Git") from error
    if dirty:
        raise RuntimeError("Modal equivalence audit requires a clean committed tree")
    return {"commit": commit, "tree_dirty": False}


@app.local_entrypoint()
def main(
    unified_manifest_path: str = str(DEFAULT_UNIFIED_MANIFEST),
    audit_root: str = str(DEFAULT_AUDIT_ROOT),
    mmp_root: str = str(DEFAULT_MMP_ROOT),
    contract_path: str = str(DEFAULT_CONTRACT),
    inventory_path: str = str(DEFAULT_INVENTORY),
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
):
    _validate_output_root(output_root)
    result = driver.remote(
        unified_manifest_path,
        audit_root,
        mmp_root,
        contract_path,
        inventory_path,
        output_root,
        _local_git_revision(),
    )
    print(
        json.dumps(
            {
                "phase": "cycle_open_component_equivalence_complete",
                "output_root": output_root,
                "max_map_containers": _MAX_MAP_CONTAINERS,
                "training_launched": False,
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
