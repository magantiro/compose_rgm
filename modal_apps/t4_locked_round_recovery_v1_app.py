"""One-shot settlement of an already locked T4 controller round.

The recovery consumes only immutable receipts from the original run.  It does
not propose, select, dock, resubmit, or replace any candidate.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
RECOVERY_PATH = ROOT / "configs/t4_locked_round_recovery_p1_2_r3_v1.json"
RECOVERY_MODULE_PATH = (
    ROOT / "src/compose_v4/experiments/t4_locked_round_recovery.py"
)
CAPSULE_ROOT = (
    ROOT / "diagnostics/t4_shared_controller_completion_v1/source_capsule_v4"
)
CONTRACT_PATH = (
    ROOT / "diagnostics/t4_shared_controller_completion_v1/scored_contract_v4.json"
)
AUTHORIZATION_PATH = (
    ROOT / "diagnostics/t4_shared_controller_completion_v1/scored_authorization_v4.json"
)
CAPSULE_MANIFEST_PATH = (
    ROOT
    / "diagnostics/t4_shared_controller_completion_v1/source_capsule_manifest_v4.json"
)
CELL_MOUNT = Path("/cell")
REMOTE_ROOT = Path("/capsule")
REMOTE_SEALED = Path("/sealed")


def _runtime_image() -> modal.Image:
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
                "PYTHONPATH": f"/:{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}",
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
    )
    required = (
        RECOVERY_PATH,
        RECOVERY_MODULE_PATH,
        CAPSULE_ROOT,
        CONTRACT_PATH,
        AUTHORIZATION_PATH,
        CAPSULE_MANIFEST_PATH,
    )
    if not all(path.exists() for path in required):
        return image
    return (
        image.add_local_file(
            Path(__file__).resolve(), "/t4_locked_round_recovery_v1_app.py", copy=True
        )
        .add_local_dir(CAPSULE_ROOT, str(REMOTE_ROOT), copy=True)
        .add_local_file(
            RECOVERY_MODULE_PATH,
            str(
                REMOTE_ROOT
                / "src/compose_v4/experiments/t4_locked_round_recovery.py"
            ),
            copy=True,
        )
        .add_local_file(
            RECOVERY_PATH, str(REMOTE_SEALED / "recovery_contract.json"), copy=True
        )
        .add_local_file(
            CONTRACT_PATH, str(REMOTE_SEALED / "scored_contract.json"), copy=True
        )
        .add_local_file(
            AUTHORIZATION_PATH,
            str(REMOTE_SEALED / "scored_authorization.json"),
            copy=True,
        )
        .add_local_file(
            CAPSULE_MANIFEST_PATH,
            str(REMOTE_SEALED / "source_capsule_manifest.json"),
            copy=True,
        )
    )


def _load_envelope(path: Path, payload_identity: Any) -> dict[str, Any]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != (
        payload_identity(payload)
    ):
        raise ValueError(f"invalid self-hashed envelope: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_local_implementation(recovery: dict[str, Any]) -> None:
    expected = recovery.get("implementation_sha256")
    paths = {
        "modal_apps/t4_locked_round_recovery_v1_app.py": Path(__file__).resolve(),
        "src/compose_v4/experiments/t4_locked_round_recovery.py": RECOVERY_MODULE_PATH,
    }
    if not isinstance(expected, dict) or set(expected) != set(paths):
        raise ValueError("recovery implementation identity census drift")
    for relative, path in paths.items():
        if _sha256_file(path) != expected[relative]:
            raise ValueError(f"recovery implementation identity drift: {relative}")


PREPARATION = json.loads(
    (ROOT / "configs/t4_shared_controller_completion_v1.json").read_text()
)
CELL = next(row for row in PREPARATION["cells"] if row["cell_key"] == "parp1_2_d04")
volume = modal.Volume.from_name(CELL["volume"], create_if_missing=False)
app = modal.App("compose-t4-locked-round-recovery-v1")
image = _runtime_image()


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=4096,
    timeout=600,
    retries=0,
    volumes={str(CELL_MOUNT): volume},
    include_source=False,
)
def settle_locked_round(task: dict[str, Any]) -> dict[str, Any]:
    import time

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.t4_locked_round_recovery import (
        recover_locked_round,
    )
    from compose_v4.experiments.t4_shared_controller_cell_runtime import ReceiptStore
    from compose_v4.experiments.t4_shared_controller_completion_contract import (
        payload_identity,
        sha256_file,
    )
    from compose_v4.experiments.t4_shared_controller_scored_contract import (
        validate_scored_contract,
    )
    from compose_v4.experiments.t4_shared_controller_scored_runtime import (
        checkpoint_controller_config,
        validate_launch_task,
    )

    recovery = _load_envelope(
        REMOTE_SEALED / "recovery_contract.json", payload_identity
    )
    if sha256_file(
        REMOTE_ROOT / "src/compose_v4/experiments/t4_locked_round_recovery.py"
    ) != recovery["implementation_sha256"][
        "src/compose_v4/experiments/t4_locked_round_recovery.py"
    ]:
        raise ValueError("remote recovery module identity drift")
    if task != {"recovery_payload_sha256": payload_identity(recovery)}:
        raise ValueError("recovery task identity drift")
    contract = validate_scored_contract(
        repository_root=REMOTE_ROOT,
        contract_path=REMOTE_SEALED / "scored_contract.json",
        authorization_path=REMOTE_SEALED / "scored_authorization.json",
        capsule_root=REMOTE_ROOT,
        capsule_manifest_path=REMOTE_SEALED / "source_capsule_manifest.json",
    )
    contract_payload = _load_envelope(
        REMOTE_SEALED / "scored_contract.json", payload_identity
    )
    authorization = _load_envelope(
        REMOTE_SEALED / "scored_authorization.json", payload_identity
    )
    manifest = _load_envelope(
        REMOTE_SEALED / "source_capsule_manifest.json", payload_identity
    )
    launch = recovery["launch"]
    validate_launch_task(
        launch,
        contract=contract_payload,
        contract_payload_sha256=payload_identity(contract_payload),
        contract_file_sha256=sha256_file(REMOTE_SEALED / "scored_contract.json"),
        authorization_receipt=authorization,
        authorization_receipt_sha256=sha256_file(
            REMOTE_SEALED / "scored_authorization.json"
        ),
        source_capsule_payload_sha256=payload_identity(manifest),
    )
    if contract != contract_payload:
        raise ValueError("validated scored contract drift")
    cells = [row for row in contract["cells"] if row["cell_key"] == recovery["cell_key"]]
    if len(cells) != 1 or cells[0]["volume"] != recovery["volume"]:
        raise ValueError("recovery cell or volume drift")
    cell = cells[0]

    volume.reload()
    store = ReceiptStore(CELL_MOUNT / recovery["run_id"], flush=volume.commit)
    completion_path = recovery["completion_path"]
    try:
        return store.read(completion_path)
    except FileNotFoundError:
        pass
    if store.read("launch.json") != launch:
        raise ValueError("durable launch receipt drift")
    try:
        store.read("result.json")
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("completed cell may not be recovered")

    paths = recovery["paths"]
    checkpoint = store.read("checkpoint.json")
    plan = store.read(paths["round_plan"])
    lock = store.read(paths["query_lock"])
    dispatch = store.read(paths["query_dispatch"])
    reservations = {
        query_id: store.read(path)
        for query_id, path in paths["reservations"].items()
    }
    receipts = {}
    for query_id, path in paths["receipts"].items():
        if path is None:
            try:
                store.read(
                    f"rounds/{recovery['expected']['round_after']:03d}/"
                    f"query_receipts/{query_id}.json"
                )
            except FileNotFoundError:
                continue
            raise ValueError(f"missing query {query_id} acquired a late receipt")
        receipts[query_id] = store.read(path)

    recovered = recover_locked_round(
        checkpoint=checkpoint,
        round_plan=plan,
        query_lock=lock,
        dispatch_intent=dispatch,
        reservations=reservations,
        receipts=receipts,
        expected=recovery["expected"],
        cell=cell,
        controller_config=checkpoint_controller_config(contract),
        budget_ceiling=contract["charged_calls_per_cell"],
        now=time.time(),
    )
    next_checkpoint = recovered["checkpoint"]
    immutable_checkpoint_path = paths["round_checkpoint"]
    try:
        existing = store.read(immutable_checkpoint_path)
    except FileNotFoundError:
        store.publish_once(immutable_checkpoint_path, next_checkpoint)
    else:
        if existing != next_checkpoint:
            raise ValueError("immutable recovered round checkpoint drift")
    store.replace("checkpoint.json", next_checkpoint)
    completion = {
        "schema_version": "t4_locked_round_recovery_result_v1",
        "run_id": recovery["run_id"],
        "cell_key": recovery["cell_key"],
        "round": recovery["expected"]["round_after"],
        "charged_before": checkpoint["charged_count"],
        "charged_after": next_checkpoint["charged_count"],
        "prior_checkpoint_payload_sha256": payload_identity(checkpoint),
        "recovered_checkpoint_payload_sha256": payload_identity(next_checkpoint),
        "query_lock_payload_sha256": payload_identity(lock),
        "settlement_action": recovered["settlement"]["action"],
        "receipt_statuses": recovered["settlement"]["receipt_statuses"],
        "new_oracle_calls": 0,
        "query_resubmissions": 0,
        "candidate_changes": 0,
    }
    store.publish_once(completion_path, completion)
    return completion


@app.local_entrypoint()
def main(mode: str = "preflight") -> None:
    from compose_v4.experiments.t4_shared_controller_completion_contract import (
        payload_identity,
    )

    missing = [
        str(path)
        for path in (
            RECOVERY_PATH,
            RECOVERY_MODULE_PATH,
            CAPSULE_ROOT,
            CONTRACT_PATH,
            AUTHORIZATION_PATH,
            CAPSULE_MANIFEST_PATH,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing recovery inputs: {missing}")
    recovery = _load_envelope(RECOVERY_PATH, payload_identity)
    _verify_local_implementation(recovery)
    task = {"recovery_payload_sha256": payload_identity(recovery)}
    if mode == "preflight":
        print(
            json.dumps(
                {
                    "ready": True,
                    "run_id": recovery["run_id"],
                    "cell_key": recovery["cell_key"],
                    "new_oracle_calls": 0,
                    "modal_calls_created": 0,
                },
                sort_keys=True,
            )
        )
        return
    if mode != "settle":
        raise ValueError("mode must be preflight or settle")
    print(json.dumps(settle_locked_round.remote(task), sort_keys=True))
