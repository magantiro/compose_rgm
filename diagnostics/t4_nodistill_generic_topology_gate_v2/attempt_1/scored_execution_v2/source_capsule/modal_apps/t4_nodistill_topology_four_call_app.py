"""Modal app for the exact authorized four-call topology diagnostic.

Importing this module defines functions only.  The source-bound execution
contract and capsule must already exist before the app can be deployed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from compose_v4.experiments.t4_nodistill_topology_four_call_contract import (
    APP_NAME,
    APP_SOURCE,
    AUTHORIZATION_RELATIVE_PATH,
    CANDIDATE_PROPOSAL_RELATIVE_PATH,
    CAPSULE_MANIFEST_RELATIVE_PATH,
    CAPSULE_ROOT_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    LAUNCHER_SOURCE,
    PREPARATION_CAPSULE_MANIFEST_RELATIVE_PATH,
    SCORED_CONTRACT_RELATIVE_PATH,
    VOLUME_NAME,
    VOLUME_ROOT,
)

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/t4_nodistill_topology_four_call")
MOOD = "https://raw.githubusercontent.com/SeulLee05/MOOD/main/scorer"

PRIVATE_ROOT = Path(VOLUME_ROOT)
_REQUIRED_FILES = (
    SCORED_CONTRACT_RELATIVE_PATH,
    AUTHORIZATION_RELATIVE_PATH,
    CANDIDATE_PROPOSAL_RELATIVE_PATH,
    PREPARATION_CAPSULE_MANIFEST_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    CAPSULE_MANIFEST_RELATIVE_PATH,
    APP_SOURCE,
    LAUNCHER_SOURCE,
)
for _relative in _REQUIRED_FILES:
    if not (ROOT / _relative).is_file():
        raise FileNotFoundError(
            f"four-call executable package is not sealed: {ROOT / _relative}"
        )
if not (ROOT / CAPSULE_ROOT_RELATIVE_PATH).is_dir():
    raise FileNotFoundError("four-call exact source capsule is absent")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("openbabel", "curl", "ca-certificates")
    .pip_install("numpy==1.26.4", "rdkit==2024.3.5")
    .run_commands(
        "mkdir -p /opt/dock/receptors",
        f"curl -sSL -o /opt/dock/qvina02 {MOOD}/qvina02",
        "chmod +x /opt/dock/qvina02",
        f"curl -sSL -o /opt/dock/receptors/parp1.pdbqt {MOOD}/receptors/parp1.pdbqt",
    )
    .add_local_dir(
        ROOT / CAPSULE_ROOT_RELATIVE_PATH,
        str(REMOTE_ROOT),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
    .add_local_dir(
        ROOT / CAPSULE_ROOT_RELATIVE_PATH,
        str(REMOTE_ROOT / CAPSULE_ROOT_RELATIVE_PATH),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
    .add_local_file(
        ROOT / EXECUTION_CONTRACT_RELATIVE_PATH,
        str(REMOTE_ROOT / EXECUTION_CONTRACT_RELATIVE_PATH),
        copy=True,
    )
    .add_local_file(
        ROOT / CAPSULE_MANIFEST_RELATIVE_PATH,
        str(REMOTE_ROOT / CAPSULE_MANIFEST_RELATIVE_PATH),
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT}",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
)

app = modal.App(APP_NAME)
private_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _dock_once(smiles: str, tag: str, seed: int, box: dict[str, Any]) -> float | None:
    from compose_v4.experiments.t4_docking_adapter import dock_t4

    return dock_t4(smiles, tag, seed, box=box, cpu=1)


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=2048,
    timeout=300,
    max_containers=1,
    retries=0,
    volumes={str(PRIVATE_ROOT): private_volume},
)
def t4_nodistill_topology_four_call_preflight(
    task: dict[str, Any],
) -> dict[str, Any]:
    from compose_v4.experiments.t4_nodistill_topology_four_call_runtime import (
        remote_preflight,
        validate_capsule_image_revision,
    )

    return remote_preflight(
        task,
        REMOTE_ROOT,
        PRIVATE_ROOT,
        private_volume,
        validate_revision=lambda value: validate_capsule_image_revision(
            value, REMOTE_ROOT
        ),
    )


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=2048,
    timeout=600,
    max_containers=1,
    retries=0,
    volumes={str(PRIVATE_ROOT): private_volume},
)
def t4_nodistill_topology_four_call_driver(
    task: dict[str, Any],
) -> dict[str, Any]:
    from compose_v4.experiments.t4_nodistill_topology_four_call_runtime import (
        prepare_dispatch,
        publish_dispatch_receipt,
        validate_capsule_image_revision,
    )

    intent, reservations = prepare_dispatch(
        task,
        REMOTE_ROOT,
        PRIVATE_ROOT,
        private_volume,
        validate_revision=lambda value: validate_capsule_image_revision(
            value, REMOTE_ROOT
        ),
    )
    calls = []
    for ordinal, reservation in enumerate(reservations):
        query_id = reservation["query_id"]
        call = t4_nodistill_topology_four_call_worker.spawn(task, query_id)
        receipt = publish_dispatch_receipt(
            task=task,
            artifact_root=PRIVATE_ROOT,
            volume=private_volume,
            query_id=query_id,
            query_ordinal=ordinal,
            function_call_id=call.object_id,
        )
        calls.append(receipt)
    return {
        "schema_version": "t4_nodistill_topology_four_call_driver_result_v1",
        "run_id": task["run_id"],
        "dispatch_intent": intent,
        "calls": calls,
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "docking_calls_by_driver": 0,
    }


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=4096,
    timeout=480,
    max_containers=4,
    retries=0,
    scaledown_window=60,
    volumes={str(PRIVATE_ROOT): private_volume},
)
def t4_nodistill_topology_four_call_worker(
    task: dict[str, Any], query_id: str
) -> dict[str, Any]:
    from compose_v4.experiments.t4_nodistill_topology_four_call_runtime import (
        run_query,
        validate_capsule_image_revision,
    )

    return run_query(
        task,
        query_id,
        REMOTE_ROOT,
        PRIVATE_ROOT,
        private_volume,
        validate_revision=lambda value: validate_capsule_image_revision(
            value, REMOTE_ROOT
        ),
        dock=_dock_once,
    )


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=2048,
    timeout=300,
    max_containers=1,
    retries=0,
    volumes={str(PRIVATE_ROOT): private_volume},
)
def t4_nodistill_topology_four_call_reduce(
    task: dict[str, Any], finalize_incomplete: bool = False
) -> dict[str, Any]:
    from compose_v4.experiments.t4_nodistill_topology_four_call_runtime import (
        reduce_run,
        validate_capsule_image_revision,
    )

    return reduce_run(
        task,
        REMOTE_ROOT,
        PRIVATE_ROOT,
        private_volume,
        validate_revision=lambda value: validate_capsule_image_revision(
            value, REMOTE_ROOT
        ),
        finalize_incomplete=finalize_incomplete,
    )
