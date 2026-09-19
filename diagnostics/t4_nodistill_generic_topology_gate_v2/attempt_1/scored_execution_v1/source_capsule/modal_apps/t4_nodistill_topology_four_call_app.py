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
from modal_apps.genmol_t4_opt_app import image as _scoring_image
from modal_apps.run_process_v2_p50_app import (
    REMOTE_ROOT,
    ROOT,
    _validate_remote_revision,
)

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

image = _scoring_image
for _relative in _REQUIRED_FILES:
    image = image.add_local_file(
        ROOT / _relative, str(REMOTE_ROOT / _relative), copy=True
    )
image = image.add_local_dir(
    ROOT / CAPSULE_ROOT_RELATIVE_PATH,
    str(REMOTE_ROOT / CAPSULE_ROOT_RELATIVE_PATH),
    copy=True,
    ignore=("**/__pycache__/**", "**/*.pyc"),
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
    )

    return remote_preflight(
        task,
        REMOTE_ROOT,
        PRIVATE_ROOT,
        private_volume,
        validate_revision=_validate_remote_revision,
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
    )

    intent, reservations = prepare_dispatch(
        task,
        REMOTE_ROOT,
        PRIVATE_ROOT,
        private_volume,
        validate_revision=_validate_remote_revision,
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
    )

    return run_query(
        task,
        query_id,
        REMOTE_ROOT,
        PRIVATE_ROOT,
        private_volume,
        validate_revision=_validate_remote_revision,
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
    )

    return reduce_run(
        task,
        REMOTE_ROOT,
        PRIVATE_ROOT,
        private_volume,
        validate_revision=_validate_remote_revision,
        finalize_incomplete=finalize_incomplete,
    )
