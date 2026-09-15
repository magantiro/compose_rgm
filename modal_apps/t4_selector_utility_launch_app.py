"""Modal workers for the authorized two-request selector utility panel."""

from __future__ import annotations

from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import _dock
from modal_apps.genmol_t4_opt_app import image as _scoring_image
from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    _validate_remote_revision,
    artifact_volume,
)

APP_SOURCE = "modal_apps/t4_selector_utility_launch_app.py"
TOOL_SOURCE = "tools/t4_selector_utility_launch.py"
REQUEST_LOCK_SOURCE = (
    "diagnostics/t4_selector_utility_prelock/attempt_1/request_lock.json"
)
image = (
    _scoring_image.add_local_file(
        ROOT / REQUEST_LOCK_SOURCE, str(REMOTE_ROOT / REQUEST_LOCK_SOURCE), copy=True
    )
    .add_local_file(ROOT / APP_SOURCE, str(REMOTE_ROOT / APP_SOURCE), copy=True)
    .add_local_file(ROOT / TOOL_SOURCE, str(REMOTE_ROOT / TOOL_SOURCE), copy=True)
)
app = modal.App("compose-t4-selector-utility")


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=2048,
    timeout=180,
    max_containers=1,
    retries=0,
)
def t4_selector_utility_preflight(task: dict[str, Any]) -> dict[str, Any]:
    from compose_v4.experiments.t4_selector_utility_launch import remote_preflight

    return remote_preflight(
        task, REMOTE_ROOT, validate_revision=_validate_remote_revision
    )


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=4096,
    timeout=480,
    max_containers=2,
    retries=0,
    scaledown_window=60,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def t4_selector_utility_worker(task: dict[str, Any]) -> dict[str, Any]:
    from compose_v4.experiments.t4_selector_utility_launch import run_worker

    return run_worker(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        validate_revision=_validate_remote_revision,
        dock=_dock,
    )


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=2048,
    timeout=300,
    max_containers=1,
    retries=0,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def t4_selector_utility_reduce(task: dict[str, Any]) -> dict[str, Any]:
    from compose_v4.experiments.t4_selector_utility_launch import reduce_run

    return reduce_run(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        validate_revision=_validate_remote_revision,
    )
