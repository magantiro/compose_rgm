"""Five independent zero-oracle route lowerings on the qualified CPU image."""

import modal

from modal_apps.pmo_trajectory_value_app import image as prior_image
from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    _validate_remote_revision,
    artifact_volume,
)

image = prior_image
for path in (
    "modal_apps/pmo_route_support_app.py",
    "docs/PMO_ROUTE_SUPPORT_REPAIR.md",
    "diagnostics/pmo_route_support/prior_spawn.json",
):
    image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)
app = modal.App("compose-pmo-route-support")
shared = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 8192,
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
}


@app.function(**shared, max_containers=5, timeout=900)
def worker(task):
    from compose_v4.experiments.pmo_route_support import worker_remote

    return worker_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _validate_remote_revision
    )


@app.function(**shared, max_containers=1, timeout=1200)
def run(task):
    from compose_v4.experiments.pmo_archive_pilot import Store
    from compose_v4.experiments.pmo_route_support import KIND, load_contract

    _validate_remote_revision(task["image_revision"])
    c, _ = load_contract(REMOTE_ROOT)
    results = list(
        worker.map([{**task, "slot": i} for i in range(len(c["sources"]))], order_outputs=False)
    )
    results.sort(key=lambda r: r["source"])
    result = {
        "schema_version": "route_support_collection_v1",
        "run_id": task["run_id"],
        "contract_sha256": c["contract_sha256"],
        "code_revision": task["image_revision"]["commit"],
        "oracle_calls": 0,
        "results": results,
    }
    store = Store(ARTIFACT_ROOT / KIND / task["run_id"], artifact_volume.commit)
    store.save("result", result)
    store.flush(force=True)
    return result
