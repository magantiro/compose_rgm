"""Two durable CPU cases for the authorized 100-query archive pilot."""

import modal

from modal_apps.genmol_t4_opt_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, _runtime, artifact_volume
from modal_apps.pmo_macro_probe_app import image as base_image
from modal_apps.run_process_v2_p50_app import _validate_remote_revision

image = base_image.add_local_file(
    ROOT / "modal_apps/pmo_archive_pilot_app.py",
    str(REMOTE_ROOT / "modal_apps/pmo_archive_pilot_app.py"),
    copy=True,
)
app = modal.App("compose-pmo-archive-pilot")


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=8192,
    timeout=4500,
    retries=0,
    max_containers=2,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def archive_case(task):
    from compose_v4.experiments.pmo_archive_pilot import run_remote

    return run_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _runtime, _validate_remote_revision
    )
