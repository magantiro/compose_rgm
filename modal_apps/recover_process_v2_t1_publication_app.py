"""Launch only the authenticated Process-V2 T1 publication recovery."""

from __future__ import annotations

from modal_apps import run_process_v2_t1_app as t1_app


@t1_app.app.local_entrypoint()
def main(
    prepared_completion_path: str,
    step_500_checkpoint_path: str,
    expected_step_500_file_sha256: str,
    selected_checkpoint_path: str,
    expected_selected_checkpoint_file_sha256: str,
    expected_commit: str,
    output_prefix: str = t1_app.PUBLICATION_RECOVERY_OUTPUT_PREFIX,
) -> None:
    """Spawn one disconnect-safe, zero-update publication recovery."""

    revision = t1_app.local_image_revision(expected_commit=expected_commit)
    call = t1_app.recover_t1_publication_remote.spawn(
        prepared_completion_path,
        step_500_checkpoint_path,
        expected_step_500_file_sha256,
        selected_checkpoint_path,
        expected_selected_checkpoint_file_sha256,
        output_prefix,
        revision,
    )
    print(
        {
            "phase": "process_v2_t1_publication_recovery_spawned",
            "function_call_id": call.object_id,
            "image_revision_sha256": revision["image_revision_sha256"],
            "optimizer_updates_planned": 0,
            "p50_launched": False,
        },
        flush=True,
    )
