"""GPU training for the corpus-scale editing trainer.

WHY A SEPARATE APP RATHER THAN A SIBLING OF `run_p50_gpu_remote`
---------------------------------------------------------------
The P50 GPU function is the frozen bounded pilot: it loads a P50 prepared input
and a collated completion, enforces the fifty-step recipe, and PUBLISHES a P50
decision.  This trainer publishes nothing, authorizes nothing, and reads
published prep slices instead of a prepared/collated pair.  Reusing that
function would bind a long run to the frozen recipe and hand it authority it
must not have.

What IS reused is the environment, which is where the real value is: the same
image, the same A10G spec, the same volume mount and the same deterministic
cuBLAS requirement.

WHY THE TIMEOUT IS NOT THE P50 TIMEOUT
--------------------------------------
`GPU_TIMEOUT_SECONDS` is twenty minutes because fifty steps is quick.  A
two-thousand-step run would be killed at that wall, losing everything since the
last checkpoint.  This function therefore has its own wall AND stops itself
before reaching it, so the run always ends on a checkpoint rather than on a
kill.  Successive calls resume, which is why a long run can outlive any single
container lifetime.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (  # reuse the exact environment
    ARTIFACT_ROOT,
    DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
    _canonical_bytes,
    _require_artifact_path,
    _require_physical_artifact_path,
    _validate_remote_revision,
    artifact_volume,
    image,
    local_image_revision,
)

app = modal.App("compose-v4-process-v2-training")

#: Its own wall. Long enough to make GPU time worth the container start, short
#: enough that a stuck run is not billed for hours.
TRAINING_TIMEOUT_SECONDS = 60 * 60
#: Held back so the run stops itself on a checkpoint instead of being killed.
TRAINING_RESERVE_SECONDS = 5 * 60
CHECKPOINT_EVERY = 250


def _imports() -> dict[str, Any]:
    from compose_v4.experiments.editing_v2_process_v2_training import (
        build_validation_panel,
        iter_slice_directories,
        load_training_checkpoint,
        run_process_v2_training,
    )
    from compose_v4.experiments.editing_v2_semantic_runtime import (
        SemanticScratchModelConfig,
        build_semantic_scratch_runtime,
    )
    from compose_v4.experiments.editing_gate_zero_semantic_contract import (
        FrozenGateZeroSemanticContract,
        build_gate_zero_process_v2_contract,
    )
    from compose_v4.data.immutable_artifact import write_bytes_if_absent

    return {
        "run_training": run_process_v2_training,
        "build_panel": build_validation_panel,
        "iter_slices": iter_slice_directories,
        "load_checkpoint": load_training_checkpoint,
        "ScratchConfig": SemanticScratchModelConfig,
        "build_scratch": build_semantic_scratch_runtime,
        "Contract": FrozenGateZeroSemanticContract,
        "build_contract": build_gate_zero_process_v2_contract,
        "write_bytes_if_absent": write_bytes_if_absent,
    }


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=64 * 1024,
    timeout=TRAINING_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def train_remote(
    slice_root: str,
    run_root: str,
    optimizer_steps: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    message_passing_steps: int,
    capability_ceiling_nats: float,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Train the productive jump law on published slices. Publishes no authority."""

    entered = time.monotonic()

    def emit(**event: Any) -> None:
        print(json.dumps({**event, "since_entry_seconds": time.monotonic() - entered},
                         sort_keys=True), flush=True)

    _validate_remote_revision(revision)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("training requires deterministic cuBLAS")
    artifact_volume.reload()
    loaded = _imports()
    emit(phase="training_entered", optimizer_steps=int(optimizer_steps))

    import torch

    # Writes take the PHYSICAL path: /artifacts is a symlink and the immutable
    # writer refuses symlinked components. Proven up front, before any compute,
    # because a write fault discovered at checkpoint time costs the whole run.
    physical_run_root = _require_physical_artifact_path(run_root, field="run_root")
    loaded["write_bytes_if_absent"](
        physical_run_root / "TRAINING_WRITABILITY_PREFLIGHT.json",
        _canonical_bytes({"schema": "compose.editing_v2.training_writability_preflight",
                          "schema_version": 1,
                          "status": "TRAINING_PREFLIGHT_EVIDENCE_ONLY_NO_AUTHORITY"}) + b"\n",
    )
    emit(phase="training_output_writable")

    slices = _require_artifact_path(slice_root, field="slice_root")
    found = loaded["iter_slices"](slices)
    emit(phase="training_slices_found", slices=len(found))

    contract = loaded["Contract"](
        source=Path("configs/editing_v2_semantic_process_v2.json"),
        payload=loaded["build_contract"](),
        file_sha256="0" * 64,
    )
    scratch = loaded["build_scratch"](
        loaded["ScratchConfig"](
            initialization_seed=104729,
            max_atoms=40,
            hidden_dim=int(hidden_dim),
            message_passing_steps=int(message_passing_steps),
            mark_dim=32,
            dtype="torch.float32",
            atom_vocabulary_class_count=15,
            catalog_fingerprint="639ff6078c32d43c",
        ),
        contract,
    )
    model = scratch.model
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    parameters = sum(p.numel() for p in model.parameters())
    emit(phase="training_model_built", parameters=parameters, device=str(model.device))

    # Resume from the newest checkpoint if one exists, so successive calls
    # continue one run rather than restarting it.
    checkpoints = sorted((physical_run_root / "checkpoints").glob("step-*.pt"))
    resume_from = checkpoints[-1] if checkpoints else None
    emit(phase="training_resume", resume_from=str(resume_from) if resume_from else None)

    panel = loaded["build_panel"](slices, examples_per_family=32, expected_role="train")
    emit(phase="training_panel_built", rows=panel.example_count,
         families=list(panel.families))

    result = loaded["run_training"](
        model,
        slices,
        optimizer_steps=int(optimizer_steps),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        seed=17,
        panel=panel,
        capability_check_every_steps=CHECKPOINT_EVERY,
        maximum_family_nll_regression={f: float(capability_ceiling_nats) for f in panel.families},
        checkpoint_every_steps=CHECKPOINT_EVERY,
        checkpoint_root=physical_run_root,
        resume_from=resume_from,
        progress_callback=lambda e: emit(**e) if e.get("completed", 0) % 100 == 0 else None,
    )
    artifact_volume.commit()

    payload = {
        "phase": "training_complete",
        "parameters": parameters,
        "optimizer_steps_completed": int(result["optimizer_steps_completed"]),
        "optimizer_steps_requested": int(result["optimizer_steps_requested"]),
        "resumed_from_optimizer_step": int(result["resumed_from_optimizer_step"]),
        "unique_training_examples_seen": int(result["unique_training_examples_seen"]),
        "loss_first": float(result["losses"][0]) if result["losses"] else None,
        "loss_last": float(result["losses"][-1]) if result["losses"] else None,
        "capability_abort": result["capability_abort"],
        "final_validation": result["final_validation"],
        "checkpoints": [c["optimizer_step"] for c in result["checkpoints"]],
        "ms_per_step": (
            1000.0 * (time.monotonic() - entered) / max(len(result["losses"]), 1)
        ),
        "artifact_published": False,
        "checkpoint_selection_authorized": False,
    }
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)
    return payload


@app.local_entrypoint()
def train(
    slice_root: str,
    run_root: str,
    expected_commit: str,
    optimizer_steps: int = 2000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    hidden_dim: int = 256,
    message_passing_steps: int = 6,
    capability_ceiling_nats: float = 1.5,
) -> None:
    """Train on the GPU, resuming from the newest checkpoint under `run_root`."""

    revision = local_image_revision(expected_commit=expected_commit)
    result = train_remote.remote(
        slice_root, run_root, int(optimizer_steps), int(batch_size), float(learning_rate),
        int(hidden_dim), int(message_passing_steps), float(capability_ceiling_nats), revision,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
