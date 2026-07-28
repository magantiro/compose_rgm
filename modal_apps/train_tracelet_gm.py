"""Modal launch surface for the frozen carbon-tree Generator Matching stage.

The recipe remains the single source of truth.  The remote entrypoint changes
only artifact paths (to the persistent Modal volume) and, for ``--smoke``, a
small explicit set of resource-saving preflight values.

Run a commit-labelled remote preflight:

    modal run modal_apps/train_tracelet_gm.py \
      --preflight --run-label compose-v4-stage3-a100-preflight-<commit>-v1

Launch the quality run so it survives local disconnects:

    modal run --detach modal_apps/train_tracelet_gm.py \
      --run-label compose-v4-stage3-production-<commit>-v1
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")
RECIPE_NAME = "tree_fcd_transfer_stage3_flexible_graft.json"
INTEGRATION_SMOKE_RECIPE_NAME = (
    "tree_fcd_transfer_stage3_integration_smoke.json"
)
FINAL_ROLLOUT_SAMPLES = 2000
INTEGRATION_SMOKE_ROLLOUT_SAMPLES = 16
ROLLOUT_BASE_SEED = 20260717
ROLLOUT_MAX_ATOMS = 40
ROLLOUT_OPERATIONAL_HORIZON = 16.0
ROLLOUT_TIME_STEP = 0.1
ROLLOUT_MAX_EVENTS = 128
TRAINING_SUPPORT_CACHE_DIR = "/artifacts/_shared/training_support"
TRAINING_SUPPORT_SHARD_SIZE = 16000
TRAINING_SUPPORT_BATCH_SIZE = 64
TRAINING_SUPPORT_SHARD_STEPS = (
    TRAINING_SUPPORT_SHARD_SIZE // TRAINING_SUPPORT_BATCH_SIZE
)
TRAINING_SUPPORT_MINIMUM_ROWS_PER_SECOND = 64.0
SOURCE_TREE_IGNORE = ("**/__pycache__/**", "**/*.pyc")
SOURCE_FINGERPRINT_SUFFIXES = frozenset({".py", ".json"})
RUN_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

# Modal executes the entrypoint module from /root while the snapshotted package
# lives under REMOTE_ROOT/src.  Install that path before any remote helper
# imports compose_v4; the subprocess receives the same path through PYTHONPATH.
if REMOTE_ROOT.is_dir():
    sys.path.insert(0, str(REMOTE_ROOT / "src"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
        "fcd-torch==1.0.7",
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    .add_local_dir(
        ROOT / "src",
        str(REMOTE_ROOT / "src"),
        copy=True,
        ignore=SOURCE_TREE_IGNORE,
    )
    .add_local_dir(
        ROOT / "scripts",
        str(REMOTE_ROOT / "scripts"),
        copy=True,
        ignore=SOURCE_TREE_IGNORE,
    )
    .add_local_dir(
        ROOT / "recipes",
        str(REMOTE_ROOT / "recipes"),
        copy=True,
        ignore=SOURCE_TREE_IGNORE,
    )
)

app = modal.App("compose-v4-tracelet-gm")
guacamol_volume = modal.Volume.from_name("guacamol", create_if_missing=False)
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts",
    create_if_missing=True,
)


def _materialize_remote_recipe(
    *,
    run_label: str,
    smoke: bool,
    preflight: bool = False,
    recipe_name: str = RECIPE_NAME,
) -> tuple[dict, Path]:
    from compose_v4.experiments.recipe import load_tracelet_recipe

    if Path(recipe_name).name != recipe_name or not recipe_name.endswith(".json"):
        raise ValueError("recipe_name must be a JSON basename under recipes/")
    recipe_path = REMOTE_ROOT / "recipes" / recipe_name
    if not recipe_path.is_file():
        visible = sorted(path.name for path in (REMOTE_ROOT / "recipes").glob("*.json"))
        raise FileNotFoundError(f"unknown recipe {recipe_name!r}; available recipes={visible}")
    recipe = load_tracelet_recipe(recipe_path)
    recipe = json.loads(json.dumps(recipe))
    arguments = recipe["arguments"]
    run_dir = Path("/artifacts") / run_label
    arguments.update(
        {
            "output": str(run_dir / "metrics.json"),
            "checkpoint": str(run_dir / "checkpoint.pt"),
            "path_cache": str(run_dir / "compiled_paths.pt"),
            "rollout_cache": str(run_dir / "rollouts.pt"),
            "evaluation_cache_dir": "/artifacts/_shared/evaluation_batches",
            "training_support_cache_dir": TRAINING_SUPPORT_CACHE_DIR,
            "training_support_shard_size": TRAINING_SUPPORT_SHARD_SIZE,
        }
    )
    if smoke and preflight:
        raise ValueError("smoke and preflight profiles are mutually exclusive")
    if smoke:
        arguments.update(
            {
                "max_atoms": 12,
                "train_size": 32,
                "validation_size": 4,
                "test_size": 4,
                "fast_split": True,
                "steps": 2,
                "recovery_every": 1,
                "warmup_steps": 1,
                "minimum_learning_rate_fraction": 0.5,
                "evaluation_every": 1,
                "early_stopping_patience": 0,
                "batch_size": 2,
                "hidden_dim": 16,
                "message_passing_steps": 1,
                "tree_couplings_per_target": 2,
                "validation_examples": 8,
                "test_examples": 8,
                "operational_horizon": 0.2,
                "max_events": 2,
                "rollout_samples": 2,
                "rollout_workers": 1,
                "fiber_workers": 1,
                "data_workers": 0,
                "data_prefetch_factor": 2,
                "path_workers": 0,
                "corpus_workers": 0,
                "quality_metrics": False,
                "include_fcd": False,
            }
        )
    elif preflight:
        arguments.update(
            {
                "train_size": 256,
                "validation_size": 64,
                "test_size": 64,
                "fast_split": True,
                "steps": 200,
                "recovery_every": 50,
                "warmup_steps": 20,
                "minimum_learning_rate_fraction": 0.1,
                "evaluation_every": 50,
                "early_stopping_patience": 4,
                "validation_examples": 128,
                "test_examples": 256,
                "operational_horizon": 16.0,
                "max_events": 128,
                "rollout_samples": 16,
                "rollout_workers": 8,
                "fiber_workers": 12,
                "data_workers": 24,
                "data_prefetch_factor": 2,
                "path_workers": 16,
                "corpus_workers": 0,
                "quality_metrics": False,
                "include_fcd": False,
            }
        )
    return recipe, run_dir


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    entrypoint = Path(__file__).resolve()
    digest.update(b"modal_apps/train_tracelet_gm.py")
    digest.update(entrypoint.read_bytes())
    for directory_name in ("src", "scripts", "recipes"):
        directory = REMOTE_ROOT / directory_name
        for path in sorted(
            item
            for item in directory.rglob("*")
            if item.is_file() and item.suffix in SOURCE_FINGERPRINT_SUFFIXES
        ):
            digest.update(str(path.relative_to(REMOTE_ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_run_label(value: str, *, field: str = "run label") -> None:
    """Reject traversal, nested paths, and ambiguous artifact labels."""

    if Path(value).name != value or RUN_LABEL_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field} must be a 1-128 character ASCII basename containing only "
            "letters, digits, '.', '_', or '-', and must begin with a letter or digit"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _early_rollout_decision(
    checkpoint_payload: dict[str, object],
    *,
    completed_steps: int,
    warmup_steps: int,
    minimum_relative_improvement: float = 0.65,
    minimum_family_accuracy: float = 0.75,
    minimum_selected_step: int = 1000,
) -> dict[str, float] | None:
    """Return auditable launch metadata for a sufficiently trained checkpoint."""

    if completed_steps < warmup_steps:
        return None
    initial = checkpoint_payload.get("initial_validation")
    selected = checkpoint_payload.get("selected_validation")
    if not isinstance(initial, dict) or not isinstance(selected, dict):
        return None
    try:
        initial_loss = float(initial["factorized_gm_loss"])
        selected_loss = float(selected["factorized_gm_loss"])
        selected_step = float(selected["selected_step"])
        family_accuracy = float(selected["family_accuracy"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (
        math.isfinite(initial_loss)
        and math.isfinite(selected_loss)
        and initial_loss > 0.0
        and 0.0 <= minimum_relative_improvement < 1.0
        and math.isfinite(family_accuracy)
        and 0.0 <= minimum_family_accuracy <= 1.0
    ):
        return None
    relative_improvement = (initial_loss - selected_loss) / initial_loss
    if (
        relative_improvement < minimum_relative_improvement
        or family_accuracy < minimum_family_accuracy
        or selected_step < minimum_selected_step
        or selected_step > completed_steps
    ):
        return None
    return {
        "completed_steps": float(completed_steps),
        "selected_step": selected_step,
        "initial_validation_loss": initial_loss,
        "selected_validation_loss": selected_loss,
        "selected_family_accuracy": family_accuracy,
        "relative_improvement": relative_improvement,
        "minimum_relative_improvement": float(minimum_relative_improvement),
        "minimum_family_accuracy": float(minimum_family_accuracy),
        "minimum_selected_step": float(minimum_selected_step),
    }


def _atomic_json_write(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _early_launch_marker_blocks_retry(marker_path: Path) -> bool:
    """Return whether a prior launch is active, complete, or not yet stale."""

    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(marker, dict):
        return False
    status = str(marker.get("status", ""))
    if status == "failed":
        return False
    if status == "completed":
        return True
    evaluation_run_label = marker.get("evaluation_run_label")
    if isinstance(evaluation_run_label, str):
        evaluation_dir = Path("/artifacts") / evaluation_run_label
        if (evaluation_dir / "metrics.json").is_file():
            return True
    updated_at = float(marker.get("updated_at_unix", 0.0))
    return time.time() - updated_at < 15.0 * 60.0


def _update_source_early_eval_marker(
    *,
    source_run_label: str,
    evaluation_run_label: str,
    status: str,
    details: dict[str, object] | None = None,
) -> None:
    """Publish evaluator liveness back to the training run's marker."""

    marker_path = Path("/artifacts") / source_run_label / "early_eval_launch.json"
    if not marker_path.is_file():
        return
    try:
        marker = json.loads(marker_path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    if (
        not isinstance(marker, dict)
        or marker.get("evaluation_run_label") != evaluation_run_label
    ):
        return
    marker.update(
        {
            "status": status,
            "updated_at_unix": time.time(),
            **({} if details is None else details),
        }
    )
    _atomic_json_write(marker, marker_path)
    artifact_volume.commit()


def _maybe_launch_early_rollout(
    *,
    run_label: str,
    run_dir: Path,
    recipe: dict[str, object],
    completed_steps: int,
) -> None:
    """Spawn one immutable 100-sample CPU preview while training continues."""

    marker_path = run_dir / "early_eval_launch.json"
    if _early_launch_marker_blocks_retry(marker_path):
        return
    marker_path.unlink(missing_ok=True)
    arguments = recipe.get("arguments")
    if not isinstance(arguments, dict):
        return
    warmup_steps = int(arguments.get("warmup_steps", 0))
    preview_gate = recipe.get("early_preview", {})
    if not isinstance(preview_gate, dict):
        raise ValueError("recipe early_preview must be an object")
    checkpoint_path = run_dir / "checkpoint.best_so_far.pt"
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        return

    import torch

    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint_payload, dict):
        return
    decision = _early_rollout_decision(
        checkpoint_payload,
        completed_steps=completed_steps,
        warmup_steps=warmup_steps,
        minimum_relative_improvement=float(
            preview_gate.get("minimum_relative_improvement", 0.65)
        ),
        minimum_family_accuracy=float(
            preview_gate.get("minimum_family_accuracy", 0.75)
        ),
        minimum_selected_step=int(preview_gate.get("minimum_selected_step", 1000)),
    )
    if decision is None:
        print(
            json.dumps(
                {
                    "phase": "early_rollout_pending",
                    "completed_steps": completed_steps,
                    "warmup_steps": warmup_steps,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    selected_step = int(decision["selected_step"])
    evaluation_run_label = f"{run_label}-early-step{selected_step}-eval100"
    immutable_checkpoint = run_dir / f"checkpoint.early_step{selected_step}.pt"
    temporary_checkpoint = immutable_checkpoint.with_suffix(".pt.tmp")
    shutil.copyfile(checkpoint_path, temporary_checkpoint)
    os.replace(temporary_checkpoint, immutable_checkpoint)
    marker: dict[str, object] = {
        "status": "reserved",
        "source_run_label": run_label,
        "source_checkpoint": immutable_checkpoint.name,
        "source_checkpoint_sha256": _file_sha256(immutable_checkpoint),
        "evaluation_run_label": evaluation_run_label,
        "rollout_samples": 100,
        "updated_at_unix": time.time(),
        **decision,
    }
    _atomic_json_write(marker, marker_path)
    artifact_volume.commit()
    try:
        call = rollout_evaluate_stage.spawn(
            evaluation_run_label,
            run_label,
            immutable_checkpoint.name,
            100,
        )
    except Exception as error:
        marker_path.unlink(missing_ok=True)
        artifact_volume.commit()
        print(
            json.dumps(
                {
                    "phase": "early_rollout_launch_failed",
                    "completed_steps": completed_steps,
                    "error": repr(error),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    marker.update(
        {
            "status": "spawned",
            "function_call_id": call.object_id,
            "updated_at_unix": time.time(),
        }
    )
    _atomic_json_write(marker, marker_path)
    artifact_volume.commit()
    print(json.dumps({"phase": "early_rollout_spawned", **marker}, sort_keys=True), flush=True)


def _final_rollout_spec(
    source_run_label: str,
    train_result: dict[str, object],
    *,
    rollout_samples: int = FINAL_ROLLOUT_SAMPLES,
) -> tuple[str, str, int]:
    """Name the immutable post-training CPU evaluation from selected state."""

    _validate_run_label(source_run_label, field="source run label")
    selected = train_result.get("selected_validation")
    if not isinstance(selected, dict):
        raise ValueError("training result lacks selected_validation")
    try:
        selected_step_value = float(selected["selected_step"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("training result lacks a valid selected_step") from error
    if (
        not math.isfinite(selected_step_value)
        or selected_step_value < 0.0
        or not selected_step_value.is_integer()
    ):
        raise ValueError("selected_step must be a finite non-negative integer")
    if rollout_samples <= 0:
        raise ValueError("rollout_samples must be positive")
    selected_step = int(selected_step_value)
    return (
        f"{source_run_label}-final-step{selected_step}-eval{rollout_samples}",
        "checkpoint.pt",
        rollout_samples,
    )


def _rollout_evaluation_profile(name: str) -> dict[str, object]:
    """Return one frozen CPU-evaluation profile for artifact identity."""

    if name == "production_v1":
        return {
            "name": name,
            "seed": ROLLOUT_BASE_SEED,
            "max_atoms": ROLLOUT_MAX_ATOMS,
            "train_size": 50000,
            "validation_size": 2000,
            "test_size": 2000,
            "fast_split": False,
            "corpus_workers": 16,
            "rollout_workers": 16,
            "torch_threads": 1,
            "operational_horizon": ROLLOUT_OPERATIONAL_HORIZON,
            "time_step": ROLLOUT_TIME_STEP,
            "max_events": ROLLOUT_MAX_EVENTS,
            "quality_reference_limit": 5000,
            "fcd_generated_limit": 2000,
        }
    if name == "integration_smoke_v1":
        return {
            "name": name,
            "seed": ROLLOUT_BASE_SEED,
            "max_atoms": 12,
            "train_size": 32,
            "validation_size": 4,
            "test_size": 4,
            "fast_split": True,
            "corpus_workers": 1,
            "rollout_workers": 4,
            "torch_threads": 1,
            "operational_horizon": 0.2,
            "time_step": 0.1,
            "max_events": 2,
            "quality_reference_limit": 128,
            "fcd_generated_limit": INTEGRATION_SMOKE_ROLLOUT_SAMPLES,
        }
    raise ValueError(f"unknown rollout evaluation profile: {name}")


def _rollout_manifest_signature(
    *,
    source_run_label: str,
    checkpoint_name: str,
    checkpoint_sha256: str,
    rollout_samples: int,
    disabled_rule_names: tuple[str, ...],
    train_sha256: str,
    reference_sha256: str,
    source_sha256: str,
    evaluation_profile: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return the fields that must match before rollout artifacts are reused."""

    profile = dict(
        _rollout_evaluation_profile("production_v1")
        if evaluation_profile is None
        else evaluation_profile
    )
    return {
        "version": 1,
        "source_run_label": source_run_label,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "rollout_samples": rollout_samples,
        "sampling_seed": int(profile["seed"]) + 4,
        "max_atoms": profile["max_atoms"],
        "operational_horizon": profile["operational_horizon"],
        "time_step": profile["time_step"],
        "max_events": profile["max_events"],
        "evaluation_profile": profile,
        "disabled_rule_names": list(disabled_rule_names),
        "source_prior": "carbon_tree",
        "train_sha256": train_sha256,
        "reference_sha256": reference_sha256,
        "source_sha256": source_sha256,
    }


def _rollout_remote_summary(
    *,
    run_label: str,
    run_dir: Path,
    report: dict[str, object],
) -> dict[str, object]:
    rollout = report.get("rollout")
    if not isinstance(rollout, dict):
        rollout = {}
    molecular_quality = report.get("molecular_quality")
    if not isinstance(molecular_quality, dict):
        molecular_quality = {}
    return {
        "run_label": run_label,
        "artifact_dir": str(run_dir),
        "generated_nonnull_smiles": report.get("generated_nonnull_smiles"),
        "valid_fraction": rollout.get("valid_fraction"),
        "trajectory_diagnostics_available": report.get(
            "trajectory_diagnostics_available"
        ),
        "selected_validation": report.get("selected_validation"),
        "fcd": molecular_quality.get("frechet_chemnet_distance"),
    }


def _stable_json_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _partition_training_support_steps(
    *,
    start_step: int,
    steps: int,
    containers: int,
) -> tuple[tuple[int, int], ...]:
    """Partition complete cache shards into deterministic contiguous ranges."""

    if start_step < 0 or steps <= 0 or containers <= 0:
        raise ValueError("support range and container count must be positive")
    if (
        start_step % TRAINING_SUPPORT_SHARD_STEPS
        or steps % TRAINING_SUPPORT_SHARD_STEPS
    ):
        raise ValueError(
            "support start and length must align to complete cache shards"
        )
    shard_count = steps // TRAINING_SUPPORT_SHARD_STEPS
    if containers > shard_count:
        raise ValueError("support containers cannot exceed complete cache shards")
    base, remainder = divmod(shard_count, containers)
    ranges = []
    cursor = int(start_step)
    for index in range(containers):
        range_shards = base + int(index < remainder)
        range_steps = range_shards * TRAINING_SUPPORT_SHARD_STEPS
        ranges.append((cursor, range_steps))
        cursor += range_steps
    if cursor != start_step + steps:
        raise RuntimeError("support range partitioning lost deterministic coverage")
    return tuple(ranges)


def _remote_stage_kind(
    *,
    smoke: bool,
    preflight: bool,
    compile_paths_only: bool,
    compile_training_support_steps: int = 0,
    evaluation_source_run: str | None,
) -> str:
    if evaluation_source_run is not None:
        return "checkpoint_evaluation"
    if compile_paths_only:
        return "compile"
    if compile_training_support_steps:
        return "support_compile"
    if smoke:
        return "smoke"
    if preflight:
        return "preflight"
    return "training"


def _run_remote(
    *,
    run_label: str,
    smoke: bool,
    preflight: bool = False,
    compile_paths_only: bool = False,
    compile_training_support_start_step: int = 0,
    compile_training_support_steps: int = 0,
    require_path_cache: bool = False,
    require_training_support_cache: bool = False,
    skip_rollouts: bool = False,
    recipe_name: str = RECIPE_NAME,
    evaluation_source_run: str | None = None,
    evaluation_checkpoint_name: str = "checkpoint.best_step3000.pt",
    path_cache_source_run: str | None = None,
    resume_source_run: str | None = None,
    initialization_source_run: str | None = None,
    initialization_checkpoint_name: str = "checkpoint.best_so_far.pt",
    compatible_initialization: bool = False,
    training_steps: int | None = None,
    schedule_steps: int | None = None,
    disable_early_stopping: bool = False,
    snapshot_checkpoints: bool = False,
    training_support_workers: int = 12,
    training_support_microbatch_size: int = 4,
    training_support_prefetch_factor: int = 2,
    training_support_wait_seconds: float = 600.0,
    minimum_training_support_rows_per_second: float = 0.0,
    corrupted_prior_mix: bool = False,
    cycle_op_mix: bool = False,
    disable_ring_grow_macro: bool = False,
    dry_launch: bool = False,
    dry_launch_output: str | None = None,
    organic_vocabulary: bool = False,
    analogue_trace_pool: str | None = None,
    analogue_trace_count: int = 0,
    corrupted_prior_count: int | None = None,
    scaled_manifest: str | None = None,
) -> dict[str, object]:
    from compose_v4.experiments.recipe import build_tracelet_recipe_argv

    _validate_run_label(run_label)
    guacamol_volume.reload()
    artifact_volume.reload()
    started = time.perf_counter()
    recipe, run_dir = _materialize_remote_recipe(
        run_label=run_label,
        smoke=smoke,
        preflight=preflight,
        recipe_name=recipe_name,
    )
    if compile_training_support_start_step < 0:
        raise ValueError("compile-training-support start step must be non-negative")
    if compile_training_support_steps < 0:
        raise ValueError("compile-training-support steps must be non-negative")
    if compile_training_support_start_step and not compile_training_support_steps:
        raise ValueError("support start step requires a non-empty compile range")
    if training_steps is not None and training_steps <= 0:
        raise ValueError("training steps must be positive")
    if schedule_steps is not None and schedule_steps <= 0:
        raise ValueError("schedule steps must be positive")
    if min(
        training_support_workers,
        training_support_microbatch_size,
        training_support_prefetch_factor,
    ) <= 0:
        raise ValueError("training-support compiler settings must be positive")
    if training_support_wait_seconds < 0.0:
        raise ValueError("training-support wait must be non-negative")
    if minimum_training_support_rows_per_second < 0.0:
        raise ValueError("training-support throughput gate must be non-negative")
    if compile_training_support_steps and require_training_support_cache:
        raise ValueError("support compilation and support consumption are exclusive")
    if compile_training_support_steps and (
        smoke or preflight or compile_paths_only or evaluation_source_run is not None
    ):
        raise ValueError("support compilation is a dedicated CPU stage")
    if training_steps is not None:
        recipe["arguments"]["steps"] = int(training_steps)
    if schedule_steps is not None:
        recipe["arguments"]["schedule_steps"] = int(schedule_steps)
    if disable_early_stopping:
        # Run the full declared horizon: patience 0 disables performance-based early stopping, so the run
        # terminates only on numerical/correctness failure and a post-hoc selection rule scores every
        # snapshot. Required when the training objective is NOT the checkpoint-selection metric.
        recipe["arguments"]["early_stopping_patience"] = 0
    if snapshot_checkpoints:
        recipe["arguments"]["snapshot_checkpoints"] = True
    if corrupted_prior_mix:
        # Emit --corrupted-prior-mix to the gate (fine-tune B -> B-edit): mixed corrupted-prior records
        # + the ring_system_restate / cyclic bond_reroute editing marks. Weights preserved by warm-start.
        recipe["arguments"]["corrupted_prior_mix"] = True
    if cycle_op_mix:
        # Emit --cycle-op-mix: compositional cycle_close/cycle_open ring families + real-molecule ring-bond
        # (cycle_open/cycle_close) supervision -- support-complete ring growth/closure.
        recipe["arguments"]["cycle_op_mix"] = True
    if disable_ring_grow_macro:
        # Emit --disable-ring-grow-macro (RING_CORE_V1): disable the legacy whole-ring grow macro so ring
        # addition is purely compositional. Grow head params retained (warm-start-safe), family masked dead.
        recipe["arguments"]["disable_ring_grow_macro"] = True
    if dry_launch:
        # CPU dry-launch: the gate runs the real trainer to the first editing batch + one validation forward,
        # then exits before backward/optimizer (proves the zero-mixture path; optimizer_steps=0).
        recipe["arguments"]["dry_launch"] = True
        if dry_launch_output:
            recipe["arguments"]["dry_launch_output"] = dry_launch_output
    if organic_vocabulary:
        # Emit --organic-vocabulary: predict/edit the whole drug-like organic subset (15 (element,
        # valence) classes over C/N/O/F/S/P/Cl/Br/I/B) instead of CNOF-only. Warm-start B compatibly.
        recipe["arguments"]["organic_vocabulary"] = True
    if analogue_trace_pool:
        # Layer-2 MMP pool (the mined, executor-verified real-molecule edit traces) appended in memory.
        recipe["arguments"]["analogue_trace_pool"] = analogue_trace_pool
        if analogue_trace_count:
            recipe["arguments"]["analogue_trace_count"] = int(analogue_trace_count)
    if corrupted_prior_count:
        recipe["arguments"]["corrupted_prior_count"] = int(corrupted_prior_count)
    if scaled_manifest:
        # Drive the hierarchical training sampler from the locked scaled manifest (mixture + curriculum).
        recipe["arguments"]["scaled_manifest"] = scaled_manifest
    if path_cache_source_run is not None:
        _validate_run_label(path_cache_source_run, field="path-cache source run")
        source_path_cache = (
            Path("/artifacts") / path_cache_source_run / "compiled_paths.pt"
        )
        source_manifest = Path(f"{source_path_cache}.shards") / "manifest.pt"
        has_single_cache = (
            source_path_cache.is_file() and source_path_cache.stat().st_size > 0
        )
        has_sharded_cache = (
            source_manifest.is_file() and source_manifest.stat().st_size > 0
        )
        if not (has_single_cache or has_sharded_cache):
            raise FileNotFoundError(
                "path-cache source is missing both cache formats: "
                f"{path_cache_source_run}"
            )
        recipe["arguments"]["path_cache"] = str(source_path_cache)
        require_path_cache = True
    if resume_source_run is not None:
        if smoke or preflight or compile_paths_only or compile_training_support_steps:
            raise ValueError("cross-run recovery is reserved for training")
        if evaluation_source_run is not None:
            raise ValueError("cross-run recovery and checkpoint evaluation are exclusive")
        _validate_run_label(resume_source_run, field="resume source run")
        source_recovery = (
            Path("/artifacts") / resume_source_run / "checkpoint.recovery.pt"
        )
        if not source_recovery.is_file() or source_recovery.stat().st_size == 0:
            raise FileNotFoundError(
                f"resume source recovery checkpoint is missing: {source_recovery}"
            )
        recipe["arguments"].update(
            {
                "resume_checkpoint": str(source_recovery),
                "allow_resume_provenance_mismatch": True,
                "allow_resume_step_extension": True,
            }
        )
    if initialization_source_run is not None:
        if resume_source_run is not None:
            raise ValueError(
                "fresh checkpoint initialization and recovery resume are exclusive"
            )
        if smoke or preflight or compile_paths_only or compile_training_support_steps:
            raise ValueError("checkpoint initialization is reserved for training")
        if evaluation_source_run is not None:
            raise ValueError(
                "checkpoint initialization and checkpoint evaluation are exclusive"
            )
        _validate_run_label(
            initialization_source_run,
            field="initialization source run",
        )
        if Path(initialization_checkpoint_name).name != initialization_checkpoint_name:
            raise ValueError("initialization checkpoint must be a file basename")
        source_checkpoint = (
            Path("/artifacts")
            / initialization_source_run
            / initialization_checkpoint_name
        )
        if not source_checkpoint.is_file() or source_checkpoint.stat().st_size == 0:
            raise FileNotFoundError(
                f"initialization checkpoint is missing: {source_checkpoint}"
            )
        if compatible_initialization:
            recipe["arguments"]["initialize_compatible_checkpoint"] = str(
                source_checkpoint
            )
        else:
            recipe["arguments"]["initialize_checkpoint"] = str(source_checkpoint)
    elif compatible_initialization:
        raise ValueError(
            "compatible initialization requires an initialization source run"
        )
    if skip_rollouts and (
        smoke
        or preflight
        or compile_paths_only
        or compile_training_support_steps
        or evaluation_source_run is not None
    ):
        raise ValueError(
            "skip_rollouts is reserved for the production training stage"
        )
    if skip_rollouts:
        recipe["arguments"]["skip_rollouts"] = True
    if evaluation_source_run is not None:
        _validate_run_label(evaluation_source_run, field="evaluation source run")
        if Path(evaluation_checkpoint_name).name != evaluation_checkpoint_name:
            raise ValueError("evaluation checkpoint must be a file basename")
        source_dir = Path("/artifacts") / evaluation_source_run
        checkpoint_path = source_dir / evaluation_checkpoint_name
        source_path_cache = source_dir / "compiled_paths.pt"
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"evaluation source artifact is missing or empty: {checkpoint_path}"
            )
        sharded_manifest = Path(f"{source_path_cache}.shards") / "manifest.pt"
        has_single_cache = source_path_cache.is_file() and source_path_cache.stat().st_size > 0
        has_sharded_cache = sharded_manifest.is_file() and sharded_manifest.stat().st_size > 0
        if not (has_single_cache or has_sharded_cache):
            raise FileNotFoundError(
                "evaluation source path cache is missing in both single-file "
                f"and sharded formats: {source_path_cache}"
            )
        arguments = recipe["arguments"]
        arguments.pop("resume_checkpoint", None)
        arguments.update(
            {
                "load_checkpoint": str(checkpoint_path),
                "path_cache": str(source_path_cache),
                "require_path_cache": True,
                "allow_legacy_evaluation_cache": True,
            }
        )
        require_path_cache = True
    if compile_paths_only:
        recipe["arguments"].update(
            {
                "compile_paths_only": True,
                "compile_evaluation_cache": True,
                "evaluation_workers": 56,
                "device": "cpu",
            }
        )
    if compile_training_support_steps:
        if path_cache_source_run is None:
            raise ValueError(
                "support compilation requires an immutable path-cache source run"
            )
        recipe["arguments"].update(
            {
                "compile_training_support_steps": int(
                    compile_training_support_steps
                ),
                "compile_training_support_start_step": int(
                    compile_training_support_start_step
                ),
                "training_support_workers": int(training_support_workers),
                "training_support_microbatch_size": int(
                    training_support_microbatch_size
                ),
                "training_support_prefetch_factor": int(
                    training_support_prefetch_factor
                ),
                "device": "cpu",
            }
        )
    if require_training_support_cache:
        recipe["arguments"].update(
            {
                "require_training_support_cache": True,
                "training_support_wait_seconds": float(
                    training_support_wait_seconds
                ),
            }
        )
    if require_path_cache:
        recipe["arguments"]["require_path_cache"] = True
        # A conditioned pilot reuses the immutable path and training-support
        # caches, but its evaluation tensors have a distinct property
        # signature.  Requiring an unrelated unconditioned cache here causes
        # a late failure after all paths are loaded.  Let the trainer compile
        # and persist the bounded conditioned evaluation batch exactly once.
        if (
            not compile_paths_only
            and not compile_training_support_steps
            and recipe["arguments"].get("property_condition") in {None, "none"}
        ):
            recipe["arguments"]["require_evaluation_cache"] = True
    run_dir.mkdir(parents=True, exist_ok=True)
    # The shared volume's nominal full-training SMILES file is a zero-byte
    # placeholder.  This populated, seeded 500k subset is the corpus used for
    # the first quality-bearing transfer stage and makes the data provenance
    # explicit instead of silently falling back at runtime.
    train_file = Path("/guacamol/guacamol_subset_500000_seed0.smiles")
    reference_file = Path("/guacamol/guacamol_heldout_val_5000_seed0.smiles")
    for required in (train_file, reference_file):
        if not required.is_file() or required.stat().st_size == 0:
            visible = sorted(path.name for path in Path("/guacamol").iterdir())
            raise FileNotFoundError(
                f"required Modal dataset is missing or empty: {required}; "
                f"visible root entries={visible[:30]}"
            )

    source_sha256 = _source_fingerprint()
    data_manifest = {
        "train": {
            "path": str(train_file),
            "bytes": train_file.stat().st_size,
            "sha256": _file_sha256(train_file),
        },
        "fcd_reference": {
            "path": str(reference_file),
            "bytes": reference_file.stat().st_size,
            "sha256": _file_sha256(reference_file),
        },
    }
    stage_kind = _remote_stage_kind(
        smoke=smoke,
        preflight=preflight,
        compile_paths_only=compile_paths_only,
        compile_training_support_steps=compile_training_support_steps,
        evaluation_source_run=evaluation_source_run,
    )
    run_identity = {
        "version": 1,
        "run_label": run_label,
        "stage_kind": stage_kind,
        "recipe_name": recipe_name,
        "recipe": recipe,
        "source_sha256": source_sha256,
        "data": data_manifest,
        "resume_source_run": resume_source_run,
        "initialization_source_run": initialization_source_run,
        "initialization_checkpoint_name": (
            initialization_checkpoint_name
            if initialization_source_run is not None
            else None
        ),
        "compatible_initialization": compatible_initialization,
    }
    run_identity_sha256 = _stable_json_sha256(run_identity)
    stage_manifest_path = run_dir / f"manifest.{stage_kind}.json"
    stage_manifest_existed = stage_manifest_path.is_file()
    if stage_manifest_existed:
        try:
            stage_manifest = json.loads(stage_manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"invalid immutable stage manifest: {stage_manifest_path}") from error
        if (
            not isinstance(stage_manifest, dict)
            or stage_manifest.get("run_identity") != run_identity
            or stage_manifest.get("run_identity_sha256") != run_identity_sha256
        ):
            raise ValueError(
                "run label already exists with a different recipe, source, or dataset identity"
            )
    else:
        existing_entries = tuple(run_dir.iterdir())
        if stage_kind != "training" and existing_entries:
            raise ValueError(
                f"fresh {stage_kind} run requires an empty immutable run label"
            )
        training_artifacts = tuple(
            run_dir / name
            for name in (
                "checkpoint.pt",
                "checkpoint.recovery.pt",
                "checkpoint.best_so_far.pt",
                "metrics.json",
                "early_eval_launch.json",
            )
        )
        if stage_kind == "training" and any(path.exists() for path in training_artifacts):
            raise ValueError(
                "training artifacts exist without a matching immutable training manifest"
            )
        _atomic_json_write(
            {
                "run_identity": run_identity,
                "run_identity_sha256": run_identity_sha256,
                "created_at_unix": time.time(),
            },
            stage_manifest_path,
        )
        artifact_volume.commit()

    recipe["arguments"]["provenance_sha256"] = run_identity_sha256
    recovery_path = run_dir / "checkpoint.recovery.pt"
    if stage_kind == "training" and recovery_path.is_file():
        if recovery_path.stat().st_size == 0:
            raise ValueError("training recovery checkpoint is empty")
        recipe["arguments"]["resume_checkpoint"] = str(recovery_path)
    elif (
        stage_kind == "training"
        and stage_manifest_existed
        and (run_dir / "checkpoint.pt").is_file()
    ):
        raise ValueError(
            "completed checkpoint exists without a recovery state; use a new run label"
        )

    import rdkit
    import torch

    manifest = {
        "run_label": run_label,
        "smoke": smoke,
        "preflight": preflight,
        "compile_paths_only": compile_paths_only,
        "compile_training_support_start_step": (
            compile_training_support_start_step
        ),
        "compile_training_support_steps": compile_training_support_steps,
        "require_path_cache": require_path_cache,
        "require_training_support_cache": require_training_support_cache,
        "skip_rollouts": skip_rollouts,
        "evaluation_source_run": evaluation_source_run,
        "path_cache_source_run": path_cache_source_run,
        "training_steps": training_steps,
        "schedule_steps": schedule_steps,
        "minimum_training_support_rows_per_second": (
            minimum_training_support_rows_per_second
        ),
        "evaluation_checkpoint_name": (
            evaluation_checkpoint_name if evaluation_source_run is not None else None
        ),
        "recipe_name": recipe_name,
        "recipe": recipe,
        "stage_kind": stage_kind,
        "stage_manifest": str(stage_manifest_path),
        "run_identity_sha256": run_identity_sha256,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "rdkit": rdkit.__version__,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "source_sha256": source_sha256,
        "data": data_manifest,
    }
    _atomic_json_write(manifest, run_dir / "manifest.json")
    artifact_volume.commit()

    command = (
        sys.executable,
        str(REMOTE_ROOT / "scripts" / "train_tracelet_cnof_gate.py"),
        *build_tracelet_recipe_argv(
            recipe,
            smiles_file=train_file,
            quality_reference_file=reference_file,
        ),
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(REMOTE_ROOT), str(REMOTE_ROOT / "src")))
    environment["PYTHONUNBUFFERED"] = "1"
    print(json.dumps({"phase": "remote_command", "argv": command}), flush=True)
    auto_early_rollout = (
        not smoke
        and not preflight
        and not compile_paths_only
        and not compile_training_support_steps
        and evaluation_source_run is None
        and int(recipe["arguments"].get("early_stopping_patience", 0)) > 0
    )
    support_ready_event: dict[str, object] | None = None
    support_gate_failure: dict[str, float] | None = None
    process = subprocess.Popen(
        command,
        cwd=REMOTE_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("phase") in {
            "compiled_path_cache_saved",
            "compiled_path_manifest_saved",
            "compiled_proposal_shard_saved",
            "compiled_path_shard_saved",
            "compiled_paths_ready",
            "evaluation_batch_cache_saved",
            "compiled_evaluation_batches_ready",
            "training_support_shard_saved",
            "training_support_cache_ready",
            "recovery_checkpoint_saved",
        }:
            # Recovery frequency is deliberately coarse: committing a Modal
            # volume is much slower than a GPU update and synchronously
            # committing every few hundred steps throttles the child process.
            artifact_volume.commit()
            if event.get("phase") == "recovery_checkpoint_saved" and auto_early_rollout:
                _maybe_launch_early_rollout(
                    run_label=run_label,
                    run_dir=run_dir,
                    recipe=recipe,
                    completed_steps=int(event.get("completed_steps", 0)),
                )
        if isinstance(event, dict) and event.get("phase") == "training_support_cache_ready":
            support_ready_event = dict(event)
        if (
            isinstance(event, dict)
            and event.get("phase") == "training_support_shard_saved"
            and minimum_training_support_rows_per_second > 0.0
        ):
            observed_rate = float(event.get("rows_per_second", 0.0))
            if observed_rate < minimum_training_support_rows_per_second:
                support_gate_failure = {
                    "observed_rows_per_second": observed_rate,
                    "minimum_rows_per_second": float(
                        minimum_training_support_rows_per_second
                    ),
                    "compiled_rows": float(event.get("compiled_rows", 0.0)),
                }
                process.terminate()
                break
    return_code = process.wait()
    artifact_volume.commit()
    if support_gate_failure is not None:
        raise RuntimeError(
            "training-support compiler missed the CPU throughput gate: "
            f"{support_gate_failure}"
        )
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    if compile_paths_only:
        summary = {
            "run_label": run_label,
            "recipe_name": recipe_name,
            "elapsed_seconds": time.perf_counter() - started,
            "artifact_dir": str(run_dir),
            "path_cache": str(run_dir / "compiled_paths.pt"),
        }
        print(
            json.dumps({"phase": "remote_compile_complete", **summary}, sort_keys=True),
            flush=True,
        )
        return summary

    if compile_training_support_steps:
        if support_ready_event is None:
            raise RuntimeError(
                "support compiler exited without a validated cache-ready event"
            )
        summary = {
            "run_label": run_label,
            "recipe_name": recipe_name,
            "elapsed_seconds": time.perf_counter() - started,
            "artifact_dir": str(run_dir),
            "training_support": support_ready_event,
        }
        print(
            json.dumps(
                {"phase": "remote_support_compile_complete", **summary},
                sort_keys=True,
            ),
            flush=True,
        )
        return summary

    if dry_launch:
        # The dry launch exits the trainer after the first forward + validation forward (no full training),
        # so it writes dry_launch.json, not metrics.json. Return its verdict artifact and commit the volume.
        dry_path = run_dir / "dry_launch.json"
        dry_report = json.loads(dry_path.read_text()) if dry_path.is_file() else {}
        artifact_volume.commit()
        summary = {"phase": "dry_launch_stage_complete", "run_label": run_label, **dry_report}
        print(json.dumps(summary, sort_keys=True), flush=True)
        return summary

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"training completed without metrics artifact: {metrics_path}")
    report = json.loads(metrics_path.read_text())
    rollout_report = report.get("rollout")
    if not isinstance(rollout_report, dict):
        rollout_report = {}
    summary = {
        "run_label": run_label,
        "smoke": smoke,
        "preflight": preflight,
        "recipe_name": recipe_name,
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_dir": str(run_dir),
        "model": report.get("model"),
        "generated_nonnull_smiles": report.get("generated_nonnull_smiles"),
        "valid_fraction": rollout_report.get("valid_fraction"),
        "rollouts_skipped": rollout_report.get("skipped") is True,
        "selected_validation": report.get("selected_validation"),
        "fcd": report.get("molecular_quality", {}).get("frechet_chemnet_distance"),
    }
    print(json.dumps({"phase": "remote_complete", **summary}, sort_keys=True), flush=True)
    return summary


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=30 * 60,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
)
def smoke_stage(
    run_label: str,
    recipe_name: str = RECIPE_NAME,
) -> dict[str, object]:
    return _run_remote(run_label=run_label, smoke=True, recipe_name=recipe_name)


@app.function(
    image=image,
    gpu="A100",
    cpu=32.0,
    memory=65536,
    timeout=60 * 60,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
)
def preflight_stage(
    run_label: str,
    recipe_name: str = RECIPE_NAME,
) -> dict[str, object]:
    return _run_remote(
        run_label=run_label,
        smoke=False,
        preflight=True,
        recipe_name=recipe_name,
    )


@app.function(
    image=image,
    gpu="H100",
    cpu=32.0,
    memory=65536,
    timeout=60 * 60,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
)
def h100_preflight_stage(
    run_label: str,
    recipe_name: str = RECIPE_NAME,
) -> dict[str, object]:
    """Resource-matched H100 control for cost-per-update selection."""

    return _run_remote(
        run_label=run_label,
        smoke=False,
        preflight=True,
        recipe_name=recipe_name,
    )


@app.function(
    image=image,
    cpu=64.0,
    memory=131072,
    timeout=24 * 3600,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
def compile_stage(
    run_label: str,
    recipe_name: str = RECIPE_NAME,
    path_cache_source_run: str | None = None,
    organic_vocabulary: bool = False,
) -> dict[str, object]:
    """Compile restart-safe path shards without renting a GPU. organic_vocabulary selects the broad-organic
    corpus split (so the de-novo carbon-tree path cache matches the B-edit training corpus)."""

    return _run_remote(
        run_label=run_label,
        smoke=False,
        compile_paths_only=True,
        recipe_name=recipe_name,
        path_cache_source_run=path_cache_source_run,
        organic_vocabulary=organic_vocabulary,
    )


@app.function(
    image=image,
    cpu=14.0,
    memory=114688,
    timeout=24 * 3600,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
)
def compile_training_support_stage(
    run_label: str,
    path_cache_source_run: str,
    steps: int = 2000,
    recipe_name: str = RECIPE_NAME,
    start_step: int = 0,
    workers: int = 12,
    microbatch_size: int = 4,
    prefetch_factor: int = 2,
    minimum_rows_per_second: float = 0.0,
) -> dict[str, object]:
    """Compile one exact, shard-aligned support range in an isolated container."""

    return _run_remote(
        run_label=run_label,
        smoke=False,
        compile_training_support_start_step=start_step,
        compile_training_support_steps=steps,
        require_path_cache=True,
        recipe_name=recipe_name,
        path_cache_source_run=path_cache_source_run,
        training_support_workers=workers,
        training_support_microbatch_size=microbatch_size,
        training_support_prefetch_factor=prefetch_factor,
        minimum_training_support_rows_per_second=minimum_rows_per_second,
    )


@app.function(
    image=image,
    cpu=16.0,
    memory=65536,
    timeout=2 * 3600,
    volumes={"/artifacts": artifact_volume},
)
def audit_teacher_stage(
    run_label: str,
    recipe_name: str = RECIPE_NAME,
    batch_size: int = 2048,
    workers: int = 12,
) -> dict[str, object]:
    """Reject a finalized cache with any zero-probability validation teacher."""

    from compose_v4.experiments.recipe import load_tracelet_recipe

    _validate_run_label(run_label)
    if batch_size <= 0 or workers < 0:
        raise ValueError("audit batch size must be positive and workers non-negative")
    recipe = load_tracelet_recipe(REMOTE_ROOT / "recipes" / recipe_name)
    ring_electronic_mode = str(
        recipe["arguments"].get("ring_electronic_mode", "factorized_local")
    )
    if ring_electronic_mode not in {"factorized_local", "catalog_exact"}:
        raise ValueError(f"unsupported ring electronic mode: {ring_electronic_mode}")
    artifact_volume.reload()
    cache_root = Path("/artifacts") / run_label / "compiled_paths.pt.shards"
    manifest = cache_root / "manifest.pt"
    shard = cache_root / "transport" / "validation-00000.pt"
    output = Path("/artifacts") / run_label / "teacher_support_audit.json"
    for required in (manifest, shard):
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(f"missing teacher-audit input: {required}")
    command = (
        sys.executable,
        str(REMOTE_ROOT / "scripts" / "audit_factorized_teacher_support.py"),
        str(manifest),
        str(shard),
        "--batch-size",
        str(batch_size),
        "--workers",
        str(workers),
        "--ring-electronic-mode",
        ring_electronic_mode,
        "--output",
        str(output),
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(REMOTE_ROOT), str(REMOTE_ROOT / "src")))
    environment["PYTHONUNBUFFERED"] = "1"
    print(json.dumps({"phase": "remote_audit_command", "argv": command}), flush=True)
    result = subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        env=environment,
        check=False,
    )
    artifact_volume.commit()
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    report = json.loads(output.read_text())
    if report.get("family_failures"):
        raise RuntimeError(f"factorized teacher-support audit failed: {report['family_failures']}")
    summary = {
        "run_label": run_label,
        "examples": report.get("examples"),
        "family_counts": report.get("family_counts"),
        "family_failures": report.get("family_failures"),
        "output": str(output),
    }
    print(json.dumps({"phase": "remote_audit_complete", **summary}), flush=True)
    return summary


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=2 * 3600,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
)
def dry_launch_stage(
    run_label: str,
    recipe_name: str = RECIPE_NAME,
    initialization_source_run: str | None = None,
    initialization_checkpoint_name: str = "checkpoint.best_so_far.pt",
    compatible_initialization: bool = False,
    corrupted_prior_mix: bool = False,
    cycle_op_mix: bool = False,
    disable_ring_grow_macro: bool = False,
    organic_vocabulary: bool = False,
    analogue_trace_pool: str = "",
    analogue_trace_count: int = 0,
    corrupted_prior_count: int = 0,
    scaled_manifest: str = "",
    training_steps: int | None = None,
    schedule_steps: int | None = None,
) -> dict[str, object]:
    # CPU-only dry launch of the REAL trainer (owner mandate §2): zero-mixture, no de-novo cache required,
    # no A100. resolve_torch_device auto-selects CPU (no CUDA on this container).
    return _run_remote(
        run_label=run_label,
        smoke=False,
        require_path_cache=False,
        require_training_support_cache=False,
        skip_rollouts=True,
        recipe_name=recipe_name,
        path_cache_source_run=None,
        dry_launch=True,
        dry_launch_output=f"/artifacts/{run_label}/dry_launch.json",
        training_steps=training_steps,
        schedule_steps=schedule_steps,
        initialization_source_run=initialization_source_run,
        initialization_checkpoint_name=initialization_checkpoint_name,
        compatible_initialization=compatible_initialization,
        corrupted_prior_mix=corrupted_prior_mix,
        cycle_op_mix=cycle_op_mix,
        disable_ring_grow_macro=disable_ring_grow_macro,
        organic_vocabulary=organic_vocabulary,
        analogue_trace_pool=analogue_trace_pool or None,
        analogue_trace_count=analogue_trace_count,
        corrupted_prior_count=corrupted_prior_count or None,
        scaled_manifest=scaled_manifest or None,
    )


@app.function(
    image=image,
    gpu="A100",
    cpu=16.0,
    memory=65536,
    timeout=3600,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
)
def gpu_smoke_stage(
    run_label: str,
    recipe_name: str = RECIPE_NAME,
    initialization_source_run: str | None = None,
    initialization_checkpoint_name: str = "checkpoint.best_so_far.pt",
    compatible_initialization: bool = False,
    corrupted_prior_mix: bool = False,
    cycle_op_mix: bool = False,
    disable_ring_grow_macro: bool = False,
    organic_vocabulary: bool = False,
    analogue_trace_pool: str = "",
    analogue_trace_count: int = 0,
    corrupted_prior_count: int = 0,
    scaled_manifest: str = "",
    training_steps: int | None = None,
    schedule_steps: int | None = None,
) -> dict[str, object]:
    # One-step GPU smoke (Part D): the SAME early-exit dry-launch path as dry_launch_stage but on an A100, so
    # it exercises the CUDA device path the CPU dry-launch cannot -- warm-start (where DEV-CUDA bit), the first
    # training forward, and one edit-validation forward -- then exits before backward/optimizer. Runs with the
    # real schedule-check args (training_steps=1000, schedule_steps=3000) so the launch-identity + scheduler
    # guard fire on GPU exactly as the full run will; the early exit keeps it cheap.
    return _run_remote(
        run_label=run_label,
        smoke=False,
        require_path_cache=False,
        require_training_support_cache=False,
        skip_rollouts=True,
        recipe_name=recipe_name,
        path_cache_source_run=None,
        dry_launch=True,
        dry_launch_output=f"/artifacts/{run_label}/gpu_smoke.json",
        training_steps=training_steps,
        schedule_steps=schedule_steps,
        initialization_source_run=initialization_source_run,
        initialization_checkpoint_name=initialization_checkpoint_name,
        compatible_initialization=compatible_initialization,
        corrupted_prior_mix=corrupted_prior_mix,
        cycle_op_mix=cycle_op_mix,
        disable_ring_grow_macro=disable_ring_grow_macro,
        organic_vocabulary=organic_vocabulary,
        analogue_trace_pool=analogue_trace_pool or None,
        analogue_trace_count=analogue_trace_count,
        corrupted_prior_count=corrupted_prior_count or None,
        scaled_manifest=scaled_manifest or None,
    )


@app.function(
    image=image,
    gpu="A100",
    cpu=32.0,
    memory=65536,
    timeout=24 * 3600,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
)
def train_stage(
    run_label: str,
    recipe_name: str = RECIPE_NAME,
    path_cache_source_run: str | None = None,
    require_training_support_cache: bool = True,
    training_steps: int | None = None,
    schedule_steps: int | None = None,
    resume_source_run: str | None = None,
    initialization_source_run: str | None = None,
    initialization_checkpoint_name: str = "checkpoint.best_so_far.pt",
    compatible_initialization: bool = False,
    corrupted_prior_mix: bool = False,
    cycle_op_mix: bool = False,
    disable_ring_grow_macro: bool = False,
    organic_vocabulary: bool = False,
    analogue_trace_pool: str = "",
    analogue_trace_count: int = 0,
    corrupted_prior_count: int = 0,
    scaled_manifest: str = "",
    disable_early_stopping: bool = False,
    snapshot_checkpoints: bool = False,
) -> dict[str, object]:
    # Zero-mixture (RING_CORE_V1 / --scaled-manifest): denovo_keep=0 so NO de-novo path cache is opened and
    # the training-support cache is built on-the-fly; a warm-started zero-mixture run must not require (or
    # reuse) B's de-novo caches. Standard (non-scaled) runs keep the strict cache requirements.
    zero_mixture = bool(scaled_manifest)
    return _run_remote(
        run_label=run_label,
        smoke=False,
        require_path_cache=not zero_mixture,
        require_training_support_cache=require_training_support_cache and not zero_mixture,
        skip_rollouts=True,
        recipe_name=recipe_name,
        path_cache_source_run=None if zero_mixture else path_cache_source_run,
        training_steps=training_steps,
        schedule_steps=schedule_steps,
        resume_source_run=resume_source_run,
        initialization_source_run=initialization_source_run,
        initialization_checkpoint_name=initialization_checkpoint_name,
        compatible_initialization=compatible_initialization,
        corrupted_prior_mix=corrupted_prior_mix,
        cycle_op_mix=cycle_op_mix,
        disable_ring_grow_macro=disable_ring_grow_macro,
        organic_vocabulary=organic_vocabulary,
        analogue_trace_pool=analogue_trace_pool or None,
        analogue_trace_count=analogue_trace_count,
        corrupted_prior_count=corrupted_prior_count or None,
        scaled_manifest=scaled_manifest or None,
        disable_early_stopping=disable_early_stopping,
        snapshot_checkpoints=snapshot_checkpoints,
    )


@app.function(
    image=image,
    gpu="H100",
    cpu=16.0,
    memory=65536,
    timeout=24 * 3600,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
)
def evaluate_stage(
    run_label: str,
    source_run_label: str,
    recipe_name: str = RECIPE_NAME,
    checkpoint_name: str = "checkpoint.pt",
) -> dict[str, object]:
    """Evaluate a selected checkpoint without resuming its training state."""

    return _run_remote(
        run_label=run_label,
        smoke=False,
        require_path_cache=True,
        recipe_name=recipe_name,
        evaluation_source_run=source_run_label,
        evaluation_checkpoint_name=checkpoint_name,
    )


@app.function(
    image=image,
    cpu=16.0,
    memory=65536,
    timeout=24 * 3600,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
def rollout_evaluate_stage(
    run_label: str,
    source_run_label: str,
    checkpoint_name: str = "checkpoint.best_so_far.pt",
    rollout_samples: int = 2000,
    disable_ring_ear: bool = False,
    evaluation_profile_name: str = "production_v1",
) -> dict[str, object]:
    """Evaluate ancestral samples without loading endpoint-conditioned teachers."""

    _validate_run_label(run_label)
    _validate_run_label(source_run_label, field="source run label")
    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint name must be a basename")
    if rollout_samples <= 0:
        raise ValueError("rollout_samples must be positive")
    evaluation_profile = _rollout_evaluation_profile(evaluation_profile_name)

    guacamol_volume.reload()
    artifact_volume.reload()
    source_checkpoint = Path("/artifacts") / source_run_label / checkpoint_name
    train_file = Path("/guacamol/guacamol_subset_500000_seed0.smiles")
    reference_file = Path("/guacamol/guacamol_heldout_val_5000_seed0.smiles")
    for required in (source_checkpoint, train_file, reference_file):
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(f"missing rollout-evaluation input: {required}")

    run_dir = Path("/artifacts") / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "manifest.json"
    source_checkpoint_sha256 = _file_sha256(source_checkpoint)
    disabled_rule_names = ("ring_ear_insert",) if disable_ring_ear else ()
    signature = _rollout_manifest_signature(
        source_run_label=source_run_label,
        checkpoint_name=checkpoint_name,
        checkpoint_sha256=source_checkpoint_sha256,
        rollout_samples=rollout_samples,
        disabled_rule_names=disabled_rule_names,
        train_sha256=_file_sha256(train_file),
        reference_sha256=_file_sha256(reference_file),
        source_sha256=_source_fingerprint(),
        evaluation_profile=evaluation_profile,
    )
    checkpoint_snapshot = run_dir / "checkpoint.snapshot.pt"
    rollout_cache_path = run_dir / "rollouts.pt"
    manifest = {
        "evaluation_kind": "checkpoint_rollout_only",
        "run_label": run_label,
        "source_run_label": source_run_label,
        "source_checkpoint": str(source_checkpoint),
        "checkpoint": str(checkpoint_snapshot),
        "checkpoint_sha256": source_checkpoint_sha256,
        "rollout_samples": rollout_samples,
        "disabled_rule_names": list(disabled_rule_names),
        "teacher_path_cache_loaded": False,
        "source_sha256": signature["source_sha256"],
        "data": {
            "train_sha256": signature["train_sha256"],
            "reference_sha256": signature["reference_sha256"],
        },
        "signature": signature,
    }
    has_manifest = manifest_path.is_file()
    if not has_manifest and any(
        path.exists() for path in (metrics_path, checkpoint_snapshot, rollout_cache_path)
    ):
        raise ValueError(
            "rollout evaluation directory contains artifacts without an auditable manifest"
        )
    if has_manifest:
        try:
            existing_manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"invalid existing rollout manifest: {manifest_path}") from error
        if (
            not isinstance(existing_manifest, dict)
            or existing_manifest.get("signature") != signature
        ):
            raise ValueError(
                "rollout evaluation run label already exists with a different signature"
            )
    else:
        # Commit request identity before copying the checkpoint.  A retry can
        # reconstruct a missing snapshot from its hash-bound source, whereas a
        # snapshot without a manifest is deliberately rejected as unaudited.
        _atomic_json_write(manifest, manifest_path)
        artifact_volume.commit()

    if (
        not checkpoint_snapshot.is_file()
        or checkpoint_snapshot.stat().st_size == 0
        or _file_sha256(checkpoint_snapshot) != source_checkpoint_sha256
    ):
        temporary_snapshot = checkpoint_snapshot.with_suffix(".pt.tmp")
        shutil.copyfile(source_checkpoint, temporary_snapshot)
        os.replace(temporary_snapshot, checkpoint_snapshot)
    if _file_sha256(checkpoint_snapshot) != source_checkpoint_sha256:
        raise RuntimeError("checkpoint snapshot hash does not match its source")

    if has_manifest:
        if metrics_path.is_file() and metrics_path.stat().st_size > 0:
            report = json.loads(metrics_path.read_text())
            if not isinstance(report, dict):
                raise ValueError("rollout metrics artifact must contain an object")
            summary = _rollout_remote_summary(
                run_label=run_label,
                run_dir=run_dir,
                report=report,
            )
            _update_source_early_eval_marker(
                source_run_label=source_run_label,
                evaluation_run_label=run_label,
                status="completed",
                details={"metrics_sha256": _file_sha256(metrics_path)},
            )
            print(
                json.dumps(
                    {"phase": "remote_rollout_reused", **summary},
                    sort_keys=True,
                ),
                flush=True,
            )
            return summary
    _update_source_early_eval_marker(
        source_run_label=source_run_label,
        evaluation_run_label=run_label,
        status="running",
        details={"checkpoint_snapshot_sha256": manifest["checkpoint_sha256"]},
    )

    command = (
        sys.executable,
        str(REMOTE_ROOT / "scripts" / "evaluate_tracelet_rollouts.py"),
        str(train_file),
        "--checkpoint",
        str(checkpoint_snapshot),
        "--quality-reference-file",
        str(reference_file),
        "--output",
        str(run_dir / "metrics.json"),
        "--rollout-cache",
        str(run_dir / "rollouts.pt"),
        "--rollout-samples",
        str(rollout_samples),
        "--rollout-workers",
        str(evaluation_profile["rollout_workers"]),
        "--corpus-workers",
        str(evaluation_profile["corpus_workers"]),
        "--seed",
        str(evaluation_profile["seed"]),
        "--max-atoms",
        str(evaluation_profile["max_atoms"]),
        "--train-size",
        str(evaluation_profile["train_size"]),
        "--validation-size",
        str(evaluation_profile["validation_size"]),
        "--test-size",
        str(evaluation_profile["test_size"]),
        "--torch-threads",
        str(evaluation_profile["torch_threads"]),
        "--operational-horizon",
        str(evaluation_profile["operational_horizon"]),
        "--time-step",
        str(evaluation_profile["time_step"]),
        "--max-events",
        str(evaluation_profile["max_events"]),
        "--quality-reference-limit",
        str(evaluation_profile["quality_reference_limit"]),
        "--fcd-generated-limit",
        str(evaluation_profile["fcd_generated_limit"]),
    )
    if bool(evaluation_profile["fast_split"]):
        command += ("--fast-split",)
    if disable_ring_ear:
        command += ("--disable-rule", "ring_ear_insert")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REMOTE_ROOT / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    print(json.dumps({"phase": "remote_rollout_command", "argv": command}), flush=True)
    process = subprocess.Popen(
        command,
        cwd=REMOTE_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("phase") == "rollouts_saved":
            artifact_volume.commit()
    return_code = process.wait()
    artifact_volume.commit()
    if return_code != 0:
        _update_source_early_eval_marker(
            source_run_label=source_run_label,
            evaluation_run_label=run_label,
            status="failed",
            details={"return_code": return_code},
        )
        raise subprocess.CalledProcessError(return_code, command)

    if not metrics_path.is_file():
        _update_source_early_eval_marker(
            source_run_label=source_run_label,
            evaluation_run_label=run_label,
            status="failed",
            details={"error": "missing metrics.json"},
        )
        raise RuntimeError("rollout evaluation completed without metrics.json")
    report = json.loads(metrics_path.read_text())
    summary = _rollout_remote_summary(
        run_label=run_label,
        run_dir=run_dir,
        report=report,
    )
    _update_source_early_eval_marker(
        source_run_label=source_run_label,
        evaluation_run_label=run_label,
        status="completed",
        details={"metrics_sha256": _file_sha256(metrics_path)},
    )
    print(
        json.dumps({"phase": "remote_rollout_complete", **summary}, sort_keys=True),
        flush=True,
    )
    return summary


@app.function(
    image=image,
    cpu=16.0,
    memory=65536,
    timeout=24 * 3600,
    volumes={"/artifacts": artifact_volume},
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
)
def property_rollout_evaluate_stage(
    run_label: str,
    source_run_label: str,
    checkpoint_name: str = "checkpoint.pt",
    targets: str = "0.3,0.5,0.7,0.9",
    samples_per_target: int = 25,
    include_classifier_free_control: bool = True,
) -> dict[str, object]:
    """Run matched direct-property rollouts from a conditioned checkpoint."""

    _validate_run_label(run_label)
    _validate_run_label(source_run_label, field="source run label")
    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint name must be a basename")
    if samples_per_target <= 0:
        raise ValueError("samples per target must be positive")
    try:
        target_values = tuple(float(value) for value in targets.split(","))
    except ValueError as error:
        raise ValueError("property targets must be comma-separated floats") from error
    if not target_values or not all(math.isfinite(value) for value in target_values):
        raise ValueError("property targets must be finite and non-empty")

    artifact_volume.reload()
    source_checkpoint = Path("/artifacts") / source_run_label / checkpoint_name
    if not source_checkpoint.is_file() or source_checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"missing conditioned checkpoint: {source_checkpoint}")
    run_dir = Path("/artifacts") / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    rollout_cache = run_dir / "rollouts.pt"
    manifest_path = run_dir / "manifest.json"
    signature = {
        "format": "compose_v4_property_rollout_request_v1",
        "source_run_label": source_run_label,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": _file_sha256(source_checkpoint),
        "targets": target_values,
        "samples_per_target": samples_per_target,
        "include_classifier_free_control": include_classifier_free_control,
        "source_sha256": _source_fingerprint(),
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if not isinstance(existing, dict) or existing.get("signature") != signature:
            raise ValueError("property-evaluation run label has a different signature")
        if metrics_path.is_file() and metrics_path.stat().st_size > 0:
            report = json.loads(metrics_path.read_text())
            return {
                "run_label": run_label,
                "artifact_dir": str(run_dir),
                "target_achieved_spearman": report.get("target_achieved_spearman"),
                "reused": True,
            }
    else:
        _atomic_json_write({"signature": signature}, manifest_path)
        artifact_volume.commit()

    command: tuple[str, ...] = (
        sys.executable,
        str(REMOTE_ROOT / "scripts" / "evaluate_property_conditioned_rollouts.py"),
        "--checkpoint",
        str(source_checkpoint),
        "--output",
        str(metrics_path),
        "--rollout-cache",
        str(rollout_cache),
        "--samples-per-target",
        str(samples_per_target),
        "--workers",
        "12",
        "--seed",
        str(ROLLOUT_BASE_SEED),
        "--max-atoms",
        str(ROLLOUT_MAX_ATOMS),
        "--operational-horizon",
        str(ROLLOUT_OPERATIONAL_HORIZON),
        "--time-step",
        str(ROLLOUT_TIME_STEP),
        "--max-events",
        str(ROLLOUT_MAX_EVENTS),
    )
    if include_classifier_free_control:
        command += ("--include-classifier-free-control",)
    for value in target_values:
        command += ("--target", str(value))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}"
    environment["PYTHONUNBUFFERED"] = "1"
    print(json.dumps({"phase": "remote_property_rollout_command", "argv": command}), flush=True)
    process = subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        env=environment,
        check=False,
    )
    artifact_volume.commit()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise RuntimeError("property rollout evaluation produced no metrics")
    report = json.loads(metrics_path.read_text())
    summary = {
        "run_label": run_label,
        "artifact_dir": str(run_dir),
        "target_achieved_spearman": report.get("target_achieved_spearman"),
        "reused": False,
    }
    print(json.dumps({"phase": "remote_property_rollout_complete", **summary}), flush=True)
    return summary


@app.function(
    image=image,
    cpu=16.0,
    memory=65536,
    timeout=4 * 3600,
    volumes={"/artifacts": artifact_volume},
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
)
def qed_successor_guidance_gate_stage(
    run_label: str,
    source_run_label: str,
    rollout_source_run_label: str,
    checkpoint_name: str = "checkpoint.pt",
    rollout_cache_name: str = "rollouts.pt",
    rollout_condition_label: str = "qed_0.9",
    target_qed: float = 0.9,
    states: int = 100,
    proposals_per_state: int = 4,
    beta: float = 8.0,
    classifier_free_base: bool = False,
) -> dict[str, object]:
    """Audit QED control on shared valid-successor proposal sets."""

    for value, field in (
        (run_label, "run label"),
        (source_run_label, "source run label"),
        (rollout_source_run_label, "rollout source run label"),
    ):
        _validate_run_label(value, field=field)
    for value, field in (
        (checkpoint_name, "checkpoint name"),
        (rollout_cache_name, "rollout cache name"),
    ):
        if Path(value).name != value:
            raise ValueError(f"{field} must be a basename")
    if not 0.0 <= target_qed <= 1.0 or not math.isfinite(target_qed):
        raise ValueError("target QED must be finite and lie in [0, 1]")
    if states <= 0 or proposals_per_state <= 0:
        raise ValueError("state and proposal counts must be positive")
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("guidance beta must be finite and positive")

    artifact_volume.reload()
    source_checkpoint = Path("/artifacts") / source_run_label / checkpoint_name
    rollout_cache = (
        Path("/artifacts") / rollout_source_run_label / rollout_cache_name
    )
    for required in (source_checkpoint, rollout_cache):
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(f"missing QED guidance-gate input: {required}")
    run_dir = Path("/artifacts") / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "manifest.json"
    signature = {
        "format": "compose_v4_qed_common_successor_gate_request_v1",
        "source_run_label": source_run_label,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": _file_sha256(source_checkpoint),
        "rollout_source_run_label": rollout_source_run_label,
        "rollout_cache_name": rollout_cache_name,
        "rollout_cache_sha256": _file_sha256(rollout_cache),
        "rollout_condition_label": rollout_condition_label,
        "target_qed": target_qed,
        "states": states,
        "proposals_per_state": proposals_per_state,
        "beta": beta,
        "classifier_free_base": classifier_free_base,
        "seed": 20260722,
        "source_sha256": _source_fingerprint(),
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if not isinstance(existing, dict) or existing.get("signature") != signature:
            raise ValueError("QED guidance-gate run label has a different signature")
        if metrics_path.is_file() and metrics_path.stat().st_size > 0:
            report = json.loads(metrics_path.read_text())
            return {
                "run_label": run_label,
                "artifact_dir": str(run_dir),
                "gate": report.get("gate"),
                "reused": True,
            }
    else:
        _atomic_json_write({"signature": signature}, manifest_path)
        artifact_volume.commit()

    command: tuple[str, ...] = (
        sys.executable,
        str(REMOTE_ROOT / "scripts" / "evaluate_qed_successor_guidance.py"),
        "--checkpoint",
        str(source_checkpoint),
        "--rollout-cache",
        str(rollout_cache),
        "--rollout-condition-label",
        rollout_condition_label,
        "--output",
        str(metrics_path),
        "--target-qed",
        str(target_qed),
        "--states",
        str(states),
        "--proposals-per-state",
        str(proposals_per_state),
        "--beta",
        str(beta),
        "--seed",
        "20260722",
        "--max-atoms",
        str(ROLLOUT_MAX_ATOMS),
    )
    if classifier_free_base:
        command += ("--classifier-free-base",)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}"
    environment["PYTHONUNBUFFERED"] = "1"
    print(
        json.dumps({"phase": "remote_qed_successor_gate_command", "argv": command}),
        flush=True,
    )
    process = subprocess.run(command, cwd=REMOTE_ROOT, env=environment, check=False)
    artifact_volume.commit()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise RuntimeError("QED successor guidance gate produced no metrics")
    report = json.loads(metrics_path.read_text())
    summary = {
        "run_label": run_label,
        "artifact_dir": str(run_dir),
        "gate": report.get("gate"),
        "reused": False,
    }
    print(json.dumps({"phase": "remote_qed_successor_gate_complete", **summary}), flush=True)
    return summary


@app.function(
    image=image,
    cpu=16.0,
    memory=65536,
    timeout=24 * 3600,
    volumes={"/artifacts": artifact_volume},
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
)
def qed_controlled_rollout_evaluate_stage(
    run_label: str,
    source_run_label: str,
    checkpoint_name: str = "checkpoint.pt",
    target_qed: float = 0.9,
    samples: int = 10,
    proposals_per_event: int = 4,
    guidance_start_event: int = 0,
    max_guided_events: int = 12,
    beta: float = 8.0,
) -> dict[str, object]:
    """Run a small beta-zero versus target-distance controlled CTMC."""

    _validate_run_label(run_label)
    _validate_run_label(source_run_label, field="source run label")
    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint name must be a basename")
    if not 0.0 <= target_qed <= 1.0 or not math.isfinite(target_qed):
        raise ValueError("target QED must be finite and lie in [0, 1]")
    if samples <= 0 or proposals_per_event <= 0 or max_guided_events <= 0:
        raise ValueError("controlled rollout counts must be positive")
    if guidance_start_event < 0:
        raise ValueError("guidance start event must be nonnegative")
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("guidance beta must be finite and positive")

    artifact_volume.reload()
    source_checkpoint = Path("/artifacts") / source_run_label / checkpoint_name
    if not source_checkpoint.is_file() or source_checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"missing controlled-rollout checkpoint: {source_checkpoint}")
    run_dir = Path("/artifacts") / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    rollout_cache = run_dir / "rollouts.pt"
    manifest_path = run_dir / "manifest.json"
    signature = {
        "format": "compose_v4_qed_controlled_rollout_request_v1",
        "source_run_label": source_run_label,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": _file_sha256(source_checkpoint),
        "target_qed": target_qed,
        "samples": samples,
        "proposals_per_event": proposals_per_event,
        "guidance_start_event": guidance_start_event,
        "max_guided_events": max_guided_events,
        "beta": beta,
        "seed": ROLLOUT_BASE_SEED,
        "max_atoms": ROLLOUT_MAX_ATOMS,
        "operational_horizon": ROLLOUT_OPERATIONAL_HORIZON,
        "time_step": ROLLOUT_TIME_STEP,
        "max_events": ROLLOUT_MAX_EVENTS,
        "source_sha256": _source_fingerprint(),
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if not isinstance(existing, dict) or existing.get("signature") != signature:
            raise ValueError("controlled-rollout run label has a different signature")
        if metrics_path.is_file() and metrics_path.stat().st_size > 0:
            report = json.loads(metrics_path.read_text())
            return {
                "run_label": run_label,
                "artifact_dir": str(run_dir),
                "paired": report.get("paired"),
                "reused": True,
            }
    else:
        _atomic_json_write({"signature": signature}, manifest_path)
        artifact_volume.commit()

    command = (
        sys.executable,
        str(REMOTE_ROOT / "scripts" / "evaluate_qed_controlled_rollouts.py"),
        "--checkpoint",
        str(source_checkpoint),
        "--output",
        str(metrics_path),
        "--rollout-cache",
        str(rollout_cache),
        "--target-qed",
        str(target_qed),
        "--samples",
        str(samples),
        "--workers",
        str(min(samples, 10)),
        "--proposals-per-event",
        str(proposals_per_event),
        "--guidance-start-event",
        str(guidance_start_event),
        "--max-guided-events",
        str(max_guided_events),
        "--beta",
        str(beta),
        "--seed",
        str(ROLLOUT_BASE_SEED),
        "--max-atoms",
        str(ROLLOUT_MAX_ATOMS),
        "--operational-horizon",
        str(ROLLOUT_OPERATIONAL_HORIZON),
        "--time-step",
        str(ROLLOUT_TIME_STEP),
        "--max-events",
        str(ROLLOUT_MAX_EVENTS),
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}"
    environment["PYTHONUNBUFFERED"] = "1"
    print(
        json.dumps({"phase": "remote_qed_controlled_rollout_command", "argv": command}),
        flush=True,
    )
    process = subprocess.run(command, cwd=REMOTE_ROOT, env=environment, check=False)
    artifact_volume.commit()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise RuntimeError("controlled QED rollout produced no metrics")
    report = json.loads(metrics_path.read_text())
    summary = {
        "run_label": run_label,
        "artifact_dir": str(run_dir),
        "paired": report.get("paired"),
        "reused": False,
    }
    print(
        json.dumps({"phase": "remote_qed_controlled_rollout_complete", **summary}),
        flush=True,
    )
    return summary


def _run_pipeline(
    run_label: str,
    recipe_name: str,
    *,
    final_rollout_samples: int,
    audit_batch_size: int,
    audit_workers: int,
    evaluation_profile_name: str,
    require_fcd: bool = False,
) -> dict[str, object]:
    """Run the same CPU -> GPU -> CPU boundary for production and its smoke."""

    compile_result = compile_stage.remote(run_label, recipe_name)
    audit_result = audit_teacher_stage.remote(
        run_label,
        recipe_name,
        audit_batch_size,
        audit_workers,
    )
    # The legacy all-in-one smoke pipeline intentionally exercises the online
    # oracle on a tiny corpus. Production launches use --support-compile-only
    # followed by --train-only, where cached support is mandatory.
    train_result = train_stage.remote(run_label, recipe_name, None, False, None)
    if train_result.get("rollouts_skipped") is not True:
        raise RuntimeError("GPU training unexpectedly performed in-container rollouts")
    evaluation_run_label, checkpoint_name, rollout_samples = _final_rollout_spec(
        run_label,
        train_result,
        rollout_samples=final_rollout_samples,
    )
    final_rollout_result = rollout_evaluate_stage.remote(
        evaluation_run_label,
        run_label,
        checkpoint_name,
        rollout_samples,
        False,
        evaluation_profile_name,
    )
    if require_fcd:
        generated = final_rollout_result.get("generated_nonnull_smiles")
        valid_fraction = final_rollout_result.get("valid_fraction")
        fcd = final_rollout_result.get("fcd")
        if not isinstance(generated, int) or generated < 2:
            raise RuntimeError("integration smoke produced fewer than two molecules")
        if (
            isinstance(valid_fraction, bool)
            or not isinstance(valid_fraction, (int, float))
            or not math.isfinite(float(valid_fraction))
        ):
            raise RuntimeError(
                "integration smoke did not report a numeric valid fraction"
            )
        if final_rollout_result.get("trajectory_diagnostics_available") is not True:
            raise RuntimeError(
                "integration smoke did not produce trajectory diagnostics"
            )
        if not isinstance(fcd, (int, float)) or not math.isfinite(float(fcd)):
            raise RuntimeError("integration smoke did not complete finite CPU FCD")
    return {
        "compile": compile_result,
        "audit": audit_result,
        "train": train_result,
        "final_rollout": final_rollout_result,
    }


@app.function(
    image=image,
    cpu=0.25,
    memory=512,
    timeout=24 * 3600,
)
def pipeline_stage(
    run_label: str,
    recipe_name: str = RECIPE_NAME,
) -> dict[str, object]:
    """Run CPU preprocessing, A100 training, then CPU final evaluation."""

    return _run_pipeline(
        run_label,
        recipe_name,
        final_rollout_samples=FINAL_ROLLOUT_SAMPLES,
        audit_batch_size=2048,
        audit_workers=12,
        evaluation_profile_name="production_v1",
    )


@app.function(
    image=image,
    cpu=0.25,
    memory=512,
    timeout=2 * 3600,
)
def integration_smoke_pipeline_stage(run_label: str) -> dict[str, object]:
    """Exercise compile -> audit -> A100 train -> CPU rollout/FCD cheaply."""

    return _run_pipeline(
        run_label,
        INTEGRATION_SMOKE_RECIPE_NAME,
        final_rollout_samples=INTEGRATION_SMOKE_ROLLOUT_SAMPLES,
        audit_batch_size=16,
        audit_workers=1,
        evaluation_profile_name="integration_smoke_v1",
        require_fcd=True,
    )


@app.local_entrypoint()
def main(
    smoke: bool = False,
    integration_smoke: bool = False,
    preflight: bool = False,
    h100_preflight: bool = False,
    compile_only: bool = False,
    support_compile_only: bool = False,
    train_only: bool = False,
    evaluate_only: bool = False,
    rollout_evaluate_only: bool = False,
    property_rollout_evaluate_only: bool = False,
    qed_successor_guidance_gate_only: bool = False,
    qed_controlled_rollout_evaluate_only: bool = False,
    run_label: str = "",
    recipe_name: str = RECIPE_NAME,
    source_run_label: str = "",
    rollout_source_run_label: str = "",
    resume_source_run_label: str = "",
    initialization_source_run_label: str = "",
    checkpoint_name: str = "checkpoint.best_so_far.pt",
    rollout_samples: int = 100,
    property_targets: str = "0.3,0.5,0.7,0.9",
    property_include_classifier_free_control: bool = True,
    rollout_cache_name: str = "rollouts.pt",
    rollout_condition_label: str = "qed_0.9",
    guidance_target_qed: float = 0.9,
    guidance_states: int = 100,
    guidance_proposals_per_state: int = 4,
    guidance_beta: float = 8.0,
    guidance_classifier_free_base: bool = False,
    guidance_max_controlled_events: int = 12,
    guidance_start_event: int = 0,
    support_start_step: int = 0,
    support_steps: int = 2000,
    support_containers: int = 1,
    support_workers: int = 12,
    training_steps: int = 0,
    schedule_steps: int = 0,
    disable_early_stopping: bool = False,
    snapshot_checkpoints: bool = False,
    initialize_from_source_checkpoint: bool = False,
    initialize_compatible_from_source_checkpoint: bool = False,
    corrupted_prior_mix: bool = False,
    cycle_op_mix: bool = False,
    disable_ring_grow_macro: bool = False,
    dry_launch: bool = False,
    gpu_smoke: bool = False,
    organic_vocabulary: bool = False,
    analogue_trace_pool: str = "",
    analogue_trace_count: int = 0,
    corrupted_prior_count: int = 0,
    scaled_manifest: str = "",
) -> None:
    modes = sum(
        (
            smoke,
            integration_smoke,
            preflight,
            h100_preflight,
            compile_only,
            support_compile_only,
            train_only,
            dry_launch,
            gpu_smoke,
            evaluate_only,
            rollout_evaluate_only,
            property_rollout_evaluate_only,
            qed_successor_guidance_gate_only,
            qed_controlled_rollout_evaluate_only,
        )
    )
    if modes > 1:
        raise ValueError(
            "smoke, integration-smoke, preflight, h100-preflight, compile-only, "
            "support-compile-only, train-only, "
            "evaluate-only, and "
            "rollout-evaluate-only, property-rollout-evaluate-only, and "
            "QED guidance modes are exclusive"
        )
    if not run_label:
        raise ValueError(
            "--run-label is required; use a fresh commit-bearing immutable label"
        )
    _validate_run_label(run_label)
    if support_steps <= 0:
        raise ValueError("--support-steps must be positive")
    if support_start_step < 0:
        raise ValueError("--support-start-step must be non-negative")
    if support_containers <= 0:
        raise ValueError("--support-containers must be positive")
    if not 0 < support_workers < 14:
        raise ValueError("--support-workers must lie in [1, 13]")
    if training_steps < 0:
        raise ValueError("--training-steps must be non-negative")
    if schedule_steps < 0:
        raise ValueError("--schedule-steps must be non-negative")
    if schedule_steps and not (train_only or dry_launch or gpu_smoke):
        raise ValueError("--schedule-steps requires --train-only")
    if resume_source_run_label and not train_only:
        raise ValueError("--resume-source-run-label requires --train-only")
    if resume_source_run_label and (
        initialize_from_source_checkpoint
        or initialize_compatible_from_source_checkpoint
    ):
        raise ValueError("recovery resume and fresh initialization are exclusive")
    if initialize_from_source_checkpoint and not train_only:
        raise ValueError(
            "--initialize-from-source-checkpoint requires --train-only"
        )
    if initialize_from_source_checkpoint and not (
        initialization_source_run_label or source_run_label
    ):
        raise ValueError(
            "--initialize-from-source-checkpoint requires either "
            "--initialization-source-run-label or --source-run-label"
        )
    if initialize_compatible_from_source_checkpoint and not (
        train_only or dry_launch or gpu_smoke
    ):
        raise ValueError(
            "--initialize-compatible-from-source-checkpoint requires --train-only"
        )
    if initialize_compatible_from_source_checkpoint and not (
        initialization_source_run_label or source_run_label
    ):
        raise ValueError(
            "--initialize-compatible-from-source-checkpoint requires either "
            "--initialization-source-run-label or --source-run-label"
        )
    if initialize_from_source_checkpoint and initialize_compatible_from_source_checkpoint:
        raise ValueError(
            "strict and compatible source-checkpoint initialization are exclusive"
        )
    if integration_smoke:
        call = integration_smoke_pipeline_stage.spawn(run_label)
        print(
            json.dumps(
                {
                    "phase": "integration_smoke_pipeline_spawned",
                    "function_call_id": call.object_id,
                    "recipe_name": INTEGRATION_SMOKE_RECIPE_NAME,
                }
            )
        )
        print("Artifacts: Modal volume compose-v4-artifacts / " + run_label)
        return
    if smoke:
        result = smoke_stage.remote(run_label, recipe_name)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if preflight:
        result = preflight_stage.remote(run_label, recipe_name)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if h100_preflight:
        result = h100_preflight_stage.remote(run_label, recipe_name)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if qed_controlled_rollout_evaluate_only:
        if not source_run_label:
            raise ValueError(
                "qed-controlled-rollout-evaluate-only requires --source-run-label"
            )
        call = qed_controlled_rollout_evaluate_stage.spawn(
            run_label,
            source_run_label,
            checkpoint_name,
            guidance_target_qed,
            rollout_samples,
            guidance_proposals_per_state,
            guidance_start_event,
            guidance_max_controlled_events,
            guidance_beta,
        )
        phase = "qed_controlled_rollout_evaluation_spawned"
    elif qed_successor_guidance_gate_only:
        if not source_run_label or not rollout_source_run_label:
            raise ValueError(
                "qed-successor-guidance-gate-only requires --source-run-label and "
                "--rollout-source-run-label"
            )
        call = qed_successor_guidance_gate_stage.spawn(
            run_label,
            source_run_label,
            rollout_source_run_label,
            checkpoint_name,
            rollout_cache_name,
            rollout_condition_label,
            guidance_target_qed,
            guidance_states,
            guidance_proposals_per_state,
            guidance_beta,
            guidance_classifier_free_base,
        )
        phase = "qed_successor_guidance_gate_spawned"
    elif property_rollout_evaluate_only:
        if not source_run_label:
            raise ValueError(
                "property-rollout-evaluate-only requires --source-run-label"
            )
        call = property_rollout_evaluate_stage.spawn(
            run_label,
            source_run_label,
            checkpoint_name,
            property_targets,
            rollout_samples,
            property_include_classifier_free_control,
        )
        phase = "property_rollout_evaluation_spawned"
    elif rollout_evaluate_only:
        if not source_run_label:
            raise ValueError("rollout-evaluate-only requires --source-run-label")
        call = rollout_evaluate_stage.spawn(
            run_label,
            source_run_label,
            checkpoint_name,
            rollout_samples,
        )
        phase = "rollout_evaluation_spawned"
    elif evaluate_only:
        if not source_run_label:
            raise ValueError("evaluate-only requires --source-run-label")
        call = evaluate_stage.spawn(
            run_label,
            source_run_label,
            recipe_name,
            checkpoint_name,
        )
        phase = "evaluation_spawned"
    elif support_compile_only:
        if not source_run_label:
            raise ValueError(
                "support-compile-only requires --source-run-label for paths"
            )
        ranges = _partition_training_support_steps(
            start_step=support_start_step,
            steps=support_steps,
            containers=support_containers,
        )
        calls = []
        for index, (range_start, range_steps) in enumerate(ranges):
            range_label = (
                run_label
                if len(ranges) == 1
                else f"{run_label}-part{index:02d}"
            )
            _validate_run_label(range_label)
            call = compile_training_support_stage.spawn(
                run_label=range_label,
                path_cache_source_run=source_run_label,
                steps=range_steps,
                recipe_name=recipe_name,
                start_step=range_start,
                workers=support_workers,
                minimum_rows_per_second=0.0,
            )
            calls.append(
                {
                    "function_call_id": call.object_id,
                    "run_label": range_label,
                    "start_step": range_start,
                    "steps": range_steps,
                }
            )
        phase = "support_compile_spawned"
    elif compile_only:
        call = compile_stage.spawn(
            run_label,
            recipe_name,
            source_run_label or None,
            organic_vocabulary,
        )
        phase = "compile_spawned"
    elif dry_launch:
        call = dry_launch_stage.spawn(
            run_label,
            recipe_name,
            initialization_source_run=(
                initialization_source_run_label or source_run_label
                if (
                    initialize_from_source_checkpoint
                    or initialize_compatible_from_source_checkpoint
                )
                else None
            ),
            initialization_checkpoint_name=checkpoint_name,
            compatible_initialization=initialize_compatible_from_source_checkpoint,
            corrupted_prior_mix=corrupted_prior_mix,
            cycle_op_mix=cycle_op_mix,
            disable_ring_grow_macro=disable_ring_grow_macro,
            organic_vocabulary=organic_vocabulary,
            analogue_trace_pool=analogue_trace_pool,
            analogue_trace_count=analogue_trace_count,
            corrupted_prior_count=corrupted_prior_count,
            scaled_manifest=scaled_manifest,
            training_steps=training_steps or None,
            schedule_steps=schedule_steps or None,
        )
        phase = "dry_launch_spawned"
    elif gpu_smoke:
        call = gpu_smoke_stage.spawn(
            run_label,
            recipe_name,
            initialization_source_run=(
                initialization_source_run_label or source_run_label
                if (
                    initialize_from_source_checkpoint
                    or initialize_compatible_from_source_checkpoint
                )
                else None
            ),
            initialization_checkpoint_name=checkpoint_name,
            compatible_initialization=initialize_compatible_from_source_checkpoint,
            corrupted_prior_mix=corrupted_prior_mix,
            cycle_op_mix=cycle_op_mix,
            disable_ring_grow_macro=disable_ring_grow_macro,
            organic_vocabulary=organic_vocabulary,
            analogue_trace_pool=analogue_trace_pool,
            analogue_trace_count=analogue_trace_count,
            corrupted_prior_count=corrupted_prior_count,
            scaled_manifest=scaled_manifest,
            training_steps=training_steps or None,
            schedule_steps=schedule_steps or None,
        )
        phase = "gpu_smoke_spawned"
    elif train_only:
        call = train_stage.spawn(
            run_label,
            recipe_name,
            source_run_label or None,
            True,
            training_steps or None,
            schedule_steps or None,
            resume_source_run_label or None,
            (
                initialization_source_run_label or source_run_label
                if (
                    initialize_from_source_checkpoint
                    or initialize_compatible_from_source_checkpoint
                )
                else None
            ),
            checkpoint_name,
            initialize_compatible_from_source_checkpoint,
            corrupted_prior_mix,
            cycle_op_mix,
            disable_ring_grow_macro,
            organic_vocabulary,
            analogue_trace_pool,
            analogue_trace_count,
            corrupted_prior_count,
            scaled_manifest,
            disable_early_stopping=disable_early_stopping,
            snapshot_checkpoints=snapshot_checkpoints,
        )
        phase = "train_spawned"
    else:
        call = pipeline_stage.spawn(run_label, recipe_name)
        phase = "pipeline_spawned"
    payload = {"phase": phase, "recipe_name": recipe_name}
    if train_only:
        payload["training_steps"] = training_steps or None
        payload["schedule_steps"] = schedule_steps or None
    if support_compile_only:
        payload["calls"] = calls
    else:
        payload["function_call_id"] = call.object_id
    print(json.dumps(payload))
    print("Artifacts: Modal volume compose-v4-artifacts / " + run_label)
