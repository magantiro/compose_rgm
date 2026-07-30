"""Freeze a completed Modal training run without mutating its source directory.

The function hashes every declared checkpoint where the artifacts already live,
loads each exact-recovery snapshot on CPU to verify its step and provenance,
and writes one immutable JSON inventory under ``_frozen_run_inventories``.

Example:

    MODAL_PROFILE=nitya modal run modal_apps/freeze_training_run.py \
      --run-label compose-v4-ringcore-v1-scientific-a7546e2-v1 \
      --maximum-step 16000 \
      --snapshot-interval 500
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping

import modal


RUN_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SNAPSHOT_PATTERN = re.compile(r"checkpoint\.step([1-9][0-9]*)\.pt\Z")
INVENTORY_SCHEMA_VERSION = 1
FROZEN_INVENTORY_DIRECTORY = "_frozen_run_inventories"
ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")

ESSENTIAL_RUN_FILES = (
    "manifest.json",
    "manifest.training.json",
    "metrics.json",
    "checkpoint.pt",
    "checkpoint.best_so_far.pt",
    "checkpoint.recovery.pt",
)

CHECKPOINT_METADATA_KEYS = (
    "model",
    "training_backend",
    "hidden_dim",
    "message_passing_steps",
    "rate_factorization",
    "bond_representation",
    "organic_vocabulary",
    "corpus_scope",
    "corpus_scope_hash",
    "ring_catalog_fingerprint",
    "training_steps",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "optimizer_kind",
    "warmup_steps",
    "schedule_steps",
    "minimum_learning_rate_fraction",
    "evaluation_every",
    "evaluation_batch_size",
    "early_stopping_patience",
    "early_stopping_min_relative_delta",
    "late_time_fraction",
    "operational_horizon",
    "progress_stratification_fraction",
    "use_bf16",
    "trainable_parameter_scope",
    "provenance_sha256",
    "corrupted_prior_mix",
    "enable_cycle_ops",
    "enable_ring_grow_macro",
    "enable_ring_macros",
    "max_atoms",
    "teacher_representability_filter_version",
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(
        ROOT / "src",
        str(REMOTE_ROOT / "src"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
)
app = modal.App("compose-v4-freeze-training-run")
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts",
    create_if_missing=False,
)


def _validate_run_label(run_label: str) -> None:
    if Path(run_label).name != run_label or RUN_LABEL_PATTERN.fullmatch(run_label) is None:
        raise ValueError(
            "run_label must be a 1-128 character ASCII basename containing only "
            "letters, digits, '.', '_', or '-', and must begin with a letter or digit"
        )


def _expected_snapshot_steps(
    *,
    maximum_step: int,
    snapshot_interval: int,
) -> tuple[int, ...]:
    if maximum_step <= 0 or snapshot_interval <= 0:
        raise ValueError("maximum_step and snapshot_interval must be positive")
    if maximum_step % snapshot_interval:
        raise ValueError("maximum_step must be divisible by snapshot_interval")
    return tuple(range(snapshot_interval, maximum_step + 1, snapshot_interval))


def _snapshot_step(path: Path) -> int:
    match = SNAPSHOT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not a step snapshot: {path.name}")
    return int(match.group(1))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_scalar(value: object, *, key: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"checkpoint metadata {key!r} is not a finite JSON scalar: {value!r}")


def _checkpoint_metadata(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: _json_scalar(payload.get(key), key=key)
        for key in CHECKPOINT_METADATA_KEYS
    }


def _state_schema(state: object, *, field: str) -> list[dict[str, object]]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"checkpoint {field} must be a non-empty mapping")
    schema = []
    for name, value in sorted(state.items()):
        if not isinstance(name, str):
            raise TypeError(f"checkpoint {field} contains a non-string parameter name")
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is None or dtype is None:
            raise TypeError(f"checkpoint {field}[{name!r}] is not tensor-like")
        schema.append(
            {
                "name": name,
                "shape": [int(dimension) for dimension in shape],
                "dtype": str(dtype),
            }
        )
    return schema


def _checkpoint_summary(
    payload: object,
    *,
    expected_step: int,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    required = {
        "checkpoint_kind",
        "completed_steps",
        "current_state_dict",
        "optimizer_state_dict",
        "best_state_dict",
        "best_metrics",
        "history",
        "initial_validation",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"snapshot is missing exact-recovery fields: {missing}")
    if payload["checkpoint_kind"] != "exact_training_recovery":
        raise ValueError(
            "step snapshot is not an exact training recovery checkpoint: "
            f"{payload['checkpoint_kind']!r}"
        )
    completed_steps = int(payload["completed_steps"])
    if completed_steps != expected_step:
        raise ValueError(
            f"snapshot filename step {expected_step} disagrees with payload {completed_steps}"
        )
    history = payload["history"]
    if not isinstance(history, list) or not history:
        raise ValueError("snapshot history must be a non-empty list")
    last_history = history[-1]
    if not isinstance(last_history, Mapping):
        raise TypeError("snapshot history rows must be mappings")
    last_history_step = int(float(last_history["step"]))
    if last_history_step != expected_step:
        raise ValueError(
            f"snapshot history ends at {last_history_step}, expected {expected_step}"
        )
    current_schema = _state_schema(
        payload["current_state_dict"],
        field="current_state_dict",
    )
    best_schema = _state_schema(
        payload["best_state_dict"],
        field="best_state_dict",
    )
    if current_schema != best_schema:
        raise ValueError("current and selected-best model schemas differ")
    best_metrics = payload["best_metrics"]
    if not isinstance(best_metrics, Mapping):
        raise TypeError("best_metrics must be a mapping")
    initial_validation = payload["initial_validation"]
    if not isinstance(initial_validation, Mapping):
        raise TypeError("initial_validation must be a mapping")
    return {
        "completed_steps": completed_steps,
        "history_last_step": last_history_step,
        "history_rows": len(history),
        "trainer_selected_step": float(best_metrics["selected_step"]),
        "trainer_selected_factorized_gm_loss": float(
            best_metrics["factorized_gm_loss"]
        ),
        "initial_validation_factorized_gm_loss": float(
            initial_validation["factorized_gm_loss"]
        ),
        "state_tensor_count": len(current_schema),
        "state_schema_sha256": _stable_json_sha256(current_schema),
        "metadata": _checkpoint_metadata(payload),
        "metadata_sha256": _stable_json_sha256(_checkpoint_metadata(payload)),
    }


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def _artifact_entry(path: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _verify_manifests(
    *,
    run_label: str,
    maximum_step: int,
    manifest: Mapping[str, object],
    training_manifest: Mapping[str, object],
) -> dict[str, object]:
    if manifest.get("run_label") != run_label:
        raise ValueError("manifest run label disagrees with requested run")
    if manifest.get("stage_kind") != "training":
        raise ValueError("manifest does not describe a training stage")
    if int(manifest.get("training_steps", -1)) != maximum_step:
        raise ValueError("manifest training horizon disagrees with requested maximum step")
    run_identity_sha256 = str(manifest.get("run_identity_sha256", ""))
    if len(run_identity_sha256) != 64:
        raise ValueError("manifest lacks a full SHA-256 run identity")
    if training_manifest.get("run_identity_sha256") != run_identity_sha256:
        raise ValueError("training manifest run identity disagrees with manifest")
    recipe = manifest.get("recipe")
    if not isinstance(recipe, Mapping):
        raise TypeError("manifest recipe must be a mapping")
    arguments = recipe.get("arguments")
    if not isinstance(arguments, Mapping):
        raise TypeError("manifest recipe arguments must be a mapping")
    required_arguments = {
        "steps": maximum_step,
        "schedule_steps": maximum_step,
        "snapshot_checkpoints": True,
        "skip_rollouts": True,
        "organic_vocabulary": True,
        "cycle_op_mix": True,
        "disable_ring_grow_macro": True,
        "require_scientific_contract": True,
    }
    mismatches = {
        key: (arguments.get(key), value)
        for key, value in required_arguments.items()
        if arguments.get(key) != value
    }
    if mismatches:
        raise ValueError(f"manifest recipe contract mismatch: {mismatches}")
    if arguments.get("provenance_sha256") != run_identity_sha256:
        raise ValueError("recipe provenance does not equal the frozen run identity")
    return {
        "run_identity_sha256": run_identity_sha256,
        "source_sha256": manifest.get("source_sha256"),
        "recipe_name": manifest.get("recipe_name"),
        "required_scientific_contract": True,
        "organic_vocabulary": True,
        "cycle_operations_enabled": True,
        "ring_growth_macro_disabled": True,
    }


def _metrics_summary(metrics: Mapping[str, object], *, maximum_step: int) -> dict[str, object]:
    training = metrics.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("metrics training section must be a mapping")
    history = training.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("metrics training history must be a non-empty list")
    last_row = history[-1]
    if not isinstance(last_row, Mapping) or int(float(last_row["step"])) != maximum_step:
        raise ValueError("metrics history does not end at the requested maximum step")
    selected = metrics.get("selected_validation")
    initial = metrics.get("initial_validation")
    final_test = metrics.get("final_test")
    if not all(isinstance(section, Mapping) for section in (selected, initial, final_test)):
        raise TypeError("metrics lacks initial, selected, or final-test mappings")
    return {
        "history_rows": len(history),
        "history_last_step": int(float(last_row["step"])),
        "trainer_selected_step_not_preregistered": float(selected["selected_step"]),  # type: ignore[index]
        "initial_validation_factorized_gm_loss": float(  # type: ignore[index]
            initial["factorized_gm_loss"]
        ),
        "trainer_selected_factorized_gm_loss": float(  # type: ignore[index]
            selected["factorized_gm_loss"]
        ),
        "inspected_final_test_factorized_gm_loss": float(  # type: ignore[index]
            final_test["factorized_gm_loss"]
        ),
        "final_test_is_sealed": False,
    }


def _build_inventory(
    *,
    run_label: str,
    maximum_step: int,
    snapshot_interval: int,
    artifact_root: Path,
) -> dict[str, object]:
    _validate_run_label(run_label)
    expected_steps = _expected_snapshot_steps(
        maximum_step=maximum_step,
        snapshot_interval=snapshot_interval,
    )
    run_directory = artifact_root / run_label
    if not run_directory.is_dir():
        raise FileNotFoundError(f"training run directory is missing: {run_directory}")

    snapshot_paths = sorted(
        run_directory.glob("checkpoint.step*.pt"),
        key=_snapshot_step,
    )
    observed_steps = tuple(_snapshot_step(path) for path in snapshot_paths)
    if observed_steps != expected_steps:
        raise ValueError(
            "snapshot step inventory mismatch: "
            f"observed={observed_steps}, expected={expected_steps}"
        )

    essential_paths = [run_directory / name for name in ESSENTIAL_RUN_FILES]
    missing_essential = [path.name for path in essential_paths if not path.is_file()]
    if missing_essential:
        raise FileNotFoundError(f"run is missing essential artifacts: {missing_essential}")

    manifest = _load_json(run_directory / "manifest.json")
    training_manifest = _load_json(run_directory / "manifest.training.json")
    metrics = _load_json(run_directory / "metrics.json")
    manifest_summary = _verify_manifests(
        run_label=run_label,
        maximum_step=maximum_step,
        manifest=manifest,
        training_manifest=training_manifest,
    )

    import torch

    snapshot_inventory = []
    reference_metadata = None
    reference_state_schema_sha256 = None
    for path, expected_step in zip(snapshot_paths, expected_steps):
        entry = _artifact_entry(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        summary = _checkpoint_summary(payload, expected_step=expected_step)
        if reference_metadata is None:
            reference_metadata = summary["metadata"]
            reference_state_schema_sha256 = summary["state_schema_sha256"]
        elif summary["metadata"] != reference_metadata:
            raise ValueError(f"checkpoint metadata drift detected at step {expected_step}")
        if summary["state_schema_sha256"] != reference_state_schema_sha256:
            raise ValueError(f"model tensor schema drift detected at step {expected_step}")
        if summary["metadata"]["provenance_sha256"] != manifest_summary["run_identity_sha256"]:
            raise ValueError(f"checkpoint provenance drift detected at step {expected_step}")
        snapshot_inventory.append({**entry, **summary})
        del payload
        gc.collect()

    essential_inventory = [_artifact_entry(path) for path in essential_paths]
    inventory: dict[str, object] = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "artifact_kind": "frozen_training_run_inventory",
        "source_run_label": run_label,
        "source_run_directory_mutated": False,
        "snapshot_contract": {
            "maximum_step": maximum_step,
            "snapshot_interval": snapshot_interval,
            "expected_count": len(expected_steps),
            "expected_steps": list(expected_steps),
            "observed_count": len(observed_steps),
            "observed_steps": list(observed_steps),
        },
        "manifest_summary": manifest_summary,
        "metrics_summary": _metrics_summary(metrics, maximum_step=maximum_step),
        "checkpoint_metadata": reference_metadata,
        "checkpoint_metadata_sha256": _stable_json_sha256(reference_metadata),
        "state_schema_sha256": reference_state_schema_sha256,
        "snapshots": snapshot_inventory,
        "essential_artifacts": essential_inventory,
        "verification": {
            "passed": True,
            "snapshot_names_complete": True,
            "snapshot_payload_steps_match_names": True,
            "snapshot_histories_end_at_payload_steps": True,
            "checkpoint_metadata_constant": True,
            "checkpoint_provenance_matches_manifest": True,
            "model_tensor_schema_constant": True,
            "source_run_directory_mutated": False,
            "checkpoint_selection_performed": False,
            "test_metrics_used_for_selection": False,
        },
    }
    inventory["inventory_sha256"] = _stable_json_sha256(inventory)
    return inventory


def _atomic_immutable_write(payload: Mapping[str, object], path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(
                f"frozen inventory already exists with different content: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(encoded)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=2 * 3600,
    volumes={"/artifacts": artifact_volume},
)
def freeze_run(
    run_label: str,
    maximum_step: int,
    snapshot_interval: int,
) -> dict[str, object]:
    artifact_volume.reload()
    inventory = _build_inventory(
        run_label=run_label,
        maximum_step=maximum_step,
        snapshot_interval=snapshot_interval,
        artifact_root=Path("/artifacts"),
    )
    output_path = (
        Path("/artifacts")
        / FROZEN_INVENTORY_DIRECTORY
        / f"{run_label}.json"
    )
    _atomic_immutable_write(inventory, output_path)
    artifact_volume.commit()
    summary = {
        "phase": "training_run_frozen",
        "run_label": run_label,
        "inventory_path": str(output_path),
        "inventory_sha256": inventory["inventory_sha256"],
        "snapshot_count": inventory["snapshot_contract"]["observed_count"],  # type: ignore[index]
        "verification_passed": inventory["verification"]["passed"],  # type: ignore[index]
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


@app.local_entrypoint()
def main(
    run_label: str,
    maximum_step: int = 16000,
    snapshot_interval: int = 500,
) -> None:
    result = freeze_run.remote(
        run_label,
        maximum_step,
        snapshot_interval,
    )
    print(json.dumps(result, sort_keys=True))


__all__ = [
    "CHECKPOINT_METADATA_KEYS",
    "ESSENTIAL_RUN_FILES",
    "FROZEN_INVENTORY_DIRECTORY",
    "INVENTORY_SCHEMA_VERSION",
    "_build_inventory",
    "_checkpoint_summary",
    "_expected_snapshot_steps",
    "_snapshot_step",
    "_stable_json_sha256",
    "_validate_run_label",
    "freeze_run",
]
