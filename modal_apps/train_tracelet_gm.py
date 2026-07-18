"""Modal launch surface for the frozen carbon-tree Generator Matching stage.

The recipe remains the single source of truth.  The remote entrypoint changes
only artifact paths (to the persistent Modal volume) and, for ``--smoke``, a
small explicit set of resource-saving preflight values.

Run the mandatory remote preflight:

    modal run modal_apps/train_tracelet_gm.py --smoke

Launch the quality run so it survives local disconnects:

    modal run --detach modal_apps/train_tracelet_gm.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")
RECIPE_NAME = "tree_fcd_transfer_stage3_flexible_graft.json"

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
    .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True)
    .add_local_dir(ROOT / "recipes", str(REMOTE_ROOT / "recipes"), copy=True)
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
                "data_workers": 16,
                "data_prefetch_factor": 2,
                "path_workers": 16,
                "corpus_workers": 0,
                "quality_metrics": False,
                "include_fcd": False,
            }
        )
    recovery_path = run_dir / "checkpoint.recovery.pt"
    if recovery_path.is_file() and recovery_path.stat().st_size > 0:
        arguments["resume_checkpoint"] = str(recovery_path)
    return recipe, run_dir


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for directory_name in ("src", "scripts", "recipes"):
        directory = REMOTE_ROOT / directory_name
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            digest.update(str(path.relative_to(REMOTE_ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


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
    minimum_relative_improvement: float = 0.05,
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
    except (KeyError, TypeError, ValueError):
        return None
    if not (
        math.isfinite(initial_loss)
        and math.isfinite(selected_loss)
        and initial_loss > 0.0
        and 0.0 <= minimum_relative_improvement < 1.0
    ):
        return None
    relative_improvement = (initial_loss - selected_loss) / initial_loss
    if relative_improvement < minimum_relative_improvement:
        return None
    return {
        "completed_steps": float(completed_steps),
        "selected_step": selected_step,
        "initial_validation_loss": initial_loss,
        "selected_validation_loss": selected_loss,
        "relative_improvement": relative_improvement,
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


def _run_remote(
    *,
    run_label: str,
    smoke: bool,
    preflight: bool = False,
    compile_paths_only: bool = False,
    require_path_cache: bool = False,
    recipe_name: str = RECIPE_NAME,
    evaluation_source_run: str | None = None,
    evaluation_checkpoint_name: str = "checkpoint.best_step3000.pt",
) -> dict[str, object]:
    from compose_v4.experiments.recipe import build_tracelet_recipe_argv

    guacamol_volume.reload()
    artifact_volume.reload()
    started = time.perf_counter()
    recipe, run_dir = _materialize_remote_recipe(
        run_label=run_label,
        smoke=smoke,
        preflight=preflight,
        recipe_name=recipe_name,
    )
    if evaluation_source_run is not None:
        if Path(evaluation_source_run).name != evaluation_source_run:
            raise ValueError("evaluation source run must be a volume-directory name")
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
    if compile_paths_only and require_path_cache:
        raise ValueError("compile_paths_only and require_path_cache are mutually exclusive")
    if compile_paths_only:
        recipe["arguments"].update(
            {
                "compile_paths_only": True,
                "device": "cpu",
            }
        )
    if require_path_cache:
        recipe["arguments"]["require_path_cache"] = True
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

    import rdkit
    import torch

    manifest = {
        "run_label": run_label,
        "smoke": smoke,
        "preflight": preflight,
        "compile_paths_only": compile_paths_only,
        "require_path_cache": require_path_cache,
        "evaluation_source_run": evaluation_source_run,
        "evaluation_checkpoint_name": (
            evaluation_checkpoint_name if evaluation_source_run is not None else None
        ),
        "recipe_name": recipe_name,
        "recipe": recipe,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "rdkit": rdkit.__version__,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "source_sha256": _source_fingerprint(),
        "data": {
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
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
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
        and evaluation_source_run is None
        and int(recipe["arguments"].get("early_stopping_patience", 0)) > 0
    )
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
    return_code = process.wait()
    artifact_volume.commit()
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

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"training completed without metrics artifact: {metrics_path}")
    report = json.loads(metrics_path.read_text())
    summary = {
        "run_label": run_label,
        "smoke": smoke,
        "preflight": preflight,
        "recipe_name": recipe_name,
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_dir": str(run_dir),
        "model": report.get("model"),
        "generated_nonnull_smiles": report.get("generated_nonnull_smiles"),
        "valid_fraction": report.get("rollout", {}).get("valid_fraction"),
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
    run_label: str = "tree_fcd_transfer_modal_smoke",
    recipe_name: str = RECIPE_NAME,
) -> dict[str, object]:
    return _run_remote(run_label=run_label, smoke=True, recipe_name=recipe_name)


@app.function(
    image=image,
    gpu="A100",
    cpu=24.0,
    memory=65536,
    timeout=60 * 60,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
)
def preflight_stage(
    run_label: str = "tree_fcd_transfer_a100_preflight",
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
    cpu=24.0,
    memory=65536,
    timeout=60 * 60,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
)
def h100_preflight_stage(
    run_label: str = "tree_fcd_transfer_h100_preflight",
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
    cpu=16.0,
    memory=65536,
    timeout=24 * 3600,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
def compile_stage(
    run_label: str = "tree_fcd_transfer_stage1",
    recipe_name: str = RECIPE_NAME,
) -> dict[str, object]:
    """Compile restart-safe path shards without renting a GPU."""

    return _run_remote(
        run_label=run_label,
        smoke=False,
        compile_paths_only=True,
        recipe_name=recipe_name,
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

    if Path(run_label).name != run_label:
        raise ValueError("run label must be a basename")
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
    gpu="A100",
    cpu=24.0,
    memory=65536,
    timeout=24 * 3600,
    volumes={"/guacamol": guacamol_volume, "/artifacts": artifact_volume},
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
)
def train_stage(
    run_label: str = "tree_fcd_transfer_stage1",
    recipe_name: str = RECIPE_NAME,
) -> dict[str, object]:
    return _run_remote(
        run_label=run_label,
        smoke=False,
        require_path_cache=True,
        recipe_name=recipe_name,
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
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
)
def rollout_evaluate_stage(
    run_label: str,
    source_run_label: str,
    checkpoint_name: str = "checkpoint.best_so_far.pt",
    rollout_samples: int = 2000,
    disable_ring_ear: bool = False,
) -> dict[str, object]:
    """Evaluate ancestral samples without loading endpoint-conditioned teachers."""

    for value, label in (
        (run_label, "run label"),
        (source_run_label, "source run label"),
        (checkpoint_name, "checkpoint name"),
    ):
        if Path(value).name != value:
            raise ValueError(f"{label} must be a basename")
    if rollout_samples <= 0:
        raise ValueError("rollout_samples must be positive")

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
    checkpoint_snapshot = run_dir / "checkpoint.snapshot.pt"
    temporary_snapshot = checkpoint_snapshot.with_suffix(".pt.tmp")
    shutil.copyfile(source_checkpoint, temporary_snapshot)
    os.replace(temporary_snapshot, checkpoint_snapshot)
    manifest = {
        "evaluation_kind": "checkpoint_rollout_only",
        "run_label": run_label,
        "source_run_label": source_run_label,
        "source_checkpoint": str(source_checkpoint),
        "checkpoint": str(checkpoint_snapshot),
        "checkpoint_sha256": _file_sha256(checkpoint_snapshot),
        "rollout_samples": rollout_samples,
        "disabled_rule_names": ["ring_ear_insert"] if disable_ring_ear else [],
        "teacher_path_cache_loaded": False,
        "source_sha256": _source_fingerprint(),
        "data": {
            "train_sha256": _file_sha256(train_file),
            "reference_sha256": _file_sha256(reference_file),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
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
        "16",
        "--corpus-workers",
        "16",
    )
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

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        _update_source_early_eval_marker(
            source_run_label=source_run_label,
            evaluation_run_label=run_label,
            status="failed",
            details={"error": "missing metrics.json"},
        )
        raise RuntimeError("rollout evaluation completed without metrics.json")
    report = json.loads(metrics_path.read_text())
    summary = {
        "run_label": run_label,
        "artifact_dir": str(run_dir),
        "generated_nonnull_smiles": report.get("generated_nonnull_smiles"),
        "valid_fraction": report.get("rollout", {}).get("valid_fraction"),
        "selected_validation": report.get("selected_validation"),
        "fcd": report.get("molecular_quality", {}).get("frechet_chemnet_distance"),
    }
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
    cpu=0.25,
    memory=512,
    timeout=24 * 3600,
)
def pipeline_stage(
    run_label: str = "tree_fcd_transfer_stage1",
    recipe_name: str = RECIPE_NAME,
) -> dict[str, object]:
    """Run CPU preprocessing to completion before allocating the training GPU."""

    compile_result = compile_stage.remote(run_label, recipe_name)
    audit_result = audit_teacher_stage.remote(run_label, recipe_name)
    train_result = train_stage.remote(run_label, recipe_name)
    return {"compile": compile_result, "audit": audit_result, "train": train_result}


@app.local_entrypoint()
def main(
    smoke: bool = False,
    preflight: bool = False,
    h100_preflight: bool = False,
    compile_only: bool = False,
    train_only: bool = False,
    evaluate_only: bool = False,
    rollout_evaluate_only: bool = False,
    run_label: str = "tree_fcd_transfer_stage1",
    recipe_name: str = RECIPE_NAME,
    source_run_label: str = "",
    checkpoint_name: str = "checkpoint.best_so_far.pt",
    rollout_samples: int = 100,
) -> None:
    modes = sum(
        (
            smoke,
            preflight,
            h100_preflight,
            compile_only,
            train_only,
            evaluate_only,
            rollout_evaluate_only,
        )
    )
    if modes > 1:
        raise ValueError(
            "smoke, preflight, h100-preflight, compile-only, train-only, "
            "evaluate-only, and "
            "rollout-evaluate-only are exclusive"
        )
    if smoke:
        if run_label == "tree_fcd_transfer_stage1":
            run_label = "tree_fcd_transfer_modal_smoke"
        result = smoke_stage.remote(run_label, recipe_name)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if preflight:
        if run_label == "tree_fcd_transfer_stage1":
            run_label = "tree_fcd_transfer_a100_preflight"
        result = preflight_stage.remote(run_label, recipe_name)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if h100_preflight:
        if run_label == "tree_fcd_transfer_stage1":
            run_label = "tree_fcd_transfer_h100_preflight"
        result = h100_preflight_stage.remote(run_label, recipe_name)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if rollout_evaluate_only:
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
    elif compile_only:
        call = compile_stage.spawn(run_label, recipe_name)
        phase = "compile_spawned"
    elif train_only:
        call = train_stage.spawn(run_label, recipe_name)
        phase = "train_spawned"
    else:
        call = pipeline_stage.spawn(run_label, recipe_name)
        phase = "pipeline_spawned"
    print(
        json.dumps(
            {
                "phase": phase,
                "function_call_id": call.object_id,
                "recipe_name": recipe_name,
            }
        )
    )
    print("Artifacts: Modal volume compose-v4-artifacts / " + run_label)
