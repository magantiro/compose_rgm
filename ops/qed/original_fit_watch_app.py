"""Launch one QED terminal-value fit after the frozen feature corpus is complete."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

LOCAL_ROOT = Path("/Users/rmaganti/compose_rgm_git/.worktrees/qed_feature_snapshot_9c6")
SOURCE_REVISION = "9c6db0b8dc16a9cbde93b21cba080f0c72c3b648"
SOURCE_SPLIT_SHA256 = "7e41720b5b7aed7ba3cb08cbdc9508422212238580af37058c68486e811a244c"
BUILDER_SHA256 = "da425e6f4127d667978d6fbad4817320ccee33a52beaeb763ae527cf6a7f8f6d"
APP_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
REMOTE_ROOT = Path("/root/compose")
TRAIN_FEATURES = Path("/artifacts/qed_shared_train16_features_v1")
VALIDATION_FEATURES = Path("/artifacts/qed_shared_validation_features_v1")
OUTPUT = Path("/artifacts/qed_shared_value_attempt_v3")
LAUNCH_RECEIPT = Path("/artifacts/qed_shared_value_attempt_v3_launch.json")
TRAIN_COUNT = 1023
VALIDATION_COUNT = 128

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
        "torch==2.4.0",
    )
    .add_local_dir(LOCAL_ROOT / "src", str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(LOCAL_ROOT / "tools", str(REMOTE_ROOT / "tools"), copy=True)
    .add_local_file(
        LOCAL_ROOT / "experiments/qed/shared_sources.json",
        str(REMOTE_ROOT / "experiments/qed/shared_sources.json"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "experiments/fragments/assets.json",
        str(REMOTE_ROOT / "experiments/fragments/assets.json"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "data/jin/hphi_train_1024.txt",
        str(REMOTE_ROOT / "data/jin/hphi_train_1024.txt"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "data/jin/hphi_valid_128.txt",
        str(REMOTE_ROOT / "data/jin/hphi_valid_128.txt"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "data/jin/qed_test.txt",
        str(REMOTE_ROOT / "data/jin/qed_test.txt"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "local_assets/fragments/r_theta_nll.pt",
        str(REMOTE_ROOT / "local_assets/fragments/r_theta_nll.pt"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "local_assets/fragments/catalog.json",
        str(REMOTE_ROOT / "local_assets/fragments/catalog.json"),
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT}",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "COMPOSE_SOURCE_REVISION": SOURCE_REVISION,
        }
    )
)

app = modal.App("compose-qed-shared-train16-fit-watch")
volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


def _names(directory: Path, prefix: str) -> set[str]:
    return {path.name for path in directory.glob(f"{prefix}_*.npz")}


def _expected(prefix: str, count: int) -> set[str]:
    return {f"{prefix}_{index:04d}.npz" for index in range(count)}


def _inventory() -> dict:
    train = _names(TRAIN_FEATURES, "train")
    validation = _names(VALIDATION_FEATURES, "validation")
    expected_train = _expected("train", TRAIN_COUNT)
    expected_validation = _expected("validation", VALIDATION_COUNT)
    if train - expected_train or validation - expected_validation:
        raise ValueError("QED feature namespace contains an unexpected source index")
    return {
        "train": len(train),
        "validation": len(validation),
        "complete": train == expected_train and validation == expected_validation,
    }


@app.function(image=image, cpu=1, memory=4096, timeout=300, volumes={"/artifacts": volume})
def preflight() -> dict:
    from compose_v4.experiments.qed_shared_sources import load_qed_source_roles

    roles = load_qed_source_roles(REMOTE_ROOT, REMOTE_ROOT / "experiments/qed/shared_sources.json")
    if (
        len(roles.train) != TRAIN_COUNT
        or len(roles.validation) != VALIDATION_COUNT
        or roles.manifest_sha256 != SOURCE_SPLIT_SHA256
    ):
        raise ValueError("QED fit source roles differ from the frozen split")
    builder = REMOTE_ROOT / "src/compose_v4/experiments/qed_shared_training.py"
    if hashlib.sha256(builder.read_bytes()).hexdigest() != BUILDER_SHA256:
        raise ValueError("QED fit builder differs from the frozen feature corpus")
    volume.reload()
    return {
        **_inventory(),
        "source_revision": SOURCE_REVISION,
        "source_split_sha256": SOURCE_SPLIT_SHA256,
        "builder_sha256": BUILDER_SHA256,
        "app_sha256": APP_SHA256,
        "new_oracle_calls": 0,
    }


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=24 * 3600,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def fit_value() -> dict:
    volume.reload()
    inventory = _inventory()
    if not inventory["complete"]:
        raise ValueError("QED fit refused an incomplete train/validation corpus")
    if OUTPUT.exists():
        if (OUTPUT / "assets.json").is_file():
            return {"status": "already_complete", "output": str(OUTPUT)}
        raise FileExistsError(f"QED fit output is partial: {OUTPUT}")
    scratch = Path("/tmp/qed_shared_train16_fit")
    if scratch.exists():
        raise FileExistsError(f"QED fit scratch already exists: {scratch}")
    shards = scratch / "shards"
    shards.mkdir(parents=True)
    for directory, prefix, count in (
        (TRAIN_FEATURES, "train", TRAIN_COUNT),
        (VALIDATION_FEATURES, "validation", VALIDATION_COUNT),
    ):
        for index in range(count):
            name = f"{prefix}_{index:04d}.npz"
            os.symlink(directory / name, shards / name)
    environment = os.environ.copy()
    subprocess.run(
        [
            sys.executable,
            str(REMOTE_ROOT / "tools/fit_qed_shared_value.py"),
            "--shards",
            str(shards),
            "--time",
            "0.5",
            "--checkpoint",
            str(REMOTE_ROOT / "local_assets/fragments/r_theta_nll.pt"),
            "--budget-max",
            "24",
            "--device",
            "cuda",
            "--output",
            str(scratch / "value"),
        ],
        cwd=REMOTE_ROOT,
        env=environment,
        check=True,
    )
    pending = OUTPUT.parent / f".{OUTPUT.name}.pending"
    if pending.exists() or OUTPUT.exists():
        raise FileExistsError("QED fit output or pending directory already exists")
    shutil.copytree(scratch / "value", pending)
    os.rename(pending, OUTPUT)
    volume.commit()
    metadata = json.loads((OUTPUT / "metadata.json").read_text())
    return {
        "status": "complete",
        "output": str(OUTPUT),
        "guidance_qualified": metadata["guidance_target"]["qualified"],
        "selected_epoch": metadata["findings"]["selected_epoch"],
        "new_oracle_calls": 0,
    }


@app.function(
    image=image,
    cpu=1,
    memory=4096,
    timeout=300,
    max_containers=1,
    schedule=modal.Period(minutes=30),
    volumes={"/artifacts": volume},
)
def trigger_when_complete() -> dict:
    volume.reload()
    inventory = _inventory()
    if OUTPUT.exists():
        if not (OUTPUT / "assets.json").is_file():
            raise FileExistsError(f"QED fit output is partial: {OUTPUT}")
        return {**inventory, "status": "complete"}
    if LAUNCH_RECEIPT.exists():
        return {**inventory, "status": "already_launched"}
    if not inventory["complete"]:
        return {**inventory, "status": "waiting"}
    launch = {
        "schema_version": "compose.qed.shared_fit_launch.v1",
        "source_revision": SOURCE_REVISION,
        "source_split_sha256": SOURCE_SPLIT_SHA256,
        "builder_sha256": BUILDER_SHA256,
        "app_sha256": APP_SHA256,
        "train_sources": TRAIN_COUNT,
        "validation_sources": VALIDATION_COUNT,
        "new_oracle_calls": 0,
    }
    with LAUNCH_RECEIPT.open("x") as stream:
        json.dump(launch, stream, sort_keys=True)
        stream.write("\n")
    volume.commit()
    call = fit_value.spawn()
    return {**inventory, "status": "launched", "function_call_id": call.object_id}


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(preflight.remote(), sort_keys=True), flush=True)
