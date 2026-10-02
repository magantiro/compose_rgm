"""Fit one QED terminal-value head after every frozen source shard is complete."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

LOCAL_ROOT = Path("/Users/rmaganti/compose_rgm_git/.worktrees/qed_encoder_fastpath")
SOURCE_REVISION = "98d8345dab6fe404786305adcfc9a1801e71ddee"
SOURCE_SPLIT_SHA256 = "7e41720b5b7aed7ba3cb08cbdc9508422212238580af37058c68486e811a244c"
BUILDER_SHA256 = "57b85ba7bba7ea1362ebe7d4cb921c3f65f4c5ba690b0d38f32b945e9e36a51b"
APP_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
REMOTE_ROOT = Path("/root/compose")
FEATURES = Path("/artifacts/qed_shared_features_fast_v1")
OUTPUT = Path("/artifacts/qed_shared_value_attempt_v4")
LAUNCH_RECEIPT = Path("/artifacts/qed_shared_value_attempt_v4_launch.json")
SOURCE_COUNTS = {"train": 1023, "validation": 128}

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

app = modal.App("compose-qed-shared-fast-fit-watch")
volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


def _expected(role: str, count: int) -> set[str]:
    return {f"{role}_{index:04d}.npz" for index in range(count)}


def _inventory() -> dict:
    observed = {path.name for path in FEATURES.glob("*.npz")}
    expected = set().union(*(_expected(role, count) for role, count in SOURCE_COUNTS.items()))
    if observed - expected:
        raise ValueError("QED feature namespace contains an unexpected source index")
    return {
        "train": len(observed & _expected("train", SOURCE_COUNTS["train"])),
        "validation": len(observed & _expected("validation", SOURCE_COUNTS["validation"])),
        "complete": observed == expected,
    }


@app.function(image=image, cpu=1, memory=4096, timeout=300, volumes={"/artifacts": volume})
def preflight() -> dict:
    from compose_v4.experiments.qed_shared_sources import load_qed_source_roles

    roles = load_qed_source_roles(REMOTE_ROOT, REMOTE_ROOT / "experiments/qed/shared_sources.json")
    if (
        len(roles.train) != SOURCE_COUNTS["train"]
        or len(roles.validation) != SOURCE_COUNTS["validation"]
        or roles.manifest_sha256 != SOURCE_SPLIT_SHA256
    ):
        raise ValueError("QED source roles differ from the frozen split")
    builder = REMOTE_ROOT / "src/compose_v4/experiments/qed_shared_training.py"
    if hashlib.sha256(builder.read_bytes()).hexdigest() != BUILDER_SHA256:
        raise ValueError("QED builder differs from the frozen feature corpus")
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
    if not _inventory()["complete"]:
        raise ValueError("QED fit refused an incomplete source-disjoint corpus")
    if OUTPUT.exists():
        if (OUTPUT / "assets.json").is_file():
            return {"status": "already_complete", "output": str(OUTPUT)}
        raise FileExistsError(f"QED fit output is partial: {OUTPUT}")
    scratch = Path("/tmp/qed_shared_fast_fit")
    if scratch.exists():
        raise FileExistsError(f"QED fit scratch already exists: {scratch}")
    shards = scratch / "shards"
    shards.mkdir(parents=True)
    for role, count in SOURCE_COUNTS.items():
        for index in range(count):
            name = f"{role}_{index:04d}.npz"
            os.symlink(FEATURES / name, shards / name)
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
        "train_sources": SOURCE_COUNTS["train"],
        "validation_sources": SOURCE_COUNTS["validation"],
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
