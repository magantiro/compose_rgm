"""Build verified QED value features as train-only reference rollouts arrive."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

LOCAL_ROOT = Path("/Users/rmaganti/compose_rgm_git/.worktrees/qed_feature_snapshot_9c6")
SOURCE_REVISION = "9c6db0b8dc16a9cbde93b21cba080f0c72c3b648"
APP_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
REMOTE_ROOT = Path("/root/compose")
ROLLOUTS = Path("/artifacts/qed_shared_train16_v1")
FEATURES = Path("/artifacts/qed_shared_train16_features_v1")
SOURCE_COUNT = 1023
BUILDER_SHA256 = "da425e6f4127d667978d6fbad4817320ccee33a52beaeb763ae527cf6a7f8f6d"

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
    .env({"PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
)

app = modal.App("compose-qed-shared-train16-features")
volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.function(image=image, cpu=1, memory=6144, timeout=300, volumes={"/artifacts": volume})
def preflight() -> dict:
    from compose_v4.experiments.qed_shared_sources import load_qed_source_roles

    roles = load_qed_source_roles(REMOTE_ROOT, REMOTE_ROOT / "experiments/qed/shared_sources.json")
    if len(roles.train) != SOURCE_COUNT:
        raise ValueError("QED train source count differs from frozen split")
    builder = REMOTE_ROOT / "src/compose_v4/experiments/qed_shared_training.py"
    if sha256(builder) != BUILDER_SHA256:
        raise ValueError("QED feature builder differs from the frozen source snapshot")
    volume.reload()
    return {
        "train_sources": len(roles.train),
        "available_rollouts": len(list(ROLLOUTS.glob("train_*.json"))),
        "source_split_sha256": roles.manifest_sha256,
        "builder_sha256": sha256(builder),
        "source_revision": SOURCE_REVISION,
        "app_sha256": APP_SHA256,
        "new_oracle_calls": 0,
    }


@app.function(
    image=image,
    cpu=1,
    memory=6144,
    timeout=2 * 3600,
    max_containers=16,
    volumes={"/artifacts": volume},
)
def build_feature(index: int) -> dict:
    from compose_v4.experiments.qed_shared_sources import load_qed_source_roles

    if not 0 <= index < SOURCE_COUNT:
        raise ValueError("QED feature source index outside frozen train role")
    volume.reload()
    roles = load_qed_source_roles(REMOTE_ROOT, REMOTE_ROOT / "experiments/qed/shared_sources.json")
    if len(roles.train) != SOURCE_COUNT:
        raise ValueError("QED train source count differs from frozen split")
    rollout = ROLLOUTS / f"train_{index:04d}.json"
    output = FEATURES / f"train_{index:04d}.npz"
    if output.exists():
        raise FileExistsError(f"QED feature output already exists: {output}")
    if not rollout.is_file():
        raise FileNotFoundError(f"QED train rollout is absent: {rollout}")
    local_output = Path("/tmp/qed_train16_feature") / output.name
    local_output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "COMPOSE_SOURCE_REVISION": SOURCE_REVISION,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    subprocess.run(
        [
            sys.executable,
            str(REMOTE_ROOT / "tools/build_qed_shared_features.py"),
            "--rollout",
            str(rollout),
            "--role",
            "train",
            "--index",
            str(index),
            "--budget-max",
            "24",
            "--time",
            "0.5",
            "--checkpoint",
            str(REMOTE_ROOT / "local_assets/fragments/r_theta_nll.pt"),
            "--output",
            str(local_output),
        ],
        env=environment,
        cwd=REMOTE_ROOT,
        check=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"QED feature output appeared during build: {output}")
    output.write_bytes(local_output.read_bytes())
    volume.commit()
    return {"index": index, "sha256": sha256(output), "new_oracle_calls": 0}


@app.function(image=image, timeout=24 * 3600, max_containers=1, volumes={"/artifacts": volume})
def drive_available() -> dict:
    completed_this_call = 0
    while True:
        volume.reload()
        rollouts = {int(path.stem.removeprefix("train_")) for path in ROLLOUTS.glob("train_*.json")}
        features = {int(path.stem.removeprefix("train_")) for path in FEATURES.glob("train_*.npz")}
        if any(not 0 <= index < SOURCE_COUNT for index in rollouts | features):
            raise ValueError("QED train artifact name is outside frozen source range")
        if not features <= rollouts:
            raise ValueError("QED feature exists without its source rollout")
        if len(features) == SOURCE_COUNT:
            return {"completed": len(features), "created_this_call": completed_this_call}
        pending = sorted(rollouts - features)
        if not pending:
            time.sleep(60)
            continue
        failures = []
        for result in build_feature.map(
            pending, order_outputs=False, return_exceptions=True, wrap_returned_exceptions=False
        ):
            if isinstance(result, Exception):
                failures.append(f"{type(result).__name__}: {result}")
            else:
                completed_this_call += 1
                if completed_this_call % 16 == 0:
                    print(f"QED train16 features created: {completed_this_call}", flush=True)
        if failures:
            raise RuntimeError(f"{len(failures)} QED feature builds failed: {failures[:3]}")


@app.local_entrypoint()
def main(mode: str = "preflight", index: int = 0) -> None:
    if mode == "preflight":
        print(json.dumps(preflight.remote(), sort_keys=True), flush=True)
    elif mode == "build_one":
        print(json.dumps(build_feature.remote(index), sort_keys=True), flush=True)
    elif mode == "drive":
        call = drive_available.spawn()
        print(f"QED train16 feature drive: {call.object_id}", flush=True)
    else:
        raise ValueError("mode must be preflight, build_one or drive")
