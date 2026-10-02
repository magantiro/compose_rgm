"""Build source-disjoint QED value features with the parity-tested encoder path."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

LOCAL_ROOT = Path("/Users/rmaganti/compose_rgm_git/.worktrees/qed_encoder_fastpath")
SOURCE_REVISION = "98d8345dab6fe404786305adcfc9a1801e71ddee"
REMOTE_ROOT = Path("/root/compose")
TRAIN_ROLLOUTS = Path("/artifacts/qed_shared_train16_v1")
VALIDATION_ROLLOUTS = Path("/artifacts/qed_shared_validation_v1")
FEATURES = Path("/artifacts/qed_shared_features_fast_v1")
SOURCE_COUNTS = {"train": 1023, "validation": 128}
APP_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

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
            "COMPOSE_SOURCE_REVISION": SOURCE_REVISION,
        }
    )
)

app = modal.App("compose-qed-shared-features-fast")
volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


def _rollout_path(role: str, index: int) -> Path:
    root = TRAIN_ROLLOUTS if role == "train" else VALIDATION_ROLLOUTS
    return root / f"{role}_{index:04d}.json"


def _feature_path(role: str, index: int) -> Path:
    return FEATURES / f"{role}_{index:04d}.npz"


def _verify_existing(output: Path, role: str, index: int, rollout_sha256: str) -> str:
    import numpy as np

    raw = output.read_bytes()
    with np.load(output, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        examples = len(archive["labels"])
    expected = {
        "schema_version": "compose.qed.shared_features.v4",
        "role": role,
        "source_index": index,
        "rollout_sha256": rollout_sha256,
        "code_revision": SOURCE_REVISION,
        "examples": examples,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"QED feature output has a different identity: {output}")
    return hashlib.sha256(raw).hexdigest()


@app.function(
    image=image,
    cpu=1,
    memory=6144,
    timeout=2 * 3600,
    max_containers=16,
    volumes={"/artifacts": volume},
)
def build_feature(role: str, index: int) -> dict:
    if role not in SOURCE_COUNTS or not 0 <= index < SOURCE_COUNTS[role]:
        raise ValueError("QED feature role or source index is outside the frozen split")
    volume.reload()
    rollout = _rollout_path(role, index)
    if not rollout.is_file():
        raise FileNotFoundError(f"QED source rollout is absent: {rollout}")
    rollout_sha256 = hashlib.sha256(rollout.read_bytes()).hexdigest()
    output = _feature_path(role, index)
    if output.is_file():
        return {
            "role": role,
            "index": index,
            "sha256": _verify_existing(output, role, index, rollout_sha256),
            "status": "existing",
        }
    local_output = Path("/tmp/qed_fast_feature") / output.name
    local_output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(REMOTE_ROOT / "tools/build_qed_shared_features.py"),
            "--rollout",
            str(rollout),
            "--role",
            role,
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
        cwd=REMOTE_ROOT,
        check=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(f".{output.name}.pending")
    if pending.exists() or output.exists():
        raise FileExistsError(f"QED feature output already exists: {output}")
    pending.write_bytes(local_output.read_bytes())
    if output.exists():
        pending.unlink()
        raise FileExistsError(f"QED feature output appeared during build: {output}")
    os.rename(pending, output)
    digest = _verify_existing(output, role, index, rollout_sha256)
    volume.commit()
    return {"role": role, "index": index, "sha256": digest, "status": "created"}


@app.function(
    image=image,
    cpu=1,
    memory=4096,
    timeout=2 * 3600,
    max_containers=1,
    schedule=modal.Period(minutes=10),
    volumes={"/artifacts": volume},
)
def trigger() -> dict:
    volume.reload()
    pending = []
    counts = {}
    for role, total in SOURCE_COUNTS.items():
        rollout_root = TRAIN_ROLLOUTS if role == "train" else VALIDATION_ROLLOUTS
        rollouts = {int(path.stem.removeprefix(f"{role}_")) for path in rollout_root.glob(f"{role}_*.json")}
        features = {int(path.stem.removeprefix(f"{role}_")) for path in FEATURES.glob(f"{role}_*.npz")}
        if any(not 0 <= index < total for index in rollouts | features) or not features <= rollouts:
            raise ValueError("QED feature inventory disagrees with the frozen source split")
        counts[role] = {"rollouts": len(rollouts), "features": len(features)}
        pending.extend((role, index) for index in sorted(rollouts - features))
    if not pending:
        return {"counts": counts, "new_features": 0, "app_sha256": APP_SHA256}
    failures = []
    created = 0
    for result in build_feature.starmap(
        pending[:64], order_outputs=False, return_exceptions=True, wrap_returned_exceptions=False
    ):
        if isinstance(result, Exception):
            failures.append(f"{type(result).__name__}: {result}")
        elif result["status"] == "created":
            created += 1
    if failures:
        raise RuntimeError(f"{len(failures)} QED feature builds failed: {failures[:3]}")
    return {"counts": counts, "new_features": created, "app_sha256": APP_SHA256}


@app.local_entrypoint()
def main(mode: str = "trigger", role: str = "train", index: int = 0) -> None:
    if mode == "trigger":
        print(json.dumps(trigger.remote(), sort_keys=True), flush=True)
    elif mode == "build_one":
        print(json.dumps(build_feature.remote(role, index), sort_keys=True), flush=True)
    else:
        raise ValueError("mode must be trigger or build_one")
