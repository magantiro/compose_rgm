"""Generate a source-disjoint, sixteen-replicate QED training corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

LOCAL_ROOT = Path("/Users/rmaganti/compose_rgm_git/.worktrees/qed_feature_snapshot_9c6")
SOURCE_REVISION = "9c6db0b8dc16a9cbde93b21cba080f0c72c3b648"
APP_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
REMOTE_ROOT = Path("/root/compose")
OUTPUT = Path("/artifacts/qed_shared_train16_v1")
HORIZON = 24
REPLICATES = 16
SOURCE_COUNT = 1023

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
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src"), "OMP_NUM_THREADS": "1"})
)

app = modal.App("compose-qed-shared-train16")
volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.function(image=image, cpu=1, memory=6144, timeout=300, volumes={"/artifacts": volume})
def preflight() -> dict:
    from compose_v4.experiments.qed_shared_reference import (
        QEDSharedReference,
        SharedReferenceConfig,
    )
    from compose_v4.experiments.qed_shared_sources import load_qed_source_roles

    roles = load_qed_source_roles(REMOTE_ROOT, REMOTE_ROOT / "experiments/qed/shared_sources.json")
    if len(roles.train) != SOURCE_COUNT or len(roles.validation) != 128 or len(roles.test) != 800:
        raise ValueError("QED source split counts differ from the frozen manifest")
    manifest = json.loads((REMOTE_ROOT / "experiments/fragments/assets.json").read_text())
    reference = QEDSharedReference.load(
        SharedReferenceConfig(
            REMOTE_ROOT / "local_assets/fragments/r_theta_nll.pt",
            manifest["assets"]["checkpoint"]["sha256"],
            manifest["catalog_fingerprint"],
            0.5,
            catalog_path=REMOTE_ROOT / "local_assets/fragments/catalog.json",
            catalog_sha256=manifest["assets"]["catalog"]["sha256"],
        )
    )
    return {
        "train_sources": len(roles.train),
        "validation_sources": len(roles.validation),
        "test_sources": len(roles.test),
        "excluded_train_rows": list(roles.excluded_train_indices),
        "source_split_sha256": roles.manifest_sha256,
        "reference": reference.identity(),
        "source_revision": SOURCE_REVISION,
        "app_sha256": APP_SHA256,
        "horizon": HORIZON,
        "replicates": REPLICATES,
        "new_oracle_calls": 0,
    }


@app.function(
    image=image,
    cpu=1,
    memory=6144,
    timeout=24 * 3600,
    max_containers=128,
    volumes={"/artifacts": volume},
)
def run_source(index: int) -> dict:
    from compose_v4.experiments.qed_shared_reference import (
        QEDSharedReference,
        SharedReferenceConfig,
    )
    from compose_v4.experiments.qed_shared_rollouts import QEDRolloutConfig, rollout_source
    from compose_v4.experiments.qed_shared_sources import load_qed_source_roles

    volume.reload()
    roles = load_qed_source_roles(REMOTE_ROOT, REMOTE_ROOT / "experiments/qed/shared_sources.json")
    if not 0 <= index < SOURCE_COUNT or len(roles.train) != SOURCE_COUNT:
        raise ValueError("QED train source index or frozen source count is invalid")
    manifest_path = REMOTE_ROOT / "experiments/fragments/assets.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = REMOTE_ROOT / "local_assets/fragments/r_theta_nll.pt"
    catalog = REMOTE_ROOT / "local_assets/fragments/catalog.json"
    reference = QEDSharedReference.load(
        SharedReferenceConfig(
            checkpoint,
            manifest["assets"]["checkpoint"]["sha256"],
            manifest["catalog_fingerprint"],
            0.5,
            catalog_path=catalog,
            catalog_sha256=manifest["assets"]["catalog"]["sha256"],
        )
    )
    output = OUTPUT / f"train_{index:04d}.json"
    expected = {
        "schema_version": "compose.qed.shared_rollout.v1",
        "source_index": index,
        "source_original": roles.train[index],
        "source_role": "train",
        "source_input_row_index": roles.train_input_indices[index],
        "source_split_sha256": roles.manifest_sha256,
        "reference": reference.identity(),
        "configuration": {"horizon": HORIZON, "replicates": REPLICATES},
    }
    if output.is_file():
        existing = json.loads(output.read_text())
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError(f"existing QED train rollout has a different identity: {output}")
        if len(existing.get("trajectories", ())) != REPLICATES:
            raise ValueError(f"existing QED train rollout is incomplete: {output}")
        return {"index": index, "sha256": sha256(output), "status": "existing"}

    result = rollout_source(
        reference, roles.train[index], index, QEDRolloutConfig(HORIZON, REPLICATES)
    )
    result.update(
        {
            "source_role": "train",
            "source_input_row_index": roles.train_input_indices[index],
            "source_split_sha256": roles.manifest_sha256,
            "provenance": {
                "source_split_sha256": roles.manifest_sha256,
                "reference_manifest_sha256": sha256(manifest_path),
                "checkpoint_sha256": sha256(checkpoint),
                "catalog_sha256": sha256(catalog),
                "rollout_code_sha256": sha256(
                    REMOTE_ROOT / "src/compose_v4/experiments/qed_shared_rollouts.py"
                ),
                "app_sha256": APP_SHA256,
                "source_revision": SOURCE_REVISION,
                "new_oracle_calls": 0,
            },
        }
    )
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("generated QED train rollout has a different identity")
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise FileExistsError(f"different QED train rollout exists: {output}")
    else:
        output.write_bytes(payload)
    volume.commit()
    return {"index": index, "sha256": sha256(output), "status": "created"}


@app.function(image=image, timeout=24 * 3600, max_containers=1)
def drive(start: int, limit: int) -> dict:
    if not 0 <= start < limit <= SOURCE_COUNT:
        raise ValueError("QED train drive range is outside the frozen source split")
    failures = []
    completed = 0
    for result in run_source.map(range(start, limit), order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            failures.append(f"{type(result).__name__}: {result}")
        else:
            completed += 1
            if completed % 32 == 0:
                print(f"QED train rollouts: {completed}/{limit - start}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} QED train rollouts failed: {failures[:3]}")
    return {"completed": completed, "start": start, "limit": limit}


@app.local_entrypoint()
def main(mode: str = "preflight", start: int = 0, limit: int = SOURCE_COUNT) -> None:
    if mode == "preflight":
        print(json.dumps(preflight.remote(), sort_keys=True), flush=True)
    elif mode == "run_one":
        print(json.dumps(run_source.remote(start), sort_keys=True), flush=True)
    elif mode == "drive":
        call = drive.spawn(start, limit)
        print(f"QED train16 drive: {call.object_id}", flush=True)
    else:
        raise ValueError("mode must be preflight, run_one or drive")
