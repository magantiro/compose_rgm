"""Fit a QED value head on complete source-disjoint shared-reference shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rdkit
import torch

from compose_v4.experiments.hphi_rollout import registered_regions
from compose_v4.experiments.qed_shared_fit import (
    FeatureShard,
    QEDFitConfig,
    fit_value_head,
    inspect_feature_corpus,
)
from compose_v4.experiments.qed_shared_reference import QEDSharedReference, SharedReferenceConfig
from compose_v4.experiments.qed_shared_sources import load_qed_source_roles

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def input_shard_manifest(shards: tuple[FeatureShard, ...], root: Path) -> list[dict]:
    """Record portable paths while keeping each shard bound by its hash."""
    return [
        {
            "path": shard.path.relative_to(root).as_posix(),
            "sha256": shard.sha256,
            "role": shard.role,
            "index": shard.index,
        }
        for shard in shards
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--time", required=True, type=float)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "local_assets/fragments/r_theta_nll.pt"
    )
    parser.add_argument("--budget-max", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--bellman-weight", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (
        ":16:8",
        ":4096:8",
    ):
        parser.error("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA training")
    config = QEDFitConfig(
        budget_max=args.budget_max,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bellman_weight=args.bellman_weight,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    config.validate()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite a QED value artifact: {output}")
    split_path = ROOT / "experiments/qed/shared_sources.json"
    roles = load_qed_source_roles(ROOT, split_path)
    manifest_path = ROOT / "experiments/fragments/assets.json"
    manifest = json.loads(manifest_path.read_text())
    assets = manifest["assets"]
    checkpoint, catalog = assets["checkpoint"], assets["catalog"]
    reference = QEDSharedReference.load(
        SharedReferenceConfig(
            args.checkpoint,
            checkpoint["sha256"],
            manifest["catalog_fingerprint"],
            args.time,
            catalog_path=ROOT / "local_assets/fragments" / catalog["path"],
            catalog_sha256=catalog["sha256"],
        )
    )
    shards_root = args.shards.resolve()
    shards = inspect_feature_corpus(
        shards_root,
        reference_identity=reference.identity(),
        source_split_sha256=roles.manifest_sha256,
        train_input_indices=roles.train_input_indices,
        validation_count=len(roles.validation),
        budget_max=config.budget_max,
        builder_sha256=sha256(ROOT / "src/compose_v4/experiments/qed_shared_training.py"),
        goal_regions=registered_regions(),
    )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    goal_regions = registered_regions()
    target_region = (0.90, 0.40)
    model, mean, scale, findings = fit_value_head(
        shards, config, guidance_region_index=goal_regions.index(target_region)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        head_path = staging / "head.pt"
        norm_path = staging / "normalization.json"
        metadata_path = staging / "metadata.json"
        torch.jit.save(torch.jit.script(model), str(head_path))
        _write_json(norm_path, {"mean": mean.tolist(), "scale": scale.tolist()})
        _write_json(
            metadata_path,
            {
                "schema_version": "compose.qed.shared_value.v1",
                "reference": reference.identity(),
                "reference_manifest_sha256": sha256(manifest_path),
                "source_split_sha256": roles.manifest_sha256,
                "feature_schema": "region_features_v1",
                "target_semantics": "terminal_region",
                "budget_max": config.budget_max,
                "training_configuration": asdict(config),
                "train_sources": len(roles.train),
                "validation_sources": len(roles.validation),
                "train_examples": sum(s.examples for s in shards if s.role == "train"),
                "validation_examples": sum(s.examples for s in shards if s.role == "validation"),
                "input_shards": input_shard_manifest(shards, shards_root),
                "findings": findings,
                "guidance_target": {
                    "region": list(target_region),
                    "qualified": findings["qualified_for_guidance"],
                },
                "code_revision": revision(),
                "code_sha256": {
                    "fit": sha256(ROOT / "src/compose_v4/experiments/qed_shared_fit.py"),
                    "script": sha256(Path(__file__)),
                    "feature_builder": sha256(
                        ROOT / "src/compose_v4/experiments/qed_shared_training.py"
                    ),
                },
                "software": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "rdkit": rdkit.__version__,
                    "torch": torch.__version__,
                },
                "hardware": {
                    "device": config.device,
                    "cuda_device": torch.cuda.get_device_name()
                    if config.device == "cuda"
                    else None,
                },
                "created_utc": datetime.now(UTC).isoformat(),
            },
        )
        _write_json(
            staging / "assets.json",
            {
                "schema_version": "compose.qed.shared_value_assets.v1",
                "head": {"path": "head.pt", "sha256": sha256(head_path)},
                "normalization": {"path": "normalization.json", "sha256": sha256(norm_path)},
                "metadata": {"path": "metadata.json", "sha256": sha256(metadata_path)},
            },
        )
        if output.exists():
            raise FileExistsError(f"refusing to overwrite a QED value artifact: {output}")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(
        json.dumps(
            {
                "output": str(output),
                "assets_sha256": sha256(output / "assets.json"),
                "selected_epoch": findings["selected_epoch"],
                "validation_brier": findings["selected_validation_source_mean_brier"],
                "constant_brier": findings["validation_constant_brier"],
                "guidance_qualified": findings["qualified_for_guidance"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
