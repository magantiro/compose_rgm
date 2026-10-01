"""Verify the shared QED reference and an optional matched value head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compose_v4.experiments.hphi_region_features import input_dim
from compose_v4.experiments.qed_shared_sources import load_qed_source_roles
from compose_v4.experiments.qed_shared_value import QEDValueAssets

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, expected: str, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label}: SHA-256 mismatch at {path}; got {actual}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def verify_assets(root: Path, checkpoint_path: Path, value_dir: Path | None = None) -> dict:
    """Check file and model identities without deserializing model weights."""
    root = Path(root)
    reference_manifest = root / "experiments/fragments/assets.json"
    reference_assets = json.loads(reference_manifest.read_text())["assets"]
    expected_checkpoint = reference_assets["checkpoint"]["sha256"]
    checkpoint = _verified(checkpoint_path, expected_checkpoint, "shared_reference_checkpoint")
    catalog_spec = reference_assets["catalog"]
    catalog = _verified(
        root / "local_assets/fragments" / catalog_spec["path"],
        catalog_spec["sha256"],
        "shared_reference_catalog",
    )
    roles = load_qed_source_roles(root, root / "experiments/qed/shared_sources.json")
    result = {
        "schema_version": "compose.qed.asset_check.v2",
        "status": "reference_only",
        "shared_reference_checkpoint": checkpoint,
        "shared_reference_catalog": catalog,
        "reference_manifest_sha256": sha256_file(reference_manifest),
        "source_split_sha256": roles.manifest_sha256,
        "source_counts": {
            "train": len(roles.train),
            "validation": len(roles.validation),
            "test": len(roles.test),
            "excluded_training_rows": len(roles.excluded_train_indices),
        },
    }
    if value_dir is None:
        return result
    value_dir = Path(value_dir)
    assets = QEDValueAssets.from_directory(value_dir)
    result["value_assets"] = {
        "head": _verified(assets.head, assets.head_sha256, "value_head"),
        "normalization": _verified(
            assets.normalization, assets.normalization_sha256, "value_normalization"
        ),
        "metadata": _verified(assets.metadata, assets.metadata_sha256, "value_metadata"),
        "manifest_sha256": sha256_file(value_dir / "assets.json"),
    }
    metadata = json.loads(assets.metadata.read_text())
    if metadata.get("schema_version") != "compose.qed.shared_value.v1":
        raise ValueError("unsupported QED shared value metadata")
    head_reference = metadata.get("reference")
    if (
        not isinstance(head_reference, dict)
        or head_reference.get("checkpoint_sha256") != expected_checkpoint
        or head_reference.get("catalog_sha256") != catalog_spec["sha256"]
    ):
        raise ValueError("QED value head belongs to another reference checkpoint")
    if metadata.get("source_split_sha256") != roles.manifest_sha256:
        raise ValueError("QED value head belongs to another source split")
    if metadata.get("guidance_target") != {"region": [0.9, 0.4], "qualified": True}:
        raise ValueError("QED value head did not qualify on the benchmark goal")
    if metadata.get("feature_schema") != "region_features_v1":
        raise ValueError("unsupported QED value feature schema")
    if metadata.get("target_semantics") != "terminal_region":
        raise ValueError("QED value head must estimate the terminal region")
    budget_max = metadata.get("budget_max")
    if type(budget_max) is not int or budget_max < 1:
        raise ValueError("QED value head has no positive budget_max")
    normalization = json.loads(assets.normalization.read_text())
    mean = np.asarray(normalization.get("mean"), dtype=np.float64)
    scale = np.asarray(normalization.get("scale"), dtype=np.float64)
    width = input_dim(budget_max)
    if (
        mean.shape != (width,)
        or scale.shape != (width,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("QED value head normalization is invalid")
    result["status"] = "ready"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "local_assets/fragments/r_theta_nll.pt"
    )
    parser.add_argument("--value", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_assets(ROOT, args.checkpoint, args.value), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
