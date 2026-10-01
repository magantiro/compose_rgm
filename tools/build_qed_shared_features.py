"""Build one source-local QED value-feature shard from a verified rollout."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import rdkit
import torch

from compose_v4.experiments.hphi_rollout import registered_regions
from compose_v4.experiments.qed_shared_reference import QEDSharedReference, SharedReferenceConfig
from compose_v4.experiments.qed_shared_sources import load_qed_source_roles
from compose_v4.experiments.qed_shared_training import value_examples_with_bellman

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str | None:
    declared = os.environ.get("COMPOSE_SOURCE_REVISION")
    if declared is not None and (
        len(declared) != 40 or any(character not in "0123456789abcdef" for character in declared)
    ):
        raise ValueError("COMPOSE_SOURCE_REVISION must be a lowercase full Git SHA")
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        local = None
    else:
        local = result.stdout.strip() if result.returncode == 0 else None
    if declared is not None and local is not None and declared != local:
        raise ValueError("COMPOSE_SOURCE_REVISION differs from the checked-out commit")
    return declared if declared is not None else local


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("train", "validation"))
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--budget-max", required=True, type=int)
    parser.add_argument("--time", required=True, type=float)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "local_assets/fragments/r_theta_nll.pt"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    split_path = ROOT / "experiments/qed/shared_sources.json"
    roles = load_qed_source_roles(ROOT, split_path)
    sources = roles.train if args.role == "train" else roles.validation
    if not 0 <= args.index < len(sources):
        parser.error(f"{args.role} index must be in [0, {len(sources) - 1}]")
    rollout_path = args.rollout.resolve()
    rollout_bytes = rollout_path.read_bytes()
    rollout = json.loads(rollout_bytes)
    if rollout.get("source_role") != args.role or rollout.get("source_index") != args.index:
        raise ValueError(
            "QED rollout role or source index does not match the requested feature shard"
        )
    if rollout.get("source_original") != sources[args.index]:
        raise ValueError("QED rollout source differs from the frozen source role")
    if rollout.get("source_split_sha256") != roles.manifest_sha256:
        raise ValueError("QED rollout belongs to another source split")
    expected_input_index = (
        roles.train_input_indices[args.index] if args.role == "train" else args.index
    )
    if rollout.get("source_input_row_index") != expected_input_index:
        raise ValueError("QED rollout input row differs from the frozen source role")

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
    values = value_examples_with_bellman(reference, rollout, budget_max=args.budget_max)
    features, labels, region_indices = (
        values.features,
        values.labels,
        values.region_indices,
    )
    if len(features) == 0:
        raise ValueError("QED source produced no positive-budget value examples")
    portable_rollout_path = (
        rollout_path.relative_to(ROOT).as_posix()
        if rollout_path.is_relative_to(ROOT)
        else rollout_path.name
    )
    metadata = {
        "schema_version": "compose.qed.shared_features.v4",
        "role": args.role,
        "source_index": args.index,
        "source_input_row_index": expected_input_index,
        "source_split_sha256": roles.manifest_sha256,
        "rollout_path": portable_rollout_path,
        "rollout_sha256": hashlib.sha256(rollout_bytes).hexdigest(),
        "rollout_schema": rollout["schema_version"],
        "import_provenance": rollout.get("import_provenance"),
        "reference": reference.identity(),
        "reference_manifest_sha256": sha256(manifest_path),
        "budget_max": args.budget_max,
        "feature_schema": "region_features_v1",
        "target_semantics": "terminal_region",
        "goal_regions": [list(region) for region in registered_regions()],
        "examples": len(labels),
        "positive_labels": int(labels.sum()),
        "region_examples": [
            int(np.sum(region_indices == i)) for i in range(len(registered_regions()))
        ],
        "region_positive_labels": [
            int(np.sum(labels[region_indices == i])) for i in range(len(registered_regions()))
        ],
        "bellman_pairs": int(
            np.sum((values.next_row_indices >= 0) | (values.next_terminal_targets >= 0))
        ),
        "bellman_boundary_pairs": int(np.sum(values.next_terminal_targets >= 0)),
        "builder_sha256": sha256(ROOT / "src/compose_v4/experiments/qed_shared_training.py"),
        "code_revision": git_revision(),
        "code_sha256": {
            "script": sha256(Path(__file__)),
            "reference": sha256(ROOT / "src/compose_v4/experiments/qed_shared_reference.py"),
            "features": sha256(ROOT / "src/compose_v4/experiments/hphi_region_features.py"),
            "regions": sha256(ROOT / "src/compose_v4/experiments/hphi_rollout.py"),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdkit.__version__,
            "torch": torch.__version__,
        },
        "device": "cpu",
        "model_precision": "float32",
    }
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        features=features,
        labels=labels,
        region_indices=region_indices,
        next_row_indices=values.next_row_indices,
        next_terminal_targets=values.next_terminal_targets,
        metadata=np.str_(json.dumps(metadata, sort_keys=True)),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite QED feature shard: {output}")
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(buffer.getvalue())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite QED feature shard: {output}")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {output} ({sha256(output)}), {len(labels)} examples")


if __name__ == "__main__":
    main()
