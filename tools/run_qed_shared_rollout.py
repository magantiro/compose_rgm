"""Write one training or validation rollout from the shared reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy
import rdkit
import torch

from compose_v4.experiments.qed_shared_reference import QEDSharedReference, SharedReferenceConfig
from compose_v4.experiments.qed_shared_rollouts import QEDRolloutConfig, rollout_source
from compose_v4.experiments.qed_shared_sources import load_qed_source_roles

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def worktree_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout) if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("train", "validation"))
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--replicates", required=True, type=int)
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
    source_entry = json.loads(split_path.read_text())[args.role]
    reference_manifest_path = ROOT / "experiments/fragments/assets.json"
    manifest = json.loads(reference_manifest_path.read_text())
    checkpoint = manifest["assets"]["checkpoint"]
    catalog = manifest["assets"]["catalog"]
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
    config = QEDRolloutConfig(args.horizon, args.replicates)
    result = rollout_source(reference, sources[args.index], args.index, config)
    result["source_role"] = args.role
    result["source_input_row_index"] = (
        roles.train_input_indices[args.index] if args.role == "train" else args.index
    )
    result["source_split_sha256"] = roles.manifest_sha256
    result["provenance"] = {
        "code_revision": git_revision(),
        "worktree_dirty": worktree_dirty(),
        "input_paths": {
            "source_file": source_entry["path"],
            "source_file_sha256": source_entry["sha256"],
            "source_split": str(split_path.relative_to(ROOT)),
            "reference_manifest": str(reference_manifest_path.relative_to(ROOT)),
            "reference_manifest_sha256": sha256(reference_manifest_path),
            "checkpoint": str(args.checkpoint.resolve()),
        },
        "code_sha256": {
            "rollout": sha256(ROOT / "src/compose_v4/experiments/qed_shared_rollouts.py"),
            "reference": sha256(ROOT / "src/compose_v4/experiments/qed_shared_reference.py"),
            "runner": sha256(Path(__file__)),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "rdkit": rdkit.__version__,
            "torch": torch.__version__,
        },
        "new_oracle_calls": 0,
        "device": "cpu",
        "model_precision": "float32",
    }
    payload = json.dumps(result, sort_keys=True, indent=2).encode() + b"\n"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite QED rollout: {output}")
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite QED rollout: {output}")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {output} ({sha256(output)})")


if __name__ == "__main__":
    main()
