"""Evaluate one frozen QED test source with the shared molecular reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rdkit
import torch

from compose_v4.experiments.qed_shared_reference import QEDSharedReference, SharedReferenceConfig
from compose_v4.experiments.qed_shared_smc import QEDSMCConfig, run_source
from compose_v4.experiments.qed_shared_sources import load_qed_source_roles
from compose_v4.experiments.qed_shared_value import QEDSharedValueHead, QEDValueAssets

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--value", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "local_assets/fragments/r_theta_nll.pt"
    )
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=8)
    args = parser.parse_args()

    roles = load_qed_source_roles(ROOT, ROOT / "experiments/qed/shared_sources.json")
    if not 0 <= args.index < len(roles.test):
        parser.error(f"test index must be in [0, {len(roles.test) - 1}]")
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
    assets = QEDValueAssets.from_directory(args.value.resolve())
    value_head = QEDSharedValueHead.load(
        reference, assets, expected_source_split_sha256=roles.manifest_sha256
    )
    config = QEDSMCConfig(
        horizon=args.horizon,
        particles=args.particles,
        candidates=args.candidates,
    )
    result = run_source(reference, value_head, roles.test[args.index], config)
    record = {
        "schema_version": "compose.qed.shared_result.v1",
        "test_index": args.index,
        "source_split_sha256": roles.manifest_sha256,
        "reference_manifest_sha256": sha256(manifest_path),
        "value_assets_manifest_sha256": sha256(args.value.resolve() / "assets.json"),
        "value_metadata_sha256": assets.metadata_sha256,
        "code_revision": revision(),
        "code_sha256": {
            "runner": sha256(Path(__file__)),
            "smc": sha256(ROOT / "src/compose_v4/experiments/qed_shared_smc.py"),
            "reference": sha256(ROOT / "src/compose_v4/experiments/qed_shared_reference.py"),
            "value": sha256(ROOT / "src/compose_v4/experiments/qed_shared_value.py"),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdkit.__version__,
            "torch": torch.__version__,
        },
        "hardware": {"device": "cpu", "model_precision": "float32"},
        "created_utc": datetime.now(UTC).isoformat(),
        "result": result,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite QED source result: {output}")
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite QED source result: {output}")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {output} ({sha256(output)}) success={result['success']}")


if __name__ == "__main__":
    main()
