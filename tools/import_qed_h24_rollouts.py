"""Import hash-verified H24 trajectories as terminal-label QED training inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import rdkit

from compose_v4.experiments.qed_shared_legacy_import import imported_train_rollouts
from compose_v4.experiments.qed_shared_reference import QEDSharedReference, SharedReferenceConfig
from compose_v4.experiments.qed_shared_sources import load_qed_source_roles
from compose_v4.experiments.qed_shared_training import LEGACY_H24_CORPUS_SHA256

ROOT = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--training-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "local_assets/fragments/r_theta_nll.pt"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite QED imported rollouts: {output}")
    raw = args.corpus.read_bytes()
    if _sha256(raw) != LEGACY_H24_CORPUS_SHA256:
        raise ValueError(f"H24 corpus hash mismatch: {args.corpus}")
    record_raw = args.training_record.read_bytes()
    record = json.loads(record_raw)
    manifest_path = ROOT / "experiments/fragments/assets.json"
    manifest = json.loads(manifest_path.read_text())
    assets = manifest["assets"]
    if (
        record.get("corpus_sha256") != LEGACY_H24_CORPUS_SHA256
        or record.get("r_theta_sha256") != assets["checkpoint"]["sha256"]
        or record.get("r_theta_retrained") is not False
        or record.get("target") != "finite-budget HITTING reachability; boundary h=1 enforced"
    ):
        raise ValueError("H24 training record does not bind the shared reference and corpus")
    split_path = ROOT / "experiments/qed/shared_sources.json"
    roles = load_qed_source_roles(ROOT, split_path)
    split = json.loads(split_path.read_text())
    input_raw = (ROOT / split["train"]["path"]).read_bytes()
    if _sha256(input_raw) != split["train"]["sha256"]:
        raise ValueError("QED original training source hash mismatch")
    train_input = tuple(line.strip() for line in input_raw.decode().splitlines() if line.strip())
    reference = QEDSharedReference.load(
        SharedReferenceConfig(
            args.checkpoint,
            assets["checkpoint"]["sha256"],
            manifest["catalog_fingerprint"],
            0.5,
            catalog_path=ROOT / "local_assets/fragments" / assets["catalog"]["path"],
            catalog_sha256=assets["catalog"]["sha256"],
        )
    )
    corpus = json.loads(gzip.decompress(raw))
    imported = imported_train_rollouts(corpus, roles, train_input, reference.identity())
    terminal_hits = sum(
        trajectory["path"][-1]["qed"] >= 0.9
        and trajectory["path"][-1]["similarity_to_source"] >= 0.4
        for rollout in imported
        for trajectory in rollout["trajectories"]
    )
    revision_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for rollout in imported:
            path = staging / f"train_{rollout['source_index']:04d}.json"
            path.write_text(json.dumps(rollout, sort_keys=True, separators=(",", ":")) + "\n")
        (staging / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "compose.qed.h24_import.v1",
                    "corpus_sha256": LEGACY_H24_CORPUS_SHA256,
                    "training_record_sha256": _sha256(record_raw),
                    "source_split_sha256": roles.manifest_sha256,
                    "source_input_sha256": _sha256(input_raw),
                    "reference": reference.identity(),
                    "configuration": {"horizon": 24, "replicates": 2, "goal": [0.9, 0.4]},
                    "excluded_train_input_indices": list(roles.excluded_train_indices),
                    "train_sources": len(imported),
                    "terminal_goal_hits": terminal_hits,
                    "software": {"python": platform.python_version(), "rdkit": rdkit.__version__},
                    "code_revision": revision_result.stdout.strip()
                    if revision_result.returncode == 0
                    else None,
                    "code_sha256": {
                        "script": _sha256(Path(__file__).read_bytes()),
                        "importer": _sha256(
                            (
                                ROOT / "src/compose_v4/experiments/qed_shared_legacy_import.py"
                            ).read_bytes()
                        ),
                    },
                    "output_files": {
                        path.name: _sha256(path.read_bytes())
                        for path in sorted(staging.glob("train_*.json"))
                    },
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        if output.exists():
            raise FileExistsError(f"refusing to overwrite QED imported rollouts: {output}")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"output": str(output), "train_sources": len(imported)}, sort_keys=True))


if __name__ == "__main__":
    main()
