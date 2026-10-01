"""Build source-local terminal-value features from verified H24 trajectories."""

from __future__ import annotations

import argparse
import gzip
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

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.hphi_rollout import registered_regions
from compose_v4.experiments.qed_shared_reference import QEDSharedReference, SharedReferenceConfig
from compose_v4.experiments.qed_shared_sources import load_qed_source_roles
from compose_v4.experiments.qed_shared_training import (
    LEGACY_H24_CORPUS_SHA256,
    ValueExamples,
    value_examples_with_bellman,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    payload = (json.dumps(data, sort_keys=True, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if path.exists():
            if path.read_bytes() != payload:
                raise FileExistsError(f"refusing to change existing QED embedding record: {path}")
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _embeddings(directory: Path, reference: QEDSharedReference) -> tuple[dict, dict]:
    files = sorted(directory.glob("shard_*.json.gz"))
    if [path.name for path in files] != [f"shard_{i:03d}.json.gz" for i in range(80)]:
        raise ValueError("QED H24 embedding set must contain exactly shards 000 through 079")
    mapping: dict[str, np.ndarray] = {}
    hashes = {}
    parity = []
    for shard_index, path in enumerate(files):
        raw = path.read_bytes()
        hashes[path.name] = _sha256(raw)
        rows = json.loads(gzip.decompress(raw))
        if not isinstance(rows, dict) or not rows:
            raise ValueError(f"QED embedding shard is empty or invalid: {path}")
        if shard_index % 10 == 0:
            smiles = min(rows)
            state = pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)
            expected = reference.encode(state)
            observed = np.asarray(rows[smiles], dtype=np.float64)
            difference = float(np.max(np.abs(expected - observed)))
            if not np.isfinite(difference) or difference > 1e-6:
                raise ValueError(f"QED embedding parity failed for {path}: {difference}")
            parity.append({"shard": path.name, "smiles": smiles, "max_abs_error": difference})
        for smiles, values in rows.items():
            embedding = np.asarray(values, dtype=np.float64)
            if embedding.shape != (256,) or not np.isfinite(embedding).all():
                raise ValueError(f"QED embedding is invalid in {path}: {smiles!r}")
            if smiles in mapping:
                raise ValueError(f"QED embedding key occurs in more than one shard: {smiles!r}")
            mapping[smiles] = embedding
    if len(mapping) != 39663:
        raise ValueError(
            f"QED H24 embedding count differs from the training record: {len(mapping)}"
        )
    return mapping, {"shards_sha256": hashes, "states": len(mapping), "parity": parity}


def _feature_bytes(
    rollout: dict, reference: QEDSharedReference, lookup: dict[str, np.ndarray]
) -> tuple[ValueExamples, dict]:
    values = value_examples_with_bellman(reference, rollout, budget_max=24, embedding_lookup=lookup)
    if len(values.labels) == 0:
        raise ValueError("QED imported source has no positive-budget examples")
    regions = registered_regions()
    region_indices = values.region_indices
    labels = values.labels
    details = {
        "examples": len(labels),
        "positive_labels": int(labels.sum()),
        "region_examples": [int(np.sum(region_indices == i)) for i in range(len(regions))],
        "region_positive_labels": [
            int(np.sum(labels[region_indices == i])) for i in range(len(regions))
        ],
        "bellman_pairs": int(
            np.sum((values.next_row_indices >= 0) | (values.next_terminal_targets >= 0))
        ),
        "bellman_boundary_pairs": int(np.sum(values.next_terminal_targets >= 0)),
    }
    return values, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "local_assets/fragments/r_theta_nll.pt"
    )
    args = parser.parse_args()
    roles = load_qed_source_roles(ROOT, ROOT / "experiments/qed/shared_sources.json")
    import_manifest_path = args.rollouts / "manifest.json"
    import_manifest = json.loads(import_manifest_path.read_text())
    if (
        import_manifest.get("schema_version") != "compose.qed.h24_import.v1"
        or import_manifest.get("corpus_sha256") != LEGACY_H24_CORPUS_SHA256
        or import_manifest.get("source_split_sha256") != roles.manifest_sha256
        or import_manifest.get("train_sources") != len(roles.train)
    ):
        raise ValueError("QED imported rollout manifest differs from the frozen roles")
    manifest_path = ROOT / "experiments/fragments/assets.json"
    manifest = json.loads(manifest_path.read_text())
    assets = manifest["assets"]
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
    if import_manifest.get("reference") != reference.identity():
        raise ValueError("QED imported rollouts belong to another reference")
    lookup, embedding_record = _embeddings(args.embeddings, reference)
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite QED feature corpus: {output}")
    output.mkdir(parents=True, exist_ok=True)
    batch_script_sha = _sha256(Path(__file__).read_bytes())
    builder_sha = _sha256((ROOT / "src/compose_v4/experiments/qed_shared_training.py").read_bytes())
    embedding_sha = _sha256(json.dumps(embedding_record, sort_keys=True).encode())
    missing_cache = set()
    completed = 0
    for index, (input_index, source) in enumerate(zip(roles.train_input_indices, roles.train)):
        name = f"train_{index:04d}"
        rollout_path = args.rollouts / f"{name}.json"
        rollout_raw = rollout_path.read_bytes()
        rollout_sha = _sha256(rollout_raw)
        if import_manifest["output_files"].get(rollout_path.name) != rollout_sha:
            raise ValueError(f"QED imported rollout changed after verification: {rollout_path}")
        rollout = json.loads(rollout_raw)
        if (
            rollout.get("source_original") != source
            or rollout.get("source_index") != index
            or rollout.get("source_input_row_index") != input_index
        ):
            raise ValueError(
                f"QED imported rollout source differs from frozen role: {rollout_path}"
            )
        for trajectory in rollout["trajectories"]:
            for step, node in enumerate(trajectory["path"]):
                smiles = node["smiles"]
                if smiles not in lookup:
                    if step != 0:
                        raise ValueError(f"QED H24 embedding absent for non-source state: {smiles}")
                    missing_cache.add(smiles)
        path = output / f"{name}.npz"
        if path.exists():
            if not args.resume:
                raise FileExistsError(f"refusing to overwrite QED feature shard: {path}")
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
            if (
                metadata.get("rollout_sha256") != rollout_sha
                or metadata.get("builder_sha256") != builder_sha
                or metadata.get("embedding_manifest_sha256") != embedding_sha
                or metadata.get("source_index") != index
            ):
                raise ValueError(f"QED existing feature shard is incompatible: {path}")
            completed += 1
            continue
        values, details = _feature_bytes(rollout, reference, lookup)
        metadata = {
            "schema_version": "compose.qed.shared_features.v4",
            "role": "train",
            "source_index": index,
            "source_input_row_index": input_index,
            "source_split_sha256": roles.manifest_sha256,
            "rollout_path": rollout_path.name,
            "rollout_sha256": rollout_sha,
            "rollout_schema": rollout["schema_version"],
            "import_provenance": rollout["import_provenance"],
            "reference": reference.identity(),
            "reference_manifest_sha256": _sha256(manifest_path.read_bytes()),
            "budget_max": 24,
            "feature_schema": "region_features_v1",
            "target_semantics": "terminal_region",
            "goal_regions": [list(region) for region in registered_regions()],
            "builder_sha256": builder_sha,
            "embedding_manifest_sha256": embedding_sha,
            "code_sha256": {"batch_script": batch_script_sha},
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "rdkit": rdkit.__version__,
                "torch": torch.__version__,
            },
            **details,
        }
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            features=values.features,
            labels=values.labels,
            region_indices=values.region_indices,
            next_row_indices=values.next_row_indices,
            next_terminal_targets=values.next_terminal_targets,
            metadata=np.str_(json.dumps(metadata, sort_keys=True)),
        )
        with tempfile.NamedTemporaryFile(dir=output, prefix=f".{name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        completed += 1
        if completed % 25 == 0:
            print(f"QED H24 feature shards: {completed}/{len(roles.train)}", flush=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    _write_json(
        output / "h24_embeddings.json",
        {
            "schema_version": "compose.qed.h24_embeddings.v1",
            "corpus_sha256": LEGACY_H24_CORPUS_SHA256,
            "source_split_sha256": roles.manifest_sha256,
            "reference": reference.identity(),
            "embedding_record": embedding_record,
            "embedding_manifest_sha256": embedding_sha,
            "source_representations_encoded_live": sorted(missing_cache),
            "feature_shards": completed,
            "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "code_sha256": {"script": batch_script_sha, "builder": builder_sha},
        },
    )
    print(json.dumps({"feature_shards": completed, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
