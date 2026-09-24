"""Trace de novo endpoint quality back to the exact sampled source-tree sizes.

This is a read-only diagnostic.  It does not change the source law, checkpoint,
sampler, or generated endpoints.  Each saved trajectory seed is replayed only
through the source prior, which is the first RNG draw in every de novo arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import rdkit
import torch
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
from rdkit import Chem

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.eval.denovo_benchmark import denovo_benchmark_metrics

SCHEMA = "denovo_source_size_audit_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: list[float | int]) -> float:
    return float(np.mean(values)) if values else 0.0


def carbon_branchpoints(smiles: str) -> int:
    """Count nonaromatic carbon atoms with at least three heavy neighbors."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid saved/source SMILES: {smiles}")
    return sum(
        atom.GetAtomicNum() == 6 and not atom.GetIsAromatic() and atom.GetDegree() >= 3
        for atom in mol.GetAtoms()
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(
    records_path: Path,
    checkpoint_path: Path,
    reference_records_path: Path,
    output: Path,
) -> dict:
    rows = json.loads(records_path.read_text())
    c1 = sorted((row for row in rows if row["arm"] == "C1"), key=lambda row: row["index"])
    if not c1 or len({row["index"] for row in c1}) != len(c1):
        raise ValueError(f"C1 rows absent or duplicate indices: {records_path}")
    torch.set_num_threads(1)
    _model, checkpoint = load_factorized_rollout_checkpoint(str(checkpoint_path))
    source_prior = checkpoint["tree_source_prior"]
    probabilities = source_prior.probabilities
    if probabilities is None:
        probabilities = tuple(1.0 / len(source_prior.sizes) for _ in source_prior.sizes)
    expected_size = sum(
        float(size) * float(probability)
        for size, probability in zip(source_prior.sizes, probabilities, strict=True)
    )
    traced: list[dict] = []
    for row in c1:
        source = source_prior.sample(np.random.default_rng(row["trajectory_seed"]), n_slots=40)
        source_size = int(np.count_nonzero(source.atom_types > 0))
        source_smiles = molecular_graph_to_smiles(source)
        quality_pass = row["qed"] >= 0.6 and row["sa"]["score"] <= 4.0
        traced.append(
            {
                "index": row["index"],
                "trajectory_seed": row["trajectory_seed"],
                "source_size": source_size,
                "source_smiles": source_smiles,
                "source_carbon_branchpoints": carbon_branchpoints(source_smiles),
                "endpoint_size": row["heavy_atoms"],
                "endpoint_smiles": row["canonical_smiles"],
                "endpoint_carbon_branchpoints": carbon_branchpoints(row["canonical_smiles"]),
                "size_delta": row["heavy_atoms"] - source_size,
                "endpoint_qed": row["qed"],
                "endpoint_sa": row["sa"]["score"],
                "endpoint_quality_pass": quality_pass,
                "endpoint_aromatic_rings": row["aromatic_rings"],
                "endpoint_rotatable_bonds": row["rotatable_bonds"],
            }
        )
    if [row["trajectory_seed"] for row in traced] != [row["trajectory_seed"] for row in c1]:
        raise AssertionError("trajectory seed alignment failed")
    if any(row["source_size"] not in source_prior.sizes for row in traced):
        raise AssertionError("source prior produced a size outside declared support")
    reference = json.loads(reference_records_path.read_text())
    matched = [row for row in reference if row["arm"] == "train_for_C1"]
    if len(matched) != 3 * len(traced):
        raise ValueError(
            f"expected three support/size-matched reference records per endpoint: {reference_records_path}"
        )
    generated_by_index = {row["index"]: row for row in traced}
    if any(
        row["matched_generated_index"] not in generated_by_index
        or row["heavy_atoms"] != generated_by_index[row["matched_generated_index"]]["endpoint_size"]
        for row in matched
    ):
        raise ValueError("reference match identity or heavy-atom equality failed")
    matched_branchpoints = [carbon_branchpoints(row["canonical_smiles"]) for row in matched]
    bins = ((0, 19), (20, 25), (26, 30), (31, 40))
    by_size = []
    for lower, upper in bins:
        subset = [row for row in traced if lower <= row["source_size"] <= upper]
        by_size.append(
            {
                "source_size_bin": [lower, upper],
                "n": len(subset),
                "quality_pass": sum(row["endpoint_quality_pass"] for row in subset),
                "quality_rate": _mean([row["endpoint_quality_pass"] for row in subset]),
                "mean_source_size": _mean([row["source_size"] for row in subset]),
                "mean_endpoint_size": _mean([row["endpoint_size"] for row in subset]),
                "mean_size_delta": _mean([row["size_delta"] for row in subset]),
                "mean_source_carbon_branchpoints": _mean(
                    [row["source_carbon_branchpoints"] for row in subset]
                ),
                "mean_endpoint_carbon_branchpoints": _mean(
                    [row["endpoint_carbon_branchpoints"] for row in subset]
                ),
                "mean_endpoint_qed": _mean([row["endpoint_qed"] for row in subset]),
                "mean_endpoint_sa": _mean([row["endpoint_sa"] for row in subset]),
                "mean_aromatic_rings": _mean([row["endpoint_aromatic_rings"] for row in subset]),
                "mean_rotatable_bonds": _mean([row["endpoint_rotatable_bonds"] for row in subset]),
            }
        )
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in traced:
        groups["quality_pass" if row["endpoint_quality_pass"] else "quality_fail"].append(row)
    report = {
        "schema_version": SCHEMA,
        "evidence_class": "retrospective_source_seed_replay",
        "inputs": {
            "records": {"path": str(records_path), "sha256": sha256(records_path)},
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
            "matched_reference_records": {
                "path": str(reference_records_path),
                "sha256": sha256(reference_records_path),
            },
        },
        "configuration": {"arm": "C1", "n_slots": 40, "source_draw": "first use of trajectory RNG"},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "rdkit": rdkit.__version__,
        },
        "n": len(traced),
        "prior_expected_size": expected_size,
        "sampled_mean_source_size": _mean([row["source_size"] for row in traced]),
        "mean_endpoint_size": _mean([row["endpoint_size"] for row in traced]),
        "mean_size_delta": _mean([row["size_delta"] for row in traced]),
        "mean_source_carbon_branchpoints": _mean(
            [row["source_carbon_branchpoints"] for row in traced]
        ),
        "mean_endpoint_carbon_branchpoints": _mean(
            [row["endpoint_carbon_branchpoints"] for row in traced]
        ),
        "mean_size_and_support_matched_reference_carbon_branchpoints": _mean(matched_branchpoints),
        "quality_pass": len(groups["quality_pass"]),
        "by_source_size": by_size,
        "by_quality": {
            label: {
                "n": len(group),
                "mean_source_size": _mean([row["source_size"] for row in group]),
                "mean_endpoint_size": _mean([row["endpoint_size"] for row in group]),
                "mean_size_delta": _mean([row["size_delta"] for row in group]),
                "mean_source_carbon_branchpoints": _mean(
                    [row["source_carbon_branchpoints"] for row in group]
                ),
                "mean_endpoint_carbon_branchpoints": _mean(
                    [row["endpoint_carbon_branchpoints"] for row in group]
                ),
            }
            for label, group in sorted(groups.items())
        },
        "limitations": [
            "Source size is observational; it is not a source-prior intervention.",
            "The C1 arm contains 150 saved trajectories of a planned 200 and is not an official final benchmark.",
            "Source tree sampling is verified from the sampler's first RNG use; later atom identity/edge retention is not inferred from endpoint SMILES.",
        ],
    }
    # Reuse the versioned metric implementation to ensure quality denominators
    # still agree with the saved endpoint count before publishing the census.
    metrics = denovo_benchmark_metrics([row["endpoint_smiles"] for row in traced])
    if metrics["high_quality"] != report["quality_pass"]:
        raise AssertionError("source-size audit quality disagrees with official-like metric")
    _atomic_json(output / "records.json", traced)
    _atomic_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.records, args.checkpoint, args.reference_records, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "n",
                    "prior_expected_size",
                    "sampled_mean_source_size",
                    "mean_endpoint_size",
                    "mean_size_delta",
                    "mean_source_carbon_branchpoints",
                    "mean_endpoint_carbon_branchpoints",
                    "mean_size_and_support_matched_reference_carbon_branchpoints",
                    "quality_pass",
                    "by_source_size",
                    "by_quality",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
