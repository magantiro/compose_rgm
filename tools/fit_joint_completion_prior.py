"""Fit a structural histogram from exactly the catalog's admitted training rows."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from rdkit import Chem, rdBase

from compose_v4.benchmark.fragment_constrained import load_genmol_prompts
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.data.scaffold_partition import murcko_scaffold, partition_for_scaffold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    catalog_path, manifest_path = (
        args.catalog_dir / "catalog.json",
        args.catalog_dir / "manifest.json",
    )
    manifest = json.loads(manifest_path.read_text())
    if physical_sha256(catalog_path) != manifest["catalog_sha256"]:
        raise ValueError("frozen catalog hash mismatch")
    catalog = json.loads(catalog_path.read_text())
    if catalog["schema"] != "split_first_training_region_catalog_v1" or catalog["split"] != {
        "algorithm": "murcko+carbonized-wl3",
        "version": 2,
        "salt": "ringcore-v1",
        "ratios": [0.9, 0.05, 0.05],
        "partition": "train",
    }:
        raise ValueError("catalog training partition contract differs")
    source = Path(catalog["source"])
    if physical_sha256(source) != catalog["source_sha256"]:
        raise ValueError("frozen molecular source hash mismatch")
    rows = catalog["accepted_source_rows"]
    if rows != sorted(set(rows)) or len(rows) != catalog["training_molecules"] or min(rows) < 1:
        raise ValueError("catalog accepted-row identity is malformed")
    prompts = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
    excluded = {
        Chem.MolToSmiles(Chem.MolFromSmiles(p.original_smiles))
        for p in load_genmol_prompts(prompts)
    }
    counts, seen, accepted = Counter(), set(), set(rows)
    verified = []
    with source.open() as handle:
        for number, text in enumerate(handle, 1):
            if number > rows[-1]:
                break
            if number not in accepted:
                continue
            mol = Chem.MolFromSmiles(text.strip())
            if (
                mol is None
                or len(Chem.GetMolFrags(mol)) != 1
                or not 1 <= mol.GetNumHeavyAtoms() <= 40
            ):
                raise ValueError(f"admitted source row {number} no longer in molecular support")
            canonical = Chem.MolToSmiles(mol)
            scaffold = murcko_scaffold(canonical)
            if (
                canonical in excluded
                or canonical in seen
                or scaffold is None
                or partition_for_scaffold(scaffold) != "train"
            ):
                raise ValueError(f"source row {number} violates training-only unique admission")
            seen.add(canonical)
            verified.append(number)
            counts[(mol.GetNumHeavyAtoms(), mol.GetRingInfo().NumRings())] += 1
    if verified != rows:
        raise ValueError("not every admitted training row was recovered")
    cells = tuple((a, r, n) for (a, r), n in sorted(counts.items()))
    JointCompletionPrior(cells)
    inputs = (
        catalog_path,
        manifest_path,
        source,
        prompts,
        Path(__file__),
        Path("src/compose_v4/benchmark/joint_completion_prior.py"),
        Path("src/compose_v4/data/scaffold_partition.py"),
    )
    result = {
        "schema": "joint_completion_structural_prior_v1",
        "counts": cells,
        "pseudocount_mass": 1.0,
        "catalog_sha256": manifest["catalog_sha256"],
        "source_sha256": catalog["source_sha256"],
        "training_molecules": len(rows),
        "source_rows": rows,
        "split": catalog["split"],
        "weighting": "one vote per admitted training molecule; no raw cut-count weighting",
        "qed_sa_used": False,
        "benchmark_drugs_used": False,
        "largest_cells": [
            {"heavy_atoms": a, "rings": r, "molecules": n} for (a, r), n in counts.most_common(10)
        ],
        "input_sha256": {str(p.resolve()): physical_sha256(p) for p in inputs},
        "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent, prefix=".joint-fit-") as temp:
        stage = Path(temp) / "complete"
        stage.mkdir()
        (stage / "prior.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        stage.rename(args.output_dir)
    print(
        json.dumps(
            {
                "training_molecules": len(rows),
                "cells": len(cells),
                "largest_cells": result["largest_cells"],
            }
        )
    )


if __name__ == "__main__":
    main()
