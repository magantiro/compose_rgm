"""Publish a split-first broad region content library with physical provenance."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from rdkit import Chem, rdBase
from run_fragment_attachment_library_pilot import _atomic_json

from compose_v4.benchmark.fragment_constrained import load_genmol_prompts
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.benchmark.training_region_catalog import build_region_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-molecules", type=int, default=10_000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prompts = root / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    protected = [
        Path(__file__).resolve(),
        root / "src/compose_v4/benchmark/training_region_catalog.py",
        root / "src/compose_v4/data/scaffold_partition.py",
        prompts,
        args.source,
    ]
    hashes = {str(p.resolve()): physical_sha256(p) for p in protected}
    excluded = frozenset(
        Chem.MolToSmiles(Chem.MolFromSmiles(p.original_smiles), canonical=True)
        for p in load_genmol_prompts(prompts)
    )
    catalog = build_region_catalog(
        args.source,
        expected_sha256=args.source_sha256,
        excluded_canonical=excluded,
        training_molecules=args.training_molecules,
    )
    if hashes != {str(p.resolve()): physical_sha256(p) for p in protected}:
        raise RuntimeError("catalog material inputs changed during extraction")
    _atomic_json(args.output_dir / "catalog.json", catalog)
    _atomic_json(
        args.output_dir / "manifest.json",
        {
            "schema": "split_first_training_region_catalog_build_v1",
            "inputs": hashes,
            "catalog_sha256": physical_sha256(args.output_dir / "catalog.json"),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "training_molecules": args.training_molecules,
            "entries": len(catalog["entries"]),
            "source_access_basis": "existing local GuacaMol research asset; no redistribution license asserted",
            "selection_uses_qed_sa": False,
            "benchmark_reference_identities_excluded": len(excluded),
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "entries": len(catalog["entries"]),
                "exclusions": catalog["exclusions"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
