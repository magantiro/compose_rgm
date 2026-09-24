"""Measure whether the frozen one-boundary catalog describes small decorations.

BRICS yields both sides of a cut. This training-only audit separates the smaller
side from the larger side without using benchmark prompts or QED/SA labels.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, rdBase

from compose_v4.benchmark.training_attachment_fragments import physical_sha256


def audit(catalog_path: Path) -> dict:
    manifest_path = catalog_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if physical_sha256(catalog_path) != manifest["catalog_sha256"]:
        raise ValueError("frozen training catalog differs from its manifest")
    catalog = json.loads(catalog_path.read_text())
    if catalog["schema"] != "split_first_training_region_catalog_v1":
        raise ValueError("not the split-first training region catalog")
    source = Path(catalog["source"])
    if physical_sha256(source) != catalog["source_sha256"]:
        raise ValueError("training molecular corpus changed")
    accepted = catalog["accepted_source_rows"]
    if accepted != sorted(set(accepted)) or len(accepted) != catalog["training_molecules"]:
        raise ValueError("training row identities are malformed")
    sizes = {}
    selected = set(accepted)
    with source.open() as handle:
        for row, text in enumerate(handle, 1):
            if row > accepted[-1]:
                break
            if row in selected:
                molecule = Chem.MolFromSmiles(text.strip())
                if molecule is None:
                    raise ValueError(f"training row {row} no longer parses")
                sizes[row] = molecule.GetNumHeavyAtoms()
    if set(sizes) != selected:
        raise ValueError("not every accepted training molecule was recovered")
    one_boundary = [entry for entry in catalog["entries"] if len(entry["contexts"]) == 1]
    raw_sizes = Counter(entry["heavy_atoms"] for entry in one_boundary)
    smaller_by_source = defaultdict(Counter)
    larger_instances = smaller_instances = 0
    for entry in one_boundary:
        size = entry["heavy_atoms"]
        for row in entry["source_rows"]:
            if row not in sizes or not 1 <= size < sizes[row]:
                raise ValueError("region/source-row size identity changed")
            if size <= sizes[row] - size:
                smaller_by_source[row][size] += 1
                smaller_instances += 1
            else:
                larger_instances += 1
    weighted = Counter()
    for histogram in smaller_by_source.values():
        count = sum(histogram.values())
        for size, n in histogram.items():
            weighted[size] += n / count
    total_weight = sum(weighted.values())
    if abs(total_weight - len(smaller_by_source)) > 1e-7:
        raise ValueError("source-balanced region weighting failed")
    inputs = (catalog_path, manifest_path, source, Path(__file__))
    return {
        "schema": "fragment_training_region_orientation_audit_v1",
        "role": "training-only catalog audit; not a proposed generator or benchmark score",
        "rule": "at each BRICS cut, region is smaller-side when region atoms <= remaining atoms",
        "catalog_entries": len(catalog["entries"]),
        "one_boundary_entries": len(one_boundary),
        "one_boundary_entry_mean_atoms": sum(size * count for size, count in raw_sizes.items())
        / len(one_boundary),
        "one_boundary_entry_min_atoms": min(raw_sizes),
        "one_boundary_entry_size_histogram": dict(sorted(raw_sizes.items())),
        "smaller_side_instances": smaller_instances,
        "larger_side_instances": larger_instances,
        "training_molecules_with_smaller_side": len(smaller_by_source),
        "source_balanced_smaller_side_size_probability": {
            size: weight / total_weight for size, weight in sorted(weighted.items())
        },
        "input_sha256": {str(path.resolve()): physical_sha256(path) for path in inputs},
        "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "qed_sa_used": False,
        "benchmark_prompts_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    result = audit(args.catalog)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=args.output_dir.parent, prefix=".region-orientation-"
    ) as temp:
        stage = Path(temp) / "complete"
        stage.mkdir()
        (stage / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        stage.rename(args.output_dir)
    print(
        json.dumps(
            {
                "one_boundary_entries": result["one_boundary_entries"],
                "entry_mean_atoms": result["one_boundary_entry_mean_atoms"],
                "smaller_side_instances": result["smaller_side_instances"],
                "larger_side_instances": result["larger_side_instances"],
            }
        )
    )


if __name__ == "__main__":
    main()
