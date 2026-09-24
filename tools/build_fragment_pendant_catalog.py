"""Publish a source-balanced pendant-region catalog from frozen train rows."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
from pathlib import Path

from rdkit import rdBase

from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.benchmark.training_pendant_regions import build_pendant_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-region-catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    base_path = args.training_region_catalog
    base_manifest_path = base_path.parent / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    if physical_sha256(base_path) != base_manifest["catalog_sha256"]:
        raise ValueError("frozen training-region catalog differs from its manifest")
    base = json.loads(base_path.read_text())
    source = Path(base["source"])
    inputs = (
        base_path,
        base_manifest_path,
        source,
        Path(__file__).resolve(),
        Path("src/compose_v4/benchmark/training_pendant_regions.py"),
        Path("src/compose_v4/benchmark/training_attachment_fragments.py"),
        Path("docs/FRAGMENT_PENDANT_DECORATION_DEV_2026-09-24.md"),
    )
    hashes = {str(path.resolve()): physical_sha256(path) for path in inputs}
    catalog = build_pendant_catalog(base, source)
    if hashes != {str(path.resolve()): physical_sha256(path) for path in inputs}:
        raise ValueError("pendant-catalog material inputs changed during extraction")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=args.output_dir.parent, prefix=".pendant-catalog-"
    ) as temp:
        stage = Path(temp) / "complete"
        stage.mkdir()
        catalog_path = stage / "catalog.json"
        catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
        (stage / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "split_first_training_pendant_catalog_build_v1",
                    "inputs": dict(sorted(hashes.items())),
                    "catalog_sha256": physical_sha256(catalog_path),
                    "code_revision": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
                    "training_molecules": catalog["training_molecules"],
                    "entries": len(catalog["entries"]),
                    "source_access_basis": base_manifest["source_access_basis"],
                    "qed_sa_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        stage.rename(args.output_dir)
    print(
        json.dumps(
            {"entries": len(catalog["entries"]), "census": catalog["census"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
