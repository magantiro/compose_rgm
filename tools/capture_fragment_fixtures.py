"""Copy identity-selected historical fragment parity evidence, without generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads((args.root / "experiments/fragments/manifest.json").read_text())
    paths = {
        "motif_extension": ["attempts/seed2/BARICITINIB_000.json"],
        "scaffold_decoration": ["seed8/attempts/frozen/BARICITINIB_000.json"],
        "linker_design": [f"seed6/attempts/BARICITINIB_{i:03d}.json" for i in (0, 1)],
        "superstructure_generation": ["shards/superstructure_generation__BARICITINIB__seed0.json"],
        "superstructure_uniform": ["shards/superstructure_generation__BARICITINIB__seed0.json"],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "compose_fragment_parity_fixtures_v1",
        "selection": "BARICITINIB, first declared seed, first attempt; first TWO linker attempts. Chosen by identity, not quality.",
        "tasks": {},
    }
    for record in records["historical_generation"]:
        task = record["task"]
        source = args.root / ".worktrees" / record["branch"]
        run_name = Path(record["contract_path"]).stem
        files = [record["contract_path"], *(f"diagnostics/{run_name}/{p}" for p in paths[task])]
        items = []
        for relative in files:
            origin = source / relative
            data = origin.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if relative == record["contract_path"] and digest != record["contract_sha256"]:
                raise ValueError(f"historical contract hash mismatch: {origin}")
            target = args.output / task / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"copy mismatch: {target}")
            items.append(
                {
                    "path": str(target.relative_to(args.output)),
                    "source_path": relative,
                    "sha256": digest,
                    "source_revision": record["inspected_revision"],
                }
            )
        manifest["tasks"][task] = items
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Preserved fixture bytes and contract identities in {args.output}")


if __name__ == "__main__":
    main()
