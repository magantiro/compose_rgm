#!/usr/bin/env python3
"""Prove the benchmark prompts we condition on are the OFFICIAL ones.

``data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv`` is the
asset every reported row is generated from.  This script fetches the pinned
upstream ``references/fragments.csv`` from the InVirtuoGen results repository
and checks the two agree -- textually, and again after canonicalizing every
SMILES through RDKit, so a difference in spelling cannot hide a difference in
molecule.

It also records the one place where the executable upstream protocol does
something our sampler does not, so the difference is on the record rather than
discovered in review: for SUPERSTRUCTURE, ``in_virtuo_gen/evaluation/downstream.py``
does not prompt with the bare core.  It calls
``list_individual_attach_points(core, depth=2)`` and then, for each of the 100
samples, ``random.choice`` over that enumeration -- so upstream's 100 samples
come from 100 randomly chosen attachment-point variants of the same core, each
pinning where growth may occur.  That draw is taken from the unseeded global
``random`` module, so it is not reproducible upstream either.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from rdkit import Chem

REPO_MANIFEST = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
UPSTREAM_RELATIVE = "references/fragments.csv"
FRAGMENT_COLUMNS = (
    "smiles",
    "linker_design",
    "motif_extension",
    "scaffold_decoration",
    "superstructure_generation",
)


def canonical(value: str) -> str | None:
    """Canonical SMILES for a possibly dot-separated, possibly dummy-bearing cell."""
    parts = []
    for part in value.split("."):
        part = part.strip()
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        parts.append(Chem.MolToSmiles(mol, canonical=True))
    return ".".join(sorted(parts))


def load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["name"]: row for row in csv.DictReader(handle)}


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_official_fragment_evaluator import fetch, verify_only

    pkg_root = fetch()
    verify_only()
    upstream_path = Path(pkg_root) / UPSTREAM_RELATIVE

    repo = load(REPO_MANIFEST)
    upstream = load(upstream_path)

    findings: list[dict] = []
    mismatches = 0
    if set(repo) != set(upstream):
        print(f"FAIL drug sets differ: {set(repo) ^ set(upstream)}")
        return 1

    for name in sorted(repo):
        for column in FRAGMENT_COLUMNS:
            raw_repo = repo[name][column]
            raw_up = upstream[name][column]
            can_repo, can_up = canonical(raw_repo), canonical(raw_up)
            ok = raw_repo == raw_up
            same_molecule = can_repo is not None and can_repo == can_up
            if not (ok and same_molecule):
                mismatches += 1
                findings.append(
                    {
                        "drug": name,
                        "column": column,
                        "repo": raw_repo,
                        "upstream": raw_up,
                        "identical_text": ok,
                        "identical_molecule": same_molecule,
                    }
                )

    payload = {
        "schema": "compose_fragment_conditioning_instance_v1",
        "upstream": {
            "repo": "invirtuolabs/InVirtuoGen_results",
            "commit": "b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb",
            "path": UPSTREAM_RELATIVE,
            "sha256": "43db9adbcf7f8895be6197bccdf729517c4433b7fdbf637f413fa9c8055c3a31",
        },
        "repo_manifest": str(REPO_MANIFEST),
        "drugs_checked": len(repo),
        "cells_checked": len(repo) * len(FRAGMENT_COLUMNS),
        "mismatches": mismatches,
        "findings": findings,
        "protocol_deltas": {
            "superstructure_attach_point_randomization": (
                "Upstream downstream.py wraps this same core in "
                "list_individual_attach_points(core, depth=2) and draws one "
                "variant per sample with random.choice from the UNSEEDED global "
                "random module. COMPOSE conditions on the bare core and does not "
                "pin growth positions. Same input molecule, different prompt "
                "decoration; upstream's own draw is not reproducible."
            ),
            "scaffold_morphing": (
                "Upstream copies the linker_design result into scaffold_morphing "
                "rather than running a separate task; we do the same."
            ),
        },
    }
    out = Path("diagnostics/fragment_official_suite_v2/conditioning_instance.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"{'PASS' if mismatches == 0 else 'FAIL'} "
        f"{payload['cells_checked'] - mismatches}/{payload['cells_checked']} cells "
        f"identical to upstream {UPSTREAM_RELATIVE}"
    )
    print(f"wrote {out}")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
