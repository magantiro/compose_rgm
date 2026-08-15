#!/usr/bin/env python
"""Draw the Task 3 initialization sets ONCE, and seal the official ones.

WHY THIS IS FROZEN BEFORE ANY POLICY EXISTS
-------------------------------------------
The benchmark says "120 random ZINC-250k molecules, 5 seeds".  If those 120x5
molecules are drawn at the moment the official run is launched, then every
failed development run is a chance -- however unintentional -- to end up with a
kinder draw.  Drawing them now, before any policy exists, costs nothing and
removes the possibility entirely.  The official sets are written once and this
script REFUSES to overwrite them; redrawing requires deleting the file
deliberately, which leaves a trace in git.

DEVELOPMENT SETS ARE DISJOINT BY CONSTRUCTION
---------------------------------------------
Policy development happens on separate initialization sets drawn from the pool
with every official molecule removed.  A policy tuned on development data
therefore never saw an official starting molecule.  It may of course GENERATE
one during search -- that is the search working, not a leak.

    python scripts/freeze_molleo_task3_init_sets.py \
        --zinc local_runtime/zinc250k/250k_rndm_zinc_drugs_clean_3.csv \
        --out artifacts/benchmarks/molleo_task3_init_v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: The benchmark's stated initialization size.
POPULATION = 120
#: The five official seeds. Never run during development.
OFFICIAL_SEEDS = (0, 1, 2, 3, 4)
#: Development seeds. Iterate here as much as you like.
DEVELOPMENT_SEEDS = tuple(range(100, 108))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pool(path: Path) -> list[str]:
    """Canonical, deduplicated, parseable ZINC-250k.

    Canonicalised here so that "is this molecule in an official set" is a
    string comparison later, with no chance of two spellings disagreeing.
    """

    from compose_v4.benchmark.oracles import canonical

    with open(path, newline="") as handle:
        raw = [row["smiles"].strip() for row in csv.DictReader(handle)]
    seen: dict[str, None] = {}
    for smiles in raw:
        key = canonical(smiles)
        if key is not None:
            seen.setdefault(key, None)
    return list(seen)


def draw(pool: list[str], seed: int, size: int) -> list[str]:
    """One initialization set. Independent draws, as "5 random seeds" implies."""

    rng = np.random.default_rng(seed)
    index = rng.choice(len(pool), size=size, replace=False)
    return [pool[int(i)] for i in index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zinc", type=Path,
                        default=Path("local_runtime/zinc250k/"
                                     "250k_rndm_zinc_drugs_clean_3.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/benchmarks/molleo_task3_init_v1"))
    parser.add_argument("--population", type=int, default=POPULATION)
    parser.add_argument("--redraw-official", action="store_true",
                        help="deliberately re-seal the official sets; this is "
                             "not something to do after seeing a result")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    official_path = args.out / "official_init_sets.json"
    development_path = args.out / "development_init_sets.json"

    pool = load_pool(args.zinc)
    print(f"pool: {len(pool):,} canonical molecules from {args.zinc}")

    if official_path.exists() and not args.redraw_official:
        official = json.loads(official_path.read_text())
        print(f"official sets already sealed ({official_path}); leaving them alone")
    else:
        official = {
            "population": args.population,
            "seeds": list(OFFICIAL_SEEDS),
            "pool": str(args.zinc),
            "pool_sha256": sha256_file(args.zinc),
            "pool_size": len(pool),
            "status": "SEALED -- do not run until the policy is frozen",
            "sets": {str(seed): draw(pool, seed, args.population)
                     for seed in OFFICIAL_SEEDS},
        }
        official["sets_sha256"] = {
            seed: hashlib.sha256("\n".join(molecules).encode()).hexdigest()
            for seed, molecules in official["sets"].items()}
        official_path.write_text(json.dumps(official, indent=1) + "\n")
        print(f"sealed {len(OFFICIAL_SEEDS)} official sets -> {official_path}")

    reserved = {smiles for molecules in official["sets"].values()
                for smiles in molecules}
    print(f"official molecules held out of development: {len(reserved):,}")

    remaining = [smiles for smiles in pool if smiles not in reserved]
    development = {
        "population": args.population,
        "seeds": list(DEVELOPMENT_SEEDS),
        "pool": str(args.zinc),
        "pool_sha256": official["pool_sha256"],
        "disjoint_from": str(official_path),
        "pool_size_after_holdout": len(remaining),
        "sets": {str(seed): draw(remaining, seed, args.population)
                 for seed in DEVELOPMENT_SEEDS},
    }
    development_path.write_text(json.dumps(development, indent=1) + "\n")
    print(f"drew {len(DEVELOPMENT_SEEDS)} development sets -> {development_path}")

    overlap = reserved & {smiles for molecules in development["sets"].values()
                          for smiles in molecules}
    if overlap:
        raise SystemExit(f"development sets share {len(overlap)} molecules with "
                         f"the official sets; the holdout failed")
    print("verified: no development molecule appears in any official set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
