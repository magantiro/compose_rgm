#!/usr/bin/env python3
"""Rebuild docs/DUDE_HOLDOUT_POOL.json from the public DUD-E target pages.

The dev seeds were drawn from DUD-E `actives_final` for parp1/fa7/braf/jak2, but
those source files were never checked in -- only the 22 selected seeds. That made
the pool unreproducible and, for one round, looked like a hard blocker. It is
not: the target pages serve the active sets directly.

The same selection rule as the dev seeds is applied, so the holdout is the SAME
DISTRIBUTION as the training molecules. That matters: the MOLLEO/PMO pool
completes at 0.7% against the dev seeds' 10.8%, so qualifying a controller there
would confound "does guidance execute better" with "does it transfer across
distributions".

Usage:  python3 tools/build_dude_pool.py [--out docs/DUDE_HOLDOUT_POOL.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("parp1", "fa7", "braf", "jak2")
URL = "https://dude.docking.org/targets/{t}/actives_final.ism"


def fetch(target: str, cache: Path) -> list[str]:
    f = cache / f"{target}.ism"
    if not f.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL.format(t=target), timeout=60) as r:
            f.write_bytes(r.read())
    return [ln.split()[0] for ln in f.read_text().splitlines() if ln.split()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/DUDE_HOLDOUT_POOL.json")
    ap.add_argument("--cache", default=".cache/dude")
    args = ap.parse_args()
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED
    RDLogger.DisableLog("rdApp.*")

    def canon(s):
        m = Chem.MolFromSmiles(s)
        return (Chem.MolToSmiles(m), m) if m else (None, None)

    dev = [s["smiles"] for s in
           json.loads((ROOT / "docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]]
    bench = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    bench = [b["smiles"] if isinstance(b, dict) else b for b in bench]
    used = {c for c, _ in map(canon, dev + bench) if c}
    ha = [canon(s)[1].GetNumHeavyAtoms() for s in dev]
    lo, hi = min(ha), max(ha)

    rows = []
    for t in TARGETS:
        for smi in fetch(t, ROOT / args.cache):
            c, m = canon(smi)
            if not c or c in used:
                continue
            n = m.GetNumHeavyAtoms()
            if not (lo <= n <= hi):          # heavy-atom range matched to the seeds
                continue
            rows.append({"smiles": c, "target": t, "heavy": n,
                         "qed": round(QED.qed(m), 4)})
    seen, keep = set(), []
    for r in sorted(rows, key=lambda r: hashlib.sha256(r["smiles"].encode()).hexdigest()):
        if r["smiles"] in seen:
            continue
        seen.add(r["smiles"])
        keep.append(r)

    (ROOT / args.out).write_text(json.dumps(
        {"rule": "DUD-E actives_final for parp1/fa7/braf/jak2 from dude.docking.org; "
                 "canonicalized; canonical-SMILES disjoint from GENMOL_T4_DEV_SEEDS "
                 "and GENMOL_T4_SEEDS; heavy-atom range matched to the dev seeds; "
                 "deterministic SHA-256 ordering",
         "heavy_range": [lo, hi], "n": len(keep), "molecules": keep}, indent=1))
    from collections import Counter
    print(f"excluded (dev+benchmark, canonical)={len(used)}  heavy range={lo}-{hi}")
    print(f"wrote {args.out}: {len(keep)} molecules {dict(Counter(r['target'] for r in keep))}")


if __name__ == "__main__":
    main()
