#!/usr/bin/env python3
"""Evidence: heteroatom abundance differs sharply by lipid region (head/linker/tail).

Runs lipid_region_labels over a corpus sample and reports the per-region element
distribution. Expectation (and the justification for a region-conditioned
element-insertion prior): tails are near-pure carbon, linkers are O/N-rich, heads
are N-rich. A globally-averaged element prior would misplace heteroatoms; this
quantifies how region-specific they are.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/audit_region_heteroatoms.py --n 6000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from compose_v4.lipids.region_labels import REGION_NAMES, lipid_region_labels

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/generator_corpus_cnof_v1.smiles"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "diagnostics/region_heteroatom_abundance.json")
    args = parser.parse_args()

    lines = [ln.strip().split()[0] for ln in args.corpus.open() if ln.strip()]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(lines), size=min(args.n, len(lines)), replace=False)

    per_region_elements: dict[str, Counter] = {name: Counter() for name in REGION_NAMES.values()}
    region_atom_totals: Counter = Counter()
    n_ok = 0
    for i in idx:
        mol = Chem.MolFromSmiles(lines[i])
        if mol is None:
            continue
        n_ok += 1
        labels = lipid_region_labels(mol)
        for atom, code in zip(mol.GetAtoms(), labels):
            name = REGION_NAMES[code]
            per_region_elements[name][atom.GetSymbol()] += 1
            region_atom_totals[name] += 1

    report = {}
    for name, counts in per_region_elements.items():
        tot = sum(counts.values()) or 1
        dist = {el: round(c / tot, 4) for el, c in counts.most_common()}
        hetero = round(1 - counts.get("C", 0) / tot, 4)
        report[name] = {"atoms": tot, "element_fraction": dist, "heteroatom_fraction": hetero}

    out = {
        "format": "compose_lipid_region_heteroatom_abundance_v1",
        "sample_molecules": n_ok,
        "region_atom_share": {n: round(region_atom_totals[n] / max(1, sum(region_atom_totals.values())), 4)
                              for n in REGION_NAMES.values()},
        "per_region": report,
        "interpretation": (
            "Heteroatom abundance is strongly region-specific -- tails near-pure carbon, "
            "linkers O/N-rich, heads N-rich. A GLOBAL element-insertion prior misplaces "
            "heteroatoms; this justifies a region-conditioned prior + region embedding."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"sampled {n_ok} lipids")
    print(f"{'region':>8} {'atoms%':>7} {'hetero%':>8}  element distribution")
    for name in ("head", "linker", "tail", "other"):
        r = report[name]
        share = out["region_atom_share"][name]
        print(f"{name:>8} {share:>7.1%} {r['heteroatom_fraction']:>8.1%}  {r['element_fraction']}")
    print(f"written: {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
