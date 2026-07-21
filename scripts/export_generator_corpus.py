#!/usr/bin/env python3
"""Export the generator training corpus as a plain SMILES file.

The COMPOSE generator's trainer takes a positional SMILES file (one SMILES per
line, whitespace-first-field) and its loader keeps only neutral, connected,
CNOF (C/N/O/F) molecules <= max_atoms -- because the current generative heads can
only build C/N/O/F atoms (S/P are representable but not generatable; that is a
v2 CNOSP head-widening). So v1 trains on the CNOF-representable subset of our
corpus (R0 real anchor + R1 reachable support), which is ~82% of it and includes
all BEAE-class chemistry (amine + acrylate + propiolate enamine-ester = C/N/O).

Emits a deduplicated SMILES file and reports what was kept vs dropped and why.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/export_generator_corpus.py --max-atoms 96
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1"
CNOF = {"C", "N", "O", "F"}


def _cnof_neutral_within(smiles: str, max_atoms: int) -> tuple[str | None, str]:
    """Return (canonical_smiles, 'kept') or (None, reason) for the generator scope."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "unparseable"
    if Chem.GetFormalCharge(mol) != 0:
        return None, "charged"
    heavy = mol.GetNumHeavyAtoms()
    if heavy > max_atoms:
        return None, "too_large"
    if heavy < 1:
        return None, "empty"
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    if not elements <= CNOF:
        return None, "non_cnof"  # S/P/halogen -> deferred to CNOSP v2
    return Chem.MolToSmiles(mol), "kept"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-atoms", type=int, default=96,
                        help="drop molecules with more heavy atoms (kernel size cap)")
    parser.add_argument("--r0", type=Path, default=DATASET / "r0_observed_real_structures.csv")
    parser.add_argument("--r1", type=Path, default=DATASET / "r1_reaction_grounded_corpus_v1.csv")
    parser.add_argument("--out", type=Path, default=DATASET / "generator_corpus_cnof_v1.smiles")
    args = parser.parse_args()

    kept: dict[str, None] = {}
    reasons: Counter = Counter()
    per_layer: dict[str, Counter] = {"r0": Counter(), "r1": Counter()}

    # R0 real anchor
    with args.r0.open() as fh:
        for row in csv.DictReader(fh):
            smi = row.get("canonical_isomeric_smiles") or row.get("canonical_smiles")
            canon, reason = _cnof_neutral_within(str(smi), args.max_atoms)
            reasons[reason] += 1
            per_layer["r0"][reason] += 1
            if canon is not None:
                kept.setdefault(canon, None)

    # R1 reachable support
    with args.r1.open() as fh:
        for row in csv.DictReader(fh):
            canon, reason = _cnof_neutral_within(row["canonical_smiles"], args.max_atoms)
            reasons[reason] += 1
            per_layer["r1"][reason] += 1
            if canon is not None:
                kept.setdefault(canon, None)

    args.out.write_text("".join(f"{s}\n" for s in kept))

    total = sum(reasons.values())
    print(f"wrote {len(kept):,} unique CNOF-generatable SMILES -> {args.out.relative_to(REPO_ROOT)}")
    print(f"scanned {total:,} rows (R0+R1)")
    for layer, c in per_layer.items():
        k = c["kept"]; n = sum(c.values())
        print(f"  {layer}: kept {k:,}/{n:,} ({k/max(1,n):.1%})  dropped: "
              f"{ {r: v for r, v in c.items() if r != 'kept'} }")
    print(f"drop reasons (total): { {r: v for r, v in reasons.most_common() if r != 'kept'} }")


if __name__ == "__main__":
    main()
