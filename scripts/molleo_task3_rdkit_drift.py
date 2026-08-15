#!/usr/bin/env python
"""Measure how much QED and SA move between RDKit versions.

Three of the five Task 3 objectives are frozen to arrays and cannot drift.  QED
and SA are RDKit computations and therefore CAN: RDKit's stereo perception,
ring perception and descriptor code all evolve.  A benchmark that does not know
the size of that drift is quietly reporting a different quantity than the one it
claims to reproduce.

So it is measured, on a large ZINC-250k sample, in both directions:

    old-env:  python molleo_task3_rdkit_drift.py --emit  panel.json
    new-env:  python molleo_task3_rdkit_drift.py --compare panel.json

The emitting environment should be the one whose RDKit is closest to the
benchmark's era.  The comparing environment is the one the experiments will
actually run in.  What comes out is a rate and a magnitude, not a reassurance.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_ZINC = Path("local_runtime/zinc250k/250k_rndm_zinc_drugs_clean_3.csv")


def sample_zinc(path: Path, count: int, seed: int) -> list[str]:
    with open(path, newline="") as handle:
        rows = [row["smiles"].strip() for row in csv.DictReader(handle)]
    rng = np.random.default_rng(seed)
    index = rng.choice(len(rows), size=min(count, len(rows)), replace=False)
    return [rows[int(i)] for i in sorted(index)]


def score(smiles_list: list[str]) -> list[dict]:
    """QED and SA via RDKit's OWN reference implementations, whatever version."""

    import rdkit
    from rdkit import Chem
    from rdkit.Chem import QED

    sys.path.insert(0, str(Path(rdkit.__file__).resolve().parent
                          / "Contrib" / "SA_Score"))
    import sascorer

    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        rows.append({
            "smiles": smiles,
            "qed": float(QED.qed(mol)),
            "sa": float(sascorer.calculateScore(mol)),
            "n_chiral_unassigned": len(
                Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--compare", action="store_true")
    parser.add_argument("--zinc", type=Path, default=DEFAULT_ZINC)
    parser.add_argument("--smiles-list", type=Path,
                        help="score these molecules instead of a ZINC sample; "
                             "ZINC is drug-like and well-behaved, so exotic "
                             "cohorts are where drift actually shows up")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--record", type=Path,
                        help="append this measurement to an oracle manifest under "
                             "the given label")
    parser.add_argument("--label", default="cohort")
    args = parser.parse_args()

    import rdkit

    if args.emit:
        if args.smiles_list is not None:
            smiles = [line.strip() for line in
                      args.smiles_list.read_text().splitlines() if line.strip()]
            smiles = smiles[: args.count]
            source = str(args.smiles_list)
        else:
            smiles = sample_zinc(args.zinc, args.count, args.seed)
            source = str(args.zinc)
        payload = {"rdkit": rdkit.__version__, "count": len(smiles),
                   "seed": args.seed, "zinc": source,
                   "rows": score(smiles)}
        args.panel.write_text(json.dumps(payload))
        print(f"emitted {len(payload['rows'])} molecules under RDKit "
              f"{rdkit.__version__} -> {args.panel}")
        return 0

    payload = json.loads(args.panel.read_text())
    reference = payload["rows"]
    actual = {row["smiles"]: row for row in score([r["smiles"] for r in reference])}
    print(f"reference RDKit {payload['rdkit']}  ->  this RDKit {rdkit.__version__}")
    print(f"{len(reference)} molecules\n")

    measurement: dict = {
        "cohort": payload.get("zinc"),
        "molecules": len(reference),
        "reference_rdkit": payload["rdkit"],
        "this_rdkit": rdkit.__version__,
    }
    for field in ("qed", "sa"):
        expected = np.array([r[field] for r in reference])
        got = np.array([actual[r["smiles"]][field] for r in reference])
        delta = np.abs(got - expected)
        moved = delta > 1e-9
        worst = int(np.argmax(delta))
        measurement[field] = {
            "n_moved": int(moved.sum()), "fraction_moved": float(moved.mean()),
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta_over_movers":
                float(delta[moved].mean()) if moved.any() else 0.0,
            "worst_molecule": reference[worst]["smiles"] if moved.any() else None,
        }
        print(f"{field}: {int(moved.sum())}/{len(delta)} molecules moved "
              f"({100 * moved.mean():.3f}%)")
        if moved.any():
            print(f"      max |delta| = {delta.max():.4f}  "
                  f"mean over movers = {delta[moved].mean():.4f}")
            print(f"      worst: {reference[worst]['smiles']}")
            print(f"             {expected[worst]:.6f} -> {got[worst]:.6f}")

    stereo_ref = np.array([r["n_chiral_unassigned"] for r in reference])
    stereo_now = np.array([actual[r["smiles"]]["n_chiral_unassigned"]
                           for r in reference])
    changed = stereo_ref != stereo_now
    print(f"\npotential stereocentres: {int(changed.sum())}/{len(changed)} molecules "
          f"changed ({100 * changed.mean():.3f}%)")
    measurement["potential_stereocentres_changed"] = int(changed.sum())
    if changed.any():
        print("  this is the mechanism: SA's stereo penalty is "
              "log10(n_centres + 1), and RDKit's stereo perception changed")

    if args.record is not None:
        manifest = json.loads(args.record.read_text())
        manifest.setdefault("rdkit_drift_measurements", {})[args.label] = measurement
        args.record.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nrecorded as '{args.label}' in {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
