#!/usr/bin/env python
"""Hold all five Task 3 oracles to the originals, and measure what cannot be held.

TWO CLAIMS THAT MUST NOT BE CONFLATED
-------------------------------------
1. IMPLEMENTATION PARITY -- does this code compute what the original computes,
   here, now?  Every one of the five is checked against the original estimator
   or the original reference implementation IN THIS ENVIRONMENT, to 1e-9.  Both
   sides are the same float64 arithmetic, so anything above rounding is a real
   discrepancy and not a tolerance to widen.  A failure here is fatal.

2. VERSION DRIFT -- does this environment's RDKit still agree with the
   environment the reference panel was frozen in?  JNK3, GSK3B and DRD2 are
   frozen to arrays and cannot drift.  QED and SA are RDKit computations and
   CAN: RDKit's stereo perception moved between 2023.09 and 2025.09, which
   changes SA's `log10(n_stereocentres + 1)` penalty.  This is MEASURED and
   recorded, never assumed away.  Drift is reported with exit code 2 -- a real
   finding about comparability, not an implementation bug.

Orientation is checked separately and is fatal.  Agreement on inactives proves
nothing: a function returning 0.0 everywhere would pass.  Known JNK3 and GSK3B
actives must score ABOVE the ZINC background, or the objective is inverted.

Exit codes: 0 = parity holds and nothing drifted; 1 = an implementation
disagrees with its original, or an objective is inverted; 2 = implementations
agree but RDKit has moved under QED/SA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compose_v4.benchmark.oracles import Task3Objectives  # noqa: E402

#: Implementations of the same arithmetic must agree to rounding noise.
EXACT_TOLERANCE = 1e-9
#: Below this a difference is float noise rather than a moved value.
DRIFT_FLOOR = 1e-9


def rdkit_reference(panel: list[str]) -> dict[str, np.ndarray]:
    """QED and SA from RDKit's OWN implementations, in THIS environment.

    SA is `Contrib/SA_Score/sascorer.py`, the original that TDC copied. Checking
    our vendored copy against it is what makes the vendoring safe; checking it
    against a second copy of itself would prove nothing.
    """

    import rdkit
    from rdkit import Chem
    from rdkit.Chem import QED

    sys.path.insert(0, str(Path(rdkit.__file__).resolve().parent
                          / "Contrib" / "SA_Score"))
    import sascorer

    mols = [Chem.MolFromSmiles(s) for s in panel]
    return {
        "qed": np.array([QED.qed(m) for m in mols], dtype=np.float64),
        "sa": np.array([sascorer.calculateScore(m) for m in mols], dtype=np.float64),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path,
                        default=Path("artifacts/oracles/molleo_task3_v1"))
    parser.add_argument("--drd2", type=Path,
                        default=Path("artifacts/oracles/drd2_svm_v1"))
    parser.add_argument("--write-manifest", action="store_true",
                        help="record the parity result into the manifest")
    args = parser.parse_args()

    manifest_path = args.bundle / "molleo_task3_oracle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    reference = manifest["reference_scores"]
    panel = [row["smiles"] for row in reference]

    oracles = Task3Objectives(args.bundle, args.drd2)
    scored = oracles.raw_many(panel)
    unparseable = [panel[i] for i, row in enumerate(scored) if row is None]
    if unparseable:
        print(f"FAIL: {len(unparseable)} reference molecules no longer parse")
        return 1

    import rdkit
    print(f"panel: {len(panel)} molecules | reference produced by "
          f"{manifest['reference_panel']['produced_by']}")
    print(f"checking under RDKit {rdkit.__version__}, numpy {np.__version__}\n")

    failures: list[str] = []
    drifted: list[str] = []
    report: dict[str, dict] = {}

    # ---- 1. implementation parity, in THIS environment -------------------
    print("IMPLEMENTATION PARITY -- this code vs the original, here, now")
    live = rdkit_reference(panel)
    for name in ("jnk3", "gsk3b", "drd2", "qed", "sa"):
        actual = np.array([getattr(row, name) for row in scored], dtype=np.float64)
        if name in live:                       # QED/SA: RDKit's own code, this env
            expected = live[name]
            against = f"RDKit {rdkit.__version__} reference implementation"
        else:                                  # frozen: the original estimator
            expected = np.array([row[name] for row in reference], dtype=np.float64)
            against = manifest["reference_panel"]["produced_by"]
        delta = np.abs(actual - expected)
        worst = int(np.argmax(delta))
        entry = {"against": against, "tolerance": EXACT_TOLERANCE,
                 "max_abs_delta": float(delta.max()),
                 "n_over_tolerance": int((delta > EXACT_TOLERANCE).sum()),
                 "worst_molecule": panel[worst]}
        report[name] = {"implementation_parity": entry}
        ok = delta.max() <= EXACT_TOLERANCE
        if not ok:
            failures.append(name)
        print(f"  {'OK  ' if ok else 'FAIL'} {name:6s} max|delta| = {delta.max():.3e} "
              f"vs {against}")
        if not ok:
            print(f"       worst: {panel[worst]}")
            print(f"       expected {expected[worst]!r} got {actual[worst]!r}")

    # ---- 2. version drift against the frozen panel -----------------------
    print("\nVERSION DRIFT -- this environment vs the environment the panel was "
          "frozen in")
    for name in ("qed", "sa"):
        frozen = np.array([row[name] for row in reference], dtype=np.float64)
        actual = np.array([getattr(row, name) for row in scored], dtype=np.float64)
        delta = np.abs(actual - frozen)
        moved = delta > DRIFT_FLOOR
        worst = int(np.argmax(delta))
        report[name]["version_drift"] = {
            "reference_environment": manifest["reference_panel"]["produced_by"],
            "n_moved": int(moved.sum()), "n_panel": len(panel),
            "max_abs_delta": float(delta.max()),
            "worst_molecule": panel[worst] if moved.any() else None,
        }
        if moved.any():
            drifted.append(name)
            print(f"  DRIFT {name:4s} {int(moved.sum())}/{len(panel)} molecules moved, "
                  f"max |delta| = {delta.max():.4f}")
            print(f"        worst: {panel[worst]}")
            print(f"        frozen {frozen[worst]:.6f} -> now {actual[worst]:.6f}")
        else:
            print(f"  OK    {name:4s} nothing moved")
    print("  (jnk3, gsk3b, drd2 are frozen to arrays and cannot drift)")

    # ---- orientation ----------------------------------------------------
    print()
    groups = np.array([row["group"] for row in reference])
    for name, active_group in (("jnk3", "jnk3_active"), ("gsk3b", "gsk3b_active")):
        values = np.array([getattr(row, name) for row in scored], dtype=np.float64)
        active = values[groups == active_group].mean()
        background = values[groups == "zinc"].mean()
        report[name]["orientation"] = {
            "known_actives_mean": float(active),
            "zinc_background_mean": float(background),
            "separation": float(active - background),
        }
        if active <= background:
            failures.append(f"{name}-orientation")
            print(f"FAIL {name}: actives {active:.4f} <= background {background:.4f} "
                  f"-- THE OBJECTIVE IS INVERTED")
        else:
            print(f"OK   {name}: known actives {active:.4f} vs ZINC background "
                  f"{background:.4f} (separation {active - background:+.4f})")

    # DRD2 discrimination is established in drd2_svm_v1; here the only claim is
    # that the frozen evaluator reproduces the original estimator.
    drd2 = np.array([row.drd2 for row in scored])
    print(f"     drd2 on this panel: max {drd2.max():.4f}, "
          f"{int((drd2 >= 0.5).sum())} of {len(drd2)} above 0.5")

    if args.write_manifest:
        manifest["parity"] = {
            "checked_under": {"rdkit": rdkit.__version__, "numpy": np.__version__,
                              "python": sys.version.split()[0]},
            "exact_tolerance": EXACT_TOLERANCE,
            "objectives": report,
            "implementation_parity_passed": not failures,
            "version_drift_detected": sorted(drifted),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nrecorded parity into {manifest_path}")

    if failures:
        print(f"\nPARITY FAILED: {', '.join(failures)}")
        return 1
    if drifted:
        print(f"\nIMPLEMENTATION PARITY PASSED for all five. RDKit has moved "
              f"under: {', '.join(drifted)}.")
        print("This is a comparability finding, not a bug: record the RDKit "
              "version with every run and do not place these numbers beside a "
              "published number as if the environments matched.")
        return 2
    print("\nPARITY PASSED: all five objectives reproduce their references, "
          "nothing drifted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
