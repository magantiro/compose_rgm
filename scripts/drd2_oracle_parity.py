"""Hold the numpy DRD2 oracle to numerical agreement with the original.

Two things are settled here, and neither is assumed:

1. THE PLATT ORIENTATION.  libsvm's binary sign convention is easy to invert,
   and an inverted oracle still returns plausible-looking probabilities in
   [0, 1] -- it would simply reward the wrong molecules.  Both orientations are
   evaluated against the original estimator's stored scores and the one that
   agrees is pinned into the manifest.  If neither agrees to tolerance, this
   exits non-zero rather than picking the closer one.

2. THAT THE ORACLE DISCRIMINATES.  Agreement on a panel of inactives is not
   evidence of a working oracle: a function returning 0.0 everywhere would pass.
   So a known-active panel is scored too, and the run fails if actives are not
   separated from the source molecules.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compose_v4.drd2_oracle import (  # noqa: E402
    DRD2Oracle,
    oracle_fingerprint,
)

#: Agreement demanded of the reimplementation against the original estimator.
#: Both evaluate the same float64 arithmetic, so anything above rounding noise
#: means a real discrepancy, not a tolerance to be widened.
PARITY_TOLERANCE = 1e-9


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", required=True, type=Path)
    parser.add_argument("--actives", type=Path,
                        help="tab-separated pairs; column 2 is a known active")
    parser.add_argument("--actives-count", type=int, default=200)
    args = parser.parse_args()

    manifest_path = args.oracle_dir / "drd2_oracle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    reference = manifest["reference_scores"]
    print(f"parity panel: {len(reference)} molecules from "
          f"{manifest['parity_panel']['source']}")

    fingerprints = np.vstack([oracle_fingerprint(r["smiles"]) for r in reference])
    expected = np.array([r["predict_proba_class1"] for r in reference])
    expected_decision = np.array([r["decision_function"] for r in reference])

    # ---- 1. orientation -------------------------------------------------
    results = {}
    for orientation in ("direct", "flipped"):
        oracle = DRD2Oracle(args.oracle_dir / "drd2_svm_parameters.npz",
                            platt_orientation=orientation)
        got = oracle.probabilities(fingerprints)
        results[orientation] = float(np.abs(got - expected).max())
        print(f"  orientation {orientation:8}  max |Δ P(active)| = "
              f"{results[orientation]:.3e}")

    chosen = min(results, key=results.get)
    if results[chosen] > PARITY_TOLERANCE:
        print(f"\nFAILED: neither orientation reaches {PARITY_TOLERANCE:g}. "
              f"The reimplementation does not reproduce the original; do not "
              f"use this oracle.")
        return 1

    oracle = DRD2Oracle(args.oracle_dir / "drd2_svm_parameters.npz",
                        platt_orientation=chosen)

    # The decision value is checked separately from the probability: Platt
    # scaling is monotone and saturating, so a wrong kernel or gamma can still
    # produce near-identical probabilities down in the tail where every score is
    # ~0. Matching the raw decision values rules that out.
    decision_gap = float(np.abs(-oracle.decision_values(fingerprints)
                                - expected_decision).max())
    flipped_gap = float(np.abs(oracle.decision_values(fingerprints)
                               - expected_decision).max())
    decision_gap = min(decision_gap, flipped_gap)
    print(f"  max |Δ decision_function|        = {decision_gap:.3e}")

    # ---- 2. discrimination ----------------------------------------------
    source_scores = oracle.score_many([r["smiles"] for r in reference])
    print(f"\nsource panel  P(active): mean {source_scores.mean():.4f}  "
          f"max {source_scores.max():.4f}  >=0.5: {(source_scores >= 0.5).sum()}")

    active_summary = None
    if args.actives and args.actives.exists():
        actives = []
        for line in args.actives.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                actives.append(parts[1])
            if len(actives) >= args.actives_count:
                break
        active_scores = oracle.score_many(actives)
        print(f"active panel  P(active): mean {active_scores.mean():.4f}  "
              f"min {active_scores.min():.4f}  >=0.5: "
              f"{(active_scores >= 0.5).sum()}/{len(active_scores)}")
        active_summary = {
            "molecules": len(active_scores),
            "mean": float(active_scores.mean()),
            "min": float(active_scores.min()),
            "fraction_above_0.5": float((active_scores >= 0.5).mean()),
        }
        if active_scores.mean() <= source_scores.mean():
            print("\nFAILED: known actives do not score above the source panel. "
                  "The oracle is not discriminating; a sign or fingerprint "
                  "mismatch is the likely cause.")
            return 1

    manifest["platt_orientation"] = chosen
    manifest["parity"] = {
        "tolerance": PARITY_TOLERANCE,
        "max_abs_probability_gap": results[chosen],
        "max_abs_decision_gap": decision_gap,
        "orientation_rejected": {k: v for k, v in results.items() if k != chosen},
        "source_panel": {
            "molecules": len(source_scores),
            "mean": float(source_scores.mean()),
            "max": float(source_scores.max()),
            "fraction_above_0.5": float((source_scores >= 0.5).mean()),
        },
        "active_panel": active_summary,
        "verdict": "numpy reimplementation reproduces the original estimator",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nPARITY OK -- orientation '{chosen}' pinned into "
          f"{manifest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
