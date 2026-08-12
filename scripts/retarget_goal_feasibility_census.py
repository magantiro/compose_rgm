"""Is a potency/developability goal switch well-posed on the held-out reserve?

Counts only. This freezes nothing, selects no panel and sets no threshold. Its
one job is to answer, before any controller is written, whether the intended
goal language can produce a meaningful intervention experiment or whether it
walks straight into a preregistered stop rule:

  1. If DRD2 potency and developability are strongly ANTI-correlated, the
     conjunction P&D may be nearly empty and every arm fails it -- the "goal is
     too hard" stop. Many DRD2 ligands are lipophilic bases, so this is the
     live risk, not a hypothetical one.
  2. If they are strongly POSITIVELY correlated, continue-A already does well on
     B and there is no intervention to measure -- the "goals too correlated"
     stop.
  3. If almost every reserve molecule already satisfies a goal at step zero,
     the source eligibility rule ("does not already satisfy both phase goals")
     removes the panel.

Properties: DRD2 P(active) from the frozen SVM, QED and Crippen cLogP from
RDKit. All three are computed on the SAME molecule set so the correlation is
measured, not assumed.

Reported at several candidate thresholds rather than one, because choosing the
threshold is a later, separate, held-in calibration step. Deliberately no
recommendation is emitted from this script: seeing the reachability numbers and
picking the threshold in the same pass is how a floor/ceiling problem gets
tuned away instead of found.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Candidate cut points. Reported as a grid; NOT selected here.
POTENCY_GRID = (0.3, 0.5, 0.7)
QED_GRID = (0.5, 0.6, 0.7)
LOGP_BOX = (1.0, 4.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve", type=Path,
                        default=Path("diagnostics/editing_v2_matched_validation_reserve_ids.json.gz"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, QED

    from compose_v4.drd2_oracle import load_default_oracle

    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()

    reserve = json.load(gzip.open(args.reserve, "rt"))
    smiles = sorted(reserve["reserve_source_keys"])
    print(f"held-out reserve: {len(smiles):,} molecules")

    oracle = load_default_oracle(str(args.manifest))
    potency = oracle.score_many(smiles)          # P(active)
    print(f"[{time.perf_counter()-started:5.1f}s] DRD2 scored", flush=True)

    qed, logp, heavy, ok = [], [], [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            qed.append(np.nan); logp.append(np.nan); heavy.append(0); ok.append(False)
            continue
        try:
            qed.append(QED.qed(mol))
        except Exception:
            qed.append(np.nan)
        logp.append(Crippen.MolLogP(mol))
        heavy.append(mol.GetNumHeavyAtoms())
        ok.append(True)
    qed = np.asarray(qed); logp = np.asarray(logp)
    heavy = np.asarray(heavy); ok = np.asarray(ok)
    potency = np.asarray(potency, dtype=float)

    valid = ok & np.isfinite(qed) & np.isfinite(logp) & np.isfinite(potency)
    print(f"[{time.perf_counter()-started:5.1f}s] RDKit properties; "
          f"{int(valid.sum()):,} of {len(smiles):,} usable\n")

    p, q, l, h = potency[valid], qed[valid], logp[valid], heavy[valid]

    def quantiles(name: str, values: np.ndarray) -> dict:
        cuts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        qs = np.percentile(values, cuts)
        print(f"  {name:>10} " + "  ".join(f"p{c}={v:.3f}" for c, v in zip(cuts, qs)))
        return {f"p{c}": float(v) for c, v in zip(cuts, qs)}

    print("distributions on the held-out reserve")
    dist = {"potency": quantiles("DRD2", p), "qed": quantiles("QED", q),
            "clogp": quantiles("cLogP", l), "heavy_atoms": quantiles("heavy", h)}

    # THE question: are potency and developability at odds on this pool?
    def corr(a: np.ndarray, b: np.ndarray) -> dict:
        from scipy import stats
        return {"pearson": float(np.corrcoef(a, b)[0, 1]),
                "spearman": float(stats.spearmanr(a, b).statistic)}

    in_box = ((l >= LOGP_BOX[0]) & (l <= LOGP_BOX[1])).astype(float)
    correlations = {
        "potency_vs_qed": corr(p, q),
        "potency_vs_clogp": corr(p, l),
        "potency_vs_logp_in_box": corr(p, in_box),
        "qed_vs_clogp": corr(q, l),
    }
    print("\ncorrelation across the reserve  (the well-posedness question)")
    for name, c in correlations.items():
        print(f"  {name:>24}  pearson {c['pearson']:+.3f}  spearman {c['spearman']:+.3f}")

    # Base rates: how much of the pool already satisfies each goal?
    print(f"\nbase rates at candidate thresholds (cLogP box {LOGP_BOX})")
    print(f"{'potency>=':>10} {'QED>=':>7} {'P':>8} {'D':>8} {'P and D':>9} "
          f"{'neither':>9} {'lift':>7}")
    rates = {}
    for pt in POTENCY_GRID:
        for qt in QED_GRID:
            sat_p = p >= pt
            sat_d = (q >= qt) & (l >= LOGP_BOX[0]) & (l <= LOGP_BOX[1])
            both = sat_p & sat_d
            # lift > 1 means potency and developability co-occur MORE than
            # independence predicts; < 1 means they are at odds.
            expected = sat_p.mean() * sat_d.mean()
            lift = (both.mean() / expected) if expected > 0 else float("nan")
            key = f"potency>={pt}_qed>={qt}"
            rates[key] = {"P": float(sat_p.mean()), "D": float(sat_d.mean()),
                          "P_and_D": float(both.mean()),
                          "P_and_D_count": int(both.sum()),
                          "neither": float((~sat_p & ~sat_d).mean()),
                          "lift_over_independence": float(lift)}
            print(f"{pt:>10} {qt:>7} {sat_p.mean():>7.1%} {sat_d.mean():>7.1%} "
                  f"{both.mean():>8.2%} {(~sat_p & ~sat_d).mean():>8.1%} {lift:>7.2f}")

    # Source eligibility: a source must not ALREADY satisfy both phase goals.
    print("\neligible sources (satisfy NEITHER goal at step zero), by threshold")
    for key, r in rates.items():
        print(f"  {key:>26}  {r['neither']:>6.1%} of {int(valid.sum()):,} "
              f"= {int(r['neither'] * valid.sum()):,} molecules")

    # Conditional structure: among potent molecules, what fails developability?
    print("\namong the most potent decile, what blocks developability?")
    top = p >= np.percentile(p, 90)
    for qt in QED_GRID:
        fails_qed = (q[top] < qt).mean()
        fails_logp = ((l[top] < LOGP_BOX[0]) | (l[top] > LOGP_BOX[1])).mean()
        too_greasy = (l[top] > LOGP_BOX[1]).mean()
        print(f"  QED>={qt}: fails QED {fails_qed:>5.1%}   "
              f"outside cLogP box {fails_logp:>5.1%}   (too lipophilic {too_greasy:>5.1%})")

    payload = {
        "schema": "compose.retarget.goal_feasibility_census",
        "status": "COUNTS_ONLY_FREEZES_NOTHING_SELECTS_NO_THRESHOLD",
        "purpose": ("Decide whether a potency->developability goal switch is "
                    "well-posed BEFORE writing a controller. Threshold choice is "
                    "a separate held-in calibration step, deliberately not made "
                    "here."),
        "pool": "held-out matched validation reserve, disjoint from R_theta training sources",
        "molecules_total": len(smiles),
        "molecules_usable": int(valid.sum()),
        "logp_box": list(LOGP_BOX),
        "distributions": dist,
        "correlations": correlations,
        "base_rates": rates,
        "rdkit_caveat": ("Local RDKit, which differs from the Modal image holding "
                         "the authoritative kernel. Feasibility estimate only."),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
