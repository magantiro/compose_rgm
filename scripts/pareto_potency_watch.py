"""Answer the five potency watch-items from the smoke shards. WATCH ONLY.

The census passed G3 at a binding share of 0.76 against a 0.90 line, and G4 at a
potency reach of 0.700 against a 0.85 line. Both passed. **Neither line moves.**
Tightening the task now because a passing number sits near its line would be
moving a line after seeing the census, which is the thing the preregistration
exists to prevent.

So this script does not adjust anything. It measures whether the risk that those
numbers flagged actually materialised in the controlled runs:

  W1  do extreme preference weights produce different ENDPOINTS, not merely
      different one-step choices?
  W2  does the potency-heavy side stop moving early?
  W3  does one objective dominate most final Pareto points?
  W4  do final preference selections stay differentiated?
  W5  does HV stop improving because potency effectively saturates?

If these come back badly the smoke can legitimately FAIL the Pareto experiment
despite Stage 0 passing. That is an acceptable outcome and is reported as a
failure, not repaired.

Every state committed by every arm is recorded in the shards, so all five are
answered by re-scoring those sequences locally with RDKit and the frozen DRD2
oracle. No kernel and no Modal.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from pareto_tradeoff_census import (  # noqa: E402
    z_developability,
    z_potency,
)

#: A step counts as "moving" when it changes the objective by more than this in
#: held-in IQR units. Small enough to catch real movement, large enough not to
#: count floating-point noise as progress.
MOVEMENT_EPS = 1e-3

#: The preference whose branch is the potency-heavy side, and its opposite.
POTENCY_HEAVY_W = 0.9
DEVELOP_HEAVY_W = 0.1


def score_states(states: list[str], oracle, key_a: str, key_b: str,
                 cache: dict[str, tuple[float, float]]) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Crippen, QED

    unseen = [s for s in states if s not in cache]
    if unseen:
        logodds = np.asarray(oracle.margin_many(unseen), dtype=float)
        for smi, lo in zip(unseen, logodds):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                cache[smi] = (float("nan"), float("nan"))
                continue
            try:
                qed = float(QED.qed(mol))
            except Exception:  # noqa: BLE001
                qed = float("nan")
            zp = float(z_potency(np.array([lo]))[0])
            zd = float(z_developability(np.array([qed]),
                                        np.array([float(Crippen.MolLogP(mol))]))[0])
            cache[smi] = (zp, zd)
    values = {"P": 0, "D": 1}
    ia, ib = values[key_a], values[key_b]
    return np.array([[cache[s][ia], cache[s][ib]] for s in states], dtype=float)


def last_productive_step(trace: np.ndarray, column: int) -> int:
    """The last step index that changed the objective by more than MOVEMENT_EPS.

    0 means it never moved after the start. The horizon is the budget, so a
    value well below it says the arm stopped improving that objective early --
    which is exactly what W2 asks.
    """
    deltas = np.abs(np.diff(trace[:, column]))
    moved = np.flatnonzero(deltas > MOVEMENT_EPS)
    return int(moved[-1] + 1) if len(moved) else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json")
    parser.add_argument("--arm", default="greedy_pref",
                        help="arm whose branches carry the preference signal")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from rdkit import RDLogger

    from compose_v4.drd2_oracle import load_default_oracle

    RDLogger.DisableLog("rdApp.*")
    oracle = load_default_oracle(str(args.manifest))

    rows = [json.loads(Path(f).read_text())
            for f in sorted(glob.glob(str(args.shards / "*.json")))
            if not f.endswith(".partial.json")]
    if not rows:
        raise SystemExit(f"no shards under {args.shards}")

    key_a = rows[0].get("objective_a", "P")
    key_b = rows[0].get("objective_b", "D")
    preferences = rows[0].get("preferences", [0.1, 0.3, 0.5, 0.7, 0.9])
    i_hi = preferences.index(POTENCY_HEAVY_W)
    i_lo = preferences.index(DEVELOP_HEAVY_W)

    cache: dict[str, tuple[float, float]] = {}
    w1_differ, w2_last, w3_binding, w4_distinct, w5_stall = [], [], [], [], []
    per_source = {}

    for row in rows:
        arm = row.get("arms", {}).get(args.arm)
        if arm is None or not arm.get("action_sequences"):
            continue
        seqs = arm["action_sequences"]
        endpoints = arm["endpoints"]

        # W1 -- extreme weights: different ENDPOINTS, not just different steps.
        differ = endpoints[i_hi] != endpoints[i_lo]
        w1_differ.append(bool(differ))

        # W2 -- does the potency-heavy side stop moving early?
        hi_trace = score_states(list(seqs[i_hi]), oracle, key_a, key_b, cache)
        last_p = last_productive_step(hi_trace, 0)
        w2_last.append(last_p)

        # W3 -- does one objective dominate the final Pareto points?
        z = np.asarray(arm["endpoint_z"], dtype=float)
        finite = z[np.all(np.isfinite(z), axis=1)]
        if len(finite) > 1:
            # RATIO of the two axes' spreads across the final front. Naming the
            # smaller axis alone would have no falsifying range -- argmin always
            # returns something, even when the two vary identically. The ratio
            # can land anywhere in [0, 1]: 1 means both axes sweep equally, near
            # 0 means one axis is effectively flat and the front is really
            # one-dimensional.
            spread = finite.max(axis=0) - finite.min(axis=0)
            hi = float(np.max(spread))
            w3_binding.append(float(np.min(spread) / hi) if hi > 0 else 0.0)

        # W4 -- do final selections stay differentiated?
        w4_distinct.append(len(set(endpoints)))

        # W5 -- does HV stall because potency saturates? Track when the
        # potency-heavy branch stops gaining potency relative to the horizon.
        horizon = len(seqs[i_hi]) - 1
        w5_stall.append(last_p / horizon if horizon else 0.0)

        per_source[str(row["index"])] = {
            "extreme_endpoints_differ": bool(differ),
            "potency_heavy_last_productive_step": last_p,
            "horizon": horizon,
            "distinct_endpoints": len(set(endpoints)),
        }

    horizon = rows[0].get("budget", 6)
    payload = {
        "schema": "compose.pareto.potency_watch",
        "status": "SMOKE_HELD_IN",
        "framing": ("WATCH ONLY. The census lines are NOT moved. G3 binding share "
                    "0.76 faced the 0.90 line and G4 potency reach 0.700 faced the "
                    "0.85 line; both passed, and both stay where they were frozen "
                    "at d206d55."),
        "arm_examined": args.arm,
        "n_sources": len(w1_differ),
        "movement_eps": MOVEMENT_EPS,
        "W1_extreme_weights_give_different_endpoints": {
            "fraction": float(np.mean(w1_differ)) if w1_differ else None,
            "falsifying_range": "[0, 1]; 0 means the preference changed nothing",
            "question": "different ENDPOINTS, not merely different one-step choices",
        },
        "W2_potency_heavy_side_stops_early": {
            "median_last_productive_step": (float(np.median(w2_last))
                                            if w2_last else None),
            "horizon": horizon,
            "fraction_stalling_before_half_horizon": (
                float(np.mean([s < horizon / 2 for s in w2_last]))
                if w2_last else None),
            "falsifying_range": f"[0, {horizon}]; {horizon} means it moved to the end",
        },
        "W3_one_objective_dominates_final_points": {
            "median_spread_ratio": (float(np.median(w3_binding))
                                    if w3_binding else None),
            "fraction_below_quarter": (
                float(np.mean([r < 0.25 for r in w3_binding]))
                if w3_binding else None),
            "statistic": ("min(spread) / max(spread) of the two objectives across "
                          "each source's final endpoint set"),
            "falsifying_range": ("[0, 1]; 1 means both axes sweep equally, near 0 "
                                 "means one axis is effectively flat and the front "
                                 "is really one-dimensional"),
            "why_not_argmin": ("naming the smaller axis alone has no falsifying "
                               "range -- argmin always returns something, even "
                               "when the two vary identically"),
        },
        "W4_selections_stay_differentiated": {
            "mean_distinct_endpoints_of_5": (float(np.mean(w4_distinct))
                                             if w4_distinct else None),
            "all_five_identical_fraction": (
                float(np.mean([d == 1 for d in w4_distinct])) if w4_distinct else None),
            "falsifying_range": "[1, 5]; 1 means the preference is inert",
        },
        "W5_hv_stalls_on_potency_saturation": {
            "median_fraction_of_horizon_still_moving": (
                float(np.median(w5_stall)) if w5_stall else None),
            "note": ("low values mean potency stops improving well before the budget "
                     "is spent, which is the saturation the G4 reach fraction "
                     "warned about"),
        },
        "per_source": per_source,
        "verdict_rule": ("These do not adjust any threshold. If they come back "
                         "badly the smoke may legitimately FAIL the Pareto "
                         "experiment despite Stage 0 passing, and that is reported "
                         "as a failure rather than repaired."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")

    print(f"POTENCY WATCH -- arm {args.arm}, {len(w1_differ)} sources")
    print(f"  W1 extreme weights give different endpoints  "
          f"{payload['W1_extreme_weights_give_different_endpoints']['fraction']}")
    print(f"  W2 median last productive potency step       "
          f"{payload['W2_potency_heavy_side_stops_early']['median_last_productive_step']}"
          f" / {horizon}")
    print(f"  W3 median spread ratio (1 = both sweep)      "
          f"{payload['W3_one_objective_dominates_final_points']['median_spread_ratio']}")
    print(f"  W4 mean distinct endpoints of 5              "
          f"{payload['W4_selections_stay_differentiated']['mean_distinct_endpoints_of_5']}")
    print(f"  W5 median fraction of horizon still moving   "
          f"{payload['W5_hv_stalls_on_potency_saturation']['median_fraction_of_horizon_still_moving']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
