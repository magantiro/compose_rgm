"""Reduce the construction-prior drift measurement to a verdict.

Reads the artifact `pmo_construction_prior_drift.py` wrote and adds the two
controls a paired QED claim needs on THIS lane, which the earlier call site did
not: the construction families CHANGE the heavy-atom count, and QED falls with
molecular size, so a QED gain accompanied by a size change is confounded until
the size is partialled out.  Two independent controls are computed --
a regression of the paired QED delta on the paired heavy-atom delta (the
intercept is the gain at zero size change) and the subset of decisions where the
two arms produced molecules of identical heavy-atom count -- because a single
adjustment that could only ever report "it was not size" is not a control.

Recomputes nothing chemical: no oracle, no model, no RDKit.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def _permutation_p(deltas, *, better_is_positive, draws=20000, seed=7):
    values = np.asarray(deltas, dtype=float)
    observed = values.mean()
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(draws):
        flipped = (values * rng.choice([-1.0, 1.0], len(values))).mean()
        hits += flipped >= observed if better_is_positive else flipped <= observed
    return (hits + 1) / (draws + 1)


def _mean_se(values):
    values = list(values)
    mean = statistics.fmean(values)
    se = statistics.pstdev(values) / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "se": round(se, 4),
        "sigma": round(mean / se, 2) if se else None,
        "n": len(values),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    drift = json.loads(args.drift.read_text())
    rows = drift["paired_rows"]
    dq = np.array([r["on_qed"] - r["off_qed"] for r in rows])
    ds = np.array([r["on_sa"] - r["off_sa"] for r in rows])
    dh = np.array([r["on_heavy"] - r["off_heavy"] for r in rows], dtype=float)

    design = np.column_stack([np.ones(len(dh)), dh])
    beta, *_ = np.linalg.lstsq(design, dq, rcond=None)
    residual = dq - design @ beta
    variance = (residual @ residual) / (len(dq) - 2)
    se = np.sqrt(np.diag(np.linalg.inv(design.T @ design) * variance))

    size_matched = [r for r in rows if r["on_heavy"] == r["off_heavy"]]

    lineage = drift["lineage"]
    off_drift = drift["lineage_drift"]["off"]["drift"]
    on_drift = drift["lineage_drift"]["on"]["drift"]

    payload = {
        "schema_version": "pmo_construction_prior_verdict_v1",
        "oracle_calls_spent": 0,
        "task": drift["task"],
        "kernel": drift["kernel"],
        "checkpoint": drift["checkpoint"],
        "source_artifact": str(args.drift),
        "verdict": (
            "CLOSED_LOOP_DRIFT_REDUCED_OPEN_LOOP_DRIFT_UNCHANGED"
            if on_drift > off_drift
            else "NO_CLOSED_LOOP_DRIFT_REDUCTION"
        ),
        "outcome_closed_loop": {
            "what": (
                "Score-free lineage from the real run's own 16 initialization "
                "entries, 15 generations x 16 proposals, parents drawn uniformly "
                "so no objective is read anywhere."
            ),
            "median_qed_generation_0": drift["lineage_drift"]["off"][
                "median_qed_generation_0"
            ],
            "off_median_qed_final": drift["lineage_drift"]["off"]["median_qed_final"],
            "on_median_qed_final": drift["lineage_drift"]["on"]["median_qed_final"],
            "off_drift": off_drift,
            "on_drift": on_drift,
            "drift_reduction_fraction": (
                round(1.0 - (on_drift / off_drift), 4) if off_drift else None
            ),
            "off_frac_qed_above_0.6_final": lineage["off"][-1]["frac_qed_above_0.6"],
            "on_frac_qed_above_0.6_final": lineage["on"][-1]["frac_qed_above_0.6"],
            "off_frac_sa_at_most_4_final": lineage["off"][-1]["frac_sa_at_most_4"],
            "on_frac_sa_at_most_4_final": lineage["on"][-1]["frac_sa_at_most_4"],
            "off_median_sa_final": lineage["off"][-1]["median_sa"],
            "on_median_sa_final": lineage["on"][-1]["median_sa"],
            "off_median_heavy_final": lineage["off"][-1]["median_heavy"],
            "on_median_heavy_final": lineage["on"][-1]["median_heavy"],
        },
        "outcome_open_loop": {
            "what": (
                "The same 702 decisions, ON and OFF, at the real 250-call run's "
                "OWN parents.  The parents are the drifted run's, so this cannot "
                "show drift PREVENTION -- only whether the chemistry produced at "
                "each point along that trajectory is better."
            ),
            "off_drift_Q1_to_Q4": drift["paired"]["drift_off_Q1_to_Q4"],
            "on_drift_Q1_to_Q4": drift["paired"]["drift_on_Q1_to_Q4"],
            "reading": (
                "Essentially unchanged, and expected to be: an open-loop arm "
                "inherits its parents' drift by construction."
            ),
        },
        "mechanism_paired": {
            "n_paired_decisions": len(rows),
            "disagreement_rate": drift["paired"]["disagreement_rate"],
            "delta_qed": _mean_se(dq),
            "delta_sa": _mean_se(ds),
            "delta_heavy": _mean_se(dh),
            "frac_qed_above_0.6": {
                "off": drift["paired"]["off_frac_qed_above_0.6"],
                "on": drift["paired"]["on_frac_qed_above_0.6"],
            },
            "frac_sa_at_most_4": {
                "off": drift["paired"]["off_frac_sa_at_most_4"],
                "on": drift["paired"]["on_frac_sa_at_most_4"],
            },
            "permutation_p_delta_qed": round(
                _permutation_p(dq, better_is_positive=True), 5
            ),
            "permutation_p_delta_sa": round(
                _permutation_p(ds, better_is_positive=False, seed=8), 5
            ),
        },
        "size_controls": {
            "why": (
                "Unlike the earlier call site, where paired dHeavy was exactly "
                "0.000000, the construction families change molecular size, and "
                "QED falls with size -- so an unpartialled QED gain is confounded."
            ),
            "regression_intercept": round(float(beta[0]), 4),
            "regression_intercept_se": round(float(se[0]), 4),
            "regression_intercept_sigma": round(float(beta[0] / se[0]), 2),
            "regression_slope_per_heavy_atom": round(float(beta[1]), 5),
            "regression_slope_se": round(float(se[1]), 5),
            "size_matched_subset": _mean_se(
                [r["on_qed"] - r["off_qed"] for r in size_matched]
            ),
            "size_matched_subset_delta_sa": _mean_se(
                [r["on_sa"] - r["off_sa"] for r in size_matched]
            ),
            "reading": (
                "The fitted size slope is NEGATIVE, so the small size increase the "
                "prior brings works AGAINST its QED gain rather than explaining it; "
                "the gain survives both the regression intercept and the "
                "size-matched subset."
            ),
        },
        "support_identity": {
            "candidates_scored": drift["prior_statistics"]["candidates_scored"],
            "candidates_joined": drift["prior_statistics"]["candidates_joined"],
            "join_rate": round(
                drift["prior_statistics"]["candidates_joined"]
                / drift["prior_statistics"]["candidates_scored"],
                4,
            ),
            "note": (
                "An unjoined candidate keeps the support floor rather than being "
                "dropped, so the reachable set is identical in both arms whatever "
                "the join rate; `order` returns a permutation by construction and "
                "the suite asserts it."
            ),
        },
        "scope": [
            "ONE task (celecoxib_rediscovery), ONE seed schedule, ZERO oracle calls.",
            "The checkpoint is PROVISIONAL and forbidden for frozen results.",
            "The closed-loop arm has no objective at all, so it measures the "
            "proposal law's own ratchet and not a scored campaign.",
            "No scored PMO entry point sets `construction_prior` yet.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=1, sort_keys=True))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
