"""What empirical reward law do successful PMO trajectories follow at the MACRO scale?

Valleys are common between individual primitive edits.  The open question is whether they survive
once a trajectory is read at the ~12-18 primitive option scale COMPOSE actually acts at, because
if they mostly do not, a PMO controller needs to predict immediate macro reward and very little
else -- and the heavy long-horizon machinery would be solving a problem the action scale already
solved.

METHOD, and the choice that decides whether the answer means anything.  Segmentation is UNIFORM:
every macro is `cap` primitives, boundaries fall where they fall.  Choosing boundaries to maximise
monotonicity would make "mostly monotone" true by construction -- the segmenter would be selecting
the answer.  The monotone-optimal segmentation is reported alongside as an explicit UPPER BOUND so
the two cannot be confused.

SCOPE.  These are the compiled witness routes, so the macro mix is partly a property of that path
compiler, and the trajectories carry SMILES and scores but no action records -- so WHERE and HOW
are measured structurally (heavy-atom and ring deltas) and macro FAMILY is not recoverable here.
Zero oracle calls: every score was already counted when the routes were compiled.
"""
from __future__ import annotations

import json
import pathlib
import sys
from itertools import pairwise

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from compose_v4.experiments.macro_segmentation import min_segments

ROUTES = "diagnostics/pmo_macro_horizon_v1/celecoxib_routes_v1.json"
OUT = "diagnostics/pmo_macro_reward_law_v1/macro_reward_law_v1.json"
CAPS = (8, 10, 12, 14, 16, 18)
TERMINAL = 0.99


def _structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    return mol.GetNumHeavyAtoms(), mol.GetRingInfo().NumRings()


def _boundaries(scores, smiles, cap):
    """Uniform macros of `cap` primitives; the final macro absorbs the remainder."""
    rows = []
    cuts = list(range(0, len(scores) - 1, cap)) + [len(scores) - 1]
    for start, stop in pairwise(cuts):
        if stop <= start:
            continue
        before, after = float(scores[start]), float(scores[stop])
        h0, r0 = _structure(smiles[start])
        h1, r1 = _structure(smiles[stop])
        rows.append(
            {
                "start": start,
                "stop": stop,
                "primitives": stop - start,
                "u_before": before,
                "u_after": after,
                "delta": after - before,
                "heavy_delta": (h1 - h0) if None not in (h0, h1) else None,
                "ring_delta": (r1 - r0) if None not in (r0, r1) else None,
                "fraction_through_route": stop / (len(scores) - 1),
                "regime": "terminal" if after >= TERMINAL else "climbing",
            }
        )
    return rows


def main():
    with open(ROUTES) as handle:
        document = json.load(handle)
    witnesses = [r for r in document["routes"] if r["status"] == "witness_found"]
    report = {
        "schema_version": "pmo_macro_reward_law_v1",
        "evidence_role": "zero_oracle_offline_reward_law_audit",
        "new_charged_oracle_calls": 0,
        "segmentation": "UNIFORM at each cap; monotone-optimal reported separately as an upper bound",
        "witness_routes": len(witnesses),
        "caps": {},
        "monotone_optimal_upper_bound": {},
    }
    for cap in CAPS:
        rows = []
        for route in witnesses:
            for row in _boundaries(route["trajectory_scores"], route["trajectory_smiles"], cap):
                rows.append({**row, "route": route["source_name"]})
        climbing = [r for r in rows if r["regime"] == "climbing"]
        terminal = [r for r in rows if r["regime"] == "terminal"]

        def stats(group):
            if not group:
                return None
            delta = np.array([g["delta"] for g in group])
            positive = delta[delta > 0]
            negative = delta[delta < 0]
            out = {
                "n": len(group),
                "fraction_positive": float((delta > 0).mean()),
                "fraction_zero": float((delta == 0).mean()),
                "median_delta": float(np.median(delta)),
                "median_positive_delta": float(np.median(positive)) if positive.size else None,
                "n_negative": int(negative.size),
                "worst_dip": float(negative.min()) if negative.size else 0.0,
                "median_dip": float(np.median(negative)) if negative.size else 0.0,
                "dips_shallower_than_0.01": (
                    int((negative > -0.01).sum()) if negative.size else 0
                ),
            }
            if len(group) > 2:
                parent = np.array([g["u_before"] for g in group])
                length = np.array([g["primitives"] for g in group], dtype=float)
                through = np.array([g["fraction_through_route"] for g in group])
                for name, axis in (
                    ("corr_delta_parent_score", parent),
                    ("corr_delta_primitives", length),
                    ("corr_delta_progress", through),
                ):
                    out[name] = (
                        float(np.corrcoef(axis, delta)[0, 1]) if axis.std() > 0 else None
                    )
            return out

        report["caps"][str(cap)] = {
            "all": stats(rows),
            "climbing": stats(climbing),
            "terminal": stats(terminal),
        }
        monotone = sum(
            1
            for route in witnesses
            if min_segments(route["trajectory_scores"], max_len=cap)[0] is not None
        )
        report["monotone_optimal_upper_bound"][str(cap)] = {
            "routes_with_a_monotone_segmentation": monotone,
            "of": len(witnesses),
        }
        c = report["caps"][str(cap)]["climbing"]
        print(
            f"cap {cap:2d} | climbing n={c['n']:3d} "
            f"P(d>0)={c['fraction_positive']:.3f} med+={c['median_positive_delta'] or 0:.4f} "
            f"dips={c['n_negative']:2d} worst={c['worst_dip']:+.4f} "
            f"| monotone-optimal {monotone}/{len(witnesses)}",
            flush=True,
        )
    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print("WROTE", OUT)


if __name__ == "__main__":
    sys.exit(main())
