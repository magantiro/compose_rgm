#!/usr/bin/env python
"""Rank the proposal bottlenecks by LOST PROBABILITY MASS.

The owner's question is "what fraction of the failure is region selection, what
fraction is scale, what fraction is completion content".  Those are not
independent, so a marginal table cannot answer it.  This computes a SEQUENTIAL
conditional decomposition: starting from all proposals, apply the generic
requirements one at a time and record how much of the controller's mass each one
removes, in both orders that matter, so the attribution is not an artifact of the
order chosen.

ZERO ORACLE CALLS.
"""

from __future__ import annotations

import collections
import itertools
import json
import os

OUT = "diagnostics/pmo_global_delta_census_v1"

AXES = {
    # region selection: are several separated sites edited together
    "region_selection": ("n_anchor_groups", lambda d: d["n_anchor_groups"] >= 2),
    # scale: is the largest coherent changed region substituent-sized
    "scale": ("largest_changed_region", lambda d: d["largest_changed_region"] >= 6),
    # completion content: is what gets installed richer than a linear CNO chain
    "completion_content": (
        "installed_region_has_ring",
        lambda d: d["installed_region_has_ring"],
    ),
}


def load(path):
    return json.load(open(path))


def deltas(records, pool):
    return [
        r["delta"]
        for r in records
        if r.get("pool") == pool and r.get("delta", {}).get("status") == "ok"
    ]


def main() -> None:
    atlas = load(os.path.join(OUT, "productive_basin_atlas_v1.json"))["records"]
    proposals = load(os.path.join(OUT, "proposal_delta_census_v1.json"))["records"]

    productive = deltas(atlas, "POOL_1_MEASURED_VLOCAL")
    clean = deltas(atlas, "POOL_4_BLIND_TO_BLIND_MEASURED")
    transfer = deltas(atlas, "POOL_3_CONSTRUCTIBLE_TRANSFER")
    null = deltas(proposals, "NULL_RANDOM_PAIR")
    charged = deltas(proposals, "PROPOSED_CHARGED")

    populations = {
        "PRODUCTIVE (POOL_1, measured V_local)": productive,
        "PRODUCTIVE (POOL_4, no answer-known material)": clean,
        "PRODUCTIVE (POOL_3, constructible transfer)": transfer,
        "NULL random drug-like pair": null,
        "CONTROLLER proposals (charged)": charged,
    }

    print("=== MARGINAL SATISFACTION, per generic axis ===")
    print(f"{'population':46s}" + "".join(f"{a:>22s}" for a in AXES))
    for name, rows in populations.items():
        line = f"{name:46s}"
        for _, (_, predicate) in AXES.items():
            value = sum(1 for d in rows if predicate(d)) / len(rows)
            line += f"{value * 100:19.1f}%  "
        print(line + f"   n={len(rows)}")

    print("\n=== SEQUENTIAL LOST MASS on the CONTROLLER distribution ===")
    print("(fraction of ALL charged proposals surviving, as each requirement is added)")
    results = {}
    for order in itertools.permutations(AXES):
        surviving = list(charged)
        trail = []
        for axis in order:
            _, predicate = AXES[axis]
            before = len(surviving)
            surviving = [d for d in surviving if predicate(d)]
            trail.append(
                {
                    "axis": axis,
                    "survived": len(surviving),
                    "removed": before - len(surviving),
                    "removed_fraction_of_all": (before - len(surviving)) / len(charged),
                }
            )
        results["|".join(order)] = trail
    for key, trail in results.items():
        print(f"\n  order: {key}")
        for step in trail:
            print(
                f"    {step['axis']:20s} removes {step['removed_fraction_of_all'] * 100:5.1f}%"
                f" of all proposals -> {step['survived']:5d} left"
            )

    print("\n=== SHAPLEY-STYLE AVERAGE ATTRIBUTION (order-independent) ===")
    attribution: dict[str, list[float]] = collections.defaultdict(list)
    for trail in results.values():
        for step in trail:
            attribution[step["axis"]].append(step["removed_fraction_of_all"])
    total = sum(sum(v) / len(v) for v in attribution.values())
    rows = []
    for axis, values in attribution.items():
        mean = sum(values) / len(values)
        rows.append((axis, mean, mean / total if total else 0.0))
    rows.sort(key=lambda r: -r[1])
    for axis, mean, share in rows:
        print(f"  {axis:20s} mean removed {mean * 100:5.1f}% of all proposals"
              f"   -> {share * 100:5.1f}% of the total loss")

    print("\n=== THE SHARPEST SINGLE DISCRIMINATOR ===")
    for threshold in (6, 10, 13, 16):
        line = f"  largest_changed_region >= {threshold:2d}: "
        for name, rows_ in populations.items():
            value = sum(1 for d in rows_ if d["largest_changed_region"] >= threshold) / len(rows_)
            line += f"{name.split('(')[0].strip()[:12]} {value * 100:5.1f}%  "
        print(line)

    payload = {
        "schema_version": "pmo_lost_mass_decomposition_v1",
        "information_regime": "DEVELOPMENT_INFORMED_DIAGNOSTIC",
        "new_oracle_calls": 0,
        "axes": {k: v[0] for k, v in AXES.items()},
        "marginal": {
            name: {
                axis: round(sum(1 for d in rows if p(d)) / len(rows), 4)
                for axis, (_, p) in AXES.items()
            }
            | {"n": len(rows)}
            for name, rows in populations.items()
        },
        "sequential_orders": results,
        "average_attribution": {a: {"mean_removed": round(m, 4), "share": round(s, 4)}
                                for a, m, s in rows},
    }
    with open(os.path.join(OUT, "lost_mass_decomposition_v1.json"), "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(f"\nwrote {OUT}/lost_mass_decomposition_v1.json")


if __name__ == "__main__":
    main()
