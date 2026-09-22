#!/usr/bin/env python
"""PART 1b/2b -- cluster the productive transitions into a generic macro grammar,
report the coverage curve, and quantify where the current controller's proposal
mass is missing relative to it.

The grammar is a set of predicates over the GLOBAL DELTA ONLY.  No target
identity, no task name, no fragment content, no teacher endpoint appears in any
family definition, so the vocabulary could be frozen before a no-prescreen run.

Bottlenecks are ranked by LOST PROBABILITY MASS: for each generic requirement
the productive census establishes, what fraction of the controller's realized
proposal distribution satisfies it, marginally and jointly.

ZERO ORACLE CALLS.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OUT = "diagnostics/pmo_global_delta_census_v1"

# ---- Generic macro grammar ----------------------------------------------------
# Axes chosen because the census shows they are what a controller would have to
# DECLARE differently.  Cut points are stated here rather than fitted per task.

SMALL_REGION = 5      # at or below: a local edit
LARGE_REGION = 13     # at or above: a coherent whole-substituent scale


def scale_band(delta: dict) -> str:
    size = delta["largest_changed_region"]
    if size == 0:
        return "none"
    if size <= SMALL_REGION:
        return "local"
    if size < LARGE_REGION:
        return "medium"
    return "large"


def direction(delta: dict) -> str:
    excised, installed = delta["excised_atoms"], delta["installed_atoms"]
    if excised == 0 and installed == 0:
        return "restate"
    if installed == 0:
        return "excise"
    if excised == 0:
        return "grow"
    ratio = installed / max(excised, 1)
    if ratio > 2.0:
        return "grow_dominant"
    if ratio < 0.5:
        return "excise_dominant"
    return "balanced_replace"


def sites(delta: dict) -> str:
    groups = delta["n_anchor_groups"]
    if groups <= 1:
        return "single_site"
    if groups == 2:
        return "two_site"
    return "many_site"


def family(delta: dict) -> str:
    """One generic macro family.  Task-independent by construction."""
    if delta.get("status") != "ok":
        return "F_unclassified"
    band, direct, site = scale_band(delta), direction(delta), sites(delta)
    ring = delta["installed_region_has_ring"] or delta["excised_region_has_ring"]
    if band == "none":
        return "F0_restate_only"
    if site == "single_site":
        if band == "local":
            return "F1_local_single_site_edit"
        return "F2_single_site_substituent_exchange"
    if band == "local":
        return "F3_distributed_local_decoration"
    if direct in ("balanced_replace", "grow_dominant", "excise_dominant") and ring:
        return "F4_multi_site_scaffold_replacement"
    return "F5_multi_site_large_edit"


FAMILIES = (
    "F0_restate_only",
    "F1_local_single_site_edit",
    "F2_single_site_substituent_exchange",
    "F3_distributed_local_decoration",
    "F4_multi_site_scaffold_replacement",
    "F5_multi_site_large_edit",
    "F_unclassified",
)


# ---- Generic requirements the census establishes ------------------------------

REQUIREMENTS = {
    "region_scale_ge_6": lambda d: d["largest_changed_region"] >= 6,
    "region_scale_ge_13": lambda d: d["largest_changed_region"] >= LARGE_REGION,
    "multi_site_ge_2_anchor_groups": lambda d: d["n_anchor_groups"] >= 2,
    "multi_site_ge_3_anchor_groups": lambda d: d["n_anchor_groups"] >= 3,
    "installs_ring_content": lambda d: d["installed_region_has_ring"],
    "excises_ring_content": lambda d: d["excised_region_has_ring"],
    "both_excises_and_installs": lambda d: d["excised_atoms"] > 0 and d["installed_atoms"] > 0,
    "ring_count_changes": lambda d: d["ring_count_delta"] != 0,
}

JOINT = ("region_scale_ge_6", "multi_site_ge_2_anchor_groups", "installs_ring_content")


def rate(rows: list[dict], predicate) -> tuple[float, int, int]:
    usable = [r["delta"] for r in rows if r.get("delta", {}).get("status") == "ok"]
    if not usable:
        return 0.0, 0, 0
    hits = sum(1 for d in usable if predicate(d))
    return hits / len(usable), hits, len(usable)


def wilson(hits: int, total: int) -> tuple[float, float]:
    """95% interval, so a rate near 0 or 1 is not quoted as exact."""
    if total == 0:
        return 0.0, 0.0
    z = 1.959963985
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def coverage_curve(rows: list[dict]) -> list[dict]:
    counts = collections.Counter(family(r["delta"]) for r in rows if r.get("delta"))
    total = sum(counts.values())
    curve, cumulative = [], 0
    for index, (name, count) in enumerate(counts.most_common(), start=1):
        cumulative += count
        curve.append(
            {
                "families_used": index,
                "family_added": name,
                "count": count,
                "cumulative_coverage": round(cumulative / total, 4),
            }
        )
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", default=os.path.join(OUT, "productive_basin_atlas_v1.json"))
    parser.add_argument("--proposals", default=os.path.join(OUT, "proposal_delta_census_v1.json"))
    parser.add_argument("--out", default=os.path.join(OUT, "macro_grammar_v1.json"))
    args = parser.parse_args()

    atlas = json.load(open(args.atlas))["records"]
    proposals = json.load(open(args.proposals))["records"]

    populations: dict[str, list[dict]] = collections.defaultdict(list)
    for row in atlas:
        populations[row["pool"]].append(row)
    for row in proposals:
        if row.get("delta"):
            populations[row["pool"]].append(row)
            if row.get("channel"):
                populations[f"{row['pool']}::{row['channel']}"].append(row)

    print("[strata] populations with a computed delta")
    for name in sorted(populations):
        usable = sum(
            1 for r in populations[name] if r.get("delta", {}).get("status") == "ok"
        )
        print(f"  {name:56s} {usable}")

    report: dict = {}
    print("\n=== GENERIC MACRO FAMILY COVERAGE ===")
    for name in sorted(populations):
        rows = [r for r in populations[name] if r.get("delta", {}).get("status") == "ok"]
        if len(rows) < 10:
            continue
        counts = collections.Counter(family(r["delta"]) for r in rows)
        curve = coverage_curve(rows)
        report.setdefault(name, {})["n"] = len(rows)
        report[name]["family_counts"] = dict(counts)
        report[name]["coverage_curve"] = curve
        print(f"\n{name}  n={len(rows)}")
        for point in curve:
            print(
                f"   top-{point['families_used']}  {point['cumulative_coverage']:.3f}"
                f"   (+{point['family_added']} {point['count']})"
            )

    print("\n=== GENERIC REQUIREMENT SATISFACTION RATES (95% Wilson) ===")
    header = f"{'requirement':34s}" + "".join(
        f"{n.replace('POOL_','').replace('PROPOSED_','')[:18]:>20s}"
        for n in [
            "POOL_1_MEASURED_VLOCAL",
            "POOL_3_CONSTRUCTIBLE_TRANSFER",
            "NULL_RANDOM_PAIR",
            "PROPOSED_CHARGED",
            "PROPOSED_DISCARDED",
        ]
    )
    print(header)
    for key, predicate in REQUIREMENTS.items():
        line = f"{key:34s}"
        for name in [
            "POOL_1_MEASURED_VLOCAL",
            "POOL_3_CONSTRUCTIBLE_TRANSFER",
            "NULL_RANDOM_PAIR",
            "PROPOSED_CHARGED",
            "PROPOSED_DISCARDED",
        ]:
            value, hits, total = rate(populations.get(name, []), predicate)
            low, high = wilson(hits, total)
            report.setdefault(name, {}).setdefault("requirements", {})[key] = {
                "rate": round(value, 4),
                "hits": hits,
                "n": total,
                "ci95": [round(low, 4), round(high, 4)],
            }
            line += f"{value * 100:17.1f}%  "
        print(line)

    joint_predicate = lambda d: all(REQUIREMENTS[k](d) for k in JOINT)  # noqa: E731
    print(f"\nJOINT ({' AND '.join(JOINT)}):")
    for name in sorted(populations):
        value, hits, total = rate(populations[name], joint_predicate)
        if total < 10:
            continue
        low, high = wilson(hits, total)
        report.setdefault(name, {})["joint"] = {
            "rate": round(value, 4),
            "hits": hits,
            "n": total,
            "ci95": [round(low, 4), round(high, 4)],
        }
        print(f"  {name:56s} {value * 100:6.2f}%  ({hits}/{total})")

    payload = {
        "schema_version": "pmo_macro_grammar_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "information_regime": "DEVELOPMENT_INFORMED_DIAGNOSTIC",
        "information_regime_statement": (
            "Every family and requirement below is a predicate over the global graph delta "
            "alone. No task identity, target structure or fragment content enters any "
            "definition. Nothing here may become a prior, library or initialization for a "
            "scored no-prescreen run without a separate authorization."
        ),
        "new_oracle_calls": 0,
        "families": list(FAMILIES),
        "requirement_definitions": {k: "predicate over global delta" for k in REQUIREMENTS},
        "joint_requirement": list(JOINT),
        "cut_points": {"SMALL_REGION": SMALL_REGION, "LARGE_REGION": LARGE_REGION},
        "populations": report,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
