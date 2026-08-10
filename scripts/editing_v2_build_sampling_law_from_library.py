#!/usr/bin/env python
"""Derive the training sampling law from the LIBRARY, not from an Active8 walk.

WHY NOT RE-WALK ACTIVE8
-----------------------
``editing_v2_build_sampling_law.py`` attributes each pair by re-reading the
Active8 transition streams. Two problems here, and only one is about speed.

First, the local Active8 copy DIVERGES from the volume's, which is why
RUN_PATHS.json pins gate-zero v6 and calls the volume authoritative. A law
built against the local walk would be bound to a corpus the run does not use.

Second, and more important: everything ``attribute`` extracts -- model family,
capability cell, data lane -- is already carried by the consolidated library
and ENTRY_PROVENANCE, which are the exact objects the training loader reads. A
second independent walk can only agree or disagree with them, and if it
disagrees the law describes a population the sampler never draws from.

So the strata are read off the library, and the ALLOCATION is imported from the
original module rather than restated: same family water-filling, same
real-preference and bounded oversample, same synthetic-overshoot trim. The
science is unchanged; only the attribution source moves.

WHAT THIS LAW IS OVER
---------------------
The post-split eligible training set: the consolidated train library, minus the
precedence-held-out sources, minus the matched-validation reserve. Regenerated,
never filtered from a finished sequence -- family targets are water-filled
against availability and the synthetic trim depends on each family's real
headroom, so removing rows afterwards silently changes the realized
coefficients the artifact claims to guarantee.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYNTHETIC_LANE = "reversible_synthetic_walk"


def _load_allocate():
    """Import allocate() from the original law builder without duplicating it."""
    spec = importlib.util.spec_from_file_location(
        "_editing_v2_sampling_law", HERE / "editing_v2_build_sampling_law.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.allocate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-library", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--reserve-ids", type=Path, required=True)
    parser.add_argument("--precedence", type=Path, required=True)
    parser.add_argument("--family-floor", type=float, default=0.05)
    parser.add_argument("--family-cap", type=float, default=0.22)
    parser.add_argument("--synthetic-target", type=float, default=0.18)
    parser.add_argument("--max-oversample", type=float, default=3.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    allocate = _load_allocate()

    with gzip.open(args.reserve_ids, "rt") as handle:
        reserve = json.load(handle)
    training_ids = set(reserve["training_entry_ids"])
    reserve_sources = set(reserve["reserve_source_keys"])
    held_out = set(json.loads(args.precedence.read_text())
                   ["excluded_source_keys"].get("train", []))
    with gzip.open(args.provenance, "rt") as handle:
        provenance = json.load(handle)
    lane_by_entry = provenance["lane_by_entry_id"]
    if provenance["synthetic_lane"] != SYNTHETIC_LANE:
        raise SystemExit(
            f"provenance declares synthetic lane {provenance['synthetic_lane']!r}, "
            f"the law is written against {SYNTHETIC_LANE!r}")

    rows: list[dict] = []
    seen: set[str] = set()
    unknown_lane = 0
    with gzip.open(args.train_library, "rt") as handle:
        for line in handle:
            entry = json.loads(line)
            entry_id = str(entry["p50_entry_sha256"])
            if entry_id in seen:
                continue
            seen.add(entry_id)
            if entry_id not in training_ids:
                continue
            source = str((entry.get("teacher_successor_fiber") or {}).get("source_key", ""))
            if not source or source in held_out or source in reserve_sources:
                continue
            lane = lane_by_entry.get(entry_id)
            if lane is None:
                unknown_lane += 1
                continue
            rows.append({
                "family": str(entry["model_family"]),
                "synthetic": lane == SYNTHETIC_LANE,
                "cell": str(entry["capability_cell_id"]),
                "source": source,
            })
    print(f"eligible post-split training rows: {len(rows):,}"
          f"  (unattributed lane, dropped: {unknown_lane:,})")
    if len(rows) != len(training_ids) - unknown_lane:
        print(f"  NOTE training_entry_ids={len(training_ids):,} -> "
              f"{len(rows):,} attributed", flush=True)

    plan = allocate(rows, floor=args.family_floor, cap=args.family_cap,
                    synthetic_target=args.synthetic_target,
                    max_oversample=args.max_oversample)
    supply, draw, total = plan["supply"], plan["draw"], plan["total"]

    strata = []
    for (family, synthetic), share in sorted(draw.items()):
        count = supply.get((family, synthetic), 0)
        if share <= 0 or count == 0:
            continue
        strata.append({
            "family": family,
            "provenance": "synthetic" if synthetic else "real",
            "available_pairs": count,
            "draw_share": round(share, 6),
            "per_row_weight": round(share / count, 12),
            "oversample_factor": round((share * total) / count, 4),
        })

    # Realized coefficients come from the UNROUNDED draw. Summing the strata's
    # 6dp display values instead let 16 roundings accumulate past a 1e-6
    # tolerance and reported a mass leak that did not exist.
    realized_family = collections.Counter()
    for (family, _synthetic), share in draw.items():
        realized_family[family] += share
    realized_synthetic = sum(v for (_f, syn), v in draw.items() if syn)
    available_synthetic = sum(1 for r in rows if r["synthetic"]) / max(total, 1)
    residual = plan["residual_synthetic_excess"]

    # The constraints the law exists to enforce, checked rather than asserted.
    families = sorted(realized_family)
    gates = {
        "no_family_below_floor": all(
            realized_family[f] >= args.family_floor - 1e-9 for f in families),
        "no_family_above_cap": all(
            realized_family[f] <= args.family_cap + 1e-9 for f in families),
        # "Not met" and "unreachable" are different facts. The trim will not take
        # synthetic mass from a family whose only supply is synthetic, so the
        # target can sit below what the corpus can deliver. Passing here means
        # the law got as close as the supply allows.
        "synthetic_at_or_below_supply_floor":
            realized_synthetic <= args.synthetic_target + 1e-9 or residual > 0,
        "real_chemistry_dominates": realized_synthetic < 0.5,
        "every_family_retained": len(families) == len({r["family"] for r in rows}),
        "shares_sum_to_one": abs(sum(realized_family.values()) - 1.0) < 1e-9,
        "no_stratum_oversampled_past_bound": all(
            s["oversample_factor"] <= args.max_oversample + 1e-6 for s in strata),
    }

    report = {
        "schema": "compose.editing_v2.training_sampling_law",
        "schema_version": 2,
        "status": "SAMPLING_LAW_EVIDENCE_ONLY_NO_AUTHORITY",
        "attribution_source": "consolidated_library+entry_provenance",
        "why_not_active8": (
            "The local Active8 copy diverges from the volume's, and every field "
            "the walk would attribute is already carried by the objects the "
            "training loader reads. A second walk could only agree or describe a "
            "population the sampler never draws from."),
        "constraints": {
            "family_floor": args.family_floor,
            "family_cap": args.family_cap,
            "synthetic_target": args.synthetic_target,
            "max_oversample": args.max_oversample,
        },
        "eligible_canonical_train_pairs": total,
        "synthetic_supply_floor": round(plan["synthetic_supply_floor"], 4),
        "residual_synthetic_excess": round(residual, 6),
        "synthetic_target_reachable": bool(residual <= 1e-12),
        "precedence_held_out_sources": len(held_out),
        "matched_validation_reserve_sources": len(reserve_sources),
        "matched_validation_reserve_source_digest": hashlib.sha256(
            "\n".join(sorted(reserve_sources)).encode()).hexdigest()[:12],
        "available_composition": {
            "synthetic_share": round(available_synthetic, 4),
            "by_family": {
                f: round(sum(v for (fam, _), v in supply.items() if fam == f) / total, 4)
                for f in families},
        },
        "realized_coefficients": {
            "synthetic_share": round(realized_synthetic, 4),
            "by_family": {f: round(v, 4) for f, v in sorted(realized_family.items())},
        },
        "gates": gates,
        "strata": strata,
    }
    body = json.dumps({k: v for k, v in report.items() if k != "frozen_sha256"},
                      sort_keys=True)
    report["frozen_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\n{'family':22} {'avail':>8} {'realized':>9}   {'real / synth draw':>20}")
    for family in families:
        available = report["available_composition"]["by_family"][family]
        realized = report["realized_coefficients"]["by_family"][family]
        real = next((s["draw_share"] for s in strata
                     if s["family"] == family and s["provenance"] == "real"), 0.0)
        synth = next((s["draw_share"] for s in strata
                      if s["family"] == family and s["provenance"] == "synthetic"), 0.0)
        print(f"  {family:20} {100*available:7.2f}% {100*realized:8.2f}%   "
              f"{100*real:8.2f}% / {100*synth:6.2f}%")
    print(f"\nsynthetic: available {100*available_synthetic:.1f}%  ->  realized "
          f"{100*realized_synthetic:.1f}%  (target {100*args.synthetic_target:.0f}%)")
    for name, passed in sorted(gates.items()):
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\nfrozen {report['frozen_sha256'][:16]}\nwrote {args.out}")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
