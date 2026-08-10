#!/usr/bin/env python
"""Pre-freeze gate for the post-split training identities.

Five properties must hold before any GPU time is spent, checked against the
artifacts themselves rather than against the reports that describe them.

    1. every family carries positive law mass
    2. every declared capability cell is positive IN THE REALIZED SEQUENCE
    3. every positive-weight (family, lane) stratum supports a stable estimate
    4. no reserve source appears anywhere in the training sequence
    5. coefficients RECOUNTED from the epoch sequence equal the frozen law

Point 5 is the one with history: a previous law existed on paper and not in
execution -- realized synthetic came out at 44.6%, identical to the library,
because the allocator's real-preference step had been capping the real draw at
its available share. A law that is never recounted from the sequence it
produced can claim anything.

WHAT "ENOUGH VALIDATION EXAMPLES" MEANS HERE
--------------------------------------------
Not a bare row count. The primary metric is a WEIGHTED mean, so a stratum
destabilises it in proportion to w/sqrt(n), not 1/sqrt(n). MEASURED:
ring_system_restate|real holds 43 reserve entries, which looks alarming until
its 1.11% law weight is applied -- its contribution to the standard error of
the weighted mean is about 0.004 nats, against checkpoint differences of order
0.1. Gating on raw n would either reject a perfectly usable panel or force
enlarging a stratum that is already scarce on the training side.

So the gate bounds the standard error of the primary metric itself, and thin
strata are additionally DECLARED so a per-stratum reading is never mistaken for
a stable one.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
from pathlib import Path

SYNTHETIC_LANE = "reversible_synthetic_walk"
#: Conservative per-entry NLL spread, near the observed panel dispersion. Used
#: only to turn counts into a standard-error bound before any scoring exists.
ASSUMED_NLL_SIGMA = 2.5
#: The primary metric must be resolvable well below checkpoint-to-checkpoint
#: differences, which run around 0.1 nats.
MAXIMUM_WEIGHTED_STANDARD_ERROR = 0.05
#: Below this a per-stratum mean is declared unstable. It does not gate.
THIN_STRATUM_ENTRIES = 100
#: Law mass allowed to sit in strata the reserve cannot measure. This bounds
#: the BIAS of the primary metric, where the SE bound above bounds its noise.
MAXIMUM_UNREPRESENTED_LAW_MASS = 1e-3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--law", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--reserve-ids", type=Path, required=True)
    parser.add_argument("--train-library", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--precedence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    law = json.loads(args.law.read_text())
    manifest = json.loads(args.manifest.read_text())
    sequence = json.loads(args.sequence.read_text())["sequence"]
    with gzip.open(args.reserve_ids, "rt") as handle:
        reserve = json.load(handle)
    reserve_ids = set(reserve["reserve_entry_ids"])
    reserve_sources = set(reserve["reserve_source_keys"])
    training_ids = set(reserve["training_entry_ids"])
    with gzip.open(args.provenance, "rt") as handle:
        lane_by_entry = json.load(handle)["lane_by_entry_id"]

    family_of: dict[str, str] = {}
    cell_of: dict[str, str] = {}
    source_of: dict[str, str] = {}
    lane_of: dict[str, str] = {}
    reserve_cells: collections.Counter = collections.Counter()
    reserve_strata: collections.Counter = collections.Counter()
    declared_cells: set[str] = set()
    seen: set[str] = set()
    with gzip.open(args.train_library, "rt") as handle:
        for line in handle:
            entry = json.loads(line)
            entry_id = str(entry["p50_entry_sha256"])
            if entry_id in seen:
                continue
            seen.add(entry_id)
            family = str(entry["model_family"])
            cell = str(entry["capability_cell_id"])
            declared_cells.add(cell)
            lane = ("synthetic" if lane_by_entry.get(entry_id) == SYNTHETIC_LANE
                    else "real")
            family_of[entry_id] = family
            cell_of[entry_id] = cell
            lane_of[entry_id] = lane
            source_of[entry_id] = str(
                (entry.get("teacher_successor_fiber") or {}).get("source_key", ""))
            if entry_id in reserve_ids:
                reserve_cells[cell] += 1
                reserve_strata[f"{family}|{lane}"] += 1

    held_out_sources = set(json.loads(args.precedence.read_text())
                           ["excluded_source_keys"].get("train", []))
    eligible_ids = {i for i in seen if source_of.get(i) not in held_out_sources}

    # --- 1. families ---------------------------------------------------------
    law_family = law["realized_coefficients"]["by_family"]
    families = sorted({family_of[i] for i in family_of})
    families_positive = sorted(f for f in families if law_family.get(f, 0.0) > 0)

    # --- 2. capability cells, in the SEQUENCE not the library ----------------
    drawn = collections.Counter(sequence)
    sequence_cells = {cell_of[i] for i in drawn if i in cell_of}
    starved_cells = sorted(declared_cells - sequence_cells)

    # --- 5. recount (family, lane) coefficients from the sequence ------------
    law_strata = {f'{s["family"]}|{s["provenance"]}': s["draw_share"]
                  for s in law["strata"]}
    counted: collections.Counter = collections.Counter()
    for entry_id in sequence:
        counted[f"{family_of[entry_id]}|{lane_of[entry_id]}"] += 1
    recounted = {k: v / len(sequence) for k, v in counted.items()}
    stratum_drift = {
        k: round(recounted.get(k, 0.0) - law_strata.get(k, 0.0), 6)
        for k in sorted(set(law_strata) | set(recounted))
    }
    worst_drift = max((abs(v) for v in stratum_drift.values()), default=0.0)

    # --- 3. stability of the PRIMARY metric ----------------------------------
    # Weights come from the law over positive-mass strata, renormalized: a
    # zero-mass stratum contributes nothing to the number it is excluded from.
    positive = {k: v for k, v in law_strata.items() if v > 0}
    scale = sum(positive.values())
    weights = {k: v / scale for k, v in positive.items()}
    variance = 0.0
    contributions = {}
    missing_from_reserve = []
    # A stratum absent from the reserve cannot be measured, so it is dropped and
    # the remaining weights renormalize. What matters is not WHETHER any are
    # dropped but how much law mass goes with them: the bias in the primary
    # metric is bounded by that mass times the omitted strata's NLL.
    #
    # MEASURED: cycle_insert|real has a supply of exactly ONE row in the whole
    # corpus, oversampled 3x, carrying 2.2e-05 of the draw, and that single
    # source sits on the training side. Refusing the freeze over it would demand
    # validation representation the corpus cannot supply, for a stratum that
    # cannot move the metric by 1e-4 nats.
    for stratum, weight in sorted(weights.items()):
        n = reserve_strata.get(stratum, 0)
        if n == 0:
            missing_from_reserve.append(stratum)
            continue
        term = (weight ** 2) * (ASSUMED_NLL_SIGMA ** 2) / n
        variance += term
        contributions[stratum] = {
            "law_weight": round(weight, 6),
            "reserve_entries": n,
            "standard_error_contribution": round(math.sqrt(term), 5),
        }
    weighted_standard_error = math.sqrt(variance)
    unrepresented_mass = sum(weights[s] for s in missing_from_reserve)
    thin_strata = sorted(s for s in weights if 0 < reserve_strata.get(s, 0)
                         < THIN_STRATUM_ENTRIES)

    # --- 4. reserve sources must not appear in the training sequence ---------
    sequence_sources = {source_of[i] for i in drawn if i in source_of}
    leaked_sources = sorted(sequence_sources & reserve_sources)
    leaked_ids = sorted(set(sequence) & reserve_ids)

    gates = {
        "all_families_positive": len(families_positive) == len(families) == 8,
        "all_declared_cells_positive_in_sequence": not starved_cells,
        "primary_metric_standard_error_within_bound":
            weighted_standard_error <= MAXIMUM_WEIGHTED_STANDARD_ERROR,
        "unrepresented_law_mass_within_bound":
            unrepresented_mass <= MAXIMUM_UNREPRESENTED_LAW_MASS,
        "no_reserve_source_in_training_sequence": not leaked_sources,
        "no_reserve_entry_in_training_sequence": not leaked_ids,
        "sequence_recount_matches_law": worst_drift <= 0.005,
        "law_bound_to_this_reserve": (
            law.get("matched_validation_reserve_source_digest")
            == manifest.get("matched_validation_reserve_source_digest")),
        # The split partitions the library MINUS the precedence-excluded
        # entries, not the raw library. Comparing against the raw count made
        # this gate fail on the 19 entries the carve correctly drops.
        "training_and_reserve_partition_the_eligible_library": (
            not (training_ids & reserve_ids)
            and len(training_ids) + len(reserve_ids) == len(eligible_ids)),
    }

    report = {
        "schema": "compose.editing_v2.post_split_freeze_gate",
        "status": "FROZEN" if all(gates.values()) else "REFUSED",
        "law_frozen_sha256": law["frozen_sha256"],
        "manifest_frozen_sha256": manifest["frozen_sha256"],
        "reserve_source_digest": law.get("matched_validation_reserve_source_digest"),
        "families": len(families),
        "declared_capability_cells": len(declared_cells),
        "capability_cells_in_sequence": len(sequence_cells),
        "starved_capability_cells": starved_cells,
        "positive_weight_strata": len(weights),
        "strata_unrepresented_in_reserve": {
            s: {"law_weight": round(weights[s], 8),
                "training_supply": next(
                    (x["available_pairs"] for x in law["strata"]
                     if f'{x["family"]}|{x["provenance"]}' == s), 0)}
            for s in missing_from_reserve},
        "unrepresented_law_mass": round(unrepresented_mass, 8),
        "unrepresented_law_mass_bound": MAXIMUM_UNREPRESENTED_LAW_MASS,
        "eligible_library_entries": len(eligible_ids),
        "precedence_excluded_entries": len(seen) - len(eligible_ids),
        "zero_mass_strata": sorted(set(law_strata) - set(positive)),
        "assumed_nll_sigma": ASSUMED_NLL_SIGMA,
        "primary_metric_standard_error": round(weighted_standard_error, 5),
        "primary_metric_standard_error_bound": MAXIMUM_WEIGHTED_STANDARD_ERROR,
        "standard_error_by_stratum": contributions,
        "thin_strata_declared_not_gated": {
            s: reserve_strata.get(s, 0) for s in thin_strata},
        "thin_stratum_note": (
            "Declared, not gated. A per-stratum mean over this many entries is "
            "not a stable reading on its own, but its contribution to the "
            "weighted primary metric is bounded by its law weight and is "
            "reported above."),
        "worst_stratum_drift_from_law": round(worst_drift, 6),
        "stratum_drift": stratum_drift,
        "gates": gates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"families {len(families)}  declared cells {len(declared_cells)}  "
          f"cells in sequence {len(sequence_cells)}")
    print(f"positive-weight strata {len(weights)}  zero-mass "
          f"{report['zero_mass_strata']}")
    print(f"\n{'stratum':32} {'law w':>8} {'reserve n':>10} {'SE contrib':>11}")
    for stratum, item in sorted(contributions.items(),
                                key=lambda kv: -kv[1]["standard_error_contribution"]):
        flag = "  THIN" if stratum in thin_strata else ""
        print(f"  {stratum:30} {item['law_weight']:8.4f} "
              f"{item['reserve_entries']:10,} "
              f"{item['standard_error_contribution']:11.5f}{flag}")
    if missing_from_reserve:
        print(f"\nunrepresented in reserve: {missing_from_reserve} "
              f"carrying {unrepresented_mass:.2e} of the law")
    print(f"\nprimary-metric standard error {weighted_standard_error:.5f} nats "
          f"(bound {MAXIMUM_WEIGHTED_STANDARD_ERROR})")
    print(f"worst stratum drift from law  {worst_drift:.6f}")
    print()
    for name, passed in sorted(gates.items()):
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\n{report['status']}  ->  {args.out}")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
