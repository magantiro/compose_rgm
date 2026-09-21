"""Reduce the schedule-probe artifact into the three measurements and a verdict.

The verdict is applied mechanically from the thresholds written into
PREDECLARATION.json before the run, so it cannot be tuned to the numbers after
seeing them.  Arms are also compared PAIRWISE on the same trace, because the
marginal means share every source draw and a matched delta is the stronger
statement.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

ARM_A = "A_sequential"
ARM_B = "B_exact_early_ring_default_barriers"
ARM_C = "C_exact_early_ring_barrier_relaxed"
ARMS = (ARM_A, ARM_B, ARM_C)


def _summary(values) -> dict:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None,
                "p10": None, "p90": None, "stderr": None}
    array = np.asarray(sorted(values), dtype=float)
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "median": float(median(array.tolist())),
        "min": float(array[0]),
        "max": float(array[-1]),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "stderr": float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else None,
    }


def _histogram(values) -> dict:
    counts: dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    return {str(key): counts[key] for key in sorted(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-uniform-support", type=float, default=0.44307839688899203)
    parser.add_argument("--trainer-supported-only", action="store_true",
                        help="restrict to traces the trainer's catalog filter keeps")
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text())
    events = payload["events"]
    if args.trainer_supported_only:
        events = [event for event in events if event.get("trainer_supported")]
    by_arm = defaultdict(list)
    for event in events:
        by_arm[event["arm"]].append(event)

    arms: dict[str, dict] = {}
    for arm in ARMS:
        rows = by_arm.get(arm, [])
        grow = [row for row in rows if row["rule_name"] == "ring_system_grow"]
        counters = {"attempted_swaps": 0, "accepted_swaps": 0, "traces": 0,
                    "dropped": 0, "position_report_disagreements": 0}
        for shard in payload.get("shard_results", []):
            shard_counters = shard["arm_counters"].get(arm, {})
            for key in counters:
                counters[key] += shard_counters.get(key, 0)
        arms[arm] = {
            "ring_events": len(rows),
            "ring_grow_events": len(grow),
            "rule_census": _histogram_str([row["rule_name"] for row in rows]),
            **counters,
            "m1_index": _summary([row["index"] for row in rows]),
            "m1_fraction_of_trace": _summary([row["fraction_of_trace"] for row in rows]),
            "m2_free_slots": _summary([row["free_slots"] for row in rows]),
            "m2_free_slots_histogram": _histogram([row["free_slots"] for row in rows]),
            "m3_small_ring_support_mass": _summary(
                [row["small_mass_uniform_support"] for row in grow]
            ),
            "m3_legal_template_count": _summary(
                [row["legal_template_count"] for row in grow]
            ),
            "m3_legal_small_template_count": _summary(
                [row["legal_small_template_count"] for row in grow]
            ),
        }

    # ---- Paired deltas on the same (target, ring-event ordinal) ----
    def keyed(arm: str) -> dict:
        ordinal: dict[int, int] = defaultdict(int)
        out = {}
        for row in sorted(by_arm.get(arm, []), key=lambda r: (r["offset"], r["index"])):
            index = ordinal[row["offset"]]
            ordinal[row["offset"]] += 1
            out[(row["offset"], index)] = row
        return out

    base = keyed(ARM_A)
    paired = {}
    for arm in (ARM_B, ARM_C):
        other = keyed(arm)
        shared = sorted(set(base) & set(other))
        paired[arm] = {
            "pairs": len(shared),
            "d_index": _summary([other[k]["index"] - base[k]["index"] for k in shared]),
            "d_fraction": _summary(
                [other[k]["fraction_of_trace"] - base[k]["fraction_of_trace"] for k in shared]
            ),
            "d_free_slots": _summary(
                [other[k]["free_slots"] - base[k]["free_slots"] for k in shared]
            ),
            "events_that_moved": sum(
                1 for k in shared if other[k]["index"] != base[k]["index"]
            ),
            "events_gaining_free_slots": sum(
                1 for k in shared if other[k]["free_slots"] > base[k]["free_slots"]
            ),
            "d_support_mass": _summary(
                [
                    other[k]["small_mass_uniform_support"] - base[k]["small_mass_uniform_support"]
                    for k in shared
                    if other[k]["small_mass_uniform_support"] is not None
                    and base[k]["small_mass_uniform_support"] is not None
                ]
            ),
        }

    # ---- Predeclared decision rule ----
    def arm_verdict(arm: str) -> dict:
        a, x = arms[ARM_A], arms[arm]
        moved = (
            a["m1_fraction_of_trace"]["median"] is not None
            and x["m1_fraction_of_trace"]["median"] is not None
            and (a["m1_fraction_of_trace"]["median"] - x["m1_fraction_of_trace"]["median"]) >= 0.10
        )
        slots = (
            a["m2_free_slots"]["mean"] is not None
            and x["m2_free_slots"]["mean"] is not None
            and (x["m2_free_slots"]["mean"] - a["m2_free_slots"]["mean"]) >= 1.0
        )
        crashed = (
            x["m3_small_ring_support_mass"]["mean"] is not None
            and x["m3_small_ring_support_mass"]["mean"] <= 0.20
        )
        inert = x["accepted_swaps"] == 0 or (
            a["m1_index"]["median"] == x["m1_index"]["median"]
            and paired[arm]["events_that_moved"] == 0
        )
        return {
            "moved_materially": bool(moved),
            "free_slots_rose_materially": bool(slots),
            "support_mass_crashed": bool(crashed),
            "inert": bool(inert),
            "achieves_repair": bool(moved and slots and crashed),
        }

    verdicts = {arm: arm_verdict(arm) for arm in (ARM_B, ARM_C)}
    # A run with no ring events has not tested anything.  Emitting
    # FIX_COMPILER_ORDERING from an empty arm would dress a failed measurement up
    # as a clean negative, which is the single most expensive mistake this probe
    # could make -- it is the branch that tells the owner NOT to retrain.  The
    # smoke did exactly this from 0 compiled traces before this guard existed.
    grow_a = arms[ARM_A]["ring_grow_events"]
    if grow_a == 0 or payload["cost_counters"]["traces_compiled"] == 0:
        decision = "INCONCLUSIVE_NO_DATA"
    elif verdicts[ARM_B]["achieves_repair"]:
        decision = "RETRAIN"
    elif verdicts[ARM_C]["achieves_repair"]:
        decision = "RELAX_BARRIER"
    else:
        decision = "FIX_COMPILER_ORDERING"

    instrument = payload.get("instrument_check")
    instrument_verdict = None
    if instrument:
        instrument_verdict = {
            "rows_agreeing": instrument["rows_agreeing"],
            "rows_total": instrument["rows_total"],
            "passes": instrument["rows_agreeing"] == instrument["rows_total"],
            "catalog_template_count": instrument["catalog_template_count"],
            "catalog_small_template_count": instrument["catalog_small_template_count"],
        }

    out = {
        "probe": "denovo_ring_schedule_probe_v1",
        "artifact": str(args.artifact),
        "trains_nothing": True,
        "oracle_calls": 0,
        "realized_n_targets_compiled": payload["cost_counters"]["traces_compiled"],
        "trainer_supported_traces": sum(
            shard.get("trainer_supported_traces", 0)
            for shard in payload.get("shard_results", [])
        ),
        "restricted_to_trainer_supported": bool(args.trainer_supported_only),
        "compile_failures": payload["cost_counters"]["compile_failures"],
        "instrument_check": instrument_verdict,
        "instrument_reference_population_note": (
            "The reference 0.443 is a mean over 12 ROLLOUT states under step-2500. "
            "Arm A measures TRAINING-TRACE states. A difference between them is a "
            "population fact; the row-by-row replication above is what validates the "
            "enumeration itself."
        ),
        "reference_uniform_support": args.reference_uniform_support,
        "arms": arms,
        "paired_vs_arm_a": paired,
        "predeclared_verdicts": verdicts,
        "decision": decision,
        "cost_counters": payload["cost_counters"],
        "wall_seconds": payload["wall_seconds"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision,
                      "instrument": instrument_verdict,
                      "armA_support": arms[ARM_A]["m3_small_ring_support_mass"],
                      "armB_support": arms[ARM_B]["m3_small_ring_support_mass"],
                      "armC_support": arms[ARM_C]["m3_small_ring_support_mass"]}, indent=1))


def _histogram_str(values) -> dict:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


if __name__ == "__main__":
    main()
