#!/usr/bin/env python3
"""Aggregate pathwise-constraint shards into gate verdicts. LOCAL ONLY.

The aggregate is fully reconstructible from the per-source shards written by
`modal_apps/pathwise_constraints_app.py`; nothing here needs the kernel.

WHAT THIS SCRIPT REFUSES TO REPORT AS A FINDING
-----------------------------------------------
"the pathwise arms had zero violations". That is the construction. It is
checked here as a BUG DETECTOR -- a non-zero count voids the run -- and is
never emitted as a headline number, never given a denominator that suggests it
was estimated, and never given a p-value.

Likewise `pathwise_verified >= pathwise_greedy` in utility and in binary
success. Greedy's action is always in the verified candidate set and strict
improvement never commits a lower V_G, so the sign is fixed by the policy-
improvement theorem before any molecule exists. Only the MAGNITUDE of the gap,
the TOP-1 DISAGREEMENT rate, and binary headroom over the denominator of
sources where greedy actually FAILED are admissible.

Every gate below can come out the other way. That is what makes it a gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

VACUITY_FLOOR = 0.10        # below this, the constraint is vacuous
MASK_VACUOUS_CEILING = 0.02  # mask removes ~nothing
MASK_TOTAL_FLOOR = 0.95      # mask removes ~everything


def summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True,
                        help="directory of per-source shard JSON files")
    parser.add_argument("--out", type=Path,
                        default=REPO / "diagnostics/pathwise_constraints_smoke.json")
    parser.add_argument("--panel", type=Path,
                        default=REPO / "diagnostics/pathwise_constraints_smoke_panel.json")
    args = parser.parse_args()

    shards = [json.loads(p.read_text()) for p in sorted(args.shards.glob("*.json"))]
    if not shards:
        raise SystemExit(f"no shards under {args.shards}")

    void = [s for s in shards if s.get("status") == "INVALID_INSTRUMENT"]
    good = [s for s in shards if s.get("status") == "SMOKE_HELD_IN"]
    if not good:
        raise SystemExit(f"every shard is INVALID_INSTRUMENT ({len(void)})")

    arm_names = sorted({name for s in good for name in s["arms"]})

    # ---------------------------------------------------------------- G0 bug
    # Definitional, not a finding. Any leak voids the run.
    leaks = {s["index"]: s["mask_leak"] for s in good if s.get("mask_leak")}

    # ------------------------------------------------- G1 vacuity of the motif
    unconstrained = [s["arms"]["unconstrained_greedy"] for s in good
                     if "unconstrained_greedy" in s["arms"]]
    ever_violates = sum(1 for a in unconstrained if a["audit"]["any_violation"])
    endpoint_valid_path_invalid = sum(
        1 for a in unconstrained if a["audit"]["endpoint_valid_path_invalid"])

    # The larger and cleaner denominator: every ENDPOINT-VALID rollout that
    # endpoint-only filtering would have been willing to return.
    admissible, admissible_path_invalid = 0, 0
    for shard in good:
        arm = shard["arms"].get("endpoint_only")
        for roll in (arm or {}).get("rollout_audits", []):
            if roll["audit"]["endpoint_valid"]:
                admissible += 1
                admissible_path_invalid += int(roll["audit"]["any_violation"])
    selected_path_invalid = sum(
        1 for s in good
        if s["arms"].get("endpoint_only", {}).get(
            "audit", {}).get("endpoint_valid_path_invalid"))
    endpoint_only_present = sum(1 for s in good if "endpoint_only" in s["arms"])

    # -------------------------------------------- G2 how much support is left
    removed = [c["removed_fraction"] for s in good for c in s["mask_census"]]
    removed_mass = [
        c["removed_reference_mass"] / (c["removed_reference_mass"] + c["kept_reference_mass"])
        for s in good for c in s["mask_census"]
        if (c["removed_reference_mass"] + c["kept_reference_mass"]) > 0
    ]
    empty_support = sum(1 for s in good for c in s["mask_census"] if c["kept"] == 0)
    census_states = sum(len(s["mask_census"]) for s in good)

    # ------------------------------------------------------- G3 feasibility
    feasibility = {}
    for name in arm_names:
        arms = [s["arms"][name] for s in good if name in s["arms"]]
        if not arms:
            continue
        improved = [
            a["b_worst_margin"] - s["b_worst_at_source"]
            for s, a in ((s, s["arms"][name]) for s in good if name in s["arms"])
        ]
        feasibility[name] = {
            "sources": len(arms),
            "completed": sum(1 for a in arms if a["completed"]),
            "selection_failed": sum(1 for a in arms if a.get("selection_failed")),
            "b_success": sum(1 for a in arms if a["b_success"]),
            "endpoint_motif_valid": sum(1 for a in arms if a["audit"]["endpoint_valid"]),
            "b_worst_margin": summarise([a["b_worst_margin"] for a in arms]),
            "improvement_over_source": summarise(improved),
            "marginal_kernel_calls": summarise(
                [float(a["marginal_kernel_calls"]) for a in arms]),
            # DEFINITIONAL for masked arms. Recorded, not headlined.
            "any_violation_DEFINITIONAL_IF_MASKED": sum(
                1 for a in arms if a["audit"]["any_violation"]),
        }

    # ------------------------------- G4 are the treatments distinguishable?
    distinct = [s["distinct_landings"] for s in good]
    paired_mask_price = []
    for shard in good:
        free = shard["arms"].get("endpoint_only")
        held = shard["arms"].get("pathwise_stochastic")
        if free and held and not free.get("selection_failed"):
            paired_mask_price.append(held["b_worst_margin"] - free["b_worst_margin"])
    endpoint_vs_pathwise_differ = sum(
        1 for s in good
        if "endpoint_only" in s["arms"] and "pathwise_stochastic" in s["arms"]
        and s["arms"]["endpoint_only"]["landing"]
        != s["arms"]["pathwise_stochastic"]["landing"])

    # ----------------------------------------- planning inside the mask (B)
    planning = None
    if all(n in arm_names for n in ("pathwise_greedy", "pathwise_verified")):
        pairs = [(s["arms"]["pathwise_greedy"], s["arms"]["pathwise_verified"])
                 for s in good
                 if "pathwise_greedy" in s["arms"] and "pathwise_verified" in s["arms"]]
        greedy_failed = [(g, v) for g, v in pairs if not g["b_success"]]
        planning = {
            "sources": len(pairs),
            "GUARANTEED_SIGN_WARNING": (
                "verified >= greedy in utility AND in binary success is the "
                "policy-improvement theorem, not a measurement. Only magnitude, "
                "top-1 disagreement, and headroom over the greedy-failure "
                "denominator are admissible."
            ),
            "top1_disagreements": summarise(
                [float(v["top1_disagreements"] or 0) for _g, v in pairs]),
            "overrides": summarise([float(v["overrides"] or 0) for _g, v in pairs]),
            "utility_gap_magnitude": summarise(
                [v["b_worst_margin"] - g["b_worst_margin"] for g, v in pairs]),
            "binary_headroom": {
                "denominator_greedy_failures": len(greedy_failed),
                "verified_rescued": sum(1 for _g, v in greedy_failed if v["b_success"]),
                "note": ("headroom 0 with denominator 0 is a CEILING, not a null: "
                         "a real planning advantage had nowhere to show"),
            },
        }

    def verdict(condition_pass: bool, condition_fail: bool) -> str:
        if condition_pass:
            return "PASS"
        return "FAIL" if condition_fail else "INCONCLUSIVE"

    n = len(unconstrained) or 1
    violation_rate = ever_violates / n
    path_invalid_rate = endpoint_valid_path_invalid / n
    admissible_rate = admissible_path_invalid / admissible if admissible else 0.0
    removed_mean = statistics.fmean(removed) if removed else 0.0

    gates = {
        "G0_mask_integrity": {
            "verdict": "FAIL" if leaks else "PASS",
            "kind": "BUG_DETECTOR_NOT_A_FINDING",
            "leaks": leaks or None,
        },
        "G1_constraint_non_vacuous": {
            "verdict": verdict(
                max(path_invalid_rate, admissible_rate) >= VACUITY_FLOOR,
                max(path_invalid_rate, admissible_rate) < VACUITY_FLOOR),
            "unconstrained_any_violation": f"{ever_violates}/{n}",
            "unconstrained_endpoint_valid_path_invalid": (
                f"{endpoint_valid_path_invalid}/{n}"),
            "endpoint_only_selected_path_invalid": (
                f"{selected_path_invalid}/{endpoint_only_present}"),
            "endpoint_valid_rollouts_that_are_path_invalid": (
                f"{admissible_path_invalid}/{admissible}"),
            "threshold": VACUITY_FLOOR,
            "note": ("this is the headline measurement: it is 0 whenever "
                     "endpoint-only filtering is already sufficient"),
        },
        "G2_mask_leaves_room_to_act": {
            "verdict": verdict(
                MASK_VACUOUS_CEILING < removed_mean < MASK_TOTAL_FLOOR,
                not (MASK_VACUOUS_CEILING < removed_mean < MASK_TOTAL_FLOOR)),
            "removed_fraction_along_unconstrained_states": summarise(removed),
            "removed_reference_mass_fraction": summarise(removed_mass),
            "states_with_empty_masked_support": f"{empty_support}/{census_states}",
            "thresholds": {"vacuous_at_or_below": MASK_VACUOUS_CEILING,
                           "infeasible_at_or_above": MASK_TOTAL_FLOOR},
        },
        "G3_feasible_paths_can_improve_the_objective": {
            "verdict": verdict(
                any(f["improvement_over_source"].get("max", 0) > 0
                    for name, f in feasibility.items() if name.startswith("pathwise")),
                all(f["improvement_over_source"].get("max", 0) <= 0
                    for name, f in feasibility.items() if name.startswith("pathwise"))),
            "per_arm": {name: f["improvement_over_source"]
                        for name, f in feasibility.items()},
        },
        "G4_endpoint_only_and_pathwise_are_distinguishable": {
            "verdict": verdict(
                endpoint_vs_pathwise_differ > 0, endpoint_vs_pathwise_differ == 0),
            "landings_differ": f"{endpoint_vs_pathwise_differ}/{len(good)}",
            "distinct_landings_per_source": summarise([float(d) for d in distinct]),
        },
    }

    payload = {
        "schema": "compose.pathwise.smoke_analysis",
        "status": "INVALID_INSTRUMENT" if leaks else "SMOKE_HELD_IN",
        "held_out_opened": False,
        "shards": len(shards),
        "shards_void": len(void),
        "void_reasons": [s.get("error") for s in void] or None,
        "panel_sha256": json.loads(args.panel.read_text())["panel_sha256"]
        if args.panel.exists() else None,
        "arms": arm_names,
        "gates": gates,
        "price_of_the_guarantee": {
            "comparison": "pathwise_stochastic minus endpoint_only, terminal worst margin",
            "why_admissible": ("identical rollout budget, identical policy; the "
                               "mask is the only difference, so the sign is free"),
            "paired_delta": summarise(paired_mask_price),
        },
        "planning_inside_the_mask": planning,
        "per_arm": feasibility,
        "cost": {
            "kernel_calls": summarise([float(s["kernel_calls"]) for s in good]),
            "seconds": summarise([float(s["seconds"]) for s in good]),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
