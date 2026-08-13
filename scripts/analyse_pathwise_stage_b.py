#!/usr/bin/env python3
"""Stage B estimands: cost and benefit of enforcing the corridor throughout.

THE SOURCE IS THE INDEPENDENT UNIT. Every interval is a source-clustered
bootstrap. Pooled trajectory statistics are reported without intervals.

WHAT IS BARRED FROM THE RESULTS TABLE
-------------------------------------
"The pathwise arms achieved zero violations." That is the construction. It is
checked as a bug detector -- a non-zero count voids the shard -- and never
given a denominator, an interval, or a p-value.

"Verified never loses to greedy under the same support." Verified contains
greedy's action and overrides only on strict improvement, so the direction is
fixed before any molecule exists. No sign test is computed. Only effect size
and constrained-performance recovered are reported, and binary headroom is
given over the denominator of sources where greedy actually failed -- a
headroom of 0 over a denominator of 0 is a CEILING, not a null.

THE PRIMARY ESTIMAND, which can come out zero
---------------------------------------------
    P( exists t < H : x_t not in C  |  x_H in C )
computed on the endpoint-only arms, whose conditioning set is the trajectories
that actually delivered a compliant product. The ring-motif family returned
exactly zero here; that is what makes it a measurement.

SUPPORT-TIGHT HANDLING, predeclared before the run
--------------------------------------------------
A source is SUPPORT_TIGHT when its median retained legal-successor fraction is
below 0.10, measured along the DESCRIPTIVE arm's states. All sources stay in
the primary intention-to-treat analysis; the sensitivity analysis excluding
them is reported alongside, and the threshold is not redefined here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SUPPORT_TIGHT_THRESHOLD = 0.10
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260815

PAIRS = {
    "delta_G_greedy_parity": ("pathwise_greedy", "endpoint_greedy"),
    "delta_V_verified_parity": ("pathwise_verified", "endpoint_verified"),
}


def summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "mean": round(statistics.fmean(ordered), 4),
            "median": round(statistics.median(ordered), 4),
            "min": round(ordered[0], 4), "max": round(ordered[-1], 4)}


def cluster_bootstrap(per_source: list[float], draws: int = BOOTSTRAP_DRAWS) -> dict:
    if not per_source:
        return {"n_sources": 0}
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(per_source)
    means = []
    for _ in range(draws):
        means.append(statistics.fmean(per_source[rng.randrange(n)] for _ in range(n)))
    means.sort()
    return {"n_sources": n, "point": round(statistics.fmean(per_source), 4),
            "ci95_low": round(means[int(0.025 * draws)], 4),
            "ci95_high": round(means[int(0.975 * draws)], 4),
            "draws": draws, "unit": "source"}


def paired_delta(shards: list[dict], treated: str, control: str) -> dict:
    """U_P(treated) - U_P(control), paired within source.

    Both arms share a controller and a source and differ only in WHERE the
    constraint is enforced, so the sign is free. A negative value is a
    legitimate finding: a pathwise requirement removes options.
    """
    deltas, pairs = [], 0
    for shard in shards:
        arms = shard["arms"]
        if treated not in arms or control not in arms:
            continue
        pairs += 1
        deltas.append(arms[treated]["U_P"] - arms[control]["U_P"])
    return {"pairs": pairs, "summary": summarise(deltas),
            "bootstrap": cluster_bootstrap(deltas),
            "sign_is_free": True,
            "note": ("controller parity: the two arms differ only in where the "
                     "constraint is enforced")}


def analyse(shards: list[dict], label: str) -> dict:
    # ---- primary and secondary pathwise estimands -------------------------
    #
    # PRIMARY is the UNCONDITIONAL hidden-path rate. Its denominator is every
    # eligible source, so it cannot collapse. The conditional fraction is
    # intuitive but its denominator is CONTROLLER-DEPENDENT: an arm that
    # rarely delivers an acceptable endpoint can post a dramatic-looking rate
    # on a handful of trajectories. Both are reported; the conditional one
    # always carries its denominator.
    #
    # Reported SEPARATELY for greedy and verified endpoint-only control and
    # never pooled, because the controller changes which endpoints become
    # acceptable and pooling would mix two different denominators.
    estimands: dict[str, dict] = {}
    for arm in ("endpoint_greedy", "endpoint_verified"):
        eligible, delivered, hidden = 0, 0, 0
        uncond_per_source, cond_per_source = [], []
        for shard in shards:
            body = shard["arms"].get(arm)
            if body is None:
                continue
            eligible += 1
            in_c = bool(body["endpoint_in_C"]) and not body.get(
                "endpoint_terminal_infeasible")
            hit = in_c and bool(body["any_intermediate_violation"])
            hidden += int(hit)
            uncond_per_source.append(1.0 if hit else 0.0)
            if in_c:
                delivered += 1
                cond_per_source.append(1.0 if hit else 0.0)
        estimands[arm] = {
            "PRIMARY_hidden_path_rate": {
                "definition": "P(x_H in C AND exists t<H: x_t not in C)",
                "numerator": hidden,
                "denominator_all_eligible_sources": eligible,
                "estimate": round(hidden / eligible, 4) if eligible else None,
                "bootstrap": cluster_bootstrap(uncond_per_source),
                "denominator_is_controller_independent": True,
                "can_be_zero": True,
            },
            "SECONDARY_hidden_path_fraction": {
                "definition": "P(exists t<H: x_t not in C | x_H in C)",
                "numerator": hidden,
                "denominator_accepted_endpoints": delivered,
                "estimate": round(hidden / delivered, 4) if delivered else None,
                "bootstrap": cluster_bootstrap(cond_per_source),
                "DENOMINATOR_IS_CONTROLLER_DEPENDENT": True,
                "warning": ("read only with the denominator: a small "
                            "denominator can make this look dramatic"),
            },
            "endpoints_not_in_C": eligible - delivered,
        }

    # ---- terminal cost ----------------------------------------------------
    cost = {name: paired_delta(shards, t, c) for name, (t, c) in PAIRS.items()}

    # ---- future-aware, guaranteed sign ------------------------------------
    gains, greedy_failures, rescued, disagreements = [], 0, 0, []
    recoveries = []
    for shard in shards:
        arms = shard["arms"]
        if "pathwise_verified" not in arms or "pathwise_greedy" not in arms:
            continue
        gain = arms["pathwise_verified"]["U_P"] - arms["pathwise_greedy"]["U_P"]
        gains.append(gain)
        disagreements.append(float(arms["pathwise_verified"].get(
            "top1_disagreements") or 0))
        if not arms["pathwise_greedy"]["completed"]:
            greedy_failures += 1
            if arms["pathwise_verified"]["completed"]:
                rescued += 1
        if "endpoint_greedy" in arms:
            mask_cost = (arms["pathwise_greedy"]["U_P"]
                         - arms["endpoint_greedy"]["U_P"])
            if mask_cost < 0:
                recoveries.append(gain / (-mask_cost))
    planning = {
        "GUARANTEED_SIGN_WARNING": (
            "verified contains greedy's action and overrides only on strict "
            "improvement, so U_P(pathwise verified) >= U_P(pathwise greedy) holds "
            "by policy improvement. No sign test is computed. Only effect size, "
            "top-1 disagreement and headroom over the greedy-failure denominator "
            "are admissible."),
        "effect_size": summarise(gains),
        "effect_size_bootstrap": cluster_bootstrap(gains),
        "top1_disagreements": summarise(disagreements),
        "constrained_performance_recovered": {
            "definition": ("planning gain inside the mask as a fraction of the "
                           "terminal potency the mask cost under greedy parity"),
            "summary": summarise(recoveries),
            "sources_with_a_positive_mask_cost": len(recoveries),
            "note": "undefined where the mask cost nothing; those sources are omitted",
        },
        "binary_headroom": {
            "denominator_pathwise_greedy_failures": greedy_failures,
            "verified_rescued": rescued,
            "verdict": ("CEILING_NOT_A_NULL" if greedy_failures == 0
                        else "headroom measurable"),
            "note": ("headroom 0 over a denominator of 0 is a ceiling: a real "
                     "planning advantage had nowhere to show"),
        },
    }

    # ---- per-arm description ---------------------------------------------
    per_arm = {}
    arm_names = sorted({n for s in shards for n in s["arms"]})
    for name in arm_names:
        bodies = [s["arms"][name] for s in shards if name in s["arms"]]
        per_arm[name] = {
            "sources": len(bodies),
            "U_P": summarise([b["U_P"] for b in bodies]),
            "potency_gain_over_source": summarise([b["potency_gain"] for b in bodies]),
            "completed": sum(1 for b in bodies if b["completed"]),
            "endpoint_in_C": sum(1 for b in bodies if b["endpoint_in_C"]),
            "endpoint_terminal_infeasible": sum(
                1 for b in bodies if b.get("endpoint_terminal_infeasible")),
            "any_intermediate_violation_DEFINITIONAL_IF_PATHWISE": sum(
                1 for b in bodies if b["any_intermediate_violation"]),
            "marginal_kernel_calls": summarise(
                [float(b["marginal_kernel_calls"]) for b in bodies]),
        }

    # ---- terminal-failure attribution, per arm ----------------------------
    attribution: dict[str, dict[str, int]] = {}
    for name in arm_names:
        counts: dict[str, int] = {}
        for shard in shards:
            body = shard["arms"].get(name)
            if body is None:
                continue
            reason = body.get("terminal_failure_attribution")
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
        attribution[name] = counts

    return {"label": label, "sources": len(shards),
            "pathwise_estimands": estimands,
            "terminal_cost": cost,
            "terminal_cost_interpretation": {
                "no_expectation_that_pathwise_beats_endpoint_only": True,
                "reason": ("the mask can only shrink the reachable set, so "
                           "framing a potency win as the goal would be suspicious"),
                "declared_informative_in_advance": [
                    "little or no potency cost -> strongest pathwise result",
                    "moderate potency cost -> still meaningful; the constraint "
                    "genuinely restricts molecular evolution",
                    "large cost or frequent support collapse -> pathwise control "
                    "works formally but is practically too restrictive for this "
                    "corridor, which is a real finding",
                ],
                "none_of_these_is_a_failure": True,
            },
            "future_aware": planning,
            "terminal_failure_attribution": attribution,
            "per_arm": per_arm}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--panel", type=Path,
                        default=REPO / "diagnostics/pathwise_stage_b_panel.json")
    parser.add_argument("--out", type=Path,
                        default=REPO / "diagnostics/pathwise_stage_b_results.json")
    args = parser.parse_args()

    raw = [json.loads(p.read_text()) for p in sorted(args.shards.glob("*.json"))]
    if not raw:
        raise SystemExit(f"no shards under {args.shards}")
    void = [s for s in raw if s.get("status") == "INVALID_INSTRUMENT"]
    good = [s for s in raw if s.get("status") == "SMOKE_HELD_IN"]
    if not good:
        raise SystemExit(f"every shard is INVALID_INSTRUMENT ({len(void)})")
    leaks = {s["index"]: s["mask_leak"] for s in good if s.get("mask_leak")}

    tight = [s for s in good if s.get("support_tight")]
    not_tight = [s for s in good if not s.get("support_tight")]

    payload = {
        "schema": "compose.pathwise.stage_b_results",
        "status": "INVALID_INSTRUMENT" if leaks else "SMOKE_HELD_IN",
        "held_out_opened": False,
        "panel_sha256": (json.loads(args.panel.read_text())["panel_sha256"]
                         if args.panel.exists() else None),
        "shards": len(raw), "shards_void": len(void),
        "void_reasons": [s.get("error") for s in void] or None,
        "objective": "DRD2 potency alone (goal P)",
        "independent_unit": "source",
        "mask_integrity": {
            "verdict": "FAIL" if leaks else "PASS",
            "kind": "BUG_DETECTOR_NOT_A_FINDING",
            "leaks": leaks or None,
        },
        "support_tight": {
            "threshold": SUPPORT_TIGHT_THRESHOLD,
            "predeclared": True,
            "measured_on": "states visited by the DESCRIPTIVE unconstrained arm",
            "sources_tight": len(tight),
            "fraction_tight": round(len(tight) / len(good), 4) if good else None,
            "tight_indices": [s["index"] for s in tight],
            "median_retention_by_source": {
                str(s["index"]): s.get("median_retention") for s in good},
        },
        "mask_empty_by_source": {
            str(s["index"]): {
                "empty_states": s.get("mask_empty_states"),
                "fraction": s.get("mask_empty_fraction_this_source"),
            } for s in good},
        "primary_analysis_ITT": analyse(good, "intention-to-treat (all sources)"),
        "sensitivity_excluding_support_tight": (
            analyse(not_tight, "predeclared sensitivity: support-tight excluded")
            if not_tight and tight else
            {"label": "not applicable: no support-tight sources" if not tight
             else "not applicable: every source is support-tight",
             "sources": len(not_tight)}),
        "cost": {
            "kernel_calls": summarise([float(s["kernel_calls"]) for s in good]),
            "seconds": summarise([float(s["seconds"]) for s in good]),
        },
    }
    payload["results_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    itt = payload["primary_analysis_ITT"]
    print(f"sources {len(good)}  support-tight {len(tight)}")
    print("\nPRIMARY  hidden-path RATE  P(x_H in C AND exists t<H: x_t not in C)")
    for arm, body in itt["pathwise_estimands"].items():
        p = body["PRIMARY_hidden_path_rate"]
        print(f"  {arm:20s} {p['numerator']}/"
              f"{p['denominator_all_eligible_sources']} = {p['estimate']}")
    print("\nSECONDARY  hidden-path FRACTION among accepted endpoints")
    for arm, body in itt["pathwise_estimands"].items():
        s = body["SECONDARY_hidden_path_fraction"]
        print(f"  {arm:20s} {s['numerator']}/"
              f"{s['denominator_accepted_endpoints']} = {s['estimate']}"
              f"   (denominator shown; controller-dependent)")
    print("\nTERMINAL COST (sign is free)")
    for name, body in itt["terminal_cost"].items():
        print(f"  {name:26s} mean {body['summary'].get('mean')} "
              f"CI {body['bootstrap'].get('ci95_low')}..{body['bootstrap'].get('ci95_high')}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
