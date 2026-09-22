"""Reduce the matched macro-option arms against the PREDECLARED falsifier.

The thresholds, the statistic, the sample size and the VOID conditions were sealed in
diagnostics/pmo_macro_option_v1/falsifier_predeclaration_v1.json before any mechanism or
measurement code existed. This script reads them from that file rather than restating
them, so a number cannot be compared against a threshold edited to suit it.

Usage:
  PYTHONPATH=src:scripts python scripts/pmo_macro_option_verdict.py \
      --arms diagnostics/pmo_macro_option_v1/arms \
      --out diagnostics/pmo_macro_option_v1/verdict_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

PREDECLARATION = Path("diagnostics/pmo_macro_option_v1/falsifier_predeclaration_v1.json")


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(Binomial(b + c, 0.5) >= b). Exact, so a small discordant count is honest."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(b, n + 1)) / (2.0**n)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    sealed = json.loads(PREDECLARATION.read_text())
    sealed_sha = hashlib.sha256(PREDECLARATION.read_bytes()).hexdigest()
    alpha = float(sealed["alpha"])
    minimum_effect = float(sealed["minimum_effect_of_interest"])
    # The one threshold the seal states in prose rather than as a field. It is NOT added
    # to the sealed file -- editing a predeclaration after sealing is the thing sealing
    # exists to prevent -- so the transcription is checked against the prose instead, and
    # a drift in either direction aborts before any verdict is computed.
    reject_below = 0.20
    underpowered = sealed["sample_size"]["underpowered_region_declared_in_advance"]
    if f"below {reject_below:.2f}" not in underpowered:
        raise ValueError(
            "the REJECTED threshold transcribed here is not the one the sealed "
            f"predeclaration states: {underpowered!r}"
        )
    required_n = int(sealed["sample_size"]["n_declared_options_minimum"])
    required_seeds = int(sealed["sample_size"]["independent_seeds_minimum"])

    seeds, failures = [], []
    for path in sorted(arguments.arms.glob("seed_*.json")):
        row = json.loads(path.read_text())
        (seeds if row.get("status") == "complete" else failures).append(row)

    pairs, void_reasons = [], []
    bridges_in_valley = {"protected": 0, "declared_unprotected": 0}
    bridges_total = {"protected": 0, "declared_unprotected": 0}
    charged = {"protected": {"bridge": 0, "destination": 0, "total": 0},
               "declared_unprotected": {"bridge": 0, "destination": 0, "total": 0}}
    finals = {"protected": [], "declared_unprotected": []}
    for row in seeds:
        arms = row["arms"]
        by_arm = {
            name: {option["option_id"]: option for option in arms[name]["options"]}
            for name in ("protected", "declared_unprotected")
        }
        if set(by_arm["protected"]) != set(by_arm["declared_unprotected"]):
            void_reasons.append(
                f"seed {row['seed']}: declared option sets differ across arms"
            )
        for name in ("protected", "declared_unprotected"):
            charged[name]["total"] += arms[name]["charged_calls"]
            finals[name].append(
                {"best": arms[name]["best"], "top10": arms[name]["top10"]}
            )
            for option in arms[name]["options"]:
                bridges_in_valley[name] += option["bridges_in_valley"]
                bridges_total[name] += len(option["bridge_endpoints"])
                charged[name]["bridge"] += option["bridges_charged"]
                charged[name]["destination"] += int(
                    option["destination_score"] is not None
                )
        for option_id in sorted(set(by_arm["protected"]) & set(by_arm["declared_unprotected"])):
            pairs.append(
                {
                    "seed": row["seed"],
                    "option_id": option_id,
                    "protected": bool(by_arm["protected"][option_id]["destination_reached"]),
                    "unprotected": bool(
                        by_arm["declared_unprotected"][option_id]["destination_reached"]
                    ),
                    "stages": by_arm["protected"][option_id]["stages"],
                    "total_primitives": by_arm["protected"][option_id]["total_primitives"],
                }
            )

    n = len(pairs)
    both = sum(p["protected"] and p["unprotected"] for p in pairs)
    b = sum(p["protected"] and not p["unprotected"] for p in pairs)
    c = sum(p["unprotected"] and not p["protected"] for p in pairs)
    reach_protected = (both + b) / n if n else 0.0
    reach_unprotected = (both + c) / n if n else 0.0
    effect = reach_protected - reach_unprotected
    p_value = mcnemar_one_sided(b, c)

    # ---- The predeclared instrument checks, which can VOID the run ----
    for name in ("protected", "declared_unprotected"):
        if bridges_total[name] and not bridges_in_valley[name]:
            void_reasons.append(
                f"{name}: zero declared bridges fell in the valley -- the landscape "
                "does not exercise the mechanism"
            )
    if n and reach_unprotected in (0.0, 1.0):
        void_reasons.append(
            f"unprotected reach rate is {reach_unprotected}, not strictly inside (0, 1): "
            "the comparison is tautological at this landscape calibration"
        )

    if void_reasons:
        verdict = "VOID"
    elif n < required_n or len({p['seed'] for p in pairs}) < required_seeds:
        verdict = "INSUFFICIENT_SAMPLE"
    elif effect >= minimum_effect and p_value < alpha:
        verdict = "MECHANISM_NOT_INERT"
    elif effect < reject_below:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE_UNDERPOWERED"

    mean = lambda rows, key: (
        sum(r[key] for r in rows if r[key] is not None) / max(1, len([r for r in rows if r[key] is not None]))
    )
    cost_regression = mean(finals["protected"], "top10") < mean(
        finals["declared_unprotected"], "top10"
    )
    if verdict == "MECHANISM_NOT_INERT" and cost_regression:
        verdict = "MECHANISM_WORKS_COST_UNACCEPTABLE"

    payload = {
        "schema_version": "pmo_macro_option_verdict_v1",
        "predeclaration": str(PREDECLARATION),
        "predeclaration_sha256": sealed_sha,
        "information_regime": sealed["information_regime"],
        "benchmark_oracle_calls": 0,
        "seeds_complete": len(seeds),
        "seeds_failed": len(failures),
        "declared_option_pairs": n,
        "required_pairs": required_n,
        "required_seeds": required_seeds,
        "reach_protected": round(reach_protected, 4),
        "reach_unprotected": round(reach_unprotected, 4),
        "effect": round(effect, 4),
        "minimum_effect_of_interest": minimum_effect,
        "discordant_protected_only": b,
        "discordant_unprotected_only": c,
        "both_reached": both,
        "neither_reached": n - both - b - c,
        "p_value_one_sided_exact_mcnemar": round(p_value, 6),
        "alpha": alpha,
        "instrument_checks": {
            "bridges_in_valley": bridges_in_valley,
            "bridges_total": bridges_total,
            "unprotected_reach_strictly_inside_unit_interval": bool(
                n and 0.0 < reach_unprotected < 1.0
            ),
        },
        "void_reasons": void_reasons,
        "cost": {
            "charged_calls": charged,
            "mean_best": {k: round(mean(v, "best"), 4) for k, v in finals.items()},
            "mean_top10": {k: round(mean(v, "top10"), 4) for k, v in finals.items()},
            "protected_arm_top10_worse": bool(cost_regression),
        },
        "verdict": verdict,
        "scope": sealed["what_a_pass_does_and_does_not_license"],
        "pairs": pairs,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in payload.items() if k != "pairs"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
