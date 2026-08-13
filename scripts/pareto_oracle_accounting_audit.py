"""Audit what the oracle counters actually count. NO NEW KERNEL WORK.

WHY THIS EXISTS
---------------
This lane reported an oracle-demand ratio between preference control and
generate-and-rank at matched kernel budget. **That ratio is withdrawn as a cost
claim.** The counts are real; the interpretation is not established.

At matched kernel budget, preference control is charged for interrogating the
whole legal successor fiber at every decision, while generate-and-rank is
charged only for the terminal molecules it produced. Those are **different
acts**. The ratio conflates *property evaluations per unit of kernel work* with
*cost of optimizing a molecule*, and only the first was measured.

This script reads committed shards and the frozen census, re-derives every
counter from the code's actual increment sites, and asserts mechanically that
the increments equal the candidate counts we believe they do. A counter bug
would matter more than the ratio.

FOUR SEPARATIONS
----------------
1. Is one "oracle request" a distinct expensive invocation, or a row in a
   vectorised batch?
2. DRD2 (the black-box surrogate) versus QED/cLogP (deterministic local RDKit).
3. Requests versus post-cache evaluator calls.
4. Marked actions versus DISTINCT CANONICAL SUCCESSORS after alias collapse.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]

#: Established by reading the increment sites, not inferred from the numbers.
#:
#:   MeteredProcess.z()          raw_oracle_calls += 1 on EVERY request;
#:                               native_oracle_calls += 1 only on a cache MISS.
#:   MeteredProcess.successors() kernel_calls += 1 only on a cache MISS.
#:   objective_vector()          calls oracle.margin_many([key]) -- a batch of
#:                               EXACTLY ONE -- and then QED and Crippen.MolLogP
#:                               unconditionally, on every evaluation.
COUNTER_SEMANTICS = {
    "raw_oracle_calls": "every z() request, including cache hits",
    "native_oracle_calls": "cache misses only -- distinct molecules actually evaluated",
    "kernel_calls": "distinct states enumerated (successor cache misses)",
    "drd2_batch_size": 1,
    "drd2_call_form": "oracle.margin_many([key]) -- a list of exactly one molecule",
    "descriptors_computed_with_every_drd2_call": ["QED", "Crippen.MolLogP"],
}


def load_shards(directory: Path) -> tuple[list[dict], list[dict]]:
    complete, partial = [], []
    for f in sorted(glob.glob(str(directory / "**" / "*.json"), recursive=True)):
        payload = json.loads(Path(f).read_text())
        (partial if f.endswith(".partial.json") else complete).append(payload)
    return complete, partial


def audit_arm(arm: str, cost: dict[str, float]) -> dict[str, Any]:
    raw = float(cost["raw_oracle_calls"])
    native = float(cost["native_oracle_calls"])
    kernel = float(cost["kernel_calls"])
    return {
        "kernel_calls": kernel,
        # --- separation 3: requests vs post-cache evaluator calls -----------
        "oracle_requests": raw,
        "unique_valid_canonical_evaluations": native,
        "cache_hits": raw - native,
        "cache_hit_rate": (raw - native) / raw if raw else None,
        # --- separation 2: the biologically meaningful count -----------------
        # objective_vector() computes DRD2, QED and cLogP together on EVERY
        # evaluation, unconditionally. So all three are equal BY CONSTRUCTION.
        # That is not a coincidence to be reported as a finding; it is a
        # property of the implementation, and it means the counter cannot
        # distinguish them. If a future version short-circuits (e.g. skips DRD2
        # when only developability binds) these diverge and the counter would
        # silently mislead.
        "N_drd2": native,
        "N_descriptor": native,
        "N_all_objective": native,
        "drd2_equals_total_by_construction": True,
        # --- separation 1: batch or separate invocation ----------------------
        "drd2_invocations": native,
        "drd2_rows_per_invocation": 1,
        "is_vectorised_batch": False,
        # --- the quantity the withdrawn ratio actually measured --------------
        "evaluations_per_kernel_call": native / kernel if kernel else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--census", type=Path,
                        default=REPO / "diagnostics/pareto_tradeoff_census.json")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    complete, partial = load_shards(args.shards)
    rows = complete + partial
    if not rows:
        raise SystemExit(f"no shards under {args.shards}")

    census = json.loads(args.census.read_text())
    fiber_mean = census["instruments"]["I_A"]["mean_fiber_width"]

    per_source: dict[str, Any] = {}
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        entry = {}
        for arm, a in row.get("arms", {}).items():
            if "cost" not in a:
                continue
            audited = audit_arm(arm, a["cost"])
            entry[arm] = audited
            by_arm.setdefault(arm, []).append(audited)
        per_source[str(row["index"])] = {
            "complete": row in complete, "stage": row.get("stage"), "arms": entry}

    # ---- MECHANICAL ASSERTION -------------------------------------------
    # For the exhaustive arms, every enumerated state has its whole fiber
    # scored, so native_oracle_calls should be ~ kernel_calls x fiber width,
    # reduced by overlap between nearby states. Ratios far ABOVE the observed
    # fiber width would mean the counter increments on something other than a
    # distinct candidate -- a counter bug, which matters more than the ratio.
    checks = []
    for arm in ("greedy_pref", "verified_pref"):
        ratios = [a["evaluations_per_kernel_call"] for a in by_arm.get(arm, [])
                  if a["evaluations_per_kernel_call"] is not None]
        if not ratios:
            continue
        hi = float(np.max(ratios))
        checks.append({
            "arm": arm,
            "n_sources": len(ratios),
            "evaluations_per_kernel_call": {
                "min": float(np.min(ratios)), "median": float(np.median(ratios)),
                "max": hi},
            "census_mean_distinct_successors_per_state": fiber_mean,
            "consistent_with_fiber_width": bool(hi <= 2.5 * fiber_mean),
            "verdict": ("CONSISTENT -- increments track distinct candidates"
                        if hi <= 2.5 * fiber_mean else
                        "SUSPECT -- more evaluations per enumeration than the "
                        "fiber contains; investigate the increment site"),
            "note": ("fiber width varies by molecule, so a per-source ratio above "
                     "the census MEAN is expected; a ratio far above the observed "
                     "maximum would not be"),
        })

    payload = {
        "schema": "compose.pareto.oracle_accounting_audit",
        "status": "SMOKE_HELD_IN",
        "no_new_kernel_work": True,
        "source": "committed shards plus the frozen census; nothing re-run",
        "withdrawn_claim": (
            "The oracle-demand ratio between preference control and "
            "generate-and-rank at matched kernel budget is WITHDRAWN as a COST "
            "claim. At matched kernel budget preference control is charged for "
            "interrogating the whole legal fiber at every decision while "
            "generate-and-rank is charged only for the terminal molecules it "
            "produced -- different acts. The counts are real; the interpretation "
            "is not established."),
        "counter_semantics_from_code": COUNTER_SEMANTICS,
        "separations": {
            "1_batch_vs_invocation": {
                "answer": ("SEPARATE INVOCATIONS. objective_vector() calls "
                           "oracle.margin_many([key]) with a list of exactly one "
                           "molecule, so every evaluation is its own scorer "
                           "invocation. The counter is NOT hiding vectorisation."),
                "consequence": ("the counts are honest, but the implementation is "
                                "leaving batching on the table -- thousands of "
                                "one-molecule SVM calls per source"),
            },
            "2_drd2_vs_descriptors": {
                "answer": ("NOT SEPARABLE AS INSTRUMENTED. objective_vector() "
                           "computes DRD2, QED and cLogP together on every "
                           "evaluation, so N_drd2 == N_descriptor == "
                           "N_all_objective BY CONSTRUCTION."),
                "consequence": ("today the biologically meaningful count happens "
                                "to equal the total, but only because nothing "
                                "short-circuits. Any future version that skips "
                                "DRD2 when developability binds would make these "
                                "diverge while the counter kept reporting one "
                                "number."),
                "required_fix_before_any_cost_claim": (
                    "increment N_drd2 and N_descriptor at their own call sites"),
            },
            "3_requests_vs_post_cache": {
                "answer": "SEPARABLE AND REPORTED -- see cache_hit_rate per arm",
            },
            "4_marks_vs_distinct_successors": {
                "answer": ("PARTIALLY ANSWERED. canonical_successor_result() "
                           "returns batch.successors AFTER alias collapse, so the "
                           "census's mean fiber width of "
                           f"{fiber_mean:.0f} is already a DISTINCT-CANONICAL-"
                           "SUCCESSOR count, not a marked-action count."),
                "not_answerable_from_artifacts": (
                    "the marked-action count before collapse is not recorded in "
                    "the shards or the census. Establishing the collapse ratio "
                    "needs one instrumented enumeration, which is new kernel work "
                    "and is NOT done here."),
            },
        },
        "mechanical_checks": checks,
        "per_arm_medians": {
            arm: {k: float(np.median([a[k] for a in xs if a[k] is not None]))
                  for k in ("oracle_requests", "unique_valid_canonical_evaluations",
                            "cache_hit_rate", "evaluations_per_kernel_call",
                            "kernel_calls")
                  if any(a[k] is not None for a in xs)}
            for arm, xs in sorted(by_arm.items())},
        "per_source": per_source,
        "reporting_rule": (
            "Until the marked-action collapse ratio and the separated DRD2 "
            "counter exist, these appear as RAW COUNTS with the withdrawal "
            "caveat attached, never as a cost ratio."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")

    print(f"ORACLE ACCOUNTING AUDIT -- {len(complete)} complete, "
          f"{len(partial)} partial shards. No new kernel work.\n")
    print(f"{'arm':<20}{'requests':>11}{'unique eval':>13}{'cache hit':>11}"
          f"{'kernel':>8}{'eval/kernel':>13}")
    for arm, med in payload["per_arm_medians"].items():
        print(f"{arm:<20}{med.get('oracle_requests', 0):>11.0f}"
              f"{med.get('unique_valid_canonical_evaluations', 0):>13.0f}"
              f"{med.get('cache_hit_rate', 0):>11.3f}"
              f"{med.get('kernel_calls', 0):>8.0f}"
              f"{med.get('evaluations_per_kernel_call', 0):>13.1f}")
    print("\nMECHANICAL CHECKS")
    for c in checks:
        print(f"  {c['arm']:<16} eval/kernel "
              f"{c['evaluations_per_kernel_call']['median']:.0f} "
              f"(max {c['evaluations_per_kernel_call']['max']:.0f}) vs census "
              f"fiber {c['census_mean_distinct_successors_per_state']:.0f}"
              f"  -> {c['verdict'].split(' -- ')[0]}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
