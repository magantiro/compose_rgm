"""Development readout for the same-prefix goal intervention.

Three questions, each reported SEPARATELY for the P-first and D-first histories
because the whole design is that information order changes the answer:

  Q1 INTERVENTION RESPONSIVENESS
     retarget vs continue_A. Does changing the objective at x_3 actually
     redirect the trajectory toward the new goal?

  Q2 VALUE OF HISTORY
     retarget-from-x_3 vs restart-from-x_0, same remaining budget. Was the
     realised history worth keeping?

  Q3 PRICE OF SURPRISE
     retarget vs clairvoyant, which knew B from step 0 and had the full
     horizon. How much of the no-surprise performance is recovered?

NO SIGN IS FORCED on Q2 or Q3 for the D-first history. The smoke suggested the
richer possibility -- that a prefix built for a different objective can still be
an asset -- and forcing a direction would hide it.

Primary metric is the paired CONTINUOUS final worst margin of P&D, with the
source as the independent unit. Binary conjunction success is secondary because
it may still ceiling or floor.

WHICH COMPARISONS ARE ADMISSIBLE. verified_retarget vs greedy_retarget is NOT
reported as a claim: policy improvement guarantees its direction, the error made
three times in the calibration. Q1-Q3 all differ in starting state, objective or
budget, so none has a sign fixed in advance.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

#: Six arms for the held-out confirmation. greedy_restart carries H2's
#: controller-matched sensitivity in the same run.
ARMS = ("continue_A", "greedy_retarget", "verified_retarget",
        "greedy_restart", "restart", "clairvoyant")
PAIRS_H2_SENSITIVITY = ("H2 sensitivity (greedy-matched)",
                        "greedy_retarget", "greedy_restart")
PAIRS = (("Q1 responsiveness", "greedy_retarget", "continue_A"),
         ("Q1 responsiveness (confounded, superseded)", "verified_retarget", "continue_A"),
         ("H2 sensitivity (greedy-matched)", "greedy_retarget", "greedy_restart"),
         ("Q2 value of history", "verified_retarget", "restart"),
         ("Q3 price of surprise", "verified_retarget", "clairvoyant"))


def paired(rows: list[dict], arm: str, base: str) -> dict:
    d = np.array([r["arms"][arm]["b_worst_margin"] - r["arms"][base]["b_worst_margin"]
                  for r in rows])
    n = len(d)
    boot = np.array([np.mean(np.random.default_rng(s).choice(d, n, replace=True))
                     for s in range(2000)])
    wins = int((d > 1e-9).sum())
    losses = int((d < -1e-9).sum())
    out = {"arm": arm, "base": base, "n": n,
           "mean_difference": float(d.mean()),
           "median_difference": float(np.median(d)),
           "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "arm_better": wins, "arm_worse": losses, "tied": n - wins - losses}
    if wins + losses:
        out["sign_test_two_sided_p"] = float(
            stats.binomtest(wins, wins + losses, 0.5).pvalue)
    return out


def report(rows: list[dict], label: str) -> dict:
    n = len(rows)
    print(f"\n{'=' * 66}\n{label}  (n={n} sources)\n{'=' * 66}")
    switch = np.array([r["b_worst_at_switch"] for r in rows])
    print(f"  worst P&D margin at the switch state: median {np.median(switch):+.4f}")

    print(f"\n  {'arm':>18} {'worst margin':>14} {'B success':>11} "
          f"{'P':>4} {'D':>4} {'overrides':>10}")
    summary = {}
    for arm in ARMS:
        w = np.array([r["arms"][arm]["b_worst_margin"] for r in rows])
        b = sum(r["arms"][arm]["b_success"] for r in rows)
        p = sum(r["arms"][arm]["p_success"] for r in rows)
        dd = sum(r["arms"][arm]["d_success"] for r in rows)
        ov = [r["arms"][arm]["overrides"] for r in rows
              if r["arms"][arm]["overrides"] is not None]
        summary[arm] = {"median_worst": float(np.median(w)),
                        "mean_worst": float(w.mean()),
                        "b_success": b, "p_success": p, "d_success": dd,
                        "mean_overrides": float(np.mean(ov)) if ov else None}
        print(f"  {arm:>18} {np.median(w):>14.4f} {b:>7}/{n} {p:>4} {dd:>4} "
              + (f"{np.mean(ov):>10.2f}" if ov else f"{'-':>10}"))

    comparisons = {}
    for name, arm, base in PAIRS:
        c = paired(rows, arm, base)
        comparisons[name] = c
        direction = ("higher" if c["mean_difference"] > 0 else "lower")
        print(f"\n  {name}: {arm} vs {base}")
        print(f"    paired difference in worst margin {c['mean_difference']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  ({direction})")
        print(f"    better on {c['arm_better']}/{c['n']}, worse on {c['arm_worse']}, "
              f"tied {c['tied']}"
              + (f", sign test p={c['sign_test_two_sided_p']:.4f}"
                 if "sign_test_two_sided_p" in c else ""))
    return {"n": n, "arms": summary, "comparisons": comparisons,
            "median_worst_at_switch": float(np.median(switch))}


def prefix_survival(rows: list[dict]) -> dict:
    """How much of the realised prefix survives the switch?

    The prefix is x_0..x_3. For each arm that continues from x_3, ask how much
    of the trajectory it lands on still resembles the switch state -- measured
    as whether the arm ever returns to a state it already passed through
    (an undo) and how far it travels from x_3.
    """
    out = {}
    for arm in ("continue_A", "greedy_retarget", "verified_retarget"):
        undos = 0
        for r in rows:
            path = r["arms"][arm]["trajectory"]
            undos += int(len(set(path)) < len(path))
        out[arm] = {"sources_with_a_revisit": undos, "n": len(rows)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--prefixes", type=Path,
                        default=Path("diagnostics/retarget_prefixes_committed.json"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    # Exclude *.partial.json -- per-arm checkpoints written mid-task. They
    # carry an incomplete `arms` map and would silently under-report every
    # comparison if globbed in alongside finished shards.
    rows = [json.loads(Path(f).read_text())
            for f in glob.glob(str(args.shards / "*.json"))
            if not f.endswith(".partial.json")]

    # COMPLETENESS GUARD. A source-history pair counts only when EVERY arm is
    # present. Excluding *.partial.json is not sufficient: a finished shard from
    # an earlier five-arm run, or any shard missing an arm, would otherwise
    # enter the aggregate and silently shift a paired contrast that assumes the
    # same sources on both sides.
    incomplete = [(r["history"], r["index"], sorted(set(ARMS) - set(r["arms"])))
                  for r in rows if not set(ARMS).issubset(r["arms"])]
    if incomplete:
        for history, index, missing in incomplete[:8]:
            print(f"  INCOMPLETE {history}/{index:03d} missing {missing}")
        raise SystemExit(f"{len(incomplete)} source-history pairs are missing arms. "
                         "A partially completed pair must never enter an aggregate.")
    if not rows:
        raise SystemExit(f"no shards under {args.shards}")
    prefixes = json.loads(args.prefixes.read_text())
    print(f"{len(rows)} branch points from committed prefixes "
          f"{prefixes['prefixes_sha256'][:16]}")

    by = {h: sorted((r for r in rows if r["history"] == h),
                    key=lambda r: r["index"]) for h in ("P", "D")}
    result = {}
    for history, label in (("P", "P-FIRST  (potency, then P&D)"),
                           ("D", "D-FIRST  (developability, then P&D)")):
        if by[history]:
            result[history] = report(by[history], label)
            result[history]["prefix_survival"] = prefix_survival(by[history])

    print(f"\n{'=' * 66}\nHISTORY CONTRAST -- the point of the design\n{'=' * 66}")
    if by["P"] and by["D"]:
        for name, arm, base in PAIRS:
            p = result["P"]["comparisons"][name]["mean_difference"]
            d = result["D"]["comparisons"][name]["mean_difference"]
            print(f"  {name:>22}   P-first {p:+.4f}   D-first {d:+.4f}")
        print("\n  Read this as: does the SAME intervention behave differently")
        print("  depending on what the trajectory was pursuing beforehand?")

    args.out.write_text(json.dumps({
        "schema": "compose.retarget.intervention_development",
        "status": "DEVELOPMENT_PANEL_NOT_CONFIRMATION",
        "prefixes_sha256": prefixes["prefixes_sha256"],
        "cohort_sha256": "e402e318fd9540fb",
        "horizon": 6, "switch_at": 3,
        "goal_language": ("unbounded signed IQR-normalised margins; conjunction "
                          "ranked lexicographically by (worst, mean); success = "
                          "worst >= 0"),
        "primary_metric": "paired continuous final P&D worst margin, source-paired",
        "not_a_claim": ("verified_retarget vs greedy_retarget -- policy "
                        "improvement guarantees its direction"),
        "by_history": result,
        "per_branch_point": rows,
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
