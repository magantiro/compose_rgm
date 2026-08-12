"""Read the held-in calibration and apply the four decisions. Runs ONCE.

Emits a verdict for each, and the verdicts are allowed to be negative:

  1. potency threshold  -- floor if nothing reaches it, ceiling if everything does
  2. developability region base rate, greedy AND verified
  3. horizon -- where in the remaining budget do contrastive decisions live
  4. THE GATE on subclaim B

The gate is read exactly as C0's was, against the same null. C0 reported
sacrificial actions winning 8 of 14 (57%) against a null of 0.5 and called it
STOP / INCONCLUSIVE. The bar here is the same bar.

Two failure modes are checked separately because they have opposite fixes and
are easy to confuse:

  FLOOR    no arm reaches the goal -- nothing to measure
  CEILING  greedy already reaches it -- nothing for lookahead to ADD, even if
           its decisions genuinely differ. A high disagreement rate with
           identical binary outcomes is a ceiling, not a win.

Per the anti-tuning rule, a negative gate is recorded and subclaim B leaves the
paper. No parameter is re-selected here.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

POTENCY_GRID = ("0.3", "0.5", "0.7")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--cohort", type=Path,
                        default=Path("diagnostics/retarget_calibration_cohort.json"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    files = [f for f in glob.glob(str(args.shards / "**" / "*.json"), recursive=True)
             if "aggregate" not in f]
    rows = sorted((json.loads(Path(f).read_text()) for f in files),
                  key=lambda r: r["index"])
    if not rows:
        raise SystemExit(f"no shards under {args.shards}")
    cohort = json.loads(args.cohort.read_text())
    n = len(rows)
    print(f"{n} sources from cohort {cohort['cohort_sha256'][:16]}\n")

    # --- 1. potency threshold -------------------------------------------
    print("1. POTENCY THRESHOLD -- reached after the 4-edit prefix")
    climbs = np.array([r["potency_reach"][POTENCY_GRID[0]]["climb_achieved"]
                       for r in rows])
    potency = {}
    print(f"{'threshold':>10} {'start':>7} {'end':>7} {'verdict':>12}")
    for t in POTENCY_GRID:
        start = sum(r["potency_reach"][t]["start_satisfies"] for r in rows)
        end = sum(r["potency_reach"][t]["end_satisfies"] for r in rows)
        verdict = ("FLOOR" if end == 0 else
                   "CEILING" if end == n else "USABLE")
        potency[t] = {"start": start, "end": end, "rate": end / n,
                      "verdict": verdict}
        print(f"{t:>10} {start:>7} {end:>7} {verdict:>12}")
    print(f"  climb achieved in 4 greedy edits: median {np.median(climbs):+.3f}, "
          f"p90 {np.percentile(climbs, 90):+.3f}, max {climbs.max():+.3f} log-odds")

    # --- 2. developability base rate ------------------------------------
    print("\n2. DEVELOPABILITY REGION -- reached in 4 post-switch edits")
    at_switch = sum(r["develop_switch_success"] for r in rows)
    greedy = sum(r["develop_greedy_success"] for r in rows)
    verified = sum(r["develop_verified_success"] for r in rows)
    print(f"  satisfied at the switch state : {at_switch:>3}/{n}")
    print(f"  greedy retarget reaches it    : {greedy:>3}/{n}  ({greedy/n:.0%})")
    print(f"  verified retarget reaches it  : {verified:>3}/{n}  ({verified/n:.0%})")
    develop_verdict = ("FLOOR -- neither arm reaches the region" if greedy == 0 and verified == 0
                       else "CEILING -- greedy already reaches it everywhere" if greedy == n
                       else "USABLE")
    print(f"  verdict: {develop_verdict}")
    headroom = verified - greedy
    print(f"  headroom for lookahead on the BINARY endpoint: {headroom:+d}")

    # --- 3. horizon ------------------------------------------------------
    print("\n3. HORIZON -- where the contrastive decisions live")
    states = [s for r in rows for s in r["decision_states"]]
    print(f"{'remaining':>10} {'states':>7} {'disagree':>9} {'sacrificial':>12} "
          f"{'won':>6} {'mean regret':>12}")
    horizon = {}
    for remaining in sorted({s["remaining"] for s in states}, reverse=True):
        bucket = [s for s in states if s["remaining"] == remaining]
        dis = sum(s["top1_disagreement"] for s in bucket)
        sac = sum(s["sacrificial"] for s in bucket)
        won = sum(s["sacrifice_won"] for s in bucket)
        regret = float(np.mean([s["future_regret"] for s in bucket]))
        horizon[str(remaining)] = {"states": len(bucket), "disagreement": dis,
                                   "sacrificial": sac, "sacrifice_won": won,
                                   "mean_regret": regret}
        print(f"{remaining:>10} {len(bucket):>7} {dis:>9} {sac:>12} {won:>6} "
              f"{regret:>+12.4f}")

    # --- 4. THE GATE -----------------------------------------------------
    print("\n4. THE GATE on subclaim B")
    total = len(states)
    disagree = sum(s["top1_disagreement"] for s in states)
    sacrificial = [s for s in states if s["sacrificial"]]
    won = sum(s["sacrifice_won"] for s in sacrificial)
    regrets = np.array([s["future_regret"] for s in states])
    print(f"  decision states            {total}")
    print(f"  top-1 disagreement         {disagree}/{total} ({disagree/total:.0%})"
          f"   <- non-circular: lookahead WOULD act differently")

    # ------------------------------------------------------------------
    # WHY THERE IS NO "sacrifice-to-win" STATISTIC HERE.
    #
    # The obvious one is circular and must not be reported. The lookahead
    # action is chosen as argmax(V_G), so V_G(chosen) >= V_G(greedy) holds BY
    # CONSTRUCTION and "the sacrifice won" is true whenever the two disagree
    # and do not tie. The data show it: 0 losses out of 55 disagreements.
    #
    # C0 avoided this by scoring on an INDEPENDENT evaluation sample. That
    # safeguard was dropped here on the reasoning that a deterministic V_G has
    # no winner's curse -- which is wrong. The bias is not sampling noise; it
    # is selecting and scoring with the same function, and determinism does not
    # touch it.
    #
    # The non-circular question is whether acting on the lookahead produces a
    # better OUTCOME, which is the endpoint comparison in section 2.
    # ------------------------------------------------------------------
    losses = sum(1 for s in states if s["top1_disagreement"] and s["future_regret"] < 0)
    ties = sum(1 for s in states if s["top1_disagreement"] and s["future_regret"] == 0)
    print(f"  of those: {disagree - ties - losses} higher V_G, {ties} tied, "
          f"{losses} lower")
    print("    NOT reported as evidence -- V_G(chosen) >= V_G(greedy) by")
    print("    construction, so this is definitional, not a measurement.")
    rate, pvalue = None, None

    print(f"  mean V_G gap               {regrets.mean():+.4f} "
          f"(p90 {np.percentile(regrets, 90):+.4f}, max {regrets.max():+.4f})")
    movement = np.array([r["develop_greedy_score"] - r["develop_switch_score"]
                         for r in rows])
    relative = regrets.mean() / abs(movement.mean()) if movement.mean() else float("nan")
    print(f"  typical post-switch movement {movement.mean():+.4f}; "
          f"the V_G gap is {relative:.1%} of it")

    # The only admissible evidence: did acting on the lookahead change the
    # OUTCOME, on sources where it would have acted differently at all?
    would_differ = [r for r in rows
                    if any(s["top1_disagreement"] for s in r["decision_states"])]
    changed = [r for r in would_differ
               if r["develop_greedy_success"] != r["develop_verified_success"]]
    print(f"\n  sources where lookahead would act differently : {len(would_differ)}/{n}")
    print(f"  ...where the OUTCOME actually differed         : {len(changed)}")

    gate = "OPEN" if headroom > 0 else "CLOSED"
    reasons = []
    if headroom <= 0:
        reasons.append(f"greedy already reaches the region {greedy}/{n} -- CEILING, "
                       f"so a real planning advantage would have nowhere to show")
    print(f"\n  GATE: {gate}" + (f" -- {'; '.join(reasons)}" if reasons else ""))
    if gate == "CLOSED":
        print("  Per the anti-tuning rule the region is NOT retuned. Subclaim B")
        print("  leaves the paper; subclaims A and C proceed.")

    args.out.write_text(json.dumps({
        "schema": "compose.retarget.calibration_result",
        "status": "HELD_IN_CALIBRATION_RUNS_ONCE",
        "cohort_sha256": cohort["cohort_sha256"],
        "sources": n,
        "potency": potency,
        "potency_climb_median": float(np.median(climbs)),
        "developability": {"at_switch": at_switch, "greedy": greedy,
                           "verified": verified, "verdict": develop_verdict,
                           "binary_headroom": int(headroom)},
        "horizon": horizon,
        "gate": {
            "decision_states": total, "top1_disagreement": disagree,
            "disagreements_higher_vg": disagree - ties - losses,
            "disagreements_tied": ties, "disagreements_lower_vg": losses,
            "sacrifice_to_win_NOT_REPORTED": (
                "Circular: the lookahead action is argmax(V_G), so "
                "V_G(chosen) >= V_G(greedy) by construction and 0 of 55 "
                "disagreements could have been a loss. C0 avoided this with an "
                "independent evaluation sample; that safeguard was dropped here "
                "on the incorrect reasoning that a deterministic V_G has no "
                "winner's curse. The bias is selecting and scoring with the "
                "same function, which determinism does not remove."),
            "sources_where_lookahead_would_differ": len(would_differ),
            "sources_where_outcome_differed": len(changed),
            "mean_future_regret": float(regrets.mean()),
            "regret_as_fraction_of_movement": float(relative),
            "verdict": gate, "reasons": reasons,
        },
        "anti_tuning": ("The region is set once and not retuned. A closed gate is "
                        "recorded as a result: subclaim B leaves the paper."),
        "per_source": rows,
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
