"""Controller-parity audit for the same-prefix retargeting contrasts.

A paired difference only isolates the mechanism it is named after if the two
arms differ in EXACTLY that mechanism. This script states each contrast's
controller, start state, budget and objective side by side, so a contrast that
silently changes two things at once is visible rather than inferred.

ARMS AS EXECUTED
----------------
    arm                controller   start   budget   objective
    continue_A         greedy       x_3     H-tau    A
    greedy_retarget    greedy       x_3     H-tau    B
    verified_retarget  verified     x_3     H-tau    B
    restart            verified     x_0     H-tau    B
    clairvoyant        verified     x_0     H        B

WHAT THE AUDIT FOUND
--------------------
Q2 (value of history) and Q3 (price of surprise) already hold parity: all three
arms involved run the same verified controller, so those contrasts vary only
the starting state, and for Q3 the budget -- which is the definition of the
comparison.

Q1 (intervention responsiveness) did NOT hold parity as first reported. It
compared `verified_retarget` against `continue_A`, which is verified against
greedy, so it confounded "the objective changed" with "a better controller ran".
The matched contrast is `greedy_retarget` vs `continue_A`: same controller, same
start, same budget, differing only in which objective is pursued. Both are
reported here; the matched one is primary.

The confound inflated Q1 by roughly 11%. It did not manufacture the effect --
but the confounded version reported 28/0 and 30/0, while the matched P-first
contrast shows a genuine loss, which is what makes it a measurement rather than
a foregone conclusion.

STILL MISSING
-------------
The greedy-matched form of Q2 -- `greedy` from x_3 versus a `greedy` restart
from x_0 -- cannot be computed from these shards because no greedy-restart arm
was run. Adding it is an instrument-completion run on the same 30 sources, not
a redesign. Until then Q2 rests on the verified-matched pair alone, which is
sound but single-controller.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

#: (arm, controller, start, budget, objective) exactly as executed.
ARM_SEMANTICS = {
    "continue_A":        ("greedy",   "x_3", "H-tau", "A"),
    "greedy_retarget":   ("greedy",   "x_3", "H-tau", "B"),
    "verified_retarget": ("verified", "x_3", "H-tau", "B"),
    "restart":           ("verified", "x_0", "H-tau", "B"),
    "clairvoyant":       ("verified", "x_0", "H",     "B"),
}

CONTRASTS = (
    ("Q1 responsiveness", "greedy_retarget", "continue_A", "PRIMARY"),
    ("Q1 responsiveness (confounded)", "verified_retarget", "continue_A", "SUPERSEDED"),
    ("Q2 value of history", "verified_retarget", "restart", "PRIMARY"),
    ("Q3 price of surprise", "verified_retarget", "clairvoyant", "PRIMARY"),
)


def parity(arm: str, base: str) -> dict:
    """Which of the four dimensions differ between the two arms?"""
    a, b = ARM_SEMANTICS[arm], ARM_SEMANTICS[base]
    fields = ("controller", "start", "budget", "objective")
    differs = [f for f, x, y in zip(fields, a, b) if x != y]
    return {"differs_in": differs, "arm": dict(zip(fields, a)),
            "base": dict(zip(fields, b)),
            "isolates_one_mechanism": len(differs) == 1}


def paired(rows: list[dict], arm: str, base: str, seed: int = 0) -> dict:
    d = np.array([r["arms"][arm]["b_worst_margin"] - r["arms"][base]["b_worst_margin"]
                  for r in rows])
    n = len(d)
    boot = np.array([np.mean(np.random.default_rng(s).choice(d, n, replace=True))
                     for s in range(2000)])
    wins, losses = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    return {"mean": float(d.mean()), "median": float(np.median(d)),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wins": wins, "losses": losses, "tied": n - wins - losses, "n": n}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = [json.loads(Path(f).read_text())
            for f in glob.glob(str(args.shards / "*.json"))
            if not f.endswith(".partial.json")]
    by = {h: [r for r in rows if r["history"] == h] for h in ("P", "D")}
    print(f"{len(rows)} branch points\n")

    print("PARITY CHECK -- how many dimensions does each contrast vary?")
    checks = {}
    for name, arm, base, status in CONTRASTS:
        p = parity(arm, base)
        checks[name] = p
        flag = "OK" if p["isolates_one_mechanism"] else "CONFOUNDED"
        print(f"  {name:<32} {arm} vs {base}")
        print(f"    varies: {', '.join(p['differs_in']):<28} {flag}  [{status}]")

    print("\nPAIRED DIFFERENCES in final worst P&D margin")
    results = {}
    for history in ("P", "D"):
        results[history] = {}
        print(f"\n  {history}-first (n={len(by[history])})")
        for name, arm, base, status in CONTRASTS:
            r = paired(by[history], arm, base)
            r["status"] = status
            r["parity"] = checks[name]["isolates_one_mechanism"]
            results[history][name] = r
            print(f"    {name:<32} {r['mean']:>+7.3f} "
                  f"[{r['ci95'][0]:>+6.3f}, {r['ci95'][1]:>+6.3f}]  "
                  f"{r['wins']}W/{r['losses']}L  [{status}]")

    # Price of surprise, expressed as recovery rather than as a raw deficit.
    print("\nQ3 RECOVERY -- fraction of the clairvoyant advantage recovered")
    recovery = {}
    for history in ("P", "D"):
        rs = by[history]
        switch = np.array([r["b_worst_at_switch"] for r in rs])
        retarget = np.array([r["arms"]["verified_retarget"]["b_worst_margin"] for r in rs])
        clair = np.array([r["arms"]["clairvoyant"]["b_worst_margin"] for r in rs])
        usable = np.abs(clair - switch) > 1e-9
        frac = (retarget - switch)[usable] / (clair - switch)[usable]
        recovery[history] = {"median": float(np.median(frac)),
                             "mean": float(frac.mean()), "n": int(usable.sum())}
        print(f"  {history}-first: median {np.median(frac):.1%} recovered "
              f"(n={int(usable.sum())})")

    # FULL SOURCE-LEVEL DISTRIBUTIONS, not just summaries. Two reasons.
    # A mean plus an interval cannot be re-analysed later -- if we want a
    # different estimator, a different subgroup, or a robustness check, the
    # per-source values must exist. And figure examples must be selectable by
    # a preregistered rule rather than by eye, which requires knowing where
    # each source sits in the effect distribution.
    per_source = {}
    for history in ("P", "D"):
        rsx = by[history]
        switch = np.array([r["b_worst_at_switch"] for r in rsx])
        clair = np.array([r["arms"]["clairvoyant"]["b_worst_margin"] for r in rsx])
        ret = np.array([r["arms"]["verified_retarget"]["b_worst_margin"] for r in rsx])
        rho = (ret - switch) / (clair - switch)
        entries = []
        for k, r in enumerate(rsx):
            row = {"index": r["index"], "source": r["source"],
                   "switch_state": r["switch_state"],
                   "b_worst_at_switch": r["b_worst_at_switch"],
                   "rho_recovery": float(rho[k])}
            for name, arm, base, _status in CONTRASTS:
                row[name] = (r["arms"][arm]["b_worst_margin"]
                             - r["arms"][base]["b_worst_margin"])
            for arm in ARM_SEMANTICS:
                row[f"margin_{arm}"] = r["arms"][arm]["b_worst_margin"]
                row[f"success_{arm}"] = r["arms"][arm]["b_success"]
            entries.append(row)
        per_source[history] = entries

    # PREREGISTERED FIGURE-EXAMPLE SELECTION. Fixed here, before any figure is
    # drawn, so trajectories are chosen by where they sit in the effect
    # distribution rather than by which molecules look best.
    examples = {}
    for history, entries in per_source.items():
        chosen = {}
        for name, _arm, _base, status in CONTRASTS:
            if status != "PRIMARY":
                continue
            vals = np.array([e[name] for e in entries])
            median = float(np.median(vals))
            typical = entries[int(np.argmin(np.abs(vals - median)))]
            # The informative counterexample: the source that most contradicts
            # the headline direction. If none contradicts it, say so rather
            # than substituting the weakest supporting case.
            against = vals < 0 if median > 0 else vals > 0
            counter = (entries[int(np.argmax(np.abs(vals * against)))]
                       if against.any() else None)
            chosen[name] = {
                "rule": ("typical = source nearest the median effect; "
                         "counterexample = source most strongly opposing the "
                         "headline direction, or null if none opposes it"),
                "median_effect": median,
                "typical_source_index": typical["index"],
                "typical_effect": typical[name],
                "counterexample_source_index": counter["index"] if counter else None,
                "counterexample_effect": counter[name] if counter else None,
            }
        examples[history] = chosen

    print("\nPREREGISTERED FIGURE EXAMPLES (selected by rule, not by eye)")
    for history, chosen in examples.items():
        for name, c in chosen.items():
            ce = ("none opposes" if c["counterexample_source_index"] is None
                  else f"src {c['counterexample_source_index']} ({c['counterexample_effect']:+.3f})")
            print(f"  {history}-first {name:<24} typical src "
                  f"{c['typical_source_index']} ({c['typical_effect']:+.3f})   "
                  f"counterexample {ce}")

    args.out.write_text(json.dumps({
        "schema": "compose.retarget.controller_parity_audit",
        "per_source_distributions": per_source,
        "preregistered_figure_examples": examples,
        "status": "AUDIT_OF_A_DEVELOPMENT_PANEL",
        "arm_semantics": {k: dict(zip(("controller", "start", "budget", "objective"), v))
                          for k, v in ARM_SEMANTICS.items()},
        "parity_checks": checks,
        "paired_differences": results,
        "q3_recovery_fraction": recovery,
        "finding": ("Q2 and Q3 held controller parity as executed. Q1 did not: "
                    "verified_retarget vs continue_A confounds the objective "
                    "change with controller sophistication. The matched Q1 is "
                    "greedy_retarget vs continue_A, and is now primary."),
        "still_missing": ("greedy-matched Q2 requires a greedy restart from x_0, "
                          "which was not run. Q2 currently rests on the "
                          "verified-matched pair only."),
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
