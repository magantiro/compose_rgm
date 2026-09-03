#!/usr/bin/env python3
"""Does Q(M|x,z)'s feasibility term beat uniform region choice?

5-fold cross-validation on the gate's 239 labelled regions. The contingency
table is REFIT on each training split, so no fold is scored by a table that saw
it -- otherwise the selector would be graded on rows it memorised.

Two things are measured, and the second is the one that matters:

    AUC                    does it rank feasible regions above infeasible ones
    successes per hour     what a budgeted search actually gets, since the
                           selector spends real controller time either way
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def band_of(r):
    return min(int(r["r_release"] * 5), 4)


def fit(rows):
    """Contingency table from a training split only."""
    from collections import defaultdict
    g_h = sum(1 for r in rows if r.get("any_ok"))
    g = g_h / max(1, len(rows))
    iface = defaultdict(lambda: [0, 0])
    size = defaultdict(lambda: [0, 0])

    def sb(n):
        for i, (lo, hi) in enumerate(((1, 3), (4, 6), (7, 10), (11, 20), (21, 10**6))):
            if lo <= n <= hi:
                return i
        return 4
    for r in rows:
        iface[r["interface"]][1] += 1
        size[sb(r["region_size"])][1] += 1
        if r.get("any_ok"):
            iface[r["interface"]][0] += 1
            size[sb(r["region_size"])][0] += 1
    S = 8.0

    def sm(hits, n):
        return (hits + S * g) / (n + S)

    def f(r):
        pi = sm(*iface[r["interface"]]) if r["interface"] in iface else g
        ps = sm(*size[sb(r["region_size"])]) if sb(r["region_size"]) in size else g
        return (pi * ps) ** 0.5
    return f


def auc(pairs):
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def main():
    d = json.loads((ROOT / "diagnostics/region_scale_gate.json").read_text())
    rows = [r for c in d["results"] for r in c["rows"]]
    rows = [r for r in rows if "interface" in r]
    folds = 5
    scored = []
    for k in range(folds):
        test = [r for i, r in enumerate(rows) if i % folds == k]
        train = [r for i, r in enumerate(rows) if i % folds != k]
        f = fit(train)
        for r in test:
            scored.append((f(r), bool(r.get("any_ok")), r))
    print(f"regions={len(rows)}  successes={sum(1 for r in rows if r.get('any_ok'))}")
    print(f"cross-validated AUC = {auc([(s, y) for s, y, _ in scored]):.3f}")

    # budgeted comparison: spend the same wall-clock either way
    import statistics as st
    budget = 3600.0
    ranked = sorted(scored, key=lambda t: -t[0])
    def spend(order):
        t, ok = 0.0, 0
        for s, y, r in order:
            c = r.get("sec_region", 0) or 1.0
            if t + c > budget:
                continue
            t += c
            ok += int(y)
        return ok, t
    sel_ok, sel_t = spend(ranked)
    import random
    rng = random.Random(0)
    trials = []
    for _ in range(200):
        o = scored[:]
        rng.shuffle(o)
        trials.append(spend(o)[0])
    print(f"\nwithin a {budget/3600:.0f}-hour controller budget:")
    print(f"  selector-ranked : {sel_ok} successful rewrites ({sel_t:.0f}s spent)")
    print(f"  uniform random  : {st.mean(trials):.1f} +/- {st.pstdev(trials):.1f}")
    if st.mean(trials) > 0:
        print(f"  ratio           : {sel_ok/st.mean(trials):.2f}x")


if __name__ == "__main__":
    main()
