"""B vs C on 5ht1b_s7_d0.6. READ-ONLY. Two comparisons, deliberately separated.

(1) FIXED 10-ROUND: what each arm actually did in its allotted rounds --
    unique feasible/docked count and best score. Confounded by SUPPLY: an arm
    that surfaces more feasible candidates gets more oracle draws.

(2) MATCHED ORACLE: K_s = min(N_B, N_C) per seed, comparing best-at-K_s read
    off the budget curve. Isolates candidate QUALITY from candidate supply.

Both are reported because they answer different questions and can disagree.
"""
import json, os, subprocess, sys, statistics as st
from pathlib import Path

CELL = sys.argv[1] if len(sys.argv) > 1 else "5ht1b_s7_d0.6"
SEEDS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["10","11","12"])]
D = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp/bc"; D.mkdir(parents=True, exist_ok=True)

def load(arm, r):
    suf = "narrow_hard" if arm == "B" else "broad"
    f = f"episodes_{CELL}_pooled_{suf}_r{r}.json"
    subprocess.run(["modal", "volume", "get", "compose-v4-artifacts",
                    f"macro_basin/{f}", str(D/f), "--force"], capture_output=True)
    return json.loads((D/f).read_text()) if (D/f).exists() else None

def best_at(d, k):
    """Best-so-far once k unique oracle calls have been spent."""
    bc = [e for e in (d.get("budget_curve") or [])
          if isinstance(e.get("best"), (int, float)) and e.get("oracle_calls", 0) <= k]
    return min((e["best"] for e in bc), default=None)

rows = {}
for r in SEEDS:
    for arm in ("B", "C"):
        d = load(arm, r)
        if d is None:
            continue
        rows[(arm, r)] = dict(round=d.get("round"), n=len(d.get("docked") or {}),
                              feasible=len(d.get("archive") or []),
                              best=d.get("best_ds"), d=d)

print(f"{CELL}\n\n(1) FIXED 10-ROUND  -- confounded by candidate supply")
print(f"  {'arm':<4}{'seed':>5}{'round':>7}{'feasible':>10}{'docked':>8}{'best':>8}")
for (arm, r), v in sorted(rows.items()):
    bs = f"{v['best']:.2f}" if isinstance(v['best'], (int, float)) else "--"
    rd = f"{v['round']}/9"
    print(f"  {arm:<4}{r:>5}{rd:>7}{v['feasible']:>10}{v['n']:>8}{bs:>8}")

print(f"\n(2) MATCHED ORACLE  K_s = min(N_B, N_C) per seed -- isolates quality")
print(f"  {'seed':>5}{'N_B':>6}{'N_C':>6}{'K_s':>6}{'B@K_s':>8}{'C@K_s':>8}{'delta':>8}")
deltas = []
for r in SEEDS:
    b, c = rows.get(("B", r)), rows.get(("C", r))
    if not b or not c:
        continue
    ks = min(b["n"], c["n"])
    bb, cc = best_at(b["d"], ks), best_at(c["d"], ks)
    if bb is None or cc is None:
        print(f"  {r:>5}{b['n']:>6}{c['n']:>6}{ks:>6}{'--':>8}{'--':>8}{'--':>8}")
        continue
    dl = cc - bb
    deltas.append(dl)
    print(f"  {r:>5}{b['n']:>6}{c['n']:>6}{ks:>6}{bb:>8.2f}{cc:>8.2f}{dl:>+8.2f}")
if deltas:
    print(f"\n  mean matched-oracle delta (C - B): {st.mean(deltas):+.2f} kcal/mol")
    print(f"  negative favours C.  replicate noise on this pipeline is 0.70")
print(f"\n  narrow banked 5HT1B d0.6: -11.10 / -11.30 / -10.90  (mean -11.10)")
print(f"  GenMol published 5HT1B d0.6: -12.00")
