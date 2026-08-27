"""Broad vocabulary on the two cells narrow LOSES: 5ht1b_s7 d0.4 and d0.6.

WHY BROAD IS THE ONLY ARM THAT CAN DO THIS. The gap is precisely one saturated-N
ring: our winners and IVG's match on heavy atoms (28-30 vs 31), ring count
(5 vs 5) and aromatic count (2-3 vs 3), and differ only in satN (1 vs 2). We add
saturated carbocycles where IVG adds a piperidine.

`realize()` finds 40 legal piperidine realizations on this seed, so the
chemistry is reachable -- but the LEGACY path builds rings through
build_ring_system_exact, which UNSATs at grow2 once N enters the quota (growth
commits to a top-ranked descriptor and cannot backtrack). Verified 2026-08-27:

    ring:linked/6/C6/saturated    -> ok, C1CCCCC1
    ring:linked/6/C5N1/saturated  -> UNSAT@grow2
    ring:fused/6/C5N1/saturated   -> UNSAT@grow2

semantics="broad" goes through realize() instead, so it CAN construct the ring.

COST: broad enumerates 800 specs and starts benzene at 1.6% against 33.3%
narrow -- a 21x exploration handicap. On a cell where narrow already loses by
0.6-0.9 that is a cheap gamble, not a risk to a banked win.

Narrow baseline -11.40 on both cells. GenMol -12.3 (d0.4) / -12.0 (d0.6).
"""
import json, os, time
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
CELLS = ["5ht1b_s7_d0.4", "5ht1b_s7_d0.6"]
RUN_SEEDS = [int(x) for x in os.environ.get("B5_SEEDS", "600,601,602").split(",")]
NARROW = -11.40
GEN = {"5ht1b_s7_d0.4": -12.3, "5ht1b_s7_d0.6": -12.0}
seeds = {f"{s['target']}_s{s['idx']}": s
         for s in json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())}
fn = modal.Function.from_name("macro-basin", "drive_episodes")
hs = {}
for c in CELLS:
    tgt, si, dl = c.rsplit("_", 2)
    sd = seeds[f"{tgt}_{si}"]
    for rs in RUN_SEEDS:
        hs[(c, rs)] = fn.spawn(
            dict(cell=c, seed=sd["smiles"], delta=float(dl[1:]),
                 target=sd["target"], idx=sd["idx"]),
            n_particles=64, rounds=10, episode_len=5, dock_per_round=10,
            arm="pooled", run_seed=rs, semantics="broad")
        print(f"  spawned BROAD {c} r{rs} -> episodes_{c}_pooled_broad_rf0_r{rs}.json",
              flush=True)
t0 = time.time(); best = {}
for (c, rs), h in hs.items():
    try:
        r = h.get(timeout=10800); v = r.get("best")
        best.setdefault(c, []).append(v)
        print(f"  DONE {c} r{rs} best={v} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  FAIL {c} r{rs}: {type(e).__name__} {str(e)[:150]}", flush=True)
print(f"\n{'cell':16s} {'GenMol':>8s} {'narrow':>8s} {'broad best':>11s}   verdict")
for c in CELLS:
    v = [x for x in best.get(c, []) if x is not None]
    b = min(v) if v else None
    ver = "-" if b is None else ("BEATS NARROW" if b < NARROW else "no better")
    if b is not None and b < GEN[c]: ver = "BEATS GENMOL"
    print(f"{c:16s} {GEN[c]:>8.1f} {NARROW:>8.2f} {str(b):>11s}   {ver}")
json.dump(dict(broad=best, narrow=NARROW, genmol=GEN),
          open(ROOT / "diagnostics/broad_5ht1b.json", "w"), indent=1)
print("\nwrote diagnostics/broad_5ht1b.json")
