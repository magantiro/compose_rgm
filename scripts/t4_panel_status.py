"""Live view of the official T4 panel from the per-round checkpoints.

drive_episodes writes its checkpoint INSIDE the rounds loop, so every run on the
volume shows its current round, best docking score and feasible count while it
is still running. Nothing here writes, resamples or filters.

    python scripts/t4_panel_status.py            # one snapshot
    python scripts/t4_panel_status.py --watch    # refresh every 2 min
"""
import json, re, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

SEEDS = (200, 201, 202)
ROUNDS = 10
LOCAL = Path("/tmp/t4_panel_ck")
PROG = Path(__file__).resolve().parents[1] / "diagnostics/t4_official_panel_progress.json"


def pull():
    LOCAL.mkdir(parents=True, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "compose-v4-artifacts",
                    "macro_basin", str(LOCAL), "--force"],
                   capture_output=True, text=True)
    return LOCAL / "macro_basin"


def snapshot():
    d = pull()
    pat = re.compile(r"episodes_(.+)_pooled_r(\d+)\.json$")
    runs = []
    for p in sorted(d.glob("episodes_*_pooled_r*.json")) if d.exists() else []:
        m = pat.search(p.name)
        if not m:
            continue
        cell, rs = m.group(1), int(m.group(2))
        if rs not in SEEDS:
            continue                      # ignore pre-REFINE_RING development runs
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        pr = j.get("provenance", {}) or {}
        blob = json.dumps(j)
        runs.append(dict(cell=cell, rs=rs, round=int(j.get("round", -1)) + 1,
                         best=j.get("best_ds"), nfeas=j.get("n_feasible", 0),
                         refine=pr.get("refine"),
                         refine_markers=len(re.findall(r"refine\[\d+\]", blob)),
                         mtime=p.stat().st_mtime))
    return runs


def show(runs):
    now = time.strftime("%H:%M:%S")
    done = [r for r in runs if r["round"] >= ROUNDS]
    live = [r for r in runs if 0 < r["round"] < ROUNDS]
    print(f"\n=== OFFICIAL T4 PANEL  {now} ===")
    print(f"  runs on volume {len(runs)}/90    complete {len(done)}    in-flight {len(live)}")
    bad = [r for r in runs if r["refine"] != 2]
    if bad:
        print(f"  !! {len(bad)} run(s) NOT at refine=2: {[(r['cell'], r['rs']) for r in bad][:5]}")
    if live:
        print("\n  IN FLIGHT")
        for r in sorted(live, key=lambda x: -x["mtime"])[:12]:
            age = (time.time() - r["mtime"]) / 60
            print(f"    {r['cell']:20s} r{r['rs']}  round {r['round']:2d}/{ROUNDS}"
                  f"  best={r['best']}  feasible={r['nfeas']}"
                  f"  refine_fired={r['refine_markers']}  ({age:.0f}m ago)")
    by_t = defaultdict(list)
    for r in done:
        if r["best"] is not None:
            by_t[r["cell"].split("_")[0]].append(float(r["best"]))
    if by_t:
        print("\n  COMPLETED RUNS, mean best docking score by target")
        for t in sorted(by_t):
            v = by_t[t]
            print(f"    {t:8s} n={len(v):2d}  mean {sum(v)/len(v):7.2f}   best {min(v):7.2f}")
    fired = sum(1 for r in runs if r["refine_markers"] > 0)
    print(f"\n  runs with REFINE_RING visibly fired: {fired}/{len(runs)}")
    if PROG.exists():
        try:
            pg = json.loads(PROG.read_text())
            print(f"  driver: {pg.get('completed')}/90 collected, "
                  f"budget {pg.get('budget_calls')} calls/cell, hash {pg.get('frozen_hash')}")
        except Exception:
            pass


if __name__ == "__main__":
    watch = "--watch" in sys.argv
    while True:
        show(snapshot())
        if not watch:
            break
        time.sleep(120)
