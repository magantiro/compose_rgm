"""Complete the 30-cell T4 table on the FROZEN NARROW controller.

Narrow = the exact vocabulary that produced the 30 banked runs across 10 cells:
three carbon ring specs at 0.08 initial mass each, no refinement. Against
GenMol's published Table 4, at 100 oracle calls versus their 1000, narrow is
6 better / 1 tie / 2 worse on the 9 matched cells.

This fills the remaining 20 cells x 3 independent runs. It does NOT change the
controller: a table mixing vocabularies would be two methods under one name.

Resumable: a cell/seed already holding a COMPLETED run (final round, 10x10
budget) is skipped, so a laptop sleep costs the queue, not the work.
"""
import json, os, re, subprocess, time
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
POOL = int(os.environ.get("FILL_POOL", "2"))
RUN_SEEDS = [0, 1, 2]
STATE = ROOT / "diagnostics/t4_narrow_fill_progress.json"

seeds = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
by_key = {f"{s['target']}_s{s['idx']}": s for s in seeds}
cells = [f"{s['target']}_s{s['idx']}_d{d}" for s in seeds for d in ("0.4", "0.6")]

def completed():
    dst = Path("/tmp/fill_ck"); dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "compose-v4-artifacts", "macro_basin",
                    str(dst), "--force"], capture_output=True, text=True)
    d = dst / "macro_basin"
    done = set()
    for f in (d.glob("episodes_*_pooled_r[0-9].json") if d.exists() else []):
        m = re.match(r"episodes_(.+)_pooled_r(\d+)\.json$", f.name)
        if not m: continue
        try: j = json.loads(f.read_text())
        except Exception: continue
        pr = j.get("provenance", {}) or {}
        if (int(j.get("round", -1)) + 1 >= 10
                and (pr.get("rounds"), pr.get("dock_per_round")) == (10, 10)
                and j.get("best_ds") is not None):
            done.add((m.group(1), int(m.group(2))))
    return done

have = completed()
jobs = [(c, r) for c in cells for r in RUN_SEEDS if (c, r) not in have]
print(f"  banked complete: {len(have)} runs")
print(f"  to run: {len(jobs)} runs across {len({c for c,_ in jobs})} cells", flush=True)

fn = modal.Function.from_name("macro-basin", "drive_episodes")
def spawn(cell, rs):
    tgt, si, dl = cell.rsplit("_", 2)
    sd = by_key[f"{tgt}_{si}"]
    return fn.spawn(dict(cell=cell, seed=sd["smiles"], delta=float(dl[1:]),
                         target=sd["target"], idx=sd["idx"]),
                    n_particles=64, rounds=10, episode_len=5, dock_per_round=10,
                    arm="pooled", run_seed=rs, semantics="legacy")

t0 = time.time(); pending = list(jobs); inflight = {}; results = []
while pending or inflight:
    while pending and len(inflight) < POOL:
        c, rs = pending.pop(0)
        h = spawn(c, rs); inflight[h.object_id] = (h, c, rs, time.time())
        print(f"  spawn {c} r{rs}  ({len(results)} done, {len(pending)} queued)", flush=True)
    time.sleep(20)
    for oid in list(inflight):
        h, c, rs, ts = inflight[oid]
        try:
            r = h.get(timeout=0)
        except BaseException as e:
            if isinstance(e, TimeoutError) or "Timeout" in type(e).__name__: continue
            print(f"  FAIL {c} r{rs}: {type(e).__name__} {str(e)[:120]}", flush=True)
            results.append(dict(cell=c, seed=rs, best=None)); del inflight[oid]; continue
        print(f"  DONE {c} r{rs} best={r.get('best')} ({time.time()-ts:.0f}s) "
              f"[{len(results)+1}/{len(jobs)}]", flush=True)
        results.append(dict(cell=c, seed=rs, best=r.get("best")))
        del inflight[oid]
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(dict(controller="narrow-3spec",
                                         budget_calls=100, results=results), indent=2))
print(f"\nfill complete: {len(results)} runs in {(time.time()-t0)/3600:.1f} h")
