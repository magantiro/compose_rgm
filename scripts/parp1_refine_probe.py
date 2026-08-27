"""Fast parp1 probe: does CONTROLLER-SELECTED refinement recover the loss?

Two cells where matched baselines already exist at the identical 100-call
contract (10 rounds x 10 docks, 64 particles):

    parp1_s0_d0.4   old (no refine) -10.73   forced refine=2 -10.00
    parp1_s1_d0.4   old (no refine) -11.53   forced refine=2 -10.90

Refinement is now a parameter of the ring ACTION (`ring:.../rN`), so the
controller chooses per cell from docking evidence rather than refining every
ring unconditionally. n=1 per cell is a SIGNAL, not a result: docking sd is
0.05-1.61 depending on the molecule.
"""
import json, os, time
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
CELLS = ["parp1_s0_d0.4", "parp1_s1_d0.4"]
RUN_SEED = int(os.environ.get("PROBE_SEED", "300"))
seeds = {f"{s['target']}_s{s['idx']}": s
         for s in json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())}
fn = modal.Function.from_name("macro-basin", "drive_episodes")
hs = {}
for c in CELLS:
    tgt, si, dl = c.rsplit("_", 2)
    sd = seeds[f"{tgt}_{si}"]
    payload = dict(cell=c, seed=sd["smiles"], delta=float(dl[1:]),
                   target=sd["target"], idx=sd["idx"])
    hs[c] = fn.spawn(payload, n_particles=64, rounds=10, episode_len=5,
                     dock_per_round=10, arm="pooled", run_seed=RUN_SEED,
                     semantics="legacy")
    print(f"  spawned {c} r{RUN_SEED}: {hs[c].object_id}", flush=True)
t0 = time.time()
out = {}
for c, h in hs.items():
    try:
        r = h.get(timeout=7200)
        out[c] = r.get("best")
        print(f"  DONE {c}  best={r.get('best')}  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  FAIL {c}: {type(e).__name__} {str(e)[:200]}", flush=True)
REF = {"parp1_s0_d0.4": (-10.73, -10.00), "parp1_s1_d0.4": (-11.53, -10.90)}
print(f"\n{'cell':18s} {'old(no refine)':>15s} {'forced r2':>10s} {'LEARNED':>9s}   verdict")
for c in CELLS:
    o, f2 = REF[c]; n = out.get(c)
    v = "-" if n is None else ("recovers" if n <= o + 0.05 else
                               "better than forced" if n < f2 else "no better")
    print(f"{c:18s} {o:15.2f} {f2:10.2f} {str(n):>9s}   {v}")
json.dump(dict(run_seed=RUN_SEED, learned=out, reference=REF),
          open(ROOT / "diagnostics/parp1_refine_probe.json", "w"), indent=1)
print(f"\nwrote diagnostics/parp1_refine_probe.json   elapsed {time.time()-t0:.0f}s")
