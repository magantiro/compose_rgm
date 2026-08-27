"""Can ONE 6-action vocabulary win parp1 AND 5ht1b, which want opposite chemistry?

MEASURED DISSOCIATION, from IVG's own winners:
    5ht1b_s7_d0.4  top-3 saturated-N ring counts = [2,2,2]  (piperazine)
    parp1_s0_d0.4  top-3 saturated-N ring counts = [0,0,0]

Our narrow vocabulary is carbon-only, so it cannot build a saturated-N ring at
all. That is precisely why narrow BEATS GenMol on parp1 (-10.90 vs -10.6,
-12.10 vs -11.0) and LOSES on 5ht1b (-11.40 vs -12.3 / -12.0).

REFINE_RING reaches saturated-N rings 39.4% of the time against 1.6% for
all-molecule restate2. Exposed as a per-ring ACTION (`/r0` vs `/r2`) the
controller chooses per cell, with no chemistry rule and no target prior.

Why this and not `semantics="broad"`: broad enumerates 800 specs and dilutes
benzene 33.3% -> 1.6% (21x). The 6-action vocabulary dilutes 0.08 -> 0.04 (2x)
and spans the same two chemistries. Same claim, a tenth of the exploration tax.

PREDICTION (falsifiable). On 5ht1b the controller concentrates on /r2 -- the
MIRROR of parp1, where it put 2.3-4.5x mass on /r0 -- and beats narrow's -11.40.
If instead it concentrates on /r0 here too, the mechanism is not learning
target-specific chemistry and narrow should be used for the whole table.
"""
import json, os, time
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
CELLS = ["5ht1b_s7_d0.4", "5ht1b_s7_d0.6"]
SEEDS_R = [int(x) for x in os.environ.get("HG_SEEDS", "500,501,502").split(",")]
NARROW = {"5ht1b_s7_d0.4": -11.40, "5ht1b_s7_d0.6": -11.40}
GENMOL = {"5ht1b_s7_d0.4": -12.3, "5ht1b_s7_d0.6": -12.0}
seeds = {f"{s['target']}_s{s['idx']}": s
         for s in json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())}
fn = modal.Function.from_name("macro-basin", "drive_episodes")
hs = {}
for c in CELLS:
    tgt, si, dl = c.rsplit("_", 2)
    sd = seeds[f"{tgt}_{si}"]
    for rs in SEEDS_R:
        hs[(c, rs)] = fn.spawn(
            dict(cell=c, seed=sd["smiles"], delta=float(dl[1:]),
                 target=sd["target"], idx=sd["idx"]),
            n_particles=64, rounds=10, episode_len=5, dock_per_round=10,
            arm="pooled", run_seed=rs, semantics="legacy")
        print(f"  spawned {c} r{rs}  -> episodes_{c}_pooled_rf0_r{rs}.json", flush=True)
t0 = time.time()
best = {}
for (c, rs), h in hs.items():
    try:
        r = h.get(timeout=10800); v = r.get("best")
        best.setdefault(c, []).append(v)
        print(f"  DONE {c} r{rs}  best={v}  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  FAIL {c} r{rs}: {type(e).__name__} {str(e)[:160]}", flush=True)
print(f"\n{'cell':16s} {'GenMol':>8s} {'narrow':>8s} {'learned best':>13s}   verdict")
for c in CELLS:
    v = [x for x in best.get(c, []) if x is not None]
    b = min(v) if v else None
    verdict = "-" if b is None else ("CLOSES GAP" if b < NARROW[c] else "no better than narrow")
    print(f"{c:16s} {GENMOL[c]:>8.1f} {NARROW[c]:>8.2f} {str(b):>13s}   {verdict}")
json.dump(dict(learned=best, narrow=NARROW, genmol=GENMOL),
          open(ROOT / "diagnostics/hetero_gap_test.json", "w"), indent=1)
print("\nwrote diagnostics/hetero_gap_test.json")
