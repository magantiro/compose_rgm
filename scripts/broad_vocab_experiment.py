"""Does ONE general ring vocabulary learn OPPOSITE chemistry on two targets?

The parp1 gap against InVirtuoGen is a vocabulary-EXPOSURE gap, not a chemistry
gap: ring:fused/6/C4N1O1/saturated is requestable today, but ACTION_SPACE
exposes only 6 carbon-only specs -- which is why REFINE_RING existed as a
workaround. Hand-adding a lactam spec because IVG's parp1 winners contain
lactams would fit the benchmark from its own answer key, and is barred.

So expose the FULL space (semantics="broad") and let each cell learn.

MEASURED SETUP (not assumed):
  qualified sizes   aromatic [5,6,9,10]   saturated [3,4,5,6,7,8,9,10]
                    identical for both seeds -- seed-only qualification is not
                    seed-sensitive, so ring size is NOT the limiting factor
  enumerated specs  800
  P(benzene) at uniform start = 1/2 * 1/2 * 1/4 * 1/4 = 1.6%, against 33.3%
  under the narrow menu. This reproduces the historically measured 1.0% and is
  the exploration handicap broad must overcome.

WHY IT IS NOT A 1-OF-800 DRAW. FactorizedRingPolicy samples axis by axis
(topology, state, size, n_hetero, then element identities), so concentrating on
any motif is four small decisions, and credit factorizes over the same axes --
"one N" generalizes across sizes, "N rather than O" across counts. R_theta's
semantic prior tilts the draw before learning starts.

FALSIFIABLE READOUT. The two cells want OPPOSITE chemistry per IVG's own
winners: parp1_s0_d0.4 best -13.6 with satN=0 (fused aromatic + lactam);
5ht1b_s7_d0.4 best -13.0 with a saturated-N piperidine motif. If one vocabulary
concentrates toward aromatic on parp1 AND toward saturated-N on 5ht1b, the
controller is discovering target-specific chemistry. PRIMARY readout is the
per-axis concentration, because at 100 calls docking is confounded by the 21x
exploration handicap above; docking is secondary.
"""
import json, os, time
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
CELLS = ["parp1_s0_d0.4", "5ht1b_s7_d0.4"]
RUN_SEED = int(os.environ.get("BROAD_SEED", "400"))
seeds = {f"{s['target']}_s{s['idx']}": s
         for s in json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())}
fn = modal.Function.from_name("macro-basin", "drive_episodes")
hs = {}
for c in CELLS:
    tgt, si, dl = c.rsplit("_", 2)
    sd = seeds[f"{tgt}_{si}"]
    hs[c] = fn.spawn(dict(cell=c, seed=sd["smiles"], delta=float(dl[1:]),
                          target=sd["target"], idx=sd["idx"]),
                     n_particles=64, rounds=10, episode_len=5,
                     dock_per_round=10, arm="pooled", run_seed=RUN_SEED,
                     semantics="broad")
    print(f"  spawned BROAD {c} r{RUN_SEED}: {hs[c].object_id}", flush=True)
    print(f"    checkpoint -> episodes_{c}_pooled_broad_rf0_r{RUN_SEED}.json", flush=True)
t0 = time.time()
out = {}
for c, h in hs.items():
    try:
        r = h.get(timeout=10800); out[c] = r.get("best")
        print(f"  DONE {c}  best={r.get('best')}  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  FAIL {c}: {type(e).__name__} {str(e)[:200]}", flush=True)
IVG = {"parp1_s0_d0.4": -13.6, "5ht1b_s7_d0.4": -13.0}
NARROW = {"parp1_s0_d0.4": -10.73, "5ht1b_s7_d0.4": None}
print(f"\n{'cell':18s} {'BROAD':>8s} {'narrow carbon':>14s} {'IVG':>7s}")
for c in CELLS:
    print(f"{c:18s} {str(out.get(c)):>8s} {str(NARROW[c]):>14s} {IVG[c]:>7.1f}")
json.dump(dict(run_seed=RUN_SEED, broad=out, ivg=IVG, narrow=NARROW),
          open(ROOT / "diagnostics/broad_vocab_experiment.json", "w"), indent=1)
print("\nwrote diagnostics/broad_vocab_experiment.json")
