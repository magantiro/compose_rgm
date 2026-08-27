"""Run the restate2 vs REFINE_RING controller gate against the DEPLOYED app.

Spawned as a deployed-function map rather than a `modal run` local entrypoint:
a local client death during a long map discards in-flight work, and this is the
gate that decides whether PMO launches.
"""
import json, os, sys, time
import modal

SEED = os.environ.get("GATE_SEED", "FC(F)(F)c1cc(N2CC[NH2+]CC2)ccc1")
SHARDS = int(os.environ.get("GATE_SHARDS", "40"))
PER = int(os.environ.get("GATE_PER", "75"))          # trials per arm per shard
OUT = os.environ.get("GATE_OUT", "diagnostics/restate_gate.json")

f = modal.Function.from_name("aryl-compile", "restate_gate_shard")
jobs = [dict(seed=SEED, n=PER, L=2, seed_rng=1000 + 17 * i) for i in range(SHARDS)]
print(f"gate: {SHARDS} shards x {PER} trials x 3 arms = {SHARDS*PER} per arm", flush=True)
t0 = time.time()
res = [r for r in f.map(jobs, order_outputs=False, return_exceptions=True)]
ok = [r for r in res if isinstance(r, dict) and "error" not in r and "P" in r]
bad = [r for r in res if r not in ok]
print(f"done in {time.time()-t0:.0f}s  ok={len(ok)} bad={len(bad)}", flush=True)
if bad:
    print("first bad:", str(bad[0])[:400], flush=True)
if not ok:
    sys.exit("no usable shards")

KEYS = ("n","ran","halted","touch_ring_step1","touch_ring_any","lost_arom",
        "gained_sat_n","gained_pip","both","uniq_end")
agg = {a: {k: sum(int(r[a][k]) for r in ok) for k in KEYS} for a in ("P","U","S")}
ex = sorted({s for r in ok for s in r.get("scoped_sat_n_examples", [])})
out = dict(x0=ok[0]["x0"], R=ok[0]["R"], base=ok[0]["base"], shards=len(ok),
           arms=agg, scoped_sat_n_examples=ex[:12])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=2)

print("\nX0 =", out["x0"], " new-ring atoms", out["R"], " base", out["base"])
NAME = {"P": "P production restate2", "U": "U unscoped (state space)",
        "S": "S REFINE_RING (ring-scoped)"}
for a in ("P", "U", "S"):
    d = agg[a]; n = d["n"] or 1
    print(f"\n{NAME[a]}   n={d['n']}  completed 2 steps={d['ran']}  halted={d['halted']}")
    for k in ("touch_ring_step1","touch_ring_any","lost_arom","gained_sat_n","gained_pip","both"):
        print(f"   {k:18s} {d[k]:6d}   {100.0*d[k]/n:6.2f}%")
    print(f"   {'distinct endpoints':18s} {d['uniq_end']:6d}")
print(f"\nwrote {OUT}")
