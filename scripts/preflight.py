"""60-120s smoke test. ONE question: does the code instantiate the experiment we
think we're running?

NOT a performance gate. No ESS thresholds, no feasibility rates, no ring counts,
no preregistered criteria -- those test science, and gating launches on them is
bureaucracy. This catches only execution bugs, each of which cost a 30-60 minute
run today: modes never allocated, long-segment strata selected away, a census
patch referencing `nxt` before assignment, and misread persistence.
"""
import json, sys
from pathlib import Path
import modal

ROOT = Path("/Users/rmaganti/compose_v2_work")
s = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())[0]
task = {"gamma_rule": 0.75, "dock_budget": 0, "beta_dock": 0.0,
        "smiles": s["smiles"], "target": s["target"], "idx": 0, "delta": 0.4,
        "beta": 4.0, "tag": "PREFLIGHT", "stages": 1, "stage_len": 2,
        "n_part": 100, "seed_rng": 1}
try:
    r = modal.Function.from_name("genmol-t4-particle", "run_gate").remote(task)
except Exception as e:
    print(f"  [FAIL] 3. iterations complete without exception: {type(e).__name__}: {e}")
    sys.exit(1)

pd = r.get("per_depth") or []
ok = True
def chk(n, passed, ev):
    global ok
    print(f"  [{'PASS' if passed else 'FAIL'}] {n}: {ev}")
    ok &= bool(passed)

d0 = pd[0] if pd else {}
mm, sm = d0.get("mode_mix") or {}, d0.get("seg_mix") or {}
mr, dh = d0.get("mode_realized") or {}, d0.get("mode_dheavy") or {}
chk("1. every mode has particles", bool(mm) and all(v > 0 for v in mm.values()), mm)
chk("2. every segment length has particles", bool(sm) and all(v > 0 for v in sm.values()), sm)
chk("3. iterations completed and artifact saved", bool(pd), f"{len(pd)} depth(s)")
chk("4. telemetry sane (grow>0, shrink<0, some k>1 runs)",
    dh.get("grow", 0) > 0 and dh.get("shrink", 0) < 0
    and any(v > 1.0 for v in mr.values()),
    {"grow": dh.get("grow"), "shrink": dh.get("shrink"), "realized": mr})
print("\nPREFLIGHT " + ("PASSED - go" if ok else "FAILED - do not launch"))
sys.exit(0 if ok else 1)
