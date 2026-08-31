"""Spawn Phase-1 frontier jobs against the DEPLOYED aryl-compile app.

Deliberately NOT a modal local_entrypoint: `modal run` builds an ephemeral app
whose lifetime is tied to this process, which is the exact coupling that lost
the 2026-08-26 run. Function.from_name attaches to the persistent deployment,
so spawn() hands the work to Modal and this process becomes irrelevant.

Usage:
  python3 scripts/frontier_spawn.py --only braf_s9,jak2_s14
  python3 scripts/frontier_spawn.py            # every phase-1 seed
  python3 scripts/frontier_spawn.py --status   # read the Volume, spawn nothing
"""
import argparse
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP = "aryl-compile"


def seed_jobs():
    plan = json.loads((ROOT / "diagnostics/panel30_plan.json").read_text())
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())}
    bases = []
    for cell in plan["phase1"]:
        b = cell.rsplit("_", 1)[0]
        if b not in bases:
            bases.append(b)
    return [dict(cell_base=b, seed=seeds[b]["smiles"]) for b in bases]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    if a.status:
        st = modal.Function.from_name(APP, "frontier_status").remote()
        pres = st["present"]
        print(f"frontiers on Volume: {len(pres)}")
        for c in sorted(pres):
            v = pres[c]
            tag = f"reuse of {v['reuse_of']}" if v.get("reuse_of") else f"{v['seconds']}s"
            print(f"  {c:<20} {str(v['n_feasible']):>5} feasible  "
                  f"min_sim {v['min_sim']}  {tag}")
        if st["partial"]:
            print(f"  in-flight temps (invisible to readers): {st['partial']}")
        plan = json.loads((ROOT / "diagnostics/panel30_plan.json").read_text())
        miss = [c for c in plan["phase1"] if c not in pres]
        print(f"\n{len(pres)}/{len(plan['phase1'])} phase-1 cells; missing: {miss or 'none'}")
        return

    jobs = seed_jobs()
    if a.only:
        want = {s.strip() for s in a.only.split(",") if s.strip()}
        jobs = [j for j in jobs if j["cell_base"] in want]
        unknown = want - {j["cell_base"] for j in jobs}
        if unknown:
            sys.exit(f"unknown cell_base: {sorted(unknown)}")
    f = modal.Function.from_name(APP, "frontier_seed_job")
    ids = []
    for j in jobs:
        call = f.spawn(j)
        ids.append({"cell_base": j["cell_base"], "call_id": call.object_id})
        print(f"  spawned {j['cell_base']:<12} call {call.object_id}")
    out = ROOT / "diagnostics/frontier_spawn_ids.json"
    out.write_text(json.dumps(ids, indent=1))
    print(f"\n{len(ids)} jobs handed to Modal. This process is now irrelevant:")
    print("each job commits its own artifact to the Volume before returning.")
    print(f"call ids -> {out}")


if __name__ == "__main__":
    main()
