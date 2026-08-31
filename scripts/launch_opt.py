"""Spawn the option controller on a DEPLOYED (persistent) Modal app.

`modal run --detach` builds an EPHEMERAL app, which is torn down when the client
session ends -- a function spawned inside it dies with it. A deployed app has no
client session, so a spawned call survives the laptop closing outright.
"""
import json, sys
from pathlib import Path
import modal

ROOT = Path("/Users/rmaganti/compose_v2_work")
cells = [int(c) for c in (sys.argv[1] if len(sys.argv) > 1 else "0,7").split(",")]
deltas = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0.4").split(",")]
tag = sys.argv[3] if len(sys.argv) > 3 else "OPT4"
stages = int(sys.argv[4]) if len(sys.argv) > 4 else 3
stage_len = int(sys.argv[5]) if len(sys.argv) > 5 else 12
n_part = int(sys.argv[6]) if len(sys.argv) > 6 else 100
dock_budget = int(sys.argv[7]) if len(sys.argv) > 7 else 200
rounds = int(sys.argv[8]) if len(sys.argv) > 8 else 0

seeds = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
tasks = []
for i in cells:
    s = seeds[i]
    for dd in deltas:
        tasks.append({"rounds": rounds, "gamma_rule": 0.75, "dock_budget": dock_budget,
                      "beta_dock": 1.0, "smiles": s["smiles"],
                      "target": s["target"], "idx": i, "delta": dd,
                      "beta": 4.0, "tag": tag, "stages": stages,
                      "stage_len": stage_len, "n_part": n_part,
                      "seed_rng": 1000 + i * 31 + int(dd * 10)})

drive = modal.Function.from_name("genmol-t4-particle", "drive")
h = drive.spawn(tasks)
print(f"  SPAWNED on the DEPLOYED app: {h.object_id}")
print(f"  {len(tasks)} cells, {stages}x{stage_len} edits, {n_part} particles, "
      f"{dock_budget} docking calls each")
print(f"  results -> volume t4_particle/{tag}_*.json")
