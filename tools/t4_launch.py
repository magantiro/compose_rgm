#!/usr/bin/env python3
"""Spawn T4 cells against the DEPLOYED app, so they outlive this process.

`modal run --detach` still cancels a .map() when the client dies, and spawning
from an ephemeral `modal run` is no better -- the ephemeral app is torn down the
moment the entrypoint returns, taking its spawned calls with it. Four T4
attempts died that way. A deployed app persists independently, so a call
spawned into it keeps running with nothing local attached.

Usage:
    modal deploy modal_apps/genmol_t4_opt_app.py     # once, after any edit
    python3 tools/t4_launch.py --budget 200 --n-seeds 1 --deltas 0.4 --arms mu_exec
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--deltas", default="0.4,0.6")
    ap.add_argument("--arms", default="mu_exec,Q_taskvalue")
    ap.add_argument("--lineages", type=int, default=8)
    ap.add_argument("--per-round", type=int, default=20)
    ap.add_argument("--regions-per-lineage", type=int, default=3)
    ap.add_argument("--particles-per-region", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.05)
    a = ap.parse_args()

    fn = modal.Function.from_name("genmol-t4-opt", "t4_population_cell")
    seeds = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())[: a.n_seeds]
    tbl = None
    p = ROOT / "diagnostics/task_value_qed.json"
    if p.exists():
        tbl = json.loads(p.read_text())

    spawned = []
    for i, s in enumerate(seeds):
        for d in [float(x) for x in a.deltas.split(",")]:
            for arm in [x.strip() for x in a.arms.split(",")]:
                t = {"cell": f"{arm}_{s['target']}_{i}_d{d}", "smiles": s["smiles"],
                     "target": s["target"], "delta": d, "budget": a.budget,
                     "arm": arm, "tau": a.tau, "seed_rng": 1000 + i, "kappa": 1.0,
                     "epsilon_region": 0.2, "lineages": a.lineages,
                     "per_round": a.per_round,
                     "regions_per_lineage": a.regions_per_lineage,
                     "particles_per_region": a.particles_per_region,
                     "workers": a.workers}
                if arm == "Q_taskvalue":
                    if tbl is None:
                        raise SystemExit("no task_value table for the tilt arm")
                    t["value_table"] = tbl
                spawned.append((t["cell"], fn.spawn(t).object_id))

    (ROOT / "diagnostics").mkdir(exist_ok=True)
    (ROOT / "diagnostics/t4_spawned.json").write_text(json.dumps(
        {"budget": a.budget, "cells": [{"cell": c, "call": o} for c, o in spawned]}))
    print(f"spawned {len(spawned)} cell(s) into the DEPLOYED app; "
          f"nothing local is holding them")
    for c, o in spawned[:8]:
        print(f"  {c}  {o}")


if __name__ == "__main__":
    main()
