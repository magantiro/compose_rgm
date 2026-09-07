#!/usr/bin/env python3
"""Spawn T4 cells against the DEPLOYED app, so they outlive this process.

`modal run --detach` still cancels a .map() when the client dies, and spawning
from an ephemeral `modal run` is no better -- the ephemeral app is torn down the
moment the entrypoint returns, taking its spawned calls with it. Four T4
attempts died that way. A deployed app persists independently, so a call
spawned into it keeps running with nothing local attached.

Usage:
    modal deploy modal_apps/genmol_t4_opt_app.py     # once, after any edit
    python3 tools/t4_launch.py --budget 20 --n-seeds 1 --deltas 0.4 --arms macro_prior
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--deltas", default="0.4")
    ap.add_argument("--arms", default="macro_prior")
    ap.add_argument("--session", default="compose_iclr")
    # V_z = 0 until a T4-specific estimator exists; the QED table was null and
    # learned QED-region effects, and an interface accepting it is not a reason
    # to use it here.  `macro_prior` names the new unlearned option-prior arm.
    ap.add_argument("--lineages", type=int, default=8)
    ap.add_argument("--per-round", type=int, default=20)
    ap.add_argument("--regions-per-lineage", type=int, default=3)
    ap.add_argument("--particles-per-region", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--max-frontier",
        type=int,
        default=8,
        help="search width per bundle (NOT the candidate pool)",
    )
    ap.add_argument(
        "--emit-per-bundle",
        type=int,
        default=3,
        help="representatives offered to the oracle per bundle",
    )
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--epsilon-option", type=float, default=0.1)
    ap.add_argument("--macro-temperature", type=float, default=2.0)
    ap.add_argument("--epsilon-macro", type=float, default=0.15)
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.session):
        raise SystemExit("--session must contain only letters, numbers, '_' or '-'")

    from preflight import assert_synced

    preflight = assert_synced(strict=True)
    if preflight["dirty"]:
        raise SystemExit(
            "PREFLIGHT FAIL: Modal scientific launches require a fully clean "
            "committed tree, not only clean mounted src/configs"
        )

    fn = modal.Function.from_name("genmol-t4-opt", "t4_population_cell")
    seed_manifest = ROOT / "docs/GENMOL_T4_SEEDS.json"
    seeds = json.loads(seed_manifest.read_text())[: a.n_seeds]
    tbl = None
    p = ROOT / "diagnostics/task_value_qed.json"
    if p.exists():
        tbl = json.loads(p.read_text())

    spawned = []
    for i, s in enumerate(seeds):
        for d in [float(x) for x in a.deltas.split(",")]:
            for arm in [x.strip() for x in a.arms.split(",")]:
                t = {
                    "cell": f"{a.session}_{arm}_{s['target']}_{i}_d{d}",
                    "session": a.session,
                    "smiles": s["smiles"],
                    "target": s["target"],
                    "delta": d,
                    "budget": a.budget,
                    "arm": arm,
                    "tau": a.tau,
                    "seed_rng": 1000 + i,
                    "kappa": 1.0,
                    "epsilon_region": 0.2,
                    "epsilon": 0.1,
                    "epsilon_option": a.epsilon_option,
                    "macro_temperature": a.macro_temperature,
                    "epsilon_macro": a.epsilon_macro,
                    "code_revision": preflight["commit"],
                    "seed_manifest_sha256": _sha256(seed_manifest),
                    "lineages": a.lineages,
                    "per_round": a.per_round,
                    "regions_per_lineage": a.regions_per_lineage,
                    "particles_per_region": a.particles_per_region,
                    "workers": a.workers,
                    "max_frontier": a.max_frontier,
                    "emit_per_bundle": a.emit_per_bundle,
                }
                if arm == "Q_taskvalue":
                    if tbl is None:
                        raise SystemExit("no task_value table for the tilt arm")
                    t["value_table"] = tbl
                spawned.append((t["cell"], fn.spawn(t).object_id))

    (ROOT / "diagnostics").mkdir(exist_ok=True)
    receipt_path = ROOT / "diagnostics/t4_spawned_three_level.json"
    receipt_payload = json.dumps(
        {
            "schema_version": "t4_three_level_spawn_v1",
            "session": a.session,
            "code_revision": preflight["commit"],
            "seed_manifest_sha256": _sha256(seed_manifest),
            "budget": a.budget,
            "epsilon_option": a.epsilon_option,
            "macro_temperature": a.macro_temperature,
            "epsilon_macro": a.epsilon_macro,
            "cells": [{"cell": c, "call": o} for c, o in spawned],
        },
        indent=2,
        sort_keys=True,
    )
    temporary_path = receipt_path.with_suffix(".json.tmp")
    temporary_path.write_text(receipt_payload)
    temporary_path.replace(receipt_path)
    print(f"spawned {len(spawned)} cell(s) into the DEPLOYED app; nothing local is holding them")
    print(f"receipt: {receipt_path}")
    for c, o in spawned[:8]:
        print(f"  {c}  {o}")


if __name__ == "__main__":
    main()
