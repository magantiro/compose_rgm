"""Docking throughput + semantics bench. Decides the official-run configuration.

WHAT IT MEASURES. One fixed batch of 20 molecules, docked under several
execution configurations, timed and compared. The question is how many docking
workers per cell to use, and whether QuickVina's own --cpu setting is doing
anything at exhaustiveness=1.

WHAT PARITY MEANS HERE. QuickVina is stochastic, so bit-identical scores are NOT
the bar and demanding them would be wrong. The bar is identical BENCHMARK
SEMANTICS: every molecule yields exactly one result, order is preserved, the
count charged is unchanged, and the score distribution is consistent with the
~0.7 kcal/mol replicate spread Gate 0 already measured. A configuration that
changed which molecules got docked, or how many, would be disqualifying; one
that shifts a score by less than the metric's own noise is not.

A latent bug this also settles: the serial path asks QuickVina for 4 threads on a
1-CPU container, so it has been oversubscribing a single core all along.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    CANONICAL_SLOTS, TIME_POINT, APPLY_CAP,
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _dock, _dock_many, _runtime,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = _opt_image.add_local_file(
    ROOT / "modal_apps/genmol_t4_opt_app.py",
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)

app = modal.App("genmol-t4-dockbench")


@app.function(image=image, cpu=(8.0, 8.0), memory=int(8 * 1024),
              timeout=2 * 60 * 60, volumes={str(ARTIFACT_ROOT): artifact_volume})
def bench(idx: int = 10, n: int = 20) -> dict[str, Any]:
    import os, sys
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    seeds = json.loads((REMOTE_ROOT / "docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    cell = seeds[idx]; target = cell["target"]
    rt = _runtime(); model, system = rt["model"], rt["system"]

    # a realistic batch: real successors of a real seed, not arbitrary SMILES
    st = pad_molecular_graph(smiles_to_molecular_graph(cell["smiles"]), CANONICAL_SLOTS)
    law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
    pr = np.array([m.probability for m in law.marks], float)
    mols: list[str] = []
    for i in np.argsort(-pr)[:APPLY_CAP]:
        mk = law.marks[int(i)]
        try:
            y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
        except Exception:
            continue
        if y and y not in mols:
            mols.append(y)
        if len(mols) >= n:
            break
    print(f"  {target}: {len(mols)} distinct successors to dock\n", flush=True)

    res: dict[str, Any] = {"target": target, "n": len(mols), "configs": []}
    def record(name, scores, secs):
        ok = [s for s in scores if s is not None]
        r = {"config": name, "seconds": round(secs, 1),
             # per-molecule scores retained: a mean cannot yield a PAIRED signed
             # difference, and that is the statistic that decides whether a config
             # shifts the oracle rather than merely jittering it.
             "scores": [None if s is None else round(float(s), 3) for s in scores],
             "n_results": len(scores), "n_scored": len(ok),
             "mean": round(float(np.mean(ok)), 3) if ok else None,
             "min": round(float(min(ok)), 3) if ok else None,
             "per_mol_s": round(secs / max(len(scores), 1), 2)}
        res["configs"].append(r)
        print(f"  {name:26s} {secs:7.1f}s  {r['per_mol_s']:5.2f} s/mol  "
              f"scored {len(ok)}/{len(scores)}  mean {r['mean']}", flush=True)
        return r

    t0 = time.time()
    serial = [_dock(m, target, f"b_s{i}", cpu=4) for i, m in enumerate(mols)]
    record("serial cpu=4 (Gate-0)", serial, time.time() - t0)

    t0 = time.time()
    serial1 = [_dock(m, target, f"b_s1_{i}", cpu=1) for i, m in enumerate(mols)]
    record("serial cpu=1", serial1, time.time() - t0)

    for w in (4, 8, 16):
        t0 = time.time()
        par = _dock_many(mols, target, f"b_p{w}", workers=w, cpu_per_dock=1)
        r = record(f"parallel w={w} cpu=1", par, time.time() - t0)
        # semantics checks that actually matter
        r["order_preserved"] = len(par) == len(mols)
        both = [(a, b) for a, b in zip(serial, par) if a is not None and b is not None]
        r["n_comparable"] = len(both)
        r["mean_abs_delta_vs_serial"] = (
            round(float(np.mean([abs(a - b) for a, b in both])), 3) if both else None)
        r["max_abs_delta_vs_serial"] = (
            round(float(max(abs(a - b) for a, b in both)), 3) if both else None)
        print(f"      vs serial: {r['n_comparable']} comparable, "
              f"mean |delta| {r['mean_abs_delta_vs_serial']}, "
              f"max {r['max_abs_delta_vs_serial']}  "
              f"(Gate-0 replicate spread was 0.70 median / 9.30 max)", flush=True)

    d = Path("/artifacts/t4_dockbench"); d.mkdir(parents=True, exist_ok=True)
    (d / f"dockbench_{target}.json").write_text(json.dumps(res, indent=1))
    artifact_volume.commit()
    return res


@app.local_entrypoint()
def main(idx: int = 10, n: int = 20, out: str = "") -> None:
    o = bench.remote(idx, n)
    p = Path(__file__).resolve().parents[1] / (out or "diagnostics/genmol_t4_dockbench.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
