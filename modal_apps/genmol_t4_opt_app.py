"""GenMol Table 4, Level 1 optimizer: local canonical-fiber sweep with
continuation from realized states.

    enumerate S(x)  ->  R_theta-weighted sweep  ->  select for docking
      ->  dock  ->  continue from realized winners

WHAT MAKES THIS COMPOSE-NATIVE RATHER THAN A GENETIC ALGORITHM. Every successor
in the fiber yields R_theta(y|x), QED, SA and similarity-to-seed BEFORE a single
docking call is spent. The complete legal local menu is free; the expensive
evaluation is spent only where the process chooses to spend it. And a docked
winner is itself a molecular state, so the next round continues FROM it rather
than regenerating from the seed. Edits already made are not discarded.

THE THREE-WAY CONSTRAINT SEPARATION (docs/AMENDMENT_GENMOL_T4.md, decision 3).

    hard at EVERY step   executability only: valence, connectivity, declared
                         state space, rewrite support. Enforced by the executor.
    GUIDANCE             QED, SA, similarity, docking. May rise AND FALL.
    RETURNED only        QED>=0.6, SA<=4, sim>=delta; among those, best docking.

v(x) is a NAVIGATION SIGNAL, NOT A MASK. Nine of the fifteen benchmark seeds have
v > 0, so masking on v would forbid leaving the seed at all. Trajectories may dip
below the similarity floor and climb back. Only the returned molecule is judged.

SELECTION RULES are a small predeclared family. One is chosen ONCE on disjoint
development molecules and frozen; it is never re-tuned per target or per delta.

    rtheta   sample proportional to R_theta(y|x).  Ignores v. The control that
             asks whether the learned law alone is already a useful proposal.
    tilt     sample proportional to R_theta(y|x) * exp(-v(y)/TAU_V).
    rank     take the best by v, ties broken by R_theta.

BUDGET. ROUNDS * PER_ROUND docking calls per cell, exactly. The published
protocol is 10 iterations x 100 generations = 1,000; development runs a reduced
budget so the ceiling is preserved for the official cells. The seed's own docking
score is NOT charged: it is published in actives.csv and was reproduced in Gate 0.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

MOOD = "https://raw.githubusercontent.com/SeulLee05/MOOD/main/scorer"

image = (
    _base_image
    .apt_install("openbabel", "curl", "ca-certificates")
    .run_commands(
        "mkdir -p /opt/dock/receptors",
        f"curl -sSL -o /opt/dock/qvina02 {MOOD}/qvina02",
        "chmod +x /opt/dock/qvina02",
        *[f"curl -sSL -o /opt/dock/receptors/{t}.pdbqt {MOOD}/receptors/{t}.pdbqt"
          for t in ("parp1", "fa7", "5ht1b", "braf", "jak2")],
    )
    .add_local_file(ROOT / "docs/GENMOL_T4_SEEDS.json",
                    str(REMOTE_ROOT / "docs/GENMOL_T4_SEEDS.json"), copy=True)
    .add_local_file(ROOT / "docs/GENMOL_T4_DEV_SEEDS.json",
                    str(REMOTE_ROOT / "docs/GENMOL_T4_DEV_SEEDS.json"), copy=True)
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
)

app = modal.App("genmol-t4-opt")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48

QED_MIN, SA_MAX, TAU_V = 0.6, 4.0, 0.10
APPLY_CAP = 300          # marks executed per parent, top-R_theta first
BOXES = {
    "fa7":   ((10.131, 41.879, 32.097), (20.673, 20.198, 21.362)),
    "parp1": ((26.413, 11.282, 27.238), (18.521, 17.479, 19.995)),
    "5ht1b": ((-26.602, 5.277, 17.898), (22.5, 22.5, 22.5)),
    "jak2":  ((114.758, 65.496, 11.345), (19.033, 17.929, 20.283)),
    "braf":  ((84.194, 6.949, -7.081), (22.032, 19.211, 14.106)),
}

_RT: dict[str, Any] = {}


def _runtime():
    if "model" in _RT:
        return _RT
    import sys

    import torch
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source)
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime, load_materialized_scorer_state)
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME)
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system)
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src, materialized_state=bundle)
    model = runtime.model
    ck = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    _RT.update({"model": model, "system": _default_rewrite_system(model)})
    return _RT


def _dock(smiles: str, target: str, tag: str) -> float | None:
    import os
    import subprocess
    d = f"/tmp/{tag}"
    os.makedirs(d, exist_ok=True)
    mol, lig, out = f"{d}/l.mol", f"{d}/l.pdbqt", f"{d}/o.pdbqt"
    for p in (mol, lig, out):
        if os.path.exists(p):
            os.remove(p)
    try:
        subprocess.run(["obabel", f"-:{smiles}", "--gen3D", "-O", mol],
                       capture_output=True, timeout=120, check=True)
        subprocess.run(["obabel", mol, "-O", lig],
                       capture_output=True, timeout=60, check=True)
    except Exception:
        return None
    (cx, cy, cz), (sx, sy, sz) = BOXES[target]
    try:
        subprocess.run(
            ["/opt/dock/qvina02", "--receptor", f"/opt/dock/receptors/{target}.pdbqt",
             "--ligand", lig, "--out", out,
             "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
             "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
             "--cpu", "4", "--num_modes", "10", "--exhaustiveness", "1"],
            capture_output=True, timeout=300, check=True)
        for line in open(out):
            if line.startswith("REMARK VINA RESULT"):
                return float(line.split()[3])
    except Exception:
        return None
    return None


@app.function(image=image, cpu=(1.0, 1.0), memory=int(6 * 1024),
              timeout=10 * 60 * 60, max_containers=40,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def optimize(task: dict[str, Any]) -> dict[str, Any]:
    import os
    import sys

    import numpy as np
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDConfig, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    rt = _runtime()
    model, system = rt["model"], rt["system"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed, delta, target = task["smiles"], task["delta"], task["target"]
    rounds, per_round = task["rounds"], task["per_round"]
    parents_n, rule = task["parents"], task["rule"]
    seed_fp = gen.GetFingerprint(Chem.MolFromSmiles(seed))
    rng = np.random.default_rng(task["seed_rng"])
    T = {"fiber": 0.0, "props": 0.0, "dock": 0.0}

    def props(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        q = float(QED.qed(m)); s = float(sascorer.calculateScore(m))
        sim = float(DataStructs.TanimotoSimilarity(seed_fp, gen.GetFingerprint(m)))
        v = max(max(0.0, QED_MIN - q) / QED_MIN, max(0.0, s - SA_MAX) / SA_MAX,
                max(0.0, delta - sim) / delta)
        return {"qed": q, "sa": s, "sim": sim, "v": v}

    t0 = time.perf_counter()
    archive = [{"smiles": seed, **props(seed), "ds": None, "round": 0}]
    docked: dict[str, float] = {}
    n_dock = 0

    for rd in range(1, rounds + 1):
        # parents: prefer feasible-with-good-docking, else low v.  Realized
        # states only -- this is where continuation from winners happens.
        scored = sorted(
            archive,
            key=lambda a: (a["v"] > 0,
                           a["ds"] if (a["v"] == 0 and a["ds"] is not None) else 0.0,
                           a["v"]))
        parents = scored[:parents_n]

        tf = time.perf_counter()
        cand: dict[str, dict] = {}
        for p in parents:
            try:
                st = pad_molecular_graph(smiles_to_molecular_graph(p["smiles"]),
                                         CANONICAL_SLOTS)
                law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            except Exception:
                continue
            if not law.marks:
                continue
            pr = np.array([m.probability for m in law.marks], float)
            for i in np.argsort(-pr)[:APPLY_CAP]:
                mk = law.marks[int(i)]
                try:
                    y = canonical_state_key(system.apply(
                        st, mk.executor_rule_name, mk.action))
                except Exception:
                    continue
                if not y or y == p["smiles"] or y in docked or y in cand:
                    continue
                cand[y] = {"rtheta": float(pr[int(i)])}
        T["fiber"] += time.perf_counter() - tf

        if not cand:
            break
        tp = time.perf_counter()
        for y in list(cand):
            pv = props(y)
            if pv is None:
                del cand[y]
            else:
                cand[y].update(pv)
        T["props"] += time.perf_counter() - tp
        if not cand:
            break

        keys = list(cand)
        k = min(per_round, len(keys))
        if rule == "rank":
            pick = sorted(keys, key=lambda y: (cand[y]["v"], -cand[y]["rtheta"]))[:k]
        else:
            w = np.array([cand[y]["rtheta"] for y in keys], float)
            if rule == "tilt":
                w = w * np.exp(-np.array([cand[y]["v"] for y in keys]) / TAU_V)
            w = np.clip(w, 1e-30, None); w /= w.sum()
            pick = [keys[i] for i in rng.choice(len(keys), size=k, replace=False, p=w)]

        td = time.perf_counter()
        for j, y in enumerate(pick):
            ds = _dock(y, target, f"c{task['idx']}_{rd}_{j}")
            n_dock += 1
            docked[y] = ds if ds is not None else 0.0
            if ds is not None:
                archive.append({"smiles": y, **cand[y], "ds": ds, "round": rd})
        T["dock"] += time.perf_counter() - td

    feas = [a for a in archive if a["v"] == 0.0 and a["ds"] is not None]
    best = min(feas, key=lambda a: a["ds"]) if feas else None
    return {**task, "n_docked": n_dock, "n_archive": len(archive),
            "n_feasible": len(feas),
            "best_ds": (best["ds"] if best else None),
            "best_smiles": (best["smiles"] if best else None),
            "best_round": (best["round"] if best else None),
            "timing": {k: round(v, 1) for k, v in T.items()},
            "seconds": round(time.perf_counter() - t0, 1)}


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(which: str, rounds: int, per_round: int, parents: int,
          rule: str, limit: int, deltas: str) -> dict[str, Any]:
    f = "GENMOL_T4_SEEDS.json" if which == "bench" else "GENMOL_T4_DEV_SEEDS.json"
    raw = json.loads((REMOTE_ROOT / f"docs/{f}").read_text())
    seeds = raw if isinstance(raw, list) else raw["seeds"]
    dl = [float(x) for x in deltas.split(",")]
    tasks = [{**s, "idx": i, "delta": d, "rounds": rounds, "per_round": per_round,
              "parents": parents, "rule": rule, "seed_rng": 20260820 + i}
             for i, (s, d) in enumerate((s, d) for s in seeds for d in dl)]
    if limit:
        tasks = tasks[:limit]
    print(f"{len(tasks)} cells   rule={rule}  rounds={rounds} x per_round="
          f"{per_round} = {rounds*per_round} dockings/cell   parents={parents}\n",
          flush=True)
    raw_out = list(optimize.map(tasks, order_outputs=True, return_exceptions=True,
                                wrap_returned_exceptions=False))
    out = [r for r in raw_out if isinstance(r, dict)]
    # NEVER silently drop failures. A transient Modal control-plane error once
    # produced a clean-looking "0 cells" result because the exceptions were
    # filtered out here without being counted.
    errs = [r for r in raw_out if not isinstance(r, dict)]
    if errs:
        from collections import Counter
        print(f"  !! {len(errs)}/{len(tasks)} cells FAILED", flush=True)
        for msg, k in Counter(f"{type(e).__name__}: {e}"[:180] for e in errs).most_common(5):
            print(f"     x{k}  {msg}", flush=True)
    if not out:
        raise RuntimeError(
            f"all {len(tasks)} cells failed; refusing to write an empty sweep")
    print(f"{'target':8s}{'delta':>6}{'seedDS':>8}{'bestDS':>8}{'gain':>7}"
          f"{'#feas':>7}{'#dock':>7}{'fiber':>8}{'props':>8}{'dock':>8}{'total':>8}",
          flush=True)
    for r in out:
        sd = -r.get("published_ds", 0.0) if "published_ds" in r else None
        bd = r["best_ds"]
        g = (sd - bd) if (sd is not None and bd is not None) else None
        print(f"  {r['target']:8s}{r['delta']:>6.1f}"
              f"{(f'{sd:.1f}' if sd else '-'):>8}{(f'{bd:.2f}' if bd else 'NONE'):>8}"
              f"{(f'{g:+.2f}' if g else '-'):>7}{r['n_feasible']:>7}{r['n_docked']:>7}"
              f"{r['timing']['fiber']:>8.0f}{r['timing']['props']:>8.0f}"
              f"{r['timing']['dock']:>8.0f}{r['seconds']:>8.0f}", flush=True)
    tot = sum(r["seconds"] for r in out)
    nd = sum(r["n_docked"] for r in out)
    print(f"\n  wall {tot:.0f} core-s over {len(out)} cells, {nd} dockings")
    if nd:
        print(f"  per docking: {sum(r['timing']['dock'] for r in out)/nd:.1f}s"
              f"   fiber+props share: "
              f"{sum(r['timing']['fiber']+r['timing']['props'] for r in out)/tot*100:.0f}%")
        full = tot / len(out) / (rounds * per_round) * 1000
        print(f"  EXTRAPOLATION: {full:.0f} core-s per cell at the official "
              f"1000-docking budget -> {full*30/3600:.1f} core-h for all 30 cells")
    return {"cells": out, "rule": rule, "rounds": rounds, "per_round": per_round,
            "parents": parents}


@app.local_entrypoint()
def main(which: str = "dev", rounds: int = 3, per_round: int = 8,
         parents: int = 3, rule: str = "tilt", limit: int = 2,
         deltas: str = "0.6", out: str = "diagnostics/genmol_t4_smoke.json") -> None:
    o = drive.remote(which, rounds, per_round, parents, rule, limit, deltas)
    p = Path(__file__).resolve().parents[1] / out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
