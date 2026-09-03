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


def _dock(smiles: str, target: str, tag: str, cpu: int = 4) -> float | None:
    """`cpu` is QuickVina's own thread count. It defaults to 4, which is what
    Gate 0 used to establish docking parity, so the default path is unchanged.
    Batch docking passes cpu=1 because each concurrent docking gets one core."""

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
             "--cpu", str(int(cpu)), "--num_modes", "10", "--exhaustiveness", "1"],
            capture_output=True, timeout=300, check=True)
        for line in open(out):
            if line.startswith("REMARK VINA RESULT"):
                return float(line.split()[3])
    except Exception:
        return None
    return None


def _dock_one(args) -> tuple[int, float | None]:
    """(index, smiles, target, tag, cpu) -> (index, score). Index preserves order."""
    i, smiles, target, tag, cpu = args
    return i, _dock(smiles, target, tag, cpu=cpu)


def _dock_many(smiles_list, target: str, tag_prefix: str, workers: int = 8,
               cpu_per_dock: int = 1) -> list:
    """Dock a round's molecules concurrently. Same molecules, same scores, same count.

    The 20 dockings inside a round are conditionally independent -- the archive is
    only updated after all of them -- so running them serially wastes most of the
    round. QuickVina is an external process, so threads are the right tool: the
    GIL is released across subprocess.run.

    ORDER AND ACCOUNTING ARE PRESERVED EXACTLY. Results come back indexed and are
    reassembled in the caller's order, every input yields exactly one output
    (None included), and the caller still charges one budget unit per molecule.
    Each docking gets its OWN scratch directory; the serial code reused
    /tmp/{tag}, which concurrent dockings would clobber.

    This is execution parallelism only. No candidate selection, archive rule, or
    budget semantics changes.
    """

    from concurrent.futures import ThreadPoolExecutor
    jobs = [(i, s, target, f"{tag_prefix}_w{i}", cpu_per_dock)
            for i, s in enumerate(smiles_list)]
    out: list = [None] * len(jobs)
    if not jobs:
        return out
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, r in ex.map(_dock_one, jobs):
            out[i] = r
    return out


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


# ---------------------------------------------------------------------------
# T4 with the local-to-global population search.
#
# Same task, same budget accounting, same feasibility contract as `optimize`
# above: QED >= 0.6, SA <= 4, sim >= delta on the RETURNED molecule, scored by
# the docking of the best feasible lead, with the seed's own score not charged.
#
# What changes is only the proposal. `optimize` samples the marked law and
# selects among realizations; this runs the qualified stack --
#
#     mu_exec(M|x)          which region to rewrite, feasibility/cost-aware
#     [ x exp(V_z/tau) ]    optional task tilt, the arm under test
#     q_rewrite(w|x,M)      frozen committor + KL trust region at kappa=1
#
# -- so it can choose the SCALE of each edit rather than only its content. The
# standing T4 conclusion located the failure exactly there: broad-C sampled the
# useful semantic 1.0% of the time against narrow-B's 33.3%, a proposal-
# probability bottleneck rather than a realization or ranking defect.
#
# Nothing below the region selector is retuned for T4.
# ---------------------------------------------------------------------------

POP_OUT = "/artifacts/t4_population"


@app.function(image=image, cpu=(4.0, 4.0), memory=int(6 * 1024),
              timeout=6 * 60 * 60, retries=2, max_containers=80,
              volumes={ARTIFACT_ROOT: artifact_volume})
def t4_population_cell(task: dict[str, Any]) -> dict[str, Any]:
    import os
    import sys

    import numpy as np
    import torch
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
    from compose_v4.control.region import enumerate_regions
    from compose_v4.control import region_rewrite as RR
    from compose_v4.control.region_selector import sample_region
    from compose_v4.control.task_value import TaskValue
    from compose_v4.gates.med_chem_gate import is_valid

    rt = _runtime(); model, system = rt["model"], rt["system"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed, delta, target = task["smiles"], task["delta"], task["target"]
    budget = int(task["budget"])                 # docking calls, charged exactly
    arm = task.get("arm", "mu_exec")
    tau = float(task.get("tau", 0.05))
    kappa = float(task.get("kappa", 1.0))
    eps_region = float(task.get("epsilon_region", 0.2))
    rng = np.random.default_rng(int(task["seed_rng"]))
    seed_fp = gen.GetFingerprint(Chem.MolFromSmiles(seed))
    tv = TaskValue.from_dict(task["value_table"]) if task.get("value_table") else None
    value_fn = tv.value_fn() if tv is not None else None

    artifact_volume.reload()
    ck = torch.load(f"/artifacts/region_committor/committor_bellman_v1.pt",
                    map_location="cpu")
    net = torch.nn.Sequential(torch.nn.Linear(int(ck["n_features"]), 48),
                              torch.nn.ReLU(), torch.nn.Linear(48, 1))
    net.load_state_dict(ck["state_dict"]); net.eval()
    torch.set_grad_enabled(False)

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
    p0 = props(seed)
    archive = [{"smiles": seed, **p0, "ds": None}]
    docked: dict[str, float] = {}
    n_dock, events = 0, []

    def rewrite(parent):
        """One region proposal on `parent`; returns a SMILES or None."""
        try:
            regions = [r for r in enumerate_regions(parent) if 1 <= r.size <= 24]
        except Exception:
            return None, None
        if not regions:
            return None, None
        reg, rs = sample_region(regions, rng, epsilon=eps_region,
                               value_fn=value_fn, tau=tau)
        if reg is None:
            return None, None
        raw = smiles_to_molecular_graph(parent)
        st0 = pad_molecular_graph(raw, CANONICAL_SLOTS)
        n_real = len(raw.atom_types)
        ctx = RR.context_from_region(reg)
        lin0 = RR.Lineage.initial(range(n_real))
        old_ids = frozenset(lin0.id_of[s] for s in reg.atoms if s in lin0.id_of)
        cache: dict = {}
        memo: dict = {}

        def slot_key(state):
            return (np.asarray(state.atom_types).tobytes(),
                    np.asarray(state.bonds).tobytes())

        def enum_fn(st):
            k = slot_key(st)
            if k not in cache:
                law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
                cache[k] = ([m.executor_rule_name for m in law.marks],
                            [m.action for m in law.marks],
                            np.array([m.probability for m in law.marks], float))
            return cache[k]

        def apply_fn(st, j):
            k = (slot_key(st), int(j))
            if k in memo:
                return memo[k]
            fams, acts, _ = enum_fn(st)
            try:
                y = system.apply(st, fams[j], acts[j])
            except Exception:
                y = None
            memo[k] = y
            return y

        hc: dict = {}

        def h_model(y, c, lin, b_, kind, _o=old_ids, _h=hc):
            key = (slot_key(y), int(b_), kind)
            if key not in _h:
                f = RR.features_from_shared(
                    RR.structural_features_shared(y, c, lin, _o), int(b_), kind)
                _h[key] = float(torch.sigmoid(
                    net(torch.tensor([f], dtype=torch.float32))).item())
            return _h[key]

        sched = ([("prune_old", max(1, reg.size)), ("grow_new", 2)]
                 if reg.interface == "pendant"
                 else [("grow_new", "until_handoff"),
                       ("prune_old", max(1, reg.size))])
        p = RR.propose(enum_fn, apply_fn, ctx, st0, schedule=sched,
                       rng=np.random.default_rng(int(rng.integers(0, 10 ** 6))),
                       lineage=lin0, original_region_ids=old_ids,
                       max_handoff_steps=int(task.get("max_handoff", 16)),
                       h_model=h_model, tilt="kl", kappa=kappa,
                       epsilon=float(task.get("epsilon", 0.1)))
        if p.status != "OK" or p.endpoint is None:
            return None, reg
        key = canonical_state_key(p.endpoint)
        if not key or not is_valid(key):
            return None, reg
        return key, reg

    while n_dock < budget:
        # parent: prefer feasible with good docking, else lowest constraint
        # violation -- the same rule `optimize` uses, so the arms differ only in
        # how a child is PROPOSED.
        scored = sorted(archive, key=lambda a: (
            0 if (a["v"] <= 0 and a["ds"] is not None) else 1,
            a["ds"] if a["ds"] is not None else 0.0, a["v"]))
        parent = scored[0]["smiles"] if rng.random() < 0.7 else \
            archive[int(rng.integers(0, len(archive)))]["smiles"]
        child, reg = rewrite(parent)
        ev = {"n_dock": n_dock, "parent": parent,
              "interface": reg.interface if reg is not None else None,
              "kind": reg.kind if reg is not None else None,
              "r_release": reg.released_fraction if reg is not None else None,
              "ok": child is not None}
        if child is None or child in docked:
            events.append(ev)
            continue
        pr = props(child)
        if pr is None:
            events.append(ev)
            continue
        # every distinct molecule sent to the oracle costs one budget unit,
        # feasible or not -- the same accounting `optimize` uses
        ds = _dock(child, target, f"{task['cell']}_{n_dock}")
        n_dock += 1
        docked[child] = ds if ds is not None else 0.0
        archive.append({"smiles": child, **pr, "ds": ds})
        ev.update({"qed": pr["qed"], "sa": pr["sa"], "sim": pr["sim"],
                   "v": pr["v"], "ds": ds, "charged": True})
        events.append(ev)
        # Persist after every CHARGED call. Writing only at the end would put a
        # multi-hour cell one interruption away from losing everything, and
        # would make progress unobservable while it runs.
        _f = [a for a in archive if a["v"] <= 0 and a["ds"] is not None
              and a["smiles"] != seed]
        _b = min(_f, key=lambda a: a["ds"]) if _f else None
        _d = Path(POP_OUT); _d.mkdir(parents=True, exist_ok=True)
        (_d / f"{task['cell']}.json").write_text(json.dumps({
            "cell": task["cell"], "arm": arm, "target": target, "delta": delta,
            "seed": seed, "seed_props": p0, "n_dock": n_dock,
            "n_feasible": len(_f), "best_ds": _b["ds"] if _b else None,
            "best_smiles": _b["smiles"] if _b else None, "events": events,
            "complete": False, "sec": time.perf_counter() - t0}))
        artifact_volume.commit()

    feas = [a for a in archive if a["v"] <= 0 and a["ds"] is not None
            and a["smiles"] != seed]
    best = min(feas, key=lambda a: a["ds"]) if feas else None
    res = {"cell": task["cell"], "arm": arm, "target": target, "delta": delta,
           "seed": seed, "seed_props": p0, "n_dock": n_dock,
           "n_feasible": len(feas),
           "best_ds": best["ds"] if best else None,
           "best_smiles": best["smiles"] if best else None,
           "events": events, "complete": True,
           "sec": time.perf_counter() - t0}
    d = Path(POP_OUT)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{task['cell']}.json").write_text(json.dumps(res))
    artifact_volume.commit()
    print(f"[t4pop] {task['cell']} arm={arm} dock={n_dock}/{budget} "
          f"feasible={len(feas)} best={best['ds'] if best else None} "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    return res


@app.local_entrypoint()
def t4_population(budget: int = 500, arms: str = "mu_exec,Q_taskvalue",
                  deltas: str = "0.4,0.6", n_seeds: int = 15,
                  tau: float = 0.05, dry_run: bool = True):
    """T4 with the local-to-global population search. DRY RUN BY DEFAULT.

    `dry_run=True` prints the full plan -- cells, arms, budget, total docking
    calls -- and launches nothing. Pass `--no-dry-run` to actually spend the
    oracle budget. The default is deliberate: this is the benchmark, and it
    should not be startable by a stray command.
    """
    import sys as _sys
    _sys.path.insert(0, "tools")
    from preflight import assert_synced
    assert_synced(strict=False)

    seeds = json.loads(Path("docs/GENMOL_T4_SEEDS.json").read_text())
    seeds = seeds[:n_seeds]
    table = None
    p = Path("diagnostics/task_value_qed.json")
    if p.exists():
        table = json.loads(p.read_text())
    arm_list = [a.strip() for a in arms.split(",")]
    d_list = [float(d) for d in deltas.split(",")]

    jobs = []
    for i, s in enumerate(seeds):
        for d in d_list:
            for a in arm_list:
                t = {"cell": f"{a}_{s['target']}_{i}_d{d}", "smiles": s["smiles"],
                     "target": s["target"], "delta": d, "budget": budget,
                     "arm": a, "tau": tau, "seed_rng": 1000 + i,
                     "kappa": 1.0, "epsilon_region": 0.2}
                if a == "Q_taskvalue":
                    if table is None:
                        raise SystemExit("no task_value table; run the QED "
                                         "collect first or drop that arm")
                    t["value_table"] = table
                jobs.append(t)

    total = len(jobs) * budget
    print(f"\nT4 POPULATION SEARCH -- {'PLAN ONLY (dry run)' if dry_run else 'LAUNCHING'}")
    print(f"  seeds={len(seeds)}  deltas={d_list}  arms={arm_list}")
    print(f"  cells={len(jobs)}  budget={budget} docking calls/cell")
    print(f"  TOTAL DOCKING CALLS = {total:,}")
    print(f"  published protocol is 1,000/cell; this run uses {budget}")
    print(f"  frozen: R_theta, committor, kappa=1.0, mu_exec, "
          f"tau={tau}, epsilon_region=0.2")
    print(f"  feasibility: QED>={QED_MIN}, SA<={SA_MAX}, sim>=delta; "
          f"seed's own docking not charged")
    import statistics as st
    cmp_ = json.loads(Path("docs/genmol_t4_all_methods.json").read_text())["rows"]
    print("\n  comparators (read from docs/genmol_t4_all_methods.json):")
    for m, c4, c6 in (("GenMol", "genmol_d04", "genmol_d06"),
                      ("RetMol", "retmol_d04", "retmol_d06"),
                      ("GraphGA", "graphga_d04", "graphga_d06")):
        n = sum(1 for r in cmp_ for c in (c4, c6) if r.get(c) is not None)
        v4 = [r[c4] for r in cmp_ if r.get(c4) is not None]
        v6 = [r[c6] for r in cmp_ if r.get(c6) is not None]
        print(f"    {m:<8} solved {n}/30   mean d0.4 {st.mean(v4):.2f}   "
              f"mean d0.6 {st.mean(v6):.2f}")
    if dry_run:
        print("\n  DRY RUN -- nothing launched. Re-run with --no-dry-run to spend "
              "the oracle budget.")
        return
    out = [o for o in t4_population_cell.map(jobs) if o]
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/t4_population.json").write_text(json.dumps(out, default=str))
    t4_population_report(out)


def t4_population_report(out):
    import statistics as st
    print(f"\ncells={len(out)}")
    for arm in sorted({o["arm"] for o in out}):
        v = [o for o in out if o["arm"] == arm]
        for d in sorted({o["delta"] for o in v}):
            w = [o for o in v if o["delta"] == d]
            solved = [o for o in w if o["best_ds"] is not None]
            print(f"  {arm:<12} delta={d}  solved {len(solved)}/{len(w)}  "
                  f"mean best {st.mean([o['best_ds'] for o in solved]) if solved else float('nan'):.2f}")
    ev = [e for o in out for e in o["events"]]
    print(f"\nproposals={len(ev)} accepted={sum(1 for e in ev if e['ok'])}")
    print("scale usage (does it use different scales, question 3):")
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        b = [e for e in ev if e.get("r_release") is not None
             and lo <= e["r_release"] < hi]
        if b:
            print(f"  {lo}-{hi}: proposed {len(b)} accepted "
                  f"{sum(1 for e in b if e['ok'])}")


@app.local_entrypoint()
def t4_population_report_only():
    out = json.loads(Path("diagnostics/t4_population.json").read_text())
    t4_population_report(out)
