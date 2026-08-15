"""Where do the ~5 seconds per step actually go? Measure, do not theorise.

Every speed claim tonight has been arithmetic from guessed millisecond costs,
and it has been wrong by 3x and by 40x in both directions. This times each phase
of a real step on a real source with the real frozen model, and separately tests
whether pinning torch to one thread is what makes it slow.

Answers exactly two questions and then exits:
  1. what fraction of a step is law enumeration vs apply vs encode vs props?
  2. does giving torch more threads help, and by how much?
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("hphi-profile")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
REGION = (0.90, 0.40)
STEPS = 6          # enough to see the per-step cost; not a trajectory


def _load(threads: int):
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME,
    )
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system,
    )

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
    model.eval(); torch.set_grad_enabled(False)
    torch.set_num_threads(threads)
    return model, _default_rewrite_system(model)


@app.function(image=image, cpu=(8.0, 8.0), memory=16384, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def profile(source: str) -> dict[str, Any]:
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import QED, rdFingerprintGenerator

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        _one_state_batch, canonical_state_key, enumerate_factorized_marked_law,
    )

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    src_fp = gen.GetFingerprint(Chem.MolFromSmiles(source))
    out: dict[str, Any] = {"source": source, "by_threads": {}}

    for threads in (1, 4, 8):
        model, system = _load(threads)
        T = {"prepare": 0.0, "law": 0.0, "apply": 0.0, "encode": 0.0,
             "props": 0.0}
        n_marks: list[int] = []
        cur = source
        rng = np.random.default_rng(0)
        t_all = time.perf_counter()
        for _ in range(STEPS):
            t = time.perf_counter()
            st = pad_molecular_graph(smiles_to_molecular_graph(cur),
                                     CANONICAL_SLOTS)
            T["prepare"] += time.perf_counter() - t

            t = time.perf_counter()
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            T["law"] += time.perf_counter() - t
            if not law.marks:
                break
            n_marks.append(len(law.marks))

            p = np.array([m.probability for m in law.marks], float); p /= p.sum()
            idx = rng.choice(len(p), size=16, p=p)

            t = time.perf_counter()
            ys = [canonical_state_key(system.apply(
                st, law.marks[int(i)].executor_rule_name,
                law.marks[int(i)].action)) for i in idx]
            T["apply"] += time.perf_counter() - t

            t = time.perf_counter()
            for y in ys[:4]:
                m = Chem.MolFromSmiles(y)
                if m:
                    QED.qed(m)
                    DataStructs.TanimotoSimilarity(src_fp, gen.GetFingerprint(m))
            T["props"] += time.perf_counter() - t

            t = time.perf_counter()
            for y in ys[:4]:
                s2 = pad_molecular_graph(smiles_to_molecular_graph(y),
                                         CANONICAL_SLOTS)
                b = _one_state_batch(model, s2, float(TIME_POINT),
                                     prepared_batch=None)
                with torch.no_grad():
                    model._encode_batch(b)
            T["encode"] += time.perf_counter() - t

            nxt = next((y for y in ys if y != canonical_state_key(st)), None)
            if nxt is None:
                break
            cur = nxt
        total = time.perf_counter() - t_all
        out["by_threads"][threads] = {
            "total_s": round(total, 2),
            "per_step_s": round(total / max(1, STEPS), 2),
            "median_marks": int(np.median(n_marks)) if n_marks else 0,
            "phases_s": {k: round(v, 2) for k, v in T.items()},
            "phase_pct": {k: (round(100 * v / total, 1) if total else 0)
                          for k, v in T.items()},
        }
        print(f"threads={threads}: {total:.1f}s total, "
              f"{total/STEPS:.2f}s/step, phases={out['by_threads'][threads]['phase_pct']}",
              flush=True)
    return out


@app.function(image=image, cpu=(8.0, 8.0), memory=32768, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def batch_test(smiles: list[str]) -> dict[str, Any]:
    """Does batching the encode help, and is it BITWISE identical?

    The whole h_phi cost is single-molecule encodes that each run 558 lines of
    mark-space preparation _encode_batch never reads. prepare_factorized_mark_batch
    already accepts tuples, so the question is purely empirical: how much does
    batching amortise, and does it change a single number?
    """
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.factorized_mark_conditional import (
        operator_capability_batch_kwargs,
    )
    from compose_v4.model.factorized_tracelet_rate_model import (
        prepare_factorized_mark_batch,
    )
    from compose_v4.experiments.production_successor_kernel import _one_state_batch

    model, _system = _load(8)
    states = [pad_molecular_graph(smiles_to_molecular_graph(s), CANONICAL_SLOTS)
              for s in smiles]
    out: dict[str, Any] = {"n": len(states)}

    # --- current path: one molecule at a time ---
    t0 = time.perf_counter()
    single = []
    for st in states:
        b = _one_state_batch(model, st, float(TIME_POINT), prepared_batch=None)
        with torch.no_grad():
            _n, g, _p = model._encode_batch(b)
        single.append(g[0].detach().cpu().numpy())
    t_single = time.perf_counter() - t0

    # --- batched path: all molecules in ONE preparation ---
    res = {}
    for bs in (8, 32, len(states)):
        if bs > len(states):
            continue
        t0 = time.perf_counter()
        batched = []
        for i in range(0, len(states), bs):
            chunk = states[i:i + bs]
            pb = prepare_factorized_mark_batch(
                tuple(chunk), (float(TIME_POINT),) * len(chunk),
                (None,) * len(chunk), (None,) * len(chunk),
                (0.0,) * len(chunk), use_aromatic_bond_view=True,
                ring_catalog=model.ring_catalog,
                **operator_capability_batch_kwargs(model.operator_capabilities))
            with torch.no_grad():
                _n, g, _p = model._encode_batch(pb)
            batched.extend(g[k].detach().cpu().numpy() for k in range(len(chunk)))
        dt = time.perf_counter() - t0
        maxdiff = float(max(np.abs(a - b).max() for a, b in zip(single, batched)))
        res[bs] = {"seconds": round(dt, 2),
                   "per_mol_ms": round(1000 * dt / len(states), 1),
                   "speedup_vs_single": round(t_single / dt, 2),
                   "max_abs_diff_vs_single": maxdiff,
                   "bitwise_identical": maxdiff == 0.0}
        print(f"  batch={bs:>3}: {dt:6.2f}s  {1000*dt/len(states):7.1f} ms/mol  "
              f"speedup {t_single/dt:5.2f}x  maxdiff {maxdiff:.3e}", flush=True)
    out["single"] = {"seconds": round(t_single, 2),
                     "per_mol_ms": round(1000 * t_single / len(states), 1)}
    out["batched"] = res
    print(f"  single : {t_single:6.2f}s  {1000*t_single/len(states):7.1f} ms/mol",
          flush=True)
    return out


_W: dict[str, Any] = {}


def _apply_chunk(args):
    """Worker: apply a slice of marks. Returns (index, canonical_key) pairs."""
    lo, hi = args
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key,
    )
    st, law, system = _W["st"], _W["law"], _W["system"]
    out = []
    for i in range(lo, hi):
        mk = law.marks[i]
        out.append((i, canonical_state_key(
            system.apply(st, mk.executor_rule_name, mk.action))))
    return out


def _encode_chunk(smis):
    """Worker: encode a slice of molecules. Returns (smiles, embedding) pairs."""
    import torch

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import _one_state_batch

    torch.set_num_threads(1)
    model = _W["model"]
    out = []
    for s in smis:
        stt = pad_molecular_graph(smiles_to_molecular_graph(s), CANONICAL_SLOTS)
        b = _one_state_batch(model, stt, float(TIME_POINT), prepared_batch=None)
        with torch.no_grad():
            _n, g, _p = model._encode_batch(b)
        out.append((s, g[0].detach().cpu().numpy()))
    return out


@app.function(image=image, cpu=(8.0, 8.0), memory=32768, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def parallel_test(source: str, workers: int = 8) -> dict[str, Any]:
    """Do the 606 applies and the encodes parallelise across PROCESSES?

    Threads failed because the cost is not BLAS. Batching failed because the
    cost is per-molecule Python. Per-molecule Python is exactly what forks
    parallelise -- the GIL is the reason threads could not, and separate
    processes do not share one. Model is loaded ONCE and inherited
    copy-on-write, so N workers do not mean N model loads.
    """
    import multiprocessing as mp
    import sys

    import numpy as np

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law,
    )

    model, system = _load(1)
    st = pad_molecular_graph(smiles_to_molecular_graph(source), CANONICAL_SLOTS)
    law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
    n = len(law.marks)
    _W.update({"st": st, "law": law, "system": system, "model": model})
    out: dict[str, Any] = {"n_marks": n, "workers": workers}

    # ---------- APPLIES: 606 of them, one step's worth for policy_b ----------
    t0 = time.perf_counter()
    serial = [canonical_state_key(system.apply(
        st, law.marks[i].executor_rule_name, law.marks[i].action))
        for i in range(n)]
    t_ser = time.perf_counter() - t0

    edges = [(i * n // workers, (i + 1) * n // workers) for i in range(workers)]
    ctx = mp.get_context("fork")          # fork => model shared copy-on-write
    t0 = time.perf_counter()
    with ctx.Pool(workers) as pool:
        par_pairs = [pr for chunk in pool.map(_apply_chunk, edges) for pr in chunk]
    t_par = time.perf_counter() - t0
    par = [k for _i, k in sorted(par_pairs)]
    out["applies"] = {
        "n": n, "serial_s": round(t_ser, 2), "parallel_s": round(t_par, 2),
        "speedup": round(t_ser / t_par, 2) if t_par else None,
        "identical": serial == par,
    }
    print(f"  applies  n={n}: serial {t_ser:6.2f}s  parallel {t_par:6.2f}s  "
          f"speedup {t_ser/max(t_par,1e-9):5.2f}x  identical={serial == par}",
          flush=True)

    # ---------- ENCODES: 40, one capped h_phi step's worth ----------
    mols = sorted(set(serial))[:40]
    t0 = time.perf_counter()
    ser_e = dict(_encode_chunk(mols))
    t_ser_e = time.perf_counter() - t0
    slices = [mols[i::workers] for i in range(workers)]
    t0 = time.perf_counter()
    with ctx.Pool(workers) as pool:
        par_e = dict(pr for ch in pool.map(_encode_chunk, slices) for pr in ch)
    t_par_e = time.perf_counter() - t0
    maxdiff = float(max(np.abs(ser_e[k] - par_e[k]).max() for k in ser_e))
    out["encodes"] = {
        "n": len(mols), "serial_s": round(t_ser_e, 2),
        "parallel_s": round(t_par_e, 2),
        "speedup": round(t_ser_e / t_par_e, 2) if t_par_e else None,
        "max_abs_diff": maxdiff, "bitwise_identical": maxdiff == 0.0,
    }
    print(f"  encodes  n={len(mols)}: serial {t_ser_e:6.2f}s  "
          f"parallel {t_par_e:6.2f}s  speedup {t_ser_e/max(t_par_e,1e-9):5.2f}x  "
          f"maxdiff {maxdiff:.3e}", flush=True)
    return out


@app.local_entrypoint()
def main(batch: bool = False, parallel: bool = False) -> None:
    src = [s.strip() for s in
           (Path(__file__).resolve().parents[1] / "data/jin/dev_panel_qed_64.txt"
            ).read_text().split("\n") if s.strip()][0]
    print(f"profiling {STEPS} steps on dev source 0 at 1 / 4 / 8 torch threads")
    if parallel:
        print(json.dumps(parallel_test.remote(src), indent=2))
        return
    if batch:
        pool = [s.strip() for s in
                (Path(__file__).resolve().parents[1] / "data/jin/hphi_valid_128.txt"
                 ).read_text().split("\n") if s.strip()][:64]
        print(json.dumps(batch_test.remote(pool), indent=2))
        return
    r = profile.remote(src)
    print(json.dumps(r, indent=2))
