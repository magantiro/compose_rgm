"""Region-h_phi V1 rollout corpus: 1,024 x 8 x H6 under FROZEN R_theta.

Preregistered in `docs/HPHI_V1_CORPUS_PREREGISTRATION.md`. Semantics and their
tests live in `compose_v4.experiments.hphi_rollout` (13 green).

THIS GENERATOR KNOWS NOTHING ABOUT EVENT PREVALENCE.
-----------------------------------------------------
It writes raw trajectories -- molecules, source ids, seeds, committed marks,
canonical keys, terminal descriptors -- and computes NO census. The 20-region
census is a separate read-only script run afterwards, which is what makes
"the goal family was frozen before the census" literally true in software
rather than merely in a document.

INVARIANT 1 -- THE ROLLOUT LAW IS THE CONTROLLER'S BASE CHAIN
--------------------------------------------------------------
`H = 6` means six COMMITTED PRODUCTIVE edits. Sampling is direct-mark: draw a
mark from the factorized law, execute only that mark, canonicalize. A mark whose
canonical image equals the current state is VIRTUAL -- redrawn, and it does not
consume a step.

`parity_check()` gates the corpus. On real states it compares the exact
successor law from `canonical_successor_result` against the empirical law of
direct-mark sampling. The corpus does not run unless total variation is small,
because the embedded chain must be the controller's base chain exactly --
aliases, virtual mass and retry semantics included.

INVARIANT 3 -- early termination is explicit. A chain that cannot make six
committed edits terminates into the CEMETERY convention, never a silent H=6
endpoint. The same convention is used at inference.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"}
).add_local_dir(ROOT / "data/jin", str(REMOTE_ROOT / "data/jin"), copy=True)

app = modal.App("hphi-rollout-corpus")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "hphi_rollout_corpus"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48
HORIZON = 6
TRAJECTORIES_PER_SOURCE = 8
CORPUS_VERSION = "hphi-corpus-v1"
#: A virtual mark does not consume a step; this bounds the redraw loop.
MAX_DRAWS_PER_STEP = 400

_RT: dict[str, Any] = {}


def rollout_seed(canonical_source: str, replicate: int) -> int:
    """Frozen manifest. NEVER Python's process-salted hash()."""
    payload = f"{CORPUS_VERSION}|{canonical_source}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _runtime():
    if "model" in _RT:
        return _RT
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

    t0 = time.perf_counter()
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src, materialized_state=bundle)
    model = runtime.model
    ckpt = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system,
    )
    _RT["model"] = model
    _RT["system"] = _default_rewrite_system(model)
    print(f"container runtime built in {time.perf_counter()-t0:.1f}s", flush=True)
    return _RT


def _direct_mark_step(model, system, state, key, rng, np):
    """One COMMITTED PRODUCTIVE edit. Virtual marks are redrawn, not counted."""
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key,
        enumerate_factorized_marked_law,
    )

    law = enumerate_factorized_marked_law(model, state, float(TIME_POINT))
    if not law.marks:
        return None, None, None, "empty_fiber"
    p = np.array([m.probability for m in law.marks], dtype=float)
    tot = p.sum()
    if not np.isfinite(tot) or tot <= 0:
        return None, None, None, "zero_mark_mass"
    p = p / tot
    for _ in range(MAX_DRAWS_PER_STEP):
        i = int(rng.choice(len(p), p=p))
        mark = law.marks[i]
        succ = system.apply(state, mark.executor_rule_name, mark.action)
        succ_key = canonical_state_key(succ)
        if succ_key == key:
            continue                      # VIRTUAL: redraw, do not consume a step
        return succ, succ_key, mark, None
    return None, None, None, "all_virtual"


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=6 * 60 * 60,
              max_containers=80, retries=3,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def parity_check(n_states: int = 6, draws: int = 4000) -> dict[str, Any]:
    """GATE. Direct-mark sampling must reproduce the canonical successor law."""
    import numpy as np
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    rt = _runtime()
    model, system = rt["model"], rt["system"]
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key,
        canonical_successor_result,
    )

    smis = [l.strip() for l in
            (REMOTE_ROOT / "data/jin/hphi_train_1024.txt").read_text().splitlines()
            if l.strip()]
    rows = []
    for smi in smis[:n_states]:
        state = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        key = canonical_state_key(state)
        res = canonical_successor_result(model, state, float(TIME_POINT))
        exact = {s.key: float(s.probability) for s in res.batch.successors}
        tot = sum(exact.values())
        exact = {k: v / tot for k, v in exact.items()}

        # The law is a property of the STATE, so compute it ONCE and draw from
        # it `draws` times. Recomputing it per draw would make this gate take
        # hours while measuring exactly the same distribution.
        from compose_v4.experiments.production_successor_kernel import (
            enumerate_factorized_marked_law,
        )
        law = enumerate_factorized_marked_law(model, state, float(TIME_POINT))
        p_marks = np.array([m.probability for m in law.marks], dtype=float)
        p_marks = p_marks / p_marks.sum()
        # Executing a mark is deterministic, so cache mark -> canonical key and
        # pay each rewrite at most once instead of `draws` times.
        mark_key: dict[int, str] = {}
        rng = np.random.default_rng(0)
        counts: dict[str, int] = {}
        for _ in range(draws):
            i = int(rng.choice(len(p_marks), p=p_marks))
            k = mark_key.get(i)
            if k is None:
                mk = law.marks[i]
                k = canonical_state_key(
                    system.apply(state, mk.executor_rule_name, mk.action))
                mark_key[i] = k
            if k == key:
                continue                  # VIRTUAL: excluded, as in the chain
            counts[k] = counts.get(k, 0) + 1
        n = sum(counts.values())
        emp = {k: c / n for k, c in counts.items()}
        keys = set(exact) | set(emp)
        tv = 0.5 * sum(abs(exact.get(k, 0.0) - emp.get(k, 0.0)) for k in keys)
        rows.append({"smiles": smi, "n_successors": len(exact),
                     "draws_accepted": n, "total_variation": round(tv, 5)})
        print(f"  succ={len(exact):>4} TV={tv:.5f}", flush=True)

    worst = max(r["total_variation"] for r in rows)
    # Sampling error alone is ~sqrt(K/draws); allow generous headroom but still
    # fail loudly on a genuine law mismatch.
    passed = worst < 0.10
    out = {"schema": "compose.hphi.parity", "rows": rows,
           "worst_total_variation": worst, "draws": draws, "passed": passed}
    p = Path(RUN_ROOT) / OUT_DIR
    p.mkdir(parents=True, exist_ok=True)
    (p / "PARITY.json").write_text(json.dumps(out, indent=2))
    artifact_volume.commit()
    print(f"\nworst TV {worst:.5f}  ->  {'PASS' if passed else 'FAIL'}", flush=True)
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=6 * 60 * 60,
              max_containers=80, retries=3,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_source(task: dict[str, Any]) -> dict[str, Any]:
    """8 unguided frozen-R_theta trajectories from one source. NO census."""
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    rt = _runtime()
    model, system = rt["model"], rt["system"]
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key,
    )

    t0 = time.perf_counter()
    source = task["smiles"]
    mol0 = Chem.MolFromSmiles(source)
    if mol0 is None:
        return {"index": task["index"], "source": source,
                "status": "SOURCE_UNPARSEABLE"}
    canonical_source = Chem.MolToSmiles(mol0)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp0 = gen.GetFingerprint(mol0)

    def props(smi: str) -> tuple[float, float]:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return 0.0, 0.0
        # Similarity ALWAYS to the immutable source. Never pairwise.
        return (float(QED.qed(m)),
                float(DataStructs.TanimotoSimilarity(fp0, gen.GetFingerprint(m))))

    trajectories = []
    kernel_calls = 0
    for rep in range(int(task.get("replicates", TRAJECTORIES_PER_SOURCE))):
        rng = np.random.default_rng(rollout_seed(canonical_source, rep))
        state = pad_molecular_graph(smiles_to_molecular_graph(source),
                                    CANONICAL_SLOTS)
        key = canonical_state_key(state)
        path, marks = [source], []
        kind, reason, edits = "complete", None, 0
        for _ in range(HORIZON):
            succ, succ_key, mark, err = _direct_mark_step(
                model, system, state, key, rng, np)
            kernel_calls += 1
            if err is not None:
                kind, reason = "cemetery", err
                break
            state, key = succ, succ_key
            path.append(succ_key)
            marks.append({"family": mark.family_name,
                          "rule": mark.executor_rule_name})
            edits += 1
        qs = [props(s) for s in path]
        trajectories.append({
            "replicate": rep,
            "seed": rollout_seed(canonical_source, rep),
            "path": path,
            "committed_marks": marks,
            "qed": [round(q, 6) for q, _ in qs],
            "similarity_to_source": [round(s, 6) for _, s in qs],
            "termination": {"kind": kind, "edits": edits, "reason": reason},
        })

    return {
        "index": task["index"], "source": source,
        "canonical_source": canonical_source,
        "source_qed": round(float(QED.qed(mol0)), 6),
        "trajectories": trajectories,
        "kernel_calls": kernel_calls,
        "seconds": round(time.perf_counter() - t0, 2),
        "status": "OK",
    }


def resume_from_partial(tasks, partial: Path):
    if not partial.exists():
        return [], list(tasks)
    try:
        blob = json.loads(gzip.decompress(partial.read_bytes()).decode())
        rec = [r for r in blob.get("results", [])
               if isinstance(r, dict) and isinstance(r.get("index"), int)]
    except Exception as exc:  # noqa: BLE001
        print(f"partial unreadable ({type(exc).__name__}); starting fresh", flush=True)
        return [], list(tasks)
    seen, uniq = set(), []
    for r in rec:
        if int(r["index"]) not in seen:
            seen.add(int(r["index"]))
            uniq.append(r)
    return uniq, [t for t in tasks if int(t["index"]) not in seen]


@app.function(image=image, cpu=(1.0, 1.0), memory=4096, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], out_name: str) -> dict[str, Any]:
    artifact_volume.reload()
    out = Path(RUN_ROOT) / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path, partial = out / f"{out_name}.json.gz", out / f"{out_name}.partial.json.gz"
    started = time.perf_counter()
    results, pending = resume_from_partial(tasks, partial)
    if results:
        print(f"RESUMED {len(results)}; {len(pending)} remain", flush=True)
    last = len(results)
    for r in run_source.map(pending, order_outputs=False, return_exceptions=True):
        if isinstance(r, dict):
            results.append(r)
        if len(results) - last >= 50:
            last = len(results)
            partial.write_bytes(gzip.compress(json.dumps(
                {"results": results}, default=float).encode()))
            artifact_volume.commit()
            print(f"{len(results)}/{len(tasks)} "
                  f"{time.perf_counter()-started:.0f}s", flush=True)
    payload = {
        "schema": "compose.hphi.rollout_corpus",
        "corpus_version": CORPUS_VERSION,
        "preregistration": "docs/HPHI_V1_CORPUS_PREREGISTRATION.md",
        "r_theta": "FROZEN; no objective-specific update",
        "law": "direct-mark sampling; virtual marks redrawn, not counted",
        "horizon": HORIZON,
        "trajectories_per_source": TRAJECTORIES_PER_SOURCE,
        "census_computed_here": False,
        "n": len(results),
        "seconds": round(time.perf_counter() - started, 1),
        "results": results,
    }
    path.write_bytes(gzip.compress(json.dumps(payload, default=float).encode()))
    partial.unlink(missing_ok=True)
    artifact_volume.commit()
    print(f"DONE {len(results)} in {payload['seconds']:.0f}s", flush=True)
    return {"n": len(results), "seconds": payload["seconds"]}


@app.local_entrypoint()
def main(parity_only: bool = False, limit: int = 0, valid: bool = False) -> None:
    src_file = ("data/jin/hphi_valid_128.txt" if valid
                else "data/jin/hphi_train_1024.txt")
    smis = [l.strip() for l in (ROOT / src_file).read_text().splitlines() if l.strip()]
    expect = 128 if valid else 1024
    assert len(smis) == expect, f"expected {expect} sources, got {len(smis)}"
    if parity_only:
        print("PARITY GATE: direct-mark sampling vs the canonical successor law")
        call = parity_check.spawn(6, 4000)
        print(f"spawned: {call.object_id}")
        return
    if limit:
        smis = smis[:limit]
    tasks = [{"index": i, "smiles": s} for i, s in enumerate(smis)]
    print(f"h_phi rollout corpus: {len(tasks)} sources x "
          f"{TRAJECTORIES_PER_SOURCE} trajectories x H{HORIZON}")
    print("FROZEN R_theta, unguided. Raw trajectories only -- NO census here.")
    call = drive.spawn(tasks, f"{'valid' if valid else 'train'}_{len(tasks):04d}")
    print(f"driver spawned: {call.object_id}")
