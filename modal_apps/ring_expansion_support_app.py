"""Is winner-like ring EXPANSION already reachable by a short primitive path?

Measures support and path difficulty BEFORE anyone proposes a macro. Motivated
by an observation, not by a target: 4 of 5 IVG parp1_s0_d0.4 winners carry
`8s:C6N2`, which is the seed's own `7s:C5N2` grown by one atom. We never do this
-- we only ADD rings. Whether that is an operator gap or a search gap is exactly
what this measures.

Success = some ring's size goes n -> n+1 while the ring COUNT is unchanged, i.e.
an existing ring was expanded rather than a new ring added.

Reports, per seed: the executor family histogram (so the search strategy is
checked, not assumed), whether a path exists, its length, each step's R_theta
rank and probability, and the path's total reference log-probability.

CPU only. No docking. Diagnostic only -- builds nothing.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
app = modal.App("ring-expansion-support")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
MEM_MIB = int(3 * 1024)

_RT: dict[str, Any] = {}


def _runtime():
    if "model" in _RT:
        return _RT
    import sys
    import torch
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source)
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state)
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME)
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system)
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=Path(REMOTE_ROOT))
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src, materialized_state=bundle)
    model = runtime.model
    ck = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["selected_model_state"], strict=True)
    model.eval(); torch.set_grad_enabled(False); torch.set_num_threads(1)
    _RT.update({"model": model, "system": _default_rewrite_system(model)})
    return _RT


def _ring_sizes(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return sorted(len(r) for r in m.GetRingInfo().AtomRings())


def _is_expansion(seed_smi, cand_smi):
    """One ring grew by exactly 1; ring COUNT unchanged (not a new ring)."""
    a, b = _ring_sizes(seed_smi), _ring_sizes(cand_smi)
    if a is None or b is None or len(a) != len(b) or not a:
        return False
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    return len(diff) == 1 and diff[0][1] == diff[0][0] + 1


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=2 * 60 * 60,
              retries=0, volumes={ARTIFACT_ROOT: artifact_volume})
def expansion_audit(job: dict) -> dict:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    import numpy as np
    from collections import Counter
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.gates.med_chem_gate import is_valid
    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed = job["seed_smiles"]; beam = int(job.get("beam", 10)); depth = int(job.get("depth", 3))
    t0 = time.time()

    def enum(st):
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        return ([m.executor_rule_name for m in law.marks],
                [m.action for m in law.marks],
                np.array([m.probability for m in law.marks], float))

    st0 = pad_molecular_graph(smiles_to_molecular_graph(seed), CANONICAL_SLOTS)
    fams0, _a0, p0 = enum(st0)
    hist = dict(Counter(fams0))

    # beam over RING-RELEVANT families only, ranked by R_theta
    # bond_reroute and cycle_open are the plausible expansion primitives and
    # must be explored on their OWN rank, not against 216 atom_inserts.
    RING_FAMS = ("cycle_open", "bond_reroute", "cycle_close", "atom_insert",
                 "bond_reorder", "ring_system_restate")
    PER_FAMILY = int(job.get("per_family", 6))
    found, explored = [], 0
    frontier = [(st0, [], 0.0)]
    for d in range(depth):
        nxt = []
        for st, path, lp in frontier:
            fams, acts, probs = enum(st)
            order = sorted(range(len(fams)), key=lambda j: -float(probs[j]))
            per = {f: 0 for f in RING_FAMS}
            for j in order:
                f = fams[j]
                if f not in per:
                    continue
                if per[f] >= PER_FAMILY:
                    continue
                per[f] += 1
                try:
                    y = system.apply(st, fams[j], acts[j])
                except Exception:
                    continue
                smi = canonical_state_key(y)
                if not smi or not is_valid(smi):
                    continue
                explored += 1
                step = {"family": fams[j], "rank": int(order.index(j)),
                        "p": float(probs[j])}
                npath = path + [step]
                nlp = lp + math.log(max(float(probs[j]), 1e-300))
                if _is_expansion(seed, smi):
                    found.append({"depth": d + 1, "smiles": smi, "path": npath,
                                  "logp": nlp,
                                  "ring_sizes": _ring_sizes(smi)})
                nxt.append((y, npath, nlp))
        by_first: dict = {}
        for t in nxt:
            k = t[1][0]["family"] if t[1] else "-"
            by_first.setdefault(k, []).append(t)
        frontier = []
        for k, group in by_first.items():
            frontier.extend(sorted(group, key=lambda t: -t[2])[:max(2, beam // 3)])
        frontier = frontier[: beam * 3]
        if found:
            break
    best = max(found, key=lambda f: f["logp"]) if found else None
    return {"cell": job["cell"], "seed": seed, "seed_ring_sizes": _ring_sizes(seed),
            "family_histogram": hist, "n_expansions_found": len(found),
            "states_explored": explored, "best": best, "sec": time.time() - t0}


@app.local_entrypoint()
def main(n_seeds: int = 15, beam: int = 10, depth: int = 3):
    seeds = json.loads(Path("docs/GENMOL_T4_SEEDS.json").read_text())[:n_seeds]
    jobs = [{"cell": f"{s['target']}_s{s['idx']}", "seed_smiles": s["smiles"],
             "beam": beam, "depth": depth, "per_family": 6} for s in seeds]
    res = list(expansion_audit.map(jobs))
    Path("diagnostics").mkdir(exist_ok=True)
    p = Path("diagnostics/ring_expansion_support.json")
    p.write_text(json.dumps({"beam": beam, "depth": depth, "results": res}, indent=2))
    n = sum(1 for r in res if r["n_expansions_found"])
    print(f"cells={len(res)} with_expansion={n} wrote={p}")
    print("family histogram (first cell):", json.dumps(res[0]["family_histogram"]))
    for r in res:
        b = r["best"]
        lp = f"{b['logp']:.1f}" if b else "-"
        dp = b["depth"] if b else "-"
        print(f"  {r['cell']:12s} rings={r['seed_ring_sizes']} "
              f"found={r['n_expansions_found']:>3d} depth={dp} logp={lp}")
