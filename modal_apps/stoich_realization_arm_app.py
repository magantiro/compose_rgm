"""Causal test: expose a structural option's admissible chemical realizations.

Matched comparison at IDENTICAL budget. Held fixed across arms: R_theta, seeds,
topology/size/state of the structural option, primitive edit budget, anchor
sweep, and -- critically -- the NUMBER OF CANONICAL ENDPOINTS ADVANCED. Only the
realization-selection layer changes.

  A  narrow_frozen      composition='carbon_rich', stoich=None   (frozen control)
  B1 free_composition   allowed = DECLARED ring elements, no quota; R_theta picks
                        the element at each growth step. Composition emerges from
                        mass already present in the frozen reference process.
  B2 stoich_enumerated  admissible exact stoichiometries for the SAME fixed
                        (topology,size,state), enumerated generically from
                        RING_ELEMENTS, drawn UNIFORMLY per proposal. Uniform on
                        purpose: no benchmark-derived prior, so any gain is
                        attributable to exposing the space, not to a clever
                        ranking of it.

No IVG winner composition is referenced anywhere. No per-cell tuning. Every arm
makes exactly K proposals per seed and therefore yields exactly K endpoints.

Feasibility here is the T4 endpoint gate (QED, SA, similarity) computed on CPU.
DOCKING IS NOT RUN in this app -- the oracle comparison is priced separately.

CPU only. Development cells only. Nothing here is a benchmark number.
"""

from __future__ import annotations

import json
import time
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
app = modal.App("stoich-realization-arm")
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


def _ring_profile(smi):
    from rdkit import Chem
    from collections import Counter
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    out = []
    for r in m.GetRingInfo().AtomRings():
        c = Counter(m.GetAtomWithIdx(i).GetSymbol() for i in r)
        aro = all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)
        out.append(f"{len(r)}{'a' if aro else 's'}:" + ''.join(f"{k}{v}" for k, v in sorted(c.items())))
    return sorted(out)


def _added_rings(seed_smi, prod_smi):
    from collections import Counter
    a, b = _ring_profile(seed_smi), _ring_profile(prod_smi)
    if a is None or b is None:
        return []
    return list((Counter(b) - Counter(a)).elements())


def _endpoint_metrics(seed_smi, prod_smi):
    """T4 endpoint gates, CPU only. No docking."""
    import os
    import sys
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    m = Chem.MolFromSmiles(prod_smi) if prod_smi else None
    s = Chem.MolFromSmiles(seed_smi)
    if m is None or s is None:
        return None
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    added = _added_rings(seed_smi, prod_smi)
    het = [t for t in added if any(x in t.split(":")[1] for x in ("N", "O", "S", "P"))]
    return {"qed": float(QED.qed(m)), "sa": float(sascorer.calculateScore(m)),
            "sim": float(DataStructs.TanimotoSimilarity(gm.GetFingerprint(s),
                                                        gm.GetFingerprint(m))),
            "heavy": int(m.GetNumHeavyAtoms()),
            "added_rings": added, "added_hetero_rings": het,
            "n_added_hetero": len(het)}


def _admissible_stoichs(size, max_hetero=3):
    """Generic: every stoichiometry of `size` over the DECLARED ring elements.

    Identical construction to control.semantic_actions.enumerate_specs, but
    restricted to the ALREADY-CHOSEN structural option so topology/size/state
    are not varied -- this isolates realization from structural search.
    """
    from compose_v4.control.semantic_actions import RING_ELEMENTS, normalize_stoich
    het = [e for e in RING_ELEMENTS if e != "C"]
    out = [normalize_stoich({}, size)]
    for n in range(1, min(max_hetero, size - 1) + 1):
        for combo in combinations_with_replacement(het, n):
            c: dict = {}
            for e in combo:
                c[e] = c.get(e, 0) + 1
            out.append(normalize_stoich(c, size))
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq


def _build(model, system, seed_smi, size, topology, composition, stoich,
           anchor_rank, allowed_override=None):
    import numpy as np
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control import macro_engine as ME
    from compose_v4.gates.med_chem_gate import is_valid
    _c: dict = {}

    def to_smiles(st):
        return canonical_state_key(st)

    def enum_full(st):
        k = to_smiles(st)
        if k not in _c:
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            _c[k] = ([m.executor_rule_name for m in law.marks],
                     [m.action for m in law.marks],
                     np.array([m.probability for m in law.marks], float))
        return _c[k]

    def apply_fn(st, j):
        fams, acts, _ = enum_full(st)
        try:
            return system.apply(st, fams[j], acts[j])
        except Exception:
            return None

    st0 = pad_molecular_graph(smiles_to_molecular_graph(seed_smi), CANONICAL_SLOTS)
    saved = dict(ME.COMPOSITION_CODES)
    try:
        if allowed_override is not None:
            # B1: widen the ELEMENT SET for this call only, from the DECLARED
            # ring vocabulary. Restored in `finally`; the frozen table is not
            # edited on disk and no other arm sees the change.
            ME.COMPOSITION_CODES = dict(saved)
            ME.COMPOSITION_CODES["__free__"] = allowed_override
            composition = "__free__"
        return ME.build_ring_system_exact(
            enum_full, apply_fn, to_smiles, is_valid, st0,
            size=size, topology=topology, composition=composition,
            state="aromatic", stoich=stoich, anchor_rank=int(anchor_rank))
    except Exception as e:
        return {"status": "ERROR", "stage": repr(e)[:200]}
    finally:
        ME.COMPOSITION_CODES = saved


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=3 * 60 * 60,
              retries=0, volumes={ARTIFACT_ROOT: artifact_volume})
def compare_seed(job: dict) -> dict:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    import numpy as np
    rt = _runtime(); model, system = rt["model"], rt["system"]
    from compose_v4.control.semantic_actions import RING_ELEMENTS
    from compose_v4.control.macro_engine import ELEMENT_CODE

    seed = job["seed_smiles"]; size = job["size"]; topo = job["topology"]
    K = int(job["k_proposals"])
    free_set = {ELEMENT_CODE[e] for e in RING_ELEMENTS if e in ELEMENT_CODE}
    stoichs = _admissible_stoichs(size, job.get("max_hetero", 3))
    rng = np.random.default_rng(int(job.get("seed_rng", 7)))
    draw = rng.integers(0, len(stoichs), size=K)          # uniform, no prior

    out = {"cell": job["cell"], "seed": seed, "size": size, "topology": topo,
           "n_admissible_stoichs": len(stoichs),
           "admissible_stoichs": [list(map(list, s)) for s in stoichs],
           "declared_ring_elements": list(RING_ELEMENTS), "arms": {}}
    t0 = time.time()
    for arm in ("narrow_frozen", "free_composition", "stoich_enumerated"):
        rows = []
        for k in range(K):                                # SAME K for every arm
            if arm == "narrow_frozen":
                r = _build(model, system, seed, size, topo, "carbon_rich", None, k)
                used = None
            elif arm == "free_composition":
                r = _build(model, system, seed, size, topo, None, None, k,
                           allowed_override=free_set)
                used = "free"
            else:
                s = stoichs[int(draw[k])]
                r = _build(model, system, seed, size, topo, "mixed",
                           [list(x) for x in s], k)
                used = [list(x) for x in s]
            met = _endpoint_metrics(seed, r.get("smiles")) if r.get("smiles") else None
            rows.append({"k": k, "requested_stoich": used,
                         "status": r.get("status"), "stage": r.get("stage"),
                         "product": r.get("smiles"), "metrics": met})
        out["arms"][arm] = rows
    out["sec"] = time.time() - t0
    return out


@app.local_entrypoint()
def main(n_seeds: int = 6, k: int = 8, size: int = 6, probe: int = 0):
    seeds = json.loads(Path("docs/GENMOL_T4_SEEDS.json").read_text())
    dev = seeds[:n_seeds] if not probe else seeds[:probe]
    jobs = [{"cell": f"{s['target']}_s{s['idx']}", "seed_smiles": s["smiles"],
             "size": size, "topology": "pendant", "k_proposals": k,
             "seed_rng": 7 + s["idx"]} for s in dev]
    res = list(compare_seed.map(jobs))
    Path("diagnostics").mkdir(exist_ok=True)
    p = Path("diagnostics/stoich_realization_arm.json")
    p.write_text(json.dumps({"k_per_arm": k, "results": res}, indent=2))
    print(f"cells={len(res)} k_per_arm={k} wrote={p}")
    for r in res:
        print(f"  {r['cell']:12s} admissible_stoichs={r['n_admissible_stoichs']}")
        for arm, rows in r["arms"].items():
            ok = [x for x in rows if x["metrics"]]
            het = sum(x["metrics"]["n_added_hetero"] for x in ok)
            print(f"     {arm:18s} built {len(ok):>2d}/{len(rows)}  hetero_rings={het}")
