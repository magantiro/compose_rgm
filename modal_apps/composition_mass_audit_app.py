"""Authoritative conditional-composition audit at TRUE ring-growth decision states.

Instruments an exact replica of `build_ring_system_exact`'s growth loop, so the
RingFrontier, tip, anchors and running stoichiometry quota are the real ones.
At every genuine decision point it records four layers:

    canonical support  ->  R_theta  ->  Q_spec  ->  realized endpoint

All probability is computed on CANONICAL SUCCESSORS after aggregating the
successor fibers G_y(x) = {a : T(x,a) ~= y}. Mark-level counts are recorded ONLY
as a contrast field to exhibit the (element,valence) encoding artifact; they are
never the scientific quantity.

Counterfactual specs are evaluated AT THE SAME STATE without changing it, so
controller bias and reference bias are separable.

Completed rings are verified from the endpoint graph. `trace_ok` (compiler says
OK) and `semantic_ok` (graph agrees) are separate fields; only `semantic_ok`
counts as success, because a previous sample found 3/14 agreement.

Ring expansion is deliberately NOT audited here -- separate causal lane.

CPU only. No docking, no training, no oracle. Nothing here is a benchmark number.
"""

from __future__ import annotations

import json
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
app = modal.App("composition-mass-audit")
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


# ---------------------------------------------------------------- ring shape
def _ring_profile(smi):
    """Sorted per-ring '<size><a|s>:<composition>' descriptors."""
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


def _semantic_verify(seed_smi, prod_smi, size, stoich, topology, state="aromatic"):
    """Independent endpoint check. `trace_ok` is NOT trusted."""
    from rdkit import Chem
    from collections import Counter
    out = {"ring_exists": False, "size_ok": False, "stoich_ok": False,
           "topology_ok": False, "connected_ok": False, "added_rings": [],
           "state_ok": False, "semantic_ok": False}
    sm, pm = Chem.MolFromSmiles(seed_smi), Chem.MolFromSmiles(prod_smi)
    if sm is None or pm is None:
        return out
    out["connected_ok"] = (len(Chem.GetMolFrags(pm)) == 1)
    added = list((Counter(_ring_profile(prod_smi)) - Counter(_ring_profile(seed_smi))).elements())
    out["added_rings"] = added
    if not added:
        return out
    out["ring_exists"] = True
    want_n = {}
    for e, n in (stoich or []):
        want_n[e] = want_n.get(e, 0) + int(n)
    want_state = "a" if state == "aromatic" else "s"
    for d in added:
        head, comp = d.split(":")
        if int(head[:-1]) != int(size):
            continue
        out["size_ok"] = True
        if head[-1] == want_state:
            out["state_ok"] = True
        if not want_n:
            out["stoich_ok"] = True
        else:
            got = {}
            i = 0
            while i < len(comp):
                j = i + 1
                while j < len(comp) and comp[j].islower():
                    j += 1
                k = j
                while k < len(comp) and comp[k].isdigit():
                    k += 1
                got[comp[i:j]] = int(comp[j:k] or 1)
                i = k
            if got == want_n:
                out["stoich_ok"] = True
    # linked vs fused: does ring count rise by more than ring-system count?
    def _nsys(mm):
        rings = [set(r) for r in mm.GetRingInfo().AtomRings()]
        sysm = []
        for r in rings:
            merged, rest = [r], []
            for s in sysm:
                (merged if s & r else rest).append(s)
            sysm = rest + [set().union(*merged)]
        return len(sysm)
    d_sys = _nsys(pm) - _nsys(sm)
    d_ring = pm.GetRingInfo().NumRings() - sm.GetRingInfo().NumRings()
    out["d_ring_systems"], out["d_rings"] = int(d_sys), int(d_ring)
    out["topology_ok"] = bool(d_sys >= 1) if topology in ("pendant", "linked") else bool(d_ring >= 1 and d_sys == 0)
    out["semantic_ok"] = bool(out["ring_exists"] and out["size_ok"]
                              and out["stoich_ok"] and out["topology_ok"]
                              and out["connected_ok"] and out["state_ok"])
    return out


# ------------------------------------------------------- instrumented growth
def _run_one(model, system, seed_smi, size, topology, composition,
             stoich, anchor_rank, record):
    """Exact replica of build_ring_system_exact's growth loop, instrumented."""
    import numpy as np
    from rdkit import Chem
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import (
        RingFrontier, match_growth_descriptors, COMPOSITION_CODES, ELEMENT_CODE)
    from compose_v4.gates.med_chem_gate import is_valid

    Z2E = {v: k for k, v in ELEMENT_CODE.items()}
    _cache: dict = {}

    def to_smiles(st):
        return canonical_state_key(st)

    def enum_full(st):
        k = to_smiles(st)
        if k not in _cache:
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            _cache[k] = ([m.executor_rule_name for m in law.marks],
                         [m.action for m in law.marks],
                         np.array([m.probability for m in law.marks], float))
        return _cache[k]

    def apply_idx(st, j):
        fams, acts, _ = enum_full(st)
        try:
            return system.apply(st, fams[j], acts[j])
        except Exception:
            return None

    allowed = COMPOSITION_CODES.get(composition, COMPOSITION_CODES["carbon_rich"])
    _quota = None
    if stoich:
        _quota = {}
        for e, n in stoich:
            z = ELEMENT_CODE.get(e) if isinstance(e, str) else int(e)
            if z is None:
                return {"status": "UNSAT", "stage": "stoich_element"}
            _quota[z] = _quota.get(z, 0) + int(n)
        if sum(_quota.values()) != int(size):
            return {"status": "UNSAT", "stage": "stoich_sum"}

    fr = RingFrontier(size=int(size), topology=topology)
    cur = pad_molecular_graph(smiles_to_molecular_graph(seed_smi), CANONICAL_SLOTS)
    anchors = None
    trace = []

    for step in range(int(size)):
        fams, acts, probs = enum_full(cur)
        tip = fr.path[-1] if fr.path else None
        _allowed_step = allowed
        if _quota is not None:
            _allowed_step = {z for z, n in _quota.items() if n > 0}
            if not _allowed_step:
                return {"status": "UNSAT", "stage": f"quota_exhausted{step}", "trace": trace}

        # ---- LAYERS 1-3, at this genuine decision state --------------------
        if record is not None:
            # the macro-step-compatible mark set: single-neighbour atom_insert at tip
            step_idx = []
            for j, f in enumerate(fams):
                if f != "atom_insert":
                    continue
                nb = list(getattr(acts[j], "neighbors", ()) or ())
                if len(nb) != 1:
                    continue
                if tip is not None and int(nb[0][0]) != int(tip):
                    continue
                step_idx.append(j)
            # aggregate the fibers: canonical successor key -> summed mass
            by_key: dict = {}
            for j in step_idx:
                y = apply_idx(cur, j)
                if y is None:
                    continue
                k = canonical_state_key(y)
                z = int(getattr(acts[j], "atom_type", -1))
                e = Z2E.get(z, f"Z{z}")
                d = by_key.setdefault(k, {"mass": 0.0, "alias": 0, "elems": set()})
                d["mass"] += float(probs[j]); d["alias"] += 1; d["elems"].add(e)
            uniq, mass, alias = {}, {}, {}
            for k, d in by_key.items():
                if len(d["elems"]) != 1:
                    continue
                e = next(iter(d["elems"]))
                uniq[e] = uniq.get(e, 0) + 1
                mass[e] = mass.get(e, 0.0) + d["mass"]
                alias.setdefault(e, []).append(d["alias"])
            tm = sum(mass.values()) or 1.0
            tu = sum(uniq.values()) or 1
            # conditional odds vs the REMAINING quota
            live = {Z2E.get(z, f"Z{z}") for z in _allowed_step}
            r_compatible = sum(m for e, m in mass.items() if e in live) / tm
            # Q under the frozen spec and counterfactuals, SAME STATE
            q = {}
            specs = {"carbon_rich": COMPOSITION_CODES["carbon_rich"],
                     "mixed": COMPOSITION_CODES["mixed"],
                     "this_step_allowed": _allowed_step}
            for name, al in specs.items():
                idx = match_growth_descriptors(fams, acts, probs, tip, al,
                                               anchors=anchors if tip is None else None)
                qm = sum(float(probs[j]) for j in idx)
                q[name] = {"n_candidates": len(idx),
                           "mass_share_of_step": qm / (sum(float(probs[j]) for j in step_idx) or 1.0)}
            record.append({
                "seed": seed_smi, "step": step, "tip": (None if tip is None else int(tip)),
                "composition_spec": composition,
                "stoich": stoich, "remaining_quota": ({Z2E.get(z, z): n for z, n in _quota.items()}
                                                      if _quota else None),
                "ring_so_far": len(fr.path),
                "n_step_compatible_marks": len(step_idx),
                "n_unique_canonical_successors": len(by_key),
                "support_share": {e: uniq[e] / tu for e in uniq},
                "R_e": {e: mass[e] / tm for e in mass},
                "mean_alias": {e: sum(v) / len(v) for e, v in alias.items()},
                "R_compatible_with_remaining_quota": r_compatible,
                "Q": q,
            })

        # ---- selection, exactly as the real loop ---------------------------
        cands = match_growth_descriptors(fams, acts, probs, tip, _allowed_step,
                                         anchors=anchors if tip is None else None)
        if tip is None and cands:
            by_site = {}
            for j in cands:
                by_site.setdefault(int(acts[j].neighbors[0][0]), []).append(j)
            order = sorted(by_site, key=lambda s: -float(probs[by_site[s][0]]))
            if int(anchor_rank) >= len(order):
                return {"status": "UNSAT", "stage": "anchor_rank", "trace": trace}
            cands = by_site[order[int(anchor_rank)]]
            fr.anchors = [order[int(anchor_rank)]]
        if not cands:
            return {"status": "UNSAT", "stage": f"grow{step}", "trace": trace}
        placed = False
        for j in cands[:6]:
            y = apply_idx(cur, j)
            if y is None or not is_valid(to_smiles(y)):
                continue
            fr.observe_insert(acts[j])
            if _quota is not None:
                zp = int(getattr(acts[j], "atom_type", -1))
                _quota[zp] = _quota.get(zp, 0) - 1
            if record is not None:
                record[-1]["realized_element"] = Z2E.get(int(getattr(acts[j], "atom_type", -1)))
            trace.append(f"insert@{acts[j].slot}")
            cur, placed = y, True
            break
        if not placed:
            return {"status": "UNSAT", "stage": f"grow{step}_blocked", "trace": trace}

    return {"status": "GREW", "smiles": to_smiles(cur), "trace": trace,
            "frontier": list(fr.path)}


def _real_build(model, system, seed_smi, arm, anchor_rank):
    """The FROZEN build_ring_system_exact, unmodified. Its `status` is the
    compiler's claim; `_semantic_verify` independently checks the graph."""
    import numpy as np
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import build_ring_system_exact
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
    try:
        return build_ring_system_exact(
            enum_full, apply_fn, to_smiles, is_valid, st0,
            size=arm["size"], topology=arm["topology"],
            composition=arm["composition"], state="aromatic",
            stoich=arm.get("stoich"), anchor_rank=int(anchor_rank))
    except Exception as e:
        return {"status": "ERROR", "stage": repr(e)[:200]}


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=2 * 60 * 60,
              retries=0, volumes={ARTIFACT_ROOT: artifact_volume})
def audit_seed(job: dict) -> dict:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    rt = _runtime(); model, system = rt["model"], rt["system"]
    t0 = time.time()
    rec, arms = [], []
    for arm in job["arms"]:
        # pass 1 -- instrumented replica, ONLY to record decision-state layers
        _run_one(model, system, job["seed_smiles"], arm["size"], arm["topology"],
                 arm["composition"], arm.get("stoich"), job.get("anchor_rank", 0),
                 rec)
        # pass 2 -- the REAL frozen macro decides completion; trace vs graph
        real = _real_build(model, system, job["seed_smiles"], arm,
                           job.get("anchor_rank", 0))
        ver = None
        if real.get("smiles"):
            ver = _semantic_verify(job["seed_smiles"], real["smiles"], arm["size"],
                                   arm.get("stoich"), arm["topology"], "aromatic")
        arms.append({"arm": arm,
                     "trace_ok": real.get("status") == "OK",
                     "trace_status": real.get("status"), "stage": real.get("stage"),
                     "product": real.get("smiles"), "semantic": ver})
    return {"cell": job["cell"], "sec": time.time() - t0, "decisions": rec, "arms": arms}


@app.local_entrypoint()
def main(n_seeds: int = 15, probe: int = 0):
    seeds = json.loads(Path("docs/GENMOL_T4_SEEDS.json").read_text())[:n_seeds]
    ARMS = [
        {"size": 6, "topology": "pendant", "composition": "carbon_rich", "stoich": None},
        {"size": 6, "topology": "pendant", "composition": "mixed", "stoich": [["C", 5], ["N", 1]]},
        {"size": 6, "topology": "pendant", "composition": "mixed", "stoich": [["C", 4], ["N", 2]]},
        {"size": 5, "topology": "pendant", "composition": "mixed", "stoich": [["C", 4], ["N", 1]]},
    ]
    if probe:
        ARMS = ARMS[:2]; seeds = seeds[:probe]
    jobs = [{"cell": f"{s['target']}_s{s['idx']}", "seed_smiles": s["smiles"],
             "arms": ARMS, "anchor_rank": 0} for s in seeds]
    out = list(audit_seed.map(jobs))
    Path("diagnostics").mkdir(exist_ok=True)
    p = Path("diagnostics/composition_mass_audit.json")
    p.write_text(json.dumps({"arms": ARMS, "results": out}, indent=2))
    nd = sum(len(r["decisions"]) for r in out)
    print(f"cells={len(out)} decision_states={nd} wrote={p}")
    for r in out:
        for a in r["arms"]:
            st = a["semantic"]
            print(f"  {r['cell']:14s} {str(a['arm'].get('stoich')):22s} "
                  f"trace_ok={a['trace_ok']!s:5s} semantic_ok="
                  f"{(st or {}).get('semantic_ok') if st else None} stage={a['stage']}")
