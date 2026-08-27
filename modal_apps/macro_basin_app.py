"""Can the macro engine reach the IVG structural basin at all? (development only)

STAGE 1 of a deliberate information-removal sequence:

    1  target-aware  : basin score = similarity to the published IVG winners
    2  target-blind  : identical engine, basin score removed
    3  protein       : docking chooses the basin

This is stage 1. It uses the IVG winners as a DEVELOPMENT signal to answer one
question with no docking at all: can grow/cyclize/aromatize/rebuild macros move
COMPOSE into IVG-like topology, which the old controller demonstrably could not?
Old COMPOSE on parp1 s0 reached 26 heavy atoms but stayed at 3 rings / 2
aromatic, against IVG's 30-33 heavy / 5-6 rings / 3 aromatic; on 5ht1b s7 it
grew to 32 heavy atoms and produced a 13-membered macrocycle with 1 aromatic
ring, against IVG's 5 rings / 3 aromatic.

A NULL HERE IS INFORMATIVE and must not be explained away: if the engine cannot
reach the basin even when TOLD the target, the defect is not the blind reward.

Nothing here is a benchmark number. Any cell whose winners are read at this
stage can never again support a "never inspected" claim -- already true of all
25, recorded in the handoff.
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
app = modal.App("macro-basin")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
MEM_MIB = int(3 * 1024)   # R_theta is single-threaded; 6 GiB was copied,
                          # not measured. Round 0 probes that 3 GiB holds.

_RT: dict[str, Any] = {}


def _runtime():
    """Inlined: the image mounts only src/ and configs/, so a sibling
    modal_apps module does not exist inside the container."""
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


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=3 * 60 * 60,
              retries=0, max_containers=80, enable_memory_snapshot=True,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def expand(job: dict) -> list[dict]:
    """One (state, macro, length) program, repeated n_rep times. Endpoint only."""
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
    from rdkit.Chem import rdMolDescriptors as rdMD
    import sys, os
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import rollout
    from compose_v4.gates.med_chem_gate import is_valid

    rt = _runtime(); model, system = rt["model"], rt["system"]
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    _cache: dict[str, Any] = {}

    def enumerate_fn(smi):
        if smi in _cache:
            return _cache[smi]
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        fams = [m.executor_rule_name for m in law.marks]
        probs = np.array([m.probability for m in law.marks], float)
        handles = [(st, m.executor_rule_name, m.action) for m in law.marks]
        _cache[smi] = (fams, probs, handles)
        return _cache[smi]

    def apply_fn(smi, handle):
        st, rule, action = handle
        try:
            return canonical_state_key(system.apply(st, rule, action))
        except Exception:
            return None

    winners = [Chem.MolFromSmiles(w) for w in job["winners"]]
    wfps = [gm.GetFingerprint(m) for m in winners if m is not None]
    seed_m = Chem.MolFromSmiles(job["seed"])
    seedfp = gm.GetFingerprint(seed_m)

    def profile(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        fp = gm.GetFingerprint(m)
        return dict(
            smi=smi,
            basin=round(max(DataStructs.TanimotoSimilarity(fp, w) for w in wfps), 4),
            sim_seed=round(DataStructs.TanimotoSimilarity(fp, seedfp), 4),
            heavy=m.GetNumHeavyAtoms(),
            rings=int(rdMD.CalcNumRings(m)),
            arom=int(rdMD.CalcNumAromaticRings(m)),
            qed=round(float(QED.qed(m)), 3),
            sa=round(float(sascorer.calculateScore(m)), 2))

    out = []
    t0 = time.time()
    for rep in range(int(job["n_rep"])):
        rng = np.random.default_rng(job["seed_rng"] + 1000 * rep)
        r = rollout(enumerate_fn, apply_fn, is_valid, job["state"], job["macro"],
                    job["length"], rng, temperature=job.get("temperature", 2.0),
                    epsilon=job.get("epsilon", 0.15))
        p = profile(r.endpoint)
        if p is None:
            continue
        p.update(macro=job["macro"], length=job["length"], executed=r.length,
                 halted=r.halted, parent=job["state"],
                 wf_ranks=[s.within_family_rank for s in r.steps])
        out.append(p)
    print(f"{job['macro']}/L{job['length']}: {len(out)} endpoints "
          f"in {time.time()-t0:.0f}s", flush=True)
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=int(1 * 1024),
              timeout=6 * 60 * 60, retries=0,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(payload: dict, rounds: int = 3, per_stratum: int = 3, n_rep: int = 2):
    """Beam over macro PROGRAMS with STRATIFIED survival.

    Ranking every macro boundary against a single global basin score recreates
    exactly the myopia we spent the day diagnosing: the parp1 route moves AWAY
    from the target before it pays (basin 0.483 at the seed; the witness dips to
    0.128 similarity for nine states before the reclose recovers it). Under
    global best-carried-forward the seed simply keeps winning and every opened
    or grown state is deleted before it can ever cyclize.

    So survival is per (macro, depth) STRATUM: each program type keeps its own
    survivors regardless of how it compares to the seed. Only COMPLETED
    multi-macro programs are ranked against each other at the end.
    """
    # docs/ and diagnostics/ are NOT mounted into the image (only src/ and
    # configs/ are), so the caller ships the two blobs this needs.
    seeds = payload["seeds"]
    wins = payload["winners"]
    cells = payload["cells"]
    CONSTRUCTIVE = ["grow", "open", "rebuild", "cyclize", "restate"]
    LENGTHS = [2, 4, 6]
    out = {}
    for cell in cells:
        tgt, si, _d = cell.rsplit("_", 2)
        sd = seeds[f"{tgt}_{si}"]
        W = [w["smiles"] for w in wins[cell]["winners"]]
        # frontier is a dict: (macro, length) -> [states], each stratum protected
        frontier = {None: [sd["smiles"]]}
        all_endpoints, history = [], []
        for rd in range(int(rounds)):
            jobs = []
            k = 0
            for _key, states in frontier.items():
                for stt in states:
                    for m in CONSTRUCTIVE:
                        for L in LENGTHS:
                            for rep in range(int(n_rep)):
                                # one job per REP: 15 jobs could only ever use 15
                                # containers, which is why the first attempt ran
                                # on 3. This fans out n_rep times wider.
                                jobs.append(dict(state=stt, macro=m, length=L,
                                                 n_rep=1, winners=W,
                                                 seed=sd["smiles"],
                                                 seed_rng=20260825 + 7919 * k))
                                k += 1
            print(f"\n[{cell}] round {rd}: {len(jobs)} programs from "
                  f"{sum(len(v) for v in frontier.values())} states", flush=True)
            res = [e for chunk in expand.map(jobs) for e in chunk]
            if not res:
                print("  no endpoints produced", flush=True); break
            all_endpoints.extend(res)

            # STRATIFIED survival: top per_stratum within each (macro, length),
            # never ranked against the seed or against other strata.
            nxt, seen = {}, set()
            for m in CONSTRUCTIVE:
                for L in LENGTHS:
                    cand = sorted((r for r in res
                                   if r["macro"] == m and r["length"] == L),
                                  key=lambda r: -r["basin"])
                    keep = []
                    for r in cand:
                        if r["smi"] in seen:
                            continue
                        seen.add(r["smi"]); keep.append(r["smi"])
                        if len(keep) >= int(per_stratum):
                            break
                    if keep:
                        nxt[(m, L)] = keep
            frontier = nxt or {None: [sd["smiles"]]}

            best = max(res, key=lambda r: r["basin"])
            rings = max(res, key=lambda r: (r["rings"], r["arom"]))
            history.append(dict(round=rd, n=len(res), best=best, most_rings=rings))
            print(f"  best basin={best['basin']} (rings={best['rings']} "
                  f"arom={best['arom']} heavy={best['heavy']}) via "
                  f"{best['macro']}/L{best['length']}", flush=True)
            print(f"  most rings={rings['rings']} arom={rings['arom']} "
                  f"basin={rings['basin']} via {rings['macro']}/L{rings['length']}",
                  flush=True)
            ck = Path("/artifacts/macro_basin") / f"{cell}_round{rd}.json"
            ck.parent.mkdir(parents=True, exist_ok=True)
            ck.write_text(json.dumps(dict(cell=cell, round=rd,
                                          endpoints=sorted(res, key=lambda r: -r["basin"])[:300],
                                          frontier={f"{a}_{b}": v for (a, b), v in
                                                    (frontier or {}).items() if a}),
                                     indent=1))
            artifact_volume.commit()
            print(f"  checkpointed -> {ck}", flush=True)

        # COMPOSED programs ranked only now, at completion
        all_endpoints.sort(key=lambda r: -r["basin"])
        out[cell] = dict(history=history, top=all_endpoints[:50],
                         seed=sd["smiles"], n_total=len(all_endpoints))
    fin = Path("/artifacts/macro_basin/stage1_final.json")
    fin.parent.mkdir(parents=True, exist_ok=True)
    fin.write_text(json.dumps(out, indent=1))
    artifact_volume.commit()
    print(f"\nwrote {fin}", flush=True)
    return out


@app.local_entrypoint()
def launch(cells: str = "parp1_s0_d0.4,5ht1b_s7_d0.4", rounds: int = 3,
           per_stratum: int = 3, n_rep: int = 2):
    """Spawn on the DEPLOYED app so the run outlives this client session."""
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    payload = dict(
        cells=[c.strip() for c in cells.split(",") if c.strip()],
        seeds={f"{s['target']}_s{s['idx']}": s for s in
               json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())},
        winners=json.loads((root / "diagnostics/ivg_winners.json").read_text()))
    fn = _m.Function.from_name("macro-basin", "drive")
    h = fn.spawn(payload, rounds=rounds, per_stratum=per_stratum, n_rep=n_rep)
    print(f"  SPAWNED on deployed macro-basin: {h.object_id}")
    print(f"  per-round checkpoints -> /artifacts/macro_basin/<cell>_round<N>.json")


# ---------------------------------------------------------------------------
# PROGRAM GRAMMAR. Round 0 gave every macro the same duration ladder, which is
# semantically wrong: "cyclize, L=6" means "close a ring, then another, then
# another" and turned a 19-atom scaffold into a 10-ring cage at SA 6.96. That is
# not evidence against cyclization -- it fired perfectly when commanded. It is
# evidence that a CYCLIZATION IS ONE STRUCTURAL EVENT, and the horizon must be
# macro-specific. After a closure the controller must replan, not close again.
# ---------------------------------------------------------------------------
MACRO_HORIZON = {"grow": (3, 4, 6), "open": (1,), "cyclize": (1,),
                 "restate": (1, 2, 3), "rebuild": (1, 2)}

LIPO_PROGRAMS = [
    [("local_extend:2", 8), ("close_disjoint", 1), ("restate", 2)],
    [("local_extend:2:carbon_rich", 8), ("close_disjoint", 1), ("restate", 2)],
    [("local_extend:2:mixed", 8), ("close_disjoint", 1), ("restate", 2)],
]

REGULARITY_PROGRAMS = [
    [("local_extend:2", 8), ("close_disjoint", 1), ("restate", 2)],   # control
    [("local_extend:2", 8), ("regular_ring", 1), ("restate", 2)],     # test
]

LOCALITY_PROGRAMS = [
    [("local_extend:1", 8), ("close_disjoint", 1), ("restate", 2)],
    [("local_extend:2", 8), ("close_disjoint", 1), ("restate", 2)],
    [("local_extend:3", 8), ("close_disjoint", 1), ("restate", 2)],
]

STATE_PROGRAMS = [
    # identical precursor + topology; ONLY the electronic-state intent differs
    [("scaffold_extend:carbon_rich", 8), ("close_disjoint", 1), ("restate", 2)],
    [("scaffold_extend:carbon_rich", 8), ("close_disjoint:saturated", 1), ("restate", 2)],
    [("scaffold_extend:carbon_rich", 8), ("close_disjoint:aromatic", 1), ("restate", 2)],
    [("scaffold_extend:carbon_rich", 8), ("close_disjoint:aromatic", 1), ("aromatize", 3)],
]

COMPOSITION_PROGRAMS = [
    # matched three-way: identical topology + refinement, composition varies
    [("grow", 8), ("close_disjoint", 1), ("restate", 2)],
    [("scaffold_extend:mixed", 8), ("close_disjoint", 1), ("restate", 2)],
    [("scaffold_extend:carbon_rich", 8), ("close_disjoint", 1), ("restate", 2)],
    [("scaffold_extend:carbon_rich", 10), ("close_disjoint", 1), ("restate", 2)],
    # carbon-rich with a longer refinement, in case aromatisation needs room
    [("scaffold_extend:carbon_rich", 8), ("close_disjoint", 1), ("restate", 5)],
]

SCAFFOLD_PROGRAMS = [
    # the earned program: backbone-only extension, then a substantial pendant
    [("scaffold_extend", 8), ("append_system", 1), ("restate", 2)],
    [("scaffold_extend", 10), ("append_system", 1), ("restate", 2)],
    # CONTROLS at matched budget: plain grow, and grow + plain disjoint closure
    [("grow", 8), ("append_system", 1), ("restate", 2)],
    [("scaffold_extend", 8), ("close_disjoint", 1), ("restate", 2)],
    [("grow", 8), ("close_disjoint", 1), ("restate", 2)],
]

APPEND_PROGRAMS = [
    # long enough chain that a ring can close among NEW atoms only
    [("grow", 8), ("close_disjoint", 1), ("restate", 2)],
    [("grow", 10), ("close_disjoint", 1), ("restate", 2)],
    [("grow", 8), ("close_disjoint", 1), ("restate", 2),
     ("grow", 8), ("close_disjoint", 1), ("restate", 2)],
    # CONTROLS, same atom budget, ordinary cyclize -> should keep annulating
    [("grow", 8), ("cyclize", 1), ("restate", 2)],
    [("grow", 10), ("cyclize", 1), ("restate", 2)],
    [("grow", 4), ("cyclize", 1), ("restate", 2)],
]

PROGRAMS = [
    [("grow", 4), ("cyclize", 1), ("restate", 2)],
    [("grow", 6), ("cyclize", 1), ("restate", 2)],
    [("open", 1), ("grow", 4), ("cyclize", 1), ("restate", 2)],
    [("open", 1), ("grow", 6), ("cyclize", 1), ("restate", 2)],
    # two constructive cycles: add, organise, add again, organise again
    [("grow", 4), ("cyclize", 1), ("restate", 2),
     ("grow", 4), ("cyclize", 1), ("restate", 2)],
    [("grow", 6), ("cyclize", 1), ("restate", 1),
     ("grow", 4), ("cyclize", 1), ("restate", 2)],
]


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=2 * 60 * 60,
              retries=0, max_containers=80, enable_memory_snapshot=True,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_program(job: dict) -> dict:
    """Execute ONE full program end to end. Endpoint scored only at completion."""
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
    from rdkit.Chem import rdMolDescriptors as rdMD
    import sys, os
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import (rollout, synthesis_preference,
                                                 disjoint_closure_only)
    from compose_v4.gates.med_chem_gate import is_valid

    rt = _runtime(); model, system = rt["model"], rt["system"]
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    _cache = {}

    def enumerate_fn(smi):
        if smi not in _cache:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            _cache[smi] = ([m.executor_rule_name for m in law.marks],
                           np.array([m.probability for m in law.marks], float),
                           [(st, m.executor_rule_name, m.action) for m in law.marks])
        return _cache[smi]

    def apply_fn(smi, handle):
        st, rule, action = handle
        try: return canonical_state_key(system.apply(st, rule, action))
        except Exception: return None

    wfps = [gm.GetFingerprint(Chem.MolFromSmiles(w)) for w in job["winners"]]
    seedfp = gm.GetFingerprint(Chem.MolFromSmiles(job["seed"]))
    rng = np.random.default_rng(job["seed_rng"])
    cur = job["seed"]; trace = []
    for macro, L in job["program"]:
        # "close_disjoint" is cyclize with the pendant-ring constraint: the new
        # ring must share no atom with any pre-existing ring, so it builds a
        # SECOND ring system instead of annulating onto the first. The predicate
        # is anchored at the state BEFORE the closure, which is why it is built
        # here rather than passed in.
        mode = None
        if ":" in macro:
            macro, mode = macro.split(":", 1)
        if macro == "local_extend":
            from compose_v4.control.macro_engine import local_extend, backbone_only
            # mode is "<anchors>" or "<anchors>:<composition>". Locality fixes
            # similarity; composition fixes lipophilicity. Each was measured
            # alone -- locality alone gives logP 1.3 against IVG's 3.7 because
            # it applies no composition restriction and R_theta's insert law is
            # carbon-poor; carbon_rich alone gives sim 0.143 because it applies
            # no locality restriction. They have never been combined.
            parts = (mode or "2").split(":")
            _loc = local_extend(cur, job["seed"], max_anchors=int(parts[0]))
            if len(parts) > 1:
                _cmp = backbone_only(cur, composition=parts[1])
                pref = (lambda y, _a=_loc, _b=_cmp: _a(y) and _b(y))
            else:
                pref = _loc
            r = rollout(enumerate_fn, apply_fn, is_valid, cur, "grow", L, rng,
                        temperature=job.get("temperature", 2.0),
                        epsilon=job.get("epsilon", 0.15), prefer_fn=pref)
            trace.append(dict(macro=f"local_extend:{mode}", L=L, executed=r.length,
                              halted=r.halted, smi=r.endpoint))
            cur = r.endpoint
            if r.halted and r.length == 0:
                break
            continue
        if macro in ("scaffold_extend", "append_system", "annulate",
                     "small_ring", "decorate", "shrink", "aromatize"):
            # macro CONTRACT: hard within the macro, never global. Falls back to
            # the gate alone if nothing satisfies it, so a contract can never
            # make a macro unexecutable.
            from compose_v4.control.macro_engine import contract_for
            pref = (contract_for(macro, cur, composition=mode) if mode
                    else contract_for(macro, cur))
            eff = {"scaffold_extend": "grow", "append_system": "cyclize",
                   "annulate": "cyclize", "small_ring": "cyclize",
                   "decorate": "grow", "shrink": "shrink",
                   "aromatize": "restate"}[macro]
        elif macro == "regular_ring":
            from compose_v4.control.macro_engine import (regular_ring_closure,
                                                         disjoint_closure_only)
            _d = disjoint_closure_only(cur); _r = regular_ring_closure(cur)
            pref = (lambda y, _a=_d, _b=_r: _a(y) and _b(y))
            eff = "cyclize"
        elif macro.startswith("close_disjoint"):
            from compose_v4.control.macro_engine import ring_state_contract
            base = disjoint_closure_only(cur)
            want = macro.split(":", 1)[1] if ":" in macro else None
            if want:
                st_ok = ring_state_contract(cur, want)
                pref = (lambda y, _b=base, _s=st_ok: _b(y) and _s(y))
            else:
                pref = base
            eff = "cyclize"
        elif job.get("prefer", True):
            pref = synthesis_preference; eff = macro
        else:
            pref = None; eff = macro
        eps = job.get("epsilon", 0.15)
        if isinstance(eps, dict):          # per-macro epsilon
            eps = float(eps.get(eff, eps.get("_default", 0.15)))
        r = rollout(enumerate_fn, apply_fn, is_valid, cur, eff, L, rng,
                    temperature=job.get("temperature", 2.0),
                    epsilon=float(eps), prefer_fn=pref)
        trace.append(dict(macro=macro, L=L, executed=r.length, halted=r.halted,
                          smi=r.endpoint))
        cur = r.endpoint
        if r.halted and r.length == 0:
            break
    m = Chem.MolFromSmiles(cur)
    if m is None:
        return dict(program=job["program"], failed=True, trace=trace)
    fp = gm.GetFingerprint(m)
    from compose_v4.control.macro_engine import ring_systems
    return dict(program=job["program"], trace=trace, smi=cur,
                systems=len(ring_systems(m)),
                basin=round(max(DataStructs.TanimotoSimilarity(fp, w) for w in wfps), 4),
                sim_seed=round(DataStructs.TanimotoSimilarity(fp, seedfp), 4),
                heavy=m.GetNumHeavyAtoms(), rings=int(rdMD.CalcNumRings(m)),
                arom=int(rdMD.CalcNumAromaticRings(m)),
                qed=round(float(QED.qed(m)), 3),
                sa=round(float(sascorer.calculateScore(m)), 2))


@app.local_entrypoint()
def grammar(cell: str = "parp1_s0_d0.4", realizations: int = 16):
    """Test a handful of explicit programs, not a generic macro search."""
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    wins = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    tgt, si, _d = cell.rsplit("_", 2)
    sd = seeds[f"{tgt}_{si}"]
    W = [w["smiles"] for w in wins[cell]["winners"]]
    jobs = [dict(program=p, seed=sd["smiles"], winners=W,
                 seed_rng=20260825 + 7919 * (pi * 1000 + r))
            for pi, p in enumerate(PROGRAMS) for r in range(int(realizations))]
    print(f"[{cell}] {len(PROGRAMS)} programs x {realizations} realizations "
          f"= {len(jobs)} jobs")
    fn = _m.Function.from_name("macro-basin", "run_program")
    res = [r for r in fn.map(jobs) if r and not r.get("failed")]
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD
    sm = Chem.MolFromSmiles(sd["smiles"])
    print(f"\nSEED: basin=? rings={rdMD.CalcNumRings(sm)} "
          f"arom={rdMD.CalcNumAromaticRings(sm)} heavy={sm.GetNumHeavyAtoms()}")
    by = {}
    for r in res:
        key = " -> ".join(f"{m}{L}" for m, L in r["program"])
        by.setdefault(key, []).append(r)
    print(f"\n{'program':44s} {'n':>3s} {'bestBasin':>9s} {'rings':>5s} {'arom':>4s} "
          f"{'heavy':>5s} {'feas':>5s}")
    for k, v in by.items():
        b = max(v, key=lambda r: r["basin"])
        t = max(v, key=lambda r: (r["rings"], r["arom"]))
        feas = sum(1 for r in v if r["qed"] >= 0.6 and r["sa"] <= 4)
        print(f"{k:44s} {len(v):3d} {b['basin']:9.4f} {t['rings']:5d} {t['arom']:4d} "
              f"{b['heavy']:5d} {feas:4d}/{len(v)}")
    out = Path("/tmp/grammar_" + cell + ".json")
    (root / f"diagnostics/grammar_{cell}.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote diagnostics/grammar_{cell}.json")


@app.function(image=image, cpu=(1.0, 1.0), memory=int(1 * 1024),
              timeout=4 * 60 * 60, retries=0,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive_grammar(payload: dict, realizations: int = 16):
    """Fan out the grammar test FROM INSIDE Modal.

    A local entrypoint calling fn.map() keeps the fan-out tethered to the client:
    a laptop sleeping mid-run kills it and loses everything (that is exactly how
    the first attempt died, with zero results written). The driver runs here and
    commits per program, so a dropped client costs nothing.
    """
    out = []
    for pi, prog in enumerate(PROGRAMS):
        jobs = [dict(program=prog, seed=payload["seed"], winners=payload["winners"],
                     seed_rng=20260825 + 7919 * (pi * 1000 + r))
                for r in range(int(realizations))]
        res = [r for r in run_program.map(jobs) if r and not r.get("failed")]
        out.extend(res)
        key = " -> ".join(f"{m}{L}" for m, L in prog)
        if res:
            b = max(res, key=lambda r: r["basin"])
            t = max(res, key=lambda r: (r["rings"], r["arom"]))
            feas = sum(1 for r in res if r["qed"] >= 0.6 and r["sa"] <= 4)
            print(f"{key:46s} best_basin={b['basin']:.4f} "
                  f"maxrings={t['rings']} arom={t['arom']} feasible={feas}/{len(res)}",
                  flush=True)
        ck = Path("/artifacts/macro_basin") / f"grammar_{payload['cell']}.json"
        ck.parent.mkdir(parents=True, exist_ok=True)
        ck.write_text(json.dumps(out, indent=1))
        artifact_volume.commit()
    return out


@app.local_entrypoint()
def launch_grammar(cell: str = "parp1_s0_d0.4", realizations: int = 16):
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    wins = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    tgt, si, _d = cell.rsplit("_", 2)
    payload = dict(cell=cell, seed=seeds[f"{tgt}_{si}"]["smiles"],
                   winners=[w["smiles"] for w in wins[cell]["winners"]])
    fn = _m.Function.from_name("macro-basin", "drive_grammar")
    h = fn.spawn(payload, realizations=realizations)
    print(f"  SPAWNED grammar test: {h.object_id}")
    print(f"  -> /artifacts/macro_basin/grammar_{cell}.json (committed per program)")


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024),
              timeout=4 * 60 * 60, retries=0,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive_dock_test(payload: dict, realizations: int = 48, dock_budget: int = 60):
    """END-TO-END: macro grammar -> T4 feasibility + med-chem gate -> docking.

    TARGET-BLIND. The IVG winners are NOT used to select what gets docked; only
    the T4 constraints and the chemistry gate filter, and docking ranks. The
    winners are carried solely to REPORT basin similarity afterwards, so we can
    see whether docking and structural similarity agree.

    Reports best-at-20 as well as best-at-N, because 20 calls was the
    precommitted development budget and reporting only the larger number would
    quietly move the goalpost.
    """
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
    from rdkit.Chem import rdMolDescriptors as rdMD
    import sys, os
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.gates.med_chem_gate import is_valid, t4_feasible
    import modal as _m

    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seedfp = gm.GetFingerprint(Chem.MolFromSmiles(payload["seed"]))
    delta = float(payload["delta"])

    # HALF the realizations run WITHOUT the preference, as the control. A yield
    # improvement measured only against a previous run confounds the preference
    # with everything else that changed; this makes it a within-run comparison.
    # the repaired controller: carbon-rich backbone -> disjoint pendant closure
    # -> refine. Feasible yield measured at ~35%, up from 3.1%.
    progs = payload.get("programs") or PROGRAMS
    jobs = [dict(program=prog, seed=payload["seed"], winners=payload["winners"],
                 prefer=False,
                 seed_rng=20260825 + 7919 * (pi * 10000 + r))
            for pi, prog in enumerate(progs) for r in range(int(realizations))]
    print(f"generating: {len(jobs)} program realizations", flush=True)
    gen = []
    for j, r in zip(jobs, run_program.map(jobs)):
        if r and not r.get("failed"):
            r["prefer"] = j["prefer"]; gen.append(r)
    print(f"generated {len(gen)} endpoints", flush=True)

    # TARGET-BLIND filter: T4 constraints + chemistry gate only.
    keep, seen = [], set()
    for r in gen:
        smi = r["smi"]
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        can = Chem.MolToSmiles(m)
        if can in seen:
            continue
        # THE real filter: QED, SA, similarity AND gate. Every earlier macro
        # experiment measured only QED+SA and called it feasible; the true rate
        # was 1.7% because similarity is what binds.
        if not t4_feasible(smi, payload["seed"], delta):
            continue
        q, sa = float(QED.qed(m)), float(sascorer.calculateScore(m))
        sim = float(DataStructs.TanimotoSimilarity(seedfp, gm.GetFingerprint(m)))
        seen.add(can)
        keep.append(dict(smi=can, qed=round(q, 3), sa=round(sa, 2),
                         sim=round(sim, 3), rings=int(rdMD.CalcNumRings(m)),
                         arom=int(rdMD.CalcNumAromaticRings(m)),
                         heavy=m.GetNumHeavyAtoms(), basin=r["basin"],
                         program=r["program"]))
    npref = sum(1 for r in gen if r.get("prefer"))
    print(f"feasible + gate-clean + unique: {len(keep)} "
          f"(prefer arm {npref}/{len(gen)} of generated)", flush=True)
    if not keep:
        return dict(error="no feasible candidates", n_generated=len(gen))

    # Prefer structurally AMBITIOUS candidates (most rings) -- a blind heuristic,
    # not a target-aware one. Ties broken by displacement from the seed.
    keep.sort(key=lambda r: (-r["rings"], -r["arom"], r["sim"]))
    cands = keep[: int(dock_budget)]
    dock = _m.Function.from_name("genmol-t4-gate4", "dock_frontier")
    chunks = [cands[i::6] for i in range(6)]      # 6-way fan-out over docking
    dj = [dict(smiles=[c["smi"] for c in ch], target=payload["target"],
               idx=payload["idx"], delta=delta, beta=4.0,
               budget=len(ch), tag=f"macrodock{i}")
          for i, ch in enumerate(chunks) if ch]
    print(f"docking {sum(len(c['smiles']) for c in dj)} molecules "
          f"across {len(dj)} containers", flush=True)
    pairs = []
    for r in dock.map(dj, order_outputs=False):
        pairs.extend(r.get("top") or [])
    pairs.sort(key=lambda p: p[1])
    prof = {c["smi"]: c for c in cands}
    out = dict(n_generated=len(gen), n_feasible=len(keep), n_docked=len(pairs),
               top=[dict(smi=s, ds=d, **{k: v for k, v in prof.get(s, {}).items()
                                         if k != "smi"}) for s, d in pairs[:20]])
    ck = Path("/artifacts/macro_basin") / f"docktest_{payload['cell']}.json"
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps(out, indent=1))
    artifact_volume.commit()
    if pairs:
        print(f"BEST ds={pairs[0][1]}  {pairs[0][0]}", flush=True)
    return out


@app.local_entrypoint()
def dock_test(cell: str = "parp1_s0_d0.4", realizations: int = 48,
              dock_budget: int = 20, repaired: bool = True):
    """Frozen localized controller on any cell."""
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    wins = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    tgt, si, dl = cell.rsplit("_", 2)
    sd = seeds[f"{tgt}_{si}"]
    alt = f"{tgt}_{si}_d0.4" if cell not in wins else cell
    ws = [w["smiles"] for w in wins.get(alt, {}).get("winners", [])] or [sd["smiles"]]
    payload = dict(cell=cell, seed=sd["smiles"], target=sd["target"],
                   idx=sd["idx"], delta=float(dl[1:]), winners=ws)
    if repaired:
        # FROZEN: locality was the structural fix. carbon_rich blew the
        # similarity budget (median 0.143); regular_ring made SA worse and had a
        # predicate bug. This program produced the first true T4-feasible
        # molecule: sim 0.597, QED 0.861, SA 3.67.
        payload["programs"] = [
            [("local_extend:2", 8), ("close_disjoint", 1), ("restate", 2)],
        ]
    fn = _m.Function.from_name("macro-basin", "drive_dock_test")
    h = fn.spawn(payload, realizations=realizations, dock_budget=dock_budget)
    print(f"  SPAWNED dock test: {h.object_id}")
    print(f"  -> /artifacts/macro_basin/docktest_{cell}.json")


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=2 * 60 * 60,
              retries=0, max_containers=80, enable_memory_snapshot=True,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def closure_arms(job: dict) -> dict:
    """A/B/C closure selection on the SAME pre-cyclization state.

    A  raw R_theta            -- highest-probability legal closure
    B  R_theta + SA/QED/SIM   -- terminal properties, no topology term
    C  B + topology prior     -- fused/aromatic favoured, bridged penalised

    Enumerating every legal closure and choosing deterministically (rather than
    sampling) is what makes the three arms comparable: identical state, identical
    candidate set, only the ranking differs. Seven-membered rings are NOT banned
    in any arm -- our own data showed a hard ban removes the only successes.
    """
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
    from rdkit.Chem import rdMolDescriptors as rdMD
    import sys, os
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import (rollout, MACRO_FAMILIES,
                                                 synthesis_preference)
    from compose_v4.gates.med_chem_gate import is_valid

    rt = _runtime(); model, system = rt["model"], rt["system"]
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seedfp = gm.GetFingerprint(Chem.MolFromSmiles(job["seed"]))
    delta = float(job["delta"])
    _cache = {}

    def enum(smi):
        if smi not in _cache:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            _cache[smi] = ([m.executor_rule_name for m in law.marks],
                           np.array([m.probability for m in law.marks], float),
                           [(st, m.executor_rule_name, m.action) for m in law.marks])
        return _cache[smi]

    def apply_h(smi, h):
        st, rule, action = h
        try: return canonical_state_key(system.apply(st, rule, action))
        except Exception: return None

    def prof(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None: return None
        ri = m.GetRingInfo(); rings = ri.AtomRings()
        return dict(smi=smi, sa=float(sascorer.calculateScore(m)),
                    qed=float(QED.qed(m)),
                    sim=float(DataStructs.TanimotoSimilarity(
                        seedfp, gm.GetFingerprint(m))),
                    bridge=int(rdMD.CalcNumBridgeheadAtoms(m)),
                    spiro=int(rdMD.CalcNumSpiroAtoms(m)),
                    stereo=len(Chem.FindMolChiralCenters(
                        m, includeUnassigned=True, useLegacyImplementation=False)),
                    rings=len(rings), arom=int(rdMD.CalcNumAromaticRings(m)),
                    heavy=m.GetNumHeavyAtoms(),
                    sizes=sorted(len(r) for r in rings))

    # 1. build ONE pre-cyclization state by growing from the seed
    rng = np.random.default_rng(job["seed_rng"])
    g = rollout(enum, apply_h, is_valid, job["seed"], "grow", int(job["grow_len"]),
                rng, prefer_fn=None)
    pre = g.endpoint
    p0 = prof(pre)
    if p0 is None or g.length == 0:
        return dict(error="grow failed", state=pre)

    # 2. enumerate EVERY legal closure from that state
    fams, probs, handles = enum(pre)
    cands = []
    for j in range(len(fams)):
        if fams[j] not in MACRO_FAMILIES["cyclize"]:
            continue
        y = apply_h(pre, handles[j])
        if not y or not is_valid(y):
            continue
        pr = prof(y)
        if pr is None: continue
        pr["r_theta"] = float(probs[j])
        pr["d_bridge"] = pr["bridge"] - p0["bridge"]
        pr["d_arom"] = pr["arom"] - p0["arom"]
        pr["new_ring"] = (sorted(set(pr["sizes"]) - set(p0["sizes"])) or [0])[-1]
        cands.append(pr)
    if not cands:
        return dict(error="no legal closure", state=pre, pre=p0)

    def scoreA(c): return c["r_theta"]
    def scoreB(c):
        return (c["r_theta"] * 1e3
                - 2.0 * max(0.0, c["sa"] - 4.0)
                - 2.0 * max(0.0, 0.6 - c["qed"])
                - 2.0 * max(0.0, delta - c["sim"]))
    def scoreC(c):
        # topology prior: bridging penalised, aromatisation rewarded.
        # ring SIZE is deliberately not penalised -- 7-rings are legitimate
        # medicinal chemistry and our data showed banning them removes successes.
        return (scoreB(c)
                - 3.0 * max(0, c["d_bridge"])
                - 1.5 * c["spiro"]
                - 1.0 * max(0, c["stereo"] - p0["stereo"])
                + 2.0 * max(0, c["d_arom"]))

    out = dict(pre=p0, n_closures=len(cands), arms={})
    for name, fn in (("A_raw_rtheta", scoreA), ("B_plus_props", scoreB),
                     ("C_plus_topology", scoreC)):
        best = max(cands, key=fn)
        r = rollout(enum, apply_h, is_valid, best["smi"], "restate",
                    int(job["restate_len"]), np.random.default_rng(job["seed_rng"] + 1),
                    prefer_fn=None)
        end = prof(r.endpoint) or best
        end["feasible"] = bool(end["qed"] >= 0.6 and end["sa"] <= 4
                               and end["sim"] >= delta)
        end["closure"] = {k: best[k] for k in
                          ("sa", "qed", "bridge", "arom", "rings", "new_ring")}
        out["arms"][name] = end
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=int(1 * 1024),
              timeout=3 * 60 * 60, retries=0,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive_closure_arms(payload: dict, n_states: int = 60):
    """Run A/B/C over many independent pre-cyclization states, then report."""
    import collections, statistics as st
    jobs = [dict(seed=payload["seed"], delta=payload["delta"], grow_len=4,
                 restate_len=2, seed_rng=20260825 + 7919 * i)
            for i in range(int(n_states))]
    res = [r for r in closure_arms.map(jobs) if r and not r.get("error")]
    print(f"{len(res)}/{n_states} states produced a legal closure", flush=True)
    summary = {}
    for arm in ("A_raw_rtheta", "B_plus_props", "C_plus_topology"):
        v = [r["arms"][arm] for r in res if arm in r.get("arms", {})]
        if not v: continue
        sizes = collections.Counter(s for x in v for s in x["sizes"])
        summary[arm] = dict(
            n=len(v),
            feasible=sum(1 for x in v if x["feasible"]),
            yield_pct=round(100 * sum(1 for x in v if x["feasible"]) / len(v), 1),
            median_sa=round(st.median(x["sa"] for x in v), 2),
            median_qed=round(st.median(x["qed"] for x in v), 3),
            bridgehead_rate=round(sum(1 for x in v if x["bridge"] > 0) / len(v), 3),
            median_arom=st.median(x["arom"] for x in v),
            median_rings=st.median(x["rings"] for x in v),
            ring_sizes=dict(sorted(sizes.items())))
        s2 = summary[arm]
        print(f"{arm:18s} yield={s2['yield_pct']:5.1f}%  medSA={s2['median_sa']:.2f}  "
              f"bridge_rate={s2['bridgehead_rate']:.2f}  medArom={s2['median_arom']}  "
              f"rings={s2['ring_sizes']}", flush=True)
    ck = Path("/artifacts/macro_basin") / f"closure_arms_{payload['cell']}.json"
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps(dict(summary=summary, detail=res), indent=1))
    artifact_volume.commit()
    print(f"wrote {ck}", flush=True)
    return summary


@app.local_entrypoint()
def closure_test(cell: str = "parp1_s0_d0.4", n_states: int = 60):
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    tgt, si, dl = cell.rsplit("_", 2)
    payload = dict(cell=cell, seed=seeds[f"{tgt}_{si}"]["smiles"],
                   delta=float(dl[1:]))
    fn = _m.Function.from_name("macro-basin", "drive_closure_arms")
    h = fn.spawn(payload, n_states=n_states)
    print(f"  SPAWNED closure A/B/C: {h.object_id}")


@app.function(image=image, cpu=(1.0, 1.0), memory=int(1 * 1024),
              timeout=3 * 60 * 60, retries=0,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive_append(payload: dict, realizations: int = 40):
    """APPEND vs annulating CONTROL at matched atom budget."""
    import collections, statistics as st
    progs = payload.get("programs") or APPEND_PROGRAMS
    jobs = [dict(program=prog, seed=payload["seed"], winners=payload["winners"],
                 prefer=False, seed_rng=20260825 + 7919 * (pi * 10000 + r))
            for pi, prog in enumerate(progs) for r in range(int(realizations))]
    print(f"mapping ALL {len(jobs)} jobs at once (was {int(realizations)} at a "
          f"time, which capped fan-out at half the container limit)", flush=True)
    out = [r for r in run_program.map(jobs) if r and not r.get("failed")]
    for pi, prog in enumerate(progs):
        res = [r for r in out if r["program"] == prog]
        key = " -> ".join(f"{m}{L}" for m, L in prog)
        if res:
            sysc = collections.Counter(r.get("systems", 1) for r in res)
            feas = [r for r in res if r["qed"] >= 0.6 and r["sa"] <= 4]
            print(f"{key:48s} systems={dict(sorted(sysc.items()))} "
                  f"medSA={st.median(r['sa'] for r in res):.2f} "
                  f"medHeavy={st.median(r['heavy'] for r in res):.0f} "
                  f"medArom={st.median(r['arom'] for r in res):.0f} "
                  f"feas={len(feas)}/{len(res)}", flush=True)
    ck = Path("/artifacts/macro_basin") / f"append_{payload['cell']}.json"
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps(out, indent=1))
    artifact_volume.commit()
    return out


@app.local_entrypoint()
def append_test(cell: str = "parp1_s0_d0.4", realizations: int = 40,
                controls_only: bool = False):
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    wins = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    tgt, si, dl = cell.rsplit("_", 2)
    payload = dict(cell=cell, seed=seeds[f"{tgt}_{si}"]["smiles"],
                   winners=[w["smiles"] for w in wins[cell]["winners"]])
    if cells == "__scaffold__":
        pass
    if controls_only:
        # the two matched controls plus the old baseline; the append arms are
        # already banked, and the 22-edit double program is the least
        # informative and most expensive of the six.
        payload["programs"] = [[["grow", 8], ["cyclize", 1], ["restate", 2]],
                               [["grow", 10], ["cyclize", 1], ["restate", 2]],
                               [["grow", 4], ["cyclize", 1], ["restate", 2]]]
        payload["cell"] = cell + "_controls"
    fn = _m.Function.from_name("macro-basin", "drive_append")
    h = fn.spawn(payload, realizations=realizations)
    print(f"  SPAWNED append test: {h.object_id}")


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=60 * 60,
              retries=0, enable_memory_snapshot=True,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def element_audit(payload: dict) -> dict:
    """Does R_theta prefer carbon, or does MY epsilon floor inject the halogens?

    Measured on 80 grows: carbon fraction 0.33, longest carbon run median 1,
    P(run >= 6) = 0. A benzene pendant is therefore impossible. Two candidate
    causes, and they imply completely different fixes:

      R_theta itself is halogen-heavy   -> `grow` is semantically too weak and
                                           needs an extendable-atom restriction
      the epsilon=0.15 uniform floor    -> MY bug: the term added to reach the
                                           rank-137 tail is overriding the
                                           learned chemistry prior during growth

    So this reports the element distribution under R_theta alone AND under the
    mixture the sampler actually draws from, on the same states.
    """
    import numpy as np, collections
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import (smiles_to_molecular_graph,
                                                 IDX_TO_ELEMENT)
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import (proposal_support,
                                                 macro_action_distribution)
    rt = _runtime(); model = rt["model"]
    out = {}
    for label, smi in payload["states"]:
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        fams = [m.executor_rule_name for m in law.marks]
        probs = np.array([m.probability for m in law.marks], float)
        vocab = getattr(model, "atom_vocabulary", None) or getattr(
            rt["system"], "atom_vocabulary", None)

        # GROUND TRUTH, not a decode. The first version read
        # IDX_TO_ELEMENT[vocab.element_of(atom_type)] and reported ZERO carbon
        # inserts at every state -- contradicted by the composition audit, which
        # found 236 carbons among actually-grown atoms. It failed by returning
        # plausible-but-wrong element names rather than visibly. So: apply the
        # action and read the element off the product.
        from rdkit import Chem as _C
        from compose_v4.experiments.production_successor_kernel import (
            canonical_state_key as _ck)
        base = _C.MolFromSmiles(smi)
        base_n = base.GetNumAtoms() if base else 0

        def elem_of(mark):
            if getattr(mark.action, "atom_type", None) is None:
                return None
            try:
                y = _ck(rt["system"].apply(st, mark.executor_rule_name, mark.action))
            except Exception:
                return None
            m2 = _C.MolFromSmiles(y) if y else None
            if m2 is None or m2.GetNumAtoms() != base_n + 1:
                return None
            b = collections.Counter(a.GetSymbol() for a in base.GetAtoms())
            n = collections.Counter(a.GetSymbol() for a in m2.GetAtoms())
            d = n - b
            return next(iter(d)) if len(d) == 1 else None

        ins = [j for j in range(len(fams)) if fams[j] == "atom_insert"]
        raw = collections.Counter(); rawm = collections.Counter()
        for j in ins:
            e = elem_of(law.marks[j])
            if e is None: continue
            raw[e] += 1
            rawm[e] += float(probs[j])
        tot = sum(rawm.values()) or 1.0
        r_theta = {k: round(v / tot, 4) for k, v in rawm.most_common()}

        support = proposal_support(fams, probs)
        clean = np.ones(len(probs), dtype=bool)
        idx, q = macro_action_distribution(fams, probs, support, "grow", clean,
                                           temperature=2.0, epsilon=0.15)
        mix = collections.Counter()
        for k, jj in enumerate(idx):
            e = elem_of(law.marks[int(jj)])
            if e is not None: mix[e] += float(q[k])
        mtot = sum(mix.values()) or 1.0
        sampler = {k: round(v / mtot, 4) for k, v in mix.most_common()}
        HAL = {"F", "Cl", "Br", "I"}
        out[label] = dict(
            n_insert_actions=len(ins), n_distinct_elements=len(raw),
            action_counts=dict(raw.most_common()),
            R_theta=r_theta, sampler_q=sampler,
            R_theta_carbon=r_theta.get("C", 0.0),
            sampler_carbon=sampler.get("C", 0.0),
            R_theta_halogen=round(sum(v for k, v in r_theta.items() if k in HAL), 4),
            sampler_halogen=round(sum(v for k, v in sampler.items() if k in HAL), 4))
        o = out[label]
        print(f"{label}: R_theta C={o['R_theta_carbon']:.3f} hal={o['R_theta_halogen']:.3f} "
              f"| sampler C={o['sampler_carbon']:.3f} hal={o['sampler_halogen']:.3f}",
              flush=True)
    ck = Path("/artifacts/macro_basin") / "element_audit.json"
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps(out, indent=1))
    artifact_volume.commit()
    return out


@app.local_entrypoint()
def elements(cell: str = "parp1_s0_d0.4"):
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    tgt, si, _dl = cell.rsplit("_", 2)
    states = [["seed", seeds[f"{tgt}_{si}"]["smiles"]]]
    # plus real grown states from the banked append run
    ap = root / "diagnostics/append_arms_parp1_s0_d0.4.json"
    if ap.exists():
        A = json.loads(ap.read_text())
        seen = set()
        for r in A:
            tr = r.get("trace") or []
            if tr and tr[0].get("smi") and tr[0]["smi"] not in seen:
                seen.add(tr[0]["smi"])
                states.append([f"grown{len(states)}", tr[0]["smi"]])
            if len(states) >= 4: break
    fn = _m.Function.from_name("macro-basin", "element_audit")
    res = fn.remote(dict(states=states))
    print(f"\n{'state':10s} {'insert_actions':>14s} {'R_theta(C)':>11s} {'q(C)':>7s} "
          f"{'R_theta(hal)':>13s} {'q(hal)':>8s}")
    for k, v in res.items():
        print(f"{k:10s} {v['n_insert_actions']:14d} {v['R_theta_carbon']:11.3f} "
              f"{v['sampler_carbon']:7.3f} {v['R_theta_halogen']:13.3f} "
              f"{v['sampler_halogen']:8.3f}")
    for k, v in res.items():
        print(f"\n{k} R_theta : {v['R_theta']}")
        print(f"{k} sampler : {v['sampler_q']}")
    (root / "diagnostics/element_audit.json").write_text(json.dumps(res, indent=1))


@app.function(image=image, cpu=(1.0, 1.0), memory=int(1 * 1024),
              timeout=2 * 60 * 60, retries=0,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive_eps_ab(payload: dict, n: int = 60):
    """epsilon=0 vs epsilon=0.15 on grow, PAIRED by RNG seed.

    Same seed molecule, same rng stream, same macro, same length -- the ONLY
    difference is the uniform exploration floor. Measures the realized endpoint
    molecules directly (element counts vs the starting molecule), so no
    atom_type decode and no MCS is involved; two attempts at decoding
    atom_type both produced answers contradicted by the endpoint molecules,
    which are ground truth.
    """
    import collections, statistics as st
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    seed = payload["seed"]
    sm = Chem.MolFromSmiles(seed)
    sc = collections.Counter(a.GetSymbol() for a in sm.GetAtoms())
    prog = [["grow", 8]]
    jobs = []
    for i in range(int(n)):
        for tag, eps in (("eps0", {"grow": 0.0, "_default": 0.15}),
                         ("eps15", {"grow": 0.15, "_default": 0.15})):
            jobs.append(dict(program=prog, seed=seed, winners=payload["winners"],
                             prefer=False, epsilon=eps,
                             seed_rng=20260825 + 7919 * i,   # PAIRED
                             _tag=tag))
    res = []
    for j, r in zip(jobs, run_program.map(jobs)):
        if r and not r.get("failed"):
            r["_tag"] = j["_tag"]; res.append(r)

    def longest_c_run(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None: return 0
        cs = {a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "C"}
        adj = {a: [x.GetIdx() for x in m.GetAtomWithIdx(a).GetNeighbors()
                   if x.GetIdx() in cs] for a in cs}
        best = 0
        def dfs(u, seen):
            nonlocal best; best = max(best, len(seen))
            for v in adj[u]:
                if v not in seen: dfs(v, seen | {v})
        for a in cs: dfs(a, {a})
        return best

    summary = {}
    for tag in ("eps0", "eps15"):
        v = [r for r in res if r["_tag"] == tag]
        if not v: continue
        dc, dh, runs = [], [], []
        for r in v:
            m = Chem.MolFromSmiles(r["smi"])
            if m is None: continue
            c = collections.Counter(a.GetSymbol() for a in m.GetAtoms())
            dc.append(c["C"] - sc["C"])
            dh.append(sum(c[x] for x in ("F", "Cl", "Br", "I")))
            runs.append(longest_c_run(r["smi"]))
        summary[tag] = dict(
            n=len(dc), dC_median=st.median(dc), dC_mean=round(st.mean(dc), 2),
            halogens_median=st.median(dh),
            c_run_median=st.median(runs),
            frac_run_ge4=round(sum(1 for x in runs if x >= 4) / len(runs), 3),
            frac_run_ge6=round(sum(1 for x in runs if x >= 6) / len(runs), 3),
            sa_median=round(st.median(r["sa"] for r in v), 2),
            qed_median=round(st.median(r["qed"] for r in v), 3),
            dC_dist=dict(sorted(collections.Counter(dc).items())))
        s2 = summary[tag]
        print(f"{tag:6s} dC={s2['dC_median']:+.0f} (mean {s2['dC_mean']:+.2f})  "
              f"hal={s2['halogens_median']:.0f}  Crun={s2['c_run_median']:.0f}  "
              f"run>=4 {100*s2['frac_run_ge4']:.0f}%  run>=6 {100*s2['frac_run_ge6']:.0f}%  "
              f"SA={s2['sa_median']:.2f}", flush=True)
    ck = Path("/artifacts/macro_basin") / "eps_ab.json"
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps(dict(summary=summary, seed=seed), indent=1))
    artifact_volume.commit()
    return summary


@app.local_entrypoint()
def eps_ab(cell: str = "parp1_s0_d0.4", n: int = 60):
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    wins = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    tgt, si, _dl = cell.rsplit("_", 2)
    payload = dict(seed=seeds[f"{tgt}_{si}"]["smiles"],
                   winners=[w["smiles"] for w in wins[cell]["winners"]])
    fn = _m.Function.from_name("macro-basin", "drive_eps_ab")
    res = fn.remote(payload, n=n)
    print(json.dumps(res, indent=1))
    (root / "diagnostics/eps_ab.json").write_text(json.dumps(res, indent=1))


@app.local_entrypoint()
def scaffold_test(cell: str = "parp1_s0_d0.4", realizations: int = 40):
    """SCAFFOLD_EXTEND + APPEND_SYSTEM vs matched controls."""
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    wins = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    tgt, si, _dl = cell.rsplit("_", 2)
    payload = dict(cell=cell + "_scaffold", seed=seeds[f"{tgt}_{si}"]["smiles"],
                   winners=[w["smiles"] for w in wins[cell]["winners"]],
                   programs=SCAFFOLD_PROGRAMS)
    fn = _m.Function.from_name("macro-basin", "drive_append")
    h = fn.spawn(payload, realizations=realizations)
    print(f"  SPAWNED scaffold test: {h.object_id}")


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=60 * 60,
              retries=0, enable_memory_snapshot=True,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def insert_support(payload: dict) -> dict:
    """Of every LEGAL atom_insert at a state, what element does it add?

    Applies each action and diffs the element counts. No atom_type decode (two
    attempts at that produced answers the endpoint molecules contradicted), no
    MCS. Separates three possibilities that the composition audit could not:

      support is carbon-poor            -> the fiber cannot offer carbon; a
                                           support restriction will not help
      support is carbon-rich, mass low  -> R_theta deprioritises carbon
      support carbon-rich, mass high    -> something downstream drops it
    """
    import numpy as np, collections
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import proposal_support
    rt = _runtime(); model, system = rt["model"], rt["system"]
    out = {}
    for label, smi in payload["states"]:
        base = Chem.MolFromSmiles(smi)
        bc = collections.Counter(a.GetSymbol() for a in base.GetAtoms())
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        fams = [m.executor_rule_name for m in law.marks]
        probs = np.array([m.probability for m in law.marks], float)
        sup = set(int(i) for i in proposal_support(fams, probs))
        n_by_elem = collections.Counter()          # how many LEGAL actions
        mass_by_elem = collections.Counter()       # R_theta mass on them
        n_in_support = collections.Counter()
        for j in range(len(fams)):
            if fams[j] != "atom_insert":
                continue
            try:
                y = canonical_state_key(system.apply(st, fams[j], law.marks[j].action))
            except Exception:
                continue
            m2 = Chem.MolFromSmiles(y) if y else None
            if m2 is None or m2.GetNumAtoms() != base.GetNumAtoms() + 1:
                continue
            d = collections.Counter(a.GetSymbol() for a in m2.GetAtoms()) - bc
            if len(d) != 1:
                continue
            e = next(iter(d))
            n_by_elem[e] += 1
            mass_by_elem[e] += float(probs[j])
            if j in sup:
                n_in_support[e] += 1
        tot_n = sum(n_by_elem.values()) or 1
        tot_m = sum(mass_by_elem.values()) or 1.0
        out[label] = dict(
            legal_actions=dict(n_by_elem.most_common()),
            frac_legal_carbon=round(n_by_elem["C"] / tot_n, 4),
            frac_mass_carbon=round(mass_by_elem["C"] / tot_m, 4),
            in_support=dict(n_in_support.most_common()),
            mass=dict((k, round(v / tot_m, 4)) for k, v in mass_by_elem.most_common()))
        o = out[label]
        print(f"{label}: legal_C={o['frac_legal_carbon']:.3f} "
              f"mass_C={o['frac_mass_carbon']:.3f} legal={o['legal_actions']}",
              flush=True)
    ck = Path("/artifacts/macro_basin") / "insert_support.json"
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps(out, indent=1))
    artifact_volume.commit()
    return out


@app.local_entrypoint()
def support(cell: str = "parp1_s0_d0.4"):
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    tgt, si, _dl = cell.rsplit("_", 2)
    states = [["seed", seeds[f"{tgt}_{si}"]["smiles"]]]
    ap = root / "diagnostics/append_arms_parp1_s0_d0.4.json"
    if ap.exists():
        for r in json.loads(ap.read_text())[:2]:
            tr = r.get("trace") or []
            if tr and tr[0].get("smi"):
                states.append([f"grown{len(states)}", tr[0]["smi"]])
    fn = _m.Function.from_name("macro-basin", "insert_support")
    res = fn.remote(dict(states=states))
    print(f"\n{'state':9s} {'legal C%':>9s} {'mass C%':>9s}   legal actions by element")
    for k, v in res.items():
        print(f"{k:9s} {100*v['frac_legal_carbon']:8.1f}% {100*v['frac_mass_carbon']:8.1f}%   "
              f"{v['legal_actions']}")
    print()
    for k, v in res.items():
        print(f"{k} R_theta mass: {v['mass']}")
    (root / "diagnostics/insert_support.json").write_text(json.dumps(res, indent=1))


@app.local_entrypoint()
def composition_test(cell: str = "parp1_s0_d0.4", realizations: int = 40):
    """grow vs scaffold_extend(mixed) vs scaffold_extend(carbon_rich), matched."""
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    wins = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    tgt, si, _dl = cell.rsplit("_", 2)
    payload = dict(cell=cell + "_composition", seed=seeds[f"{tgt}_{si}"]["smiles"],
                   winners=[w["smiles"] for w in wins[cell]["winners"]],
                   programs=COMPOSITION_PROGRAMS)
    fn = _m.Function.from_name("macro-basin", "drive_append")
    h = fn.spawn(payload, realizations=realizations)
    print(f"  SPAWNED composition test: {h.object_id}")


@app.local_entrypoint()
def state_test(cell: str = "parp1_s0_d0.4", realizations: int = 50):
    """Same precursor and topology; saturated vs aromatic endpoint intent."""
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    wins = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    tgt, si, _dl = cell.rsplit("_", 2)
    payload = dict(cell=cell + "_state", seed=seeds[f"{tgt}_{si}"]["smiles"],
                   winners=[w["smiles"] for w in wins[cell]["winners"]],
                   programs=STATE_PROGRAMS)
    fn = _m.Function.from_name("macro-basin", "drive_append")
    h = fn.spawn(payload, realizations=realizations)
    print(f"  SPAWNED state test: {h.object_id}")


@app.local_entrypoint()
def locality_test(cell: str = "parp1_s0_d0.4", realizations: int = 24):
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    wins = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    tgt, si, _dl = cell.rsplit("_", 2)
    payload = dict(cell=cell + "_locality", seed=seeds[f"{tgt}_{si}"]["smiles"],
                   winners=[w["smiles"] for w in wins[cell]["winners"]],
                   programs=LOCALITY_PROGRAMS)
    fn = _m.Function.from_name("macro-basin", "drive_append")
    h = fn.spawn(payload, realizations=realizations)
    print(f"  SPAWNED locality test: {h.object_id}")


@app.local_entrypoint()
def regularity_test(cell: str = "parp1_s0_d0.4", realizations: int = 12):
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    wins = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    tgt, si, _dl = cell.rsplit("_", 2)
    payload = dict(cell=cell + "_regularity", seed=seeds[f"{tgt}_{si}"]["smiles"],
                   winners=[w["smiles"] for w in wins[cell]["winners"]],
                   programs=REGULARITY_PROGRAMS)
    fn = _m.Function.from_name("macro-basin", "drive_append")
    h = fn.spawn(payload, realizations=realizations)
    print(f"  SPAWNED regularity test: {h.object_id}")


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024),
              timeout=4 * 60 * 60, retries=0,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive_topology_dock(payload: dict, realizations: int = 200,
                        per_arm_dock: int = 15):
    """Which EXISTING topology gives the best docking distribution?

    Locality and exact T4 feasibility are frozen. No new macros, no chemistry
    rules. The only variable is which existing closure topology the program
    uses. Docking is applied AFTER endpoint feasibility and PER ARM, so the
    oracle spend buys a comparison rather than more samples of one distribution.

    Motivating observation: 10 blind docking calls on the localized program
    reached -10.2, where the old full search needed 1000 calls to reach -10.5 --
    but all 10 molecules were 27 heavy atoms / 4 rings / 2 aromatic, i.e. one
    motif found ten ways. Scaling that would improve the extreme value and teach
    us nothing about where the binding signal lives.
    """
    import collections, statistics as st
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED, RDConfig
    import sys, os
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.gates.med_chem_gate import t4_feasible
    import modal as _m

    ARMS = payload.get("arms") or [
        ("close_disjoint", [("local_extend:2", 8), ("close_disjoint", 1), ("restate", 2)]),
        ("append_system",  [("local_extend:2", 8), ("append_system", 1), ("restate", 2)]),
        ("annulate",       [("local_extend:2", 8), ("annulate", 1), ("restate", 2)])]
    delta = float(payload["delta"]); seed = payload["seed"]
    jobs = [dict(program=prog, seed=seed, winners=payload["winners"], prefer=False,
                 seed_rng=20260826 + 7919 * (ai * 100000 + r), _arm=name)
            for ai, (name, prog) in enumerate(ARMS) for r in range(int(realizations))]
    print(f"generating {len(jobs)} rollouts across {len(ARMS)} topologies", flush=True)
    gen = []
    for j, r in zip(jobs, run_program.map(jobs)):
        if r and not r.get("failed"):
            r["_arm"] = j["_arm"]; gen.append(r)

    feas = collections.defaultdict(list)
    seen = set()
    for r in gen:
        smi = r["smi"]
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        can = Chem.MolToSmiles(m)
        if can in seen or not t4_feasible(smi, seed, delta):
            continue
        seen.add(can)
        feas[r["_arm"]].append(dict(smi=can, qed=round(float(QED.qed(m)), 3),
                                    sa=round(float(sascorer.calculateScore(m)), 2),
                                    heavy=m.GetNumHeavyAtoms(), rings=r["rings"],
                                    arom=r["arom"], basin=r["basin"]))
    for a, v in feas.items():
        print(f"  {a}: {len(v)} T4-feasible", flush=True)

    dock = _m.Function.from_name("genmol-t4-gate4", "dock_frontier")
    djobs, tags = [], []
    for arm, v in feas.items():
        cand = v[: int(per_arm_dock)]
        if not cand:
            continue
        djobs.append(dict(smiles=[c["smi"] for c in cand], target=payload["target"],
                          idx=payload["idx"], delta=delta, beta=4.0,
                          budget=len(cand), tag=f"topo_{arm}"))
        tags.append(arm)
    print(f"docking {sum(len(j['smiles']) for j in djobs)} molecules "
          f"across {len(djobs)} arms", flush=True)
    out = {}
    for arm, res in zip(tags, dock.map(djobs, order_outputs=True)):
        pairs = sorted(res.get("top") or [], key=lambda p: p[1])
        prof = {c["smi"]: c for c in feas[arm]}
        out[arm] = dict(n_feasible=len(feas[arm]), n_docked=len(pairs),
                        best=pairs[0][1] if pairs else None,
                        median=st.median([p[1] for p in pairs]) if pairs else None,
                        top=[dict(smi=sm, ds=ds, **{k: v for k, v in
                                                    prof.get(sm, {}).items() if k != "smi"})
                             for sm, ds in pairs[:5]])
        print(f"  {arm}: best={out[arm]['best']} median={out[arm]['median']}", flush=True)
    ck = Path("/artifacts/macro_basin") / f"topology_dock_{payload['cell']}.json"
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps(out, indent=1))
    artifact_volume.commit()
    return out


@app.local_entrypoint()
def topology_dock(cell: str = "parp1_s0_d0.4", realizations: int = 200,
                  per_arm_dock: int = 15):
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    wins = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    tgt, si, dl = cell.rsplit("_", 2)
    sd = seeds[f"{tgt}_{si}"]
    alt = f"{tgt}_{si}_d0.4" if cell not in wins else cell
    ws = [w["smiles"] for w in wins.get(alt, {}).get("winners", [])] or [sd["smiles"]]
    payload = dict(cell=cell, seed=sd["smiles"], target=sd["target"], idx=sd["idx"],
                   delta=float(dl[1:]), winners=ws)
    fn = _m.Function.from_name("macro-basin", "drive_topology_dock")
    h = fn.spawn(payload, realizations=realizations, per_arm_dock=per_arm_dock)
    print(f"  SPAWNED topology dock: {h.object_id}")


@app.local_entrypoint()
def lipo_dock(cell: str = "parp1_s0_d0.4", realizations: int = 200,
              per_arm_dock: int = 15):
    """Locality AND carbon-rich composition together, vs each alone."""
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    wins = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    tgt, si, dl = cell.rsplit("_", 2)
    sd = seeds[f"{tgt}_{si}"]
    payload = dict(cell=cell + "_lipo", seed=sd["smiles"], target=sd["target"],
                   idx=sd["idx"], delta=float(dl[1:]),
                   winners=[w["smiles"] for w in wins[cell]["winners"]],
                   arms=[("locality_only",
                          [("local_extend:2", 8), ("close_disjoint", 1), ("restate", 2)]),
                         ("locality_carbon",
                          [("local_extend:2:carbon_rich", 8), ("close_disjoint", 1), ("restate", 2)]),
                         ("locality_mixed",
                          [("local_extend:2:mixed", 8), ("close_disjoint", 1), ("restate", 2)])])
    fn = _m.Function.from_name("macro-basin", "drive_topology_dock")
    h = fn.spawn(payload, realizations=realizations, per_arm_dock=per_arm_dock)
    print(f"  SPAWNED lipo dock: {h.object_id}")



# BUILD_RING_SYSTEM is a pure function of
#   (smiles, size, topology, composition, ring_state, anchor_rank)
# -- it selects the top-ranked descriptor at every step and carries no RNG, so
# the same key always yields the same product. Across a 640-episode sentinel the
# same few trajectories are otherwise recomputed hundreds of times at 8 model
# enumerations each; from the seed there are only 6 distinct first-macro
# outcomes in total.
#
# A modal.Dict rather than a Volume file: 80 containers write concurrently, and
# a shared JSON file would race. Encoding it once and reusing it is the same
# lesson as the embeddings that were recomputed until a stopped job destroyed
# 93.5 core-hours.
_MACRO_MEMO = None


def _macro_memo():
    global _MACRO_MEMO
    if _MACRO_MEMO is None:
        import modal as _m
        _MACRO_MEMO = _m.Dict.from_name("build-ring-system-memo",
                                        create_if_missing=True)
    return _MACRO_MEMO


def _build_ring_cached(model, system, smiles, *, size, topology="pendant",
                       composition="carbon_rich", ring_state="aromatic",
                       anchor_rank=0, anchors=None, refine=0, refine_seed=0):
    """Memoised BUILD_RING_SYSTEM. Returns the macro's result dict.

    `refine` is REFINE_RING: that many ring-scoped restate steps applied to the
    ring this call just built, inside the builder so slot provenance survives.
    It is part of the MEMO KEY together with `refine_seed` -- refinement is
    stochastic, and keying without the seed would freeze one draw and hand every
    particle in the ensemble the identical refined ring.
    """
    import numpy as np
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import (smiles_to_molecular_graph,
                                                 molecular_graph_to_smiles)
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import build_ring_system_exact
    from compose_v4.gates.med_chem_gate import is_executable

    akey = "" if anchors is None else ",".join(str(int(a)) for a in sorted(anchors))
    key = (f"{smiles}|{size}|{topology}|{composition}|{ring_state}|{anchor_rank}"
           f"|{akey}|rf{int(refine)}s{int(refine_seed)}")
    memo = _macro_memo()
    try:
        hit = memo.get(key)
        if hit is not None:
            return hit
    except Exception:
        pass

    _fc = {}

    def _to_smiles(st):
        try: return molecular_graph_to_smiles(st)
        except Exception: return None

    def _enum_full(st):
        k = _to_smiles(st)
        if k not in _fc:
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            _fc[k] = ([m.executor_rule_name for m in law.marks],
                      [m.action for m in law.marks],
                      np.array([m.probability for m in law.marks], float))
        return _fc[k]

    def _apply(st, j):
        fams, acts, _p = _enum_full(st)
        try: return system.apply(st, fams[j], acts[j])
        except Exception: return None

    st0 = pad_molecular_graph(smiles_to_molecular_graph(smiles), CANONICAL_SLOTS)
    import numpy as _np
    # FAST PATH. 98.9-99.6% of a ring build's wall time was full-law
    # enumeration (diagnostics/enum_profile.json) to rank a handful of
    # descriptors the deterministic builder had already compiled. The partial
    # enumerator returns weights proportional to the eager law over only the
    # families each phase consumes, which induces the identical ordering --
    # qualified 6/6 identical (status, smiles) at 6.39x aggregate
    # (diagnostics/ring_fast_ab.json). Set COMPOSE_RING_EAGER=1 to force the
    # old path.
    import os as _os
    _enum_used = _enum_full
    if not _os.environ.get("COMPOSE_RING_EAGER"):
        try:
            from compose_v4.control.ring_fastpath import ring_phase_enumerator
            from compose_v4.experiments.hphi_lazy_helpers import make_helpers
            _hlp = make_helpers(model, time_point=float(TIME_POINT),
                                canonical_slots=CANONICAL_SLOTS)
            _enum_used = ring_phase_enumerator(model, _hlp, float(TIME_POINT),
                                               int(size))
        except Exception as _exc:
            print(f"  ring fastpath unavailable ({type(_exc).__name__}); eager",
                  flush=True)
            _enum_used = _enum_full

    def _apply_fast(st, j):
        fams, acts, _p = _enum_used(st)
        try: return system.apply(st, fams[j], acts[j])
        except Exception: return None

    r = build_ring_system_exact(_enum_used, _apply_fast, _to_smiles, is_executable,
                                st0, size=int(size), topology=topology,
                                composition=composition, state=ring_state,
                                anchor_rank=int(anchor_rank), anchors=anchors,
                                refine=int(refine),
                                refine_rng=_np.random.default_rng(int(refine_seed)))
    r = {k: v for k, v in r.items() if isinstance(v, (str, int, float, bool, list, type(None)))}
    try:
        memo[key] = r
    except Exception:
        pass
    return r


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=2 * 60 * 60,
              retries=0, max_containers=80, enable_memory_snapshot=True,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def search_step(job: dict) -> dict:
    """One particle takes ONE chosen macro action. The controller picks the
    action; R_theta picks the primitive realisation inside it."""
    import numpy as np
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import rollout, contract_for, local_extend
    from compose_v4.control.constrained_search import margins
    from compose_v4.gates.med_chem_gate import is_executable

    rt = _runtime(); model, system = rt["model"], rt["system"]
    _cache = {}

    def enumerate_fn(smi):
        if smi not in _cache:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            _cache[smi] = ([m.executor_rule_name for m in law.marks],
                           np.array([m.probability for m in law.marks], float),
                           [(st, m.executor_rule_name, m.action) for m in law.marks])
        return _cache[smi]

    def apply_fn(smi, h):
        st, rule, action = h
        try: return canonical_state_key(system.apply(st, rule, action))
        except Exception: return None

    macro, L = job["macro"], int(job["length"])
    cur = job["smiles"]
    rng = np.random.default_rng(job["seed_rng"])
    if macro == "STOP" or L == 0:
        m = margins(cur, job["seed"], float(job["delta"]))
        return dict(smiles=cur, macro=macro, length=L, halted="stop",
                    shortfall=m.shortfall, feasible=m.feasible)
    if macro.startswith("build_ring_system"):
        rr = _build_ring_cached(model, system, cur, size=L,
                                topology=job.get("topology", "pendant"),
                                composition=job.get("composition", "carbon_rich"),
                                ring_state=job.get("ring_state", "aromatic"),
                                anchor_rank=int(job.get("anchor_rank",
                                                        rng.integers(0, 6))))
        out = rr.get("smiles") or cur
        m = margins(out, job["seed"], float(job["delta"]))
        return dict(smiles=out, macro=macro, length=L,
                    executed=len(rr.get("trace", []) or []), halted=rr.get("stage"),
                    status=rr.get("status"), trace=rr.get("trace"),
                    sys_gain=rr.get("sys_gain"), arom_gain=rr.get("arom_gain"),
                    shortfall=m.shortfall, feasible=m.feasible,
                    sim=round(m.sim, 3), qed=round(m.qed, 3), sa=round(m.sa, 2))
    if macro == "local_extend":
        pref = local_extend(cur, job["seed"], max_anchors=int(job.get("anchors", 2)))
        eff = "grow"
    elif macro in ("close_disjoint",):
        from compose_v4.control.macro_engine import disjoint_closure_only
        pref = disjoint_closure_only(cur); eff = "cyclize"
    elif macro in ("annulate", "shrink", "decorate", "restate"):
        pref = contract_for(macro, cur) if macro != "restate" else None
        eff = {"annulate": "cyclize", "shrink": "shrink",
               "decorate": "grow", "restate": "restate"}[macro]
    else:
        pref = None; eff = macro
    r = rollout(enumerate_fn, apply_fn, is_executable, cur, eff, L, rng,
                prefer_fn=pref)
    m = margins(r.endpoint, job["seed"], float(job["delta"]))
    return dict(smiles=r.endpoint, macro=macro, length=L, executed=r.length,
                halted=r.halted, shortfall=m.shortfall, feasible=m.feasible,
                sim=round(m.sim, 3), qed=round(m.qed, 3), sa=round(m.sa, 2))


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024),
              timeout=4 * 60 * 60, retries=0,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive_constrained(payload: dict, n_particles: int = 64, rounds: int = 6,
                      dock_per_round: int = 0):
    """Goal-conditioned search. NO fixed program, NO docking, NO IVG info.

    Gate: >= 20 distinct T4-feasible endpoints per cell. If a plain CEM over the
    existing vocabulary cannot manage that where the fixed recipe scored 0/300,
    the controller is the problem, not the chemistry.
    """
    import numpy as np
    from compose_v4.control.constrained_search import ACTION_SPACE, CEMController
    rng = np.random.default_rng(20260826)
    ctrl = CEMController(len(ACTION_SPACE), rng)
    seed, delta = payload["seed"], float(payload["delta"])
    parts = [seed] * int(n_particles)
    archive, seen = [], set()
    import modal as _m
    dock = _m.Function.from_name("genmol-t4-gate4", "dock_frontier")
    from rdkit import Chem as _C
    seed_can = _C.MolToSmiles(_C.MolFromSmiles(seed))
    docked_smi, by_smiles_action = {}, {}
    for rd in range(int(rounds)):
        idx = ctrl.sample(len(parts))
        jobs = []
        for p, ai in zip(parts, idx):
            macro, L = ACTION_SPACE[int(ai)]
            jobs.append(dict(smiles=p, seed=seed, delta=delta, macro=macro,
                             length=L, anchors=min(L, 4),
                             seed_rng=int(rng.integers(0, 2**31)), _ai=int(ai)))
        res = list(search_step.map(jobs))
        used, scores, nxt = [], [], []
        for j, r in zip(jobs, res):
            if not r:
                continue
            used.append(j["_ai"]); scores.append(r["shortfall"])
            nxt.append(r["smiles"])
            if r.get("feasible") and r["smiles"] not in seen:
                # the unchanged seed is feasible by construction; dock it once
                # as a baseline, never let it consume repeated oracle calls
                try:
                    if _C.MolToSmiles(_C.MolFromSmiles(r["smiles"])) == seed_can:
                        continue
                except Exception:
                    pass
                seen.add(r["smiles"]); archive.append(r)
        # THE OBJECTIVE IS DOCKING. Feasibility is a hard filter, not a term to
        # optimise -- optimising shortfall alone has the trivial optimum x = x0,
        # which is exactly what the previous run found (shrink1/STOP0 dominant,
        # >50% of every archive within one heavy atom of the seed, the
        # unmodified seed itself in all three archives).
        if int(dock_per_round) and archive:
            fresh = [a for a in archive if a["smiles"] not in docked_smi][:int(dock_per_round)]
            if fresh:
                dj = [dict(smiles=[a["smiles"] for a in fresh], target=payload["target"],
                           idx=payload["idx"], delta=delta, beta=4.0,
                           budget=len(fresh), tag=f"cem{rd}")]
                try:
                    dres = list(dock.map(dj))[0]
                    for sm_, ds_ in (dres.get("top") or []):
                        docked_smi[sm_] = ds_
                        by_smiles_action.setdefault(sm_, None)
                except Exception as exc:
                    print(f"  docking failed: {type(exc).__name__}", flush=True)
            # rank the round's actions by the docking score they produced
            dsc, dact = [], []
            for j, r in zip(jobs, res):
                if r and r.get("smiles") in docked_smi:
                    dact.append(j["_ai"]); dsc.append(docked_smi[r["smiles"]])
            if dact:
                ctrl.update_by_rank(dact, dsc)
                best_ds = min(docked_smi.values())
                print(f"  docked={len(docked_smi)} best_ds={best_ds:.1f}", flush=True)
            else:
                ctrl.update(used, scores)
        else:
            ctrl.update(used, scores)
        # resample particles toward low shortfall, keep some exploration
        order = np.argsort(scores)
        keep = [nxt[i] for i in order[: max(1, len(nxt) // 2)]]
        parts = (keep * ((int(n_particles) // max(len(keep), 1)) + 1))[: int(n_particles)]
        top = [f"{ACTION_SPACE[i][0]}{ACTION_SPACE[i][1]}"
               for i in np.argsort(-ctrl.probs())[:3]]
        print(f"round {rd}: feasible_total={len(archive)} "
              f"median_shortfall={float(np.median(scores)):.3f} favours={top}", flush=True)
        ck = Path("/artifacts/macro_basin") / f"constrained_{payload['cell']}.json"
        ck.parent.mkdir(parents=True, exist_ok=True)
        # PROVENANCE. A checkpoint must carry enough identity that a reader can
        # refuse it if it belongs to a different run or a different frozen
        # interface. Reusing a path across runs already produced a fictitious
        # "result": an old pooled-ablation checkpoint was read as a new run,
        # and its stored action INDICES were decoded against a changed
        # ACTION_SPACE, yielding plausible but entirely invented macro traces.
        import hashlib as _hl
        _cfg = dict(
            action_space=[f"{n}|{k}" for n, k in ACTION_SPACE],
            arm=arm, ring_mass=float(ring_mass), run_seed=int(run_seed),
            n_particles=int(n_particles), rounds=int(rounds),
            episode_len=int(episode_len), dock_per_round=int(dock_per_round),
            delta=float(delta), seed_smiles=seed, cell=payload["cell"],
        )
        _cfg["config_sha256"] = _hl.sha256(
            json.dumps(_cfg, sort_keys=True).encode()).hexdigest()[:16]
        _cfg["run_id"] = f"{payload['cell']}_{arm}_r{int(run_seed)}_{_cfg['config_sha256']}"
        ck.write_text(json.dumps(dict(provenance=_cfg, cell=payload["cell"], round=rd,
                                      n_feasible=len(archive), archive=archive[:200],
                                      docked={k: v for k, v in docked_smi.items()},
                                      best_ds=(min(docked_smi.values())
                                               if docked_smi else None),
                                      budget_curve=budget_curve,
                                      action_probs={f"{a}{l}": float(p) for (a, l), p
                                                    in zip(ACTION_SPACE, ctrl.probs())}),
                                 indent=1, default=str))
        artifact_volume.commit()
    return dict(cell=payload["cell"], n_feasible=len(archive))


@app.local_entrypoint()
def constrained(cell: str = "5ht1b_s7_d0.4", n_particles: int = 64, rounds: int = 6,
                dock_per_round: int = 0):
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    tgt, si, dl = cell.rsplit("_", 2)
    sd = seeds[f"{tgt}_{si}"]
    payload = dict(cell=cell, seed=sd["smiles"], delta=float(dl[1:]),
                   target=sd["target"], idx=sd["idx"])
    fn = _m.Function.from_name("macro-basin", "drive_constrained")
    h = fn.spawn(payload, n_particles=n_particles, rounds=rounds,
                 dock_per_round=dock_per_round)
    print(f"  SPAWNED constrained search [{cell}]: {h.object_id}")


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=2 * 60 * 60,
              retries=0, max_containers=256, enable_memory_snapshot=True,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def search_episode(job: dict) -> dict:
    """Execute a WHOLE multi-macro episode, evaluate only the endpoint.

    The previous driver took one macro per round and then resampled particles
    toward LOW SHORTFALL, which preferentially keeps whatever is closest to the
    seed -- so it actively pruned the long trajectories. Result: 30 docked
    molecules at a median of 17 heavy atoms against a 16-atom seed, while the
    IVG winners for this cell are 31 heavy with 5 ring systems. The controller
    was choosing well inside a reachable set far too small to contain a -12
    binder.

    An episode composes several macros, so material added by one can be built on
    by the next: grow -> close -> grow again from the NEW atoms. Endpoint-only
    evaluation is what makes that affordable, and it is the same delayed-credit
    principle that the macro horizons needed one level down.
    """
    import numpy as np
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import (smiles_to_molecular_graph,
                                                 molecular_graph_to_smiles)
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import (rollout, contract_for, local_extend,
                                                 disjoint_closure_only)
    from compose_v4.control.constrained_search import margins
    from compose_v4.gates.med_chem_gate import is_executable

    rt = _runtime(); model, system = rt["model"], rt["system"]
    _cache = {}

    def enumerate_fn(smi):
        if smi not in _cache:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            _cache[smi] = ([m.executor_rule_name for m in law.marks],
                           np.array([m.probability for m in law.marks], float),
                           [(st, m.executor_rule_name, m.action) for m in law.marks])
        return _cache[smi]

    def apply_fn(smi, h):
        st, rule, action = h
        try: return canonical_state_key(system.apply(st, rule, action))
        except Exception: return None

    rng = np.random.default_rng(job["seed_rng"])
    cur = job["smiles"]; trace = []
    # STRUCTURAL EXECUTION FLAGS. Semantic credit must reach only requests that
    # actually executed. Carrying a boolean per ring request is authoritative;
    # re-deriving it by parsing `trace` strings downstream would be inference,
    # and would silently break the moment a trace format changed.
    ring_exec: list[dict] = []
    _ringcache: dict = {}
    # Every state along an executable trajectory is a molecule that was actually
    # constructed, and T4 asks for ONE molecule -- so a feasible intermediate is
    # a legitimate answer even if a later action ruins it. Scoring only the
    # final state discarded them: build_ring_system produced a feasible product
    # and a following restate turned an F into an S, and the episode reported
    # infeasible. Round 0 of the first sentinel returned feasible 0 / docked 0
    # with the controller pinned at uniform, because nothing could ever score.
    # INVARIANT: every realized molecular state is eligible for endpoint
    # bookkeeping. Every control path below -- including `continue` branches
    # and the STOP break -- must call _note(). Violating it once already cost a
    # sentinel round.
    from compose_v4.control.constrained_search import PrefixArchive
    _arch = PrefixArchive(lambda s: margins(s, job["seed"], float(job["delta"])),
                          seed_smiles=job["seed"])

    def _note(state, k):
        return _arch.note(state, k)

    _note(cur, -1)                       # x0 is realized too
    for _k, (macro, L) in enumerate(job["program"]):
        if macro == "STOP" or int(L) == 0:
            _note(cur, _k - 1)
            break
        if macro.startswith("ring:") or macro.startswith("build_ring_system"):
            # HIERARCHY. The controller picked the ring SEMANTICS; the compiler
            # enumerates the admissible realizations with their predicted
            # endpoints; frozen R_theta ranks among them.
            #
            #   P(xi | x, m) proportional to R_theta(xi | x, m) * H(xi; x, m, z)
            #
            # H is identity today: with total ring mass matched, six site-level
            # actions and one opaque action scored identically (-11.3 vs -11.2
            # best, -8.46 vs -8.57 mean), so there is no trustworthy site-level
            # objective evidence to apply. The realization identities and their
            # endpoint molecules are kept so a contextual h_phi(y_xi, z, b) can
            # score the candidate futures directly when one exists -- what was
            # dropped is only splitting ~30 docking observations across six
            # unrelated global CEM parameters.
            from compose_v4.control.constrained_search import parse_ring_spec
            from compose_v4.control.macro_engine import (admissible_anchors,
                admissible_fused_edges, build_fused_ring_exact)
            from compose_v4.gates.med_chem_gate import is_valid as _isvalid
            spec = parse_ring_spec(macro) or dict(topology="linked", size=int(L),
                                                  composition="carbon_rich",
                                                  state="aromatic")
            _sz = int(spec.get("size", L) or L)
            _st = pad_molecular_graph(smiles_to_molecular_graph(cur), CANONICAL_SLOTS)
            _arom = spec.get("state") == "aromatic"
            # SEMANTICS ARM. Default "legacy" is byte-identical to the banked
            # panel and must stay that way.
            #
            #   legacy      admissible_* apply delta/qed_min/sa_max -- ENDPOINT
            #               criteria -- to INTERMEDIATE states. On an
            #               endpoint-constrained benchmark that is an accidental
            #               PATHWISE restriction: measured on 5HT1B seed 7 it
            #               prunes fused intermediates 4->2 and 4->3.
            #   narrow_hard same three semantics, but intermediates masked ONLY
            #               by executor/state-space legality. Isolates the
            #               gate-correction effect (B - A).
            #
            # Everything downstream is untouched, so the arms differ solely in
            # which realizations the exact R_theta-ranked builder may use.
            _sem = str(job.get("semantics", "legacy"))
            # STATE-CONDITIONED SEMANTIC DRAW. For broad semantics the tuple is
            # chosen HERE, against the support of the CURRENT state `cur`, not
            # at program-construction time against the seed. Sampling at the
            # seed meant a tuple drawn from x0's support had to execute several
            # edits later at a different state: measured 50-70% execution and
            # ~7% reaching credit. `_ringcache` memoises (state, tuple) validity
            # so a redraw against the same state is free.
            if _sem == "broad" and job.get("ring_policy"):
                import numpy as _np2

                from compose_v4.control import semantic_actions as _sa3
                from compose_v4.control.macro_engine import (
                    fusable_edges as _fe3, predict_fused_product as _pf3,
                    predict_pendant_product as _pp3)
                _rp = job["ring_policy"]
                if "_pol" not in _ringcache:
                    _p3 = _sa3.FactorizedRingPolicy(
                        _np2.random.default_rng(int(job.get("seed_rng", 0))),
                        _rp["domain"])
                    _p3.load_state_dict(_rp)
                    _ringcache["_pol"] = _p3
                _stx = pad_molecular_graph(smiles_to_molecular_graph(cur),
                                           CANONICAL_SLOTS)
                # R_THETA SEMANTIC PRIOR. rho(s|x) is the frozen reference
                # law's mass on the first committed insertion that begins each
                # transformation; the policy supplies only a learned tilt.
                # Under a FLAT prior the winning move on this cell was drawn
                # 1.0% of the time versus 33.3% under the narrow menu, and it
                # has to fire 2-3x in one program to pay off -- so docking was
                # being asked to rediscover basic plausibility from ~10 labels
                # a round. Cached per canonical state; one law enumeration.
                if "_specs" not in _ringcache:
                    _ringcache["_specs"] = _sa3.enumerate_specs(_rp["domain"])
                _pk = ("_prior", cur)
                if _pk not in _ringcache:
                    _lw = enumerate_factorized_marked_law(model, _stx,
                                                          float(TIME_POINT))
                    _ringcache[_pk] = _sa3.rtheta_semantic_prior(
                        _ringcache["_specs"],
                        [m.executor_rule_name for m in _lw.marks],
                        [m.action for m in _lw.marks],
                        _np2.array([m.probability for m in _lw.marks], float))
                _rq3, _rl3, _nt3 = _sa3.sample_prior_tilted(
                    _ringcache["_pol"], _ringcache[_pk], _stx, job["seed"],
                    predict_fn=_pp3, fused_fn=_pf3, edges_fn=_fe3,
                    cache=_ringcache, state_key=cur, tries=12,
                    specs=_ringcache["_specs"])
                if _rq3 is not None:
                    macro = _rq3.key
            if _sem == "legacy":
                if spec.get("topology") == "fused":
                    _adm = admissible_fused_edges(_st, job["seed"], float(job["delta"]),
                                                  size=_sz, aromatic=_arom,
                                                  gate_fn=_isvalid)
                else:
                    _adm = admissible_anchors(_st, job["seed"], float(job["delta"]),
                                              size=_sz, aromatic=_arom,
                                              gate_fn=_isvalid)
            else:
                from compose_v4.control import semantic_actions as _sa
                from compose_v4.control.macro_engine import (
                    fusable_edges as _fe, predict_fused_product as _pfp,
                    predict_pendant_product as _ppp)
                _comp = spec.get("composition", "carbon_rich")
                _stoi = _sa.normalize_stoich({}, _sz) if _comp in (
                    "carbon_rich", "C") else _sa.parse_stoich_label(_comp, _sz)
                _req = _sa.RingRequest(spec.get("topology", "linked"), _sz,
                                       _stoi, spec.get("state", "aromatic"))
                _reals, _why = _sa.realize(_req, _st, job["seed"],
                                           predict_fn=_ppp, fused_fn=_pfp,
                                           edges_fn=_fe, max_realizations=200)
                if spec.get("topology") == "fused":
                    _adm = [(r["edge"], r["smiles"], r["sim"], r["qed"], r.get("sa"))
                            for r in _reals if "edge" in r]
                else:
                    _adm = [(r["anchor"], r["smiles"], r["sim"], r["qed"], r.get("sa"))
                            for r in _reals if "anchor" in r]
                # de-duplicate sites, preserving first-seen order
                _seen_s, _dd = set(), []
                for _row in _adm:
                    _key = _row[0]
                    if _key in _seen_s:
                        continue
                    _seen_s.add(_key)
                    _dd.append(_row)
                _adm = _dd
            if not _adm:
                trace.append(f"{macro}:UNSAT(no_admissible_realization)")
                ring_exec.append(dict(key=str(macro), ok=False, stage="no_admissible_realization"))
                _note(cur, _k)
                continue
            if spec.get("topology") == "fused":
                _fc2 = {}

                def _ts2(st_):
                    try: return molecular_graph_to_smiles(st_)
                    except Exception: return None

                def _ef2(st_):
                    kk = _ts2(st_)
                    if kk not in _fc2:
                        law = enumerate_factorized_marked_law(model, st_, float(TIME_POINT))
                        _fc2[kk] = ([m.executor_rule_name for m in law.marks],
                                    [m.action for m in law.marks],
                                    np.array([m.probability for m in law.marks], float))
                    return _fc2[kk]

                def _ap2(st_, j):
                    fams, acts, _p = _ef2(st_)
                    try: return system.apply(st_, fams[j], acts[j])
                    except Exception: return None

                rr = build_fused_ring_exact(_ef2, _ap2, _ts2, is_executable, _st,
                                            size=_sz,
                                            composition=spec.get("composition", "carbon_rich"),
                                            state=spec.get("state", "aromatic"),
                                            anchor_rank=0,
                                            candidate_edges=[e for e, *_ in _adm],
                                            refine=int(spec.get("refine",
                                                                job.get("refine", 0))),
                                            refine_rng=np.random.default_rng(
                                                int(job["seed_rng"]) * 1000 + int(_k)))
            else:
                rr = _build_ring_cached(model, system, cur, size=_sz,
                                        topology="pendant",
                                        composition=spec.get("composition", "carbon_rich"),
                                        ring_state=spec.get("state", "aromatic"),
                                        anchor_rank=0,
                                        anchors=[a for a, *_ in _adm],
                                        refine=int(spec.get("refine",
                                                            job.get("refine", 0))),
                                        # per-particle, per-position: a shared
                                        # seed would hand all 64 particles the
                                        # same refined ring through the memo.
                                        refine_seed=int(job["seed_rng"]) * 1000 + int(_k))
            if rr.get("status") == "OK" and rr.get("smiles"):
                cur = rr["smiles"]
                trace.append(f"{macro}:ok(adm{len(_adm)})")
                # AUDITABILITY. The builder's own trace carries the REFINE_RING
                # steps, tagged @ring, and the episode used to discard it -- so
                # a checkpoint could not show whether refinement ever fired.
                # For a claim-bearing panel that has to be visible in the
                # artifact, not inferred from the config flag.
                _rf = [str(t) for t in (rr.get("trace") or []) if "@ring" in str(t)]
                if _rf:
                    trace.append("refine[" + str(len(_rf)) + "]:" + ",".join(_rf))
                ring_exec.append(dict(key=str(macro), ok=True, stage="ok",
                                      refine_steps=len(_rf)))
            else:
                trace.append(f"{macro}:UNSAT@{rr.get('stage')}")
                ring_exec.append(dict(key=str(macro), ok=False, stage=str(rr.get("stage"))))
            _note(cur, _k)
            continue
        if macro == "local_extend":
            pref = local_extend(cur, job["seed"], max_anchors=min(int(L), 4)); eff = "grow"
        elif macro == "close_disjoint":
            pref = disjoint_closure_only(cur); eff = "cyclize"
        elif macro in ("annulate", "shrink", "decorate"):
            pref = contract_for(macro, cur)
            eff = {"annulate": "cyclize", "shrink": "shrink", "decorate": "grow"}[macro]
        else:
            pref = None; eff = macro
        r = rollout(enumerate_fn, apply_fn, is_executable, cur, eff, int(L), rng,
                    prefer_fn=pref)
        trace.append(f"{macro}{L}")
        if r.length:
            cur = r.endpoint
        _note(cur, _k)
    m = margins(cur, job["seed"], float(job["delta"]))
    best = _arch.best()
    if not m.feasible and best is not None:
        cur, best_k, m = best[0], best[1], best[2]
    else:
        best_k = len(job["program"])
    # The UNCHANGED SEED is never an answer. It trivially satisfies sim=1 and,
    # on a QED-rich seed, every other constraint -- so a program that UNSATs
    # everything reported feasible=True with the seed as its endpoint. The
    # archive and drive_episodes both filter it canonically, so it could never
    # be docked, but the per-episode flag was misleading in raw output.
    try:
        from rdkit import Chem as _C
        _a, _b = _C.MolFromSmiles(cur), _C.MolFromSmiles(job["seed"])
        if _a is not None and _b is not None and \
                _C.MolToSmiles(_a) == _C.MolToSmiles(_b):
            m = margins(cur, job["seed"], float(job["delta"]))
            m = type(m)(sim=m.sim, qed=m.qed, sa=m.sa, valid=False)
    except Exception:
        pass
    from rdkit import Chem
    mm = Chem.MolFromSmiles(cur)
    # EVERY distinct feasible prefix is a candidate answer, not just the last:
    # T4 returns one molecule and the docking oracle is what decides which of
    # them was actually useful.
    cands = []
    for _s, _pl, _mm in _arch.entries():
        _m2 = Chem.MolFromSmiles(_s)
        from compose_v4.control.constrained_search import med_chem_ok as _mc
        cands.append(dict(smiles=_s, prefix_len=int(_pl),
                          heavy=_m2.GetNumHeavyAtoms() if _m2 else 0,
                          sim=round(_mm.sim, 3), qed=round(_mm.qed, 3),
                          sa=round(_mm.sa, 2),
                          # secondary validity analysis, never gating the
                          # primary benchmark number
                          med_chem=bool(_mc(_s))))
    return dict(smiles=cur, program=job["program"], trace=trace, ring_exec=ring_exec,
                start_depth=int(job.get("start_depth") or 0),
                cumulative_depth=int(job.get("start_depth") or 0) + len(trace),
                prefix_len=int(best_k), candidates=cands,
                n_feasible_prefixes=len(cands),
                shortfall=m.shortfall, feasible=m.feasible,
                heavy=mm.GetNumHeavyAtoms() if mm else 0,
                sim=round(m.sim, 3), qed=round(m.qed, 3), sa=round(m.sa, 2))


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024),
              timeout=4 * 60 * 60, retries=0,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive_episodes(payload: dict, n_particles: int = 64, rounds: int = 3,
                   episode_len: int = 5, dock_per_round: int = 10,
                   arm: str = "pooled", ring_mass: float = 0.24,
                   run_seed: int = 0, semantics: str = "legacy",
                   refine: int = 0):
    """CEM over PROGRAMS (sequences of macros), docking as the objective.

    `arm` selects the ablation arm, with TOTAL initial ring probability matched
    at `ring_mass` in both so the comparison isolates site visibility:
      "pooled" -- six site-visible realizations, partially pooled credit
      "opaque" -- one ring action, no site identity
    """
    import numpy as np, collections
    from rdkit import Chem
    from compose_v4.control.constrained_search import (ACTION_SPACE as _AS_POOLED,
                                                       ACTION_SPACE_OPAQUE,
                                                       CEMController,
                                                       realization_groups,
                                                       matched_prior)
    import modal as _m
    # GenMol/InVirtuoGen report the MEAN over independent optimization runs of
    # each run's best feasible lead. So each run must be genuinely independent.
    rng = np.random.default_rng(20260826 + int(run_seed))
    ACTION_SPACE = ACTION_SPACE_OPAQUE if arm == "opaque" else _AS_POOLED
    ctrl = CEMController(len(ACTION_SPACE), rng,
                         init_probs=matched_prior(ACTION_SPACE, ring_mass))
    # Attachment sites of one ring spec are realizations of the SAME macro.
    # Pooling their evidence stops the controller assigning full-strength
    # credit to a site on one noisy docking observation -- which it did,
    # putting 0.254 on @0 against 0.096 on @1 when replicated docking showed
    # @1 marginally better.
    _groups = realization_groups(ACTION_SPACE) if arm != "opaque" else []
    seed, delta = payload["seed"], float(payload["delta"])
    seed_can = Chem.MolToSmiles(Chem.MolFromSmiles(seed))
    dock = _m.Function.from_name("genmol-t4-gate4", "dock_frontier")
    archive, seen, docked_smi = [], set(), {}
    # WARM START: Phase 1 (cheap constraint satisfaction, ZERO oracle calls)
    # hands Phase 2 an already-feasible pool. Three cheap constraints and one
    # expensive objective get different machinery, so the controller never has
    # to learn "repair QED first" from docking feedback it cannot receive until
    # the feasible set is entered.
    from compose_v4.control.constrained_search import margins
    warm = [s for s in (payload.get("warm_start") or []) if s]
    # smiles -> (shortfall, cumulative_depth). LINEAGE IS PART OF THE STATE:
    # restarting from an archived molecule CONTINUES that trajectory, it does
    # not reset the edit horizon. Without the depth we would launch a 5-action
    # episode from a state already 3 edits deep and then report a "5-macro"
    # controller for what actually took 8.
    shortfall_pool = {}
    budget_curve = []
    # BROAD SEMANTICS (arm C). BUILD_RING_SYSTEM is ONE parent action; its
    # topology/size/stoichiometry/state are drawn from learned FACTORIZED
    # parameter distributions, masked to what the executor can realize at the
    # start state. Parent mass is therefore untouched by how many parameter
    # tuples happen to be feasible -- the count of ring specs cannot change how
    # often rings are attempted.
    _ringpol = _ringdom = None
    _sem_log = []
    _sem_elig = []
    if str(semantics) == "broad":
        from compose_v4.chem.molecular_graph import smiles_to_molecular_graph as _s2g
        from compose_v4.chem.state import pad_molecular_graph as _pad
        from compose_v4.control import semantic_actions as _sa
        from compose_v4.control.macro_engine import predict_pendant_product as _ppp
        _st_seed = _pad(_s2g(seed), CANONICAL_SLOTS)
        _ringdom = _sa.qualify_domain([_st_seed], _ppp)
        _ringpol = _sa.FactorizedRingPolicy(rng, _ringdom)
        # No seed-state SupportMask is built any more. Support is evaluated at
        # the CURRENT state inside search_episode, by rejection sampling with a
        # (state, tuple) validity cache -- 122 ms per draw, versus the 11.0 s
        # this mask cost to construct and which conditioned on the wrong state.
        print(f"BROAD semantics: qualified sizes {_ringdom['size']}", flush=True)
    _rejected = 0
    if warm:
        print(f"  warm start: {len(warm)} feasible molecules from the cheap "
              f"frontier (0 oracle calls spent building it)", flush=True)
        for _w in warm:
            try:
                _mw = Chem.MolFromSmiles(_w)
                if _mw is None:
                    continue
                _kw = Chem.MolToSmiles(_mw)
                if _kw == seed_can or _kw in seen:
                    continue
                _m2 = margins(_w, seed, delta)
                if not _m2.feasible:
                    continue
                seen.add(_kw)
                archive.append(dict(smiles=_w, _ai=[], trace=["warm_start"],
                                    heavy=_mw.GetNumHeavyAtoms(),
                                    sim=round(_m2.sim, 3), qed=round(_m2.qed, 3),
                                    sa=round(_m2.sa, 2), prefix_len=0))
            except Exception as _exc:
                # NEVER swallow silently here. A bare `except: continue` hid a
                # NameError on `margins` for all 401 warm-start molecules --
                # the archive stayed empty and every downstream check reported
                # "0 feasible" with no indication that anything had failed.
                if _rejected == 0:
                    print(f"  warm-start rejected: {type(_exc).__name__}: {_exc}",
                          flush=True)
                _rejected += 1
                continue
        if _rejected:
            print(f"  warm-start: {_rejected}/{len(warm)} rejected", flush=True)
        print(f"  archived {len(archive)} feasible warm-start molecules", flush=True)

    for rd in range(int(rounds)):
        jobs, prog_actions = [], []
        # STARTING POINTS. While nothing is feasible, restarting every episode
        # from the seed forces each one to complete the entire repair in
        # `episode_len` actions -- braf_s10 needs 5-6 edits and the controller
        # samples 5 from 22, so it never lands the sequence even though it
        # correctly learns that shrink helps. The cheap probe succeeded only
        # because it kept a beam of PARTIAL progress across depths.
        #
        # So bank partial progress: while infeasible, launch from the
        # lowest-shortfall states found so far. This is the same archive
        # mechanism used for feasible molecules, keyed on shortfall instead.
        starts = [(seed, 0)]
        if not archive and shortfall_pool:
            # VALLEY-SAFE. Measured over the 113 successful braf_s10 repairs:
            # 17% dip below delta and recover, 10% have NON-MONOTONE total
            # shortfall, and the deepest valley's FIRST step is strictly worse
            # than the seed (0.254 -> 0.275, sim 0.576) yet lies on the path to
            # a feasible endpoint. A pool keyed on lowest current shortfall
            # prunes exactly those routes -- the same myopia the endpoint-only
            # macro judging was built to avoid.
            #
            # So: a best-shortfall pool PLUS a diversity reserve that is not
            # selected on shortfall at all. Never impose sim/QED/SA as pathwise
            # hard constraints; the benchmark constrains the RETURNED lead.
            _ranked = sorted(shortfall_pool.items(), key=lambda kv: kv[1][0])
            _greedy = _ranked[:6]
            _rest = _ranked[6:]
            _reserve = []
            if _rest:
                _idx = rng.choice(len(_rest), size=min(4, len(_rest)),
                                  replace=False)
                _reserve = [_rest[int(i)] for i in _idx]
            starts = [(s, v[1]) for s, v in (_greedy + _reserve)] or [(seed, 0)]
        for _pi in range(int(n_particles)):
            ai = ctrl.sample(int(episode_len))
            prog = [ACTION_SPACE[int(i)] for i in ai]
            # The semantic tuple is NO LONGER drawn here. search_episode draws
            # it at the moment BUILD_RING_SYSTEM executes, against the support
            # of the state actually reached. Only the learned policy travels.
            prog_actions.append(list(ai))
            _start, _sdep = starts[_pi % len(starts)] if starts else (seed, 0)
            jobs.append(dict(smiles=_start, seed=seed, delta=delta, program=prog,
                             start_depth=int(_sdep),
                             semantics=str(semantics),
                             # FALLBACK ONLY. Refinement is now a parameter of
                             # the ring ACTION (`ring:.../rN`), so the controller
                             # selects it per cell from docking evidence. This
                             # job-level value is used only when a macro carries
                             # no parsable ring spec. Freezing it at 2 for every
                             # build cost 0.83 kcal/mol on parp1 and gave the
                             # controller no way to opt out.
                             refine=int(refine),
                             ring_policy=(_ringpol.state_dict()
                                          if _ringpol is not None else None),
                             seed_rng=int(rng.integers(0, 2**31))))
        res = list(search_episode.map(jobs))
        for _ai3, _r3 in zip(prog_actions, res):
            if _r3 is not None:
                _r3["_ai_full"] = [int(x) for x in _ai3]
                _sf = _r3.get("shortfall")
                _sm3 = _r3.get("smiles")
                if _sf is not None and _sm3 and not _r3.get("feasible"):
                    try:
                        if Chem.MolToSmiles(Chem.MolFromSmiles(_sm3)) != seed_can:
                            _d0 = int(_r3.get("start_depth") or 0)
                            _dep = _d0 + int(_r3.get("executed") or
                                             len(_r3.get("trace") or []))
                            _prev = shortfall_pool.get(_sm3)
                            if _prev is None or _sf < _prev[0]:
                                shortfall_pool[_sm3] = (float(_sf), _dep)
                    except Exception:
                        pass
        if len(shortfall_pool) > 400:      # keep it bounded
            shortfall_pool = dict(sorted(shortfall_pool.items(),
                                         key=lambda kv: kv[1][0])[:200])
        heavy = []
        for ai, r in zip(prog_actions, res):
            if not r: continue
            heavy.append(r["heavy"])
            if r.get("feasible"):
                try:
                    if Chem.MolToSmiles(Chem.MolFromSmiles(r["smiles"])) == seed_can:
                        continue
                except Exception: pass
                # every distinct feasible prefix is a separate candidate, each
                # crediting only the actions that produced it
                for _c in (r.get("candidates") or [r]):
                    _sm = _c.get("smiles")
                    if not _sm or _sm in seen:
                        continue
                    try:
                        if Chem.MolToSmiles(Chem.MolFromSmiles(_sm)) == seed_can:
                            continue
                    except Exception:
                        continue
                    seen.add(_sm)
                    _row = dict(_c)
                    # plain ints: ctrl.sample() returns numpy, and np.int64 is
                    # not JSON serialisable, so the round checkpoint write threw
                    # and killed the whole run -- but ONLY once the archive was
                    # non-empty, so the earlier 0-feasible round looked fine.
                    _row["_ai"] = [int(x) for x in
                                   list(ai)[: int(_c.get("prefix_len") or len(ai))]]
                    _row["trace"] = r.get("trace")
                    archive.append(_row)
        fresh = [a for a in archive if a["smiles"] not in docked_smi][: int(dock_per_round)]
        if fresh:
            dj = [dict(smiles=[a["smiles"] for a in fresh], target=payload["target"],
                       idx=payload["idx"], delta=delta, beta=4.0,
                       budget=len(fresh), tag=f"ep{rd}")]
            try:
                for sm_, ds_ in (list(dock.map(dj))[0].get("top") or []):
                    docked_smi[sm_] = ds_
            except Exception as exc:
                print(f"  dock failed: {type(exc).__name__}", flush=True)
        acts, scs = [], []
        for a in archive:
            if a["smiles"] in docked_smi:
                for i in a.get("_ai", []):
                    acts.append(int(i)); scs.append(docked_smi[a["smiles"]])
        if acts:
            ctrl.update_by_rank(acts, scs, groups=_groups, site_shrink=0.25)
            # FACTORIZED SEMANTIC CREDIT. A particle's ring requests are scored
            # by the docking value of the endpoint that particle produced, so
            # parameter mass moves on the same evidence the CEM uses. Credit is
            # per-parameter, not per-tuple: the tuple space is far too sparse to
            # fit from ~100 oracle calls, but "saturated" or "one N" generalizes
            # across sizes and topologies.
            if _ringpol is not None:
                # ONLY EXECUTED REQUESTS ARE ELIGIBLE. A request that returned
                # UNSAT contributed no transformation, so it must not absorb the
                # trajectory's docking score. Crediting SAMPLED requests let a
                # particle whose every ring macro failed still push mass onto
                # those tuples; P(fused) reached 0.959 in an earlier run whose
                # docked set contained zero fused pairs. Eligibility is read
                # from the executor's own ok flag, never re-derived from traces.
                from compose_v4.control.semantic_actions import (
                    parse_request_key as _prk)
                _rq_all, _rq_sc = [], []
                _n_samp = _n_exec = 0
                for _pi2 in range(len(res)):
                    _r2 = res[_pi2] or {}
                    _rex = _r2.get("ring_exec") or []
                    _n_samp += len(_rex)
                    _sm2 = _r2.get("smiles")
                    if _sm2 not in docked_smi:
                        continue
                    for _e2 in _rex:
                        if not _e2.get("ok"):
                            continue            # executed nothing: no credit
                        _q = _prk(_e2.get("key"))
                        if _q is None:
                            continue
                        _n_exec += 1
                        _rq_all.append(_q); _rq_sc.append(float(docked_smi[_sm2]))
                _sem_elig.append(dict(round=rd, sampled=_n_samp, credited=_n_exec))
                if _rq_all:
                    _ringpol.update(_rq_all, _rq_sc)
        elif not archive:
            # LEXICOGRAPHIC BOOTSTRAP. With no feasible molecule there is
            # nothing to dock, so docking feedback can never arrive and the
            # controller stays at its prior forever -- braf_s10 ran all ten
            # rounds at 0 feasible / 0 docked because its seed starts at
            # QED 0.346 against a 0.6 gate, and 9 of the 15 T4 seeds start
            # below it.
            #
            # While the feasible archive is empty, update on the CHEAP endpoint
            # shortfall instead. It costs zero oracle calls, and unlike the
            # earlier shortfall objective it has a real gradient here: that one
            # failed on QED-rich seeds where the seed itself scored zero
            # shortfall, making STOP optimal. On an INFEASIBLE seed the
            # shortfall descends 0.254 -> 0.000 over ~5 edits.
            #
            # The repair program is discovered from the existing vocabulary --
            # no QED-specific macro, no hand-picked shrink/restate.
            ba, bs = [], []
            for _r in res:
                if not _r or _r.get("shortfall") is None:
                    continue
                _ai2 = _r.get("_ai_full")
                if _ai2 is None:
                    continue
                for i in _ai2:
                    ba.append(int(i)); bs.append(float(_r["shortfall"]))
            if ba:
                ctrl.update_by_rank(ba, bs, groups=_groups, site_shrink=0.25)
                print(f"  round {rd}: BOOTSTRAP shortfall best {min(bs):.3f} "
                      f"pool {len(shortfall_pool)} (0 oracle calls)", flush=True)
        best = min(docked_smi.values()) if docked_smi else None
        # BUDGET CURVE. GenMol's released lead-optimisation runs num_gen=100 x
        # num_iter=10 = 1000 docking evaluations per run, and InVirtuoGen sets
        # --max_oracle_calls 1000. So oracle EFFICIENCY is a reportable axis,
        # not just endpoint quality: record the best feasible lead as a function
        # of docking calls spent, so a 1000-call run also yields the 25/50/100/
        # 250/500 points without extra cost.
        if _ringpol is not None:
            _sem_log.append(dict(round=rd, oracle_calls=len(docked_smi),
                                 eligibility=list(_sem_elig),
                                 logit={a: dict(v) for a, v in _ringpol.logit.items()},
                                 sampled=[e["key"] for r_ in res if r_
                                          for e in (r_.get("ring_exec") or [])]))
        budget_curve.append(dict(round=rd, oracle_calls=len(docked_smi),
                                 best=best))
        print(f"round {rd}: feasible={len(archive)} docked={len(docked_smi)} "
              f"best={best} median_heavy={int(np.median(heavy)) if heavy else 0} "
              f"max_heavy={max(heavy) if heavy else 0}", flush=True)
        # `refine` is part of the frozen interface, so it must appear in the
        # PATH as well as in the config hash. Without it a refine=0 ablation
        # overwrites the refine=2 checkpoint for the same cell/arm/seed --
        # exactly the stale-checkpoint failure described just below.
        _suffix = ((f"_{arm}" if arm else "")
                   + ("" if semantics == "legacy" else f"_{semantics}")
                   + ("" if int(refine) == 2 else f"_rf{int(refine)}")
                   + f"_r{int(run_seed)}")
        ck = Path("/artifacts/macro_basin") / f"episodes_{payload['cell']}{_suffix}.json"
        ck.parent.mkdir(parents=True, exist_ok=True)
        # PROVENANCE. A checkpoint must carry enough identity that a reader can
        # refuse it if it belongs to a different run or a different frozen
        # interface. Reusing a path across runs already produced a fictitious
        # "result": an old pooled-ablation checkpoint was read as a new run,
        # and its stored action INDICES were decoded against a changed
        # ACTION_SPACE, yielding plausible but entirely invented macro traces.
        import hashlib as _hl
        _cfg = dict(
            action_space=[f"{n}|{k}" for n, k in ACTION_SPACE],
            arm=arm, ring_mass=float(ring_mass), run_seed=int(run_seed),
            n_particles=int(n_particles), rounds=int(rounds),
            episode_len=int(episode_len), dock_per_round=int(dock_per_round),
            delta=float(delta), seed_smiles=seed, cell=payload["cell"],
            refine=int(refine),
        )
        _cfg["config_sha256"] = _hl.sha256(
            json.dumps(_cfg, sort_keys=True).encode()).hexdigest()[:16]
        _cfg["run_id"] = f"{payload['cell']}_{arm}_r{int(run_seed)}_{_cfg['config_sha256']}"
        ck.write_text(json.dumps(dict(provenance=_cfg, semantic_log=_sem_log, cell=payload["cell"], round=rd,
                                      n_feasible=len(archive), docked=docked_smi,
                                      best_ds=best, archive=archive[:150],
                                      budget_curve=budget_curve,
                                      action_probs={f"{a}{l}": float(p) for (a, l), p
                                                    in zip(ACTION_SPACE, ctrl.probs())}),
                                 indent=1, default=str))
        artifact_volume.commit()
    return dict(cell=payload["cell"], best=min(docked_smi.values()) if docked_smi else None)


@app.local_entrypoint()
def episodes(cell: str = "5ht1b_s7_d0.4", n_particles: int = 64, rounds: int = 3,
             episode_len: int = 5, dock_per_round: int = 10,
             arm: str = "pooled", run_seed: int = 0, frontier: str = "",
             semantics: str = "legacy"):
    import modal as _m
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    tgt, si, dl = cell.rsplit("_", 2)
    sd = seeds[f"{tgt}_{si}"]
    payload = dict(cell=cell, seed=sd["smiles"], delta=float(dl[1:]),
                   target=sd["target"], idx=sd["idx"])
    if frontier:
        _fr = json.loads((root / frontier).read_text())
        payload["warm_start"] = [m["smiles"] for m in _fr.get("molecules", [])]
        print(f"  warm start from {frontier}: {len(payload['warm_start'])} molecules")
    fn = _m.Function.from_name("macro-basin", "drive_episodes")
    h = fn.spawn(payload, semantics=semantics, n_particles=n_particles, rounds=rounds,
                 episode_len=episode_len, dock_per_round=dock_per_round, arm=arm,
                 run_seed=run_seed)
    print(f"  SPAWNED episodes [{cell}]: {h.object_id}")


@app.local_entrypoint()
def smoke(cell: str = "5ht1b_s7_d0.6", n: int = 6, refine: int = 2):
    """1%-scale timing/smoke check before the sentinel.

    Times the ONE loop the sentinel newly depends on: build_ring_system inside
    an episode. Each application is ~8 sequential model enumerations, and that
    cost has never been measured in this path. Two multi-hour surprises this
    project came from scaling an unmeasured loop.
    """
    import time
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    tgt, si, dl = cell.rsplit("_", 2)
    sd = seeds[f"{tgt}_{si}"]
    seed, delta = sd["smiles"], float(dl[1:])
    programs = [
        [("build_ring_system", 6)],
        [("build_ring_system", 6), ("build_ring_system", 6)],
        [("build_ring_system", 6), ("build_ring_system", 6), ("restate", 1)],
        [("build_ring_system", 5), ("build_ring_system", 6)],
        [("local_extend", 4), ("build_ring_system", 6)],
        [("build_ring_system", 6), ("decorate", 1)],
    ][: int(n)]
    jobs = [dict(smiles=seed, seed=seed, delta=delta, program=p,
                 refine=int(refine), seed_rng=1000 + i)
            for i, p in enumerate(programs)]
    t0 = time.time()
    res = list(search_episode.map(jobs))
    el = time.time() - t0
    print(f"\n[refine={int(refine)}] {len(jobs)} episodes in {el:.1f}s wall "
          f"({el / max(len(jobs), 1):.1f}s/episode at this concurrency)\n")
    nfeas = 0
    for p, r in zip(programs, res):
        if not r:
            print(f"  {str(p):58s} -> None"); continue
        nfeas += bool(r.get("feasible"))
        print(f"  {'->'.join(f'{a}{b}' for a, b in p):58s}")
        print(f"      feasible={r.get('feasible')} sim={r.get('sim')} "
              f"qed={r.get('qed')} sa={r.get('sa')} heavy={r.get('heavy')} "
              f"trace={r.get('trace')}")
        print(f"      {r.get('smiles')}")
    print(f"\nfeasible {nfeas}/{len(res)} at delta={delta}")
    per_round = el / max(len(jobs), 1)
    print(f"\nPROJECTION for the 100-call sentinel (10 rounds x 64 particles):")
    print(f"  episodes: 640 x {per_round:.1f}s / (concurrency) -- at 20 containers "
          f"~= {640 * per_round / 20 / 60:.1f} min of episode time")
    print(f"  plus 100 docking calls (~9s each, 20-wide) ~= {100 * 9 / 20 / 60:.1f} min")
