"""Can the executor compile a generic aromatic pendant from the seed?

MOTIVATION. On the 5ht1b seed, attaching a plain phenyl at ANY substitutable
position is T4-feasible: 8/8 attachment sites give sim 0.43-0.62, QED 0.904,
SA 2.97-3.42, at 22 heavy atoms with 3 ring systems and 2 aromatic rings --
strictly better than every one of the 30 molecules the controller actually
docked (median QED ~0.80, SA ~3.8, sim ~0.49, and ALL of them 2 systems /
1 aromatic). The cost is 2.17 new fingerprint bits per atom added, against 8.0
for the bespoke decorations the controller does propose.

So the target region is trivially reachable and maximally feasible, and
R_theta proposes it with ~zero probability. This asks the narrow question:
given the executor's primitives, can that molecule be BUILT at all, and where
does R_theta rank each required step?

The targets are GENERIC -- a plain benzene at a substitutable position -- not
copied from any published molecule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
app = modal.App("aryl-compile")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS, APPLY_CAP = 0.5, 48, 300
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


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=60 * 60,
              retries=0, enable_memory_snapshot=True, max_containers=40,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def compile_target(job: dict) -> dict:
    """Greedy graph-diff walk from seed toward the target, reporting R_theta's
    rank for each step actually taken."""
    import numpy as np
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    rt = _runtime(); model, system = rt["model"], rt["system"]
    tgt = Chem.MolFromSmiles(job["target"])
    tgt_can = Chem.MolToSmiles(tgt)
    tn = tgt.GetNumAtoms()

    def score_to_target(smi):
        """crude progress: shared atoms via MCS-free heuristic = atom count match
        plus canonical-prefix agreement is unreliable, so use heavy-atom delta
        toward target plus formula distance."""
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return -1e9
        import collections
        a = collections.Counter(x.GetSymbol() for x in m.GetAtoms())
        b = collections.Counter(x.GetSymbol() for x in tgt.GetAtoms())
        diff = sum((a - b).values()) + sum((b - a).values())
        arom = abs(sum(1 for x in m.GetAtoms() if x.GetIsAromatic())
                   - sum(1 for x in tgt.GetAtoms() if x.GetIsAromatic()))
        return -(diff + arom)

    cur = job["seed"]; steps = []
    for depth in range(int(job.get("max_steps", 14))):
        if Chem.MolToSmiles(Chem.MolFromSmiles(cur)) == tgt_can:
            return dict(target=job["target"], reached=True, depth=depth, steps=steps)
        st = pad_molecular_graph(smiles_to_molecular_graph(cur), CANONICAL_SLOTS)
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        pr = np.array([m.probability for m in law.marks], float)
        order = list(np.argsort(-pr))
        rank_of = {int(j): r for r, j in enumerate(order)}
        best = None
        for j in range(len(law.marks)):
            mk = law.marks[j]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
            except Exception:
                continue
            if not y:
                continue
            s = score_to_target(y)
            if best is None or s > best[0]:
                best = (s, y, mk.executor_rule_name, rank_of[j], float(pr[j]))
        if best is None:
            return dict(target=job["target"], reached=False, depth=depth,
                        steps=steps, halted="no successor")
        s, y, rule, rk, p = best
        steps.append(dict(rule=rule, rank=rk, prob=p, in_cap=bool(rk < APPLY_CAP),
                          smiles=y))
        if y == cur:
            return dict(target=job["target"], reached=False, depth=depth,
                        steps=steps, halted="no progress")
        cur = y
    return dict(target=job["target"], reached=False, depth=int(job.get("max_steps", 14)),
                steps=steps, halted="budget", final=cur)


@app.function(image=image, cpu=(1.0, 1.0), memory=int(1 * 1024), timeout=600,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def make_targets(seed: str, n: int) -> list:
    """Generic phenyl pendants at each substitutable position. Built remotely
    because the local entrypoint runs in Modal's own venv, which has no rdkit."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    sm = Chem.MolFromSmiles(seed)
    out = []
    for i, a in enumerate(sm.GetAtoms()):
        if a.GetTotalNumHs() == 0:
            continue
        rw = Chem.RWMol(sm)
        idx = [rw.AddAtom(Chem.Atom(6)) for _ in range(6)]
        for k in range(6):
            rw.AddBond(idx[k], idx[(k + 1) % 6], Chem.BondType.AROMATIC)
            rw.GetAtomWithIdx(idx[k]).SetIsAromatic(True)
        rw.AddBond(i, idx[0], Chem.BondType.SINGLE)
        try:
            m = rw.GetMol(); Chem.SanitizeMol(m)
            out.append(Chem.MolToSmiles(m))
        except Exception:
            pass
    return out[: int(n)]


@app.function(image=image, cpu=(1.0, 1.0), memory=int(1 * 1024), timeout=600,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def stage_pair(seed: str) -> list:
    """Two separate compiles: chain->closed ring, and closed ring->aromatic.
    Splitting them tests whether the CLOSURE or the AROMATISATION is the blocker."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.control.macro_engine import aromatisation_targets
    chain = "CCCCCCc1ccc(N2CC[NH2+]CC2)cc1C(F)(F)F"      # where the greedy walk stalled
    closed = "C1CCCCC1c1ccc(N2CC[NH2+]CC2)cc1C(F)(F)F"   # same atoms, one more bond
    aro = aromatisation_targets(closed)
    return [dict(name="chain->closed", start=chain, target=closed),
            dict(name="closed->aromatic", start=closed,
                 target=aro[0] if aro else None)]


@app.local_entrypoint()
def stages(cell: str = "5ht1b_s7_d0.4"):
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    tgt, si, _ = cell.rsplit("_", 2)
    seed = seeds[f"{tgt}_{si}"]["smiles"]
    pair = stage_pair.remote(seed)
    jobs = [dict(seed=p["start"], target=p["target"], max_steps=8)
            for p in pair if p["target"]]
    res = list(compile_target.map(jobs))
    for p, r in zip([p for p in pair if p["target"]], res):
        print(f"\n=== {p['name']} ===")
        print(f"  from   {p['start']}")
        print(f"  to     {p['target']}")
        print(f"  reached={r['reached']} depth={r['depth']} {r.get('halted','')}")
        for s in r.get("steps", []):
            print(f"    {s['rule']:20s} rank={s['rank']:4d} p={s['prob']:.2e}  {s['smiles']}")
    (root / "diagnostics/aryl_stages.json").write_text(json.dumps(res, indent=1))


@app.local_entrypoint()
def run(cell: str = "5ht1b_s7_d0.4", n_targets: int = 4):
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    tgt, si, _ = cell.rsplit("_", 2)
    seed = seeds[f"{tgt}_{si}"]["smiles"]
    targets = make_targets.remote(seed, int(n_targets))
    jobs = [dict(seed=seed, target=t, max_steps=14) for t in targets]
    res = list(compile_target.map(jobs))
    print(f"\nseed: {seed}\n")
    for r in res:
        print(f"target {r['target']}")
        print(f"  reached={r['reached']} depth={r['depth']} {r.get('halted','')}")
        for s in r.get("steps", [])[:14]:
            print(f"    {s['rule']:22s} rank={s['rank']:4d} p={s['prob']:.2e} "
                  f"in_cap={s['in_cap']}")
    (root / "diagnostics/aryl_compile.json").write_text(json.dumps(res, indent=1))
    print("\nwrote diagnostics/aryl_compile.json")


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=60 * 60,
              retries=0, enable_memory_snapshot=True, max_containers=20,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_macro(job: dict) -> list:
    """BUILD_RING_SYSTEM(pendant, 6, carbon_rich, aromatic), end to end.
    Prints every primitive so the trace can be checked against the intended
    grow -> cycle_close -> ring_system_restate."""
    import numpy as np, os, sys
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
    from rdkit.Chem import rdMolDescriptors as rdMD
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import build_ring_system, ring_systems
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

    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed = job["seed"]; sfp = gm.GetFingerprint(Chem.MolFromSmiles(seed))
    out = []
    for k in range(int(job.get("n", 12))):
        r = build_ring_system(enumerate_fn, apply_fn, is_executable, seed,
                              np.random.default_rng(1000 + k),
                              topology="pendant", size=6,
                              composition="carbon_rich", state="aromatic")
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            out.append(dict(ok=False, trace=r["trace"])); continue
        sim = float(DataStructs.TanimotoSimilarity(sfp, gm.GetFingerprint(m)))
        q = float(QED.qed(m)); sa = float(sascorer.calculateScore(m))
        rec = dict(satisfied=r["satisfied"], reason=r.get("reason"),
                   trace=r["trace"], smiles=r["smiles"],
                   heavy=m.GetNumHeavyAtoms(), sys=len(ring_systems(m)),
                   arom=int(rdMD.CalcNumAromaticRings(m)),
                   sim=round(sim, 3), qed=round(q, 3), sa=round(sa, 2),
                   t4=bool(sim >= 0.4 and q >= 0.6 and sa <= 4))
        out.append(rec)
        print(f"{k}: satisfied={rec['satisfied']} arom={rec['arom']} sys={rec['sys']} "
              f"heavy={rec['heavy']} sim={rec['sim']} T4={rec['t4']} "
              f"trace={'->'.join(rec['trace'])}", flush=True)
    return out


@app.local_entrypoint()
def macro(cell: str = "5ht1b_s7_d0.4", n: int = 12):
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    tgt, si, _ = cell.rsplit("_", 2)
    res = run_macro.remote(dict(seed=seeds[f"{tgt}_{si}"]["smiles"], n=int(n)))
    ok = [r for r in res if r.get("t4") and r.get("arom", 0) >= 2]
    print(f"\nT4-feasible with an added aromatic ring: {len(ok)}/{len(res)}")
    for r in ok[:5]:
        print(f"  arom={r['arom']} sys={r['sys']} heavy={r['heavy']} sim={r['sim']} "
              f"qed={r['qed']} sa={r['sa']}")
        print(f"    {'->'.join(r['trace'])}")
        print(f"    {r['smiles']}")
    (root / "diagnostics/macro_endtoend.json").write_text(json.dumps(res, indent=1))


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=60 * 60,
              retries=0, enable_memory_snapshot=True, max_containers=20,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def exact_macro(job: dict) -> dict:
    """BUILD_RING_SYSTEM by descriptor compilation, ranked by frozen R_theta.

    The macro restricts support, it never invents it: A_m(x) = A(x) INTERSECT
    C_m(x), where C_m is cheap graph logic computed from the tracked frontier
    and A(x) is the executor's exact legal support. Nothing here modifies the
    frozen kernel, and an empty intersection returns UNSAT with no fallback.
    """
    import numpy as np, os, sys
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
    from rdkit.Chem import rdMolDescriptors as rdMD
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import (smiles_to_molecular_graph,
        molecular_graph_to_smiles)
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import build_ring_system_exact, ring_systems
    from compose_v4.gates.med_chem_gate import is_executable

    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed = job["seed"]; size = int(job.get("size", 6))
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    sfp = gm.GetFingerprint(Chem.MolFromSmiles(seed))
    m0 = Chem.MolFromSmiles(seed)
    arom0 = int(rdMD.CalcNumAromaticRings(m0)); sys0 = len(ring_systems(m0))
    _cache = {}

    def to_smiles(st):
        try:
            return molecular_graph_to_smiles(st)
        except Exception:
            return None

    def enum_full(st):
        # keyed on the canonical SMILES, but the STATE is what gets threaded
        # through the macro -- a SMILES round trip renumbers atoms, which
        # silently invalidates every tracked slot.
        k = to_smiles(st)
        if k not in _cache:
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            _cache[k] = ([m.executor_rule_name for m in law.marks],
                         [m.action for m in law.marks],
                         np.array([m.probability for m in law.marks], float))
        return _cache[k]

    def apply_fn(st, j):
        fams, acts, _ = enum_full(st)
        try:
            return system.apply(st, fams[j], acts[j])
        except Exception:
            return None

    st0 = pad_molecular_graph(smiles_to_molecular_graph(seed), CANONICAL_SLOTS)
    r = build_ring_system_exact(enum_full, apply_fn, to_smiles, is_executable,
                                st0, size=size, topology="pendant",
                                composition="carbon_rich", state="aromatic",
                                anchor_rank=int(job.get("anchor_rank", 0)))
    r["anchor_rank"] = int(job.get("anchor_rank", 0))
    if r.get("status") != "OK" or not r.get("smiles"):
        return r
    m = Chem.MolFromSmiles(r["smiles"])
    if m is None:
        r["status"] = "UNSAT"; r["stage"] = "unparseable"; return r
    sim = float(DataStructs.TanimotoSimilarity(sfp, gm.GetFingerprint(m)))
    q = float(QED.qed(m)); sa = float(sascorer.calculateScore(m))
    r.update(heavy=m.GetNumHeavyAtoms(), sys=len(ring_systems(m)),
             arom=int(rdMD.CalcNumAromaticRings(m)),
             sim=round(sim, 3), qed=round(q, 3), sa=round(sa, 2),
             t4=bool(sim >= 0.4 and q >= 0.6 and sa <= 4),
             t4_d06=bool(sim >= 0.6 and q >= 0.6 and sa <= 4))
    return r


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=60 * 60,
              retries=0, enable_memory_snapshot=True, max_containers=8,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def state_audit(job: dict) -> dict:
    """What does the PRODUCTION law actually offer at a given state?

    The local primitive fiber and the production marked law are different
    supports -- 229 restate/reorder actions locally against 3 in production at
    the same closed cyclohexyl state -- so any claim about reachability has to
    be measured on the production side, not inferred from the local fiber.
    """
    import numpy as np
    from collections import Counter
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law)

    rt = _runtime(); model = rt["model"]
    st = pad_molecular_graph(smiles_to_molecular_graph(job["smiles"]), CANONICAL_SLOTS)
    law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
    pr = np.array([m.probability for m in law.marks], float)
    order = list(np.argsort(-pr))
    rank = {int(j): r for r, j in enumerate(order)}
    by_exec = Counter(m.executor_rule_name for m in law.marks)
    by_fam = Counter(m.family_name for m in law.marks)
    detail = []
    for j, m in enumerate(law.marks):
        if m.executor_rule_name in ("bond_reorder", "ring_system_restate",
                                    "atom_restate"):
            a = m.action
            detail.append(dict(rule=m.executor_rule_name, rank=rank[j],
                               prob=round(float(pr[j]), 5),
                               cls=type(a).__name__,
                               fields={k: repr(getattr(a, k))[:120] for k in dir(a)
                                       if not k.startswith("_")
                                       and not callable(getattr(a, k, None))}))
    return dict(smiles=job["smiles"], n_marks=len(law.marks),
                by_executor=dict(by_exec), by_family=dict(by_fam),
                enabled=list(law.enabled_families), detail=detail[:40])


@app.local_entrypoint()
def audit(smiles: str = "FC(F)(F)c1cc(N2CC[NH2+]CC2)ccc1C1CCCCC1"):
    root = Path(__file__).resolve().parents[1]
    r = state_audit.remote(dict(smiles=smiles))
    print(f"\nstate: {r['smiles']}\nmarks: {r['n_marks']}")
    print(f"enabled families: {r['enabled']}")
    print("\nby executor rule:")
    for k, v in sorted(r["by_executor"].items(), key=lambda t: -t[1]):
        print(f"  {v:5d}  {k}")
    print("\nrestate / reorder actions actually offered:")
    for d in r["detail"]:
        print(f"  r{d['rank']:<5d} p={d['prob']:.5f}  {d['rule']:22s} {d['fields']}")
    (root / "diagnostics/state_audit.json").write_text(json.dumps(r, indent=1))


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=60 * 60,
              retries=0, enable_memory_snapshot=True, max_containers=8,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def fused_macro(job: dict) -> dict:
    """BUILD_RING_SYSTEM(topology=fused) against the PRODUCTION law.

    Fused aromatic cannot be qualified locally at all: a fused aromatic system
    is ONE correlated electronic block, the shared atoms are already aromatic
    and have no valence capacity for an independent Kekule alternation, and the
    local primitive fiber has no ring_system_restate. Pendant had a
    bond_reorder fallback; fused has none. So the block restate has to come
    from the production law, and R_theta ranks whichever realisations of it are
    legal -- no hand-written naphthalene pattern anywhere.
    """
    import numpy as np, os, sys
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
    from rdkit.Chem import rdMolDescriptors as rdMD
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import (smiles_to_molecular_graph,
        molecular_graph_to_smiles)
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import (build_fused_ring_exact,
        ring_systems, fusable_edges, ring_pair_relations)
    from compose_v4.gates.med_chem_gate import is_executable

    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed = job["seed"]; size = int(job.get("size", 6))
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    sfp = gm.GetFingerprint(Chem.MolFromSmiles(seed))
    _cache = {}

    def to_smiles(st):
        try: return molecular_graph_to_smiles(st)
        except Exception: return None

    def enum_full(st):
        k = to_smiles(st)
        if k not in _cache:
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            _cache[k] = ([m.executor_rule_name for m in law.marks],
                         [m.action for m in law.marks],
                         np.array([m.probability for m in law.marks], float))
        return _cache[k]

    def apply_fn(st, j):
        fams, acts, _ = enum_full(st)
        try: return system.apply(st, fams[j], acts[j])
        except Exception: return None

    st0 = pad_molecular_graph(smiles_to_molecular_graph(seed), CANONICAL_SLOTS)
    r = build_fused_ring_exact(enum_full, apply_fn, to_smiles, is_executable, st0,
                               size=size, composition=job.get("composition", "carbon_rich"),
                               state=job.get("ring_state", "aromatic"),
                               anchor_rank=int(job.get("anchor_rank", 0)))
    r["anchor_rank"] = int(job.get("anchor_rank", 0))
    r["ring_state"] = job.get("ring_state", "aromatic")
    if r.get("status") != "OK" or not r.get("smiles"):
        return r
    m = Chem.MolFromSmiles(r["smiles"])
    if m is None:
        r["status"], r["stage"] = "UNSAT", "unparseable"; return r
    sim = float(DataStructs.TanimotoSimilarity(sfp, gm.GetFingerprint(m)))
    q = float(QED.qed(m)); sa = float(sascorer.calculateScore(m))
    r.update(heavy=m.GetNumHeavyAtoms(), sys=len(ring_systems(m)),
             arom=int(rdMD.CalcNumAromaticRings(m)),
             relations=dict(ring_pair_relations(r["smiles"])),
             sim=round(sim, 3), qed=round(q, 3), sa=round(sa, 2),
             t4=bool(sim >= 0.4 and q >= 0.6 and sa <= 4),
             t4_d06=bool(sim >= 0.6 and q >= 0.6 and sa <= 4))
    return r


@app.local_entrypoint()
def fused(cell: str = "parp1_s0", n: int = 8, ring_state: str = "aromatic"):
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    seed = seeds[cell]["smiles"]
    # two states per edge so a failure isolates to the electronic step
    jobs = []
    for i in range(int(n)):
        jobs.append(dict(seed=seed, size=6, anchor_rank=i % 4,
                         ring_state=("aromatic" if i < int(n) // 2 else "saturated")))
    res = list(fused_macro.map(jobs))
    ok = [r for r in res if r.get("status") == "OK"]
    strict = [r for r in ok if r.get("fused_gain") == 1
              and r.get("spiro_gain") == 0 and r.get("bridged_gain") == 0]
    print(f"\nseed {seed}")
    print(f"OK {len(ok)}/{len(res)}   contract-clean (fused+1, spiro0, bridged0): {len(strict)}/{len(res)}")
    for r in res:
        st = r.get("ring_state")
        print(f"  a{r.get('anchor_rank')} {st:9s} {r.get('status'):5s} {str(r.get('stage') or ''):18s} "
              f"fused+{r.get('fused_gain')} spiro+{r.get('spiro_gain')} bridged+{r.get('bridged_gain')} "
              f"arom+{r.get('arom_gain')} sim={r.get('sim')} qed={r.get('qed')} sa={r.get('sa')} "
              f"T4={r.get('t4')}/{r.get('t4_d06')}")
        if r.get("trace"): print(f"      {'->'.join(r['trace'])}")
        if r.get("smiles"): print(f"      {r['smiles']}")
    (root / "diagnostics/fused_macro.json").write_text(json.dumps(res, indent=1))
    print(f"wrote diagnostics/fused_macro.json")


@app.local_entrypoint()
def exact(cell: str = "5ht1b_s7_d0.4", n: int = 8):
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    tgt, si, _ = cell.rsplit("_", 2)
    seed = seeds[f"{tgt}_{si}"]["smiles"]
    jobs = [dict(seed=seed, size=6, anchor_rank=i) for i in range(int(n))]
    res = list(exact_macro.map(jobs))
    ok = [r for r in res if r.get("status") == "OK"]
    t4 = [r for r in ok if r.get("t4") and r.get("arom_gain", 0) >= 1
          and r.get("sys_gain", 0) >= 1]
    print(f"\nstatus: {json.dumps({s: sum(1 for r in res if r.get('status')==s or r.get('stage')==s) for s in ('OK','grow','closure','aromatise')})}")
    print(f"systems 2->3 and aromatics 1->2 and T4-feasible: {len(t4)}/{len(res)}")
    for r in res[:8]:
        print(f"  a{r.get('anchor_rank')} {r.get('status'):5s} {r.get('stage','') or '':14s} "
              f"sys_gain={r.get('sys_gain')} arom_gain={r.get('arom_gain')} "
              f"sim={r.get('sim')} T4={r.get('t4')} trace={'->'.join(r.get('trace',[]))}")
        if r.get("smiles"): print(f"        {r['smiles']}")
    (root / "diagnostics/exact_macro.json").write_text(json.dumps(res, indent=1))


@app.function(image=image, cpu=(2.0, 2.0), memory=int(4 * 1024), timeout=3 * 60 * 60,
              retries=0, max_containers=24,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def frontier_job(job: dict) -> dict:
    """Phase 1 for one cell: cheap feasible frontier, ZERO docking calls.

    Parallelised because 20 T4 cells start outside the feasible region (9 of 15
    seeds are below the QED gate) and one frontier took 2667 s locally --
    sequentially that is ~15 hours. This is CPU-only and embarrassingly
    parallel across cells, so it belongs on the fleet.

    Internal compute is REPORTED, not hidden: oracle calls are zero here, but
    state expansions and wall time are recorded so the panel can show both
    axes rather than implying oracle calls are the only cost.
    """
    import time
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.control.feasibility_frontier import build_frontier

    t0 = time.time()
    feas, stats = build_frontier(job["seed"], float(job["delta"]),
                                 target=int(job.get("target", 80)),
                                 max_depth=int(job.get("max_depth", 7)),
                                 beam=int(job.get("beam", 140)),
                                 diversity_frac=0.35)
    return dict(cell=job["cell"], delta=float(job["delta"]), seed=job["seed"],
                n_feasible=len(feas), stats=stats,
                seconds=round(time.time() - t0, 1),
                molecules=[dict(smiles=k,
                                **{kk: (round(float(vv), 4) if isinstance(vv, float) else vv)
                                   for kk, vv in v.items()})
                           for k, v in feas.items()])


@app.local_entrypoint()
def frontiers(target: int = 80):
    """Build every Phase-1 frontier the 30-cell panel needs, in parallel."""
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "diagnostics/panel30_plan.json").read_text())
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    jobs = []
    for cell in plan["phase1"]:
        tgt, si, dl = cell.rsplit("_", 2)
        jobs.append(dict(cell=cell, delta=float(dl[1:]),
                         seed=seeds[f"{tgt}_{si}"]["smiles"], target=target))
    print(f"building {len(jobs)} feasibility frontiers in parallel (0 oracle calls)")
    tot_sec = tot_exp = 0
    for r in frontier_job.map(jobs):
        p = root / f"diagnostics/frontier_{r['cell']}.json"
        p.write_text(json.dumps(r, indent=1))
        tot_sec += r["seconds"]; tot_exp += int(r["stats"].get("expanded", 0))
        print(f"  {r['cell']:18s} {r['n_feasible']:4d} feasible  "
              f"{r['seconds']:7.0f}s  {r['stats'].get('expanded',0):8d} expansions  "
              f"seed_shortfall {r['stats'].get('seed_shortfall',0):.3f}", flush=True)
    print("")
    print("PHASE-1 INTERNAL COMPUTE (reported separately from oracle calls):")
    print(f"  total wall {tot_sec/3600:.2f} core-hours   total expansions {tot_exp:,}")
    print(f"  oracle calls consumed: 0")


# =============================================================================
# PHASE-1 FRONTIERS, v2 --- REMOTE-SIDE ATOMIC PERSISTENCE.
#
# WHY THIS EXISTS.  The v1 path above (frontier_job + frontiers) persisted only
# in the LOCAL client: `for r in frontier_job.map(jobs): p.write_text(...)`.
# Two defects compounded on 2026-08-26 and cost a run:
#   1. `.map()` is ORDERED.  parp1_s2 (17 heavy atoms, ~2 min) were jobs 0-1 and
#      landed; job 2 was fa7_s3 (32 atoms, ~18 min), so the iterator blocked
#      head-of-line.  Jobs 3-19 could have been finishing and still could not be
#      written, because the writer only sees what the ordered iterator yields.
#   2. Persistence depended on that client.  When it ended, every in-flight
#      frontier was discarded -- no server-side copy existed.
#
# THE FIX, and the invariant it must hold:
#   * EVERY frontier_seed_job writes its own provenance-tagged artifact to the
#     shared Volume and COMMITS IT BEFORE RETURNING.  Persistence therefore
#     never depends on iterator order, on the caller, or on the caller living.
#   * order_outputs=False is for MONITORING ONLY.  Nothing is persisted by the
#     consumer loop.
#   * Restart is idempotent: a job that finds a valid artifact for its own
#     config skips the work and reports skipped=True.
#
# DELTA DEDUPLICATION.  Feasibility is QED>=0.6, SA<=4, sim>=delta; only the
# similarity term depends on delta, and sim>=0.6 implies sim>=0.4.  So a
# delta=0.6 frontier is a valid delta=0.4 frontier.  We build 0.6 FIRST and
# reuse it for 0.4 when it carries enough diverse feasible molecules, which is
# why this is one job per SEED rather than one per cell.  The reuse is VERIFIED
# per molecule (assert min sim >= 0.4), never assumed.
#
# The frozen controller is untouched: build_frontier is called with the same
# parameters as v1.
# =============================================================================

FRONTIER_DIR = Path(RUN_ROOT) / "t4_frontiers"
FRONTIER_PARAMS: dict[str, Any] = dict(target=80, max_depth=7, beam=140,
                                       diversity_frac=0.35)
# A delta=0.6 frontier is reused for delta=0.4 only if it holds at least this
# many distinct feasible molecules.  Set to the frontier target so reuse never
# hands Phase 2 a thinner start than a purpose-built frontier would.
FRONTIER_REUSE_MIN = 80


def _frontier_code_sha() -> str:
    """sha256 of the frontier builder, so a code change invalidates artifacts."""
    import hashlib

    from compose_v4.control import feasibility_frontier as _ff
    return hashlib.sha256(Path(_ff.__file__).read_bytes()).hexdigest()[:16]


def _frontier_key(seed: str, delta: float) -> str:
    import hashlib

    payload = json.dumps({"seed": seed, "delta": float(delta),
                          "params": FRONTIER_PARAMS,
                          "code": _frontier_code_sha()}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _frontier_read(cell: str, key: str) -> dict[str, Any] | None:
    """Return an existing artifact iff it is COMPLETE and matches `key`."""
    p = FRONTIER_DIR / f"{cell}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    prov = d.get("provenance") or {}
    if not d.get("complete") or prov.get("config_sha256") != key:
        return None
    return d


def _frontier_write(cell: str, payload: dict[str, Any]) -> str:
    """Atomically publish one frontier artifact, then commit the Volume.

    Write-temp-then-rename means a reader never observes a partial file, and
    os.replace is atomic within the volume.  commit() is what makes it survive
    the container, so it MUST happen before the job returns -- that is the
    property the sentinel checks.
    """
    import os
    import uuid

    FRONTIER_DIR.mkdir(parents=True, exist_ok=True)
    final = FRONTIER_DIR / f"{cell}.json"
    tmp = FRONTIER_DIR / f".{cell}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, final)
    artifact_volume.commit()
    return str(final)


@app.function(image=image, cpu=(2.0, 2.0), memory=int(4 * 1024), timeout=6 * 60 * 60,
              retries=0, max_containers=24,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def frontier_seed_job(job: dict) -> dict:
    """Both delta cells for ONE seed. Persists remotely before returning."""
    import time

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.control.feasibility_frontier import build_frontier

    artifact_volume.reload()
    seed, base = job["seed"], job["cell_base"]
    out: dict[str, Any] = {"cell_base": base, "cells": {}}

    def _molecules(feas):
        return [dict(smiles=k,
                     **{kk: (round(float(vv), 4) if isinstance(vv, float) else vv)
                        for kk, vv in v.items()})
                for k, v in feas.items()]

    def _publish(delta, feas, stats, seconds, reuse_of=None):
        cell = f"{base}_d{delta}"
        mols = _molecules(feas)
        sims = [m["sim"] for m in mols if m.get("sim") is not None]
        # The reuse is only sound if every molecule really clears this delta.
        if sims:
            assert min(sims) >= float(delta) - 1e-9, (
                f"{cell}: molecule with sim {min(sims)} < delta {delta}")
        payload = {
            "cell": cell, "delta": float(delta), "seed": seed,
            "n_feasible": len(mols), "stats": stats, "seconds": seconds,
            "complete": True,
            "provenance": {
                "config_sha256": _frontier_key(seed, delta),
                "code_sha256": _frontier_code_sha(),
                "params": FRONTIER_PARAMS,
                "reuse_of": reuse_of,
                "min_sim": (round(min(sims), 4) if sims else None),
                "oracle_calls": 0,
            },
            "molecules": mols,
        }
        path = _frontier_write(cell, payload)
        out["cells"][cell] = {"n_feasible": len(mols), "seconds": seconds,
                              "reuse_of": reuse_of, "path": path,
                              "skipped": False}
        return payload

    # ---- delta = 0.6 first: it is the strict subset and may serve both.
    key06 = _frontier_key(seed, 0.6)
    have06 = _frontier_read(f"{base}_d0.6", key06)
    if have06 is not None:
        out["cells"][f"{base}_d0.6"] = {"n_feasible": have06["n_feasible"],
                                        "seconds": have06.get("seconds"),
                                        "reuse_of": None, "skipped": True}
        feas06_n = have06["n_feasible"]
        mols06 = have06["molecules"]
    else:
        t0 = time.time()
        feas06, stats06 = build_frontier(seed, 0.6, **FRONTIER_PARAMS)
        p = _publish(0.6, feas06, stats06, round(time.time() - t0, 1))
        feas06_n, mols06 = p["n_feasible"], p["molecules"]

    # ---- delta = 0.4: reuse the 0.6 frontier when it is rich enough.
    key04 = _frontier_key(seed, 0.4)
    have04 = _frontier_read(f"{base}_d0.4", key04)
    if have04 is not None:
        out["cells"][f"{base}_d0.4"] = {"n_feasible": have04["n_feasible"],
                                        "seconds": have04.get("seconds"),
                                        "reuse_of": (have04.get("provenance") or {}).get("reuse_of"),
                                        "skipped": True}
    elif feas06_n >= FRONTIER_REUSE_MIN:
        feas = {m["smiles"]: {k: v for k, v in m.items() if k != "smiles"}
                for m in mols06}
        _publish(0.4, feas, {"reused_from": f"{base}_d0.6"}, 0.0,
                 reuse_of=f"{base}_d0.6")
    else:
        t0 = time.time()
        feas04, stats04 = build_frontier(seed, 0.4, **FRONTIER_PARAMS)
        _publish(0.4, feas04, stats04, round(time.time() - t0, 1))

    return out


def _phase1_seed_jobs() -> list[dict]:
    """One job per SEED (not per cell): the seed's two delta cells share work."""
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "diagnostics/panel30_plan.json").read_text())
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    bases: list[str] = []
    for cell in plan["phase1"]:
        base = cell.rsplit("_", 1)[0]
        if base not in bases:
            bases.append(base)
    return [dict(cell_base=b, seed=seeds[b]["smiles"]) for b in bases]


@app.function(image=image, cpu=(0.25, 0.25), memory=1024, timeout=20 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def frontier_status() -> dict:
    """Read-only view of what is actually ON the Volume. No local state."""
    artifact_volume.reload()
    out: dict[str, Any] = {"present": {}, "partial": []}
    if not FRONTIER_DIR.exists():
        return out
    for p in sorted(FRONTIER_DIR.iterdir()):
        if p.name.endswith(".tmp"):
            out["partial"].append(p.name)
            continue
        if p.suffix != ".json":
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            out["partial"].append(p.name)
            continue
        prov = d.get("provenance") or {}
        out["present"][d.get("cell", p.stem)] = {
            "n_feasible": d.get("n_feasible"),
            "seconds": d.get("seconds"),
            "complete": bool(d.get("complete")),
            "reuse_of": prov.get("reuse_of"),
            "min_sim": prov.get("min_sim"),
        }
    return out


@app.local_entrypoint()
def frontier_launch(only: str = "") -> None:
    """SERVER-SIDE spawn. Returns immediately; work does not need this client.

    `only` is a comma-separated list of cell_base names, for the sentinel.
    """
    jobs = _phase1_seed_jobs()
    if only:
        want = {s.strip() for s in only.split(",") if s.strip()}
        jobs = [j for j in jobs if j["cell_base"] in want]
    print(f"spawning {len(jobs)} seed-level frontier jobs SERVER-SIDE "
          f"(each persists its own artifact to the Volume before returning)")
    for j in jobs:
        print(f"  {j['cell_base']}")
    frontier_seed_job.spawn_map(jobs)
    print("\nspawned. This client is no longer required -- killing it cannot")
    print("delete or cancel a committed artifact. Poll with:")
    print("  modal run modal_apps/aryl_compile_app.py::frontier_report")


@app.local_entrypoint()
def frontier_report() -> None:
    """Print Volume-side frontier state, and pull artifacts down locally."""
    root = Path(__file__).resolve().parents[1]
    st = frontier_status.remote()
    pres = st["present"]
    print(f"frontiers on Volume: {len(pres)}")
    for cell in sorted(pres):
        v = pres[cell]
        tag = f"reuse of {v['reuse_of']}" if v.get("reuse_of") else f"{v['seconds']}s"
        print(f"  {cell:<20} {str(v['n_feasible']):>5} feasible  "
              f"min_sim {v['min_sim']}  {tag}")
    if st["partial"]:
        print(f"  in-flight temp files (never read by consumers): {st['partial']}")
    plan = json.loads((root / "diagnostics/panel30_plan.json").read_text())
    missing = [c for c in plan["phase1"] if c not in pres]
    print(f"\n{len(pres)}/{len(plan['phase1'])} phase-1 cells present; "
          f"missing: {missing if missing else 'none'}")


@app.function(image=image, cpu=(1.0, 1.0), memory=int(4 * 1024), timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def stoich_parity(job: dict) -> dict:
    """ALL-CARBON PARITY GATE for the exact-stoichiometry quota.

    C6 is the legacy behaviour. If requesting it as a QUOTA differs from the
    legacy `composition="carbon_rich"` element set, then the refactor changed
    more than composition expressivity, and broad semantics must not be
    enabled. Compares product SMILES and descriptor trace, per topology.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    from compose_v4.chem.molecular_graph import (molecular_graph_to_smiles,
                                                 smiles_to_molecular_graph)
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.gates.med_chem_gate import is_executable
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import (build_fused_ring_exact,
                                                 build_ring_system_exact)

    rt = _runtime(); model, system = rt["model"], rt["system"]
    out = []
    for seed in job["seeds"]:
        _cache = {}

        def to_smiles(st):
            try:
                return molecular_graph_to_smiles(st)
            except Exception:
                return None

        def enum_full(st):
            k = to_smiles(st)
            if k not in _cache:
                law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
                _cache[k] = ([m.executor_rule_name for m in law.marks],
                             [m.action for m in law.marks],
                             np.array([m.probability for m in law.marks], float))
            return _cache[k]

        def apply_fn(st, j):
            fams, acts, _ = enum_full(st)
            try:
                return system.apply(st, fams[j], acts[j])
            except Exception:
                return None

        st0 = pad_molecular_graph(smiles_to_molecular_graph(seed), CANONICAL_SLOTS)
        for topo, state in (("pendant", "aromatic"), ("pendant", "saturated"),
                            ("fused", "aromatic"), ("fused", "saturated")):
            common = dict(size=6, state=state, anchor_rank=0)
            if topo == "fused":
                leg = build_fused_ring_exact(enum_full, apply_fn, to_smiles,
                                             is_executable, st0,
                                             composition="carbon_rich", **common)
                new = build_fused_ring_exact(enum_full, apply_fn, to_smiles,
                                             is_executable, st0,
                                             composition="carbon_rich",
                                             stoich=(("C", 6),), **common)
            else:
                leg = build_ring_system_exact(enum_full, apply_fn, to_smiles,
                                              is_executable, st0,
                                              topology=topo,
                                              composition="carbon_rich", **common)
                new = build_ring_system_exact(enum_full, apply_fn, to_smiles,
                                              is_executable, st0,
                                              topology=topo,
                                              composition="carbon_rich",
                                              stoich=(("C", 6),), **common)
            same = (leg.get("status") == new.get("status")
                    and leg.get("smiles") == new.get("smiles")
                    and leg.get("trace") == new.get("trace"))
            out.append(dict(seed=seed, topology=topo, state=state,
                            legacy_status=leg.get("status"),
                            quota_status=new.get("status"),
                            legacy_smiles=leg.get("smiles"),
                            quota_smiles=new.get("smiles"),
                            trace_match=(leg.get("trace") == new.get("trace")),
                            identical=bool(same)))
    return {"rows": out, "n_identical": sum(1 for r in out if r["identical"]),
            "n": len(out)}


@app.local_entrypoint()
def parity() -> None:
    import json as _j
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    seeds = [s["smiles"] for s in
             _j.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())][:4]
    r = stoich_parity.remote({"seeds": seeds})
    print(f"ALL-CARBON PARITY: {r['n_identical']}/{r['n']} identical\n")
    for row in r["rows"]:
        flag = "OK " if row["identical"] else "DIFF"
        print(f"  {flag} {row['topology']:<8}{row['state']:<10}"
              f"{str(row['legacy_status']):<8}->{str(row['quota_status']):<8}"
              f"trace_match={row['trace_match']}")
        if not row["identical"]:
            print(f"       legacy: {row['legacy_smiles']}")
            print(f"       quota : {row['quota_smiles']}")
    (root / "diagnostics/stoich_c6_parity.json").write_text(_j.dumps(r, indent=1))
    print(f"\nwrote diagnostics/stoich_c6_parity.json")


@app.function(image=image, cpu=(1.0, 1.0), memory=int(4 * 1024), timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def replay_audit(job: dict) -> dict:
    """REPLAY AUDIT. No learning, no prior, no comparator target.

    Forces linked/6/C6/aromatic repeatedly from the seed, exactly as B's
    winning trajectory did, and asks at each state:
      - does broad C's realize() contain the successor B actually took?
      - what is that successor's R_theta rank and mass among the alternatives?
      - what would broad C's CURRENT realization rule (anchor_rank=0) pick?

    Distinguishes representation/realization failure (B's successor absent or
    badly ranked) from proposal-learning failure (present and well ranked, but
    the semantic tuple is never proposed).
    """
    import numpy as np
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    from compose_v4.chem.molecular_graph import (molecular_graph_to_smiles,
                                                 smiles_to_molecular_graph)
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control import semantic_actions as sa
    from compose_v4.control.macro_engine import (build_ring_system_exact,
                                                 fusable_edges,
                                                 match_growth_descriptors,
                                                 predict_fused_product,
                                                 predict_pendant_product)
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law)
    from compose_v4.gates.med_chem_gate import is_executable

    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed = job["seed"]; target = job.get("target_smiles")
    can = lambda s: (Chem.MolToSmiles(Chem.MolFromSmiles(s))
                     if s and Chem.MolFromSmiles(s) else None)

    def to_smiles(st):
        try: return molecular_graph_to_smiles(st)
        except Exception: return None

    _c = {}
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
        try: return system.apply(st, fams[j], acts[j])
        except Exception: return None

    req = sa.RingRequest("linked", 6, sa.normalize_stoich({}, 6), "aromatic")
    steps = []
    cur = job.get("start_state") or seed
    for step in range(int(job.get("n_steps", 4))):
        stx = pad_molecular_graph(smiles_to_molecular_graph(cur), CANONICAL_SLOTS)
        # (a) every executable canonical realization of the FORCED request
        reals, why = sa.realize(req, stx, seed, predict_fn=predict_pendant_product,
                                fused_fn=predict_fused_product,
                                edges_fn=fusable_edges, max_realizations=200)
        groups = sa.aggregate_canonical(reals) if reals else []
        # (b) R_theta mass over the first insertions that begin this request
        fams, acts, probs = enum_full(stx)
        idx = match_growth_descriptors(fams, acts, probs, None, {2}, anchors=None)
        ranked = sorted(idx, key=lambda j: -float(probs[j]))
        # (c) what the CURRENT production rule (anchor_rank=0) actually builds
        rr = build_ring_system_exact(enum_full, apply_fn, to_smiles, is_executable,
                                     stx, size=6, topology="pendant",
                                     composition="carbon_rich", state="aromatic",
                                     anchor_rank=0)
        built = rr.get("smiles") if rr.get("status") == "OK" else None
        tc = can(target) if target else None
        hit = [i for i, g in enumerate(groups) if can(g["canonical"]) == tc]
        steps.append(dict(step=step, target_index=(hit[0] if hit else None),
                          state=cur, n_realizations=len(reals),
                          n_canonical=len(groups), unsat=why,
                          carbon_first_insertions=len(idx),
                          carbon_mass=float(sum(float(probs[j]) for j in ranked)),
                          top_carbon_prob=(float(probs[ranked[0]]) if ranked else None),
                          endpoints=[g["canonical"] for g in groups[:40]],
                          production_built=built,
                          production_stage=rr.get("stage")))
        if not built:
            break
        cur = built
    return dict(seed=seed, target=target, steps=steps, final=cur,
                target_can=can(target),
                final_is_target=(can(cur) == can(target) if target else None))


@app.local_entrypoint()
def replay(start_state: str = "", n_steps: int = 2) -> None:
    import json as _j
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s["smiles"]
             for s in _j.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    tgt = "FC(F)(c1ccc(-c2ccccc2)cc1)c1cccc(N2CC[NH2+]CC2)c1"
    r = replay_audit.remote({"seed": seeds["5ht1b_s7"], "target_smiles": tgt,
                             "start_state": (start_state or None),
                             "n_steps": int(n_steps)})
    print(f"SEED   {r['seed']}")
    print(f"TARGET {tgt}   (B r12 endpoint, -11.30)\n")
    for s in r["steps"]:
        print(f"--- step {s['step']}  state {s['state']}")
        print(f"    forced linked/6/C6/aromatic -> {s['n_realizations']} realizations, "
              f"{s['n_canonical']} distinct canonical endpoints  unsat={s['unsat']}")
        print(f"    R_theta carbon first-insertions: {s['carbon_first_insertions']}  "
              f"mass {s['carbon_mass']:.4f}  top {s['top_carbon_prob']}")
        print(f"    production rule (anchor_rank=0) built: {s['production_built']}")
        ti = s.get("target_index")
        print(f"    IS B's TARGET among the canonical endpoints? "
              f"{'YES at index ' + str(ti) if ti is not None else 'no'}")
    print(f"\nfinal after forced chain: {r['final']}")
    print(f"equals B target? {r.get('final_is_target')}")
    (root / "diagnostics/replay_audit_5ht1b.json").write_text(_j.dumps(r, indent=1))
    print("wrote diagnostics/replay_audit_5ht1b.json")


@app.function(image=image, cpu=(1.0, 1.0), memory=int(4 * 1024), timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def piperidine_reachability(job: dict) -> dict:
    """Is Ar-N1CCCCC1 reachable from a narrow-B linked/6/C/aromatic product?

    STATE IS THREADED THROUGHOUT. A SMILES round trip renumbers slots and would
    destroy the new-ring atom provenance this test depends on, so the ring is
    built here rather than reconstructed from the builder's SMILES output.

    Exhaustive: every ring-local ring_system_restate successor, then every
    atom_restate successor on the six new-ring atoms. No R_theta truncation and
    no docking anywhere in the reachability decision -- R_theta rank/mass are
    RECORDED for the required actions but never used to prune.
    """
    import numpy as np
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    from compose_v4.chem.molecular_graph import (molecular_graph_to_smiles,
                                                 smiles_to_molecular_graph)
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.macro_engine import (RingFrontier,
                                                 match_growth_descriptors,
                                                 match_closure_descriptors)
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law)

    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed = job["seed"]
    def sm(st):
        try: return molecular_graph_to_smiles(st)
        except Exception: return None
    _c = {}
    def law(st):
        k = sm(st)
        if k not in _c:
            L = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            _c[k] = ([m.executor_rule_name for m in L.marks],
                     [m.action for m in L.marks],
                     np.array([m.probability for m in L.marks], float))
        return _c[k]
    def apply_j(st, j):
        f, a, _ = law(st)
        try: return system.apply(st, f[j], a[j])
        except Exception: return None

    # ---- build one linked/6/C/aromatic ring, KEEPING the state + atom ids ----
    cur = pad_molecular_graph(smiles_to_molecular_graph(seed), CANONICAL_SLOTS)
    fr = RingFrontier(size=6, topology="pendant")
    build = []
    for step in range(6):
        f, a, p = law(cur)
        tip = fr.path[-1] if fr.path else None
        order = None if step == 0 else (2 if step % 2 == 1 else 1)
        cands = match_growth_descriptors(f, a, p, tip, {2}, bond_order=order)
        if not cands:
            cands = match_growth_descriptors(f, a, p, tip, {2})
        placed = False
        for j in cands[:8]:
            y = apply_j(cur, j)
            if y is None or sm(y) is None:
                continue
            fr.observe_insert(a[j]); cur = y; placed = True
            build.append(dict(step=step, slot=int(a[j].slot), smiles=sm(cur)))
            break
        if not placed:
            return {"error": f"growth failed at step {step}", "build": build}
    # close the ring
    f, a, p = law(cur)
    pairs = fr.closure_pairs() if hasattr(fr, "closure_pairs") else []
    cl = match_closure_descriptors(f, a, pairs) if pairs else []
    for j in cl[:8]:
        y = apply_j(cur, j)
        if y is not None and sm(y):
            cur = y; break
    ring_ids = [int(x) for x in fr.path]
    base = sm(cur)
    from rdkit.Chem import rdMolDescriptors as _rdMD0
    _bm = Chem.MolFromSmiles(base)
    _base_arom = int(_rdMD0.CalcNumAromaticRings(_bm)) if _bm else 0

    # ---- exhaustive ring-local restate successors (dearomatization) ---------
    f, a, p = law(cur)
    rs = [j for j, fam in enumerate(f) if fam == "ring_system_restate"]
    order_all = sorted(range(len(p)), key=lambda k: -float(p[k]))
    tot = float(p.sum()) or 1.0
    sat = []
    for j in rs:
        ch = getattr(a[j], "changes", None) or ()
        touched = {int(getattr(c, "a", -1)) for c in ch} | {int(getattr(c, "b", -1)) for c in ch}
        if not touched or not (touched & set(ring_ids)):
            continue
        y = apply_j(cur, j)
        ys = sm(y) if y is not None else None
        if ys is None:
            continue
        m = Chem.MolFromSmiles(ys)
        if m is None:
            continue
        # Count aromatic rings and compare with the base. Asking "is ANY
        # 6-ring aromatic" is wrong here: the scaffold's own benzene stays
        # aromatic, so that test reports True even when the NEW ring has been
        # fully dearomatized, and silently skips every saturated successor.
        from rdkit.Chem import rdMolDescriptors as _rdMD
        arom = int(_rdMD.CalcNumAromaticRings(m)) >= _base_arom
        sat.append(dict(j=j, smiles=ys, still_aromatic=arom,
                        rank=order_all.index(j), mass=float(p[j]) / tot,
                        state=y))
    # ---- from every saturated successor, every C->N retype on ring atoms ----
    hits = []
    dbg = {"sat_seen": 0, "sat_skipped_aromatic": 0, "restate_fam": 0,
           "on_ring": 0, "target_N": 0, "applied_ok": 0}
    for s in sat:
        dbg["sat_seen"] += 1
        if s["still_aromatic"]:
            dbg["sat_skipped_aromatic"] += 1
            continue
        st2 = s["state"]
        f2, a2, p2 = law(st2)
        ord2 = sorted(range(len(p2)), key=lambda k: -float(p2[k]))
        tot2 = float(p2.sum()) or 1.0
        for j2, fam2 in enumerate(f2):
            if "atom_restate" not in str(fam2):
                continue
            dbg["restate_fam"] += 1
            # SemanticAtomRestate carries `v` (the atom) and
            # `target_class_index` (an index into ATOM_VALENCE_CLASSES) -- NOT
            # `slot`/`atom_type`. Reading the wrong names made every one of the
            # 158 restate actions fail the filter and reported 0 reachable
            # retypes, which looked like a chemistry result and was a typo.
            from compose_v4.chem.molecular_graph import ATOM_VALENCE_CLASSES
            slot = int(getattr(a2[j2], "v", -1))
            if slot not in ring_ids:
                continue
            dbg["on_ring"] += 1
            tci = int(getattr(a2[j2], "target_class_index", -1))
            if not (0 <= tci < len(ATOM_VALENCE_CLASSES)):
                continue
            if int(ATOM_VALENCE_CLASSES[tci][0]) != 3:          # element N
                continue
            dbg["target_N"] += 1
            y2 = apply_j(st2, j2)
            ys2 = sm(y2) if y2 is not None else None
            if ys2 is None:
                continue
            m2 = Chem.MolFromSmiles(ys2)
            if m2 is None:
                continue
            dbg["applied_ok"] += 1
            pip = m2.HasSubstructMatch(Chem.MolFromSmarts("[c,C]-[NX3;R]1[CH2][CH2][CH2][CH2][CH2]1"))
            hits.append(dict(dearom_j=s["j"], dearom_rank=s["rank"],
                             dearom_mass=s["mass"], dearom_smiles=s["smiles"],
                             retype_slot=slot, retype_rank=ord2.index(j2),
                             retype_mass=float(p2[j2]) / tot2,
                             final=ys2, piperidine_match=bool(pip),
                             canonical=Chem.MolToSmiles(m2)))
    return dict(seed=seed, base=base, ring_ids=ring_ids, build=build, dbg=dbg,
                n_restate_ring_local=len(sat),
                n_dearomatized=sum(1 for s in sat if not s["still_aromatic"]),
                restate=[{k: v for k, v in s.items() if k != "state"} for s in sat],
                n_retype_hits=len(hits),
                hits=sorted(hits, key=lambda h: (not h["piperidine_match"],
                                                 h["dearom_rank"] + h["retype_rank"]))[:20])


@app.local_entrypoint()
def reach() -> None:
    import json as _j
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s["smiles"]
             for s in _j.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    r = piperidine_reachability.remote({"seed": seeds["5ht1b_s7"]})
    if r.get("error"):
        print("ERROR:", r["error"]); print(r.get("build")); return
    print(f"base (narrow-B linked/6/C/aromatic product): {r['base']}")
    print(f"new-ring atom ids (state-threaded, no SMILES round trip): {r['ring_ids']}")
    print(f"\nring-local ring_system_restate successors: {r['n_restate_ring_local']}"
          f"   of which dearomatized: {r['n_dearomatized']}")
    for s in r["restate"][:6]:
        print(f"   rank {s['rank']:>4} mass {s['mass']:.5f} arom={s['still_aromatic']}  {s['smiles']}")
    print(f"\nC->N retypes on new-ring atoms from saturated states: {r['n_retype_hits']}")
    for h in r["hits"][:8]:
        print(f"   dearom(rank {h['dearom_rank']}, mass {h['dearom_mass']:.5f})"
              f" + retype slot {h['retype_slot']} (rank {h['retype_rank']}, mass {h['retype_mass']:.5f})"
              f"  piperidine={h['piperidine_match']}")
        print(f"      {h['canonical']}")
    hit = [h for h in r["hits"] if h["piperidine_match"]]
    print(f"\nPIPERIDINE MOTIF REACHABLE IN 2 ACTIONS? {'YES' if hit else 'NO'}"
          f"   ({len(hit)} matching paths)")
    (root / "diagnostics/piperidine_reachability.json").write_text(_j.dumps(r, indent=1))
    print("wrote diagnostics/piperidine_reachability.json")


@app.function(image=image, cpu=(1.0, 1.0), memory=int(4 * 1024), timeout=30 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def action_schema(job: dict) -> dict:
    """What do the restate/retype actions actually look like at a given state?

    Written because a reachability test reported zero C->N retypes, and a
    filter that assumes the wrong attribute names would report exactly that.
    """
    import numpy as np
    from collections import Counter
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law)
    rt = _runtime(); model = rt["model"]
    st = pad_molecular_graph(smiles_to_molecular_graph(job["smiles"]),
                             CANONICAL_SLOTS)
    L = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
    fams = [m.executor_rule_name for m in L.marks]
    acts = [m.action for m in L.marks]
    fam_counts = dict(Counter(fams))
    samples = {}
    for fam in set(fams):
        j = fams.index(fam)
        a = acts[j]
        samples[fam] = {"type": type(a).__name__,
                        "attrs": {k: str(getattr(a, k))[:60]
                                  for k in dir(a)
                                  if not k.startswith("_") and not callable(getattr(a, k))}}
    return {"smiles": job["smiles"], "n_marks": len(fams),
            "families": fam_counts, "samples": samples}


@app.local_entrypoint()
def schema(smiles: str = "FC(F)(F)c1cc(N2CC[NH2+]CC2)ccc1C1CCCCC1") -> None:
    import json as _j
    r = action_schema.remote({"smiles": smiles})
    print(f"state: {r['smiles']}   marks: {r['n_marks']}")
    print(f"families: {r['families']}\n")
    for fam, s in sorted(r["samples"].items()):
        print(f"  {fam}  ({s['type']})")
        for k, v in s["attrs"].items():
            print(f"      {k} = {v}")
    from pathlib import Path as _P
    _P("diagnostics/action_schema.json").write_text(_j.dumps(r, indent=1))


@app.local_entrypoint()
def retype_census(smiles: str = "FC(F)(F)c1cc(N2CC[NH2+]CC2)ccc1C1CCCCC1",
                  ring_ids: str = "16,17,18,19,20,21") -> None:
    """Census of atom_restate_semantic by (on-new-ring?, target element).

    Distinguishes 'N retype is not offered here' from 'the filter is wrong
    again' -- if the on-ring count is zero for EVERY element, the filter is
    still broken; if it is nonzero but N is absent, that is a real result.
    """
    import json as _j
    r = retype_census_fn.remote({"smiles": smiles,
                                 "ring_ids": [int(x) for x in ring_ids.split(",")]})
    print(f"state: {r['smiles']}")
    print(f"atom_restate_semantic actions: {r['n_restate']}")
    print(f"  on new-ring atoms: {r['n_on_ring']}")
    print(f"  by target element (on-ring): {r['on_ring_by_element']}")
    print(f"  by target element (all):     {r['all_by_element']}")
    print(f"  ring atom ids requested: {r['ring_ids']}   present as v: {r['v_seen_on_ring']}")
    from pathlib import Path as _P
    _P("diagnostics/retype_census.json").write_text(_j.dumps(r, indent=1))


@app.function(image=image, cpu=(1.0, 1.0), memory=int(4 * 1024), timeout=30 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def retype_census_fn(job: dict) -> dict:
    from collections import Counter
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import (ATOM_VALENCE_CLASSES,
                                                 IDX_TO_ELEMENT,
                                                 smiles_to_molecular_graph)
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law)
    rt = _runtime(); model = rt["model"]
    st = pad_molecular_graph(smiles_to_molecular_graph(job["smiles"]), CANONICAL_SLOTS)
    L = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
    rid = set(int(x) for x in job["ring_ids"])
    allc, onc, vseen = Counter(), Counter(), set()
    n_rs = n_on = 0
    for m in L.marks:
        if m.executor_rule_name != "atom_restate_semantic":
            continue
        n_rs += 1
        a = m.action
        v = int(getattr(a, "v", -1))
        tci = int(getattr(a, "target_class_index", -1))
        el = IDX_TO_ELEMENT.get(int(ATOM_VALENCE_CLASSES[tci][0]), "?") \
            if 0 <= tci < len(ATOM_VALENCE_CLASSES) else "?"
        allc[el] += 1
        if v in rid:
            n_on += 1; onc[el] += 1; vseen.add(v)
    return dict(smiles=job["smiles"], ring_ids=sorted(rid), n_restate=n_rs,
                n_on_ring=n_on, on_ring_by_element=dict(onc),
                all_by_element=dict(allc), v_seen_on_ring=sorted(vseen))
