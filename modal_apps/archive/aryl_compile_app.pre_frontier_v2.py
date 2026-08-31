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
