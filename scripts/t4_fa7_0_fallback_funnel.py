"""The zero-support fallback's whole support on fa7_0, decomposed by gate.

`Fiber.check` returns only survivors, so the reason a support produces nothing is
invisible from its return value.  This enumerates the fallback's full deterministic
support -- every bridge-separated region crossed with every completion it draws --
and reports each threshold separately AND their intersection, because a per-gate
pass rate hides an empty intersection completely.

Metrics come from the gate's own functions (`QED.qed`, `sascorer.calculateScore`,
the fiber's own Morgan generator and reference fingerprint), not from a
reimplementation, and each endpoint is additionally run through the unmodified
production `Fiber.check` so a disagreement would show up as `joint != gate_pass`.

ZERO oracle calls.
"""
import json
import pathlib
import sys

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import QED, DataStructs

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.control import zero_support_fallback as zsf
from compose_v4.control.bridge_region_law import FreeFeasibilityGate, free_gate_margin_law
from compose_v4.control.dynamic_program_synthesis import (
    MAX_SEGMENT_LENGTH,
    _delete_pendant_fragment,
    _grow_actions,
)
from compose_v4.experiments.t4_fiber_campaign import QED_MIN, SA_MAX, Fiber, sascorer
from compose_v4.experiments.whole_ring_plan import execute_program

SEED = "CC(C)CCN(Cc2ccc1ccc(C(N)=N)cc1c2)C(=O)c3cccc4ccccc34"
AM = Chem.MolFromSmarts("[CX3](=[NX2])[NX3]")
fiber = Fiber(SEED, 0.6, support="compose_valid")
probe = fiber.check(SEED)
print("seed gate:", probe, flush=True)
src = zsf.proposal_state(SEED)
law = free_gate_margin_law(FreeFeasibilityGate(delta=0.6), SEED, maximum=None)
rng = np.random.default_rng(zsf.fallback_seed(2026091903, round_index=1, parent_index=0))
ordered = law.order(src, rng)[:60]
uniq = {}
for region in ordered:
    try:
        da, contracted, anchor, path = _delete_pendant_fragment(src, rng, law=zsf._FixedRegionLaw(region))
    except ValueError:
        continue
    cap = min(MAX_SEGMENT_LENGTH, 40 - contracted.n_real_atoms)
    for L in zsf._completion_lengths(rng, capacity=cap, count=8):
        if L == 0:
            prod = contracted
        else:
            try:
                ga, _, els = _grow_actions(contracted, rng, length=L, elements=("C","N","O"), anchor=anchor)
                prod, _ = execute_program(src, [*da, *ga])
            except (ValueError, KeyError, IndexError):
                continue
        ep = molecular_graph_to_smiles(prod)
        if ep is None: continue
        m = Chem.MolFromSmiles(ep)
        if m is None or "." in ep: continue
        uniq.setdefault(Chem.MolToSmiles(m), {"region_size": int(region.size), "inserted": L})
res = []
for smi, meta in uniq.items():
    m = Chem.MolFromSmiles(smi)
    heavy = m.GetNumHeavyAtoms()
    sim = DataStructs.TanimotoSimilarity(fiber.seed, fiber.generator.GetFingerprint(m))
    q = QED.qed(m); sa = sascorer.calculateScore(m)
    res.append({**meta, "smiles": smi, "sim": sim, "qed": q, "sa": sa, "heavy": heavy,
                "gate": fiber.check(smi) is not None,
                "amidine": bool(m.HasSubstructMatch(AM))})
n = len(res)
ps = [x for x in res if x["sim"] >= 0.6]
pq = [x for x in res if x["qed"] >= QED_MIN]
psa = [x for x in res if x["sa"] <= SA_MAX]
joint = [x for x in res if x["sim"] >= 0.6 and x["qed"] >= QED_MIN and x["sa"] <= SA_MAX]
print(f"QED_MIN={QED_MIN} SA_MAX={SA_MAX}", flush=True)
print(f"n_distinct={n} pass_sim={len(ps)} pass_qed={len(pq)} pass_sa={len(psa)} joint={len(joint)} gate={sum(1 for x in res if x['gate'])}", flush=True)
print(f"among similarity-passing (n={len(ps)}): max_qed={max(x['qed'] for x in ps):.4f} min_sa={min(x['sa'] for x in ps):.3f}", flush=True)
print(f"among qed-passing (n={len(pq)}): max_sim={max((x['sim'] for x in pq), default=0):.4f}", flush=True)
payload = {"n_distinct": n, "pass_sim": len(ps), "pass_qed": len(pq), "pass_sa": len(psa),
           "joint": len(joint), "gate_pass": sum(1 for x in res if x["gate"]),
           "max_qed_among_similarity_passing": max(x["qed"] for x in ps),
           "min_sa_among_similarity_passing": min(x["sa"] for x in ps),
           "max_sim_among_qed_passing": max((x["sim"] for x in pq), default=0.0),
           "qed_min": QED_MIN, "sa_max": SA_MAX,
           "seed_gate_metrics": {"qed": QED.qed(Chem.MolFromSmiles(SEED)),
                                 "sa": sascorer.calculateScore(Chem.MolFromSmiles(SEED))},
           "rows": sorted(res, key=lambda r: -r["sim"])[:40]}
with pathlib.Path(sys.argv[1]).open("w") as handle:
    json.dump(payload, handle, indent=2)
