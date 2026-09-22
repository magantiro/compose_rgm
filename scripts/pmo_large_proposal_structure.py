"""What do the LARGE generic proposals actually DO, structurally?

Long programs can be long two very different ways: many scattered small edits, or one
coherent scaffold-scale replacement.  Similarity cannot tell those apart; the number of
changed REGIONS and the size of the LARGEST one can.

Each proposal endpoint is compared to its own SOURCE by MCS.  Atoms outside the MCS are
changed; connected components of that set are the changed regions.  The same measure is
computed for the REQUIRED macro (source -> witness target) as the reference.
Zero oracle calls.
"""
from __future__ import annotations
import json, sys
import numpy as np
import networkx as nx
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdFMCS

def stats(a_smi, b_smi, timeout=5):
    """Structural delta of b relative to a."""
    A, B = Chem.MolFromSmiles(a_smi), Chem.MolFromSmiles(b_smi)
    if A is None or B is None: return None
    res = rdFMCS.FindMCS([A, B], timeout=timeout, completeRingsOnly=False,
                         ringMatchesRingOnly=True)
    if res.canceled or res.numAtoms == 0:
        core = 0; matched_b = set()
    else:
        q = Chem.MolFromSmarts(res.smartsString)
        mb = B.GetSubstructMatch(q)
        core = res.numAtoms; matched_b = set(mb) if mb else set()
    changed = [x.GetIdx() for x in B.GetAtoms() if x.GetIdx() not in matched_b]
    g = nx.Graph()
    g.add_nodes_from(changed)
    cs = set(changed)
    for bd in B.GetBonds():
        i, j = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        if i in cs and j in cs: g.add_edge(i, j)
    comps = [len(c) for c in nx.connected_components(g)] if changed else []
    return {
        "d_heavy": B.GetNumHeavyAtoms() - A.GetNumHeavyAtoms(),
        "d_rings": B.GetRingInfo().NumRings() - A.GetRingInfo().NumRings(),
        "retained_core": core,
        "retained_fraction": core / max(A.GetNumHeavyAtoms(), 1),
        "n_changed_regions": len(comps),
        "largest_changed_region": max(comps) if comps else 0,
        "total_changed": sum(comps),
    }

d = json.load(open("/Users/rmaganti/compose_pmo_macro_data/coherence_v1.json"))
rows = {3: [], 8: []}
ref = []
for s in d["sources"]:
    r = stats(s["source_smiles"], s["target_smiles"], timeout=10)
    if r: ref.append({**r, "route": s["route"]})
    for a in s["arms"]:
        for ep in (a.get("large_endpoints") or []):
            v = stats(s["source_smiles"], ep)
            if v: rows[a["max_modules"]].append(v)

def col(rs, k): return np.array([x[k] for x in rs], dtype=float)
def show(label, rs):
    if not rs: print(f"{label}: none"); return
    print(f"{label}  n={len(rs)}")
    for k in ("d_heavy","d_rings","n_changed_regions","largest_changed_region",
              "total_changed","retained_fraction"):
        c = col(rs, k)
        print(f"    {k:24s} median {np.median(c):7.2f}   mean {c.mean():7.2f}   "
              f"p90 {np.percentile(c,90):7.2f}   max {c.max():7.2f}")

print("=" * 84)
print("REQUIRED MACRO  (source -> witness target)  -- the thing to beat")
show("  required", ref)
print()
print("GENERIC LARGE PROPOSALS (>=17 primitives), by module budget")
show("  mods=3", rows[3])
print()
show("  mods=8", rows[8])
print()
print("=" * 84)
if rows[8] and ref:
    print("DISCRIMINATOR -- scattered local growth vs coherent scaffold replacement:")
    print(f"  required  : {np.median(col(ref,'n_changed_regions')):.1f} changed regions, "
          f"largest {np.median(col(ref,'largest_changed_region')):.1f}, "
          f"retained {np.median(col(ref,'retained_fraction')):.2f}")
    for m in (3, 8):
        print(f"  mods={m}    : {np.median(col(rows[m],'n_changed_regions')):.1f} changed regions, "
              f"largest {np.median(col(rows[m],'largest_changed_region')):.1f}, "
              f"retained {np.median(col(rows[m],'retained_fraction')):.2f}")
json.dump({"schema_version":"pmo_large_proposal_structure_v1",
           "evidence_role":"answer_known_offline_diagnostic","new_oracle_calls":0,
           "required_macro":ref,"generic_large_mods3":rows[3],"generic_large_mods8":rows[8]},
          open("/Users/rmaganti/compose_pmo_macro_data/structure_v1.json","w"), indent=1)
