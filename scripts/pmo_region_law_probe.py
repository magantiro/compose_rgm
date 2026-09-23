"""Does an uncapped region draw move PMO's changed-region SIZE?

The autopsy located the binding constraint: MAX_SEGMENT_LENGTH = 8 bounds the
pendant-excision draw on the PMO shallow lane, and the structural census measured that
lane's realized largest_changed_region at exactly 7.  BridgeRegionLaw imposes no size
bound and is accepted by synthesize_dynamic_program(region_law=...), but PMO passes None.

ARM A  region_law=None                      -> production, v1's cap of 8
ARM B  BridgeRegionLaw(maximum=None)        -> uncapped UNIFORM; margin=None, so NO
                                               similarity reference, NO delta, NO task
                                               information of any kind enters the draw.

Matched RNG seeds; only the law differs.  Zero oracle calls.
"""
from __future__ import annotations
import json, sys, time
import numpy as np
import networkx as nx
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdFMCS

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.bridge_region_law import BridgeRegionLaw
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = "/Users/rmaganti/compose_pmo_macro_data/region_law_v2_patched.json"

def delta(a_smi, b_smi, timeout=5):
    A, B = Chem.MolFromSmiles(a_smi), Chem.MolFromSmiles(b_smi)
    if A is None or B is None: return None
    res = rdFMCS.FindMCS([A, B], timeout=timeout, ringMatchesRingOnly=True)
    if res.canceled or res.numAtoms == 0:
        core, matched = 0, set()
    else:
        q = Chem.MolFromSmarts(res.smartsString)
        mb = B.GetSubstructMatch(q)
        core, matched = res.numAtoms, set(mb) if mb else set()
    changed = [x.GetIdx() for x in B.GetAtoms() if x.GetIdx() not in matched]
    g = nx.Graph(); g.add_nodes_from(changed); cs = set(changed)
    for bd in B.GetBonds():
        i, j = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        if i in cs and j in cs: g.add_edge(i, j)
    comps = [len(c) for c in nx.connected_components(g)] if changed else []
    return {"largest_changed_region": max(comps) if comps else 0,
            "n_changed_regions": len(comps),
            "retained_fraction_mcs": core / max(A.GetNumHeavyAtoms(), 1),
            "d_heavy": B.GetNumHeavyAtoms() - A.GetNumHeavyAtoms(),
            "d_rings": B.GetRingInfo().NumRings() - A.GetRingInfo().NumRings()}

def arm(src_smi, src, n, law, seed):
    rng = np.random.default_rng(seed)
    rows, eps, fails = [], set(), 0
    for _ in range(n):
        try:
            _s, program, _a, trace, _m = synthesize_dynamic_program(
                src, rng, max_modules=3, max_primitives=32, max_blocks=8, region_law=law)
        except (ValueError, RuntimeError):
            fails += 1; continue
        ep = trace.get("endpoint")
        if not ep: continue
        eps.add(ep)
        d = delta(src_smi, ep)
        if d: rows.append({**d, "k": trace.get("primitive_edits", len(program.marks)), "endpoint": ep})
    return rows, len(eps), fails

def summarise(rows, uniq, fails, n, label):
    if not rows: return {"arm": label, "executed": 0}
    def col(k): return np.array([r[k] for r in rows], dtype=float)
    return {"arm": label, "draws": n, "executed": len(rows), "failures": fails,
            "executable_yield": len(rows)/n, "distinct_endpoints": uniq,
            "largest_changed_region_median": float(np.median(col("largest_changed_region"))),
            "largest_changed_region_p90": float(np.percentile(col("largest_changed_region"),90)),
            "largest_changed_region_max": float(col("largest_changed_region").max()),
            "retained_fraction_mcs_median": float(np.median(col("retained_fraction_mcs"))),
            "n_changed_regions_median": float(np.median(col("n_changed_regions"))),
            "d_heavy_median": float(np.median(col("d_heavy"))),
            "d_rings_median": float(np.median(col("d_rings"))),
            "k_median": float(np.median(col("k")))}

def main():
    segs = json.load(open("/Users/rmaganti/compose_pmo_macro_data/segments.json"))
    seen, sources = set(), []
    for g in segs:
        if g["source_smiles"] not in seen:
            seen.add(g["source_smiles"]); sources.append(g)
    out = {"schema_version": "pmo_region_law_probe_v1",
           "evidence_role": "zero_oracle_support_measurement", "new_charged_oracle_calls": 0,
           "arms": {"A": "region_law=None (production, v1 cap 8)",
                    "B": "BridgeRegionLaw(maximum=None) -- uncapped uniform, margin=None, no task information"},
           "baseline_from_census": {"largest_changed_region": 7, "retained_fraction": 0.64},
           "required_from_witnesses": {"largest_changed_region": 19, "retained_fraction": 0.30},
           "sources": []}
    LAW = BridgeRegionLaw(maximum=None)
    for i, g in enumerate(sources):
        src = pad_molecular_graph(smiles_to_molecular_graph(g["source_smiles"]), 48)
        assert len(src.atom_types) == 48
        rec = {"route": g["route"], "seg": g["seg"], "source_smiles": g["source_smiles"], "arms": []}
        for label, law in (("A_none", None), ("B_uncapped", LAW)):
            t = time.time()
            rows, uniq, fails = arm(g["source_smiles"], src, N, law, 20260922 + 1009*i)
            s = summarise(rows, uniq, fails, N, label); s["seconds"] = round(time.time()-t, 1)
            s["endpoints"] = [r["endpoint"] for r in rows[:40]]
            rec["arms"].append(s)
            print(f"{g['route'][:16]:16s} s{g['seg']} {label:11s} | yield {s.get('executable_yield',0):.3f} "
                  f"| largest {s.get('largest_changed_region_median',0):5.1f} "
                  f"max {s.get('largest_changed_region_max',0):5.1f} "
                  f"| retained {s.get('retained_fraction_mcs_median',0):.3f} "
                  f"| uniq {s.get('distinct_endpoints',0):4d} | {s['seconds']:.0f}s", flush=True)
        out["sources"].append(rec)
        json.dump(out, open(OUT, "w"), indent=1)
    print("WROTE", OUT, flush=True)

if __name__ == "__main__":
    main()
