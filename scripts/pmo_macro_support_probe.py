"""Does the BLIND production generator propose the macro each witness segment needs?

Zero oracle calls.  The witness endpoint is answer-known and is used ONLY to score the
proposals after the fact; it never enters the generator, which sees the source state and
nothing else.
"""
from __future__ import annotations
import json, sys, time, collections

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program

SEGS = json.load(open("/Users/rmaganti/compose_pmo_macro_data/segments.json"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = sys.argv[2] if len(sys.argv) > 2 else "/Users/rmaganti/compose_pmo_macro_data/support_v1.json"

def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

def heavy(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else m.GetNumHeavyAtoms()

def probe(seg, n, seed):
    src = pad_molecular_graph(smiles_to_molecular_graph(seg["source_smiles"]), 48)
    assert len(src.atom_types) == 48, "PMO requires a 48-slot source"
    tgt_fp = fp(seg["target_smiles"])
    tgt_h = heavy(seg["target_smiles"])
    src_h = heavy(seg["source_smiles"])
    want = seg["required_primitives"]
    rng = np.random.default_rng(seed)
    sizes, sims, endpoints, fails = [], [], set(), collections.Counter()
    exact = 0
    for _ in range(n):
        try:
            _s, program, _a, trace, _m = synthesize_dynamic_program(
                src, rng, max_modules=3, max_primitives=32, max_blocks=8)
        except (ValueError, RuntimeError) as e:
            fails[type(e).__name__] += 1
            continue
        k = trace.get("primitive_edits", len(program.marks))
        ep = trace.get("endpoint")
        sizes.append(k)
        if ep:
            endpoints.add(ep)
            f = fp(ep)
            if f is not None and tgt_fp is not None:
                sims.append(DataStructs.TanimotoSimilarity(tgt_fp, f))
            if ep == seg["target_smiles"]:
                exact += 1
    sizes = np.array(sizes) if sizes else np.array([0])
    sims = np.array(sims) if sims else np.array([0.0])
    return {
        **{k: seg[k] for k in ("route", "seg", "required_primitives", "source_score", "target_score")},
        "source_smiles": seg["source_smiles"], "target_smiles": seg["target_smiles"],
        "source_heavy": src_h, "target_heavy": tgt_h, "heavy_delta_required": tgt_h - src_h,
        "draws": n, "executed": int(len(sizes)), "failures": dict(fails),
        "distinct_endpoints": len(endpoints),
        "size_median": float(np.median(sizes)), "size_p90": float(np.percentile(sizes, 90)),
        "size_max": int(sizes.max()),
        "p_size_ge_17": float((sizes >= 17).mean()),
        "p_size_ge_required": float((sizes >= want).mean()),
        "sim_median": float(np.median(sims)), "sim_max": float(sims.max()),
        "p_sim_ge_0.5": float((sims >= 0.5).mean()),
        "exact_hits": exact,
    }

def main():
    rows, t0 = [], time.time()
    for i, seg in enumerate(SEGS):
        t = time.time()
        r = probe(seg, N, 20260922 + 1009 * i)
        r["seconds"] = round(time.time() - t, 1)
        rows.append(r)
        print(f"{r['route'][:18]:18s} s{r['seg']} K={r['required_primitives']:2d} | "
              f"exec {r['executed']:4d}/{N} med {r['size_median']:4.1f} max {r['size_max']:2d} | "
              f"P(>=17)={r['p_size_ge_17']:.4f} P(>=K)={r['p_size_ge_required']:.4f} | "
              f"simmax {r['sim_max']:.3f} exact {r['exact_hits']} | {r['seconds']:.0f}s", flush=True)
        json.dump({"schema_version": "pmo_macro_support_v1",
                   "evidence_role": "answer_known_offline_diagnostic",
                   "new_charged_oracle_calls": 0, "draws_per_segment": N,
                   "generator": "synthesize_dynamic_program (blind; region_law=None)",
                   "segments": rows}, open(OUT, "w"), indent=1)
    print(f"TOTAL {time.time()-t0:.0f}s -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
