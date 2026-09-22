"""Does a bigger module budget buy the right TRANSFORMATION, or only the right SIZE?

    P(useful macro) = P(right scale) x P(right transformation | right scale)

The support probe measured the first factor at production settings.  This measures BOTH
factors, at the production module budget (3) and at the cap (8), with matched RNG seeds
so only max_modules differs.

Every proposal's endpoint is PERSISTED, so a question posed after the fact is answerable.
Zero oracle calls; the witness endpoint scores proposals afterwards and never enters the
generator.
"""
from __future__ import annotations
import json, sys, time
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, QED

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program

SEGS = json.load(open("/Users/rmaganti/compose_pmo_macro_data/segments.json"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 800
OUT = "/Users/rmaganti/compose_pmo_macro_data/coherence_v1.json"
ARMS = [3, 8]

def fp(s):
    m = Chem.MolFromSmiles(s)
    return None if m is None else AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

def arm(src, tgt_fp, n, max_modules, seed):
    rng = np.random.default_rng(seed)
    rows, eps = [], set()
    for _ in range(n):
        try:
            _s, program, _a, trace, _m = synthesize_dynamic_program(
                src, rng, max_modules=max_modules, max_primitives=32, max_blocks=8)
        except (ValueError, RuntimeError):
            continue
        k = trace.get("primitive_edits", len(program.marks))
        ep = trace.get("endpoint")
        sim = q = None
        if ep:
            eps.add(ep)
            f = fp(ep)
            if f is not None and tgt_fp is not None:
                sim = DataStructs.TanimotoSimilarity(tgt_fp, f)
            m = Chem.MolFromSmiles(ep)
            if m is not None:
                try: q = QED.qed(m)
                except Exception: pass
        rows.append({"k": int(k), "sim": sim, "qed": q, "endpoint": ep})
    return rows, len(eps)

def summarise(rows, uniq, n, max_modules, want):
    ks = np.array([r["k"] for r in rows]) if rows else np.array([0])
    big = [r for r in rows if r["k"] >= 17]
    sims = np.array([r["sim"] for r in rows if r["sim"] is not None] or [0.0])
    bsims = np.array([r["sim"] for r in big if r["sim"] is not None] or [0.0])
    qs = [r["qed"] for r in rows if r["qed"] is not None]
    return {
        "max_modules": max_modules, "draws": n, "executed": len(rows),
        "executable_yield": len(rows) / n, "distinct_endpoints": uniq,
        "size_median": float(np.median(ks)), "size_max": int(ks.max()),
        "p_size_ge_17": float((ks >= 17).mean()),
        "p_size_ge_required": float((ks >= want).mean()),
        "n_large": len(big),
        "sim_median_all": float(np.median(sims)), "sim_max_all": float(sims.max()),
        "sim_median_large": float(np.median(bsims)) if big else None,
        "sim_max_large": float(bsims.max()) if big else None,
        "p_sim_ge_0.4_given_large": float((bsims >= 0.4).mean()) if big else None,
        "mean_qed": float(np.mean(qs)) if qs else None,
        "large_endpoints": [r["endpoint"] for r in big][:40],
    }

def main():
    seen, sources = set(), []
    for g in SEGS:
        if g["source_smiles"] not in seen:
            seen.add(g["source_smiles"]); sources.append(g)
    sources.sort(key=lambda g: -g["required_primitives"])   # large segments first
    out = {"schema_version": "pmo_macro_coherence_v1",
           "evidence_role": "answer_known_offline_diagnostic",
           "new_charged_oracle_calls": 0, "draws_per_arm": N, "arms": ARMS,
           "note": "matched RNG seed per source across arms; only max_modules differs",
           "sources": []}
    for i, g in enumerate(sources):
        src = pad_molecular_graph(smiles_to_molecular_graph(g["source_smiles"]), 48)
        assert len(src.atom_types) == 48
        tgt_fp = fp(g["target_smiles"])
        rec = {k: g[k] for k in ("route", "seg", "required_primitives",
                                 "source_smiles", "target_smiles")}
        rec["arms"] = []
        for m in ARMS:
            t = time.time()
            rows, uniq = arm(src, tgt_fp, N, m, 20260922 + 1009 * i)
            a = summarise(rows, uniq, N, m, g["required_primitives"])
            a["seconds"] = round(time.time() - t, 1)
            rec["arms"].append(a)
            print(f"{g['route'][:16]:16s} s{g['seg']} K={g['required_primitives']:2d} mods={m} | "
                  f"P(>=17)={a['p_size_ge_17']:.4f} nlarge={a['n_large']:3d} | "
                  f"simmax_all={a['sim_max_all']:.3f} simmax_LARGE="
                  f"{('%.3f'%a['sim_max_large']) if a['sim_max_large'] is not None else '  -  '} | "
                  f"yield {a['executable_yield']:.3f} uniq {a['distinct_endpoints']:4d} "
                  f"qed {('%.3f'%a['mean_qed']) if a['mean_qed'] else '-'} | {a['seconds']:.0f}s",
                  flush=True)
        out["sources"].append(rec)
        json.dump(out, open(OUT, "w"), indent=1)
    print("WROTE", OUT, flush=True)

if __name__ == "__main__":
    main()
