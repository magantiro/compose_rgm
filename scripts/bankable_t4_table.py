"""The bankable T4 table: gate applied, comparators from checked artifacts.

Written BEFORE the GATED30 results land, so the analysis cannot be shaped by
seeing them. Every molecule must pass, in this order:
    1  RDKit parse
    2  QED >= 0.6, SA <= 4, sim(x,x0) >= delta   (the T4 constraints)
    3  the calibrated med-chem gate               (what the last audit added)
A molecule failing any of these is not counted, however good its score.

Comparator numbers are read from docs/*_targets.json and never typed by hand.
"""
from __future__ import annotations
import json, os, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer
RDLogger.DisableLog("rdApp.*")
from compose_v4.gates.med_chem_gate import is_valid, validity_reasons

NOISE = 0.70          # Gate-0 median docking noise; smaller diffs are not results
GM = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def verify(smi, seed_smi, delta):
    """Return (ok, reason, profile). Constraints AND chemistry, in that order."""
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return False, "unparseable", {}
    q, sa = float(QED.qed(m)), float(sascorer.calculateScore(m))
    sim = float(DataStructs.TanimotoSimilarity(
        GM.GetFingerprint(Chem.MolFromSmiles(seed_smi)), GM.GetFingerprint(m)))
    prof = dict(qed=round(q, 3), sa=round(sa, 2), sim=round(sim, 3))
    if q < 0.6:            return False, f"QED {q:.3f}<0.6", prof
    if sa > 4:             return False, f"SA {sa:.2f}>4", prof
    if sim < delta - 1e-9: return False, f"sim {sim:.3f}<{delta}", prof
    r = validity_reasons(smi)
    if r:                  return False, "medchem:" + ",".join(r), prof
    return True, "", prof


def main(tag="GATED30", results_glob=None):
    seeds = {(s["target"], s["idx"]): s
             for s in json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())}
    gen = json.loads((ROOT / "docs/genmol_t4_targets.json").read_text())["rows"]
    ivg = json.loads((ROOT / "docs/invirtuogen_t4_targets.json").read_text())["rows"]
    for i, r in enumerate(ivg):          # alignment is asserted, never assumed
        assert r["target"] == seeds[(r["target"], i)]["target"]

    import glob
    pat = results_glob or str(ROOT / f"diagnostics/{tag}_*.json")
    best = {}
    for f in glob.glob(pat):
        try: d = json.load(open(f))
        except Exception: continue
        runs = d.get("runs") if isinstance(d, dict) else d
        for rec in (runs if isinstance(runs, list) else []):
            if not isinstance(rec, dict): continue
            t, i, dl = rec.get("target"), rec.get("idx"), rec.get("delta")
            smi = rec.get("best_smiles") or rec.get("best_docked_smi")
            ds = rec.get("best_ds") if rec.get("best_ds") is not None else rec.get("best_docked")
            if None in (t, i, dl) or not smi or ds is None: continue
            ok, why, prof = verify(smi, seeds[(t, i)]["smiles"], float(dl))
            if not ok: continue
            k = f"{t}_s{i}_d{dl}"
            if k not in best or ds < best[k]["ds"]:
                best[k] = dict(ds=float(ds), smi=Chem.MolToSmiles(Chem.MolFromSmiles(smi)), **prof)

    print(f"{'cell':22s} {'COMPOSE':>8s} {'GenMol':>8s} {'IVG':>8s} "
          f"{'vsGM':>7s} {'vsIVG':>7s}")
    rows = []
    for i, g in enumerate(gen):
        for dl, gk, ik in ((0.4, "genmol_d04", "invirtuo_d04"),
                           (0.6, "genmol_d06", "invirtuo_d06")):
            k = f"{g['target']}_s{i}_d{dl}"
            b = best.get(k); gv, iv = g[gk], ivg[i][ik]
            rows.append(dict(cell=k, ours=(b["ds"] if b else None), gen=gv, ivg=iv))
            o = f"{b['ds']:8.1f}" if b else f"{'—':>8s}"
            dg = f"{b['ds']-gv:+7.1f}" if (b and gv is not None) else f"{'—':>7s}"
            di = f"{b['ds']-iv:+7.1f}" if (b and iv is not None) else f"{'—':>7s}"
            print(f"{k:22s} {o} {gv if gv is not None else '—':>8} "
                  f"{iv if iv is not None else '—':>8} {dg} {di}")
    solved = sum(1 for r in rows if r["ours"] is not None)
    print(f"\ncells solved (constraints + med-chem gate): {solved}/30")
    for nm, key in (("GenMol", "gen"), ("InVirtuoGen", "ivg")):
        v = [r["ours"] - r[key] for r in rows if r["ours"] is not None and r[key] is not None]
        w = [r for r in rows if r["ours"] is not None and r[key] is not None
             and r["ours"] < r[key] - NOISE]
        l = [r for r in rows if r["ours"] is not None and r[key] is not None
             and r["ours"] > r[key] + NOISE]
        if v:
            print(f"  vs {nm:12s} mean {st.mean(v):+.3f} (n={len(v)})  "
                  f"beat {len(w)}  lose {len(l)}  [outside {NOISE} noise band]")
    json.dump(dict(best=best, rows=rows),
              open(ROOT / f"diagnostics/bankable_{tag}.json", "w"), indent=1)
    print(f"\nwrote diagnostics/bankable_{tag}.json")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["GATED30"]))
