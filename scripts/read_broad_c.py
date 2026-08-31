"""Read broad-C development checkpoints. READ-ONLY: never restarts or mutates."""
import json, os, subprocess, sys
from collections import Counter
from pathlib import Path
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import QED, RDConfig, rdFingerprintGenerator as rfg
RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score")); import sascorer

CELL = sys.argv[1] if len(sys.argv) > 1 else "5ht1b_s7_d0.6"
SEEDS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["20","21","22"])]
ROOT = Path(__file__).resolve().parents[1]
D = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp/c"; D.mkdir(parents=True, exist_ok=True)
gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
seeds = {f"{s['target']}_s{s['idx']}": s["smiles"]
         for s in json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())}
tgt, si, _d = CELL.rsplit("_", 2)
SEED = seeds[f"{tgt}_{si}"]
fs = gen.GetFingerprint(Chem.MolFromSmiles(SEED))

def ring_profile(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return {}
    ri = m.GetRingInfo(); rings = ri.AtomRings()
    fused = sum(1 for i in range(len(rings)) for j in range(i+1, len(rings))
                if len(set(rings[i]) & set(rings[j])) >= 2)
    arom = sum(1 for r in rings if all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    satn = any(all(not m.GetAtomWithIdx(i).GetIsAromatic() for i in r)
               and any(m.GetAtomWithIdx(i).GetSymbol() == "N" for i in r) for r in rings)
    return dict(heavy=m.GetNumHeavyAtoms(), n_rings=len(rings), aromatic=arom,
                fused=fused, linked=len(rings)-fused-1 if len(rings) else 0,
                sizes=sorted(len(r) for r in rings), sat_N_ring=bool(satn))

for r in SEEDS:
    f = f"episodes_{CELL}_pooled_broad_r{r}.json"
    subprocess.run(["modal","volume","get","compose-v4-artifacts",
                    f"macro_basin/{f}", str(D/f), "--force"],
                   capture_output=True)
    if not (D/f).exists():
        print(f"broad r{r}: no checkpoint yet"); continue
    d = json.loads((D/f).read_text())
    dk = d.get("docked") or {}
    slog = d.get("semantic_log") or []
    print(f"\n{'='*72}\nbroad r{r}   round {d['round']}/9   unique docked {len(dk)}   best {d.get('best_ds')}")
    # SAMPLED -> EXECUTED -> FEASIBLE -> DOCKED funnel, by topology.
    # Execution counts here are read from traces for REPORTING only; the credit
    # path itself uses the executor's ok flag, never trace strings.
    import re as _re
    arch = d.get("archive") or []
    samp = {"linked": 0, "fused": 0}; ok = {"linked": 0, "fused": 0}
    feas = {"linked": 0, "fused": 0}; dock = {"linked": 0, "fused": 0}
    for a in arch:
        tr = [t for t in (a.get("trace") or []) if str(t).startswith("ring:")]
        hit = None
        for t in tr:
            m = _re.match(r"ring:(\w+)/", str(t))
            if not m or m.group(1) not in samp:
                continue
            samp[m.group(1)] += 1
            if ":ok(" in t:
                ok[m.group(1)] += 1; hit = m.group(1)
        if hit:
            feas[hit] += 1
            if a.get("smiles") in (d.get("docked") or {}):
                dock[hit] += 1
    print(f"  FUNNEL   {'topology':<8}{'sampled':>9}{'executed':>10}{'feasible':>10}{'docked':>8}")
    for t in ("linked", "fused"):
        print(f"           {t:<8}{samp[t]:>9}{ok[t]:>10}{feas[t]:>10}{dock[t]:>8}")
    el = (slog[-1].get("eligibility") if slog else None) or []
    if el:
        ts = sum(e["sampled"] for e in el); tc = sum(e["credited"] for e in el)
        print(f"  CREDIT   requests sampled {ts}, credited {tc} "
              f"({tc/ts*100:.0f}% executed)  <- UNSAT draws now get none")
    if slog:
        curve = [(e["oracle_calls"], e.get("best")) for e in d.get("budget_curve", [])]
        print(f"  best-so-far curve: {curve}")
        last = slog[-1]
        lg = last.get("logit", {})
        for axis in ("topology","size","state","n_hetero","stoich"):
            if axis in lg:
                top = sorted(lg[axis].items(), key=lambda kv:-kv[1])[:5]
                print(f"  mass {axis:<9}: " + "  ".join(f"{k}={v:+.2f}" for k,v in top))
        used = Counter(s for e in slog for s in e.get("sampled", []))
        print(f"  most-used tuples: {used.most_common(6)}")
    if dk:
        top = sorted(dk.items(), key=lambda kv: kv[1])[:5]
        print("  TOP DOCKED:")
        for smi, ds in top:
            m = Chem.MolFromSmiles(smi)
            if m is None: continue
            p = ring_profile(smi)
            sim = DataStructs.TanimotoSimilarity(fs, gen.GetFingerprint(m))
            print(f"   {ds:>7.2f}  qed {QED.qed(m):.3f} sa {sascorer.calculateScore(m):.2f} "
                  f"sim {sim:.2f} heavy {p.get('heavy')} rings {p.get('n_rings')} "
                  f"arom {p.get('aromatic')} fused {p.get('fused')} sizes {p.get('sizes')} "
                  f"satN {p.get('sat_N_ring')}")
            print(f"           {smi}")
print(f"\nnarrow banked 5HT1B d0.6: -11.10 / -11.30 / -10.90 (mean -11.10)")
print(f"GenMol published 5HT1B d0.6: -12.00")
