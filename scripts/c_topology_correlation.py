"""READ-ONLY: does learned topology mass track SA headroom, growth and docking?

No prior is added anywhere. This only reads what the running controller chose.
"""
import json, os, subprocess, sys
from pathlib import Path
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import QED, RDConfig, rdFingerprintGenerator as rfg
RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score")); import sascorer

CELL = sys.argv[1] if len(sys.argv) > 1 else "5ht1b_s7_d0.6"
SEEDS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["20","21","22"])]
ROOT = Path(__file__).resolve().parents[1]
D = Path(os.environ.get("CLAUDE_JOB_DIR","/tmp"))/"tmp/c"; D.mkdir(parents=True, exist_ok=True)
gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
seeds = {f"{s['target']}_s{s['idx']}": s["smiles"]
         for s in json.loads((ROOT/"docs/GENMOL_T4_SEEDS.json").read_text())}
tgt, si, _ = CELL.rsplit("_", 2); SEED = seeds[f"{tgt}_{si}"]
ms = Chem.MolFromSmiles(SEED); seed_hv = ms.GetNumHeavyAtoms()

def topo_counts(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    rings = m.GetRingInfo().AtomRings()
    f = sum(1 for i in range(len(rings)) for j in range(i+1,len(rings))
            if len(set(rings[i]) & set(rings[j])) >= 2)
    l = sum(1 for i in range(len(rings)) for j in range(i+1,len(rings))
            if len(set(rings[i]) & set(rings[j])) == 0 and
               any(m.GetBondBetweenAtoms(a,b) for a in rings[i] for b in rings[j]))
    return m.GetNumHeavyAtoms(), len(rings), f, l, sascorer.calculateScore(m)

print(f"{CELL}  seed {seed_hv} heavy   SA ceiling 4.0   sim floor {_d if (_d:=float(CELL[-3:])) else 0}")
for r in SEEDS:
    f = f"episodes_{CELL}_pooled_broad_r{r}.json"
    subprocess.run(["modal","volume","get","compose-v4-artifacts",
                    f"macro_basin/{f}", str(D/f), "--force"], capture_output=True)
    if not (D/f).exists():
        print(f"\nr{r}: no checkpoint"); continue
    d = json.loads((D/f).read_text())
    slog = d.get("semantic_log") or []
    bc = {e["round"]: e for e in d.get("budget_curve", [])}
    print(f"\n{'='*76}\nr{r}   round {d['round']}/9   best {d.get('best_ds')}")
    print(f"  {'rnd':>3} {'calls':>6} {'P(link)':>8} {'P(fused)':>9} {'best':>7}")
    for e in slog:
        lg = e.get("logit", {}).get("topology", {})
        import math
        li, fu = lg.get("linked", 0.0), lg.get("fused", 0.0)
        mx = max(li, fu); el, ef = math.exp(li-mx), math.exp(fu-mx); t = el+ef
        b = bc.get(e["round"], {}).get("best")
        print(f"  {e['round']:>3} {e['oracle_calls']:>6} {el/t:>8.3f} {ef/t:>9.3f} "
              f"{(f'{b:.2f}' if isinstance(b,(int,float)) else '--'):>7}")
    dk = d.get("docked") or {}
    if dk:
        prof = [(v,)+tuple(topo_counts(k)) for k,v in dk.items() if topo_counts(k)]
        prof.sort(key=lambda x: x[0])
        import statistics as st
        print(f"  docked set: n={len(prof)}  mean SA {st.mean(p[5] for p in prof):.2f}  "
              f"max SA {max(p[5] for p in prof):.2f}  "
              f"mean heavy {st.mean(p[1] for p in prof):.1f} (seed {seed_hv})")
        tf = sum(p[3] for p in prof); tl = sum(p[4] for p in prof)
        print(f"  realized topology in docked molecules: FUSED pairs {tf}  LINKED pairs {tl}")
        print(f"  top 3 by docking:")
        for v,h,nr,fp,lp,sa in prof[:3]:
            print(f"     {v:>6.2f}  heavy {h:>2}  rings {nr}  fused {fp}  linked {lp}  SA {sa:.2f}")
