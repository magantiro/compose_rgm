"""Development docking read-out, in the order that separates the two hypotheses.

  1. Did the ACCESSIBLE CHEMICAL DISTRIBUTION change?
  2. Did that shift TRANSLATE INTO DOCKING (whole distribution, not the winner)?
  3. Best-so-far versus counted calls -- i.e. oracle efficiency.

IMPORTANT ON LABELS. Where a run predates provenance recording, the structural
class is derived from dHeavy/dRings and is a DESCRIPTIVE PROXY, not a causal
option label: a ring-gaining molecule may have come from `mixed`, and a
construct_ring trajectory may end with zero net ring change after later edits.
Such a run can support "high scores concentrated among strongly displaced /
ring-gaining candidates" but NOT "construct_ring caused the high score". Runs
carrying `docked_provenance` use the true option label and are marked as such.
"""
import json, sys, glob, os, collections
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, AllChem, RDConfig, rdMolDescriptors as rdMD
RDLogger.DisableLog('rdApp.*')
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

TAG = sys.argv[1] if len(sys.argv) > 1 else "OPTDOCK"
seeds = json.load(open('docs/GENMOL_T4_SEEDS.json'))
PRIOR = {("parp1", 0, 0.4): (-10.5, -10.6, -14.1),
         ("5ht1b", 7, 0.4): (-10.2, -12.3, -12.0)}
NOISE = 0.70

def q(v, p):
    return float(np.percentile(v, p)) if v else float('nan')

for f in sorted(glob.glob(f'/tmp/**/t4_particle/{TAG}_*.json', recursive=True)):
    d = json.load(open(f))
    dk = d.get('docked') or {}
    if not dk: continue
    i_ = int(d['idx']); tgt = d['target']; dl = float(d['delta'])
    x0 = seeds[i_]['smiles']; m0 = Chem.MolFromSmiles(x0)
    f0 = AllChem.GetMorganFingerprintAsBitVect(m0, 2, nBits=2048)
    prov = d.get('docked_provenance') or {}
    true_prov = bool(prov and any(v.get('mode') for v in prov.values()))
    rows = []
    for smi, ds in dk.items():
        m = Chem.MolFromSmiles(smi)
        if m is None or not ds: continue
        sim = DataStructs.TanimotoSimilarity(
            f0, AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
        dh = m.GetNumHeavyAtoms() - m0.GetNumHeavyAtoms()
        dr = rdMD.CalcNumRings(m) - rdMD.CalcNumRings(m0)
        try: qd = QED.qed(m); sa = sascorer.calculateScore(m)
        except Exception: continue
        rows.append({"ds": float(ds), "dh": dh, "dr": dr, "sim": sim,
                     "smi": smi, "mode": (prov.get(smi) or {}).get('mode'),
                     "feasible": qd >= 0.6 and sa <= 4 and sim >= dl})
    if not rows: continue
    feas = sorted([r for r in rows if r["feasible"]], key=lambda r: r["ds"])
    ours, gm, iv = PRIOR.get((tgt, i_, dl), (None, None, None))
    print(f"\n{'='*66}\n{tgt} s{i_} d{dl}   {len(rows)} docked, {len(feas)} feasible")
    print(f"prior {ours}  GenMol {gm}  InVirtuoGen {iv}   "
          f"[{'TRUE option provenance' if true_prov else 'structural PROXY only'}]")

    print("\n1. ACCESSIBLE DISTRIBUTION (feasible candidates)")
    for lab, sel in (("dHeavy >= 8", lambda r: abs(r['dh']) >= 8),
                     ("dHeavy >= 12", lambda r: abs(r['dh']) >= 12),
                     ("dRings > 0", lambda r: r['dr'] > 0)):
        s = [r for r in feas if sel(r)]
        sims = [r['sim'] for r in s]
        print(f"   {lab:14s} n={len(s):>4d}"
              + (f"   sim median {np.median(sims):.3f} "
                 f"[{min(sims):.3f}, {max(sims):.3f}]" if s else ""))

    print("\n2. DID IT TRANSLATE? score distribution by displacement")
    deep = [r['ds'] for r in feas if abs(r['dh']) >= 8]
    mild = [r['ds'] for r in feas if abs(r['dh']) < 8]
    print(f"   {'group':22s} {'n':>4s} {'best':>7s} {'top-10%':>8s} {'median':>7s}")
    for lab, v in (("deep constructive >=8", deep), ("local / mild <8", mild)):
        if v:
            print(f"   {lab:22s} {len(v):>4d} {min(v):>7.1f} "
                  f"{q(v,10):>8.1f} {q(v,50):>7.1f}")
    if deep and mild:
        print(f"   shift in median: {q(deep,50)-q(mild,50):+.2f} kcal/mol "
              f"(deep minus mild)")

    print("\n3. BEST-SO-FAR vs COUNTED CALLS")
    order = list(dk.items())
    best = None; marks = []
    for n_, (smi, ds) in enumerate(order, 1):
        if ds and (best is None or ds < best): best = ds
        if n_ in (20, 40, 80, 120, 160, 200) or n_ == len(order):
            marks.append((n_, best))
    print("   " + "  ".join(f"@{n_}:{b:.1f}" for n_, b in marks if b))

    if feas:
        b = feas[0]
        dlt = (b['ds'] - ours) if ours else None
        verdict = ("IMPROVES beyond noise" if dlt is not None and dlt <= -NOISE
                   else "within noise band" if dlt is not None and abs(dlt) < NOISE
                   else "WORSE" if dlt is not None else "")
        print(f"\nBEST FEASIBLE {b['ds']:.1f}  dH {b['dh']:+d} dR {b['dr']:+d} "
              f"sim {b['sim']:.3f}"
              + (f"  mode={b['mode']}" if b['mode'] else "")
              + (f"\n   vs prior best {dlt:+.2f} -> {verdict}" if dlt is not None else ""))
        print(f"   {b['smi']}")
