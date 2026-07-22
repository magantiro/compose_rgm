"""Meeting-ready BEAE ranking. The potency oracle is OOD on BEAE (abstains), so we
rank by a TRANSPARENT COMPOSITE:
  composite = design_score + 3*pred_potency + 1.5*lead_similarity
design_score encodes ionizable-lipid principles (Whitehead/Arral): an ionizable amine
head, two hydrophobic tails of the right length, MW/logP in range, no aromatic/polar
tails. This term is reliable and sinks the bad decoys; potency is the provisional
(OOD-flagged) active-learning signal. Good candidates rank high; decoys rank low."""
import json, sys, os
from pathlib import Path
from collections import Counter, defaultdict
SCRATCH = Path("/private/tmp/claude-502/-Users-rmaganti-Documents-Codex-2026-07-14-ok-so-compose-rgm-claude-lipid/11425d68-331b-425e-9161-eb394e3d1e57/scratchpad")
sys.path.insert(0, "src"); sys.path.insert(0, str(SCRATCH))
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
import rank_beae
from compose_v4.lipids.beae_linker import beae_decompose
from compose_v4.lipids.head_region import basic_nitrogens

def design_score(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return -9.0, ["unparseable"], {}
    score, notes = 0.0, []
    n_basic = len(basic_nitrogens(m))
    if n_basic >= 1: score += 2.0
    else: score -= 3.0; notes.append("no-ionizable-head")
    dec = beae_decompose(m)
    tail_desc = {}
    if dec is None:
        score -= 3.0; notes.append("not-BEAE")
    else:
        for tk in ("tail1", "tail2"):
            atoms = list(dec[tk])
            nc = sum(m.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in atoms)
            nhet = sum(m.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6) for i in atoms)
            narom = sum(m.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)
            ndb = sum(1 for b in m.GetBonds()
                      if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms and b.GetBondTypeAsDouble() == 2.0)
            tail_desc[tk] = f"{nc}C/{ndb}db"
            if 8 <= nc <= 18: score += 1.0
            elif nc < 6: score -= 1.5; notes.append(f"{tk}-short{nc}")
            elif nc > 20: score -= 0.5
            if narom > 0: score -= 1.5; notes.append(f"{tk}-aromatic")
            if nhet > 3: score -= 1.0; notes.append(f"{tk}-polar")
    mw, logp = Descriptors.MolWt(m), Crippen.MolLogP(m)
    if 550 <= mw <= 1000: score += 1.0
    elif mw < 480: score -= 1.0
    if 6 <= logp <= 16: score += 1.0
    elif logp < 4: score -= 1.5; notes.append("too-polar")
    return round(score, 2), notes, tail_desc

# ---- load candidates ----
POOLS = os.environ.get("BEAE_POOLS", "beae_generated.json").split(",")
pool = []
for f in POOLS:
    p = SCRATCH / f.strip()
    if p.exists(): pool += json.load(open(p))
print(f"candidates loaded: {len(pool)} from {POOLS}")

# FP-dedup
_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
seen, uniq = set(), []
for c in pool:
    m = Chem.MolFromSmiles(c["smiles"])
    if m is None: continue
    k = tuple(_FP.GetFingerprint(m).GetOnBits())
    if k in seen: continue
    seen.add(k); uniq.append(c)
print(f"FP-unique: {len(uniq)}")

# rank_beae (potency oracle) + design score -> composite
rows = rank_beae.rank_beae([c["smiles"] for c in uniq])
meta = {Chem.MolToSmiles(Chem.MolFromSmiles(c["smiles"])): c for c in uniq}
out = []
for r in rows:
    c = meta.get(r["smiles"], {})
    ds, notes, td = design_score(r["smiles"])
    pot = r["pred_potency"] or 0.0
    r["design_score"] = ds; r["design_notes"] = notes; r["tails"] = td
    r["kind"] = c.get("kind", "generated"); r["head"] = c.get("head", "?"); r["seed"] = c.get("seed_lead", "?")
    r["composite"] = round(ds + 3.0 * pot + 1.5 * (r["lead_similarity"] or 0.0), 3)
    out.append(r)
out.sort(key=lambda r: r["composite"], reverse=True)

gens = [r for r in out if r["kind"] != "decoy"]
decoys = [r for r in out if r["kind"] == "decoy"]
print(f"\n=== RANKED: {len(gens)} generated + {len(decoys)} decoys, by composite ===")
print(f"{'rank':>4} {'kind':>9} {'composite':>9} {'design':>6} {'pred':>6} {'leadSim':>7} {'tails':>14} head")
for i, r in enumerate(out, 1):
    p = "None" if r["pred_potency"] is None else f"{r['pred_potency']:+.3f}"
    t = f"{r['tails'].get('tail1','?')}+{r['tails'].get('tail2','?')}"
    tag = "DECOY" if r["kind"] == "decoy" else ""
    print(f"{i:>4} {r['kind'][:9]:>9} {r['composite']:>9.2f} {r['design_score']:>6.1f} {p:>6} {r['lead_similarity']:>7} {t:>14} {r['head']} {tag}")

# where do decoys land?
if decoys and gens:
    worst_gen = min(g["composite"] for g in gens)
    best_decoy = max(d["composite"] for d in decoys)
    print(f"\nseparation: worst generated composite {worst_gen:.2f}  vs  best decoy composite {best_decoy:.2f}"
          f"  ->  {'CLEAN (all generated > all decoys)' if worst_gen > best_decoy else 'OVERLAP'}")

# diverse shortlist: best per head
best = {}
for r in gens:
    h = r["head"]
    if h not in best or r["composite"] > best[h]["composite"]: best[h] = r
print(f"\n=== diverse shortlist (best generated per head) ===")
for h, r in sorted(best.items(), key=lambda kv: -kv[1]["composite"]):
    print(f"  [{h}] composite={r['composite']:.2f} pred={r['pred_potency']}  {r['smiles']}")

json.dump(out, open(SCRATCH / "beae_ranked.json", "w"), indent=1)
print(f"\nsaved {len(out)} ranked -> beae_ranked.json  (all generated = active-learning batch; potency OOD/provisional)")
