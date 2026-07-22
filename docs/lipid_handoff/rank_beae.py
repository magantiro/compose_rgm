"""BEAE candidate ranking: oracle prediction (descriptor-supported extrapolation)
+ structural-OOD flag + similarity to qualified leads. Honest active-learning ranking."""
import sys
from pathlib import Path
sys.path.insert(0, "src")
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors
RDLogger.DisableLog("rdApp.*")
from compose_v4.oracles.pan_lung_filtering import (
    _load_bundle, _load_verified_joblib, molecular_features, _score_members, molecular_admission)

MANIFEST = Path("artifacts/oracles/pan_lung_filtering_v1/manifest.json")
LEADS = {
 "RM-60": "CCCCCCCCCCC(CCCCCCCC)OC(=O)/C=C/N(CCCN(C)C)CCC(=O)OCC(CCCCCC)CCCCCCCC",
 "Example-2": "CCCCC/C=C\\C/C=C\\CCCCCCCCOC(=O)/C=C/N(CCCN(C)C)CCC(=O)OCCCCCCCCC(C)C",
}
_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_lead_fps = {k: _FP.GetFingerprint(Chem.MolFromSmiles(v)) for k, v in LEADS.items()}
_manifest, _root, _contract = _load_bundle(MANIFEST)
_ad = _load_verified_joblib(_root, _manifest["domains"]["a549"]["applicability_domain"])

def lead_similarity(smi):
    fp = _FP.GetFingerprint(Chem.MolFromSmiles(smi))
    sims = {k: DataStructs.TanimotoSimilarity(fp, f) for k, f in _lead_fps.items()}
    best = max(sims, key=sims.get)
    return best, round(sims[best], 3)

def rank_beae(smiles, domain_id="a549"):
    rows = []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        can = Chem.MolToSmiles(m)
        try:
            feats = molecular_features([can])["combined"]
            scores = _score_members(manifest=_manifest, repo_root=_root, contract=_contract,
                                    domain_id=domain_id, features=feats)
            sel = min(scores, key=lambda x: x["pessimistic_score"])  # the conservative-selected ensemble
            pred_potency = float(sel["ensemble_mean"])       # <-- meaningful: calibrated predicted potency
            uncertainty = float(sel["total_uncertainty"])    # conformal radius + ensemble disagreement
            pessimistic = float(sel["pessimistic_score"])
        except Exception:
            pred_potency = uncertainty = pessimistic = None
        adm = molecular_admission(_ad, can)
        nearest, sim = lead_similarity(can)
        rows.append({"smiles": can,
                     "pred_potency": pred_potency,        # PRIMARY ranking signal (higher = more potent-predicted)
                     "uncertainty": None if uncertainty is None else round(uncertainty, 2),
                     "pessimistic": None if pessimistic is None else round(pessimistic, 3),
                     "pred_score": pred_potency,          # back-compat alias
                     "train_tanimoto": round(adm.get("maximum_training_tanimoto", 0), 3),
                     "descriptor_in_domain": bool(adm.get("admitted")),
                     "nearest_lead": nearest, "lead_similarity": sim,
                     "mw": round(Descriptors.MolWt(m), 1)})
    rows.sort(key=lambda r: (r["pred_potency"] is not None, r["pred_potency"] or -1e9), reverse=True)
    return rows

if __name__ == "__main__":
    import json
    cands = dict(LEADS)
    enum = json.load(open("artifacts/datasets/compose_lipid_pretraining_v1/beae_substrate_enumeration_v1.json"))
    for e in enum["examples"]:
        cands[f"enum:{e['head']}/{e['propiolate_tail']}"] = e["smiles"]
    inv = {Chem.MolToSmiles(Chem.MolFromSmiles(v)): k for k, v in cands.items()}
    rows = rank_beae(list(cands.values()))
    print(f"{'candidate':30s} {'pred':>7s} {'trainTani':>9s} {'nearLead':>10s} {'leadSim':>7s} {'MW':>7s}")
    for r in rows:
        name = inv.get(r["smiles"], "?")[:29]
        p = "None" if r["pred_score"] is None else f"{r['pred_score']:.3f}"
        print(f"{name:30s} {p:>7s} {r['train_tanimoto']:>9} {str(r['nearest_lead']):>10s} {str(r['lead_similarity']):>7} {r['mw']:>7}")
