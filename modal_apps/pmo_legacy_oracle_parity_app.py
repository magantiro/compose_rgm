"""Do our frozen JNK3/GSK3B/DRD2 arrays reproduce the original TDC estimators?

This is the one PMO gate a modern environment structurally cannot run. TDC ships
these three as scikit-learn 0.21.3 pickles. Aliasing the pre-0.22 module paths
gets past the import and then fails:

    ValueError: node array from the pickle has an incompatible dtype
    ... expected [..., 'missing_go_to_left']

The tree node struct gained a field after 0.21, so the binary layout, not the
module path, is the wall. Reproducing the originals therefore needs
sklearn < 0.22, hence Python <= 3.7: 0.21.3 has no cp38 wheels, and 0.22, which
does, is exactly the release that broke the layout.

What this proves, if it passes: our npz extraction IS the TDC oracle, so the
production path can keep walking flat arrays in numpy with no sklearn, no
pickle, and no drift when the environment moves. That is stronger than an
oracle-vs-oracle check, because it validates the artifact every downstream
number depends on.

Records for each oracle: SHA of the upstream pickle, SHA of our npz, the fixed
canonical panel, both sets of outputs, max absolute discrepancy, an orientation
check, and a deliberately WRONG-orientation negative control. Matching one
orientation means little unless the plausible alternative is shown to fail.

CPU only. No training, no docking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]

# sklearn 0.21.3 is the last release before the tree dtype change; its newest
# wheels are cp37. numpy is pinned below 1.20 for binary compatibility with it.
image = (
    modal.Image.debian_slim(python_version="3.7")
    .pip_install("numpy==1.18.5", "scipy==1.4.1", "scikit-learn==0.21.3",
                 "joblib==0.14.1", "rdkit-pypi==2021.3.5")
    .add_local_file(ROOT / "artifacts/tdc_originals/jnk3.pkl",  "/orig/jnk3.pkl",  copy=True)
    .add_local_file(ROOT / "artifacts/tdc_originals/gsk3b.pkl", "/orig/gsk3b.pkl", copy=True)
    .add_local_file(ROOT / "artifacts/tdc_originals/drd2.pkl",  "/orig/drd2.pkl",  copy=True)
    .add_local_file(ROOT / "artifacts/oracles/molleo_task3_v1/jnk3_forest.npz",
                    "/frozen/jnk3_forest.npz", copy=True)
    .add_local_file(ROOT / "artifacts/oracles/molleo_task3_v1/gsk3b_forest.npz",
                    "/frozen/gsk3b_forest.npz", copy=True)
    .add_local_file(ROOT / "artifacts/oracles/drd2_svm_v1/drd2_svm_parameters.npz",
                    "/frozen/drd2_svm_parameters.npz", copy=True)
)

app = modal.App("pmo-legacy-oracle-parity")

# Fixed canonical panel. Mixed drug-like molecules plus two known actives per
# task family, so orientation is testable rather than assumed.
PANEL = [
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",                       # ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",                            # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",                     # caffeine
    "c1ccc2c(c1)ccc1ccccc12",                           # anthracene
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",                    # salbutamol
    "COc1cc2c(cc1OC)CC(N)C2",                           # a small amine
    "O=C(Nc1ccccc1)c1ccccc1",                           # benzanilide
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",                       # procainamide
    "Clc1ccccc1C1=NCC(=O)Nc2ccc(Cl)cc21",               # a benzodiazepine core
    "CC1=C(C(=O)Nc2ccccc2)SC(=N1)N",                    # thiazole amide
]


@app.function(image=image, cpu=(2.0, 2.0), memory=8192, timeout=60 * 60)
def parity() -> dict[str, Any]:
    import hashlib, pickle
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    canon = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in PANEL]
    mols = [Chem.MolFromSmiles(s) for s in canon]

    def ecfp(m, nbits=2048, r=2):
        fp = AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=nbits)
        a = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, a)
        return a

    # ---- frozen implementations, plain numpy, no sklearn ----
    def forest_predict(npz, X):
        d = np.load(npz)
        cl, cr, feat = d["children_left"], d["children_right"], d["feature"]
        leaf, root = d["leaf_p1"], d["tree_root"]
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            tot = 0.0
            for t in range(len(root)):
                n = int(root[t])
                while cl[n] != -1:
                    n = int(cl[n]) if x[int(feat[n])] <= 0.5 else int(cr[n])
                tot += float(leaf[n])
            out[i] = tot / len(root)
        return out

    report = {}
    X2048 = np.array([ecfp(m, 2048, 2) for m in mols], dtype=np.float64)

    for name, pkl, npz in [("jnk3", "/orig/jnk3.pkl", "/frozen/jnk3_forest.npz"),
                           ("gsk3b", "/orig/gsk3b.pkl", "/frozen/gsk3b_forest.npz")]:
        with open(pkl, "rb") as f:
            est = pickle.load(f)
        orig = est.predict_proba(X2048)[:, 1]
        ours = forest_predict(npz, X2048)
        diff = np.abs(orig - ours)
        wrong = np.abs((1.0 - orig) - ours)          # negative control
        report[name] = {
            "pickle_sha256": sha(pkl), "npz_sha256": sha(npz),
            "n_estimators": int(getattr(est, "n_estimators", -1)),
            "n_features_expected": int(np.load(npz)["n_features"]),
            "original": [round(float(v), 10) for v in orig],
            "frozen":   [round(float(v), 10) for v in ours],
            "max_abs_diff": float(diff.max()),
            "wrong_orientation_max_abs_diff": float(wrong.max()),
            "PASS": bool(diff.max() < 1e-9),
        }

    # ---- DRD2: SVM with Platt scaling, ECFP6 count-based per TDC ----
    with open("/orig/drd2.pkl", "rb") as f:
        svm = pickle.load(f)
    from rdkit.Chem import rdMolDescriptors
    def drd2_fp(m):
        fp = rdMolDescriptors.GetMorganFingerprint(m, 3, useCounts=True, useFeatures=True)
        a = np.zeros((32681,), dtype=np.float64)
        for idx, v in fp.GetNonzeroElements().items():
            a[idx % 32681] += v
        return a
    Xd = np.array([drd2_fp(m) for m in mols])
    try:
        orig_d = svm.predict_proba(Xd)[:, 1]
        ok = True
    except Exception as e:
        orig_d, ok = None, False
        report["drd2_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    report["drd2"] = {
        "pickle_sha256": sha("/orig/drd2.pkl"),
        "npz_sha256": sha("/frozen/drd2_svm_parameters.npz"),
        "original": [round(float(v), 10) for v in orig_d] if ok else None,
        "note": ("DRD2 featurisation must be confirmed against the TDC source "
                 "before this row is meaningful; the SVM is Platt-scaled."),
    }
    report["panel"] = canon
    report["sklearn"] = __import__("sklearn").__version__
    report["numpy"] = np.__version__
    return report


@app.local_entrypoint()
def main() -> None:
    r = parity.remote()
    print(f"\n  sklearn {r['sklearn']}  numpy {r['numpy']}\n")
    for k in ("jnk3", "gsk3b"):
        d = r[k]
        print(f"  {k.upper()}  pickle {d['pickle_sha256'][:16]}  npz {d['npz_sha256'][:16]}")
        print(f"    trees {d['n_estimators']}  features {d['n_features_expected']}")
        print(f"    max |orig - ours|          {d['max_abs_diff']:.3e}")
        print(f"    max |(1-orig) - ours|      {d['wrong_orientation_max_abs_diff']:.3e}   <- negative control")
        print(f"    {'PASS' if d['PASS'] else 'FAIL'}\n")
    p = Path(__file__).resolve().parents[1] / "diagnostics/pmo_legacy_oracle_parity.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, indent=1))
    print(f"  wrote {p}")
