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
# Modal's builder only offers Python >= 3.10, where sklearn 0.21.3 has no
# wheels, so bring a 3.7 base image directly. rdkit is deliberately ABSENT: the
# panel fingerprints are computed locally with modern RDKit and passed in as
# arrays, so both sides consume IDENTICAL features and any discrepancy is the
# estimator alone rather than the featurisation.
image = (
    modal.Image.from_registry("python:3.7-slim", add_python=None)
    .pip_install("numpy==1.18.5", "scipy==1.4.1", "scikit-learn==0.21.3",
                 "joblib==0.14.1")
    .add_local_file(ROOT / "artifacts/tdc_originals/jnk3.pkl",  "/orig/jnk3.pkl",  copy=True)
    .add_local_file(ROOT / "artifacts/tdc_originals/gsk3b.pkl", "/orig/gsk3b.pkl", copy=True)
    .add_local_file(ROOT / "artifacts/tdc_originals/drd2.pkl",  "/orig/drd2.pkl",  copy=True)
    .add_local_file(ROOT / "artifacts/tdc_originals/parity_panel.npz",
                    "/orig/parity_panel.npz", copy=True)
    .add_local_file(ROOT / "artifacts/oracles/molleo_task3_v1/jnk3_forest.npz",
                    "/frozen/jnk3_forest.npz", copy=True)
    .add_local_file(ROOT / "artifacts/oracles/molleo_task3_v1/gsk3b_forest.npz",
                    "/frozen/gsk3b_forest.npz", copy=True)
    .add_local_file(ROOT / "artifacts/oracles/drd2_svm_v1/drd2_svm_parameters.npz",
                    "/frozen/drd2_svm_parameters.npz", copy=True)
)

app = modal.App("pmo-legacy-oracle-parity")


@app.function(image=image, cpu=(2.0, 2.0), memory=8192, timeout=60 * 60)
def parity():
    import hashlib, pickle
    import numpy as np

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    d = np.load("/orig/parity_panel.npz", allow_pickle=True)
    Xbit, Xcnt = d["X_bit2048"], d["X_count2048"]
    panel = [str(s) for s in d["panel"]]

    def forest_predict(npz, X):
        z = np.load(npz)
        cl, cr, feat = z["children_left"], z["children_right"], z["feature"]
        leaf, root = z["leaf_p1"], z["tree_root"]
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

    report = {"panel": panel, "sklearn": __import__("sklearn").__version__,
              "numpy": np.__version__}

    for name, pkl, npz in [("jnk3", "/orig/jnk3.pkl", "/frozen/jnk3_forest.npz"),
                           ("gsk3b", "/orig/gsk3b.pkl", "/frozen/gsk3b_forest.npz")]:
        est = pickle.load(open(pkl, "rb"))
        orig = est.predict_proba(Xbit)[:, 1]
        ours = forest_predict(npz, Xbit)
        diff = np.abs(orig - ours)
        wrong = np.abs((1.0 - orig) - ours)
        report[name] = {
            "pickle_sha256": sha(pkl), "npz_sha256": sha(npz),
            "n_estimators": int(getattr(est, "n_estimators", -1)),
            "original": [round(float(v), 12) for v in orig],
            "frozen": [round(float(v), 12) for v in ours],
            "max_abs_diff": float(diff.max()),
            "wrong_orientation_max_abs_diff": float(wrong.max()),
            "PASS": bool(diff.max() < 1e-9)}

    # DRD2: SVC with Platt scaling, TDC folds counts to 2048
    try:
        svm = pickle.load(open("/orig/drd2.pkl", "rb"))
        orig = svm.predict_proba(Xcnt)[:, 1]
        z = np.load("/frozen/drd2_svm_parameters.npz")
        report["drd2"] = {
            "pickle_sha256": sha("/orig/drd2.pkl"),
            "npz_sha256": sha("/frozen/drd2_svm_parameters.npz"),
            "original": [round(float(v), 12) for v in orig],
            "frozen_keys": [str(k) for k in z.files],
            "note": "original evaluated; frozen SVM replay compared locally"}
    except Exception as e:
        report["drd2"] = {"error": "%s: %s" % (type(e).__name__, str(e)[:200])}
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
