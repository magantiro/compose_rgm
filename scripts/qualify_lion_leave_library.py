#!/usr/bin/env python3
"""LiON leave-library-out: does the A549 oracle transfer across chemistry libraries?

The LiON A549 lung screen carries Library_ID annotations, including a
Michael-addition-branched library (~1,199 lipids) -- the novel-linker chemistry
class. Holding out an entire library and predicting it tests cross-chemistry
transfer, and specifically whether an oracle trained on other chemistries can
rank Michael-addition lipids (the core novel-linker question) with real data.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:. \
        python3 -m scripts.qualify_lion_leave_library
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

from scripts.train_lumi_oracle_baselines import molecular_features

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260720


def _enrich(t: np.ndarray, p: np.ndarray) -> float:
    n = len(t); k = max(1, int(np.ceil(0.1 * n)))
    return float((len(set(np.argsort(t)[-k:]) & set(np.argsort(p)[-k:])) / k) / (k / n))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lion", type=Path, default=Path(
        "/Users/rmaganti/Desktop/thesis_projects_ML/diffusion_project/lipid_diffusion/external/LNP_ML/data/all_data.csv"))
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "diagnostics/lion_leave_library_transfer.json")
    args = parser.parse_args()

    df = pd.read_csv(args.lion, low_memory=False)
    a = df[df["Model_type"].astype(str).eq("A549")].copy()
    a = a[a["smiles"].notna() & a["unnormalized_delivery"].notna() & a["Library_ID"].notna()]
    a["canon"] = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(str(s)) else None
                  for s in a["smiles"]]
    a = a[a["canon"].notna()].copy()
    # log-transform the delivery signal (spans orders of magnitude)
    v = a["unnormalized_delivery"].to_numpy(float)
    a["target"] = np.log10(np.clip(v, np.nanmin(v[v > 0]) if (v > 0).any() else 1e-6, None))

    # keep libraries with enough lipids to be a real held-out group
    lib_counts = a["Library_ID"].value_counts()
    keep_libs = lib_counts[lib_counts >= 30].index.tolist()
    a = a[a["Library_ID"].isin(keep_libs)].reset_index(drop=True)

    y = a["target"].to_numpy(float)
    features = np.asarray(molecular_features(a["canon"].tolist())["combined"], float)
    libs = a["Library_ID"].to_numpy()

    per_library = {}
    for held in sorted(set(libs)):
        train = libs != held
        test = libs == held
        if test.sum() < 10 or train.sum() < 30:
            continue
        model = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                                    n_jobs=-1, random_state=SEED)
        model.fit(features[train], y[train])
        pred = model.predict(features[test])
        rank = spearmanr(y[test], pred).statistic
        per_library[str(held)] = {
            "held_out_lipids": int(test.sum()), "train_lipids": int(train.sum()),
            "spearman": round(float(rank) if np.isfinite(rank) else 0.0, 3),
            "top_decile_enrichment": round(_enrich(y[test], pred), 2),
        }

    michael = next((k for k in per_library if "michael" in k.lower()), None)
    out = {
        "format": "compose_lion_leave_library_transfer_v1",
        "task": "A549 in-vitro delivery, leave-LIBRARY-out (train on other chemistries, test on held library)",
        "n_lipids": int(len(a)), "n_libraries": len(per_library),
        "per_library": per_library,
        "michael_addition_library": michael,
        "michael_transfer_spearman": per_library.get(michael, {}).get("spearman") if michael else None,
        "interpretation": (
            "Leave-library-out tests cross-chemistry transfer with real lung data. The "
            "Michael-addition library result is the most novel-linker-relevant: whether an "
            "oracle trained on OTHER chemistries can rank Michael-addition lipids. Weak "
            "transfer supports the head-aware-AD-gate + active-learning-calibration design."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"LiON A549 leave-library-out: {len(a)} lipids, {len(per_library)} libraries")
    for lib, r in sorted(per_library.items(), key=lambda x: -x[1]["held_out_lipids"]):
        star = "  <- MICHAEL (novel-linker relevant)" if lib == michael else ""
        print(f"  held={lib:34s} n={r['held_out_lipids']:4d}  Spearman={r['spearman']:+.3f}  enrich={r['top_decile_enrichment']}x{star}")
    print(f"\nwritten: {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
