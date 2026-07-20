#!/usr/bin/env python3
"""Does a RICHER (RDKit-native) representation lift weak cross-chemistry transfer?

`scripts/qualify_lion_leave_library.py` found that holding out the entire
`RM_Michael_addition_branched` library from the LiON A549 lung screen and
predicting it from an oracle trained on the OTHER chemistries yields only a
Spearman of ~+0.08 (vs ~+0.63 when the Michael chemistry is INCLUDED in a random
5-fold).  This script asks a narrow representation question: with the SAME data,
SAME leave-library-out / positive-control protocol, SAME ExtraTreesRegressor, does
concatenating additional complementary RDKit descriptors onto the Morgan+RDKit
baseline raise the held-out Michael Spearman?

HONESTY NOTE: no pretrained molecular encoder is installed (transformers /
molfeat / unimol / torch_geometric / deepchem are ABSENT -- only RDKit is
available).  The "richer" representation here is therefore a RICHER RDKit
DESCRIPTOR representation -- NOT a learned/frozen neural embedding.  It is the
baseline Morgan-2048 + RDKit descriptors concatenated with MACCS keys, the Avalon
fingerprint, a hashed atom-pair fingerprint, a hashed topological-torsion
fingerprint, and a folded Gobbi 2D-pharmacophore fingerprint.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:. \
        python3 -m scripts.qualify_lion_representation_transfer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

from scripts.train_lumi_oracle_baselines import molecular_features

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260720
FEATURE_VERSION = "richer_rdkit_v2"

# ExtraTreesRegressor hyperparameters -- identical to qualify_lion_leave_library.py
ET_KW = dict(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
             n_jobs=-1, random_state=SEED)

# --- optional richer-descriptor imports (record any that are unavailable) ------
_SKIPPED: list[dict[str, str]] = []

try:
    from rdkit.Chem import MACCSkeys as _MACCS
except Exception as exc:  # pragma: no cover - MACCS ships with rdkit
    _MACCS = None
    _SKIPPED.append({"name": "maccs_keys", "reason": f"{type(exc).__name__}: {exc}"})

try:
    from rdkit.Avalon import pyAvalonTools as _AVALON
except Exception as exc:  # pragma: no cover - depends on rdkit build
    _AVALON = None
    _SKIPPED.append({"name": "avalon_fp", "reason": f"{type(exc).__name__}: {exc}"})

try:
    from rdkit.Chem import rdMolDescriptors as _RDMD
except Exception as exc:  # pragma: no cover - core rdkit module
    _RDMD = None
    _SKIPPED.append({"name": "atom_pair_and_torsion", "reason": f"{type(exc).__name__}: {exc}"})

try:
    from rdkit.Chem.Pharm2D import Generate as _PH2D_GEN
    from rdkit.Chem.Pharm2D.Gobbi_Pharm2D import factory as _GOBBI_FACTORY
    from rdkit.DataStructs import FoldFingerprint as _FOLD
except Exception as exc:  # pragma: no cover - depends on rdkit build
    _PH2D_GEN = None
    _GOBBI_FACTORY = None
    _FOLD = None
    _SKIPPED.append({"name": "pharm2d_gobbi", "reason": f"{type(exc).__name__}: {exc}"})

# atom-pair / topological-torsion hashed sizes and pharm2d fold factor
_AP_BITS = 2048
_TT_BITS = 2048
_PHARM2D_FOLD_FACTOR = 10  # 39,972 -> ~3,997 bits (keeps it in scale with the other families)


def _enrich(t: np.ndarray, p: np.ndarray) -> float:
    n = len(t)
    k = max(1, int(np.ceil(0.1 * n)))
    return float((len(set(np.argsort(t)[-k:]) & set(np.argsort(p)[-k:])) / k) / (k / n))


def _bitvect_to_np(bitvect, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def _sparse_bitvect_to_np(bitvect, n_bits: int) -> np.ndarray:
    # SparseBitVect (e.g. Pharm2D / folded Pharm2D) is not accepted by
    # ConvertToNumpyArray; densify via its on-bit indices instead.
    arr = np.zeros(n_bits, dtype=np.float32)
    for idx in bitvect.GetOnBits():
        arr[idx] = 1.0
    return arr


def richer_extra_features(canon_smiles: list[str]) -> tuple[np.ndarray, list[dict], list[dict]]:
    """RDKit-native complementary descriptors, concatenated per molecule.

    Returns (matrix, included_spec, skipped_spec).  Each fingerprint family that
    fails to import is skipped and recorded rather than aborting the run.
    """
    mols = [Chem.MolFromSmiles(s) for s in canon_smiles]
    if any(m is None for m in mols):
        raise ValueError("richer representation received an unparsable canonical SMILES")

    columns: list[np.ndarray] = []
    included: list[dict] = []

    if _MACCS is not None:
        maccs = np.stack([_bitvect_to_np(_MACCS.GenMACCSKeys(m), 167) for m in mols])
        columns.append(maccs)
        included.append({"name": "maccs_keys", "dim": int(maccs.shape[1]),
                         "note": "rdkit.Chem.MACCSkeys.GenMACCSKeys (167 structural keys)"})

    if _AVALON is not None:
        av_bits = len(_AVALON.GetAvalonFP(mols[0]))
        avalon = np.stack([_bitvect_to_np(_AVALON.GetAvalonFP(m), av_bits) for m in mols])
        columns.append(avalon)
        included.append({"name": "avalon_fp", "dim": int(avalon.shape[1]),
                         "note": "rdkit.Avalon.pyAvalonTools.GetAvalonFP (default nBits)"})

    if _RDMD is not None:
        ap = np.stack([_bitvect_to_np(
            _RDMD.GetHashedAtomPairFingerprintAsBitVect(m, nBits=_AP_BITS), _AP_BITS) for m in mols])
        columns.append(ap)
        included.append({"name": "atom_pair_hashed", "dim": int(ap.shape[1]),
                         "note": f"GetHashedAtomPairFingerprintAsBitVect(nBits={_AP_BITS})"})

        tt = np.stack([_bitvect_to_np(
            _RDMD.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=_TT_BITS), _TT_BITS)
            for m in mols])
        columns.append(tt)
        included.append({"name": "topological_torsion_hashed", "dim": int(tt.shape[1]),
                         "note": f"GetHashedTopologicalTorsionFingerprintAsBitVect(nBits={_TT_BITS})"})

    if _PH2D_GEN is not None and _GOBBI_FACTORY is not None and _FOLD is not None:
        raw0 = _PH2D_GEN.Gen2DFingerprint(mols[0], _GOBBI_FACTORY)
        raw_bits = raw0.GetNumBits()
        folded_bits = _FOLD(raw0, _PHARM2D_FOLD_FACTOR).GetNumBits()
        ph = np.stack([_sparse_bitvect_to_np(
            _FOLD(_PH2D_GEN.Gen2DFingerprint(m, _GOBBI_FACTORY), _PHARM2D_FOLD_FACTOR), folded_bits)
            for m in mols])
        columns.append(ph)
        included.append({"name": "pharm2d_gobbi_folded", "dim": int(ph.shape[1]),
                         "note": (f"rdkit.Chem.Pharm2D Gobbi factory 2D pharmacophore, "
                                  f"folded {raw_bits}->{folded_bits} (fold factor "
                                  f"{_PHARM2D_FOLD_FACTOR}) to stay in scale with the other "
                                  f"fingerprint families; raw is very high-dimensional/sparse")})

    if not columns:
        raise RuntimeError("no richer descriptors were available to build a representation")
    matrix = np.concatenate(columns, axis=1).astype(np.float32)
    return matrix, included, list(_SKIPPED)


def _feature_cache_path(canon_smiles: list[str], included_tag: str) -> Path:
    key = hashlib.sha256(
        (FEATURE_VERSION + "|" + included_tag + "|" + "\n".join(canon_smiles)).encode()
    ).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"lion_repr_richer_{key}.npz"


def _build_richer(canon_smiles: list[str], cache: bool) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Compute (or load from an OS-temp cache) the richer extra-feature block.

    The Gobbi 2D pharmacophore is expensive on these large lipids (~1 s/mol), so a
    hash-keyed npz cache makes reruns of the exact command cheap without leaving
    artifacts in the repo.  The cache stores only the extra-descriptor block; the
    baseline is always recomputed from `molecular_features`.
    """
    included_tag = f"maccs{_MACCS is not None}_av{_AVALON is not None}_rd{_RDMD is not None}_ph{_PH2D_GEN is not None}"
    cache_path = _feature_cache_path(canon_smiles, included_tag)
    if cache and cache_path.exists():
        try:
            blob = np.load(cache_path, allow_pickle=True)
            matrix = blob["matrix"]
            included = json.loads(str(blob["included"]))
            skipped = json.loads(str(blob["skipped"]))
            print(f"richer features loaded from cache: {cache_path} shape={matrix.shape}")
            return matrix, included, skipped
        except Exception:
            pass
    t0 = time.time()
    matrix, included, skipped = richer_extra_features(canon_smiles)
    print(f"richer extra-features computed: shape={matrix.shape} in {time.time() - t0:.1f}s")
    if cache:
        try:
            np.savez_compressed(cache_path, matrix=matrix,
                                included=json.dumps(included), skipped=json.dumps(skipped))
            print(f"richer features cached to: {cache_path}")
        except Exception:
            pass
    return matrix, included, skipped


def run_protocol(features: np.ndarray, y: np.ndarray, libs: np.ndarray) -> dict:
    """Leave-library-out + Michael-included positive control (identical protocol)."""
    features = np.asarray(features, float)
    per_library: dict[str, dict] = {}
    for held in sorted(set(libs)):
        train = libs != held
        test = libs == held
        if test.sum() < 10 or train.sum() < 30:
            continue
        model = ExtraTreesRegressor(**ET_KW)
        model.fit(features[train], y[train])
        pred = model.predict(features[test])
        rank = spearmanr(y[test], pred).statistic
        per_library[str(held)] = {
            "held_out_lipids": int(test.sum()), "train_lipids": int(train.sum()),
            "spearman": round(float(rank) if np.isfinite(rank) else 0.0, 3),
            "top_decile_enrichment": round(_enrich(y[test], pred), 2),
        }

    michael = next((k for k in per_library if "michael" in k.lower()), None)

    # POSITIVE CONTROL: Michael-addition data INCLUDED via random 5-fold; report
    # ranking on the Michael subset (the deploy configuration).
    mich_mask = np.array([michael is not None and lib == michael for lib in libs])
    pooled_pred = np.zeros(len(y))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=SEED).split(features):
        m = ExtraTreesRegressor(**ET_KW)
        m.fit(features[tr], y[tr])
        pooled_pred[te] = m.predict(features[te])
    pos_michael = (spearmanr(y[mich_mask], pooled_pred[mich_mask]).statistic
                   if mich_mask.sum() >= 10 else None)
    positive_control = {
        "protocol": "random 5-fold over all A549 LiON (Michael-addition INCLUDED in training)",
        "michael_subset_spearman": round(float(pos_michael), 3) if pos_michael is not None else None,
        "michael_subset_enrichment": (round(_enrich(y[mich_mask], pooled_pred[mich_mask]), 2)
                                      if mich_mask.sum() >= 10 else None),
        "overall_spearman": round(float(spearmanr(y, pooled_pred).statistic), 3),
    }
    return {
        "feature_dim": int(features.shape[1]),
        "per_library": per_library,
        "michael_addition_library": michael,
        "michael_leaveout_spearman": per_library.get(michael, {}).get("spearman") if michael else None,
        "positive_control": positive_control,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lion", type=Path, default=Path(
        "/Users/rmaganti/Desktop/thesis_projects_ML/diffusion_project/lipid_diffusion/external/LNP_ML/data/all_data.csv"))
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "diagnostics/lion_representation_transfer.json")
    parser.add_argument("--no-cache", action="store_true",
                        help="recompute the richer descriptors instead of using the OS-temp cache")
    args = parser.parse_args()

    # --- identical LiON data loading / filtering / target as qualify_lion_leave_library ---
    df = pd.read_csv(args.lion, low_memory=False)
    a = df[df["Model_type"].astype(str).eq("A549")].copy()
    a = a[a["smiles"].notna() & a["unnormalized_delivery"].notna() & a["Library_ID"].notna()]
    a["canon"] = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(str(s)) else None
                  for s in a["smiles"]]
    a = a[a["canon"].notna()].copy()
    v = a["unnormalized_delivery"].to_numpy(float)
    a["target"] = np.log10(np.clip(v, np.nanmin(v[v > 0]) if (v > 0).any() else 1e-6, None))

    lib_counts = a["Library_ID"].value_counts()
    keep_libs = lib_counts[lib_counts >= 30].index.tolist()
    a = a[a["Library_ID"].isin(keep_libs)].reset_index(drop=True)

    y = a["target"].to_numpy(float)
    libs = a["Library_ID"].to_numpy()
    canon = a["canon"].tolist()

    # baseline representation: exactly molecular_features(...)["combined"]
    baseline = np.asarray(molecular_features(canon)["combined"], float)

    # richer representation: baseline ++ additional RDKit descriptor families
    extra, included, skipped = _build_richer(canon, cache=not args.no_cache)
    richer = np.concatenate((baseline, extra), axis=1)

    print(f"LiON A549: {len(a)} lipids | baseline dim={baseline.shape[1]} | "
          f"richer dim={richer.shape[1]} (+{extra.shape[1]} added)")

    baseline_result = run_protocol(baseline, y, libs)
    richer_result = run_protocol(richer, y, libs)

    baseline_result["description"] = (
        "molecular_features(...)['combined']: 10 RDKit descriptors + Morgan-2048 "
        "(identical to scripts/qualify_lion_leave_library.py)")
    richer_result["description"] = (
        "RICHER RDKit DESCRIPTOR representation (NOT a learned/neural embedding): "
        "baseline ++ " + " ++ ".join(f"{d['name']}({d['dim']})" for d in included))
    richer_result["added_descriptors"] = included
    richer_result["added_dim"] = int(extra.shape[1])

    b_mich = baseline_result["michael_leaveout_spearman"]
    r_mich = richer_result["michael_leaveout_spearman"]
    delta = (round(float(r_mich) - float(b_mich), 3)
             if (b_mich is not None and r_mich is not None) else None)

    b_pc = baseline_result["positive_control"]["michael_subset_spearman"]
    r_pc = richer_result["positive_control"]["michael_subset_spearman"]
    pc_delta = (round(float(r_pc) - float(b_pc), 3)
                if (b_pc is not None and r_pc is not None) else None)

    if delta is None:
        verdict = "indeterminate (Michael leave-out Spearman missing for a representation)"
    elif delta >= 0.05:
        verdict = "lifts"
    elif delta <= -0.05:
        verdict = "lowers"
    else:
        verdict = "no meaningful lift (null result)"

    interpretation = (
        f"Richer RDKit descriptors {verdict} cross-chemistry transfer to the held-out "
        f"Michael-addition library: leave-out Spearman moves {b_mich} -> {r_mich} "
        f"(delta {delta}). This is a RICHER RDKIT DESCRIPTOR representation (MACCS + Avalon "
        f"+ atom-pair + topological-torsion + folded Gobbi 2D-pharmacophore concatenated onto "
        f"the Morgan+RDKit baseline), NOT a learned/frozen neural embedding -- no pretrained "
        f"molecular encoder (transformers/molfeat/unimol/torch_geometric/deepchem) is installed. "
        f"A null result would mean the weak transfer is a genuine data/chemistry-coverage limit, "
        f"not a representation artifact, and supports the head-aware-AD-gate + active-learning "
        f"calibration design rather than richer featurization alone."
    )

    out = {
        "format": "compose_lion_representation_transfer_v1",
        "task": ("A549 in-vitro delivery, leave-LIBRARY-out (train on other chemistries, test on "
                 "held library) + Michael-included positive control, run under TWO representations"),
        "seed": SEED,
        "n_lipids": int(len(a)),
        "n_libraries": len(baseline_result["per_library"]),
        "michael_addition_library": baseline_result["michael_addition_library"],
        "representations": {
            "baseline_morgan_rdkit": baseline_result,
            "richer_rdkit_descriptors": richer_result,
        },
        "skipped_descriptors": skipped,
        "michael_leaveout_delta": delta,
        "michael_positive_control_delta": pc_delta,
        "representation_lifts_transfer": verdict,
        "interpretation": interpretation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print("\n=== Michael-addition leave-library-out (cross-chemistry transfer) ===")
    print(f"  baseline (dim {baseline_result['feature_dim']:5d}):  Spearman = {b_mich}")
    print(f"  richer   (dim {richer_result['feature_dim']:5d}):  Spearman = {r_mich}")
    print(f"  delta (richer - baseline)               = {delta}   => {verdict}")
    print("\n=== Positive control (Michael INCLUDED, random 5-fold) ===")
    print(f"  baseline Michael-subset Spearman = {b_pc}")
    print(f"  richer   Michael-subset Spearman = {r_pc}  (delta {pc_delta})")
    if skipped:
        print("\nskipped descriptors:", ", ".join(s["name"] for s in skipped))
    print(f"\nwritten: {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
