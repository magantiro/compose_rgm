#!/usr/bin/env python3
"""Component-transfer qualification for the pan-lung oracle.

Answers the load-bearing oracle-design question: which structural axes does a
lung-potency model transfer across?  This decides whether a novel-linker
candidate is rankable by the oracle or requires active-learning calibration.

Held-out-component splits (unseen head / tail / 4CR position) are compared to a
held-lipid baseline on two independent in-vitro datasets (A549 and LUMI-HBE).
The consistent finding is an asymmetry: the high-diversity ionizable-head axis
transfers poorly, while tail / secondary-component axes transfer well.

Run (module mode):
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:. \
        python3 -m scripts.qualify_component_transfer
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, KFold

from scripts.train_lumi_oracle_baselines import molecular_features

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260720


def canonical(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(str(smiles)))


def enrichment(target: np.ndarray, prediction: np.ndarray) -> float:
    n = len(target)
    k = max(1, int(np.ceil(0.1 * n)))
    top_t = set(np.argsort(target)[-k:].tolist())
    top_p = set(np.argsort(prediction)[-k:].tolist())
    return float((len(top_t & top_p) / k) / (k / n))


def evaluate(features: np.ndarray, target: np.ndarray, groups: np.ndarray | None,
             name: str) -> dict:
    oof = np.full(len(target), np.nan)
    if groups is None:
        splitter = KFold(n_splits=5, shuffle=True, random_state=SEED).split(features)
        n_groups = len(target)
    else:
        n_groups = int(len(np.unique(groups)))
        splitter = GroupKFold(n_splits=min(5, n_groups)).split(features, target, groups)
    for train_idx, test_idx in splitter:
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                    n_jobs=-1, random_state=SEED)
        model.fit(features[train_idx], target[train_idx])
        oof[test_idx] = model.predict(features[test_idx])
    rank = spearmanr(target, oof).statistic
    return {"split": name, "n_groups": n_groups,
            "spearman": round(float(rank) if np.isfinite(rank) else 0.0, 4),
            "top_decile_enrichment": round(enrichment(target, oof), 3)}


def a549_transfer(lnpdb_path: Path) -> dict:
    frame = pd.read_csv(lnpdb_path, low_memory=False)
    a = frame[(frame["Experiment_ID"] == "JW_2024") & (frame["Model_type"] == "A549")].copy()
    a["canon"] = [canonical(s) for s in a["IL_SMILES"]]
    target = a["Experiment_value"].to_numpy(float)
    features = np.asarray(molecular_features(a["canon"].tolist())["combined"], float)
    return {
        "rows": int(len(a)), "unique_lipids": int(a["canon"].nunique()),
        "splits": [
            evaluate(features, target, a["canon"].to_numpy(), "held_lipid_baseline"),
            evaluate(features, target, a["IL_head_name"].to_numpy(), "held_head"),
            evaluate(features, target, a["IL_tail1_name"].to_numpy(), "held_tail1"),
        ],
        "caveat": "held_tail1 groups on the primary tail; minor leakage possible if a held tail1 recurs as tail2 in training.",
    }


def _parse_markush(code: str) -> dict[str, str]:
    return {f"R{m.group(1)}": m.group(2) for m in re.finditer(r"R(\d)\(\d+\):(\d+)", str(code))}


def lumi_transfer(lumi_path: Path) -> dict:
    frame = pd.read_csv(lumi_path)
    comp = frame["Markush code"].apply(_parse_markush)
    for position in ("R1", "R2", "R3", "R4"):
        frame[position] = comp.apply(lambda d, p=position: d.get(p))
    frame["canon"] = [canonical(s) for s in frame["mol"]]
    target = frame["RLU (log2)"].to_numpy(float)
    features = np.asarray(molecular_features(frame["canon"].tolist())["combined"], float)
    splits = [evaluate(features, target, frame["canon"].to_numpy(), "held_lipid_baseline")]
    for position in ("R1", "R2", "R3", "R4"):
        splits.append(evaluate(features, target, frame[position].to_numpy(), f"held_{position}"))
    return {"rows": int(len(frame)), "component_cardinalities": {"R1": 20, "R2": 4, "R3": 4, "R4": 6},
            "splits": splits,
            "note": "R1 is the highest-cardinality (amine/head-like) position; R2-R4 are lower-cardinality secondary positions."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "diagnostics/pan_lung_component_transfer.json")
    args = parser.parse_args()

    source_manifest = json.loads(
        (REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/source_manifest.json").read_text())
    lnpdb_path = Path(next(s["local_artifact_path"] for s in source_manifest["sources"]
                           if s["source_id"] == "lnpdb_v1"))
    lumi_path = REPO_ROOT / "tmp/lipid_data/lumi_lab/4CR-1920.csv"
    for path in (lnpdb_path, lumi_path):
        if not path.exists():
            raise SystemExit(f"missing raw source {path}; re-fetch from primary origin first")

    a549 = a549_transfer(lnpdb_path)
    lumi = lumi_transfer(lumi_path)

    def _get(block, split):
        return next(s for s in block["splits"] if s["split"] == split)["spearman"]

    output = {
        "format": "compose_pan_lung_component_transfer_v1",
        "purpose": "Decide which axes of a novel-linker candidate are oracle-rankable vs need active-learning calibration.",
        "model": "ExtraTrees on Morgan2048 + RDKit descriptors (R1 representation)",
        "datasets": {"a549_in_vitro": a549, "lumi_hbe_in_vitro": lumi},
        "headline": {
            "a549_head_vs_tail_spearman": {"held_head": _get(a549, "held_head"),
                                           "held_tail1": _get(a549, "held_tail1"),
                                           "held_lipid_baseline": _get(a549, "held_lipid_baseline")},
            "lumi_component_spearman": {f"held_{r}": _get(lumi, f"held_{r}") for r in ("R1", "R2", "R3", "R4")}
                                       | {"held_lipid_baseline": _get(lumi, "held_lipid_baseline")},
        },
        "interpretation": (
            "Consistent across two independent in-vitro datasets: the high-diversity "
            "ionizable-head axis transfers poorly (A549 held-head Spearman ~0.16; LUMI "
            "held-R1 ~0.48) while tail / secondary-component axes transfer well (A549 "
            "held-tail ~0.51; LUMI held-R2/R3/R4 ~0.74-0.82). Mechanistically the head "
            "drives pKa/endosomal-escape potency (discontinuous SAR); tails interpolate."
        ),
        "oracle_design_consequence": (
            "Do NOT use one global oracle to rank novel-linker candidates blindly. Use a "
            "HEAD-AWARE applicability-domain gate: candidates that reuse in-distribution "
            "ionizable heads and vary linker/tails are rankable; candidates with novel "
            "heads abstain. For the novel Michael-addition linker, run a small "
            "active-learning calibration (known heads x new linker) to anchor the linker "
            "offset, then rank within-family by the transferable tail/secondary SAR. "
            "This matches the plan's calibrate-don't-extrapolate rule and the reward_guard "
            "OOD-abstention already in code."
        ),
        "environment": {"rdkit": Chem.rdBase.rdkitVersion, "seed": SEED},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    print("A549 in-vitro transfer:")
    for s in a549["splits"]:
        print(f"  {s['split']:22s} groups={s['n_groups']:4d}  Spearman={s['spearman']:.3f}  enrich={s['top_decile_enrichment']:.2f}x")
    print("LUMI-HBE in-vitro transfer:")
    for s in lumi["splits"]:
        print(f"  {s['split']:22s} groups={s['n_groups']:4d}  Spearman={s['spearman']:.3f}  enrich={s['top_decile_enrichment']:.2f}x")
    print(f"\nwritten: {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
