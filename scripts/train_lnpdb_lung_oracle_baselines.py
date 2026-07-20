#!/usr/bin/env python3
"""Leakage-resistant molecular baselines for every verified LNPDB lung task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.train_lumi_oracle_baselines import molecular_features


TASKS = {
    "airway_a549_in_vitro_expression": {
        "experiment_id": "JW_2024",
        "model_type": "A549",
    },
    "airway_hbec_ali_in_vitro_expression": {
        "experiment_id": "JW_2024",
        "model_type": "HBEC_ALI",
    },
    "local_intratracheal_functional_expression": {
        "experiment_id": "BL_2023",
        "model_type": "Mouse_B6",
    },
    "systemic_iv_barcoded_lung_uptake": {
        "experiment_id": "LX_2024",
        "model_type": "Mouse_B6",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_smiles(value: str) -> str:
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        raise ValueError(f"unparsable LNPDB lipid: {value!r}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    count = len(target)
    top_count = max(1, int(np.ceil(0.1 * count)))
    true_top = set(np.argsort(target)[-top_count:].tolist())
    predicted_top = set(np.argsort(prediction)[-top_count:].tolist())
    overlap = len(true_top & predicted_top)
    prevalence = top_count / count
    rank = spearmanr(target, prediction).statistic
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "r2": float(r2_score(target, prediction)),
        "spearman": float(rank) if np.isfinite(rank) else 0.0,
        "top_decile_recall": float(overlap / top_count),
        "top_decile_enrichment": float((overlap / top_count) / prevalence),
    }


def cross_validate(
    estimator: object,
    features: np.ndarray,
    target: np.ndarray,
    splits: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    predictions = np.full(target.shape, np.nan, dtype=np.float64)
    folds: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splits):
        fitted = clone(estimator)
        fitted.fit(features[train], target[train])
        predicted = np.asarray(fitted.predict(features[test]), dtype=np.float64)
        predictions[test] = predicted
        folds.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                **metrics(target[test], predicted),
            }
        )
    if np.isnan(predictions).any():
        raise RuntimeError("incomplete cross-validated predictions")
    return predictions, folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/Users/rmaganti/Desktop/thesis_projects_ML/diffusion_project/"
            "lipid_diffusion/external/LNPDB/LNPDB.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/lnpdb_lung_task_baselines.json"),
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()

    frame = pd.read_csv(args.data, low_memory=False)
    required = {
        "Experiment_ID",
        "Model",
        "Model_type",
        "Model_target",
        "IL_SMILES",
        "Experiment_value",
        "Experiment_method",
        "Experiment_batching",
        "Route_of_administration",
        "Publication_PMID",
    }
    if not required.issubset(frame.columns):
        raise ValueError("LNPDB artifact lacks required lung-oracle columns")
    lung = frame[
        frame["Model_target"].astype(str).str.contains("lung", case=False, na=False)
    ].copy()
    if len(lung) != 1975:
        raise ValueError(f"expected frozen 1,975-row LNPDB lung slice, found {len(lung)}")

    reports: list[dict[str, object]] = []
    task_summaries: dict[str, dict[str, object]] = {}
    for task_name, selector in TASKS.items():
        task = lung[
            (lung["Experiment_ID"] == selector["experiment_id"])
            & (lung["Model_type"] == selector["model_type"])
        ].copy()
        if task.empty:
            raise ValueError(f"empty frozen LNPDB task: {task_name}")
        task["canonical_smiles"] = [canonical_smiles(value) for value in task["IL_SMILES"]]
        target = task["Experiment_value"].to_numpy(dtype=np.float64)
        representations = molecular_features(task["canonical_smiles"])
        estimators = {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=int(args.trees),
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=int(args.seed),
            ),
        }
        groups = task["canonical_smiles"].to_numpy()
        group_folds = min(5, len(np.unique(groups)))
        splitters = {
            "random_diagnostic": tuple(
                KFold(n_splits=5, shuffle=True, random_state=args.seed).split(task)
            ),
            "held_lipid": tuple(
                GroupKFold(n_splits=group_folds).split(task, target, groups)
            ),
        }
        for representation_name, features in representations.items():
            for model_name, estimator in estimators.items():
                for split_name, splits in splitters.items():
                    predictions, folds = cross_validate(estimator, features, target, splits)
                    reports.append(
                        {
                            "task": task_name,
                            "representation": representation_name,
                            "model": model_name,
                            "split": split_name,
                            "folds": folds,
                            **metrics(target, predictions),
                        }
                    )
        task_summaries[task_name] = {
            "rows": int(len(task)),
            "unique_lipids": int(task["canonical_smiles"].nunique()),
            "experiment_id": selector["experiment_id"],
            "model_type": selector["model_type"],
            "route": sorted(task["Route_of_administration"].dropna().unique().tolist()),
            "method": sorted(task["Experiment_method"].dropna().unique().tolist()),
            "batching": sorted(task["Experiment_batching"].dropna().unique().tolist()),
            "publication_pmids": sorted(
                str(value) for value in task["Publication_PMID"].dropna().unique()
            ),
            "target_minimum": float(target.min()),
            "target_maximum": float(target.max()),
        }

    output = {
        "format": "compose_lipid_lnpdb_lung_baselines_v1",
        "artifact": str(args.data),
        "artifact_sha256": sha256(args.data),
        "lung_rows": int(len(lung)),
        "lung_unique_lipids": int(lung["IL_SMILES"].nunique()),
        "tasks": task_summaries,
        "reports": reports,
        "interpretation_boundary": (
            "Each route/readout is modeled as a separate task. Held-lipid folds prevent "
            "the same ionizable lipid from leaking across train and test, but these "
            "single-study tasks still require leave-study external validation. Barcode "
            "uptake is not interpreted as functional expression."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    best: dict[str, dict[str, object]] = {}
    for report in reports:
        if report["split"] != "held_lipid":
            continue
        task = str(report["task"])
        if task not in best or float(report["spearman"]) > float(best[task]["spearman"]):
            best[task] = report
    print(
        json.dumps(
            {
                task: {
                    key: report[key]
                    for key in (
                        "representation",
                        "model",
                        "mae",
                        "r2",
                        "spearman",
                        "top_decile_enrichment",
                    )
                }
                for task, report in sorted(best.items())
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
