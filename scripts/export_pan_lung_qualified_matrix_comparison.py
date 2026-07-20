#!/usr/bin/env python3
"""Compare frozen linear, ExtraTrees, and boosted pan-lung oracle cells.

Only the currently qualified signals enter this comparison: LNPDB A549,
LUMI HBE expression, and LuT systemic-IV selectivity.  Assay heads remain
separate, all splits hold out chemical groups, and every attempted prediction
is retained.  The completed ExtraTrees predictions are imported from the
frozen R1xM2/R2xM2 table; Ridge and XGBoost use exactly the same fold map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from scripts.export_pan_lung_matrix_cell_predictions import (
    LUT_REPRESENTATION,
    PREDICTION_DECIMALS,
    REPO_ROOT,
    artifact_label,
    canonical_smiles,
    file_sha256,
    manifest_source,
    resolve_path,
)
from scripts.train_lumi_oracle_baselines import molecular_features, parse_markush
from scripts.train_lut_oracle_baselines import make_preprocessor


FULL_REPRESENTATION = "morgan_descriptors_plus_typed_context_v1"
QUALIFIED_HEADS = {
    "airway_a549_in_vitro_expression",
    "airway_hbe_in_vitro_expression",
    "systemic_iv_lung_selectivity",
}
OUTPUT_COLUMNS = [
    "source_id",
    "measurement_id",
    "canonical_lipid_id",
    "structure_id",
    "structure_status",
    "typed_head",
    "group_id",
    "route",
    "readout_type",
    "pooling_type",
    "split_name",
    "fold",
    "target_value",
    "prediction",
    "representation",
    "model",
    "model_family",
    "attempt_id",
]


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    count = int(len(target))
    top_count = max(1, int(np.ceil(0.1 * count)))
    true_top = set(np.argsort(target)[-top_count:].tolist())
    predicted_top = set(np.argsort(prediction)[-top_count:].tolist())
    overlap = len(true_top & predicted_top)
    if np.unique(target).size < 2 or np.unique(prediction).size < 2:
        spearman = None
        pearson = None
    else:
        spearman_value = spearmanr(target, prediction).statistic
        pearson_value = pearsonr(target, prediction).statistic
        spearman = float(spearman_value) if np.isfinite(spearman_value) else None
        pearson = float(pearson_value) if np.isfinite(pearson_value) else None
    prediction_variance = float(np.var(prediction))
    if prediction_variance <= np.finfo(np.float64).eps:
        calibration_slope = None
        calibration_intercept = None
    else:
        calibration_slope = float(np.cov(prediction, target, ddof=0)[0, 1] / prediction_variance)
        calibration_intercept = float(np.mean(target) - calibration_slope * np.mean(prediction))
    rmse = float(mean_squared_error(target, prediction) ** 0.5)
    target_std = float(np.std(target))
    return {
        "rows": count,
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": rmse,
        "normalized_rmse": float(rmse / target_std) if target_std > 0 else None,
        "r2": float(r2_score(target, prediction)),
        "spearman": spearman,
        "pearson": pearson,
        "top_decile_recall": float(overlap / top_count),
        "top_decile_enrichment": float((overlap / top_count) / (top_count / count)),
        "target_mean": float(np.mean(target)),
        "prediction_mean": float(np.mean(prediction)),
        "mean_prediction_bias": float(np.mean(prediction - target)),
        "calibration_slope_target_on_prediction": calibration_slope,
        "calibration_intercept_target_on_prediction": calibration_intercept,
        "prediction_standard_deviation": float(np.std(prediction)),
    }


def ridge_estimator() -> object:
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0, solver="lsqr", tol=1e-7, max_iter=20_000),
    )


def xgboost_estimator(seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=1,
        verbosity=0,
    )


def expected_fold_map(extra_trees: pd.DataFrame) -> dict[tuple[str, str, str, str], int]:
    return {
        (row.source_id, row.measurement_id, row.typed_head, row.split_name): int(row.fold)
        for row in extra_trees.itertuples(index=False)
    }


def append_predictions(
    *,
    features: object,
    target: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    metadata: pd.DataFrame,
    split_name: str,
    representation: str,
    estimator: object,
    model: str,
    model_family: str,
    attempt_id: str,
    expected_folds: dict[tuple[str, str, str, str], int],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splits):
        fitted = clone(estimator)
        if isinstance(features, pd.DataFrame):
            fitted.fit(features.iloc[train], target[train])
            prediction = fitted.predict(features.iloc[test])
        else:
            fitted.fit(features[train], target[train])
            prediction = fitted.predict(features[test])
        prediction = np.round(
            np.asarray(prediction, dtype=np.float64),
            decimals=PREDICTION_DECIMALS,
        )
        for position, predicted in zip(test, prediction, strict=True):
            record = metadata.iloc[int(position)].to_dict()
            key = (
                str(record["source_id"]),
                str(record["measurement_id"]),
                str(record["typed_head"]),
                split_name,
            )
            if expected_folds.get(key) != fold:
                raise RuntimeError(f"new model does not match frozen fold map for {key}")
            record.update(
                {
                    "split_name": split_name,
                    "fold": int(fold),
                    "target_value": float(target[int(position)]),
                    "prediction": float(predicted),
                    "representation": representation,
                    "model": model,
                    "model_family": model_family,
                    "attempt_id": attempt_id,
                }
            )
            records.append(record)
    return records


def full_structure_attempts(
    *,
    lnpdb_path: Path,
    lumi_path: Path,
    expected_folds: dict[tuple[str, str, str, str], int],
    seed: int,
) -> list[dict[str, object]]:
    estimators = (
        ("ridge_v1", "linear", "R1xM1", ridge_estimator()),
        ("xgboost_v1", "gradient_boosted_trees", "R1xM3", xgboost_estimator(seed)),
    )
    records: list[dict[str, object]] = []

    lnpdb = pd.read_csv(lnpdb_path, low_memory=False)
    a549 = lnpdb[
        (lnpdb["Experiment_ID"] == "JW_2024")
        & (lnpdb["Model_type"] == "A549")
        & lnpdb["Model_target"].astype(str).str.contains("lung", case=False, na=False)
    ].copy()
    if len(a549) != 1801:
        raise ValueError(f"expected 1,801 A549 rows, observed {len(a549)}")
    a549["canonical_smiles"] = [canonical_smiles(value) for value in a549["IL_SMILES"]]
    a549_target = a549["Experiment_value"].to_numpy(dtype=np.float64)
    a549_features = molecular_features(a549["canonical_smiles"])["combined"]
    a549_groups = a549["canonical_smiles"].to_numpy()
    a549_splits = tuple(GroupKFold(n_splits=5).split(a549, a549_target, a549_groups))
    a549_metadata = pd.DataFrame(
        {
            "source_id": "lnpdb_2026",
            "measurement_id": [f"lnpdb_2026:{index}" for index in a549.index],
            "canonical_lipid_id": a549["canonical_smiles"].to_numpy(),
            "structure_id": a549["canonical_smiles"].to_numpy(),
            "structure_status": "full_smiles",
            "typed_head": "airway_a549_in_vitro_expression",
            "group_id": a549_groups,
            "route": a549["Route_of_administration"].astype(str).to_numpy(),
            "readout_type": a549["Experiment_method"].astype(str).to_numpy(),
            "pooling_type": a549["Experiment_batching"].astype(str).to_numpy(),
        }
    )
    for model, family, attempt_id, estimator in estimators:
        records.extend(
            append_predictions(
                features=a549_features,
                target=a549_target,
                splits=a549_splits,
                metadata=a549_metadata,
                split_name="held_lipid",
                representation=FULL_REPRESENTATION,
                estimator=estimator,
                model=model,
                model_family=family,
                attempt_id=attempt_id,
                expected_folds=expected_folds,
            )
        )

    lumi = pd.read_csv(lumi_path)
    if len(lumi) != 1920:
        raise ValueError(f"expected 1,920 LUMI rows, observed {len(lumi)}")
    lumi_canonical = np.asarray([canonical_smiles(value) for value in lumi["mol"]])
    lumi_target = lumi["RLU (log2)"].to_numpy(dtype=np.float64)
    lumi_features = molecular_features(lumi_canonical)["combined"]
    components = tuple(parse_markush(code) for code in lumi["Markush code"])
    base_metadata = pd.DataFrame(
        {
            "source_id": "lumi_lab_4cr1920",
            "measurement_id": [f"lumi_lab_4cr1920:{index}" for index in lumi.index],
            "canonical_lipid_id": lumi_canonical,
            "structure_id": lumi_canonical,
            "structure_status": "full_smiles",
            "typed_head": "airway_hbe_in_vitro_expression",
            "group_id": "",
            "route": "in_vitro",
            "readout_type": "log2_RLU",
            "pooling_type": "individual",
        }
    )
    for role in ("R1", "R2", "R3", "R4"):
        groups = np.asarray([component[role] for component in components])
        splits = tuple(
            GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(
                lumi, lumi_target, groups
            )
        )
        metadata = base_metadata.copy()
        metadata["group_id"] = groups
        for model, family, attempt_id, estimator in estimators:
            records.extend(
                append_predictions(
                    features=lumi_features,
                    target=lumi_target,
                    splits=splits,
                    metadata=metadata,
                    split_name=f"held_component_{role}",
                    representation=FULL_REPRESENTATION,
                    estimator=estimator,
                    model=model,
                    model_family=family,
                    attempt_id=attempt_id,
                    expected_folds=expected_folds,
                )
            )
    return records


def lut_attempts(
    *,
    path: Path,
    expected_folds: dict[tuple[str, str, str, str], int],
    seed: int,
) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    if len(frame) != 444 or frame["compound_id"].nunique() != 444:
        raise ValueError("LuT table does not contain 444 unique component IDs")
    target = frame["lung_selectivity_fraction"].to_numpy(dtype=np.float64)
    splits = {
        "held_head": tuple(GroupKFold(n_splits=5).split(frame, target, frame["head"])),
        "held_tail": tuple(GroupKFold(n_splits=5).split(frame, target, frame["tail"])),
        "round1_to_round2": (
            (
                np.flatnonzero(frame["round"].to_numpy() == 1),
                np.flatnonzero(frame["round"].to_numpy() == 2),
            ),
        ),
    }
    metadata = pd.DataFrame(
        {
            "source_id": "lut_444_2026",
            "measurement_id": [f"lut_444_2026:{value}" for value in frame["compound_id"]],
            "canonical_lipid_id": "UNAVAILABLE",
            "structure_id": [f"lut_component:{value}" for value in frame["compound_id"]],
            "structure_status": "component_only_full_product_smiles_unavailable",
            "typed_head": "systemic_iv_lung_selectivity",
            "group_id": "",
            "route": "intravenous",
            "readout_type": "systemic_iv_lung_selectivity",
            "pooling_type": "individual",
        }
    )
    estimators = (
        (
            "ridge_v1",
            "linear",
            "R2xM1",
            make_pipeline(make_preprocessor(frame, "combined"), Ridge(alpha=1.0, solver="lsqr")),
        ),
        (
            "xgboost_v1",
            "gradient_boosted_trees",
            "R2xM3",
            make_pipeline(make_preprocessor(frame, "combined"), xgboost_estimator(seed)),
        ),
    )
    records: list[dict[str, object]] = []
    for split_name, fold_splits in splits.items():
        split_metadata = metadata.copy()
        if split_name == "held_head":
            split_metadata["group_id"] = frame["head"].to_numpy()
        elif split_name == "held_tail":
            split_metadata["group_id"] = frame["tail"].to_numpy()
        else:
            split_metadata["group_id"] = frame["round"].astype(str).to_numpy()
        for model, family, attempt_id, estimator in estimators:
            records.extend(
                append_predictions(
                    features=frame,
                    target=target,
                    splits=fold_splits,
                    metadata=split_metadata,
                    split_name=split_name,
                    representation=LUT_REPRESENTATION,
                    estimator=estimator,
                    model=model,
                    model_family=family,
                    attempt_id=attempt_id,
                    expected_folds=expected_folds,
                )
            )
    return records


def build_reports(predictions: pd.DataFrame) -> list[dict[str, object]]:
    group_columns = [
        "attempt_id",
        "model",
        "model_family",
        "representation",
        "source_id",
        "typed_head",
        "split_name",
    ]
    reports: list[dict[str, object]] = []
    for keys, group in predictions.groupby(group_columns, sort=True):
        report = dict(zip(group_columns, keys, strict=True))
        report.update(regression_metrics(group["target_value"], group["prediction"]))
        folds = []
        for fold, fold_group in group.groupby("fold", sort=True):
            folds.append(
                {
                    "fold": int(fold),
                    **regression_metrics(
                        fold_group["target_value"], fold_group["prediction"]
                    ),
                }
            )
        report["folds"] = folds
        reports.append(report)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "configs/pan_lung_corpus_manifest_v1.json",
    )
    parser.add_argument(
        "--extra-trees-predictions",
        type=Path,
        default=REPO_ROOT / "diagnostics/pan_lung_matrix_R1M2_R2M2_predictions.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO_ROOT / "diagnostics/pan_lung_qualified_matrix_comparison_predictions.csv",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=REPO_ROOT / "diagnostics/pan_lung_qualified_matrix_comparison_metrics.json",
    )
    parser.add_argument("--lnpdb", type=Path, default=None)
    parser.add_argument("--lumi", type=Path, default=None)
    parser.add_argument("--lut", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    sources = {
        source_id: manifest_source(manifest, source_id)
        for source_id in ("lnpdb_2026", "lumi_lab_4cr1920", "lut_444_2026")
    }
    lut_table = next(
        value
        for value in sources["lut_444_2026"]["local_paths"]
        if str(value).endswith(".csv")
    )
    source_paths = {
        "lnpdb": resolve_path(args.lnpdb or Path(sources["lnpdb_2026"]["local_path"])),
        "lumi": resolve_path(args.lumi or Path(sources["lumi_lab_4cr1920"]["local_path"])),
        "lut": resolve_path(args.lut or Path(lut_table)),
    }
    expected_hashes = {
        "lnpdb": sources["lnpdb_2026"]["sha256"],
        "lumi": sources["lumi_lab_4cr1920"]["sha256"],
        "lut": sources["lut_444_2026"]["sha256"]["extracted_table"],
    }
    source_hashes = {name: file_sha256(path) for name, path in source_paths.items()}
    if source_hashes != expected_hashes:
        raise ValueError(
            f"source artifacts do not match frozen manifest: expected={expected_hashes}, "
            f"observed={source_hashes}"
        )

    extra_path = resolve_path(args.extra_trees_predictions)
    extra = pd.read_csv(extra_path)
    extra = extra[extra["typed_head"].isin(QUALIFIED_HEADS)].copy()
    extra["model_family"] = "extra_trees"
    extra["attempt_id"] = np.where(
        extra["representation"] == FULL_REPRESENTATION, "R1xM2", "R2xM2"
    )
    if len(extra) != 10_495:
        raise ValueError(f"expected 10,495 qualified ExtraTrees rows, observed {len(extra)}")
    frozen_folds = expected_fold_map(extra)

    new_records = full_structure_attempts(
        lnpdb_path=source_paths["lnpdb"],
        lumi_path=source_paths["lumi"],
        expected_folds=frozen_folds,
        seed=args.seed,
    )
    new_records.extend(
        lut_attempts(
            path=source_paths["lut"], expected_folds=frozen_folds, seed=args.seed
        )
    )
    predictions = pd.concat(
        (extra[OUTPUT_COLUMNS], pd.DataFrame.from_records(new_records)[OUTPUT_COLUMNS]),
        ignore_index=True,
    )
    key = ["attempt_id", "source_id", "measurement_id", "typed_head", "split_name"]
    if predictions.duplicated(key).any():
        raise RuntimeError("comparison contains duplicate attempt/measurement/split keys")
    if predictions["prediction"].isna().any():
        raise RuntimeError("comparison contains missing predictions")
    predictions = predictions.sort_values(key).reset_index(drop=True)
    args.predictions = resolve_path(args.predictions)
    args.metrics = resolve_path(args.metrics)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions, index=False)

    reports = build_reports(predictions)
    output = {
        "format": "compose_pan_lung_qualified_matrix_comparison_v1",
        "qualified_heads": sorted(QUALIFIED_HEADS),
        "excluded_heads": {
            "airway_hbec_ali_in_vitro_expression": "29 rows and negative held-lipid rank correlation",
            "local_intratracheal_functional_expression": "49 rows and negative held-lipid rank correlation",
            "systemic_iv_barcoded_lung_uptake": "incompatible uptake readout and near-zero held-lipid rank correlation",
            "systemic_iv_lung_expression": "LuT expression transfer weaker than selectivity; retained outside this bounded qualified comparison",
        },
        "cells": ["R1xM1", "R1xM2", "R1xM3", "R2xM1", "R2xM2", "R2xM3"],
        "seed": int(args.seed),
        "prediction_decimals": PREDICTION_DECIMALS,
        "model_specs": {
            "ridge_v1": {"alpha": 1.0, "solver": "lsqr"},
            "extra_trees_v1": {
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
            "xgboost_v1": {
                "tree_method": "hist",
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 4,
                "min_child_weight": 2.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            },
        },
        "prediction_rows": int(len(predictions)),
        "attempt_counts": {
            key: int(value) for key, value in predictions.groupby("attempt_id").size().items()
        },
        "prediction_table": artifact_label(args.predictions),
        "prediction_table_sha256": file_sha256(args.predictions),
        "extra_trees_source": {
            "path": artifact_label(extra_path),
            "sha256": file_sha256(extra_path),
        },
        "manifest": artifact_label(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "source_artifacts": {
            name: {"path": artifact_label(path), "sha256": source_hashes[name]}
            for name, path in source_paths.items()
        },
        "reports": reports,
        "metric_definitions": {
            "ranking": "Spearman correlation and top-decile recall/enrichment within one typed head and hard split.",
            "calibration": "Mean prediction bias plus slope/intercept from target regressed on held-group predictions; ideal bias=0, slope=1, intercept=0.",
            "selection": "No metric is aggregated across typed heads, routes, assays, or readout types.",
        },
        "selection_boundary": (
            "Metrics are reported separately by typed head and hard split. No score is "
            "pooled across assays, routes, or readout types; no random-fold result is used."
        ),
    }
    args.metrics.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "prediction_rows": output["prediction_rows"],
                "attempt_counts": output["attempt_counts"],
                "reports": len(reports),
                "predictions": artifact_label(args.predictions),
                "metrics": artifact_label(args.metrics),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
