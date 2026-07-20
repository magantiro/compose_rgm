#!/usr/bin/env python3
"""Reproducible baselines for the published 444-lipid LuT in-vivo screen.

The Nature Biomedical Engineering source workbooks expose paired systemic-IV
lung expression and lung-selectivity measurements for 318 round-one and 126
round-two LuT lipids.  This script reconstructs the compound table directly
from those workbooks, excludes the DOTAP controls, and evaluates potency and
selectivity as separate oracle heads.

Random folds are diagnostic only.  Held-head, held-tail, and round-transfer
tests are the decision-relevant estimates of chemical generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EXPECTED_ROUND_COUNTS = {1: 318, 2: 126}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_round_potency(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Fig.2b", header=None)
    heads = tuple(str(value) for value in raw.iloc[0, 1:] if pd.notna(value))
    records: list[dict[str, object]] = []
    for row in range(1, len(raw)):
        tail = str(raw.iloc[row, 0])
        for column, head in enumerate(heads, start=1):
            if head == "DOTAP" or pd.isna(raw.iloc[row, column]):
                continue
            records.append(
                {
                    "round": 1,
                    "head": head,
                    "tail": tail,
                    "lung_expression_log10_photons_per_second": float(raw.iloc[row, column]),
                }
            )
    return pd.DataFrame.from_records(records)


def _first_round_selectivity(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Fig.2c", header=None)
    heads = tuple(str(value) for value in raw.iloc[0, 2:] if pd.notna(value))
    records: list[dict[str, object]] = []
    for row in range(1, len(raw)):
        if str(raw.iloc[row, 1]) != "Lu":
            continue
        tail_row = row - 2
        tail = str(raw.iloc[tail_row, 0])
        for column, head in enumerate(heads, start=2):
            value = raw.iloc[row, column]
            if pd.isna(value):
                continue
            records.append(
                {
                    "round": 1,
                    "head": head,
                    "tail": tail,
                    "lung_selectivity_fraction": float(value),
                }
            )
    return pd.DataFrame.from_records(records)


def _second_round_potency(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Fig.3b", header=None)
    tails = tuple(str(value) for value in raw.iloc[0, 1:] if pd.notna(value))
    records: list[dict[str, object]] = []
    for row in range(1, len(raw)):
        head = str(raw.iloc[row, 0])
        for column, tail in enumerate(tails, start=1):
            if tail == "DOTAP" or pd.isna(raw.iloc[row, column]):
                continue
            records.append(
                {
                    "round": 2,
                    "head": head,
                    "tail": tail,
                    "lung_expression_log10_photons_per_second": float(raw.iloc[row, column]),
                }
            )
    return pd.DataFrame.from_records(records)


def _second_round_selectivity(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Fig.3c", header=None)
    tails = tuple(str(value) for value in raw.iloc[0, 1:] if pd.notna(value))
    records: list[dict[str, object]] = []
    for row in range(1, len(raw)):
        head = str(raw.iloc[row, 0])
        for column, tail in enumerate(tails, start=1):
            value = raw.iloc[row, column]
            if pd.isna(value):
                continue
            records.append(
                {
                    "round": 2,
                    "head": head,
                    "tail": tail,
                    "lung_selectivity_fraction": float(value),
                }
            )
    return pd.DataFrame.from_records(records)


def load_lut_table(fig2: Path, fig3: Path) -> pd.DataFrame:
    potency = pd.concat(
        (_first_round_potency(fig2), _second_round_potency(fig3)),
        ignore_index=True,
    )
    selectivity = pd.concat(
        (_first_round_selectivity(fig2), _second_round_selectivity(fig3)),
        ignore_index=True,
    )
    keys = ["round", "head", "tail"]
    if potency.duplicated(keys).any() or selectivity.duplicated(keys).any():
        raise ValueError("LuT source data contain duplicate compound keys")
    frame = potency.merge(selectivity, on=keys, how="inner", validate="one_to_one")
    counts = frame.groupby("round").size().to_dict()
    if counts != EXPECTED_ROUND_COUNTS or len(frame) != sum(EXPECTED_ROUND_COUNTS.values()):
        raise ValueError(
            f"LuT reconstruction mismatch: expected {EXPECTED_ROUND_COUNTS}, observed {counts}"
        )
    if frame.isna().any().any():
        raise ValueError("LuT reconstruction contains missing values")

    frame["head_family"] = frame["head"].str.extract(r"^(\d+A)", expand=False)
    frame["head_index"] = frame["head"].str.extract(r"A(\d+)$", expand=False).astype(int)
    frame["tail_index"] = frame["tail"].str.extract(r"B(\d+)$", expand=False).astype(int)
    frame["tripod_compatible_head"] = frame["head"].isin(
        {"1A1", "1A2", "1A3", "1A6", "1A7", "1A8"}
    ).astype(int)
    frame["tripod_compatible_tail"] = (frame["tail_index"] >= 4).astype(int)
    frame["branched_tail"] = frame["tail"].isin(
        {f"B{index}" for index in range(13, 19)}
    ).astype(int)
    frame["unsaturated_tail"] = frame["tail"].isin(
        {"B19", "B20", "B21", "B22", "B23", "B24", "B25"}
    ).astype(int)
    frame["near_head_ester"] = frame["tail"].isin({"B11", "B16", "B20"}).astype(int)
    frame["compound_id"] = frame["head"] + frame["tail"]
    return frame.sort_values(["round", "head", "tail"]).reset_index(drop=True)


def representation_columns(name: str) -> tuple[list[str], list[str]]:
    structural_numeric = [
        "round",
        "head_index",
        "tail_index",
        "tripod_compatible_head",
        "tripod_compatible_tail",
        "branched_tail",
        "unsaturated_tail",
        "near_head_ester",
    ]
    if name == "component_ids":
        return ["head", "tail", "head_family"], ["round"]
    if name == "published_sar_descriptors":
        return ["head_family"], structural_numeric
    if name == "combined":
        return ["head", "tail", "head_family"], structural_numeric
    raise ValueError(f"unknown representation: {name}")


def make_preprocessor(frame: pd.DataFrame, representation: str) -> ColumnTransformer:
    categorical, numeric = representation_columns(representation)
    return ColumnTransformer(
        (
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ),
        remainder="drop",
    )


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    count = len(target)
    top_count = max(1, int(np.ceil(0.1 * count)))
    true_top = set(np.argsort(target)[-top_count:].tolist())
    predicted_top = set(np.argsort(prediction)[-top_count:].tolist())
    overlap = len(true_top & predicted_top)
    prevalence = top_count / count
    precision = overlap / top_count
    statistic = spearmanr(target, prediction).statistic
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "r2": float(r2_score(target, prediction)),
        "spearman": float(statistic) if np.isfinite(statistic) else 0.0,
        "top_decile_recall": float(overlap / top_count),
        "top_decile_enrichment": float(precision / prevalence),
    }


def cross_validated_predictions(
    estimator: object,
    features: pd.DataFrame,
    target: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    prediction = np.full(target.shape, np.nan, dtype=np.float64)
    fold_reports: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splits):
        fitted = clone(estimator)
        fitted.fit(features.iloc[train], target[train])
        fold_prediction = np.asarray(fitted.predict(features.iloc[test]), dtype=np.float64)
        prediction[test] = fold_prediction
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                **regression_metrics(target[test], fold_prediction),
            }
        )
    if np.isnan(prediction).all():
        raise RuntimeError("cross-validation did not produce any predictions")
    return prediction, fold_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fig2",
        type=Path,
        default=Path("tmp/lipid_data/lut/source_data_fig2.xlsx"),
    )
    parser.add_argument(
        "--fig3",
        type=Path,
        default=Path("tmp/lipid_data/lut/source_data_fig3.xlsx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/lut_444_in_vivo_baselines.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help="Optional pre-extracted LuT table; avoids XLSX parsing in lean environments.",
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--trees", type=int, default=400)
    args = parser.parse_args()

    frame = (
        pd.read_csv(args.table)
        if args.table is not None
        else load_lut_table(args.fig2, args.fig3)
    )
    if len(frame) != 444 or frame["compound_id"].nunique() != 444:
        raise ValueError("pre-extracted LuT table does not contain 444 unique compounds")
    representations = ("component_ids", "published_sar_descriptors", "combined")
    splitters: dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]] = {
        "random_diagnostic": tuple(
            KFold(n_splits=5, shuffle=True, random_state=args.seed).split(frame)
        ),
        "held_head": tuple(
            GroupKFold(n_splits=5).split(frame, groups=frame["head"])
        ),
        "held_tail": tuple(
            GroupKFold(n_splits=5).split(frame, groups=frame["tail"])
        ),
        "round1_to_round2": (
            (
                np.flatnonzero(frame["round"].to_numpy() == 1),
                np.flatnonzero(frame["round"].to_numpy() == 2),
            ),
        ),
    }
    targets = {
        "systemic_iv_lung_expression": frame[
            "lung_expression_log10_photons_per_second"
        ].to_numpy(dtype=np.float64),
        "systemic_iv_lung_selectivity": frame["lung_selectivity_fraction"].to_numpy(
            dtype=np.float64
        ),
    }

    reports: list[dict[str, object]] = []
    for representation in representations:
        preprocessor = make_preprocessor(frame, representation)
        estimators = {
            "ridge": make_pipeline(preprocessor, Ridge(alpha=10.0)),
            "extra_trees": make_pipeline(
                preprocessor,
                ExtraTreesRegressor(
                    n_estimators=int(args.trees),
                    min_samples_leaf=2,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=int(args.seed),
                ),
            ),
        }
        for target_name, target in targets.items():
            for estimator_name, estimator in estimators.items():
                for split_name, splits in splitters.items():
                    prediction, folds = cross_validated_predictions(
                        estimator,
                        frame,
                        target,
                        splits,
                    )
                    evaluated = np.flatnonzero(np.isfinite(prediction))
                    reports.append(
                        {
                            "target": target_name,
                            "representation": representation,
                            "model": estimator_name,
                            "split": split_name,
                            "evaluated_rows": int(len(evaluated)),
                            "folds": folds,
                            **regression_metrics(target[evaluated], prediction[evaluated]),
                        }
                    )

    output = {
        "format": "compose_lipid_lut_444_baseline_v1",
        "source": {
            "article": "https://www.nature.com/articles/s41551-026-01615-9",
            "fig2": str(args.fig2),
            "fig2_sha256": sha256(args.fig2),
            "fig3": str(args.fig3),
            "fig3_sha256": sha256(args.fig3),
        },
        "rows": int(len(frame)),
        "round_counts": {
            str(key): int(value) for key, value in frame.groupby("round").size().items()
        },
        "unique_heads": int(frame["head"].nunique()),
        "unique_tails": int(frame["tail"].nunique()),
        "targets": {
            name: {
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
            for name, values in targets.items()
        },
        "reports": reports,
        "interpretation_boundary": (
            "These are within-LuT combinatorial baselines. Random folds are optimistic. "
            "Held-component and round-transfer results test extrapolation but do not yet "
            "establish cross-study or novel-linker performance; molecular structures and "
            "full formulation context are required for the final pan-lung oracle."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    best: dict[str, dict[str, object]] = {}
    for report in reports:
        key = f"{report['target']}::{report['split']}"
        if key not in best or float(report["spearman"]) > float(best[key]["spearman"]):
            best[key] = report
    print(
        json.dumps(
            {
                key: {
                    metric: report[metric]
                    for metric in (
                        "representation",
                        "model",
                        "evaluated_rows",
                        "mae",
                        "r2",
                        "spearman",
                        "top_decile_enrichment",
                    )
                }
                for key, report in sorted(best.items())
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
