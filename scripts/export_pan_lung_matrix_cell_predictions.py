#!/usr/bin/env python3
"""Export held-group predictions for the first pan-lung oracle matrix cells.

The full-structure cell is Morgan fingerprints plus RDKit descriptors with
experimental context represented by typed, non-interchangeable task heads.
The adjacent LuT cell uses released head/tail and published SAR features
because full LuT product structures are not present in the source workbooks.

No random-split predictions are exported.  Each output row records the source
measurement, chemical identifier, typed head, hard split, fold, target, and
prediction so that every reported metric is auditable.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from scripts.train_lnpdb_lung_oracle_baselines import TASKS
from scripts.train_lumi_oracle_baselines import molecular_features, parse_markush
from scripts.train_lut_oracle_baselines import make_preprocessor


FULL_REPRESENTATION = "morgan_descriptors_plus_typed_context_v1"
LUT_REPRESENTATION = "lut_component_sar_plus_typed_context_v1"
MODEL = "extra_trees_v1"
PREDICTION_DECIMALS = 12
REPO_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def artifact_label(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def manifest_source(manifest: dict[str, object], source_id: str) -> dict[str, object]:
    for source in manifest["sources"]:
        if source["source_id"] == source_id:
            return source
    raise KeyError(f"source {source_id!r} is absent from the frozen manifest")


def canonical_smiles(value: str) -> str:
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        raise ValueError(f"unparsable lipid structure: {value!r}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    count = int(len(target))
    top_count = max(1, int(np.ceil(0.1 * count)))
    true_top = set(np.argsort(target)[-top_count:].tolist())
    predicted_top = set(np.argsort(prediction)[-top_count:].tolist())
    overlap = len(true_top & predicted_top)
    if np.unique(target).size < 2 or np.unique(prediction).size < 2:
        rank = 0.0
    else:
        rank = spearmanr(target, prediction).statistic
    return {
        "rows": count,
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "r2": float(r2_score(target, prediction)),
        "spearman": float(rank) if np.isfinite(rank) else 0.0,
        "top_decile_recall": float(overlap / top_count),
        "top_decile_enrichment": float((overlap / top_count) / (top_count / count)),
    }


def extra_trees(seed: int, trees: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=int(trees),
        min_samples_leaf=2,
        max_features="sqrt",
        # A single prediction reduction order makes the frozen row table
        # byte-reproducible; parallel tree summation varies at ~1e-14.
        n_jobs=1,
        random_state=int(seed),
    )


def predict_array_splits(
    *,
    features: np.ndarray,
    target: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    metadata: pd.DataFrame,
    split_name: str,
    representation: str,
    estimator: object,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_target: list[float] = []
    all_prediction: list[float] = []
    fold_metrics: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splits):
        fitted = clone(estimator)
        fitted.fit(features[train], target[train])
        prediction = np.round(
            np.asarray(fitted.predict(features[test]), dtype=np.float64),
            decimals=PREDICTION_DECIMALS,
        )
        fold_metrics.append({"fold": fold, **metrics(target[test], prediction)})
        all_target.extend(target[test].tolist())
        all_prediction.extend(prediction.tolist())
        for position, predicted in zip(test, prediction, strict=True):
            record = metadata.iloc[int(position)].to_dict()
            record.update(
                {
                    "split_name": split_name,
                    "fold": int(fold),
                    "target_value": float(target[int(position)]),
                    "prediction": float(predicted),
                    "representation": representation,
                    "model": MODEL,
                }
            )
            rows.append(record)
    aggregate = metrics(np.asarray(all_target), np.asarray(all_prediction))
    aggregate.update({"split_name": split_name, "folds": fold_metrics})
    return rows, aggregate


def predict_frame_splits(
    *,
    frame: pd.DataFrame,
    target: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    metadata: pd.DataFrame,
    split_name: str,
    representation: str,
    estimator: object,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_target: list[float] = []
    all_prediction: list[float] = []
    fold_metrics: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splits):
        fitted = clone(estimator)
        fitted.fit(frame.iloc[train], target[train])
        prediction = np.round(
            np.asarray(fitted.predict(frame.iloc[test]), dtype=np.float64),
            decimals=PREDICTION_DECIMALS,
        )
        fold_metrics.append({"fold": fold, **metrics(target[test], prediction)})
        all_target.extend(target[test].tolist())
        all_prediction.extend(prediction.tolist())
        for position, predicted in zip(test, prediction, strict=True):
            record = metadata.iloc[int(position)].to_dict()
            record.update(
                {
                    "split_name": split_name,
                    "fold": int(fold),
                    "target_value": float(target[int(position)]),
                    "prediction": float(predicted),
                    "representation": representation,
                    "model": MODEL,
                }
            )
            rows.append(record)
    aggregate = metrics(np.asarray(all_target), np.asarray(all_prediction))
    aggregate.update({"split_name": split_name, "folds": fold_metrics})
    return rows, aggregate


def lnpdb_predictions(
    path: Path, seed: int, trees: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = pd.read_csv(path, low_memory=False)
    lung = frame[
        frame["Model_target"].astype(str).str.contains("lung", case=False, na=False)
    ].copy()
    if len(lung) != 1975:
        raise ValueError(f"expected 1,975 LNPDB lung rows, found {len(lung)}")
    predictions: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for head_id, selector in TASKS.items():
        task = lung[
            (lung["Experiment_ID"] == selector["experiment_id"])
            & (lung["Model_type"] == selector["model_type"])
        ].copy()
        task["canonical_smiles"] = [canonical_smiles(value) for value in task["IL_SMILES"]]
        target = task["Experiment_value"].to_numpy(dtype=np.float64)
        features = molecular_features(task["canonical_smiles"])["combined"]
        groups = task["canonical_smiles"].to_numpy()
        split_count = min(5, len(np.unique(groups)))
        splits = tuple(GroupKFold(n_splits=split_count).split(task, target, groups))
        metadata = pd.DataFrame(
            {
                "source_id": "lnpdb_2026",
                "measurement_id": [f"lnpdb_2026:{index}" for index in task.index],
                "canonical_lipid_id": task["canonical_smiles"].to_numpy(),
                "structure_id": task["canonical_smiles"].to_numpy(),
                "structure_status": "full_smiles",
                "typed_head": head_id,
                "group_id": task["canonical_smiles"].to_numpy(),
                "route": task["Route_of_administration"].astype(str).to_numpy(),
                "readout_type": task["Experiment_method"].astype(str).to_numpy(),
                "pooling_type": task["Experiment_batching"].astype(str).to_numpy(),
            }
        )
        head_rows, report = predict_array_splits(
            features=features,
            target=target,
            splits=splits,
            metadata=metadata,
            split_name="held_lipid",
            representation=FULL_REPRESENTATION,
            estimator=extra_trees(seed, trees),
        )
        predictions.extend(head_rows)
        reports.append({"source_id": "lnpdb_2026", "typed_head": head_id, **report})
    return predictions, reports


def lumi_predictions(
    path: Path, seed: int, trees: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = pd.read_csv(path)
    if len(frame) != 1920 or set(frame.columns) != {"RLU (log2)", "mol", "Markush code"}:
        raise ValueError("LUMI artifact does not match the frozen schema")
    canonical = np.asarray([canonical_smiles(value) for value in frame["mol"]])
    target = frame["RLU (log2)"].to_numpy(dtype=np.float64)
    features = molecular_features(canonical)["combined"]
    components = tuple(parse_markush(code) for code in frame["Markush code"])
    metadata = pd.DataFrame(
        {
            "source_id": "lumi_lab_4cr1920",
            "measurement_id": [f"lumi_lab_4cr1920:{index}" for index in frame.index],
            "canonical_lipid_id": canonical,
            "structure_id": canonical,
            "structure_status": "full_smiles",
            "typed_head": "airway_hbe_in_vitro_expression",
            "group_id": "",
            "route": "in_vitro",
            "readout_type": "log2_RLU",
            "pooling_type": "individual",
        }
    )
    predictions: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for role in ("R1", "R2", "R3", "R4"):
        groups = np.asarray([component[role] for component in components])
        split_count = min(5, len(np.unique(groups)))
        splits = tuple(GroupKFold(n_splits=split_count).split(frame, target, groups))
        role_metadata = metadata.copy()
        role_metadata["group_id"] = groups
        split_name = f"held_component_{role}"
        role_rows, report = predict_array_splits(
            features=features,
            target=target,
            splits=splits,
            metadata=role_metadata,
            split_name=split_name,
            representation=FULL_REPRESENTATION,
            estimator=extra_trees(seed, trees),
        )
        predictions.extend(role_rows)
        reports.append(
            {
                "source_id": "lumi_lab_4cr1920",
                "typed_head": "airway_hbe_in_vitro_expression",
                **report,
            }
        )
    return predictions, reports


def lut_predictions(
    path: Path, seed: int, trees: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = pd.read_csv(path)
    if len(frame) != 444 or frame["compound_id"].nunique() != 444:
        raise ValueError("LuT table does not contain 444 unique component IDs")
    targets = {
        "systemic_iv_lung_expression": frame[
            "lung_expression_log10_photons_per_second"
        ].to_numpy(dtype=np.float64),
        "systemic_iv_lung_selectivity": frame["lung_selectivity_fraction"].to_numpy(
            dtype=np.float64
        ),
    }
    splitters = {
        "held_head": tuple(GroupKFold(n_splits=5).split(frame, groups=frame["head"])),
        "held_tail": tuple(GroupKFold(n_splits=5).split(frame, groups=frame["tail"])),
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
            "typed_head": "",
            "group_id": "",
            "route": "intravenous",
            "readout_type": "",
            "pooling_type": "individual",
        }
    )
    predictions: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    preprocessor = make_preprocessor(frame, "combined")
    estimator = make_pipeline(preprocessor, extra_trees(seed, trees))
    for head_id, target in targets.items():
        for split_name, splits in splitters.items():
            split_metadata = metadata.copy()
            split_metadata["typed_head"] = head_id
            split_metadata["readout_type"] = head_id
            if split_name == "held_head":
                split_metadata["group_id"] = frame["head"].to_numpy()
            elif split_name == "held_tail":
                split_metadata["group_id"] = frame["tail"].to_numpy()
            else:
                split_metadata["group_id"] = frame["round"].astype(str).to_numpy()
            head_rows, report = predict_frame_splits(
                frame=frame,
                target=target,
                splits=splits,
                metadata=split_metadata,
                split_name=split_name,
                representation=LUT_REPRESENTATION,
                estimator=estimator,
            )
            predictions.extend(head_rows)
            reports.append({"source_id": "lut_444_2026", "typed_head": head_id, **report})
    return predictions, reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "configs/pan_lung_corpus_manifest_v1.json",
    )
    parser.add_argument(
        "--lnpdb",
        type=Path,
        default=None,
    )
    parser.add_argument("--lumi", type=Path, default=None)
    parser.add_argument("--lut", type=Path, default=None)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO_ROOT / "diagnostics/pan_lung_matrix_R1M2_R2M2_predictions.csv",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=REPO_ROOT / "diagnostics/pan_lung_matrix_R1M2_R2M2_metrics.json",
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    lnpdb_source = manifest_source(manifest, "lnpdb_2026")
    lumi_source = manifest_source(manifest, "lumi_lab_4cr1920")
    lut_source = manifest_source(manifest, "lut_444_2026")
    lut_table = next(
        value for value in lut_source["local_paths"] if str(value).endswith(".csv")
    )
    source_paths = {
        "lnpdb": resolve_path(args.lnpdb or Path(lnpdb_source["local_path"])),
        "lumi": resolve_path(args.lumi or Path(lumi_source["local_path"])),
        "lut": resolve_path(args.lut or Path(lut_table)),
    }
    expected_hashes = {
        "lnpdb": lnpdb_source["sha256"],
        "lumi": lumi_source["sha256"],
        "lut": lut_source["sha256"]["extracted_table"],
    }
    source_hashes = {name: file_sha256(path) for name, path in source_paths.items()}
    mismatches = {
        name: {"expected": expected_hashes[name], "observed": observed}
        for name, observed in source_hashes.items()
        if observed != expected_hashes[name]
    }
    if mismatches:
        raise ValueError(f"source artifacts do not match frozen manifest: {mismatches}")

    args.predictions = resolve_path(args.predictions)
    args.metrics = resolve_path(args.metrics)

    source_predictions: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for loader, path in (
        (lnpdb_predictions, source_paths["lnpdb"]),
        (lumi_predictions, source_paths["lumi"]),
        (lut_predictions, source_paths["lut"]),
    ):
        predictions, source_reports = loader(path, args.seed, args.trees)
        source_predictions.extend(predictions)
        reports.extend(source_reports)

    predictions_frame = pd.DataFrame.from_records(source_predictions)
    key = ["source_id", "measurement_id", "typed_head", "split_name"]
    if predictions_frame.duplicated(key).any():
        raise RuntimeError("prediction table contains duplicate measurement/split keys")
    predictions_frame = predictions_frame.sort_values(key).reset_index(drop=True)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_csv(args.predictions, index=False)

    output = {
        "format": "compose_pan_lung_matrix_cell_metrics_v1",
        "cells": ["R1xM2", "R2xM2"],
        "seed": int(args.seed),
        "trees": int(args.trees),
        "prediction_decimals": PREDICTION_DECIMALS,
        "prediction_rows": int(len(predictions_frame)),
        "source_counts": {
            key: int(value)
            for key, value in predictions_frame.groupby("source_id").size().items()
        },
        "typed_head_count": int(predictions_frame["typed_head"].nunique()),
        "manifest": artifact_label(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "prediction_table": artifact_label(args.predictions),
        "prediction_table_sha256": file_sha256(args.predictions),
        "source_artifacts": {
            name: {"path": artifact_label(path), "sha256": source_hashes[name]}
            for name, path in source_paths.items()
        },
        "reports": reports,
        "interpretation_boundary": (
            "R1xM2 uses full molecular fingerprints/descriptors under typed task heads. "
            "R2xM2 is the explicit LuT component/SAR branch because full LuT product "
            "SMILES are unavailable. Metrics are hard-split diagnostics, not a frozen "
            "prospective steering oracle."
        ),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "prediction_rows": output["prediction_rows"],
                "typed_head_count": output["typed_head_count"],
                "source_counts": output["source_counts"],
                "predictions": artifact_label(args.predictions),
                "metrics": artifact_label(args.metrics),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
