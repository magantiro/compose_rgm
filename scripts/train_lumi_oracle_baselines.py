#!/usr/bin/env python3
"""Reproducible first-pass baselines for the public LUMI-lab 4CR-1920 data.

Random folds are reported only as a diagnostic.  The decision-relevant tests
hold out complete Markush components (R1--R4), which directly measure transfer
to unseen heads/linkers/tails within this factorial reaction family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MARKUSH_PATTERN = re.compile(r"(R\d+)\(\d+\):(\d+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_markush(code: str) -> dict[str, str]:
    parsed = dict(MARKUSH_PATTERN.findall(str(code)))
    if set(parsed) != {"R1", "R2", "R3", "R4"}:
        raise ValueError(f"unexpected LUMI Markush code: {code!r}")
    return parsed


def descriptor_vector(molecule: Chem.Mol) -> np.ndarray:
    return np.asarray(
        (
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            Descriptors.TPSA(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            Lipinski.RingCount(molecule),
            Lipinski.FractionCSP3(molecule),
            Chem.GetFormalCharge(molecule),
            molecule.GetNumHeavyAtoms(),
        ),
        dtype=np.float32,
    )


def molecular_features(smiles: Iterable[str]) -> dict[str, np.ndarray]:
    molecules = tuple(Chem.MolFromSmiles(value) for value in smiles)
    if any(molecule is None for molecule in molecules):
        raise ValueError("LUMI baseline received an unparsable molecule")
    typed_molecules = tuple(molecule for molecule in molecules if molecule is not None)
    descriptors = np.stack(tuple(descriptor_vector(mol) for mol in typed_molecules))
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = np.zeros((len(typed_molecules), 2048), dtype=np.float32)
    for row, molecule in enumerate(typed_molecules):
        DataStructs.ConvertToNumpyArray(
            generator.GetFingerprint(molecule),
            fingerprints[row],
        )
    return {
        "descriptors": descriptors,
        "morgan": fingerprints,
        "combined": np.concatenate((descriptors, fingerprints), axis=1),
    }


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correlation = float(spearmanr(target, prediction).statistic)
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "r2": float(r2_score(target, prediction)),
        "spearman": correlation,
    }


def cross_validated_predictions(
    estimator: object,
    features: np.ndarray,
    target: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    prediction = np.full(target.shape, np.nan, dtype=np.float64)
    fold_reports: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splits):
        fitted = clone(estimator)
        fitted.fit(features[train], target[train])
        fold_prediction = np.asarray(fitted.predict(features[test]), dtype=np.float64)
        prediction[test] = fold_prediction
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                **regression_metrics(target[test], fold_prediction),
            }
        )
    if np.isnan(prediction).any():
        raise RuntimeError("cross-validation did not produce one prediction per row")
    return prediction, fold_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("tmp/lipid_data/lumi_lab/4CR-1920.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/lumi_lab_4cr1920_baselines.json"),
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--trees", type=int, default=200)
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    required = {"RLU (log2)", "mol", "Markush code"}
    if set(frame.columns) != required or len(frame) != 1920:
        raise ValueError("LUMI artifact does not match the frozen 1,920-row schema")
    if frame[list(required)].isna().any().any():
        raise ValueError("LUMI artifact contains missing required values")
    components = tuple(parse_markush(code) for code in frame["Markush code"])
    groups = {
        role: np.asarray([component[role] for component in components])
        for role in ("R1", "R2", "R3", "R4")
    }
    target = frame["RLU (log2)"].to_numpy(dtype=np.float64)
    representations = molecular_features(frame["mol"].astype(str))
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
    splitters: dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]] = {
        "random_diagnostic": tuple(
            KFold(n_splits=5, shuffle=True, random_state=args.seed).split(target)
        )
    }
    for role, role_groups in groups.items():
        fold_count = min(5, len(np.unique(role_groups)))
        splitters[f"held_component_{role}"] = tuple(
            GroupKFold(n_splits=fold_count).split(target, target, role_groups)
        )

    reports: list[dict[str, object]] = []
    for representation_name, features in representations.items():
        for estimator_name, estimator in estimators.items():
            for split_name, splits in splitters.items():
                prediction, folds = cross_validated_predictions(
                    estimator,
                    features,
                    target,
                    splits,
                )
                reports.append(
                    {
                        "representation": representation_name,
                        "model": estimator_name,
                        "split": split_name,
                        "folds": folds,
                        **regression_metrics(target, prediction),
                    }
                )

    output = {
        "format": "compose_lipid_lumi_baseline_v1",
        "artifact": str(args.data),
        "artifact_sha256": sha256(args.data),
        "rows": int(len(frame)),
        "unique_molecules": int(frame["mol"].nunique()),
        "unique_markush_codes": int(frame["Markush code"].nunique()),
        "component_cardinalities": {
            role: int(len(np.unique(values))) for role, values in groups.items()
        },
        "target_summary": {
            "mean": float(target.mean()),
            "standard_deviation": float(target.std(ddof=1)),
            "minimum": float(target.min()),
            "maximum": float(target.max()),
        },
        "reports": reports,
        "interpretation_boundary": (
            "Random folds are diagnostic. Held-component results measure "
            "within-4CR transfer and are not evidence of in-vivo lung prediction."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    best_by_split: dict[str, dict[str, object]] = {}
    for report in reports:
        split = str(report["split"])
        if split not in best_by_split or float(report["spearman"]) > float(
            best_by_split[split]["spearman"]
        ):
            best_by_split[split] = report
    print(
        json.dumps(
            {
                split: {
                    key: report[key]
                    for key in ("representation", "model", "mae", "rmse", "r2", "spearman")
                }
                for split, report in sorted(best_by_split.items())
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
