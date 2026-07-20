#!/usr/bin/env python3
"""Build leakage-resistant conservative ensembles for qualified lung heads.

Calibration is cross-fitted over the already frozen held-group folds.  Every
test row is calibrated using only other folds from the same typed head and
novelty split.  The pessimistic score subtracts both a held-fold conformal
residual radius and per-row calibrated model disagreement from the ensemble
mean.  Scores remain in assay-native units and must never be compared across
typed heads.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.export_pan_lung_matrix_cell_predictions import (
    REPO_ROOT,
    artifact_label,
    file_sha256,
    resolve_path,
)
from scripts.export_pan_lung_qualified_matrix_comparison import regression_metrics


ALPHA = 0.10
QUALIFICATION_MIN_SPEARMAN = 0.40
QUALIFICATION_MIN_COVERAGE = 0.80
QUALIFICATION_SLOPE_RANGE = (0.50, 1.50)


@dataclass(frozen=True)
class EnsemblePolicy:
    source_id: str
    typed_head: str
    split_name: str
    member_models: tuple[str, ...]
    representation: str
    deployment_scope: str

    @property
    def ensemble_id(self) -> str:
        return f"{self.source_id}__{self.typed_head}__{self.split_name}"


POLICIES = (
    EnsemblePolicy(
        source_id="lnpdb_2026",
        typed_head="airway_a549_in_vitro_expression",
        split_name="held_lipid",
        member_models=("ridge_v1", "extra_trees_v1", "xgboost_v1"),
        representation="morgan_descriptors_plus_typed_context_v1",
        deployment_scope="A549 assay domain; novel full ionizable-lipid structures",
    ),
    *(
        EnsemblePolicy(
            source_id="lumi_lab_4cr1920",
            typed_head="airway_hbe_in_vitro_expression",
            split_name=f"held_component_{role}",
            member_models=("extra_trees_v1", "xgboost_v1"),
            representation="morgan_descriptors_plus_typed_context_v1",
            deployment_scope=f"LUMI 4CR chemistry with novelty assessed along {role}",
        )
        for role in ("R1", "R2", "R3", "R4")
    ),
    EnsemblePolicy(
        source_id="lut_444_2026",
        typed_head="systemic_iv_lung_selectivity",
        split_name="held_head",
        member_models=("ridge_v1", "extra_trees_v1", "xgboost_v1"),
        representation="lut_component_sar_plus_typed_context_v1",
        deployment_scope="LuT released-component domain, evaluated under held-head generalization",
    ),
    EnsemblePolicy(
        source_id="lut_444_2026",
        typed_head="systemic_iv_lung_selectivity",
        split_name="held_tail",
        member_models=("ridge_v1", "extra_trees_v1", "xgboost_v1"),
        representation="lut_component_sar_plus_typed_context_v1",
        deployment_scope="LuT released-component domain, evaluated under held-tail generalization",
    ),
)


def affine_fit(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    design = np.column_stack((np.ones(len(prediction)), prediction))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def conformal_radius(residuals: np.ndarray, alpha: float = ALPHA) -> float:
    if len(residuals) == 0:
        raise ValueError("cannot estimate a conformal radius without residuals")
    rank = min(len(residuals), int(np.ceil((len(residuals) + 1) * (1.0 - alpha))))
    return float(np.partition(np.asarray(residuals), rank - 1)[rank - 1])


def policy_frame(predictions: pd.DataFrame, policy: EnsemblePolicy) -> pd.DataFrame:
    selected = predictions[
        (predictions["source_id"] == policy.source_id)
        & (predictions["typed_head"] == policy.typed_head)
        & (predictions["split_name"] == policy.split_name)
        & predictions["model"].isin(policy.member_models)
    ].copy()
    if selected.empty:
        raise ValueError(f"no predictions found for {policy.ensemble_id}")
    identity = [
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
    ]
    for column in identity:
        if selected.groupby("measurement_id")[column].nunique(dropna=False).max() != 1:
            raise RuntimeError(
                f"models disagree on frozen metadata {column!r} for {policy.ensemble_id}"
            )
    raw = selected.pivot(index="measurement_id", columns="model", values="prediction")
    missing = set(policy.member_models) - set(raw.columns)
    if missing or raw.isna().any().any():
        raise RuntimeError(f"missing ensemble members {sorted(missing)} for {policy.ensemble_id}")
    metadata = selected.drop_duplicates("measurement_id").set_index("measurement_id")
    frame = metadata[identity[0:1] + identity[2:]].copy()
    for model in policy.member_models:
        frame[f"raw_{model}"] = raw[model]
    return frame.reset_index()


def cross_fit_policy(
    predictions: pd.DataFrame, policy: EnsemblePolicy
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    frame = policy_frame(predictions, policy)
    target = frame["target_value"].to_numpy(dtype=np.float64)
    folds = frame["fold"].to_numpy(dtype=int)
    calibrated = {
        model: np.full(len(frame), np.nan, dtype=np.float64)
        for model in policy.member_models
    }
    radius = np.full(len(frame), np.nan, dtype=np.float64)
    fold_calibrators: list[dict[str, object]] = []

    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        if not train.any() or not test.any():
            raise RuntimeError(f"invalid calibration fold {fold} for {policy.ensemble_id}")
        training_members: list[np.ndarray] = []
        fold_models: dict[str, dict[str, float]] = {}
        for model in policy.member_models:
            raw = frame[f"raw_{model}"].to_numpy(dtype=np.float64)
            intercept, slope = affine_fit(raw[train], target[train])
            calibrated[model][test] = intercept + slope * raw[test]
            training_members.append(intercept + slope * raw[train])
            fold_models[model] = {"intercept": intercept, "slope": slope}
        training_mean = np.mean(np.column_stack(training_members), axis=1)
        fold_radius = conformal_radius(np.abs(target[train] - training_mean))
        radius[test] = fold_radius
        fold_calibrators.append(
            {
                "test_fold": int(fold),
                "calibration_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "models": fold_models,
                "conformal_absolute_residual_quantile": 1.0 - ALPHA,
                "conformal_radius": fold_radius,
            }
        )

    if any(np.isnan(values).any() for values in calibrated.values()) or np.isnan(radius).any():
        raise RuntimeError(f"cross-fitting left missing values for {policy.ensemble_id}")
    calibrated_matrix = np.column_stack(
        tuple(calibrated[model] for model in policy.member_models)
    )
    ensemble_mean = np.mean(calibrated_matrix, axis=1)
    model_disagreement = np.std(calibrated_matrix, axis=1, ddof=0)
    total_uncertainty = radius + model_disagreement
    pessimistic_score = ensemble_mean - total_uncertainty
    conformal_covered = np.abs(target - ensemble_mean) <= radius
    conservative_covered = np.abs(target - ensemble_mean) <= total_uncertainty

    output = frame.copy()
    output.insert(0, "ensemble_id", policy.ensemble_id)
    for model in policy.member_models:
        output[f"calibrated_{model}"] = calibrated[model]
    output["ensemble_mean"] = ensemble_mean
    output["model_disagreement"] = model_disagreement
    output["held_fold_conformal_radius"] = radius
    output["total_uncertainty"] = total_uncertainty
    output["pessimistic_score"] = pessimistic_score
    output["conformal_covered"] = conformal_covered
    output["conservative_covered"] = conservative_covered

    mean_metrics = regression_metrics(target, ensemble_mean)
    pessimistic_metrics = regression_metrics(target, pessimistic_score)
    slope = mean_metrics["calibration_slope_target_on_prediction"]
    qualified = (
        mean_metrics["spearman"] is not None
        and mean_metrics["spearman"] >= QUALIFICATION_MIN_SPEARMAN
        and float(np.mean(conservative_covered)) >= QUALIFICATION_MIN_COVERAGE
        and slope is not None
        and QUALIFICATION_SLOPE_RANGE[0] <= slope <= QUALIFICATION_SLOPE_RANGE[1]
    )
    qualification = {
        "ensemble_id": policy.ensemble_id,
        "source_id": policy.source_id,
        "typed_head": policy.typed_head,
        "split_name": policy.split_name,
        "deployment_scope": policy.deployment_scope,
        "representation": policy.representation,
        "member_models": list(policy.member_models),
        "rows": int(len(frame)),
        "folds": int(len(np.unique(folds))),
        "qualification_status": (
            "qualified_for_domain_bounded_filtering" if qualified else "diagnostic_only"
        ),
        "mean_spearman": mean_metrics["spearman"],
        "mean_top_decile_recall": mean_metrics["top_decile_recall"],
        "mean_top_decile_enrichment": mean_metrics["top_decile_enrichment"],
        "mean_mae": mean_metrics["mae"],
        "mean_r2": mean_metrics["r2"],
        "calibration_bias": mean_metrics["mean_prediction_bias"],
        "calibration_slope": slope,
        "pessimistic_spearman": pessimistic_metrics["spearman"],
        "pessimistic_top_decile_recall": pessimistic_metrics["top_decile_recall"],
        "pessimistic_top_decile_enrichment": pessimistic_metrics[
            "top_decile_enrichment"
        ],
        "mean_model_disagreement": float(np.mean(model_disagreement)),
        "mean_total_uncertainty": float(np.mean(total_uncertainty)),
        "cross_fitted_conformal_coverage": float(np.mean(conformal_covered)),
        "cross_fitted_conservative_coverage": float(np.mean(conservative_covered)),
    }

    final_models: dict[str, dict[str, float]] = {}
    final_calibrated_members: list[np.ndarray] = []
    for model in policy.member_models:
        raw = frame[f"raw_{model}"].to_numpy(dtype=np.float64)
        intercept, model_slope = affine_fit(raw, target)
        final_models[model] = {"intercept": intercept, "slope": model_slope}
        final_calibrated_members.append(intercept + model_slope * raw)
    final_mean = np.mean(np.column_stack(final_calibrated_members), axis=1)
    deployment = {
        "ensemble_id": policy.ensemble_id,
        "representation": policy.representation,
        "member_models": list(policy.member_models),
        "uniform_member_weight": float(1.0 / len(policy.member_models)),
        "final_oof_affine_calibrators": final_models,
        "final_oof_conformal_radius": conformal_radius(np.abs(target - final_mean)),
        "pessimistic_score_formula": (
            "mean(calibrated_member_predictions) - final_oof_conformal_radius "
            "- std(calibrated_member_predictions, ddof=0)"
        ),
        "fold_calibrators_for_audit": fold_calibrators,
    }
    return output, qualification, deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-predictions",
        type=Path,
        default=REPO_ROOT
        / "diagnostics/pan_lung_qualified_matrix_comparison_predictions.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO_ROOT / "diagnostics/pan_lung_qualified_ensemble_predictions.csv",
    )
    parser.add_argument(
        "--qualification-table",
        type=Path,
        default=REPO_ROOT / "diagnostics/pan_lung_qualified_ensemble_qualification.csv",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=REPO_ROOT / "diagnostics/pan_lung_qualified_ensemble_metrics.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO_ROOT / "configs/pan_lung_qualified_ensemble_contract_v1.json",
    )
    args = parser.parse_args()

    comparison_path = resolve_path(args.comparison_predictions)
    predictions = pd.read_csv(comparison_path)
    outputs: list[pd.DataFrame] = []
    qualifications: list[dict[str, object]] = []
    deployments: list[dict[str, object]] = []
    for policy in POLICIES:
        output, qualification, deployment = cross_fit_policy(predictions, policy)
        outputs.append(output)
        qualifications.append(qualification)
        deployments.append(deployment)

    ensemble_predictions = pd.concat(outputs, ignore_index=True, sort=False)
    ensemble_predictions = ensemble_predictions.sort_values(
        ["ensemble_id", "measurement_id"]
    ).reset_index(drop=True)
    qualification = pd.DataFrame.from_records(qualifications).sort_values("ensemble_id")
    args.predictions = resolve_path(args.predictions)
    args.qualification_table = resolve_path(args.qualification_table)
    args.metrics = resolve_path(args.metrics)
    args.contract = resolve_path(args.contract)
    for path in (args.predictions, args.qualification_table, args.metrics, args.contract):
        path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_predictions.to_csv(args.predictions, index=False)
    qualification.to_csv(args.qualification_table, index=False)

    metrics_output = {
        "format": "compose_pan_lung_qualified_ensemble_metrics_v1",
        "comparison_predictions": artifact_label(comparison_path),
        "comparison_predictions_sha256": file_sha256(comparison_path),
        "prediction_rows": int(len(ensemble_predictions)),
        "ensemble_count": int(len(qualification)),
        "qualification_counts": {
            key: int(value)
            for key, value in qualification.groupby("qualification_status").size().items()
        },
        "cross_fitted_predictions": artifact_label(args.predictions),
        "cross_fitted_predictions_sha256": file_sha256(args.predictions),
        "qualification_table": artifact_label(args.qualification_table),
        "qualification_table_sha256": file_sha256(args.qualification_table),
        "qualification_rules": {
            "minimum_spearman": QUALIFICATION_MIN_SPEARMAN,
            "minimum_conservative_coverage": QUALIFICATION_MIN_COVERAGE,
            "calibration_slope_range": list(QUALIFICATION_SLOPE_RANGE),
            "prohibited_splits": ["random", "round1_to_round2"],
        },
        "reports": qualifications,
    }
    args.metrics.write_text(json.dumps(metrics_output, indent=2, sort_keys=True) + "\n")

    contract = {
        "format": "compose_pan_lung_qualified_ensemble_contract_v1",
        "status": "filtering_bundle_verified_reward_fine_tuning_disabled",
        "purpose": "Domain-bounded filtering and reward construction for later COMPOSE-Lipid experiments.",
        "filtering_enabled": True,
        "reward_fine_tuning_enabled": False,
        "filtering_bundle_manifest": "artifacts/oracles/pan_lung_filtering_v1/manifest.json",
        "clean_runtime_verification": "diagnostics/pan_lung_filtering_clean_runtime_verification.json",
        "reward_guard": "src/compose_v4/oracles/reward_guard.py",
        "preregistration_template": "configs/pan_lung_reward_preregistration.template.json",
        "direction": "higher_is_better_within_each_typed_head_only",
        "source_comparison_predictions": artifact_label(comparison_path),
        "source_comparison_predictions_sha256": file_sha256(comparison_path),
        "qualification_table": artifact_label(args.qualification_table),
        "qualification_table_sha256": file_sha256(args.qualification_table),
        "cross_fitted_audit_predictions": artifact_label(args.predictions),
        "cross_fitted_audit_predictions_sha256": file_sha256(args.predictions),
        "qualified_ensemble_ids": qualification.loc[
            qualification["qualification_status"]
            == "qualified_for_domain_bounded_filtering",
            "ensemble_id",
        ].tolist(),
        "diagnostic_only_ensemble_ids": qualification.loc[
            qualification["qualification_status"] == "diagnostic_only",
            "ensemble_id",
        ].tolist(),
        "uncertainty_contract": {
            "calibration": "Affine calibration is fitted on held-group OOF predictions. Cross-fitted audit rows never use their fold labels to fit calibration.",
            "conformal_radius": "Finite-sample 90% absolute-residual quantile estimated from other held folds; final deployment radius uses all OOF residuals after qualification.",
            "model_disagreement": "Population standard deviation across calibrated member predictions; a disagreement proxy, not a posterior standard deviation.",
            "pessimistic_score": "calibrated ensemble mean minus conformal radius minus model disagreement",
        },
        "deployment_policies": deployments,
        "use_rules": [
            "Never compare or add raw scores across typed heads.",
            "Use only policies whose qualification_status is qualified_for_domain_bounded_filtering.",
            "For LUMI, select the policy matching the claimed novel component axis; if multiple axes are novel, take the minimum pessimistic score across those axes.",
            "For LuT, admit only released head and tail components; for new combinations of known components, take the minimum pessimistic score from held_head and held_tail policies.",
            "Round-transfer is diagnostic only and is prohibited as a reward or qualification source.",
            "Filtering may rank in-domain candidates by pessimistic score; reward fine-tuning additionally requires an applicability-domain gate and serialized full-data base estimators.",
        ],
        "excluded_heads": {
            "airway_hbec_ali_in_vitro_expression": "insufficient held-lipid evidence",
            "local_intratracheal_functional_expression": "insufficient held-lipid evidence",
            "systemic_iv_barcoded_lung_uptake": "incompatible uptake readout and weak rank signal",
            "systemic_iv_lung_expression": "not qualified in this selectivity-focused gate",
        },
        "hard_blockers_before_reward_fine_tuning": [
            "Require the present hash-verified bundle and passed clean-runtime verification artifact.",
            "Keep the hard reward guard in the optimization path so rejected candidates can never receive a reward.",
            "Create an explicit frozen preregistration bound to the exact bundle-manifest hash and separately authorize reward fine-tuning in the bundle.",
            "Pre-register score thresholds and candidate budgets without using prospective outcomes.",
            "Monitor score hacking and applicability-domain drift during any optimization run.",
        ],
    }
    args.contract.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "prediction_rows": int(len(ensemble_predictions)),
                "ensemble_count": int(len(qualification)),
                "qualification_counts": metrics_output["qualification_counts"],
                "predictions": artifact_label(args.predictions),
                "qualification_table": artifact_label(args.qualification_table),
                "metrics": artifact_label(args.metrics),
                "contract": artifact_label(args.contract),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
