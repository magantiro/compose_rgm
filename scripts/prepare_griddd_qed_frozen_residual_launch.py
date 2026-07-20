#!/usr/bin/env python3
"""Validate the frozen-residual QED launch contract without launching it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.experiments.griddd_conditional import (
    ANALYTIC_BACKBONE_QUALIFICATION_FORMAT,
    RETAINED_PANCAKE_CHECKPOINT_SHA256,
    file_sha256,
    load_unconditional_backbone_qualification,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/griddd_qed_frozen_residual_launch_v1.json"
DEFAULT_OUTPUT = ROOT / "diagnostics/griddd_qed_frozen_residual_launch_readiness.json"


def _resolve(value: object) -> Path | None:
    if value is None or not str(value):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _block(
    blockers: list[dict[str, str]],
    code: str,
    detail: str,
) -> None:
    blockers.append({"code": code, "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = _load_object(args.config)
    if config.get("format") != "compose_v4_griddd_qed_frozen_residual_launch_v1":
        raise ValueError("unsupported frozen-residual launch config")
    backbone = config.get("backbone")
    adapter = config.get("conditional_adapter")
    training = config.get("training")
    protocol_a = config.get("protocol_a")
    protocol_b = config.get("protocol_b")
    ring = config.get("ring_sensitivity")
    if not all(
        isinstance(value, dict)
        for value in (backbone, adapter, training, protocol_a, protocol_b, ring)
    ):
        raise ValueError("launch config is missing a required section")
    assert isinstance(backbone, dict)
    assert isinstance(adapter, dict)
    assert isinstance(training, dict)
    assert isinstance(protocol_a, dict)
    assert isinstance(protocol_b, dict)
    assert isinstance(ring, dict)

    contract_errors: list[dict[str, str]] = []
    if backbone.get("retained_checkpoint_sha256") != RETAINED_PANCAKE_CHECKPOINT_SHA256:
        _block(
            contract_errors,
            "retained_checkpoint_identity_changed",
            "config does not name the retained step-6250 pancake SHA-256",
        )
    required_sidecar = {
        "kind": "frozen_valid_state_qed_residual_v1",
        "base_parameters_frozen": True,
        "base_weights_in_sidecar_state_dict": False,
        "family_hazard_residual": True,
        "within_family_residual": True,
        "independent_total_hazard_residual": False,
        "missing_condition_identity": True,
    }
    for key, expected in required_sidecar.items():
        if adapter.get(key) != expected:
            _block(
                contract_errors,
                "conditional_sidecar_contract_changed",
                f"conditional_adapter.{key} must equal {expected!r}",
            )
    if protocol_a.get("candidates_per_start") != 20:
        _block(contract_errors, "protocol_a_candidate_count", "Protocol A requires 20")
    if protocol_a.get("oracle_tilt_or_rescoring_during_sampling") is not False:
        _block(contract_errors, "protocol_a_oracle_tilt", "Protocol A must be native")
    if protocol_a.get("padding_calls") is not False:
        _block(contract_errors, "protocol_a_padding", "Protocol A prohibits padding")
    expected_protocol_b = 1 + int(protocol_b.get("candidates_per_start", 0)) * (
        int(protocol_b.get("guidance_oracle_calls_per_candidate", 0))
        + int(protocol_b.get("final_candidate_oracle_calls_per_candidate", 0))
    )
    if expected_protocol_b != 981 or protocol_b.get(
        "exact_oracle_calls_per_start_seed_arm"
    ) != 981:
        _block(
            contract_errors,
            "protocol_b_budget",
            "Protocol B must remain exactly 1 + 20 * (48 + 1) = 981 calls",
        )
    if ring.get("compare_high_qed_target_subset_to_unconditional_corpus") is not True:
        _block(
            contract_errors,
            "ring_reference_boundary",
            "ring diagnostic must compare the high-QED subset to the unconditional corpus",
        )
    if ring.get("global_fused_frequency_is_target_marginal") is not False:
        _block(
            contract_errors,
            "ring_target_marginal",
            "global fused frequency cannot be declared the conditional target marginal",
        )

    training_blockers = list(contract_errors)
    qualification_path = _resolve(backbone.get("qualification_manifest"))
    qualification = None
    if qualification_path is None or not qualification_path.is_file():
        _block(
            training_blockers,
            "analytic_qualification_manifest_missing",
            "truthful passed analytic-adapter qualification manifest is absent",
        )
    else:
        raw_qualification = _load_object(qualification_path)
        if raw_qualification.get("format") != ANALYTIC_BACKBONE_QUALIFICATION_FORMAT:
            _block(
                training_blockers,
                "analytic_qualification_wrong_format",
                "qualification does not use the analytic-adapter handoff schema",
            )
        try:
            qualification = load_unconditional_backbone_qualification(
                qualification_path
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            _block(
                training_blockers,
                "analytic_qualification_not_passed",
                str(error),
            )
        if qualification is not None:
            adapter_source = _resolve(qualification.execution_adapter_source_path)
            if adapter_source is None or not adapter_source.is_file():
                _block(
                    training_blockers,
                    "qualified_adapter_source_missing",
                    "qualification-bound analytic adapter source is unavailable",
                )
            elif file_sha256(adapter_source) != qualification.execution_adapter_source_sha256:
                _block(
                    training_blockers,
                    "qualified_adapter_source_hash_mismatch",
                    "analytic adapter source changed after qualification",
                )

    retained_checkpoint = _resolve(backbone.get("retained_checkpoint_path"))
    if retained_checkpoint is None:
        _block(
            training_blockers,
            "retained_checkpoint_path_unresolved",
            "the passed rollout did not record the retained checkpoint path",
        )
    elif not retained_checkpoint.is_file():
        _block(
            training_blockers,
            "retained_checkpoint_missing",
            str(retained_checkpoint),
        )
    elif file_sha256(retained_checkpoint) != RETAINED_PANCAKE_CHECKPOINT_SHA256:
        _block(
            training_blockers,
            "retained_checkpoint_hash_mismatch",
            "checkpoint path does not resolve to the retained pancake SHA-256",
        )

    rollout_path = _resolve(backbone.get("analytic_rollout_evidence"))
    if rollout_path is None or not rollout_path.is_file():
        _block(training_blockers, "analytic_rollout_evidence_missing", "rollout gate absent")
    else:
        rollout = _load_object(rollout_path)
        if rollout.get("format") != "compose_v4_analytic_pancake_quotient_rollout_smoke_v1":
            _block(training_blockers, "analytic_rollout_wrong_format", "wrong rollout schema")
        if rollout.get("passed") is not True:
            _block(training_blockers, "analytic_rollout_not_passed", "rollout gate failed")
        profile = rollout.get("profile", {})
        if not isinstance(profile, dict) or (
            profile.get("atom_delete_log_rate_adjustment") != -0.5
            or profile.get("small_ring_log_rate_adjustment") != -1.5
        ):
            _block(
                training_blockers,
                "analytic_rollout_calibration_mismatch",
                "rollout evidence does not match the frozen calibration",
            )

    training_manifest = _resolve(training.get("training_partition_manifest"))
    if training_manifest is None or not training_manifest.is_file():
        _block(
            training_blockers,
            "training_partition_manifest_unresolved",
            "sidecar training examples and their QED normalizer are not hash-bound",
        )
    elif file_sha256(training_manifest) != str(
        training.get("training_partition_manifest_sha256", "")
    ):
        _block(
            training_blockers,
            "training_partition_manifest_hash_mismatch",
            "training partition manifest does not match its frozen SHA-256",
        )
    normalizer = adapter.get("qed_normalizer", {})
    if not isinstance(normalizer, dict) or normalizer.get("mean") is None or normalizer.get(
        "standard_deviation"
    ) is None:
        _block(
            training_blockers,
            "qed_normalizer_not_frozen",
            "training-partition QED mean and standard deviation remain unresolved",
        )
    else:
        normalizer_evidence = _resolve(normalizer.get("evidence_path"))
        if normalizer_evidence is None or not normalizer_evidence.is_file():
            _block(
                training_blockers,
                "qed_normalizer_evidence_missing",
                "frozen QED normalizer evidence artifact is absent",
            )
        elif file_sha256(normalizer_evidence) != str(
            normalizer.get("evidence_sha256", "")
        ):
            _block(
                training_blockers,
                "qed_normalizer_evidence_hash_mismatch",
                "QED normalizer evidence does not match its frozen SHA-256",
            )

    protocol_a_blockers = list(training_blockers)
    lead_manifest_path = _resolve(protocol_a.get("exact_official_lead_manifest"))
    if lead_manifest_path is None or not lead_manifest_path.is_file():
        _block(
            protocol_a_blockers,
            "griddd_lead_manifest_missing",
            "benchmark provenance manifest is absent",
        )
    else:
        lead_manifest = _load_object(lead_manifest_path)
        decision = lead_manifest.get("decision", {})
        if not isinstance(decision, dict) or decision.get(
            "griddd_release_exact_list_available"
        ) is not True:
            _block(
                protocol_a_blockers,
                "official_griddd_800_unavailable",
                "the exact GrIDDD release lead list is still unresolved; Jin800 and the local reconstruction are not substitutes",
            )

    ring_reference = _resolve(ring.get("reference_comparison_artifact"))
    for blockers in (protocol_a_blockers,):
        if ring_reference is None or not ring_reference.is_file():
            _block(
                blockers,
                "qed_ring_reference_comparison_missing",
                "high-QED target-subset versus unconditional-corpus ring comparison is absent",
            )
        else:
            reference = _load_object(ring_reference)
            if reference.get("format") != ring.get("reference_comparison_format"):
                _block(
                    blockers,
                    "qed_ring_reference_comparison_wrong_format",
                    "ring comparison artifact has the wrong schema",
                )

    protocol_b_blockers = list(training_blockers)
    output = {
        "format": "compose_v4_griddd_qed_frozen_residual_launch_readiness_v1",
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "launched": False,
        "sidecar_training_ready": not training_blockers,
        "protocol_a_ready": not protocol_a_blockers,
        "protocol_b_ready": not protocol_b_blockers,
        "training_blockers": training_blockers,
        "protocol_a_blockers": protocol_a_blockers,
        "protocol_b_blockers": protocol_b_blockers,
        "verified_contract": {
            "frozen_base_residual_sidecar": not contract_errors,
            "protocol_a_native_20_candidates": (
                protocol_a.get("candidates_per_start") == 20
                and protocol_a.get("oracle_tilt_or_rescoring_during_sampling") is False
                and protocol_a.get("padding_calls") is False
            ),
            "protocol_b_exact_981_call_ledger": expected_protocol_b == 981,
            "ring_sensitivity_evaluator_required": ring.get("required") is True,
            "global_fused_frequency_not_assumed_target": (
                ring.get("global_fused_frequency_is_target_marginal") is False
            ),
        },
        "next_action": (
            "Resolve only the named artifacts; do not launch the legacy shared-encoder "
            "QED recipe or relabel Jin800/reconstruction as official GrIDDD800."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
