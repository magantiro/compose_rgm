#!/usr/bin/env python3
"""Build a hash-bound analytic-quotient backbone qualification manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import empty_molecular_graph, pad_molecular_graph
from compose_v4.experiments.canonical_successor_distillation import (
    AnalyticPancakeQuotientSampler,
    build_calibrated_pancake_quotient_target,
)
from compose_v4.experiments.griddd_conditional import (
    ANALYTIC_BACKBONE_QUALIFICATION_FORMAT,
    RETAINED_PANCAKE_CHECKPOINT_SHA256,
    file_sha256,
)

if __package__:
    from scripts.evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
else:
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("/private/tmp/pancake_checkpoint/checkpoint.recovery.pt")
DEFAULT_PANEL = ROOT / "artifacts/canonical_successor_backbone_qualification/panel_manifest.json"
DEFAULT_ROLLOUT = ROOT / "diagnostics/canonical_successor_analytic_rollout_smoke20.json"
DEFAULT_RATE_EVIDENCE = ROOT / "diagnostics/canonical_successor_analytic_rate_equivalence_panel.json"
DEFAULT_QUALIFICATION = ROOT / "diagnostics/canonical_successor_analytic_backbone_qualification.json"
ANALYTIC_SOURCE = ROOT / "src/compose_v4/experiments/canonical_successor_distillation.py"


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = right.abs().clamp_min(1e-12)
    return float(((left - right).abs() / denominator).max()) if right.numel() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--rollout", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--rate-evidence", type=Path, default=DEFAULT_RATE_EVIDENCE)
    parser.add_argument("--qualification", type=Path, default=DEFAULT_QUALIFICATION)
    args = parser.parse_args()

    checkpoint_sha256 = file_sha256(args.checkpoint)
    if checkpoint_sha256 != RETAINED_PANCAKE_CHECKPOINT_SHA256:
        raise ValueError("checkpoint is not the retained step-6250 pancake")
    panel = _object(args.panel)
    if panel.get("format") != "compose_v4_pancake_quotient_heldout_panel_v1":
        raise ValueError("unsupported analytic rate panel")
    if panel.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("panel does not bind the retained checkpoint")
    rows = panel.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("multi-state panel is missing or too small")
    rollout = _object(args.rollout)
    if rollout.get("format") != "compose_v4_analytic_pancake_quotient_rollout_smoke_v1":
        raise ValueError("unsupported analytic rollout gate")

    model, checkpoint_payload = load_factorized_rollout_checkpoint(args.checkpoint)
    if int(model.property_condition_dim) != 0:
        raise ValueError("qualified analytic base must be unconditional")
    if model.empirical_mark_prior_mode != "none":
        raise ValueError("qualified analytic base imports prohibited P1 priors")
    if model.ring_family_mass_mode != "boolean":
        raise ValueError("qualified analytic base imports prohibited P2 mass")
    model.eval()
    sampler = AnalyticPancakeQuotientSampler(model)
    evidence_rows = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("panel row must be an object")
        source_key = str(row["source_key"])
        state = (
            empty_molecular_graph(40)
            if source_key == "<NULL>"
            else pad_molecular_graph(smiles_to_molecular_graph(source_key), 40)
        )
        model_time = float(row["model_time"])
        analytic = sampler.rate_table(state, model_time)
        reference = build_calibrated_pancake_quotient_target(
            model,
            state,
            model_time,
        )
        if analytic.graft_successor_keys != reference.graft_successor_keys:
            raise RuntimeError("analytic/reference Graft successor supports differ")
        family_abs = float(
            (analytic.productive_family_rates - reference.productive_family_rates)
            .abs()
            .max()
        )
        graft_abs = (
            float(
                (analytic.graft_successor_rates - reference.graft_successor_rates)
                .abs()
                .max()
            )
            if len(reference.graft_successor_rates)
            else 0.0
        )
        total_abs = abs(
            float(analytic.productive_total_hazard)
            - float(reference.productive_total_hazard)
        )
        family_relative = _relative_error(
            analytic.productive_family_rates,
            reference.productive_family_rates,
        )
        graft_relative = _relative_error(
            analytic.graft_successor_rates,
            reference.graft_successor_rates,
        )
        passed = bool(
            family_abs <= 1e-6
            and graft_abs <= 1e-6
            and total_abs <= 1e-6
            and family_relative <= 1e-5
            and graft_relative <= 1e-5
        )
        evidence_rows.append(
            {
                "row_id": str(row["row_id"]),
                "split": str(row["split"]),
                "stratum": str(row["stratum"]),
                "source_key": source_key,
                "model_time": model_time,
                "family_max_absolute_error": family_abs,
                "family_max_relative_error": family_relative,
                "graft_max_absolute_error": graft_abs,
                "graft_max_relative_error": graft_relative,
                "total_hazard_absolute_error": total_abs,
                "graft_successor_count": len(reference.graft_successor_keys),
                "passed": passed,
            }
        )

    heldout_rows = [row for row in evidence_rows if row["split"] == "heldout"]
    panel_passed = all(bool(row["passed"]) for row in evidence_rows)
    heldout_passed = bool(heldout_rows) and all(
        bool(row["passed"]) for row in heldout_rows
    )
    rate_evidence = {
        "format": "compose_v4_analytic_pancake_quotient_rate_equivalence_panel_v1",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha256,
        },
        "panel_manifest": {
            "path": str(args.panel),
            "sha256": file_sha256(args.panel),
            "state_count": len(evidence_rows),
            "heldout_state_count": len(heldout_rows),
        },
        "analytic_source": {
            "path": str(ANALYTIC_SOURCE.relative_to(ROOT)),
            "sha256": file_sha256(ANALYTIC_SOURCE),
        },
        "thresholds": {
            "max_absolute_error": 1e-6,
            "max_relative_error": 1e-5,
        },
        "panel_passed": panel_passed,
        "heldout_passed": heldout_passed,
        "rows": evidence_rows,
    }
    _write(args.rate_evidence, rate_evidence)

    rollout_profile = rollout.get("profile")
    candidate = rollout.get("candidate")
    baseline = rollout.get("baseline")
    if not all(isinstance(value, dict) for value in (rollout_profile, candidate, baseline)):
        raise ValueError("rollout gate lacks its matched 20+20 evidence")
    assert isinstance(rollout_profile, dict)
    assert isinstance(candidate, dict)
    assert isinstance(baseline, dict)
    candidate_metrics = candidate.get("metrics", {})
    baseline_metrics = baseline.get("metrics", {})
    rollout_samples = int(rollout_profile.get("samples", 0))
    rollout_passed = bool(
        rollout.get("passed") is True
        and rollout_samples >= 20
        and isinstance(candidate_metrics, dict)
        and isinstance(baseline_metrics, dict)
        and int(candidate_metrics.get("samples", 0)) >= 20
        and int(baseline_metrics.get("samples", 0)) >= 20
        and rollout_profile.get("atom_delete_log_rate_adjustment") == -0.5
        and rollout_profile.get("small_ring_log_rate_adjustment") == -1.5
    )
    qualification_passed = panel_passed and heldout_passed and rollout_passed
    qualification = {
        "format": ANALYTIC_BACKBONE_QUALIFICATION_FORMAT,
        "decision": {
            "qualification_passed": qualification_passed,
            "status": "passed" if qualification_passed else "failed",
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha256,
            "weight_source": "pancake_derived",
            "source_checkpoint_sha256": checkpoint_sha256,
            "property_condition_dim": 0,
        },
        "execution": {
            "backbone_execution_kind": "analytic_pancake_quotient_adapter_v1",
            "canonical_successor_execution": True,
            "molecular_self_transitions_virtualized": True,
            "analytic_adapter": {
                "source_path": str(ANALYTIC_SOURCE.relative_to(ROOT)),
                "source_sha256": file_sha256(ANALYTIC_SOURCE),
            },
            "calibration": {
                "atom_delete_log_rate_adjustment": -0.5,
                "small_ring_log_rate_adjustment": -1.5,
                "small_ring_maximum_size": 4,
            },
            "history_safety": "canonical_history_exact_thinning_v1",
        },
        "model_modes": {
            "empirical_mark_prior_mode": "none",
            "ring_family_mass_mode": "boolean",
            "p1_p2_imported": False,
        },
        "evidence": {
            "multi_state_panel": {
                "passed": panel_passed,
                "state_count": len(evidence_rows),
                "artifact": str(args.rate_evidence),
                "artifact_sha256": file_sha256(args.rate_evidence),
                "panel_manifest": str(args.panel),
                "panel_manifest_sha256": file_sha256(args.panel),
            },
            "heldout_rate_gate": {
                "passed": heldout_passed,
                "state_count": len(heldout_rows),
                "artifact": str(args.rate_evidence),
                "artifact_sha256": file_sha256(args.rate_evidence),
            },
            "rollout_gate": {
                "passed": rollout_passed,
                "artifact": str(args.rollout),
                "artifact_sha256": file_sha256(args.rollout),
                "candidate_samples": int(candidate_metrics.get("samples", 0)),
                "baseline_samples": int(baseline_metrics.get("samples", 0)),
                "matched_profile": rollout_profile,
            },
        },
        "checkpoint_payload_contract": {
            "tree_source_prior_present": "tree_source_prior" in checkpoint_payload,
            "model_property_condition_dim": int(model.property_condition_dim),
        },
        "qualification_builder": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "launch_authorized": qualification_passed,
    }
    _write(args.qualification, qualification)
    print(
        json.dumps(
            {
                "qualification_passed": qualification_passed,
                "panel_states": len(evidence_rows),
                "heldout_states": len(heldout_rows),
                "qualification": str(args.qualification),
                "rate_evidence": str(args.rate_evidence),
            },
            sort_keys=True,
        )
    )
    if not qualification_passed:
        raise RuntimeError("analytic quotient backbone qualification failed")


if __name__ == "__main__":
    main()
