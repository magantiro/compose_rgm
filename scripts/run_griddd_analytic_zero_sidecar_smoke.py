#!/usr/bin/env python3
"""Run the smallest real three-arm GrIDDD execution smoke on the qualified base.

This is deliberately an execution and accounting gate, not an efficacy gate.
The QED residual sidecar is zero initialized and untrained, so direct
conditioning is bound to the requested target but is exactly identical to the
qualified analytic unconditional base.  Only the controller arms can use QED
oracle values to select among real canonical rewrite proposals.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from time import time

import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.canonical_successor_distillation import (
    AnalyticPancakeQuotientSampler,
    FrozenQEDResidualAdapter,
    FrozenQEDResidualConfig,
)
from compose_v4.experiments.griddd_conditional import (
    CorrectedBackboneContract,
    FrozenResidualConditionalBackbone,
    GridDDConditionalEvaluator,
    GridDDProtocol,
    LeadRecord,
    RewriteLeadCandidateGenerator,
    file_sha256,
    load_unconditional_backbone_qualification,
    qed_state_oracle,
    standard_conditional_arms,
)

if __package__:
    from scripts.evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
else:
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint


ROOT = Path(__file__).resolve().parents[1]
LEAD_SMILES = "Cc1occc1C(=O)NNC(=O)Nc1ccc(F)cc1F"
DEFAULT_CHECKPOINT = Path("/private/tmp/pancake_checkpoint/checkpoint.recovery.pt")
DEFAULT_QUALIFICATION = (
    ROOT / "diagnostics/canonical_successor_analytic_backbone_qualification.json"
)
DEFAULT_CONFIG = ROOT / "configs/experiments/griddd_qed_frozen_residual_pilot_v1.json"
DEFAULT_OUTPUT = ROOT / "diagnostics/griddd_analytic_zero_sidecar_rewrite_smoke.json"


def _atomic_json_write(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, output)


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--qualification", type=Path, default=DEFAULT_QUALIFICATION)
    parser.add_argument("--sidecar-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--operational-horizon", type=float, default=2.0)
    parser.add_argument("--max-events", type=int, default=4)
    args = parser.parse_args()
    if args.operational_horizon <= 0.0 or args.max_events <= 0:
        raise ValueError("rollout horizon and event limit must be positive")

    started_unix_seconds = time()
    qualification = load_unconditional_backbone_qualification(args.qualification)
    qualification.assert_checkpoint_file(args.checkpoint)
    config = _object(args.sidecar_config)
    property_config = config.get("property")
    adapter_config = config.get("adapter")
    if not isinstance(property_config, dict) or not isinstance(adapter_config, dict):
        raise ValueError("sidecar config lacks property or adapter settings")
    if str(property_config.get("name")) != "qed":
        raise ValueError("execution smoke requires the frozen QED sidecar config")

    torch.manual_seed(args.seed)
    base_model, _base_payload = load_factorized_rollout_checkpoint(args.checkpoint)
    base_model.eval()
    adapter = FrozenQEDResidualAdapter(
        AnalyticPancakeQuotientSampler(base_model),
        FrozenQEDResidualConfig(
            qed_mean=float(property_config["normalizer_mean"]),
            qed_standard_deviation=float(
                property_config["normalizer_standard_deviation"]
            ),
            hidden_dim=int(adapter_config["hidden_dim"]),
        ),
    ).eval()
    contract = CorrectedBackboneContract(
        checkpoint_sha256=qualification.checkpoint_sha256,
        weight_source=qualification.weight_source,
        canonical_successor_execution=(
            qualification.canonical_successor_execution
        ),
        molecular_self_transitions_virtualized=(
            qualification.molecular_self_transitions_virtualized
        ),
        unconditional_qualification_passed=qualification.qualification_passed,
        empirical_mark_prior_mode=qualification.empirical_mark_prior_mode,
        ring_family_mass_mode=qualification.ring_family_mass_mode,
        unconditional_execution_kind=qualification.execution_kind,
    )
    backbone = FrozenResidualConditionalBackbone(
        adapter=adapter,
        contract=contract,
        small_ring_log_rate_adjustment=float(
            qualification.small_ring_log_rate_adjustment
        ),
        small_ring_maximum_size=4,
    )

    lead_state = pad_molecular_graph(smiles_to_molecular_graph(LEAD_SMILES), 40)
    lead_qed = qed_state_oracle(lead_state)
    target_qed = 0.90
    identity_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for model_time in (0.0, 0.2, 0.4):
            base = adapter.base_sampler.rate_table(lead_state, model_time)
            direct = adapter.rate_table(
                lead_state,
                model_time,
                target_qed=target_qed,
            )
            missing = adapter.rate_table(
                lead_state,
                model_time,
                target_qed=None,
            )
            identity_rows.append(
                {
                    "model_time": model_time,
                    "target_vs_base_max_family_rate_absolute_error": float(
                        (direct.family_rates - base.productive_family_rates).abs().max()
                    ),
                    "missing_vs_base_max_family_rate_absolute_error": float(
                        (missing.family_rates - base.productive_family_rates).abs().max()
                    ),
                    "target_vs_base_total_hazard_absolute_error": abs(
                        float(direct.total_hazard)
                        - float(base.productive_total_hazard)
                    ),
                    "maximum_absolute_family_log_hazard_residual": float(
                        direct.family_log_hazard_residuals.abs().max()
                    ),
                }
            )

    protocol = GridDDProtocol(
        candidates_per_start=1,
        guidance_oracle_calls_per_candidate=4,
        proposals_per_controlled_event=4,
        bootstrap_replicates=250,
        bootstrap_seed=args.seed + 1,
    )
    generator = RewriteLeadCandidateGenerator(
        backbone=backbone,
        n_slots=40,
        operational_horizon=args.operational_horizon,
        time_step=0.1,
        max_events=args.max_events,
        proposals_per_event=protocol.proposals_per_controlled_event,
        minimum_similarity=protocol.minimum_tanimoto_similarity,
        fingerprint_radius=protocol.fingerprint_radius,
        fingerprint_bits=protocol.fingerprint_bits,
    )
    evaluation = GridDDConditionalEvaluator(
        protocol=protocol,
        generator=generator,
        backbone_contract=contract,
    ).evaluate(
        (LeadRecord("griddd-qualified-zero-sidecar-smoke-0", lead_state),),
        seeds=(args.seed,),
        arms=standard_conditional_arms(beta=8.0),
    )
    arms = evaluation["arms"]
    records = evaluation["records"]
    assert isinstance(arms, dict)
    assert isinstance(records, dict)
    expected_calls = protocol.oracle_calls_per_start_seed_arm
    identity_exact = all(
        max(
            float(row["target_vs_base_max_family_rate_absolute_error"]),
            float(row["missing_vs_base_max_family_rate_absolute_error"]),
            float(row["target_vs_base_total_hazard_absolute_error"]),
            float(row["maximum_absolute_family_log_hazard_residual"]),
        )
        == 0.0
        for row in identity_rows
    )
    arm_gate = all(
        bool(arms[name]["exact_oracle_budget_every_start_seed"])
        and int(arms[name]["total_oracle_calls"]) == expected_calls
        and int(arms[name]["candidate_attempts"]) == 1
        and int(arms[name]["valid_candidates"]) == 1
        and bool(arms[name]["all_trajectory_states_valid"])
        and bool(arms[name]["all_trajectory_states_connected"])
        for name in ("direct", "controller", "combined")
    )
    actual_rewrite_gate = all(
        int(records[name][0]["candidates"][0]["events"]) > 0
        and not bool(records[name][0]["candidates"][0]["stalled"])
        for name in ("direct", "controller", "combined")
    )
    similarity_gate = all(
        float(records[name][0]["candidates"][0]["similarity_to_lead"])
        >= protocol.minimum_tanimoto_similarity
        for name in ("direct", "controller", "combined")
    )
    oracle_influence_gate = bool(
        int(records["direct"][0]["oracle"]["selection_calls"]) == 0
        and int(records["controller"][0]["oracle"]["selection_calls"]) > 0
        and int(records["combined"][0]["oracle"]["selection_calls"]) > 0
    )
    passed = bool(
        qualification.qualification_passed
        and 0.70 <= lead_qed <= 0.80
        and identity_exact
        and arm_gate
        and actual_rewrite_gate
        and similarity_gate
        and oracle_influence_gate
    )
    output: dict[str, object] = {
        "format": "compose_v4_griddd_analytic_zero_sidecar_rewrite_smoke_v1",
        "phase": "complete" if passed else "failed",
        "passed": passed,
        "gate_role": "real_rewrite_execution_and_exact_accounting_only",
        "real_rewrite_generator": True,
        "scripted_candidates": False,
        "learned_qed_sidecar": False,
        "zero_initialized_untrained_sidecar": True,
        "direct_conditioning_has_learned_effect": False,
        "controller_uses_exact_qed_for_selection": True,
        "efficacy_claim_authorized": False,
        "griddd_comparable_claim_authorized": False,
        "large_benchmark_authorized": False,
        "training_launched": False,
        "started_unix_seconds": started_unix_seconds,
        "completed_unix_seconds": time(),
        "inputs": {
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": file_sha256(args.checkpoint),
            },
            "qualification": {
                "path": str(args.qualification),
                "sha256": file_sha256(args.qualification),
            },
            "sidecar_config": {
                "path": str(args.sidecar_config),
                "sha256": file_sha256(args.sidecar_config),
            },
            "lead": {
                "id": "griddd-qualified-zero-sidecar-smoke-0",
                "smiles": LEAD_SMILES,
                "qed": lead_qed,
                "eligible_qed_interval": [0.70, 0.80],
            },
            "target_qed": target_qed,
            "minimum_tanimoto_similarity": 0.40,
            "seed": args.seed,
            "operational_horizon": args.operational_horizon,
            "max_events": args.max_events,
        },
        "qualification": qualification.to_dict(),
        "sidecar_contract": adapter.sidecar_contract(),
        "zero_initialization_identity": {
            "exact": identity_exact,
            "rows": identity_rows,
        },
        "expected_oracle_calls_per_arm": expected_calls,
        "gate_checks": {
            "qualified_analytic_backbone": qualification.qualification_passed,
            "lead_in_frozen_qed_window": 0.70 <= lead_qed <= 0.80,
            "zero_sidecar_identity_exact": identity_exact,
            "three_arm_validity_and_exact_accounting": arm_gate,
            "actual_rewrite_event_every_arm": actual_rewrite_gate,
            "final_similarity_at_least_0_40_every_arm": similarity_gate,
            "oracle_influence_matches_arm_semantics": oracle_influence_gate,
            "qed_target_0_90_is_an_outcome_not_an_execution_gate": True,
        },
        "evaluation": evaluation,
    }
    _atomic_json_write(output, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": passed,
                "phase": output["phase"],
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise RuntimeError("qualified zero-sidecar GrIDDD execution smoke failed")


if __name__ == "__main__":
    main()
