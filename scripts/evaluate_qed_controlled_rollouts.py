"""Run a bounded matched-seed stochastic QED controlled-CTMC diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.experiments.guided_rewrite_sampling import (
    CachedStateScorer,
    CONTROLLED_MARK_EQUATION,
    ProposalGuidedRewriteSampler,
    QEDTargetDistancePotential,
    qed_state_score,
)
from compose_v4.experiments.molecular_property_conditioning import (
    PropertyConditionNormalizer,
)
from compose_v4.experiments.parallel_tracelet_sampling import (
    sample_tracelet_ancestral_many,
)
from compose_v4.experiments.property_conditioned_sampling import (
    PropertyConditionedRewriteSampler,
)
from scripts.evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint


def _atomic_json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _normalizer_from_checkpoint(payload: dict[str, object]) -> PropertyConditionNormalizer:
    raw = payload.get("property_conditioning")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint lacks property-conditioning metadata")
    normalizer = PropertyConditionNormalizer(
        names=tuple(str(value) for value in raw["names"]),
        means=tuple(float(value) for value in raw["means"]),
        standard_deviations=tuple(float(value) for value in raw["standard_deviations"]),
    )
    if normalizer.names != ("qed",):
        raise ValueError("controlled QED rollout requires a QED-conditioned checkpoint")
    return normalizer


def _rollout_arm_report(
    label: str,
    rollouts: tuple[object, ...],
    *,
    target_qed: float,
    tolerance: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for sample_index, rollout in enumerate(rollouts):
        smiles = molecular_graph_to_smiles(rollout.final_state)
        qed = None if smiles is None else float(qed_state_score(rollout.final_state))
        audit = rollout.control_diagnostics
        if not isinstance(audit, dict):
            raise ValueError("controlled rollout lacks proposal/oracle diagnostics")
        diagnostics = rollout.diagnostics
        if diagnostics is None:
            raise ValueError("controlled rollout lacks trajectory diagnostics")
        events = tuple(audit.get("events", ()))
        proposal_valid = all(
            bool(proposal["valid"])
            for event in events
            for proposal in event["proposals"]
        )
        proposal_connected = all(
            bool(proposal["connected"])
            for event in events
            for proposal in event["proposals"]
        )
        rows.append(
            {
                "sample_index": sample_index,
                "smiles": smiles,
                "qed": qed,
                "events": len(rollout.event_rules),
                "event_rules": list(rollout.event_rules),
                "event_budget_exhausted": bool(rollout.exhausted_event_budget),
                "virtual_events": len(rollout.virtual_event_rules),
                "all_trajectory_states_valid": all(diagnostics.state_valid),
                "all_trajectory_states_connected": all(
                    diagnostics.state_connected_or_null
                ),
                "trajectory_states": len(diagnostics.state_valid),
                "all_scored_proposals_valid": proposal_valid,
                "all_scored_proposals_connected": proposal_connected,
                "control_diagnostics": audit,
            }
        )
    valid_rows = [row for row in rows if row["qed"] is not None]
    qeds = [float(row["qed"]) for row in valid_rows]
    unique_smiles = {str(row["smiles"]) for row in valid_rows}
    family_counts = Counter(
        str(rule_name) for row in rows for rule_name in row["event_rules"]
    )
    proposal_family_counts = Counter(
        str(proposal["rule_name"])
        for row in rows
        for event in row["control_diagnostics"]["events"]
        for proposal in event["proposals"]
    )
    oracle = {
        key: sum(int(row["control_diagnostics"][key]) for row in rows)
        for key in (
            "raw_proposal_marks",
            "canonical_group_requests",
            "unique_oracle_evaluations",
            "cache_hits",
        )
    }
    report = {
        "label": label,
        "attempts": len(rows),
        "valid": len(valid_rows),
        "valid_fraction": len(valid_rows) / len(rows),
        "unique_valid": len(unique_smiles),
        "unique_fraction_of_valid": len(unique_smiles) / max(len(valid_rows), 1),
        "mean_qed": mean(qeds),
        "qed_standard_deviation": pstdev(qeds),
        "minimum_qed": min(qeds),
        "maximum_qed": max(qeds),
        "mean_absolute_target_error": mean(abs(value - target_qed) for value in qeds),
        "within_tolerance": sum(abs(value - target_qed) <= tolerance for value in qeds),
        "qed_at_least_target": sum(value >= target_qed for value in qeds),
        "mean_events": mean(int(row["events"]) for row in rows),
        "event_budget_exhaustions": sum(
            bool(row["event_budget_exhausted"]) for row in rows
        ),
        "virtual_events": sum(int(row["virtual_events"]) for row in rows),
        "all_trajectory_states_valid": all(
            bool(row["all_trajectory_states_valid"]) for row in rows
        ),
        "all_trajectory_states_connected": all(
            bool(row["all_trajectory_states_connected"]) for row in rows
        ),
        "trajectory_states_audited": sum(int(row["trajectory_states"]) for row in rows),
        "all_scored_proposals_valid": all(
            bool(row["all_scored_proposals_valid"]) for row in rows
        ),
        "all_scored_proposals_connected": all(
            bool(row["all_scored_proposals_connected"]) for row in rows
        ),
        "committed_family_counts": dict(sorted(family_counts.items())),
        "proposal_family_counts": dict(sorted(proposal_family_counts.items())),
        "oracle_accounting": oracle,
        "per_attempt_oracle_accounting": [
            {
                "sample_index": int(row["sample_index"]),
                **{
                    key: int(row["control_diagnostics"][key])
                    for key in oracle
                },
            }
            for row in rows
        ],
    }
    return report, rows


def _paired_report(
    control_rows: list[dict[str, object]],
    guided_rows: list[dict[str, object]],
    *,
    target_qed: float,
) -> dict[str, object]:
    if len(control_rows) != len(guided_rows):
        raise ValueError("matched rollout arms do not align")
    qed_deltas = []
    target_error_improvements = []
    for control, guided in zip(control_rows, guided_rows):
        if control["qed"] is None or guided["qed"] is None:
            continue
        control_qed = float(control["qed"])
        guided_qed = float(guided["qed"])
        qed_deltas.append(guided_qed - control_qed)
        target_error_improvements.append(
            abs(control_qed - target_qed) - abs(guided_qed - target_qed)
        )
    return {
        "paired_valid_attempts": len(qed_deltas),
        "mean_qed_delta": mean(qed_deltas),
        "qed_improved_attempts": sum(value > 0.0 for value in qed_deltas),
        "mean_target_error_improvement": mean(target_error_improvements),
        "target_error_improved_attempts": sum(
            value > 0.0 for value in target_error_improvements
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-cache", type=Path, required=True)
    parser.add_argument("--target-qed", type=float, default=0.9)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--proposals-per-event", type=int, default=4)
    parser.add_argument("--guidance-start-event", type=int, default=0)
    parser.add_argument("--max-guided-events", type=int, default=12)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--operational-horizon", type=float, default=16.0)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=0.1)
    args = parser.parse_args()
    if args.samples <= 0 or args.workers <= 0:
        raise ValueError("sample and worker counts must be positive")
    if args.proposals_per_event <= 0 or args.max_guided_events <= 0:
        raise ValueError("proposal and controlled-event bounds must be positive")
    if args.guidance_start_event < 0:
        raise ValueError("guidance start event must be nonnegative")
    if not 0.0 <= args.target_qed <= 1.0:
        raise ValueError("target QED must lie in [0, 1]")
    if args.beta <= 0.0 or not np.isfinite(args.beta):
        raise ValueError("positive guidance beta must be finite")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    model, checkpoint = load_factorized_rollout_checkpoint(args.checkpoint)
    normalizer = _normalizer_from_checkpoint(checkpoint)
    source_prior = checkpoint["tree_source_prior"]
    target_condition = PropertyConditionedRewriteSampler(
        model,
        property_values=normalizer.transform((args.target_qed,)),
    )
    potential = QEDTargetDistancePotential(target_qed=args.target_qed)
    arms = (
        ("beta_zero", 0.0),
        ("target_distance_soft", args.beta),
    )
    reports: list[dict[str, object]] = []
    rows_by_arm: dict[str, list[dict[str, object]]] = {}
    rollouts_by_arm: dict[str, tuple[object, ...]] = {}
    for label, beta in arms:
        sampler = ProposalGuidedRewriteSampler(
            target_condition,
            CachedStateScorer(qed_state_score),
            proposals_per_event=args.proposals_per_event,
            guidance_strength=beta,
            tempering_power=0.0,
            score_potential=potential,
            guidance_start_event=args.guidance_start_event,
            max_guided_events=args.max_guided_events,
        )

        def report_progress(completed: int, total: int) -> None:
            print(
                json.dumps(
                    {
                        "phase": "controlled_rollout_progress",
                        "label": label,
                        "completed": completed,
                        "attempts": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        rollouts = sample_tracelet_ancestral_many(
            sampler,
            seed=args.seed + 4,
            samples=args.samples,
            workers=args.workers,
            n_slots=args.max_atoms,
            operational_horizon=args.operational_horizon,
            time_step=args.time_step,
            max_events=args.max_events,
            torch_threads_per_worker=1,
            progress_callback=report_progress,
            source_prior=source_prior,
        )
        report, rows = _rollout_arm_report(
            label,
            rollouts,
            target_qed=args.target_qed,
            tolerance=args.tolerance,
        )
        reports.append(report)
        rows_by_arm[label] = rows
        rollouts_by_arm[label] = rollouts
        partial = {
            "format": "compose_v4_qed_controlled_rollout_partial_v1",
            "complete": False,
            "conditions": reports,
            "samples": rows_by_arm,
        }
        _atomic_json_dump(partial, args.output.with_suffix(f"{args.output.suffix}.partial"))
        _atomic_torch_save(
            {"rollouts": rollouts_by_arm, "metadata": partial},
            args.rollout_cache.with_suffix(f"{args.rollout_cache.suffix}.partial"),
        )
        print(json.dumps({"phase": "controlled_arm_complete", **report}), flush=True)

    payload = {
        "format": "compose_v4_qed_controlled_rollout_v1",
        "complete": True,
        "checkpoint": str(args.checkpoint),
        "control": {
            "equation": CONTROLLED_MARK_EQUATION,
            "potential": "U_target(q)=-abs(q-target_qed)",
            "total_hazard_preserved": True,
            "beam_search": False,
            "beta_zero_consumes_the_same_oracle_bearing_proposal_mechanism": True,
        },
        "configuration": {
            "target_qed": args.target_qed,
            "samples_per_arm": args.samples,
            "trajectory_seed": args.seed + 4,
            "proposals_per_controlled_event": args.proposals_per_event,
            "zero_based_guidance_start_event": args.guidance_start_event,
            "max_controlled_events": args.max_guided_events,
            "maximum_raw_proposal_marks_per_attempt": (
                args.proposals_per_event * args.max_guided_events
            ),
            "positive_beta": args.beta,
            "max_atoms": args.max_atoms,
            "operational_horizon": args.operational_horizon,
            "time_step": args.time_step,
            "max_events": args.max_events,
        },
        "conditions": reports,
        "paired": _paired_report(
            rows_by_arm["beta_zero"],
            rows_by_arm["target_distance_soft"],
            target_qed=args.target_qed,
        ),
        "samples": rows_by_arm,
        "benchmark_boundary": {
            "developmental_de_novo_diagnostic_only": True,
            "griddd_benchmark": False,
            "similarity_constraint_enforced": False,
            "lead_molecule_inputs": False,
        },
    }
    _atomic_json_dump(payload, args.output)
    _atomic_torch_save(
        {"rollouts": rollouts_by_arm, "metadata": payload},
        args.rollout_cache,
    )
    print(json.dumps({"phase": "complete", "output": str(args.output), **payload["paired"]}))


if __name__ == "__main__":
    main()
