#!/usr/bin/env python3
"""Matched smallest rollout gate for analytic pancake quotient execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from compose_v4.experiments.calibrated_rewrite_sampling import (
    ThinnedRateCalibrationSampler,
)
from compose_v4.experiments.canonical_successor_distillation import (
    AnalyticPancakeQuotientSampler,
    CanonicalHistoryThinningSampler,
)
from compose_v4.experiments.cnof_conditional import corpus_rollout_metrics
from compose_v4.experiments.parallel_tracelet_sampling import (
    sample_tracelet_ancestral_many,
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _family_tv(left: dict[str, float], right: dict[str, float]) -> float:
    names = set(left) | set(right)
    return 0.5 * sum(
        abs(float(left.get(name, 0.0)) - float(right.get(name, 0.0)))
        for name in names
    )


def _selected(metrics: dict[str, object]) -> dict[str, object]:
    trajectory = metrics["trajectory_diagnostics"]
    return {
        "valid_fraction": metrics["valid_fraction"],
        "connected_or_null_fraction": metrics["connected_or_null_fraction"],
        "mean_atoms": metrics["mean_atoms"],
        "mean_cycle_rank": metrics["mean_cycle_rank"],
        "cycle_rank_total_variation": metrics["cycle_rank_total_variation"],
        "mean_events": metrics["mean_events"],
        "event_rule_fractions": metrics["event_rule_fractions"],
        "all_step_valid_trajectory_fraction": trajectory[
            "all_step_valid_trajectory_fraction"
        ],
        "canonical_self_event_count": trajectory["canonical_self_event_count"],
        "immediate_backtrack_count": trajectory["immediate_backtrack_count"],
        "mean_unique_state_fraction": trajectory["mean_unique_state_fraction"],
        "delete_to_one_then_regrow_fraction": trajectory[
            "delete_to_one_then_regrow_fraction"
        ],
        "collapsed_from_larger_source_to_one_atom_fraction": trajectory[
            "collapsed_from_larger_source_to_one_atom_fraction"
        ],
        "event_budget_exhaustion_fraction": metrics[
            "event_budget_exhaustion_fraction"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    if args.samples <= 0 or args.workers <= 0:
        raise ValueError("samples and workers must be positive")

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    teacher, checkpoint_payload = load_factorized_rollout_checkpoint(args.checkpoint)
    path_payload = __import__("torch").load(
        args.validation_paths,
        map_location="cpu",
        weights_only=False,
    )
    records = tuple(path_payload["records"])
    reference_smiles = tuple(dict.fromkeys(record.target_key for record in records))
    profile = {
        "samples": args.samples,
        "workers": args.workers,
        "seed": args.seed,
        "n_slots": 40,
        "operational_horizon": 16.0,
        "time_step": 0.1,
        "max_events": 128,
        "small_ring_log_rate_adjustment": -1.5,
        "atom_delete_log_rate_adjustment": -0.5,
    }
    baseline = ThinnedRateCalibrationSampler(
        teacher,
        family_log_rate_adjustments=(("atom_delete", -0.5),),
        small_ring_log_rate_adjustment=-1.5,
    )
    analytic = AnalyticPancakeQuotientSampler(teacher)
    candidate = CanonicalHistoryThinningSampler(
        ThinnedRateCalibrationSampler(
            analytic,
            small_ring_log_rate_adjustment=-1.5,
        )
    )

    def sample(model, label: str):
        start = perf_counter()
        rollouts = sample_tracelet_ancestral_many(
            model,
            seed=args.seed,
            samples=args.samples,
            workers=args.workers,
            n_slots=40,
            operational_horizon=16.0,
            time_step=0.1,
            max_events=128,
            torch_threads_per_worker=1,
            source_prior=checkpoint_payload["tree_source_prior"],
        )
        elapsed = perf_counter() - start
        metrics = corpus_rollout_metrics(
            rollouts,
            train_smiles=(),
            reference_smiles=reference_smiles,
        )
        controls = tuple(
            rollout.control_diagnostics
            for rollout in rollouts
            if rollout.control_diagnostics is not None
        )
        virtual_rules: dict[str, int] = {}
        for rollout in rollouts:
            for rule in rollout.virtual_event_rules:
                virtual_rules[rule] = virtual_rules.get(rule, 0) + 1
        return {
            "label": label,
            "elapsed_seconds": elapsed,
            "metrics": metrics,
            "selected_metrics": _selected(metrics),
            "control_diagnostics": controls,
            "virtual_event_rule_counts": dict(sorted(virtual_rules.items())),
        }

    baseline_result = sample(baseline, "calibrated_pancake_virtual_thinning")
    candidate_result = sample(candidate, "analytic_productive_quotient")
    baseline_metrics = baseline_result["metrics"]
    candidate_metrics = candidate_result["metrics"]
    candidate_trajectory = candidate_metrics["trajectory_diagnostics"]
    family_tv = _family_tv(
        candidate_metrics["event_rule_fractions"],
        baseline_metrics["event_rule_fractions"],
    )
    safety_checks = {
        "valid_fraction_eq_1": candidate_metrics["valid_fraction"] == 1.0,
        "connected_fraction_eq_1": (
            candidate_metrics["connected_or_null_fraction"] == 1.0
        ),
        "all_step_valid_fraction_eq_1": (
            candidate_trajectory["all_step_valid_trajectory_fraction"] == 1.0
        ),
        "committed_self_events_eq_0": (
            candidate_trajectory["canonical_self_event_count"] == 0
        ),
        "committed_backtracks_eq_0": (
            candidate_trajectory["immediate_backtrack_count"] == 0
        ),
        "event_budget_exhaustion_eq_0": (
            candidate_metrics["event_budget_exhaustion_fraction"] == 0.0
        ),
    }
    baseline_mean_events = float(baseline_metrics["mean_events"])
    candidate_mean_events = float(candidate_metrics["mean_events"])
    behavioral_checks = {
        "mean_atoms_within_3_of_baseline": (
            abs(
                float(candidate_metrics["mean_atoms"])
                - float(baseline_metrics["mean_atoms"])
            )
            <= 3.0
        ),
        "mean_cycle_rank_within_1_of_baseline": (
            abs(
                float(candidate_metrics["mean_cycle_rank"])
                - float(baseline_metrics["mean_cycle_rank"])
            )
            <= 1.0
        ),
        "event_family_tv_le_0p20": family_tv <= 0.20,
        "mean_events_ratio_0p5_to_1p5": (
            0.5
            <= candidate_mean_events / max(baseline_mean_events, 1e-12)
            <= 1.5
        ),
        "elapsed_time_le_1p5x_baseline": (
            candidate_result["elapsed_seconds"]
            <= 1.5 * baseline_result["elapsed_seconds"]
        ),
    }
    safety_passed = all(safety_checks.values())
    behavior_passed = all(behavioral_checks.values())
    no_graft = None
    if safety_passed and not behavior_passed:
        no_graft_model = CanonicalHistoryThinningSampler(
            ThinnedRateCalibrationSampler(
                AnalyticPancakeQuotientSampler(teacher),
                family_log_rate_adjustments=(("bond_reroute", -1000.0),),
                small_ring_log_rate_adjustment=-1.5,
            )
        )
        no_graft_result = sample(
            no_graft_model,
            "analytic_productive_quotient_no_graft_diagnostic",
        )
        no_graft = {
            "trigger": "base analytic arm passed safety but failed behavior",
            "not_a_promotion_candidate": True,
            "matched_profile": profile,
            "base": candidate_result["selected_metrics"],
            "no_graft": no_graft_result["selected_metrics"],
            "full_arm": no_graft_result,
        }

    output = {
        "format": "compose_v4_analytic_pancake_quotient_rollout_smoke_v1",
        "profile": profile,
        "baseline": baseline_result,
        "candidate": candidate_result,
        "comparison": {
            "event_family_total_variation": family_tv,
            "mean_atoms_delta": (
                float(candidate_metrics["mean_atoms"])
                - float(baseline_metrics["mean_atoms"])
            ),
            "mean_cycle_rank_delta": (
                float(candidate_metrics["mean_cycle_rank"])
                - float(baseline_metrics["mean_cycle_rank"])
            ),
            "mean_events_ratio": candidate_mean_events
            / max(baseline_mean_events, 1e-12),
            "elapsed_time_ratio": candidate_result["elapsed_seconds"]
            / max(baseline_result["elapsed_seconds"], 1e-12),
        },
        "safety_checks": safety_checks,
        "behavioral_checks": behavioral_checks,
        "safety_passed": safety_passed,
        "behavior_passed": behavior_passed,
        "passed": safety_passed and behavior_passed,
        "no_graft_diagnostic": no_graft,
        "decision": (
            "analytic quotient rollout gate passed"
            if safety_passed and behavior_passed
            else "inspect safety failure; no no-Graft diagnostic allowed"
            if not safety_passed
            else "behavior suspect; inspect matched no-Graft diagnostic"
        ),
        "corpus_scale_training_authorized": False,
    }
    _atomic_json(args.output, output)
    print(
        json.dumps(
            {
                "safety_passed": safety_passed,
                "behavior_passed": behavior_passed,
                "passed": output["passed"],
                "comparison": output["comparison"],
                "no_graft_ran": no_graft is not None,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
