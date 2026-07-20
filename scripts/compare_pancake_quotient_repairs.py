#!/usr/bin/env python3
"""Compare analytic and learned state-conditioned quotient repairs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from compose_v4.experiments.canonical_successor_distillation import (
    AnalyticPancakeQuotientSampler,
    build_calibrated_pancake_quotient_target,
)
from compose_v4.model.factorized_tracelet_rate_model import MARK_RULE_TO_INDEX


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _relative_l1(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (left - right).abs().sum()
        / right.abs().sum().clamp_min(1e-12)
    )


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-paths", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--failed-rate-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-repeats", type=int, default=20)
    args = parser.parse_args()
    if args.warm_repeats <= 0:
        raise ValueError("warm repeats must be positive")
    torch.set_num_threads(1)

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    teacher, checkpoint_payload = load_factorized_rollout_checkpoint(args.checkpoint)
    path_payload = torch.load(
        args.validation_paths,
        map_location="cpu",
        weights_only=False,
    )
    records = tuple(path_payload["records"])
    manifest = json.loads(args.panel_manifest.read_text())
    failed = json.loads(args.failed_rate_metrics.read_text())
    failed_rows = {row["row_id"]: row for row in failed["final_rows"]}
    sampler = AnalyticPancakeQuotientSampler(teacher)
    rng = np.random.default_rng(int(manifest["seed"]) + 31)
    graft_index = MARK_RULE_TO_INDEX["bond_reroute"]

    row_results = []
    analytic_build_times: list[float] = []
    analytic_cache_hit_times: list[float] = []
    analytic_sample_times: list[float] = []
    raw_sample_times: list[float] = []
    for descriptor in manifest["rows"]:
        row_id = descriptor["row_id"]
        record = records[int(descriptor["record_index"])]
        state = record.path.state_at(int(descriptor["progress"]))
        model_time = float(descriptor["model_time"])
        reference = build_calibrated_pancake_quotient_target(
            teacher,
            state,
            model_time,
        )

        start = perf_counter()
        analytic = sampler.rate_table(state, model_time)
        analytic_build_times.append(perf_counter() - start)
        features = sampler.state_features(state, model_time)
        cache_times = []
        sample_times = []
        base_times = []
        for _ in range(args.warm_repeats):
            start = perf_counter()
            sampler.rate_table(state, model_time)
            cache_times.append(perf_counter() - start)
            start = perf_counter()
            sampler.sample_rewrite_mark(state, model_time, rng)
            sample_times.append(perf_counter() - start)
            start = perf_counter()
            with torch.no_grad():
                teacher.sample_rewrite_mark(state, model_time, rng)
            base_times.append(perf_counter() - start)
        analytic_cache_hit_times.extend(cache_times)
        analytic_sample_times.extend(sample_times)
        raw_sample_times.extend(base_times)

        reconstructed_family = (
            features.raw_family_rates
            * features.productive_survival_fractions
        )
        failed_row = failed_rows[row_id]
        raw_graft = float(reference.raw_family_rates[graft_index])
        productive_graft = float(reference.productive_family_rates[graft_index])
        row_results.append(
            {
                **descriptor,
                "failed_head_only": {
                    "family_relative_l1": failed_row["family_relative_l1"],
                    "graft_relative_l1": failed_row["graft_relative_l1"],
                    "student_graft_inflation": failed_row[
                        "student_graft_inflation"
                    ],
                },
                "support_attribution": {
                    "raw_total_hazard": float(reference.raw_total_hazard),
                    "productive_total_hazard": float(
                        reference.productive_total_hazard
                    ),
                    "total_productive_survival_fraction": float(
                        reference.productive_total_hazard
                        / reference.raw_total_hazard.clamp_min(1e-12)
                    ),
                    "raw_graft_rate": raw_graft,
                    "productive_graft_rate": productive_graft,
                    "graft_productive_survival_fraction": (
                        0.0
                        if raw_graft <= 1e-12
                        else productive_graft / raw_graft
                    ),
                    "raw_graft_marks": reference.raw_graft_mark_count,
                    "productive_graft_marks": (
                        reference.productive_graft_mark_count
                    ),
                    "canonical_graft_successors": len(
                        reference.graft_successor_keys
                    ),
                },
                "remedy_a_analytic": {
                    "family_relative_l1": _relative_l1(
                        analytic.productive_family_rates,
                        reference.productive_family_rates,
                    ),
                    "graft_relative_l1": _relative_l1(
                        analytic.graft_successor_rates,
                        reference.graft_successor_rates,
                    ),
                    "hazard_absolute_error": abs(
                        float(analytic.productive_total_hazard)
                        - float(reference.productive_total_hazard)
                    ),
                    "context_build_seconds": analytic_build_times[-1],
                    "cache_hit_seconds": _quantiles(cache_times),
                    "sample_mark_seconds": _quantiles(sample_times),
                },
                "remedy_b_state_conditioned": {
                    "required_exact_inputs": (
                        "raw family rates",
                        "per-family productive survival fractions",
                        "raw/productive Graft log partitions",
                    ),
                    "oracle_feature_reconstruction_family_relative_l1": (
                        _relative_l1(
                            reconstructed_family,
                            reference.productive_family_rates,
                        )
                    ),
                    "requires_new_learned_residual": True,
                    "pretraining_correctness_guarantee": False,
                },
            }
        )

    strata: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in row_results:
        strata[str(row["stratum"])].append(row)
    stratum_attribution = {}
    for name, selected in sorted(strata.items()):
        stratum_attribution[name] = {
            "rows": len(selected),
            "graft_productive_survival_fraction": _quantiles(
                [
                    float(row["support_attribution"]["graft_productive_survival_fraction"])
                    for row in selected
                ]
            ),
            "failed_head_family_relative_l1": _quantiles(
                [
                    float(row["failed_head_only"]["family_relative_l1"])
                    for row in selected
                ]
            ),
        }

    maximum_family_error = max(
        float(row["remedy_a_analytic"]["family_relative_l1"])
        for row in row_results
    )
    maximum_graft_error = max(
        float(row["remedy_a_analytic"]["graft_relative_l1"])
        for row in row_results
    )
    latency = {
        "analytic_context_build_after_support_warm_seconds": _quantiles(
            analytic_build_times
        ),
        "analytic_context_cache_hit_seconds": _quantiles(
            analytic_cache_hit_times
        ),
        "analytic_cached_sample_mark_seconds": _quantiles(analytic_sample_times),
        "raw_checkpoint_sample_mark_seconds": _quantiles(raw_sample_times),
    }
    checks = {
        "analytic_family_relative_l1_max_le_1e-5": maximum_family_error <= 1e-5,
        "analytic_graft_relative_l1_max_le_1e-5": maximum_graft_error <= 1e-5,
        "analytic_context_build_p95_le_0p25s": (
            latency["analytic_context_build_after_support_warm_seconds"]["p95"]
            <= 0.25
        ),
        "analytic_cached_mark_p95_le_5ms": (
            latency["analytic_cached_sample_mark_seconds"]["p95"] <= 0.005
        ),
    }
    analytic_passed = all(checks.values())
    output = {
        "format": "compose_v4_pancake_quotient_repair_comparison_v1",
        "inputs": {
            "checkpoint": str(args.checkpoint),
            "selected_step": (
                checkpoint_payload.get("selected_validation") or {}
            ).get("selected_step"),
            "validation_paths": str(args.validation_paths),
            "panel_manifest": str(args.panel_manifest),
            "failed_head_rate_metrics": str(args.failed_rate_metrics),
            "state_count": len(row_results),
            "warm_repeats_per_state": args.warm_repeats,
        },
        "row_results": row_results,
        "stratum_attribution": stratum_attribution,
        "latency": latency,
        "checks": checks,
        "decision": {
            "remedy_a_analytic_panel_passed": analytic_passed,
            "remedy_a": (
                "GO to the predeclared smallest matched rollout throughput/safety smoke"
                if analytic_passed
                else "NO-GO; repair analytic correctness or latency"
            ),
            "remedy_b_state_conditioned": (
                "NO-GO while analytic execution is lossless and within latency; "
                "retain only as a future residual option"
            ),
            "additional_failed_head_only_steps": "FORBIDDEN",
            "corpus_scale_training_authorized": False,
            "conditional_backbone_qualified": False,
        },
    }
    _atomic_json(args.output, output)
    print(json.dumps(output["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
