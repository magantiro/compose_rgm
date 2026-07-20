"""Evaluate direct molecular-property conditioning under matched CTMC seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

import numpy as np
import torch

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.experiments.molecular_property_conditioning import (
    PROPERTY_FUNCTIONS,
    PropertyConditionNormalizer,
    molecular_property_values,
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
    try:
        names = tuple(str(value) for value in raw["names"])
        means = tuple(float(value) for value in raw["means"])
        standard_deviations = tuple(float(value) for value in raw["standard_deviations"])
    except (KeyError, TypeError) as error:
        raise ValueError("malformed property-conditioning metadata") from error
    return PropertyConditionNormalizer(
        names=names,
        means=means,
        standard_deviations=standard_deviations,
    )


def _rank_correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None

    def average_ranks(values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        order = np.argsort(array, kind="mergesort")
        ranks = np.empty(len(array), dtype=np.float64)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and array[order[stop]] == array[order[start]]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return ranks

    x_ranks = average_ranks(x)
    y_ranks = average_ranks(y)
    value = float(np.corrcoef(x_ranks, y_ranks)[0, 1])
    return value if np.isfinite(value) else None


def _condition_report(
    *,
    label: str,
    raw_target: tuple[float, ...] | None,
    rollouts: tuple[object, ...],
    property_names: tuple[str, ...],
    elapsed_seconds: float,
    tolerance: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for index, rollout in enumerate(rollouts):
        final_state = rollout.final_state
        smiles = molecular_graph_to_smiles(final_state)
        values = None if smiles is None else molecular_property_values(final_state, property_names)
        rows.append(
            {
                "sample_index": index,
                "smiles": smiles,
                "properties": None if values is None else list(values),
                "events": len(rollout.event_rules),
                "event_rules": list(rollout.event_rules),
            }
        )
    valid_rows = [row for row in rows if row["properties"] is not None]
    unique_smiles = {row["smiles"] for row in valid_rows}
    property_columns = [
        [float(row["properties"][column]) for row in valid_rows]
        for column in range(len(property_names))
    ]
    report: dict[str, object] = {
        "label": label,
        "raw_target": None if raw_target is None else list(raw_target),
        "attempts": len(rows),
        "valid": len(valid_rows),
        "valid_fraction": len(valid_rows) / max(len(rows), 1),
        "unique_valid": len(unique_smiles),
        "unique_fraction_of_valid": len(unique_smiles) / max(len(valid_rows), 1),
        "elapsed_seconds": elapsed_seconds,
        "mean_events": mean([int(row["events"]) for row in rows]) if rows else 0.0,
        "properties": {},
    }
    for index, name in enumerate(property_names):
        values = property_columns[index]
        target = None if raw_target is None else raw_target[index]
        report["properties"][name] = {
            "mean": None if not values else mean(values),
            "standard_deviation": None if not values else pstdev(values),
            "minimum": None if not values else min(values),
            "maximum": None if not values else max(values),
            "mean_absolute_error": (
                None if target is None or not values else mean(abs(value - target) for value in values)
            ),
            "tolerance_success_fraction": (
                None
                if target is None or not values
                else sum(abs(value - target) <= tolerance for value in values) / len(values)
            ),
        }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-cache", type=Path)
    parser.add_argument("--target", type=float, action="append", required=True)
    parser.add_argument("--samples-per-target", type=int, default=25)
    parser.add_argument("--include-classifier-free-control", action="store_true")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--operational-horizon", type=float, default=16.0)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.1)
    args = parser.parse_args()

    if args.samples_per_target <= 0 or args.workers <= 0:
        raise ValueError("sampling counts must be positive")
    if args.max_atoms <= 0 or args.max_events <= 0:
        raise ValueError("state and event limits must be positive")
    if args.tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    model, checkpoint = load_factorized_rollout_checkpoint(args.checkpoint)
    normalizer = _normalizer_from_checkpoint(checkpoint)
    if len(normalizer.names) != 1:
        raise ValueError("the current evaluator expects one directly conditioned property")
    if normalizer.names[0] not in PROPERTY_FUNCTIONS:
        raise ValueError("checkpoint uses an unsupported molecular property")
    source_prior = checkpoint["tree_source_prior"]
    if max(source_prior.sizes) > args.max_atoms:
        raise ValueError("checkpoint source prior exceeds max atoms")

    conditions: list[tuple[str, tuple[float, ...] | None, PropertyConditionedRewriteSampler]] = []
    if args.include_classifier_free_control:
        conditions.append(
            (
                "classifier_free",
                None,
                PropertyConditionedRewriteSampler(
                    model,
                    property_values=(0.0,),
                    property_mask=(False,),
                ),
            )
        )
    for target in args.target:
        raw = (float(target),)
        conditions.append(
            (
                f"{normalizer.names[0]}_{target:g}",
                raw,
                PropertyConditionedRewriteSampler(
                    model,
                    property_values=normalizer.transform(raw),
                ),
            )
        )

    reports: list[dict[str, object]] = []
    cached_rollouts: dict[str, tuple[object, ...]] = {}
    sample_rows: dict[str, list[dict[str, object]]] = {}
    requested: list[float] = []
    achieved: list[float] = []
    partial_output = args.output.with_suffix(f"{args.output.suffix}.partial")
    partial_rollout_cache = (
        None
        if args.rollout_cache is None
        else args.rollout_cache.with_suffix(f"{args.rollout_cache.suffix}.partial")
    )
    for label, raw_target, sampler in conditions:
        started = perf_counter()

        def report_progress(completed: int, total: int) -> None:
            print(
                json.dumps(
                    {
                        "phase": "condition_progress",
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
            samples=args.samples_per_target,
            workers=args.workers,
            n_slots=args.max_atoms,
            operational_horizon=args.operational_horizon,
            time_step=args.time_step,
            max_events=args.max_events,
            torch_threads_per_worker=args.torch_threads,
            progress_callback=report_progress,
            source_prior=source_prior,
        )
        report, rows = _condition_report(
            label=label,
            raw_target=raw_target,
            rollouts=rollouts,
            property_names=normalizer.names,
            elapsed_seconds=perf_counter() - started,
            tolerance=args.tolerance,
        )
        reports.append(report)
        cached_rollouts[label] = rollouts
        sample_rows[label] = rows
        if raw_target is not None:
            for row in rows:
                if row["properties"] is not None:
                    requested.append(raw_target[0])
                    achieved.append(float(row["properties"][0]))
        partial_payload = {
            "format": "compose_v4_direct_property_rollout_partial_v1",
            "complete": False,
            "checkpoint": str(args.checkpoint),
            "checkpoint_kind": checkpoint.get("checkpoint_kind"),
            "normalizer": normalizer.to_dict(),
            "matched_sampling": {
                "same_source_prior": True,
                "same_seed_per_condition": True,
                "seed": args.seed + 4,
                "samples_per_condition": args.samples_per_target,
                "ancestral_ctmc": True,
                "beam_search": False,
            },
            "conditions": reports,
            "samples": sample_rows,
        }
        _atomic_json_dump(partial_payload, partial_output)
        if partial_rollout_cache is not None:
            _atomic_torch_save(
                {"rollouts": cached_rollouts, "metadata": partial_payload},
                partial_rollout_cache,
            )
        print(
            json.dumps(
                {
                    "phase": "partial_checkpoint_saved",
                    "output": str(partial_output),
                    "conditions": len(reports),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print(json.dumps({"phase": "condition_complete", **report}, sort_keys=True), flush=True)

    payload = {
        "format": "compose_v4_direct_property_rollout_evaluation_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_kind": checkpoint.get("checkpoint_kind"),
        "selected_validation": checkpoint.get("selected_validation"),
        "normalizer": normalizer.to_dict(),
        "matched_sampling": {
            "same_source_prior": True,
            "same_seed_per_condition": True,
            "seed": args.seed + 4,
            "samples_per_condition": args.samples_per_target,
            "ancestral_ctmc": True,
            "beam_search": False,
        },
        "conditions": reports,
        "target_achieved_spearman": _rank_correlation(requested, achieved),
        "samples": sample_rows,
    }
    _atomic_json_dump(payload, args.output)
    if args.rollout_cache is not None:
        _atomic_torch_save(
            {"rollouts": cached_rollouts, "metadata": payload},
            args.rollout_cache,
        )
    print(json.dumps({"phase": "complete", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
