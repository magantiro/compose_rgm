"""Falsify late ring scheduling with an exact no-training support audit.

The gate selects the first bridged/fused-or-spiro ring teacher event from each
of the first eligible frozen validation paths.  It compares the original
pre-ring state with the state produced by the existing exact earliest-priority
commutation scheduler.  Checkpoint and path artifacts are read only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
import torch

from compose_v4.eval.ring_calibration import undesirable_small_ring_mask
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_TO_INDEX,
    _masked_family_logits,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.commuting_schedule import (
    schedule_priority_events_earliest,
    states_are_array_exact,
)
from compose_v4.rewrite.ring_system_fiber import (
    matching_ring_system_template_indices,
)
from compose_v4.rewrite.trace import execute_trace
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint


ELIGIBLE_TOPOLOGIES = frozenset({"bridged_or_fused", "spiro"})
TOPOLOGY_GROUPS = (
    "single_ring",
    "bridged_or_fused",
    "spiro",
    "macrocycle",
)
def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(state: Any) -> str:
    digest = hashlib.sha256()
    for array in (
        state.atom_types,
        state.formal_charges,
        state.implicit_h_counts,
        state.bonds,
    ):
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def _score_state(model: Any, state: Any, time: float) -> dict[str, Any]:
    batch = prepare_factorized_mark_batch(
        (state,),
        (float(time),),
        (None,),
        (None,),
        (0.0,),
        ring_catalog=None,
        compute_ring_grow_support=False,
    ).to(model.device)
    with torch.no_grad():
        node, global_state, pair = model._encode_batch(batch)
        masks, logits, action_log_z = model._action_tables(
            batch,
            node,
            global_state,
            pair,
        )
        enabled = torch.isfinite(action_log_z)
        family_logits = _masked_family_logits(
            model._family_base_logits(batch, global_state),
            action_log_z,
            enabled,
            rate_factorization=model.rate_factorization,
        )
        family_probabilities = torch.softmax(family_logits, dim=-1)[0]
        ring_index = MARK_RULE_TO_INDEX["ring_system_grow"]
        template_support = masks["ring_system_grow"][0]
        template_probabilities = torch.softmax(
            logits["ring_system_grow"][0].masked_fill(
                ~template_support,
                float("-inf"),
            ),
            dim=0,
        )
    return {
        "time": float(time),
        "ring_family_probability": float(
            family_probabilities[ring_index].detach().cpu()
        ),
        "template_probabilities": template_probabilities.detach().cpu(),
    }


def _conditional_group_masses(
    probabilities: torch.Tensor,
    support: torch.Tensor,
    group_masks: dict[str, torch.Tensor],
) -> dict[str, float]:
    return {
        name: float(probabilities[support & mask].sum())
        for name, mask in group_masks.items()
    }


def _support_and_scores(
    model: Any,
    *,
    state: Any,
    native_time: float,
    controlled_time: float,
    teacher_action: Any,
    group_masks: dict[str, torch.Tensor],
) -> dict[str, Any]:
    # Force a cold, exact support computation for each compared state.  This
    # changes only ephemeral process-local caches, never checkpoint weights.
    model.clear_ring_candidate_caches()
    started = perf_counter()
    support = torch.tensor(model._ring_grow_support(state), dtype=torch.bool)
    support_seconds = perf_counter() - started
    if not bool(support.any()):
        raise RuntimeError("selected teacher state unexpectedly has empty ring support")

    prior_probabilities = model.ring_system_template_log_prior.detach().cpu().exp()
    admitted_prior_mass = {
        name: float(prior_probabilities[support & mask].sum())
        for name, mask in group_masks.items()
    }
    total_admitted_prior_mass = float(prior_probabilities[support].sum())
    prior_given_support = prior_probabilities * support
    prior_given_support = prior_given_support / prior_given_support.sum()

    native_score = _score_state(model, state, native_time)
    if math.isclose(native_time, controlled_time, rel_tol=0.0, abs_tol=1e-15):
        controlled_score = native_score
    else:
        controlled_score = _score_state(model, state, controlled_time)

    matching_indices = tuple(
        int(index)
        for index in matching_ring_system_template_indices(
            teacher_action,
            model.ring_system_templates,
        )
    )
    matching_supported = tuple(index for index in matching_indices if support[index])

    def compact_score(score: dict[str, Any]) -> dict[str, Any]:
        probabilities = score["template_probabilities"]
        return {
            "time": score["time"],
            "ring_family_probability": score["ring_family_probability"],
            "conditional_model_group_mass": _conditional_group_masses(
                probabilities,
                support,
                group_masks,
            ),
            "conditional_teacher_template_probability": float(
                probabilities[list(matching_supported)].sum()
                if matching_supported
                else 0.0
            ),
        }

    legal_indices = tuple(
        int(index) for index in torch.nonzero(support, as_tuple=False).flatten()
    )
    topology_counts = {
        name: int((support & group_masks[name]).sum()) for name in TOPOLOGY_GROUPS
    }
    small_count = int((support & group_masks["small_ring_3_or_4"]).sum())
    return {
        "state_sha256": _state_sha256(state),
        "raw_catalog_template_count": len(model.ring_catalog.ring_system_templates),
        "deduplicated_production_template_count": len(support),
        "exact_support_wall_seconds": support_seconds,
        "legal_template_count": len(legal_indices),
        "legal_template_indices": legal_indices,
        "legal_topology_counts": topology_counts,
        "legal_small_ring_template_count": small_count,
        "tiny_or_degenerate_support_le_5": len(legal_indices) <= 5,
        "small_ring_only_support": small_count == len(legal_indices),
        "total_admitted_empirical_prior_mass": total_admitted_prior_mass,
        "admitted_empirical_prior_group_mass": admitted_prior_mass,
        "conditional_empirical_prior_group_mass": _conditional_group_masses(
            prior_given_support,
            support,
            group_masks,
        ),
        "matching_teacher_template_indices": matching_indices,
        "supported_matching_teacher_template_indices": matching_supported,
        "teacher_template_is_in_exact_support": bool(matching_supported),
        "native_position_score": compact_score(native_score),
        "controlled_original_time_score": compact_score(controlled_score),
    }


def _audit_chunk(
    checkpoint: str,
    rows: tuple[dict[str, Any], ...],
    torch_threads: int,
) -> tuple[dict[str, Any], ...]:
    torch.set_num_threads(torch_threads)
    model, _ = load_factorized_rollout_checkpoint(Path(checkpoint))
    templates = model.ring_system_templates
    topology_masks = {
        name: torch.tensor(
            tuple(template.topology_class == name for template in templates),
            dtype=torch.bool,
        )
        for name in TOPOLOGY_GROUPS
    }
    group_masks = {
        **topology_masks,
        "small_ring_3_or_4": undesirable_small_ring_mask(templates),
    }

    audited: list[dict[str, Any]] = []
    for row in rows:
        original_time = float(row["original_position"] / row["path_length"])
        early_time = float(row["early_position"] / row["path_length"])
        original = _support_and_scores(
            model,
            state=row["original_state"],
            native_time=original_time,
            controlled_time=original_time,
            teacher_action=row["teacher_action"],
            group_masks=group_masks,
        )
        early = _support_and_scores(
            model,
            state=row["early_state"],
            native_time=early_time,
            controlled_time=original_time,
            teacher_action=row["teacher_action"],
            group_masks=group_masks,
        )
        audited.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"original_state", "early_state", "teacher_action"}
            }
            | {
                "original": original,
                "earliest_commuting": early,
            }
        )
    return tuple(audited)


def _select_rows(paths_path: Path, row_count: int) -> tuple[dict[str, Any], ...]:
    payload = torch.load(paths_path, map_location="cpu", weights_only=False)
    records = tuple(payload["records"])
    selected: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        trace = record.path.trace
        eligible = tuple(
            (position, step)
            for position, step in enumerate(trace.steps)
            if step.rule_name == "ring_system_grow"
            and step.action.topology_class in ELIGIBLE_TOPOLOGIES
        )
        if not eligible:
            continue
        original_position, teacher_step = eligible[0]
        scheduled, report = schedule_priority_events_earliest(
            trace,
            system=record.path.system,
        )
        scheduled_positions = tuple(
            position
            for position, step in enumerate(scheduled.steps)
            if step is teacher_step
        )
        if len(scheduled_positions) != 1:
            raise RuntimeError("scheduler did not preserve the selected step object")
        early_position = scheduled_positions[0]
        scheduled_step = scheduled.steps[early_position]
        if scheduled_step is not teacher_step:
            raise RuntimeError("scheduler changed the selected teacher step")

        original_endpoint, original_states = execute_trace(
            trace.source,
            trace.steps,
            system=record.path.system,
            return_states=True,
        )
        scheduled_endpoint, scheduled_states = execute_trace(
            scheduled.source,
            scheduled.steps,
            system=record.path.system,
            return_states=True,
        )
        endpoint_exact = states_are_array_exact(original_endpoint, scheduled_endpoint)
        target_exact = states_are_array_exact(scheduled_endpoint, trace.target)
        original_successor = record.path.system.apply(
            original_states[original_position],
            teacher_step.rule_name,
            teacher_step.action,
        )
        early_successor = record.path.system.apply(
            scheduled_states[early_position],
            teacher_step.rule_name,
            teacher_step.action,
        )
        teacher_executable_original = states_are_array_exact(
            original_successor,
            original_states[original_position + 1],
        )
        teacher_executable_early = states_are_array_exact(
            early_successor,
            scheduled_states[early_position + 1],
        )

        selected.append(
            {
                "row_index": len(selected),
                "record_index": record_index,
                "target_key": record.target_key,
                "teacher_topology_class": teacher_step.action.topology_class,
                "path_length": len(trace.steps),
                "original_position": original_position,
                "early_position": early_position,
                "positions_moved_earlier": original_position - early_position,
                "schedule_attempted_swaps": report.attempted_swaps,
                "schedule_accepted_swaps": report.accepted_swaps,
                "scheduled_endpoint_array_exact": endpoint_exact,
                "scheduled_target_array_exact": target_exact,
                "teacher_action_executable_original": teacher_executable_original,
                "teacher_action_executable_early": teacher_executable_early,
                "original_state": original_states[original_position],
                "early_state": scheduled_states[early_position],
                "teacher_action": teacher_step.action,
            }
        )
        if len(selected) == row_count:
            break
    if len(selected) != row_count:
        raise ValueError(
            f"frozen paths supplied only {len(selected)} eligible rows, need {row_count}"
        )
    return tuple(selected)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return math.inf if numerator > 0.0 else 1.0
    return numerator / denominator


def _summarize(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    original_bridge_mass = tuple(
        float(row["original"]["admitted_empirical_prior_group_mass"]["bridged_or_fused"])
        for row in rows
    )
    early_bridge_mass = tuple(
        float(
            row["earliest_commuting"]["admitted_empirical_prior_group_mass"][
                "bridged_or_fused"
            ]
        )
        for row in rows
    )
    rowwise_ratios = tuple(
        _ratio(early, original)
        for original, early in zip(original_bridge_mass, early_bridge_mass)
    )
    original_degenerate = sum(
        bool(row["original"]["tiny_or_degenerate_support_le_5"]) for row in rows
    )
    early_degenerate = sum(
        bool(row["earliest_commuting"]["tiny_or_degenerate_support_le_5"])
        for row in rows
    )
    original_small_only = sum(
        bool(row["original"]["small_ring_only_support"]) for row in rows
    )
    early_small_only = sum(
        bool(row["earliest_commuting"]["small_ring_only_support"]) for row in rows
    )
    endpoint_gate = all(
        bool(row["scheduled_endpoint_array_exact"])
        and bool(row["scheduled_target_array_exact"])
        and bool(row["teacher_action_executable_original"])
        and bool(row["teacher_action_executable_early"])
        for row in rows
    )
    bridge_mass_gate = median(rowwise_ratios) >= 2.0
    degeneracy_gate = (
        original_degenerate > 0
        and early_degenerate <= 0.5 * original_degenerate
    )
    no_small_only_increase = early_small_only <= original_small_only
    passed = endpoint_gate and no_small_only_increase and (
        bridge_mass_gate or degeneracy_gate
    )
    return {
        "row_count": len(rows),
        "selection_rule": (
            "first bridged_or_fused-or-spiro ring grow from each of the first "
            "32 eligible frozen path records"
        ),
        "endpoint_and_teacher_executability_gate": endpoint_gate,
        "positions_moved_earlier": {
            "count_positive": sum(int(row["positions_moved_earlier"]) > 0 for row in rows),
            "median": float(median(int(row["positions_moved_earlier"]) for row in rows)),
            "minimum": min(int(row["positions_moved_earlier"]) for row in rows),
            "maximum": max(int(row["positions_moved_earlier"]) for row in rows),
        },
        "bridged_or_fused_admitted_prior_mass": {
            "original_median": float(median(original_bridge_mass)),
            "earliest_commuting_median": float(median(early_bridge_mass)),
            "median_rowwise_ratio": float(median(rowwise_ratios)),
            "threshold_ratio": 2.0,
            "gate_passed": bridge_mass_gate,
        },
        "tiny_or_degenerate_support_le_5": {
            "original_rows": original_degenerate,
            "earliest_commuting_rows": early_degenerate,
            "reduction_fraction": (
                float((original_degenerate - early_degenerate) / original_degenerate)
                if original_degenerate
                else None
            ),
            "threshold_reduction_fraction": 0.5,
            "gate_passed": degeneracy_gate,
        },
        "small_ring_only_support": {
            "original_rows": original_small_only,
            "earliest_commuting_rows": early_small_only,
            "no_increase_gate_passed": no_small_only_increase,
        },
        "support_wall_seconds": {
            "original_total": float(
                sum(row["original"]["exact_support_wall_seconds"] for row in rows)
            ),
            "earliest_commuting_total": float(
                sum(
                    row["earliest_commuting"]["exact_support_wall_seconds"]
                    for row in rows
                )
            ),
        },
        "preregistered_gate_passed": passed,
        "decision": (
            "PASS: earliest commutation materially improves legal multicycle support"
            if passed
            else "FAIL: earliest commutation does not meet the preregistered support gate"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.rows <= 0 or args.workers <= 0 or args.torch_threads <= 0:
        raise ValueError("rows, workers, and torch threads must be positive")

    selected = _select_rows(args.paths, args.rows)
    chunks = tuple(
        tuple(selected[offset:: args.workers]) for offset in range(args.workers)
    )
    chunks = tuple(chunk for chunk in chunks if chunk)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = tuple(
            executor.submit(
                _audit_chunk,
                str(args.checkpoint),
                chunk,
                args.torch_threads,
            )
            for chunk in chunks
        )
        rows = tuple(
            sorted(
                (row for future in futures for row in future.result()),
                key=lambda row: int(row["row_index"]),
            )
        )

    result = {
        "schema": "pancake_ring_earliest_commutation_falsification_v1",
        "inputs": {
            "paths": str(args.paths),
            "paths_sha256": _file_sha256(args.paths),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _file_sha256(args.checkpoint),
            "raw_catalog_template_count": rows[0]["original"][
                "raw_catalog_template_count"
            ],
            "deduplicated_production_template_count": rows[0]["original"][
                "deduplicated_production_template_count"
            ],
        },
        "constraints": {
            "training_launched": False,
            "model_artifacts_mutated": False,
            "checkpoint_weights_used": "selected best_state_dict",
        },
        "summary": _summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
