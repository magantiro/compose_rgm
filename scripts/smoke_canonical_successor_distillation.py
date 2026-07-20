#!/usr/bin/env python3
"""Timed one-state smoke for calibrated-pancake quotient distillation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.canonical_successor_distillation import (
    build_calibrated_pancake_quotient_target,
    canonical_successor_distillation_excess,
    predict_factorized_quotient_rates,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_TO_INDEX,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog


def _relative_l1(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left - right).abs().sum()
    denominator = right.abs().sum().clamp_min(1e-12)
    return float((numerator / denominator).detach())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_teacher(
    checkpoint: Path | None,
    *,
    hidden_dim: int,
) -> tuple[FactorizedTraceletRateModel, str, int]:
    if checkpoint is None:
        torch.manual_seed(20260720)
        model = FactorizedTraceletRateModel(
            TypedRingCatalog((), (), ()),
            hidden_dim=hidden_dim,
            message_passing_steps=1,
            ring_family_mass_mode="boolean",
        ).eval()
        return model, "deterministic_tiny_teacher", 12

    from evaluate_tracelet_rollouts import (
        load_factorized_rollout_checkpoint,
    )

    model, payload = load_factorized_rollout_checkpoint(checkpoint)
    if model.ring_family_mass_mode != "boolean":
        raise ValueError("smoke forbids the rejected P1/P2 topology scalar")
    selected = payload.get("selected_validation") or {}
    label = (
        f"{checkpoint}:step={selected.get('selected_step', 'unknown')}"
    )
    return model, label, 40


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--smiles", default="CCCCCCCC")
    parser.add_argument("--time", type=float, default=0.5)
    args = parser.parse_args()
    if args.steps <= 0 or args.learning_rate <= 0.0:
        raise ValueError("steps and learning rate must be positive")

    teacher, teacher_label, n_slots = _load_teacher(
        args.checkpoint,
        hidden_dim=args.hidden_dim,
    )
    state = pad_molecular_graph(
        smiles_to_molecular_graph(args.smiles),
        n_slots,
    )
    student = copy.deepcopy(teacher).train()
    if hasattr(student, "virtualize_legacy_self_grafts"):
        student.virtualize_legacy_self_grafts = False

    target_start = perf_counter()
    target = build_calibrated_pancake_quotient_target(
        teacher,
        state,
        args.time,
    )
    target_seconds = perf_counter() - target_start
    initial = predict_factorized_quotient_rates(student, state, args.time)
    initial_excess = float(
        canonical_successor_distillation_excess(initial, target).detach()
    )
    initial_family_relative_l1 = _relative_l1(
        initial.family_rates,
        target.productive_family_rates,
    )
    initial_graft_relative_l1 = _relative_l1(
        initial.graft_successor_rates,
        target.graft_successor_rates,
    )

    optimizer = torch.optim.Adam(
        (
            *student.total_hazard_head.parameters(),
            *student.family_head.parameters(),
        ),
        lr=args.learning_rate,
    )
    train_start = perf_counter()
    for _ in range(args.steps):
        prediction = predict_factorized_quotient_rates(student, state, args.time)
        loss = canonical_successor_distillation_excess(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    training_seconds = perf_counter() - train_start

    final = predict_factorized_quotient_rates(student, state, args.time)
    final_excess = float(
        canonical_successor_distillation_excess(final, target).detach()
    )
    final_family_relative_l1 = _relative_l1(
        final.family_rates,
        target.productive_family_rates,
    )
    final_graft_relative_l1 = _relative_l1(
        final.graft_successor_rates,
        target.graft_successor_rates,
    )
    graft_index = MARK_RULE_TO_INDEX["bond_reroute"]
    delete_index = MARK_RULE_TO_INDEX["atom_delete"]
    excess_reduction = 1.0 - final_excess / max(initial_excess, 1e-12)
    seconds_per_update = training_seconds / args.steps
    projected_500_update_seconds = 500.0 * seconds_per_update
    gate_checks = {
        "mass_balance_error_le_1e-6": float(target.mass_balance_error) <= 1e-6,
        "excess_reduction_ge_99pct": excess_reduction >= 0.99,
        "family_relative_l1_le_0p5pct": final_family_relative_l1 <= 0.005,
        "graft_relative_l1_le_0p5pct": final_graft_relative_l1 <= 0.005,
        "projected_500_updates_le_10min": projected_500_update_seconds <= 600.0,
        "p1p2_topology_scalar_absent": student.ring_family_mass_mode == "boolean",
    }
    report = {
        "format": "compose_v4_canonical_successor_distillation_smoke_v1",
        "decision_use": "cheap_nonpromotional_launch_gate",
        "teacher": teacher_label,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "checkpoint_sha256": (
            None if args.checkpoint is None else _file_sha256(args.checkpoint)
        ),
        "state": {
            "smiles": args.smiles,
            "n_slots": n_slots,
            "time": args.time,
        },
        "objective": {
            "kind": "productive_family_poisson_kl_plus_canonical_graft_conditional",
            "atom_delete_log_rate_adjustment": -0.5,
            "small_ring_calibration_included": False,
            "joint_ring_context_included": False,
            "ring_family_mass_mode": student.ring_family_mass_mode,
        },
        "teacher_mass": {
            "raw_total_hazard": float(target.raw_total_hazard),
            "productive_total_hazard": float(target.productive_total_hazard),
            "virtual_total_rate": float(target.virtual_total_rate),
            "mass_balance_error": float(target.mass_balance_error),
            "raw_graft_family_rate": float(target.raw_family_rates[graft_index]),
            "productive_graft_family_rate": float(
                target.productive_family_rates[graft_index]
            ),
            "virtual_graft_self_rate": float(target.virtual_family_rates[graft_index]),
            "raw_delete_family_rate": float(target.raw_family_rates[delete_index]),
            "productive_delete_family_rate": float(
                target.productive_family_rates[delete_index]
            ),
            "raw_graft_marks": target.raw_graft_mark_count,
            "productive_graft_marks": target.productive_graft_mark_count,
            "canonical_graft_successors": len(target.graft_successor_keys),
            "graft_alias_counts": tuple(int(value) for value in target.graft_alias_counts),
        },
        "optimization": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "initial_excess_poisson_kl": initial_excess,
            "final_excess_poisson_kl": final_excess,
            "excess_reduction_fraction": excess_reduction,
            "initial_family_relative_l1": initial_family_relative_l1,
            "final_family_relative_l1": final_family_relative_l1,
            "initial_graft_relative_l1": initial_graft_relative_l1,
            "final_graft_relative_l1": final_graft_relative_l1,
            "final_total_hazard": float(final.total_hazard.detach()),
            "target_total_hazard": float(target.productive_total_hazard),
        },
        "throughput": {
            "target_build_seconds": target_seconds,
            "training_seconds": training_seconds,
            "seconds_per_update": seconds_per_update,
            "projected_500_update_seconds": projected_500_update_seconds,
        },
        "gate": {
            "checks": gate_checks,
            "cheap_pilot_gate_passed": all(gate_checks.values()),
            "corpus_scale_training_authorized": False,
            "note": (
                "A corpus launch additionally requires a cached-state benchmark "
                "on the real pancake checkpoint and a fixed heldout family/backtrack gate."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
