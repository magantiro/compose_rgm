#!/usr/bin/env python3
"""Frozen multi-state and rollout gate for pancake quotient distillation."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import torch

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.experiments.calibrated_rewrite_sampling import (
    ThinnedRateCalibrationSampler,
)
from compose_v4.experiments.canonical_successor_distillation import (
    CanonicalHistoryThinningSampler,
    PancakeQuotientTarget,
    build_calibrated_pancake_quotient_target,
    canonical_successor_distillation_excess,
    predict_factorized_quotient_rates,
)
from compose_v4.experiments.cnof_conditional import corpus_rollout_metrics
from compose_v4.experiments.parallel_tracelet_sampling import (
    sample_tracelet_ancestral_many,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_TO_INDEX,
    SampledRewriteMark,
    _legacy_prequotient_graft_tables,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import BondReroute


PHASES = {
    "early": (0.20, 0.20),
    "mid": (0.50, 0.50),
    "late": (1.00, 0.85),
}
SIZE_BINS = ("small", "medium", "large")


@dataclass(frozen=True)
class PanelRow:
    row_id: str
    split: str
    stratum: str
    record_index: int
    target_key: str
    path_length: int
    progress: int
    progress_fraction: float
    model_time: float
    source_key: str
    atom_count: int
    bond_count: int
    cycle_rank: int
    ring_status: str
    size_bin: str
    raw_graft_marks: int
    productive_graft_marks: int
    canonical_graft_successors: int


class _TerminalSampler:
    def sample_rewrite_mark(self, state, time, rng):
        del state, time, rng
        return SampledRewriteMark(0.0, "<TERMINAL>", None)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _structure(state: MolecularGraph) -> tuple[int, int, int, str, str]:
    vertices = np.flatnonzero(is_element(state.atom_types))
    bonds = state.bonds[np.ix_(vertices, vertices)]
    atom_count = len(vertices)
    bond_count = int(np.count_nonzero(np.triu(bonds, k=1)))
    cycle_rank = max(bond_count - atom_count + int(atom_count > 0), 0)
    ring_status = "ring" if cycle_rank else "acyclic"
    size_bin = (
        "small"
        if atom_count <= 20
        else "medium"
        if atom_count <= 30
        else "large"
    )
    return atom_count, bond_count, cycle_rank, ring_status, size_bin


def _stable_score(seed: int, *values: object) -> str:
    text = "|".join((str(seed), *(str(value) for value in values)))
    return hashlib.sha256(text.encode()).hexdigest()


def _select_panel(
    records: Sequence[object],
    *,
    seed: int,
) -> tuple[tuple[PanelRow, ...], dict[str, MolecularGraph]]:
    desired = {
        (phase, ring_status, size_bin)
        for phase, ring_statuses in (
            ("early", ("acyclic",)),
            ("mid", ("acyclic",)),
            ("late", ("ring", "acyclic")),
        )
        for ring_status in ring_statuses
        for size_bin in SIZE_BINS
    }
    candidates: dict[tuple[str, str, str], list[tuple[str, int, int, float]]] = {
        key: [] for key in desired
    }
    for record_index, record in enumerate(records):
        path = record.path
        for phase, (fraction, model_time) in PHASES.items():
            progress = (
                path.path_length
                if phase == "late"
                else int(round(fraction * path.path_length))
            )
            state = path.state_at(progress)
            _, _, _, ring_status, size_bin = _structure(state)
            key = (phase, ring_status, size_bin)
            if key not in desired:
                continue
            score = _stable_score(
                seed,
                record_index,
                progress,
                record.target_key,
            )
            candidates[key].append((score, record_index, progress, model_time))

    missing = {key: len(values) for key, values in candidates.items() if len(values) < 2}
    if missing:
        raise RuntimeError(f"path cache cannot fill panel strata: {missing}")

    rows: list[PanelRow] = []
    states: dict[str, MolecularGraph] = {}
    for stratum_key in sorted(candidates):
        phase, ring_status, size_bin = stratum_key
        selected = sorted(candidates[stratum_key])[:2]
        for offset, (_, record_index, progress, model_time) in enumerate(selected):
            record = records[record_index]
            state = record.path.state_at(progress)
            atom_count, bond_count, cycle_rank, actual_ring, actual_size = _structure(
                state
            )
            if (actual_ring, actual_size) != (ring_status, size_bin):
                raise RuntimeError("frozen panel descriptor changed during selection")
            support = prepare_factorized_mark_batch(
                (state,),
                (model_time,),
                (None,),
                (None,),
                (0.0,),
            )
            raw_graft_marks = int(_legacy_prequotient_graft_tables(state)[0].sum())
            productive_graft_marks = int(support.graft_mask[0].sum())
            groups = support.graft_successor_groups[0][support.graft_mask[0]]
            canonical_successors = int(torch.unique(groups).numel())
            split = "train" if offset == 0 else "heldout"
            row_id = f"{phase}-{ring_status}-{size_bin}-{split}"
            source_key = canonical_state_key(state)
            rows.append(
                PanelRow(
                    row_id=row_id,
                    split=split,
                    stratum="/".join(stratum_key),
                    record_index=record_index,
                    target_key=record.target_key,
                    path_length=record.path.path_length,
                    progress=progress,
                    progress_fraction=progress / max(record.path.path_length, 1),
                    model_time=model_time,
                    source_key=source_key,
                    atom_count=atom_count,
                    bond_count=bond_count,
                    cycle_rank=cycle_rank,
                    ring_status=ring_status,
                    size_bin=size_bin,
                    raw_graft_marks=raw_graft_marks,
                    productive_graft_marks=productive_graft_marks,
                    canonical_graft_successors=canonical_successors,
                )
            )
            states[row_id] = state
    return tuple(rows), states


def _relative_l1(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = right.abs().sum().clamp_min(1e-12)
    return float(((left - right).abs().sum() / denominator).detach())


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _evaluate_rows(
    model,
    rows: Sequence[PanelRow],
    states: dict[str, MolecularGraph],
    targets: dict[str, PancakeQuotientTarget],
) -> tuple[list[dict[str, object]], list[float]]:
    output = []
    runtimes = []
    graft_index = MARK_RULE_TO_INDEX["bond_reroute"]
    for row in rows:
        start = perf_counter()
        prediction = predict_factorized_quotient_rates(
            model,
            states[row.row_id],
            row.model_time,
        )
        runtimes.append(perf_counter() - start)
        target = targets[row.row_id]
        target_graft = float(target.productive_family_rates[graft_index])
        predicted_graft = float(prediction.family_rates[graft_index].detach())
        raw_graft = float(target.raw_family_rates[graft_index])
        output.append(
            {
                "row_id": row.row_id,
                "split": row.split,
                "family_relative_l1": _relative_l1(
                    prediction.family_rates,
                    target.productive_family_rates,
                ),
                "graft_relative_l1": (
                    None
                    if target_graft <= 1e-12
                    else _relative_l1(
                        prediction.graft_successor_rates,
                        target.graft_successor_rates,
                    )
                ),
                "student_total_hazard": float(prediction.total_hazard.detach()),
                "teacher_productive_total_hazard": float(
                    target.productive_total_hazard
                ),
                "student_productive_graft_rate": predicted_graft,
                "teacher_productive_graft_rate": target_graft,
                "teacher_raw_graft_rate": raw_graft,
                "student_graft_inflation": (
                    None if target_graft <= 1e-12 else predicted_graft / target_graft
                ),
                "naive_raw_graft_inflation": (
                    None if target_graft <= 1e-12 else raw_graft / target_graft
                ),
                "teacher_mass_balance_error": float(target.mass_balance_error),
                "teacher_relative_mass_balance_error": float(
                    target.mass_balance_error
                    / target.raw_total_hazard.abs().clamp_min(1e-12)
                ),
                "canonical_self_successor_groups": sum(
                    key == row.source_key for key in prediction.graft_successor_keys
                ),
            }
        )
    return output, runtimes


def _split_summary(rows: Sequence[dict[str, object]], split: str) -> dict[str, object]:
    selected = [row for row in rows if row["split"] == split]
    family = [float(row["family_relative_l1"]) for row in selected]
    graft = [
        float(value)
        for row in selected
        if (value := row["graft_relative_l1"]) is not None
    ]
    inflation = [
        float(value)
        for row in selected
        if (value := row["student_graft_inflation"]) is not None
    ]
    return {
        "rows": len(selected),
        "family_relative_l1": _quantiles(family),
        "graft_relative_l1": _quantiles(graft),
        "student_graft_inflation": _quantiles(inflation),
        "maximum_teacher_mass_balance_error": max(
            float(row["teacher_mass_balance_error"]) for row in selected
        ),
        "maximum_teacher_relative_mass_balance_error": max(
            float(row["teacher_relative_mass_balance_error"])
            for row in selected
        ),
        "canonical_self_successor_groups": sum(
            int(row["canonical_self_successor_groups"]) for row in selected
        ),
    }


def _reverse_thinning_check(
    rows: Sequence[PanelRow],
    states: dict[str, MolecularGraph],
) -> dict[str, object]:
    runtime = de_novo_rewrite_system()
    for row in rows:
        state = states[row.row_id]
        support = prepare_factorized_mark_batch(
            (state,),
            (row.model_time,),
            (None,),
            (None,),
            (0.0,),
        )
        for moved_tensor, target_tensor in torch.nonzero(
            support.graft_mask[0],
            as_tuple=False,
        ):
            moved, target = int(moved_tensor), int(target_tensor)
            removed = int(support.graft_remove_neighbors[0, moved, target])
            forward_action = BondReroute(
                a=moved,
                b=removed,
                u=moved,
                v=target,
            )
            successor = runtime.apply(state, "bond_reroute", forward_action)
            reverse_action = BondReroute(
                a=moved,
                b=target,
                u=moved,
                v=removed,
            )
            try:
                restored = runtime.apply(successor, "bond_reroute", reverse_action)
            except ValueError:
                continue
            if canonical_state_key(restored) != canonical_state_key(state):
                continue
            sampler = CanonicalHistoryThinningSampler(_TerminalSampler())
            sampler.begin_rollout_audit()
            accepted = sampler.resample_rewrite_mark_after_event(
                state,
                row.model_time,
                np.random.default_rng(1),
                SampledRewriteMark(7.0, "bond_reroute", forward_action),
            )
            rejected = sampler.resample_rewrite_mark_after_event(
                successor,
                row.model_time,
                np.random.default_rng(2),
                SampledRewriteMark(7.0, "bond_reroute", reverse_action),
            )
            return {
                "row_id": row.row_id,
                "source_key": canonical_state_key(state),
                "successor_key": canonical_state_key(successor),
                "forward_committed": accepted.action is not None,
                "reverse_rule_name": rejected.rule_name,
                "reverse_action_is_none": rejected.action is None,
                "hazard_preserved": float(rejected.total_hazard) == 7.0,
                "audit": sampler.end_rollout_audit(),
            }
    raise RuntimeError("frozen panel has no exactly reversible productive Graft")


def _event_family_tv(left: dict[str, float], right: dict[str, float]) -> float:
    names = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(name, 0.0)) - float(right.get(name, 0.0))) for name in names)


def _rollout_gate(
    student,
    teacher,
    *,
    source_prior,
    reference_smiles: tuple[str, ...],
    samples: int,
    workers: int,
    seed: int,
) -> dict[str, object]:
    profile = {
        "samples": samples,
        "workers": workers,
        "seed": seed,
        "n_slots": 40,
        "operational_horizon": 16.0,
        "time_step": 0.1,
        "max_events": 128,
    }
    baseline_sampler = ThinnedRateCalibrationSampler(
        teacher,
        family_log_rate_adjustments=(("atom_delete", -0.5),),
        small_ring_log_rate_adjustment=-1.5,
    )
    student_sampler = CanonicalHistoryThinningSampler(student)

    def sample(model, label: str):
        start = perf_counter()
        rollouts = sample_tracelet_ancestral_many(
            model,
            seed=seed,
            samples=samples,
            workers=workers,
            n_slots=40,
            operational_horizon=16.0,
            time_step=0.1,
            max_events=128,
            torch_threads_per_worker=1,
            source_prior=source_prior,
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
        return rollouts, {
            "label": label,
            "elapsed_seconds": elapsed,
            "metrics": metrics,
            "control_diagnostics": controls,
        }

    _, baseline = sample(baseline_sampler, "calibrated_pancake")
    _, candidate = sample(student_sampler, "panel_distilled_quotient")
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    control_rows = candidate["control_diagnostics"]
    accepted = sum(int(row.get("accepted_events", 0)) for row in control_rows)
    reverse_rejections = sum(
        int(row.get("immediate_backtrack_rejections", 0)) for row in control_rows
    )
    self_rejections = sum(int(row.get("self_rejections", 0)) for row in control_rows)
    family_tv = _event_family_tv(
        candidate_metrics["event_rule_fractions"],
        baseline_metrics["event_rule_fractions"],
    )
    baseline_mean_events = float(baseline_metrics["mean_events"])
    candidate_mean_events = float(candidate_metrics["mean_events"])
    safety_checks = {
        "candidate_valid_fraction_eq_1": candidate_metrics["valid_fraction"] == 1.0,
        "candidate_connected_fraction_eq_1": (
            candidate_metrics["connected_or_null_fraction"] == 1.0
        ),
        "candidate_all_step_valid_eq_1": (
            candidate_metrics["trajectory_diagnostics"][
                "all_step_valid_trajectory_fraction"
            ]
            == 1.0
        ),
        "candidate_committed_self_events_eq_0": (
            candidate_metrics["trajectory_diagnostics"]["canonical_self_event_count"]
            == 0
        ),
        "candidate_committed_backtracks_eq_0": (
            candidate_metrics["trajectory_diagnostics"]["immediate_backtrack_count"]
            == 0
        ),
        "candidate_exhaustion_eq_0": (
            candidate_metrics["event_budget_exhaustion_fraction"] == 0.0
        ),
    }
    behavioral_checks = {
        "virtual_reverse_fraction_le_1pct": (
            reverse_rejections / max(accepted + reverse_rejections, 1) <= 0.01
        ),
        "mean_atoms_within_3_of_baseline": (
            abs(
                float(candidate_metrics["mean_atoms"])
                - float(baseline_metrics["mean_atoms"])
            )
            <= 3.0
        ),
        "event_family_tv_le_0p20": family_tv <= 0.20,
        "mean_events_ratio_0p5_to_1p5": (
            0.5
            <= candidate_mean_events / max(baseline_mean_events, 1e-12)
            <= 1.5
        ),
    }
    checks = {**safety_checks, **behavioral_checks}
    safety_passed = all(safety_checks.values())
    behavior_passed = all(behavioral_checks.values())

    no_graft_diagnostic = None
    if safety_passed and not behavior_passed:
        no_graft_student = copy.deepcopy(student)
        no_graft_student.disabled_sampling_rule_names = frozenset({"bond_reroute"})
        _, no_graft = sample(
            CanonicalHistoryThinningSampler(no_graft_student),
            "panel_distilled_quotient_no_graft_diagnostic",
        )
        no_graft_metrics = no_graft["metrics"]
        no_graft_trajectory = no_graft_metrics["trajectory_diagnostics"]
        candidate_trajectory = candidate_metrics["trajectory_diagnostics"]

        def selected_metrics(metrics, trajectory):
            return {
                "valid_fraction": metrics["valid_fraction"],
                "all_step_valid_trajectory_fraction": trajectory[
                    "all_step_valid_trajectory_fraction"
                ],
                "mean_atoms": metrics["mean_atoms"],
                "mean_cycle_rank": metrics["mean_cycle_rank"],
                "cycle_rank_total_variation": metrics[
                    "cycle_rank_total_variation"
                ],
                "mean_unique_state_fraction": trajectory[
                    "mean_unique_state_fraction"
                ],
                "delete_to_one_then_regrow_fraction": trajectory[
                    "delete_to_one_then_regrow_fraction"
                ],
                "collapsed_from_larger_source_to_one_atom_fraction": trajectory[
                    "collapsed_from_larger_source_to_one_atom_fraction"
                ],
                "immediate_backtrack_count": trajectory[
                    "immediate_backtrack_count"
                ],
                "trajectories_with_immediate_backtrack_fraction": trajectory[
                    "trajectories_with_immediate_backtrack_fraction"
                ],
                "event_rule_fractions": metrics["event_rule_fractions"],
            }

        no_graft_diagnostic = {
            "trigger": (
                "base quotient arm passed all safety checks but failed at least "
                "one behavioral check"
            ),
            "matched_profile": profile,
            "not_a_promotion_candidate": True,
            "base_quotient": selected_metrics(
                candidate_metrics,
                candidate_trajectory,
            ),
            "no_graft": selected_metrics(no_graft_metrics, no_graft_trajectory),
            "full_no_graft_arm": no_graft,
        }
    return {
        "profile": profile,
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "event_family_total_variation": family_tv,
            "mean_atoms_delta": float(candidate_metrics["mean_atoms"])
            - float(baseline_metrics["mean_atoms"]),
            "mean_events_ratio": candidate_mean_events
            / max(baseline_mean_events, 1e-12),
            "candidate_virtual_self_rejections": self_rejections,
            "candidate_virtual_reverse_rejections": reverse_rejections,
            "candidate_accepted_events": accepted,
            "candidate_virtual_reverse_fraction": reverse_rejections
            / max(accepted + reverse_rejections, 1),
        },
        "safety_checks": safety_checks,
        "behavioral_checks": behavioral_checks,
        "safety_passed": safety_passed,
        "behavior_passed": behavior_passed,
        "checks": checks,
        "passed": all(checks.values()),
        "no_graft_diagnostic": no_graft_diagnostic,
    }


def _save_qualified_checkpoint(
    path: Path,
    *,
    student,
    checkpoint_payload: dict[str, object],
    source_checkpoint_sha256: str,
    manifest_path: Path,
) -> None:
    """Save a rollout-loadable selected checkpoint without recovery state."""

    retained_keys = {
        "ring_catalog",
        "tree_source_prior",
        "hidden_dim",
        "message_passing_steps",
        "training_backend",
        "source_prior",
        "ring_electronic_mode",
        "rate_factorization",
        "property_condition_dim",
        "empirical_mark_prior_mode",
        "empirical_mark_priors",
        "ring_family_mass_mode",
        "selected_validation",
    }
    payload = {
        key: checkpoint_payload[key]
        for key in retained_keys
        if key in checkpoint_payload
    }
    payload.update(
        {
            "format": "compose_v4_canonical_successor_distilled_checkpoint_v1",
            "checkpoint_kind": "canonical_successor_distilled_selected",
            "state_dict": {
                name: value.detach().cpu()
                for name, value in student.state_dict().items()
            },
            "property_condition_dim": 0,
            "empirical_mark_prior_mode": "none",
            "ring_family_mass_mode": "boolean",
            "canonical_successor_execution": True,
            "molecular_self_transitions_virtualized": True,
            "required_sampling_wrapper": "CanonicalHistoryThinningSampler",
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "qualification_manifest_sha256": _file_sha256(manifest_path),
        }
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-paths", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--rollout-samples", type=int, default=20)
    parser.add_argument("--rollout-workers", type=int, default=4)
    parser.add_argument(
        "--conditional-handoff",
        type=Path,
        default=Path("diagnostics/canonical_successor_backbone_qualification.json"),
    )
    parser.add_argument("--skip-rollout", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.learning_rate <= 0.0:
        raise ValueError("training controls must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    path_payload = torch.load(
        args.validation_paths,
        map_location="cpu",
        weights_only=False,
    )
    records = tuple(path_payload["records"])
    rows, states = _select_panel(records, seed=args.seed)
    teacher, checkpoint_payload = load_factorized_rollout_checkpoint(args.checkpoint)
    if teacher.ring_family_mass_mode != "boolean":
        raise ValueError("qualification forbids the rejected P1/P2 topology scalar")
    student = copy.deepcopy(teacher).train()
    student.virtualize_legacy_self_grafts = False

    manifest = {
        "format": "compose_v4_pancake_quotient_heldout_panel_v1",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "selected_step": (checkpoint_payload.get("selected_validation") or {}).get(
            "selected_step"
        ),
        "validation_paths": str(args.validation_paths),
        "validation_paths_sha256": _file_sha256(args.validation_paths),
        "validation_paths_signature_fingerprint": path_payload.get(
            "signature_fingerprint"
        ),
        "selection": {
            "phase_progress": PHASES,
            "size_bins": {"small": "<=20", "medium": "21..30", "large": ">30"},
            "strata": (
                "early/mid acyclic plus late ring/acyclic crossed with size bin"
            ),
            "two_stable_sha256_ranked_rows_per_stratum": True,
            "first_row_train_second_row_heldout": True,
        },
        "rows": tuple(asdict(row) for row in rows),
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "trainable_prefixes": ("total_hazard_head.", "family_head."),
            "ring_family_mass_mode": teacher.ring_family_mass_mode,
            "small_ring_calibration_in_student": False,
            "joint_ring_context_in_student": False,
        },
    }
    manifest_path = args.output_dir / "panel_manifest.json"
    _atomic_json(manifest_path, manifest)

    targets = {}
    target_times = []
    for row in rows:
        start = perf_counter()
        targets[row.row_id] = build_calibrated_pancake_quotient_target(
            teacher,
            states[row.row_id],
            row.model_time,
        )
        target_times.append(perf_counter() - start)

    initial_rows, initial_prediction_times = _evaluate_rows(
        student,
        rows,
        states,
        targets,
    )
    train_rows = tuple(row for row in rows if row.split == "train")
    rng = np.random.default_rng(args.seed)
    optimizer = torch.optim.Adam(
        (
            *student.total_hazard_head.parameters(),
            *student.family_head.parameters(),
        ),
        lr=args.learning_rate,
    )
    losses = []
    train_start = perf_counter()
    for _ in range(args.steps):
        indices = rng.choice(
            len(train_rows),
            size=min(args.batch_size, len(train_rows)),
            replace=False,
        )
        batch_losses = []
        for index in indices:
            row = train_rows[int(index)]
            prediction = predict_factorized_quotient_rates(
                student,
                states[row.row_id],
                row.model_time,
            )
            batch_losses.append(
                canonical_successor_distillation_excess(
                    prediction,
                    targets[row.row_id],
                )
            )
        loss = torch.stack(batch_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    training_seconds = perf_counter() - train_start
    student.eval()

    final_rows, final_prediction_times = _evaluate_rows(
        student,
        rows,
        states,
        targets,
    )
    reverse_check = _reverse_thinning_check(rows, states)
    initial = {
        "train": _split_summary(initial_rows, "train"),
        "heldout": _split_summary(initial_rows, "heldout"),
    }
    final = {
        "train": _split_summary(final_rows, "train"),
        "heldout": _split_summary(final_rows, "heldout"),
    }
    heldout = final["heldout"]
    rate_checks = {
        "panel_rows_eq_24": len(rows) == 24,
        "train_heldout_rows_eq_12_each": (
            final["train"]["rows"] == 12 and heldout["rows"] == 12
        ),
        "target_relative_mass_balance_max_le_1e-6": (
            heldout["maximum_teacher_relative_mass_balance_error"] <= 1e-6
        ),
        "canonical_self_successor_groups_eq_0": (
            heldout["canonical_self_successor_groups"] == 0
        ),
        "heldout_family_relative_l1_p95_le_0p25": (
            heldout["family_relative_l1"]["p95"] <= 0.25
        ),
        "heldout_graft_relative_l1_p95_le_0p50": (
            heldout["graft_relative_l1"]["p95"] <= 0.50
        ),
        "heldout_graft_inflation_max_le_2": (
            heldout["student_graft_inflation"]["max"] <= 2.0
        ),
        "warm_prediction_runtime_p95_le_0p5s": (
            _quantiles(final_prediction_times)["p95"] <= 0.5
        ),
        "reverse_jump_is_virtual": (
            reverse_check["reverse_rule_name"] == "<VIRTUAL_IMMEDIATE_BACKTRACK>"
            and reverse_check["reverse_action_is_none"]
            and reverse_check["hazard_preserved"]
        ),
        "p1p2_topology_scalar_absent": teacher.ring_family_mass_mode == "boolean",
    }
    rate_passed = all(rate_checks.values())
    rate_metrics = {
        "format": "compose_v4_pancake_quotient_heldout_rate_gate_v1",
        "manifest": str(manifest_path),
        "initial": initial,
        "final": final,
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "reverse_thinning_check": reverse_check,
        "runtime": {
            "target_build_seconds": _quantiles(target_times),
            "initial_prediction_seconds": _quantiles(initial_prediction_times),
            "warm_prediction_seconds": _quantiles(final_prediction_times),
            "training_seconds": training_seconds,
            "seconds_per_update": training_seconds / args.steps,
            "projected_500_update_seconds": 500.0 * training_seconds / args.steps,
        },
        "optimization": {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "minimum_loss": min(losses),
        },
        "checks": rate_checks,
        "passed": rate_passed,
        "corpus_scale_training_authorized": False,
    }
    rate_metrics_path = args.output_dir / "rate_metrics.json"
    _atomic_json(rate_metrics_path, rate_metrics)

    head_delta = {
        name: value.detach().cpu()
        for name, value in student.state_dict().items()
        if name.startswith(("total_hazard_head.", "family_head."))
    }
    delta_path = args.output_dir / "distilled_head_delta.pt"
    torch.save(
        {
            "format": "compose_v4_pancake_quotient_head_delta_v1",
            "base_checkpoint_sha256": manifest["checkpoint_sha256"],
            "panel_manifest_sha256": _file_sha256(manifest_path),
            "state_dict": head_delta,
        },
        delta_path,
    )

    rollout = None
    if rate_passed and not args.skip_rollout:
        reference_smiles = tuple(dict.fromkeys(record.target_key for record in records))
        rollout = _rollout_gate(
            student,
            teacher,
            source_prior=checkpoint_payload["tree_source_prior"],
            reference_smiles=reference_smiles,
            samples=args.rollout_samples,
            workers=args.rollout_workers,
            seed=args.seed + 1,
        )
        _atomic_json(args.output_dir / "rollout_metrics.json", rollout)

    qualification_passed = bool(
        rate_passed and rollout is not None and rollout["passed"]
    )
    qualified_checkpoint_path = args.output_dir / "qualified_checkpoint.pt"
    if qualification_passed:
        _save_qualified_checkpoint(
            qualified_checkpoint_path,
            student=student,
            checkpoint_payload=checkpoint_payload,
            source_checkpoint_sha256=str(manifest["checkpoint_sha256"]),
            manifest_path=manifest_path,
        )
    else:
        qualified_checkpoint_path.unlink(missing_ok=True)
    qualified_checkpoint_sha256 = (
        _file_sha256(qualified_checkpoint_path)
        if qualified_checkpoint_path.exists() and qualification_passed
        else None
    )

    decision = {
        "format": "compose_v4_pancake_quotient_qualification_decision_v1",
        "decision_use": "bounded_nonpromotional_qualification",
        "rate_gate_passed": rate_passed,
        "rollout_gate_run": rollout is not None,
        "rollout_gate_passed": None if rollout is None else rollout["passed"],
        "qualification_passed": qualification_passed,
        "corpus_scale_training_authorized": False,
        "ring_calibration_authorized": False,
        "no_graft_diagnostic_policy": {
            "default_model": False,
            "run_only_if": (
                "the matched 20-rollout quotient arm passes all safety checks "
                "but fails at least one behavioral check"
            ),
            "matched_seed_and_operational_budget": True,
            "reported_axes": (
                "topology/cycle rank",
                "delete-to-one then regrow",
                "size",
                "unique-state fraction",
                "backtracking",
                "rings",
                "validity",
            ),
            "ran": bool(
                rollout is not None
                and rollout.get("no_graft_diagnostic") is not None
            ),
        },
        "next_step": (
            "inspect failed rate checks; do not roll out"
            if not rate_passed
            else "inspect failed rollout checks; do not scale"
            if rollout is not None and not rollout["passed"]
            else "rate gate passed; rollout skipped by request"
            if rollout is None
            else "bounded gate passed; seek explicit authorization for a larger cached pilot"
        ),
        "artifacts": {
            "manifest": str(manifest_path),
            "rate_metrics": str(rate_metrics_path),
            "head_delta": str(delta_path),
            "qualified_checkpoint": (
                str(qualified_checkpoint_path) if qualification_passed else None
            ),
            "rollout_metrics": (
                None if rollout is None else str(args.output_dir / "rollout_metrics.json")
            ),
        },
    }
    decision_path = args.output_dir / "decision.json"
    _atomic_json(decision_path, decision)

    conditional_handoff = {
        "format": "compose_v4_canonical_successor_backbone_qualification_v1",
        "decision": {
            "qualification_passed": qualification_passed,
            "decision_path": str(decision_path.resolve()),
        },
        "checkpoint": {
            "path": (
                str(qualified_checkpoint_path.resolve())
                if qualification_passed
                else None
            ),
            "sha256": qualified_checkpoint_sha256,
            "weight_source": (
                "step6250_pancake_plus_panel_quotient_distilled_family_hazard_heads"
            ),
            "source_checkpoint_sha256": manifest["checkpoint_sha256"],
            "property_condition_dim": 0,
        },
        "execution": {
            "canonical_successor_execution": True,
            "molecular_self_transitions_virtualized": True,
            "required_sampling_wrapper": "CanonicalHistoryThinningSampler",
        },
        "model_modes": {
            "empirical_mark_prior_mode": str(
                getattr(teacher, "empirical_mark_prior_mode", "none")
            ),
            "ring_family_mass_mode": str(teacher.ring_family_mass_mode),
            "p1_p2_imported": False,
        },
        "evidence": {
            "multi_state_panel": {
                "passed": len(rows) == 24,
                "state_count": len(rows),
                "manifest": str(manifest_path.resolve()),
            },
            "heldout_rate_gate": {
                "passed": rate_passed,
                "metrics": str(rate_metrics_path.resolve()),
            },
            "rollout_gate": {
                "passed": bool(rollout is not None and rollout["passed"]),
                "run": rollout is not None,
                "metrics": (
                    str((args.output_dir / "rollout_metrics.json").resolve())
                    if rollout is not None
                    else None
                ),
            },
        },
    }
    _atomic_json(args.conditional_handoff, conditional_handoff)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
