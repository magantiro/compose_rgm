"""Audit stochastic QED control on common valid-successor proposal sets."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from math import exp
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import (
    empty_molecular_graph,
    is_connected_or_null,
    is_valid_state,
    pad_molecular_graph,
)
from compose_v4.experiments.guided_rewrite_sampling import (
    CachedStateScorer,
    CONTROLLED_MARK_EQUATION,
    ProposalGuidedRewriteSampler,
    QEDTargetDistancePotential,
    identity_score_potential,
    kl_tilted_proposal_probabilities,
    qed_state_score,
)
from compose_v4.experiments.molecular_property_conditioning import (
    PropertyConditionNormalizer,
)
from compose_v4.experiments.property_conditioned_sampling import (
    PropertyConditionedRewriteSampler,
)
from compose_v4.rewrite.kernel import canonical_state_key
from scripts.evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint


CONTROL_EQUATION = CONTROLLED_MARK_EQUATION


@dataclass(frozen=True)
class SavedState:
    rollout_index: int
    state_index: int
    canonical_key: str
    operational_time: float
    model_time: float
    state: MolecularGraph


def _atomic_json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _normalizer_from_checkpoint(payload: dict[str, object]) -> PropertyConditionNormalizer:
    raw = payload.get("property_conditioning")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint lacks property-conditioning metadata")
    return PropertyConditionNormalizer(
        names=tuple(str(value) for value in raw["names"]),
        means=tuple(float(value) for value in raw["means"]),
        standard_deviations=tuple(float(value) for value in raw["standard_deviations"]),
    )


def _state_from_key(key: str, *, n_slots: int) -> MolecularGraph:
    if key == "<NULL>":
        return empty_molecular_graph(n_slots)
    return pad_molecular_graph(smiles_to_molecular_graph(key), n_slots)


def saved_rollout_states(
    rollout_cache: Path,
    *,
    condition_label: str,
    n_slots: int,
) -> tuple[SavedState, ...]:
    payload = torch.load(rollout_cache, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("rollouts"), dict):
        raise ValueError("rollout cache has the wrong structure")
    rollouts = payload["rollouts"].get(condition_label)
    if not isinstance(rollouts, (tuple, list)) or not rollouts:
        raise KeyError(f"rollout cache lacks condition {condition_label!r}")
    rows: list[SavedState] = []
    for rollout_index, rollout in enumerate(rollouts):
        diagnostics = rollout.diagnostics
        keys = tuple(str(value) for value in diagnostics.canonical_state_keys)
        valid = tuple(bool(value) for value in diagnostics.state_valid)
        connected = tuple(bool(value) for value in diagnostics.state_connected_or_null)
        event_times = (0.0, *(float(value) for value in rollout.event_times))
        if not (len(keys) == len(valid) == len(connected) == len(event_times)):
            raise ValueError("rollout state diagnostics do not align")
        for state_index, (key, is_valid, is_connected, operational_time) in enumerate(
            zip(keys, valid, connected, event_times)
        ):
            if not is_valid or not is_connected:
                raise ValueError("saved rollout contains an invalid or disconnected state")
            state = _state_from_key(key, n_slots=n_slots)
            if not is_valid_state(state) or not is_connected_or_null(state):
                raise ValueError("reconstructed saved state failed executor invariants")
            if canonical_state_key(state) != key:
                raise ValueError("saved-state reconstruction changed canonical identity")
            rows.append(
                SavedState(
                    rollout_index=rollout_index,
                    state_index=state_index,
                    canonical_key=key,
                    operational_time=operational_time,
                    model_time=1.0 - exp(-operational_time),
                    state=state,
                )
            )
    return tuple(rows)


def _select_with_uniform(probabilities: np.ndarray, uniform: float) -> int:
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    return min(int(np.searchsorted(cumulative, uniform, side="right")), len(cumulative) - 1)


def _arm_summary(
    records: list[dict[str, object]],
    *,
    target_qed: float,
    tolerance: float,
) -> dict[str, object]:
    qeds = [float(row["qed"]) for row in records]
    distances = [abs(value - target_qed) for value in qeds]
    families = Counter(str(row["rule_name"]) for row in records)
    unique_successors = {str(row["successor_key"]) for row in records}
    return {
        "selections": len(records),
        "mean_qed": mean(qeds),
        "qed_standard_deviation": pstdev(qeds),
        "minimum_qed": min(qeds),
        "maximum_qed": max(qeds),
        "mean_absolute_target_error": mean(distances),
        "within_tolerance": sum(value <= tolerance for value in distances),
        "qed_at_least_target": sum(value >= target_qed for value in qeds),
        "unique_selected_successors": len(unique_successors),
        "unique_selected_fraction": len(unique_successors) / len(records),
        "family_counts": dict(sorted(families.items())),
        "family_fractions": {
            name: count / len(records) for name, count in sorted(families.items())
        },
        "all_selected_valid": all(bool(row["valid"]) for row in records),
        "all_selected_connected": all(bool(row["connected"]) for row in records),
    }


def _paired_summary(
    control: list[dict[str, object]],
    candidate: list[dict[str, object]],
    *,
    target_qed: float,
) -> dict[str, object]:
    if len(control) != len(candidate):
        raise ValueError("paired arms do not align")
    qed_deltas = [
        float(candidate_row["qed"]) - float(control_row["qed"])
        for control_row, candidate_row in zip(control, candidate)
    ]
    error_improvements = [
        abs(float(control_row["qed"]) - target_qed)
        - abs(float(candidate_row["qed"]) - target_qed)
        for control_row, candidate_row in zip(control, candidate)
    ]
    return {
        "mean_paired_qed_delta": mean(qed_deltas),
        "qed_improved_states": sum(value > 0.0 for value in qed_deltas),
        "qed_tied_states": sum(value == 0.0 for value in qed_deltas),
        "mean_paired_target_error_improvement": mean(error_improvements),
        "target_error_improved_states": sum(value > 0.0 for value in error_improvements),
        "target_error_tied_states": sum(value == 0.0 for value in error_improvements),
    }


def evaluate_common_successor_sets(
    *,
    model: object,
    checkpoint: dict[str, object],
    saved_states: tuple[SavedState, ...],
    target_qed: float,
    state_samples: int,
    proposals_per_state: int,
    beta: float,
    seed: int,
    tolerance: float,
    use_target_condition: bool,
) -> dict[str, object]:
    if state_samples <= 0 or proposals_per_state <= 0:
        raise ValueError("state and proposal counts must be positive")
    if state_samples > len(saved_states):
        raise ValueError("requested more states than the rollout cache contains")
    normalizer = _normalizer_from_checkpoint(checkpoint)
    if normalizer.names != ("qed",):
        raise ValueError("QED successor guidance requires a QED-conditioned checkpoint")
    property_values = (
        normalizer.transform((target_qed,)) if use_target_condition else (0.0,)
    )
    base_sampler = PropertyConditionedRewriteSampler(
        model,
        property_values=property_values,
        property_mask=(use_target_condition,),
    )
    scorer = CachedStateScorer(qed_state_score)
    target_potential = QEDTargetDistancePotential(target_qed=target_qed)
    proposal_sampler = ProposalGuidedRewriteSampler(
        base_sampler,
        scorer,
        proposals_per_event=proposals_per_state,
        guidance_strength=beta,
        tempering_power=0.0,
        score_potential=target_potential,
    )

    order_rng = np.random.default_rng(np.random.SeedSequence((seed, 0x57A7E)))
    order = order_rng.permutation(len(saved_states))
    proposal_rows: list[dict[str, object]] = []
    arm_rows: dict[str, list[dict[str, object]]] = {
        "beta_zero": [],
        "raw_max_soft": [],
        "target_distance_soft": [],
        "raw_max_best_of_k": [],
        "target_distance_best_of_k": [],
    }
    terminal_states_skipped = 0
    state_occurrences_considered = 0
    for occurrence_index in order:
        if len(proposal_rows) >= state_samples:
            break
        state_occurrences_considered += 1
        saved = saved_states[int(occurrence_index)]
        proposal_rng = np.random.default_rng(
            np.random.SeedSequence((seed, int(occurrence_index), 0xA11CE))
        )
        initial = proposal_sampler.sample_hazard_probe(
            saved.state,
            saved.model_time,
            proposal_rng,
        )
        proposals = proposal_sampler.draw_scored_proposals_after_event(
            saved.state,
            saved.model_time,
            proposal_rng,
            initial,
        )
        if not proposals:
            terminal_states_skipped += 1
            continue
        if len(proposals) != proposals_per_state:
            raise RuntimeError("proposal sampler returned the wrong bounded set size")

        qeds = np.asarray([proposal.score for proposal in proposals], dtype=np.float64)
        beta_zero = kl_tilted_proposal_probabilities(
            qeds,
            beta=0.0,
            score_potential=target_potential,
        )
        raw_soft = kl_tilted_proposal_probabilities(
            qeds,
            beta=beta,
            score_potential=identity_score_potential,
        )
        target_soft = kl_tilted_proposal_probabilities(
            qeds,
            beta=beta,
            score_potential=target_potential,
        )
        selection_rng = np.random.default_rng(
            np.random.SeedSequence((seed, int(occurrence_index), 0x5E1EC7))
        )
        common_uniform = float(selection_rng.random())
        selected_indices = {
            "beta_zero": _select_with_uniform(beta_zero, common_uniform),
            "raw_max_soft": _select_with_uniform(raw_soft, common_uniform),
            "target_distance_soft": _select_with_uniform(target_soft, common_uniform),
            "raw_max_best_of_k": int(np.argmax(qeds)),
            "target_distance_best_of_k": int(
                np.argmax([target_potential(float(value)) for value in qeds])
            ),
        }

        group_members: dict[str, list[int]] = {}
        for proposal_index, proposal in enumerate(proposals):
            group_members.setdefault(proposal.successor_key, []).append(proposal_index)
        groups = []
        for successor_key, indices in group_members.items():
            groups.append(
                {
                    "successor_key": successor_key,
                    "proposal_indices": indices,
                    "multiplicity": len(indices),
                    "base_mass_in_set": len(indices) / proposals_per_state,
                    "qed": float(qeds[indices[0]]),
                    "raw_max_soft_mass": float(raw_soft[indices].sum()),
                    "target_distance_soft_mass": float(target_soft[indices].sum()),
                }
            )

        state_row: dict[str, object] = {
            "source": {
                "rollout_index": saved.rollout_index,
                "state_index": saved.state_index,
                "canonical_key": saved.canonical_key,
                "operational_time": saved.operational_time,
                "model_time": saved.model_time,
            },
            "total_hazard": float(initial.total_hazard),
            "common_uniform": common_uniform,
            "unique_successor_groups": len(groups),
            "groups": groups,
            "proposals": [],
            "selected_indices": selected_indices,
            "expected_qed": {
                "beta_zero": float(np.dot(beta_zero, qeds)),
                "raw_max_soft": float(np.dot(raw_soft, qeds)),
                "target_distance_soft": float(np.dot(target_soft, qeds)),
            },
            "expected_absolute_target_error": {
                "beta_zero": float(np.dot(beta_zero, np.abs(qeds - target_qed))),
                "raw_max_soft": float(np.dot(raw_soft, np.abs(qeds - target_qed))),
                "target_distance_soft": float(
                    np.dot(target_soft, np.abs(qeds - target_qed))
                ),
            },
        }
        for proposal_index, proposal in enumerate(proposals):
            valid = bool(is_valid_state(proposal.successor))
            connected = bool(is_connected_or_null(proposal.successor))
            if not valid or not connected:
                raise RuntimeError("legal proposal produced an invalid or disconnected successor")
            state_row["proposals"].append(
                {
                    "proposal_index": proposal_index,
                    "rule_name": proposal.mark.rule_name,
                    "action": repr(proposal.mark.action),
                    "successor_key": proposal.successor_key,
                    "qed": float(proposal.score),
                    "valid": valid,
                    "connected": connected,
                    "probability_beta_zero": float(beta_zero[proposal_index]),
                    "probability_raw_max_soft": float(raw_soft[proposal_index]),
                    "probability_target_distance_soft": float(target_soft[proposal_index]),
                }
            )
        proposal_rows.append(state_row)
        for arm_name, selected_index in selected_indices.items():
            selected = proposals[selected_index]
            arm_rows[arm_name].append(
                {
                    "proposal_index": selected_index,
                    "rule_name": selected.mark.rule_name,
                    "successor_key": selected.successor_key,
                    "qed": float(selected.score),
                    "valid": bool(is_valid_state(selected.successor)),
                    "connected": bool(is_connected_or_null(selected.successor)),
                }
            )

    if len(proposal_rows) != state_samples:
        raise RuntimeError("insufficient nonterminal saved states for the requested gate")
    proposal_count = state_samples * proposals_per_state
    unique_groups = sum(int(row["unique_successor_groups"]) for row in proposal_rows)
    expected = {
        arm: {
            "mean_qed": mean(float(row["expected_qed"][arm]) for row in proposal_rows),
            "mean_absolute_target_error": mean(
                float(row["expected_absolute_target_error"][arm])
                for row in proposal_rows
            ),
        }
        for arm in ("beta_zero", "raw_max_soft", "target_distance_soft")
    }
    selected_summaries = {
        name: _arm_summary(rows, target_qed=target_qed, tolerance=tolerance)
        for name, rows in arm_rows.items()
    }
    paired = {
        name: _paired_summary(
            arm_rows["beta_zero"],
            rows,
            target_qed=target_qed,
        )
        for name, rows in arm_rows.items()
        if name != "beta_zero"
    }
    expected_target_improvement = (
        expected["beta_zero"]["mean_absolute_target_error"]
        - expected["target_distance_soft"]["mean_absolute_target_error"]
    )
    support_improvement = paired["target_distance_best_of_k"][
        "mean_paired_target_error_improvement"
    ]
    all_proposals_valid = all(
        bool(proposal["valid"])
        for row in proposal_rows
        for proposal in row["proposals"]
    )
    all_proposals_connected = all(
        bool(proposal["connected"])
        for row in proposal_rows
        for proposal in row["proposals"]
    )
    return {
        "format": "compose_v4_qed_common_successor_gate_v1",
        "control": {
            "equation": CONTROL_EQUATION,
            "target_distance_potential": "U_target(q)=-abs(q-target_qed)",
            "raw_max_potential": "U_raw(q)=q",
            "finite_proposal_interpretation": (
                "canonical successors are scored once, but repeated sampled marks retain "
                "multiplicity/base mass in each stochastic softmax"
            ),
            "total_hazard_preserved": True,
            "beam_search": False,
            "deterministic_top_k_used_for_sampling": False,
            "best_of_k_is_offline_support_diagnostic_only": True,
        },
        "configuration": {
            "target_qed": target_qed,
            "tolerance": tolerance,
            "state_samples": state_samples,
            "proposals_per_state": proposals_per_state,
            "proposal_marks": proposal_count,
            "beta": beta,
            "seed": seed,
            "condition_mode": "qed_target" if use_target_condition else "classifier_free",
        },
        "source_state_accounting": {
            "saved_state_occurrences_available": len(saved_states),
            "state_occurrences_considered": state_occurrences_considered,
            "terminal_states_skipped": terminal_states_skipped,
            "evaluated_states": len(proposal_rows),
            "unique_source_state_keys": len(
                {str(row["source"]["canonical_key"]) for row in proposal_rows}
            ),
        },
        "oracle_accounting": {
            "raw_proposal_marks": proposal_count,
            "canonical_groups_requested": scorer.accounting.requests,
            "unique_oracle_evaluations": scorer.accounting.unique_evaluations,
            "cache_hits": scorer.accounting.cache_hits,
            "canonical_groups_across_state_sets": unique_groups,
        },
        "proposal_audit": {
            "all_proposals_valid": all_proposals_valid,
            "all_proposals_connected": all_proposals_connected,
            "unique_group_fraction_of_proposals": unique_groups / proposal_count,
        },
        "expected_stochastic_metrics": expected,
        "selected_arm_metrics": selected_summaries,
        "paired_vs_beta_zero": paired,
        "gate": {
            "support_target_error_improvement": support_improvement,
            "soft_expected_target_error_improvement": expected_target_improvement,
            "support_contains_exploitable_target_improvement": support_improvement > 0.0,
            "soft_target_tilt_positive": expected_target_improvement > 0.0,
            "trajectory_guidance_authorized": bool(
                all_proposals_valid
                and all_proposals_connected
                and support_improvement > 0.0
                and expected_target_improvement > 0.0
            ),
        },
        "proposal_sets": proposal_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-cache", type=Path, required=True)
    parser.add_argument("--rollout-condition-label", default="qed_0.9")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-qed", type=float, default=0.9)
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--proposals-per-state", type=int, default=4)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--classifier-free-base", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.target_qed <= 1.0:
        raise ValueError("target QED must lie in [0, 1]")
    if args.beta <= 0.0 or not np.isfinite(args.beta):
        raise ValueError("the preregistered positive beta must be finite and positive")
    if args.tolerance <= 0.0 or args.max_atoms <= 0:
        raise ValueError("tolerance and max atoms must be positive")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    model, checkpoint = load_factorized_rollout_checkpoint(args.checkpoint)
    model.eval()
    saved_states = saved_rollout_states(
        args.rollout_cache,
        condition_label=args.rollout_condition_label,
        n_slots=args.max_atoms,
    )
    report = evaluate_common_successor_sets(
        model=model,
        checkpoint=checkpoint,
        saved_states=saved_states,
        target_qed=args.target_qed,
        state_samples=args.states,
        proposals_per_state=args.proposals_per_state,
        beta=args.beta,
        seed=args.seed,
        tolerance=args.tolerance,
        use_target_condition=not args.classifier_free_base,
    )
    report["inputs"] = {
        "checkpoint": str(args.checkpoint),
        "rollout_cache": str(args.rollout_cache),
        "rollout_condition_label": args.rollout_condition_label,
    }
    _atomic_json_dump(report, args.output)
    print(json.dumps({"phase": "complete", "output": str(args.output), **report["gate"]}))


if __name__ == "__main__":
    main()
