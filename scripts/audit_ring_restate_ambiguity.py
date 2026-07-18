"""Audit electronic ambiguity inside the ring-system restate fiber.

The trace compiler presents one coordinated ring restatement as the teacher
event.  At inference time, however, every validity-closed maximum matching is
legal.  This script measures how often those legal outcomes differ in
aromaticity while receiving indistinguishable scores from a trained model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median

import numpy as np
from rdkit import Chem
import torch

from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.model.tracelet_rate_model import TraceletRateModel
from compose_v4.rewrite.fiber import MarkedTransition
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.trace import execute_trace
from compose_v4.rewrite.tracelet_compiler import compile_null_to_target_tracelets
from compose_v4.rewrite.tracelet_fiber import enumerate_tracelet_cnof_fiber


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=1067)
    parser.add_argument("--validation-size", type=int, default=133)
    parser.add_argument("--test-size", type=int, default=133)
    parser.add_argument("--limit-events", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ring_restate_ambiguity_audit.json"),
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    split = load_cnof_corpus_split(
        args.smiles_file,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        max_atoms=args.max_atoms,
        seed=args.seed,
        scan_all=True,
    )
    model, checkpoint_metadata = _load_model(args.checkpoint)
    report = audit_ring_restate_events(
        split.train,
        model=model,
        n_slots=args.max_atoms,
        limit_events=args.limit_events,
    )
    report["configuration"] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "seed": args.seed,
        "max_atoms": args.max_atoms,
        "train_molecules": len(split.train),
        "validation_molecules": len(split.validation),
        "test_molecules": len(split.test),
        "eligible_molecules": split.eligible_molecules,
        "limit_events": args.limit_events,
        "device": "cpu",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "configuration": report["configuration"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


def audit_ring_restate_events(
    smiles: tuple[str, ...],
    *,
    model: TraceletRateModel,
    n_slots: int,
    limit_events: int | None = None,
) -> dict[str, object]:
    runtime = de_novo_rewrite_system()
    events: list[dict[str, object]] = []
    candidate_count_histogram: Counter[int] = Counter()
    change_count_histogram: Counter[int] = Counter()
    teacher_aromatic_ring_count_histogram: Counter[int] = Counter()
    unique_pre_states: set[str] = set()

    for molecule_index, text in enumerate(smiles):
        target = pad_molecular_graph(smiles_to_molecular_graph(text), n_slots)
        trace = compile_null_to_target_tracelets(target)
        _, states = execute_trace(trace.source, trace.steps, return_states=True)
        for progress, step in enumerate(trace.steps):
            if step.rule_name != "ring_system_restate":
                continue
            pre_state = states[progress]
            teacher_state = states[progress + 1]
            transitions = _restate_transitions(pre_state, runtime=runtime)
            if not transitions:
                raise RuntimeError("teacher ring restate has an empty inference fiber")
            teacher_key = canonical_state_key(teacher_state)
            if teacher_key not in {item.successor_key for item in transitions}:
                raise RuntimeError("teacher ring restate is outside its inference fiber")

            # The within-family ordering is the relevant ambiguity.  Global
            # state, time, total hazard, and family probability are common to
            # every candidate and therefore cancel in the conditional choice.
            model_time = (progress + 0.5) / (len(trace.steps) + 1.0)
            with torch.no_grad():
                logits, _, _ = model.action_logits(
                    pre_state,
                    model_time,
                    transitions,
                )
                probabilities = torch.softmax(logits, dim=0).cpu().numpy()
                logit_values = logits.cpu().numpy()

            successor_indices: dict[str, list[int]] = defaultdict(list)
            successor_states: dict[str, MolecularGraph] = {}
            for index, transition in enumerate(transitions):
                successor_indices[transition.successor_key].append(index)
                successor_states.setdefault(
                    transition.successor_key,
                    transition.successor,
                )
            successor_probabilities = {
                key: float(probabilities[indices].sum())
                for key, indices in successor_indices.items()
            }
            electronic = {
                key: _electronic_features(state)
                for key, state in successor_states.items()
            }
            teacher_electronic = electronic[teacher_key]
            aromatic_keys = {
                key
                for key, features in electronic.items()
                if int(features["aromatic_ring_count"]) > 0
            }
            aromatic_mark_mask = np.asarray(
                [transition.successor_key in aromatic_keys for transition in transitions],
                dtype=bool,
            )
            teacher_probability = successor_probabilities[teacher_key]
            successor_rank = 1 + sum(
                probability > teacher_probability + 1e-12
                for probability in successor_probabilities.values()
            )
            rounded_logits = np.round(logit_values, decimals=7)
            unique_logits = len(set(float(value) for value in rounded_logits))
            largest_logit_tie = max(Counter(rounded_logits.tolist()).values())
            candidate_aromatic_fraction = float(
                aromatic_mark_mask.mean() if len(aromatic_mark_mask) else 0.0
            )
            model_aromatic_mass = float(probabilities[aromatic_mark_mask].sum())

            event = {
                "molecule_index": molecule_index,
                "target_smiles": text,
                "progress": progress,
                "pre_state": canonical_state_key(pre_state),
                "teacher_successor": teacher_key,
                "change_count": len(step.action.changes),
                "marked_candidates": len(transitions),
                "unique_successors": len(successor_indices),
                "aromatic_marked_candidates": int(aromatic_mark_mask.sum()),
                "aromatic_unique_successors": len(aromatic_keys),
                "candidate_aromatic_fraction": candidate_aromatic_fraction,
                "model_aromatic_mass": model_aromatic_mass,
                "aromatic_mass_minus_uniform": (
                    model_aromatic_mass - candidate_aromatic_fraction
                ),
                "teacher_is_aromatic": bool(
                    int(teacher_electronic["aromatic_ring_count"]) > 0
                ),
                "teacher_aromatic_ring_count": int(
                    teacher_electronic["aromatic_ring_count"]
                ),
                "teacher_successor_probability_within_family": teacher_probability,
                "teacher_successor_rank_within_family": int(successor_rank),
                "unique_logits_at_1e-7": unique_logits,
                "largest_logit_tie_at_1e-7": int(largest_logit_tie),
                "all_candidate_logits_equal_at_1e-7": bool(unique_logits == 1),
            }
            events.append(event)
            unique_pre_states.add(str(event["pre_state"]))
            candidate_count_histogram[len(transitions)] += 1
            change_count_histogram[len(step.action.changes)] += 1
            teacher_aromatic_ring_count_histogram[
                int(teacher_electronic["aromatic_ring_count"])
            ] += 1

            if len(events) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "audit",
                            "events": len(events),
                            "molecules_scanned": molecule_index + 1,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if limit_events is not None and len(events) >= limit_events:
                break
        if limit_events is not None and len(events) >= limit_events:
            break

    if not events:
        raise RuntimeError("no ring-system restate events were found")
    return {
        "summary": _summarize_events(events, unique_pre_states),
        "candidate_count_histogram": _string_keyed(candidate_count_histogram),
        "change_count_histogram": _string_keyed(change_count_histogram),
        "teacher_aromatic_ring_count_histogram": _string_keyed(
            teacher_aromatic_ring_count_histogram
        ),
        "events": events,
    }


def _summarize_events(
    events: list[dict[str, object]],
    unique_pre_states: set[str],
) -> dict[str, object]:
    def mean(name: str) -> float:
        return float(np.mean([float(event[name]) for event in events]))

    candidate_counts = [int(event["marked_candidates"]) for event in events]
    aromatic_teacher_events = [
        event for event in events if bool(event["teacher_is_aromatic"])
    ]
    aromatic_ambiguous_events = [
        event
        for event in aromatic_teacher_events
        if int(event["aromatic_marked_candidates"])
        < int(event["marked_candidates"])
    ]
    return {
        "teacher_events": len(events),
        "unique_pre_states": len(unique_pre_states),
        "teacher_aromatic_event_fraction": float(
            len(aromatic_teacher_events) / len(events)
        ),
        "aromatic_teacher_events_with_nonaromatic_competitors_fraction": float(
            len(aromatic_ambiguous_events) / max(len(aromatic_teacher_events), 1)
        ),
        "mean_marked_candidates": float(np.mean(candidate_counts)),
        "median_marked_candidates": float(median(candidate_counts)),
        "max_marked_candidates": max(candidate_counts),
        "mean_unique_successors": mean("unique_successors"),
        "mean_candidate_aromatic_fraction": mean("candidate_aromatic_fraction"),
        "mean_model_aromatic_mass": mean("model_aromatic_mass"),
        "mean_aromatic_mass_minus_uniform": mean(
            "aromatic_mass_minus_uniform"
        ),
        "mean_teacher_successor_probability_within_family": mean(
            "teacher_successor_probability_within_family"
        ),
        "mean_teacher_successor_rank_within_family": mean(
            "teacher_successor_rank_within_family"
        ),
        "all_candidate_logits_equal_fraction": mean(
            "all_candidate_logits_equal_at_1e-7"
        ),
        "mean_unique_logit_fraction": float(
            np.mean(
                [
                    int(event["unique_logits_at_1e-7"])
                    / int(event["marked_candidates"])
                    for event in events
                ]
            )
        ),
        "mean_largest_logit_tie": mean("largest_logit_tie_at_1e-7"),
    }


def _restate_transitions(state: MolecularGraph, *, runtime) -> tuple[MarkedTransition, ...]:
    return enumerate_tracelet_cnof_fiber(
        state,
        system=runtime,
    ).by_family["ring_system_restate"]


def _electronic_features(state: MolecularGraph) -> dict[str, object]:
    smiles = molecular_graph_to_smiles(state)
    if smiles is None:
        raise RuntimeError("valid ring restate successor did not decode")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError("decoded ring restate successor did not parse")
    bond_rings = tuple(tuple(ring) for ring in molecule.GetRingInfo().BondRings())
    aromatic_ring_count = sum(
        bool(ring)
        and all(molecule.GetBondWithIdx(int(index)).GetIsAromatic() for index in ring)
        for ring in bond_rings
    )
    aromatic_bond_count = sum(
        bond.GetIsAromatic() for bond in molecule.GetBonds()
    )
    return {
        "smiles": smiles,
        "aromatic_ring_count": int(aromatic_ring_count),
        "aromatic_bond_count": int(aromatic_bond_count),
    }


def _load_model(checkpoint_path: Path) -> tuple[TraceletRateModel, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("checkpoint does not contain a model state_dict")
    hidden_dim = int(checkpoint.get("hidden_dim", 64))
    message_passing_steps = int(checkpoint.get("message_passing_steps", 3))
    model = TraceletRateModel(
        hidden_dim=hidden_dim,
        message_passing_steps=message_passing_steps,
        rate_factorization=str(checkpoint.get("rate_factorization", "hierarchical")),
        use_aromatic_bond_view=(
            str(checkpoint.get("bond_representation", "kekule")) == "aromatic"
        ),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key != "state_dict" and isinstance(value, (str, int, float, bool))
    }
    return model, metadata


def _string_keyed(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


if __name__ == "__main__":
    main()
