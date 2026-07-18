"""Measure whether a trained model opens the electronic-restatement channel.

This complements ``audit_ring_restate_ambiguity.py``.  That audit conditions on
the ring-system family already having been selected; this one enumerates the
entire legal fiber and measures the probability assigned to that family.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.model.tracelet_rate_model import TraceletRateModel
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace import execute_trace
from compose_v4.rewrite.tracelet_compiler import compile_null_to_target_tracelets
from compose_v4.rewrite.tracelet_fiber import enumerate_tracelet_cnof_fiber


DEFAULT_TIMES = (0.5, 0.8, 0.95, 0.99)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=1067)
    parser.add_argument("--validation-size", type=int, default=133)
    parser.add_argument("--test-size", type=int, default=133)
    parser.add_argument("--events", type=int, default=64)
    parser.add_argument("--times", type=float, nargs="+", default=DEFAULT_TIMES)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ring_restate_family_audit.json"),
    )
    args = parser.parse_args()
    if args.events <= 0:
        raise ValueError("events must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.times):
        raise ValueError("all times must lie in [0, 1]")

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
    candidates = _collect_teacher_events(split.train, n_slots=args.max_atoms)
    rng = np.random.default_rng(args.seed + 41)
    chosen = rng.choice(
        len(candidates),
        size=min(args.events, len(candidates)),
        replace=False,
    )
    selected = tuple(candidates[int(index)] for index in sorted(chosen))
    report = audit_family_selection(
        selected,
        model=model,
        times=tuple(float(value) for value in args.times),
    )
    report["configuration"] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "seed": args.seed,
        "max_atoms": args.max_atoms,
        "available_teacher_events": len(candidates),
        "sampled_teacher_events": len(selected),
        "times": list(args.times),
        "device": "cpu",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(
        json.dumps(
            {
                "summary_by_time": report["summary_by_time"],
                "configuration": report["configuration"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


def audit_family_selection(
    events: tuple[dict[str, object], ...],
    *,
    model: TraceletRateModel,
    times: tuple[float, ...],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    fiber_cache = {}
    for event_index, event in enumerate(events):
        state = event["state"]
        state_key = str(event["pre_state"])
        fiber = fiber_cache.get(state_key)
        if fiber is None:
            fiber = enumerate_tracelet_cnof_fiber(state)
            fiber_cache[state_key] = fiber
        teacher_key = str(event["teacher_successor"])
        for model_time in times:
            with torch.no_grad():
                prediction = model.predict_tracelet_fiber(
                    state,
                    model_time,
                    fiber=fiber,
                )
            rates = prediction.marked_rates.cpu().numpy()
            total_hazard = float(rates.sum())
            family_rates: Counter[str] = Counter()
            for transition, rate in zip(prediction.transitions, rates):
                family_rates[transition.rule_name] += float(rate)
            restate_rate = float(family_rates["ring_system_restate"])
            restate_probability = restate_rate / max(total_hazard, 1e-30)
            teacher_rate = float(prediction.successor_rate(teacher_key).cpu())
            family_probabilities = {
                family: rate / max(total_hazard, 1e-30)
                for family, rate in family_rates.items()
            }
            top_family = max(family_probabilities, key=family_probabilities.get)
            rows.append(
                {
                    "event_index": event_index,
                    "target_smiles": event["target_smiles"],
                    "pre_state": state_key,
                    "teacher_successor": teacher_key,
                    "progress": event["progress"],
                    "path_length": event["path_length"],
                    "time": model_time,
                    "total_hazard": total_hazard,
                    "ring_system_restate_probability": restate_probability,
                    "teacher_successor_probability": teacher_rate
                    / max(total_hazard, 1e-30),
                    "teacher_probability_given_restate": teacher_rate
                    / max(restate_rate, 1e-30),
                    "top_family": top_family,
                    "top_family_probability": family_probabilities[top_family],
                    "family_probabilities": dict(sorted(family_probabilities.items())),
                    "fiber_family_counts": {
                        family: len(transitions)
                        for family, transitions in fiber.by_family.items()
                        if transitions
                    },
                }
            )
        if (event_index + 1) % 16 == 0:
            print(
                json.dumps(
                    {"phase": "full_fiber_audit", "events": event_index + 1}
                ),
                flush=True,
            )
    return {
        "summary_by_time": {
            str(model_time): _summarize_time(rows, model_time)
            for model_time in times
        },
        "events": rows,
    }


def _collect_teacher_events(
    smiles: tuple[str, ...],
    *,
    n_slots: int,
) -> tuple[dict[str, object], ...]:
    events = []
    for text in smiles:
        target = pad_molecular_graph(smiles_to_molecular_graph(text), n_slots)
        trace = compile_null_to_target_tracelets(target)
        _, states = execute_trace(trace.source, trace.steps, return_states=True)
        for progress, step in enumerate(trace.steps):
            if step.rule_name != "ring_system_restate":
                continue
            events.append(
                {
                    "target_smiles": text,
                    "state": states[progress],
                    "pre_state": canonical_state_key(states[progress]),
                    "teacher_successor": canonical_state_key(states[progress + 1]),
                    "progress": progress,
                    "path_length": len(trace.steps),
                }
            )
    return tuple(events)


def _summarize_time(
    rows: list[dict[str, object]],
    model_time: float,
) -> dict[str, object]:
    subset = [row for row in rows if float(row["time"]) == model_time]

    def mean(name: str) -> float:
        return float(np.mean([float(row[name]) for row in subset]))

    top_family_counts = Counter(str(row["top_family"]) for row in subset)
    return {
        "events": len(subset),
        "mean_ring_system_restate_probability": mean(
            "ring_system_restate_probability"
        ),
        "median_ring_system_restate_probability": float(
            np.median(
                [float(row["ring_system_restate_probability"]) for row in subset]
            )
        ),
        "mean_teacher_successor_probability": mean(
            "teacher_successor_probability"
        ),
        "mean_teacher_probability_given_restate": mean(
            "teacher_probability_given_restate"
        ),
        "restate_is_top_family_fraction": float(
            top_family_counts["ring_system_restate"] / len(subset)
        ),
        "top_family_counts": dict(sorted(top_family_counts.items())),
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


if __name__ == "__main__":
    main()
