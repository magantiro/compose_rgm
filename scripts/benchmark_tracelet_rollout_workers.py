"""Benchmark deterministic CPU worker counts for ancestral tracelet rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from compose_v4.experiments.parallel_tracelet_sampling import (
    sample_tracelet_ancestral_many,
)
from compose_v4.model.tracelet_rate_model import TraceletRateModel
from compose_v4.rewrite.kernel import canonical_state_key


def rollout_signature(rollout) -> tuple:
    return (
        canonical_state_key(rollout.final_state),
        rollout.event_times,
        rollout.event_rules,
        rollout.exhausted_event_budget,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--operational-horizon", type=float, default=7.0)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["model"] != "from_scratch":
        raise ValueError("this benchmark currently expects a from_scratch checkpoint")
    model = TraceletRateModel(
        hidden_dim=checkpoint["hidden_dim"],
        message_passing_steps=checkpoint["message_passing_steps"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    worker_counts = tuple(int(value) for value in args.workers.split(","))
    results = {}
    reference = None
    for workers in worker_counts:
        started = perf_counter()
        rollouts = sample_tracelet_ancestral_many(
            model,
            seed=args.seed,
            samples=args.samples,
            workers=workers,
            n_slots=args.max_atoms,
            operational_horizon=args.operational_horizon,
            time_step=args.time_step,
            max_events=args.max_events,
        )
        elapsed = perf_counter() - started
        signatures = tuple(map(rollout_signature, rollouts))
        if reference is None:
            reference = signatures
        results[str(workers)] = {
            "seconds": elapsed,
            "seconds_per_trajectory": elapsed / args.samples,
            "exactly_matches_first_configuration": signatures == reference,
        }
    fastest = min(results, key=lambda key: results[key]["seconds"])
    report = {
        "checkpoint": str(args.checkpoint),
        "samples": args.samples,
        "operational_horizon": args.operational_horizon,
        "max_events": args.max_events,
        "results": results,
        "recommended_workers": int(fastest),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
