"""Run the first learned rewrite-rate gate and target-free rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compose_v4.experiments.tiny_rate import (
    build_exact_marginal_batch,
    build_tiny_paths,
    rollout_metrics,
    sample_piecewise_ancestral,
    train_exact_marginal_batch,
)
from compose_v4.model.rate_model import WholeGraphRateModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("results/tiny_rate_gate.json"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/tiny_rate_gate.pt"),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    paths = build_tiny_paths(seed=args.seed)
    examples = build_exact_marginal_batch(paths)
    model = WholeGraphRateModel(
        hidden_dim=args.hidden_dim,
        message_passing_steps=2,
    )
    initial, final, history = train_exact_marginal_batch(
        model,
        examples,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed + 1)
    fiber_cache = {}
    rate_cache = {}
    rollouts = tuple(
        sample_piecewise_ancestral(
            model,
            rng=rng,
            fiber_cache=fiber_cache,
            rate_cache=rate_cache,
        )
        for _ in range(args.samples)
    )
    report = {
        "seed": args.seed,
        "corpus_size": len(paths),
        "exact_marginal_examples": len(examples),
        "training_steps": args.steps,
        "minibatch_size": args.batch_size,
        "training_weighting": "uniform_over_exact_state_time_groups",
        "initial": initial,
        "final": final,
        "loss_history": history,
        "rollout": rollout_metrics(rollouts),
        "sampler": {
            "clock": "cumulative_hazard_operational_time",
            "scheme": "piecewise_constant_ancestral_ctmc",
            "operational_horizon": 5.0,
            "time_step": 0.1,
            "max_events": 24,
            "target_available": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "seed": args.seed,
            "report": report,
        },
        args.checkpoint,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
