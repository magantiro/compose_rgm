"""Evaluate a frozen tiny rewrite-rate checkpoint with target-free rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compose_v4.experiments.tiny_rate import (
    rollout_metrics,
    sample_piecewise_ancestral,
)
from compose_v4.model.rate_model import WholeGraphRateModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--operational-horizon", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tiny_rate_gate_eval.json"),
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WholeGraphRateModel(
        hidden_dim=int(checkpoint["hidden_dim"]),
        message_passing_steps=2,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    rng = np.random.default_rng(args.seed)
    fiber_cache = {}
    rate_cache = {}
    rollouts = tuple(
        sample_piecewise_ancestral(
            model,
            rng=rng,
            operational_horizon=args.operational_horizon,
            time_step=args.time_step,
            fiber_cache=fiber_cache,
            rate_cache=rate_cache,
        )
        for _ in range(args.samples)
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "time_step": args.time_step,
        "operational_horizon": args.operational_horizon,
        "target_available": False,
        "rollout": rollout_metrics(rollouts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
