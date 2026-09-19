"""Subprocess worker for independent tracelet-CTMC rollout shards."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from compose_v4.experiments.parallel_tracelet_sampling import (
    ParallelTraceletSamplingConfig,
)
from compose_v4.experiments.tracelet_conditional import sample_tracelet_ancestral


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    model = torch.load(args.model, map_location="cpu", weights_only=False)
    if isinstance(model, nn.Module):
        model.eval()
    with args.tasks.open("rb") as handle:
        tasks: tuple[tuple[int, int], ...] = pickle.load(handle)
    with args.config.open("rb") as handle:
        config: ParallelTraceletSamplingConfig = pickle.load(handle)

    fiber_cache = {}
    rate_cache = {}
    with args.results.open("wb") as result_handle, args.progress.open(
        "w", encoding="utf-8"
    ) as progress_handle:
        for index, seed in tasks:
            rollout = sample_tracelet_ancestral(
                model,
                rng=np.random.default_rng(seed),
                n_slots=config.n_slots,
                operational_horizon=config.operational_horizon,
                time_step=config.time_step,
                max_events=config.max_events,
                fiber_cache=fiber_cache,
                rate_cache=rate_cache,
                source_prior=config.source_prior,
            )
            pickle.dump((index, rollout), result_handle)
            result_handle.flush()
            progress_handle.write(f"{index}\n")
            progress_handle.flush()


if __name__ == "__main__":
    main()
