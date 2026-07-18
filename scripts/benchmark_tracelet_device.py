"""Benchmark an identical tracelet-GM update on CPU and Apple MPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.experiments.tracelet_conditional import (
    build_tracelet_path_records,
    sample_tracelet_conditional_batch,
    tracelet_conditional_batch_loss,
)
from compose_v4.model.device import synchronize_device
from compose_v4.model.tracelet_rate_model import TraceletRateModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    split = load_cnof_corpus_split(
        args.smiles_file,
        train_size=64,
        validation_size=8,
        test_size=8,
        max_atoms=16,
        seed=args.seed,
        scan_all=True,
    )
    records = build_tracelet_path_records(split.train, n_slots=16)
    examples = sample_tracelet_conditional_batch(
        records,
        batch_size=args.batch_size,
        rng=np.random.default_rng(args.seed + 1),
        fiber_cache={},
        late_time_fraction=0.5,
    )
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    results = {}
    for device in devices:
        torch.manual_seed(args.seed)
        model = TraceletRateModel(hidden_dim=64, message_passing_steps=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad()
            tracelet_conditional_batch_loss(model, examples).backward()
            optimizer.step()
        synchronize_device(device)
        started = perf_counter()
        for _ in range(args.repeats):
            optimizer.zero_grad()
            tracelet_conditional_batch_loss(model, examples).backward()
            optimizer.step()
        synchronize_device(device)
        elapsed = perf_counter() - started
        results[str(device)] = {
            "seconds": elapsed,
            "seconds_per_update": elapsed / args.repeats,
        }
    fastest = min(results, key=lambda name: results[name]["seconds_per_update"])
    print(
        json.dumps(
            {
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
                "results": results,
                "recommended_device": fastest,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
