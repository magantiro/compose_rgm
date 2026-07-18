"""Deterministic multi-process ancestral sampling for tracelet generators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from time import sleep
import warnings

import numpy as np
import torch
from torch import nn
from compose_v4.chem.source_prior import MolecularSourcePrior

from compose_v4.experiments.tracelet_conditional import (
    TraceletRatePredictor,
    TraceletRollout,
    sample_tracelet_ancestral,
)


@dataclass(frozen=True)
class ParallelTraceletSamplingConfig:
    n_slots: int
    operational_horizon: float
    time_step: float
    max_events: int
    source_prior: MolecularSourcePrior | None = None


def _trajectory_seeds(seed: int, samples: int) -> tuple[int, ...]:
    sequence = np.random.SeedSequence(seed)
    return tuple(
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in sequence.spawn(samples)
    )


def _is_shared_memory_runtime_failure(details: str) -> bool:
    """Recognize host-level OpenMP failures that are safe to retry serially."""

    normalized = details.lower()
    return "can't open shm" in normalized or "cannot open shared memory" in normalized


def sample_tracelet_ancestral_many(
    model: TraceletRatePredictor,
    *,
    seed: int,
    samples: int,
    workers: int,
    n_slots: int,
    operational_horizon: float,
    time_step: float,
    max_events: int,
    torch_threads_per_worker: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
    source_prior: MolecularSourcePrior | None = None,
) -> tuple[TraceletRollout, ...]:
    """Sample independent trajectories reproducibly on one or more CPU workers.

    Every trajectory receives a seed derived solely from ``seed`` and its sample
    index, so results do not depend on worker count or completion order.
    """

    if samples <= 0 or workers <= 0 or torch_threads_per_worker <= 0:
        raise ValueError("samples, workers, and torch_threads_per_worker must be positive")
    config = ParallelTraceletSamplingConfig(
        n_slots=n_slots,
        operational_horizon=operational_horizon,
        time_step=time_step,
        max_events=max_events,
        source_prior=source_prior,
    )
    seeds = _trajectory_seeds(seed, samples)
    completed: list[TraceletRollout] = []
    if workers == 1:
        fiber_cache = {}
        rate_cache = {}
        if isinstance(model, nn.Module):
            model.eval()
        for index, trajectory_seed in enumerate(seeds, start=1):
            completed.append(
                sample_tracelet_ancestral(
                    model,
                    rng=np.random.default_rng(trajectory_seed),
                    n_slots=n_slots,
                    operational_horizon=operational_horizon,
                    time_step=time_step,
                    max_events=max_events,
                    fiber_cache=fiber_cache,
                    rate_cache=rate_cache,
                    source_prior=source_prior,
                )
            )
            if progress_callback is not None:
                progress_callback(index, samples)
        return tuple(completed)

    if isinstance(model, nn.Module):
        devices = {parameter.device.type for parameter in model.parameters()}
        if devices != {"cpu"}:
            raise ValueError("multi-process rollout requires a CPU-resident model")
    worker_count = min(workers, samples)
    indexed_seeds = tuple(enumerate(seeds))
    buckets = tuple(indexed_seeds[offset::worker_count] for offset in range(worker_count))
    with tempfile.TemporaryDirectory(prefix="compose_v4_rollouts_") as directory:
        root = Path(directory)
        model_path = root / "model.pt"
        config_path = root / "config.pkl"
        torch.save(model, model_path)
        with config_path.open("wb") as handle:
            pickle.dump(config, handle)

        processes = []
        logs = []
        progress_paths = []
        result_paths = []
        environment = dict(os.environ)
        environment["OMP_NUM_THREADS"] = str(torch_threads_per_worker)
        environment["MKL_NUM_THREADS"] = str(torch_threads_per_worker)
        for worker_index, tasks in enumerate(buckets):
            task_path = root / f"tasks_{worker_index}.pkl"
            result_path = root / f"results_{worker_index}.pkl"
            progress_path = root / f"progress_{worker_index}.txt"
            log_path = root / f"worker_{worker_index}.log"
            with task_path.open("wb") as handle:
                pickle.dump(tasks, handle)
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "compose_v4.experiments.tracelet_sampling_worker",
                    "--model",
                    str(model_path),
                    "--tasks",
                    str(task_path),
                    "--results",
                    str(result_path),
                    "--progress",
                    str(progress_path),
                    "--config",
                    str(config_path),
                    "--torch-threads",
                    str(torch_threads_per_worker),
                ],
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((process, log_path))
            logs.append(log_handle)
            progress_paths.append(progress_path)
            result_paths.append(result_path)

        reported = 0
        try:
            while any(process.poll() is None for process, _ in processes):
                observed = sum(
                    path.read_text(encoding="utf-8").count("\n")
                    if path.exists()
                    else 0
                    for path in progress_paths
                )
                while reported < observed:
                    reported += 1
                    if progress_callback is not None:
                        progress_callback(reported, samples)
                sleep(0.25)
        finally:
            for log_handle in logs:
                log_handle.close()

        failures = [
            (process.returncode, log_path)
            for process, log_path in processes
            if process.returncode != 0
        ]
        if failures:
            details = "\n".join(
                f"worker exit {code}: {path.read_text(encoding='utf-8')}"
                for code, path in failures
            )
            if _is_shared_memory_runtime_failure(details):
                warnings.warn(
                    "parallel rollout workers cannot access shared memory; "
                    "falling back to deterministic serial sampling",
                    RuntimeWarning,
                    stacklevel=2,
                )

                def fallback_progress(index: int, total: int) -> None:
                    if progress_callback is not None and index > reported:
                        progress_callback(index, total)

                return sample_tracelet_ancestral_many(
                    model,
                    seed=seed,
                    samples=samples,
                    workers=1,
                    n_slots=n_slots,
                    operational_horizon=operational_horizon,
                    time_step=time_step,
                    max_events=max_events,
                    torch_threads_per_worker=torch_threads_per_worker,
                    progress_callback=fallback_progress,
                    source_prior=source_prior,
                )
            raise RuntimeError(f"parallel rollout worker failed:\n{details}")

        indexed_rollouts = []
        for path in result_paths:
            with path.open("rb") as handle:
                while True:
                    try:
                        indexed_rollouts.append(pickle.load(handle))
                    except EOFError:
                        break
        indexed_rollouts.sort(key=lambda item: item[0])
        completed = [rollout for _, rollout in indexed_rollouts]
        while reported < samples:
            reported += 1
            if progress_callback is not None:
                progress_callback(reported, samples)
        return tuple(completed)


__all__ = [
    "ParallelTraceletSamplingConfig",
    "sample_tracelet_ancestral_many",
]
