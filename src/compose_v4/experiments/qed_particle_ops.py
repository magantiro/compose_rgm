"""Particle-weight and resampling operations for finite-horizon QED control."""

from __future__ import annotations

import numpy as np


def normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    """Return normalized weights from a nonextinct log-weight vector."""
    shifted = np.asarray(log_weights, dtype=float)
    shifted = shifted - shifted.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    """Measure degeneracy of normalized particle weights."""
    values = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(values**2))


def should_resample(weights: np.ndarray) -> bool:
    """Resample only when effective sample size is below half the population."""
    return effective_sample_size(weights) < len(weights) * 0.5


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw low-variance ancestor indices from normalized weights."""
    count = len(weights)
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def terminal_output(log_weights: np.ndarray, rng: np.random.Generator) -> tuple[int | None, str]:
    """Draw one terminal particle or record that all target mass vanished."""
    values = np.asarray(log_weights, dtype=float)
    if np.all(np.isneginf(values)):
        return None, "EXTINCT_NO_HIT"
    weights = normalized_weights(values)
    return int(rng.choice(len(weights), p=weights)), "OK"
