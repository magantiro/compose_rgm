"""Goal regions for finite-horizon similarity-constrained QED editing."""

from __future__ import annotations

QED_THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)
SIMILARITY_FLOORS = (0.30, 0.40, 0.50, 0.60)
BENCHMARK_REGION = (0.90, 0.40)


def registered_regions() -> tuple[tuple[float, float], ...]:
    """Return the fixed training grid in a deterministic order."""
    return tuple((qed, similarity) for qed in QED_THRESHOLDS for similarity in SIMILARITY_FLOORS)
