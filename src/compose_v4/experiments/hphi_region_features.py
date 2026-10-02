"""Features and terminal membership for the QED future-value model.

The head predicts goal-region membership after the remaining number of
productive transitions. Membership is an exact boundary only at budget zero.
"""

from __future__ import annotations

import numpy as np

BUDGET_MAX = 24
EMBED_DIM = 256
INPUT_DIM = 4 * EMBED_DIM + 2 + 2 + 2 + (BUDGET_MAX + 1)


def input_dim(budget_max: int = BUDGET_MAX) -> int:
    """Return the feature width for a declared transition budget."""
    return 4 * EMBED_DIM + 2 + 2 + 2 + (int(budget_max) + 1)


def in_region(qed: float, similarity: float, region: tuple[float, float]) -> bool:
    """Check inclusive QED and source-similarity thresholds."""
    q_min, s_min = region
    return bool(qed >= q_min and similarity >= s_min)


def build_features(
    e_state: np.ndarray,
    e_source: np.ndarray,
    qed: float,
    similarity_to_source: float,
    region: tuple[float, float],
    budget: int,
    budget_max: int = BUDGET_MAX,
) -> np.ndarray:
    """Encode molecular state, source, goal margins and remaining budget."""
    if e_state.shape != (EMBED_DIM,) or e_source.shape != (EMBED_DIM,):
        raise ValueError(f"embeddings must be ({EMBED_DIM},)")
    if not 0 <= budget <= int(budget_max):
        raise ValueError(f"budget {budget} outside [0, {int(budget_max)}]")
    q_min, s_min = region
    one_hot = np.zeros(int(budget_max) + 1, dtype=np.float64)
    one_hot[budget] = 1.0
    return np.concatenate(
        [
            e_state,
            e_source,
            e_state - e_source,
            e_state * e_source,
            np.array(
                [
                    qed,
                    similarity_to_source,
                    q_min,
                    s_min,
                    qed - q_min,
                    similarity_to_source - s_min,
                ],
                dtype=np.float64,
            ),
            one_hot,
        ]
    )
