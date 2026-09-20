"""Region-conditioned `h_φ`: feature contract and the exact boundary condition.

Reuses the PATTERN of `artifacts/h_phi_frozen_v1` — frozen `R_θ` encoder plus a
small head, budget one-hot, goal conditioning — and changes exactly one thing:
the goal is an **objective-space REGION**, not a target molecule.

    old   [e_y, e_z, e_y−e_z, e_y⊙e_z, sim(y,z), budget(8)]     = 1033
                    ^^^ e_z is a TARGET MOLECULE embedding

    new   [e_x, e_src, e_x−e_src, e_x⊙e_src,                    1024
           qed(x), sim(x,x_src),                                    2
           τ_q, τ_s,                                                2
           qed(x)−τ_q, sim(x,x_src)−τ_s,                            2   <- MARGINS
           budget one-hot(25)]                                      25
                                                                 = 1055

THE MARGIN FEATURES ARE THE POINT. `qed(x) − τ_q` and `sim − τ_s` tell the head
directly how far the current state is from satisfying each constraint. Asking a
GNN to infer QED, similarity and a threshold from raw graph structure would be
strictly harder for no benefit — the design rule is *do not make `h_φ`
rediscover known chemistry*.

THE BOUNDARY CONDITION IS ENFORCED, NOT LEARNED
------------------------------------------------
    h_b(x, z) = 1   whenever   x ∈ B_z        for every b

The network is never asked to learn this from trajectories, and the policy
executes STOP. `qualified@0` states are boundary cases, not evidence of
navigation, and they must not dominate the training signal.

The budget axis is `b = 0 … 24` after the H24 amendment; the old contract's
`budget_max = 8` would silently truncate it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: Frozen by `docs/HORIZON_AMENDMENT_H24.md`. This stays 24 FOREVER: the frozen
#: `head.pt` was trained against a 25-slot budget one-hot, so changing this
#: constant would change INPUT_DIM from 1055 to something else and every
#: controller loading that checkpoint would fail on a shape mismatch. A longer
#: horizon is expressed by PASSING `budget_max`, never by editing this.
BUDGET_MAX = 24
#: `R_θ` global-state embedding width, from the frozen feature contract.
EMBED_DIM = 256
#: 4 embedding blocks + 2 state props + 2 thresholds + 2 margins + budget one-hot
INPUT_DIM = 4 * EMBED_DIM + 2 + 2 + 2 + (BUDGET_MAX + 1)


def input_dim(budget_max: int = BUDGET_MAX) -> int:
    """Feature width for a given budget ceiling.

    A budget-extended h_phi needs a wider one-hot and therefore a wider first
    layer. Deriving the width here keeps the trainer and the runtime from
    disagreeing about it, which is the failure that produces a checkpoint no
    controller can load.
    """
    return 4 * EMBED_DIM + 2 + 2 + 2 + (int(budget_max) + 1)

CANONICAL_SLOTS = 48
TIME_POINT = 0.5


def in_region(qed: float, similarity: float, region: tuple[float, float]) -> bool:
    """`x ∈ B_z`. Inclusive on both constraints, matching the census."""
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
    """Region-conditioned features; 1,055-dim at the frozen budget_max = 24.

    `budget_max` is a PARAMETER so a longer-horizon model can be trained without
    touching the module constant the frozen checkpoint depends on. Callers that
    omit it get byte-identical behaviour to before.
    """
    if e_state.shape != (EMBED_DIM,) or e_source.shape != (EMBED_DIM,):
        raise ValueError(f"embeddings must be ({EMBED_DIM},)")
    if not 0 <= budget <= int(budget_max):
        raise ValueError(f"budget {budget} outside [0, {int(budget_max)}]")
    q_min, s_min = region
    one_hot = np.zeros(int(budget_max) + 1, dtype=np.float64)
    one_hot[budget] = 1.0
    return np.concatenate([
        e_state, e_source, e_state - e_source, e_state * e_source,
        np.array([qed, similarity_to_source, q_min, s_min,
                  qed - q_min,                       # MARGIN on QED
                  similarity_to_source - s_min],     # MARGIN on similarity
                 dtype=np.float64),
        one_hot,
    ])


def h_with_boundary(
    raw_prediction: float,
    qed: float,
    similarity_to_source: float,
    region: tuple[float, float],
) -> float:
    """Apply the EXACT boundary condition on top of the network output.

    `h_b(x,z) = 1` whenever `x ∈ B_z`, for every budget including `b = 0`.
    The network's opinion is discarded there — it is not an estimate, it is a
    fact, and the policy stops.
    """
    if in_region(qed, similarity_to_source, region):
        return 1.0
    return float(min(1.0, max(0.0, raw_prediction)))


def should_stop(qed: float, similarity_to_source: float,
                region: tuple[float, float]) -> bool:
    """The online stopping rule. First qualifying state, period."""
    return in_region(qed, similarity_to_source, region)


def acceptance_probabilities(
    raw_predictions: Sequence[float],
    qeds: Sequence[float],
    similarities: Sequence[float],
    region: tuple[float, float],
) -> np.ndarray:
    """Per-candidate acceptance for the Stage-A1 rejection sampler.

    Boundary-corrected, so a successor already inside the region is accepted
    with probability 1 — which is what makes the controller reach and stop.
    """
    return np.array([
        h_with_boundary(p, q, s, region)
        for p, q, s in zip(raw_predictions, qeds, similarities)
    ], dtype=np.float64)
