"""Locked-candidate batch acquisition from joint posterior samples.

Continuation value decides how to search.  This module makes the separate
decision of which complete, canonical-deduplicated molecules receive a scarce
oracle call.  It never edits candidates or evaluates an oracle.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def probability_of_optimality(posterior_samples: np.ndarray) -> np.ndarray:
    """Monte Carlo probability that each candidate is optimal under joint draws.

    Tied maxima split one draw's mass evenly, avoiding order-dependent credit.
    """

    draws = np.asarray(posterior_samples, dtype=float)
    if draws.ndim != 2 or not draws.shape[0] or not draws.shape[1]:
        raise ValueError("joint posterior samples must be a nonempty 2D matrix")
    if not np.isfinite(draws).all():
        raise ValueError("joint posterior samples must be finite")
    maxima = draws.max(axis=1, keepdims=True)
    winners = np.isclose(draws, maxima, atol=0, rtol=0)
    credit = winners / winners.sum(axis=1, keepdims=True)
    return credit.mean(axis=0)


def select_qpo_batch(
    candidates: Sequence[dict], posterior_samples: np.ndarray, *, batch_size: int
) -> dict:
    """Rank a unique locked pool by probability of optimality."""

    rows = tuple(candidates)
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError("batch_size must be an integer")
    if not 1 <= int(batch_size) <= len(rows):
        raise ValueError("batch_size must lie in 1..number of candidates")
    identifiers, molecules = [], []
    for index, row in enumerate(rows):
        candidate_id, smiles = row.get("candidate_id"), row.get("canonical_smiles")
        if not candidate_id or not smiles:
            raise ValueError(f"candidate {index} lacks canonical and lock identity")
        if row.get("locked") is not True or row.get("complete") is not True:
            raise ValueError(f"candidate {index} is not a locked complete molecule")
        identifiers.append(str(candidate_id))
        molecules.append(str(smiles))
    if len(set(identifiers)) != len(rows) or len(set(molecules)) != len(rows):
        raise ValueError("acquisition candidates must be identity- and molecule-unique")
    draws = np.asarray(posterior_samples, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != len(rows):
        raise ValueError("posterior sample columns must align with candidates")
    probabilities = probability_of_optimality(draws)
    means = draws.mean(axis=0)
    order = sorted(
        range(len(rows)),
        key=lambda i: (-probabilities[i], -means[i], identifiers[i]),
    )
    selected = order[: int(batch_size)]
    return {
        "schema_version": "joint_qpo_acquisition_v1",
        "candidate_ids": identifiers,
        "probability_of_optimality": probabilities.tolist(),
        "posterior_means": means.tolist(),
        "ranking": order,
        "selected_indices": selected,
        "selected_candidate_ids": [identifiers[i] for i in selected],
        "batch_size": int(batch_size),
        "posterior_draws": int(draws.shape[0]),
    }
