"""Reference-guided draws for an existing controller's exploration quota.

The baseline is the controller's actual uniform, without-replacement draw. This
module does not rerank exploitation slots or allocate any additional queries.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from compose_v4.control.reference_guidance import GuidedPanel

PanelGuide = Callable[[Sequence[Mapping[str, Any]]], GuidedPanel]


def draw_exploration(
    candidates: Sequence[Mapping[str, Any]],
    count: int,
    rng: np.random.Generator,
    *,
    guide: PanelGuide | None = None,
    receipts: list[dict] | None = None,
) -> list[int]:
    """Return indices into the supplied remaining panel, without replacement.

    Off, shadow, and coverage fallback use the original NumPy uniform call,
    without even supplying ``p``. Supplying uniform ``p`` would change the RNG
    algorithm and break compatibility. Guidance requires a receipt sink so
    unsupported scoring cannot silently look like active reference control.
    """
    if type(count) is not int or not 0 <= count <= len(candidates):
        raise ValueError("exploration count must be an integer within the candidate panel")
    if count == 0:
        return []
    if guide is not None and receipts is None:
        raise ValueError("reference-aware exploration requires a receipt sink")
    panel = None if guide is None else guide(candidates)
    if guide is not None and not isinstance(panel, GuidedPanel):
        raise TypeError("reference guide must return a GuidedPanel")
    if panel is not None:
        if len(panel.candidate_ids) != len(candidates):
            raise ValueError("reference exploration must preserve the entire remaining panel")
        uniform = 1.0 / len(candidates)
        if any(p != uniform for p in panel.baseline_probabilities):
            raise ValueError("exploration guidance must start from the actual uniform baseline")
    if panel is not None and panel.probabilities_changed:
        chosen = rng.choice(len(candidates), count, replace=False, p=panel.probabilities)
    else:
        chosen = rng.choice(len(candidates), count, replace=False)
    indices = [int(index) for index in np.atleast_1d(chosen)]
    if panel is not None:
        # Conditional on the previously drawn indices, not a marginal inclusion
        # probability for a weighted batch. Sum remaining mass directly to avoid
        # cancellation when previous draws consumed almost all the probability.
        remaining = set(range(len(candidates)))
        conditional = []
        for index in indices:
            total = math.fsum(panel.probabilities[i] for i in sorted(remaining))
            conditional.append(panel.probabilities[index] / total)
            remaining.remove(index)
        receipts.append(
            {
                **panel.receipt(),
                "sampling": "without_replacement",
                "allocation": "existing_exploration_quota",
                "requested_count": count,
                "selected_indices": indices,
                "selected_ids": [panel.candidate_ids[i] for i in indices],
                "conditional_draw_probabilities": conditional,
                "conditional_probability_semantics": "given previously selected indices",
            }
        )
    return indices
