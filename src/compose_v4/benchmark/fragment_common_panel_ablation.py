"""Selection-law intervention on a recorded, common fragment-program panel.

The eight structural offers and native-support admission are held fixed. This
isolates the learned score's effect on selection, not the effect of replacing
the transition reference used by a primitive sampler.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np


def supported_panel(offered: list[Mapping[str, object]]) -> tuple[tuple[str, float], ...]:
    """Return unique, finite native-supported endpoints in recorded draw order."""
    if len(offered) != 8 or [row.get("draw") for row in offered] != list(range(8)):
        raise ValueError("a common panel must contain the eight recorded offers")
    seen: set[str] = set()
    panel: list[tuple[str, float]] = []
    for row in offered:
        if row.get("status") != "model_supported":
            continue
        endpoint = row.get("endpoint")
        score = row.get("mean_log_mark")
        if not isinstance(endpoint, str) or not endpoint or not isinstance(score, (int, float)):
            raise ValueError("supported offer lacks an endpoint or score")
        if not math.isfinite(float(score)) or endpoint in seen:
            raise ValueError("recorded support contains a nonfinite or duplicate endpoint")
        seen.add(endpoint)
        panel.append((endpoint, float(score)))
    return tuple(panel)


def select_common_panel(
    offered: list[Mapping[str, object]], *, law: str, seed_material: str
) -> str | None:
    """Draw from the identical admitted panel under learned or uniform weights."""
    if law not in {"learned", "uniform"}:
        raise ValueError(f"unknown panel selection law: {law}")
    panel = supported_panel(offered)
    if not panel:
        return None
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    if law == "uniform":
        probabilities = np.full(len(panel), 1.0 / len(panel))
    else:
        scores = np.asarray([score for _, score in panel], dtype=float)
        weights = np.exp(scores - scores.max())
        probabilities = weights / weights.sum()
    return panel[int(rng.choice(len(panel), p=probabilities))][0]
