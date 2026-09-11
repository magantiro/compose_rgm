"""Batched preference choice over canonical complete-program candidate pools."""

from __future__ import annotations

import numpy as np

from compose_v4.control.branch_policy import RECIPE, BranchPolicy
from compose_v4.control.continuation import _tilted


def choose_pools(pools: list[dict], policy: BranchPolicy, *, seed: int) -> list[dict]:
    """One feature batch, then independent coupled choices per parent, without labels."""
    rows = [r for pool in pools for r in pool["candidates"]]
    utilities = policy.utilities(rows) if rows else np.empty(0)
    if not np.isfinite(utilities).all():
        raise ValueError("nonfinite complete-program preference")
    choices, offset = [], 0
    for slot, pool in enumerate(pools):
        candidates = pool["candidates"]
        if not candidates:
            choices.append({"slot": slot, "status": "empty_pool"})
            continue
        n = len(candidates)
        u = utilities[offset : offset + n]
        offset += n
        base = np.asarray(pool["reference"], dtype=np.float64)
        if (
            base.shape != (n,)
            or not np.isfinite(base).all()
            or np.any(base <= 0)
            or not np.isclose(base.sum(), 1)
        ):
            raise ValueError("invalid empirical canonical program law")
        q, eta, kl = _tilted(
            base, np.exp(u - u.max()), kappa=RECIPE["kappa"], exploration=RECIPE["exploration"]
        )
        rng = np.random.default_rng(np.random.SeedSequence([seed, slot, 313]))
        uniform = float(rng.random())
        indices = {
            name: min(int(np.searchsorted(np.cumsum(p), uniform, side="right")), n - 1)
            for name, p in (("baseline", base), ("learned", q))
        }
        choices.append(
            {
                "slot": slot,
                "status": "selected",
                "indices": indices,
                "uniform_draw": uniform,
                "utilities": u.tolist(),
                "reference": base.tolist(),
                "probabilities": q.tolist(),
                "eta": eta,
                "kl": kl,
                "model_sha256": policy.payload["model_sha256"],
            }
        )
    return choices
