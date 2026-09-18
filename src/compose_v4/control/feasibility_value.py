"""Future feasibility of a molecule under the production program kernel.

The T4 fiber at delta=0.6 is thin, and the strong molecules sit on its boundary: the best
solutions carry a median similarity margin of +0.019 and several sit at exactly 0.600.
Uncontrolled program dynamics do not respect that geometry. Measured from a parent at
similarity 0.629, random coordinated pairs land at median 0.325 and 484 of 486 fall
outside the fiber -- two edits move about twice as far, so drift compounds rather than
cancels.

This module ESTIMATES that geometry. It does not steer by it, and the distinction is the
whole point of the file.

    psi_h(G) = P_{q0}( G_1, ..., G_h all in Q_delta | G_0 = G )

An earlier design proposed from `q0(Z|G) * 1[T(G,Z) in Q_delta] * psi_{h-1}(T(G,Z))`, so
that a move was preferred when the corridor CONTINUED. Measuring psi is what refuted that
design. Over 14 feasible delta=0.6 JAK2 molecules, `corr(psi_1, docking score) = +0.424`,
and because better docking is more negative, higher survival goes with WORSE binding: the
best molecule had the lowest survival (0.08, 92% dead on arrival) and the weakest the
highest (0.50). psi_3 is 0.00 for 71% of them. Multiplying by psi therefore steers away
from the high-reward corner, which is why `fiber_control` treats feasibility as a SUPPORT
constraint with an explicit STOP and never as a factor in the policy.

Caveats that bound the measurement: n=14 molecules at 12 rollouts each, and the rollout
policy is uniform, so a psi measured under a controlled policy would be higher and might
not anti-correlate the same way. It is suggestive, not established -- but it is enough to
refuse a multiplicative psi, because the burden is on adding the factor.

This is not the maneuverability proxy that failed earlier. That was a handcrafted
correlate of a molecule's edit capacity and added nothing over current score. `psi` is
estimated by launching real program rollouts from the exact state through the exact
kernel and counting how many survive the exact task gate. It costs no docking call, which
is the whole point: feasibility is free and reward is not.
"""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "feasibility_value_v1"


def survival_profile(
    state, *, propose, admit, horizon: int = 3, rollouts: int = 32, rng=None
) -> dict:
    """Estimate `psi_h` by rolling the real kernel forward and counting survivors.

    `propose(state, rng)` returns candidate successor states as `(molecule, payload)`
    pairs; `admit(molecule)` is the exact task gate. A rollout dies the first time no
    proposed successor is admissible, which is the quantity that matters: a molecule whose
    corridor dead-ends one step out is a trap however good it looks now.
    """
    rng = rng or np.random.default_rng(0)
    alive_at = np.zeros(horizon + 1)
    alive_at[0] = rollouts
    depths = []
    for _ in range(rollouts):
        current, depth = state, 0
        for step in range(horizon):
            options = propose(current, rng)
            feasible = [m for m, _ in options if admit(m)]
            if not feasible:
                break
            current = feasible[int(rng.integers(len(feasible)))]
            depth = step + 1
            alive_at[depth] += 1
        depths.append(depth)
    return {
        "psi": {h: float(alive_at[h] / rollouts) for h in range(horizon + 1)},
        "mean_depth": float(np.mean(depths)),
        "dead_on_arrival": float(np.mean([d == 0 for d in depths])),
        "rollouts": rollouts,
        "horizon": horizon,
    }


def conditioned_weights(candidates, psi_values, *, temperature: float = 1.0) -> np.ndarray:
    """Reweight admissible candidates by their future feasibility.

    `psi` enters multiplicatively, which is what the Doob form requires: a candidate that
    is feasible now but dead one step out is suppressed relative to one that keeps the
    corridor open, without either being excluded outright.
    """
    psi = np.asarray(psi_values, dtype=float)
    if psi.size == 0:
        return psi
    if psi.max() <= 0:
        return np.full(psi.size, 1.0 / psi.size)
    weights = np.power(np.clip(psi, 1e-9, None), 1.0 / max(temperature, 1e-9))
    return weights / weights.sum()


def discriminates(psi_values, *, tolerance: float = 1e-6) -> dict:
    """Does `psi` separate candidates at all?

    If every admissible molecule has the same survival probability then conditioning on it
    is a no-op and the feasibility layer buys nothing. Reported explicitly so that a flat
    `psi` is recorded as a negative rather than silently multiplied through.
    """
    psi = np.asarray(psi_values, dtype=float)
    if psi.size < 2:
        return {"discriminates": False, "reason": "fewer than two candidates"}
    spread = float(psi.max() - psi.min())
    return {
        "discriminates": spread > tolerance,
        "spread": spread,
        "min": float(psi.min()),
        "median": float(np.median(psi)),
        "max": float(psi.max()),
        "fraction_dead": float(np.mean(psi <= tolerance)),
    }
