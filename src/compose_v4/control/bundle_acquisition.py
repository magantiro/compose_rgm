"""Finite-budget experiment selection by depth-3 belief-space rollout.

The acquisition is not `expected improvement plus lambda times information`. That form
needs a weight nobody has derived, and an arbitrary exploration constant of exactly that
kind already dominated behaviour once on this branch. Instead the value of an experiment
is read off the finite-budget recursion directly:

    Q_b(D, E) = E_{Y_E} [ V_{b-|E|}( D union (E, Y_E) ) ]

approximated by simulating the outcomes, updating the belief, and rolling the resulting
policy forward a few option decisions. Information value then appears because a different
`Y_E` leads to a different continuation, not because a term was added for it.

Two horizons are kept apart, because they are different clocks and conflating them is how
this gets expensive for no reason:

- INNER, over molecular options: measured necessary depth is about 3. Beyond 3 the
  credit signal saturates (rank correlation with one-step credit is 0.820 at h=3 against
  0.785 at h=8) and one step alone misses 17.2% of winner ancestors.
- OUTER, over docking batches: ONE step. Nothing measured supports planning three
  recursive experiments ahead, and the prior is worse than uniform on two of five
  targets, so compounding rollout error is a real cost.

The rollout prior is the length-2 option transition table. The length-3 table rests on 19
observations with one to five per cell and would be noise, so a depth-3 rollout ITERATES
the length-2 matrix rather than consulting learned longer motifs.

Computation here is cheap and docking is not, so spending many posterior samples per
candidate bundle is the correct trade.
"""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "bundle_acquisition_v1"

INNER_DEPTH = 3
OUTER_DEPTH = 1

# Measured length-2 option transitions over the 77 routes (70 observations).
# G = global remodel, L = local refinement. Used ONLY as a rollout prior.
TRANSITION_PRIOR = {
    "G": {"L": 0.357 / (0.357 + 0.214), "G": 0.214 / (0.357 + 0.214)},
    "L": {"L": 0.229 / (0.229 + 0.200), "G": 0.200 / (0.229 + 0.200)},
}


def simulate_outcomes(belief, bundle, parent_score: float, rng) -> np.ndarray:
    """One posterior draw of what this bundle would measure.

    Latent effect from the belief, plus measurement noise, so a simulated outcome is a
    measurement rather than a truth -- which is what the controller will actually see.
    """
    out = np.empty(len(bundle))
    for i, option in enumerate(bundle):
        effect = belief.sample(option["coordinate"], option["value"], rng)
        out[i] = parent_score - effect + rng.normal(0.0, belief.noise)
    return out


def _rollout(belief, incumbent: float, lane: str, depth: int, coordinates, rng) -> float:
    """Best latent utility reachable in `depth` further option decisions.

    Each step samples a lane from the transition prior, picks the coordinate the belief
    currently likes best in that lane, and advances the incumbent by the sampled latent
    effect. No measurement noise is added: the rollout estimates what is achievable, and
    the noise has already entered through the belief's own width.
    """
    best = incumbent
    for _ in range(depth):
        options = coordinates.get(lane) or []
        if not options:
            break
        gains = [belief.sample(c, v, rng) for c, v in options]
        index = int(np.argmax(gains))
        best = min(best, best - max(gains[index], 0.0))
        choices, weights = zip(*TRANSITION_PRIOR[lane].items())
        lane = str(rng.choice(choices, p=np.asarray(weights, dtype=float)))
    return best


def bundle_value(
    belief,
    bundle,
    *,
    parent_score: float,
    incumbent: float,
    coordinates,
    rng,
    samples: int = 256,
    depth: int = INNER_DEPTH,
) -> dict:
    """Q(E): expected best latent utility after running this bundle and continuing.

    Averages over posterior outcomes of the bundle. For each draw the belief is updated
    on what the bundle would have shown -- as CONTRASTS between its own arms, which is
    what a matched bundle provides -- and the continuation is rolled forward.
    """
    from copy import deepcopy

    terminal = np.empty(samples)
    direct = np.empty(samples)
    for s in range(samples):
        outcomes = simulate_outcomes(belief, bundle, parent_score, rng)
        direct[s] = min(incumbent, outcomes.min())

        trial = deepcopy(belief)
        order = np.argsort(outcomes)
        best_arm, worst_arm = bundle[order[0]], bundle[order[-1]]
        if best_arm["coordinate"] == worst_arm["coordinate"] and len(bundle) > 1:
            trial.observe_contrast(
                best_arm["coordinate"],
                best_arm["value"],
                worst_arm["value"],
                float(outcomes[order[-1]] - outcomes[order[0]]),
            )
        else:
            for option, y in zip(bundle, outcomes):
                trial.observe_absolute(
                    option["coordinate"], option["value"], float(parent_score - y)
                )
        lane = bundle[0].get("lane", "G")
        terminal[s] = _rollout(trial, direct[s], lane, depth, coordinates, rng)

    return {
        "value": float(terminal.mean()),
        "direct_only": float(direct.mean()),
        "continuation_gain": float(direct.mean() - terminal.mean()),
        "samples": samples,
        "depth": depth,
    }


def select_bundle(
    belief,
    candidates,
    *,
    parent_score: float,
    incumbent: float,
    coordinates,
    rng,
    samples: int = 256,
) -> dict:
    """Choose the experiment with the best finite-budget value. Execute only this one."""
    if not candidates:
        raise ValueError("no candidate bundle to select")
    scored = []
    for bundle in candidates:
        report = bundle_value(
            belief,
            bundle,
            parent_score=parent_score,
            incumbent=incumbent,
            coordinates=coordinates,
            rng=rng,
            samples=samples,
        )
        scored.append((report["value"], bundle, report))
    scored.sort(key=lambda t: t[0])
    value, bundle, report = scored[0]
    return {
        "bundle": bundle,
        "value": value,
        "report": report,
        "considered": len(candidates),
        "runner_up_gap": float(scored[1][0] - value) if len(scored) > 1 else float("nan"),
    }
