"""A coupled site-and-mode proposal law for constructive edits.

The production sampler reproduced zero teacher constructions in 11,762 proposals while
reproducing destructive ones at 4.5-10.8%, with every family compiling at nearly every
state. So the vocabulary is present and the joint choice of *where* to build and *what
kind* of construction to make carries no mass.

Two failure modes bracket the design, and both are measured rather than assumed.
Sampling site and mode independently is what produces the zero. Memorising joint
identities is what produced zero held-source recovery in the earlier whole-template
gate: over the 77 routes only 46% of exact `(mode, site-context)` pairs recur across
cells, against 80% for mode alone and 74% for site alone.

So the score is additive with an interaction rather than either extreme:

    S(G, r, m) = A(G, r) + B(G, m) + C(G, r, m)

normalised over the legal pairs at G. `A` and `B` pool statistical strength across all
sites and all modes; `C(G,r,m) = x_r^T W z_m` carries the coupling and is ridged hard,
because 105 examples cannot support 59 free joint bins. With little evidence the law
falls back smoothly toward the pooled effects instead of off a categorical cliff; where
the data repeatedly says a construction belongs at a kind of site, `W` earns its weight.

`W` is a FULL matrix, not a low-rank factorisation. That keeps the score linear in
`(a, b, W)`, so ridge-penalised softmax cross-entropy is convex with a unique optimum.
A rank-k factorisation `W = U^T V` is not: `dL/dU` is proportional to `V` and vice
versa, so `U = V = 0` is a stationary point, and fitted from a small random start the
factors stayed at 4e-5 -- silently reducing the law to `A + B`, the independent-marginal
model this design exists to avoid. The pathology was invisible in the held-out scores,
which were byte-identical across rank 1 to 4 and across two orders of interaction
penalty. Do not reintroduce the factorisation to "save parameters": 216 ridged
coefficients under a convex objective are cheaper to trust than 116 under a saddle.

Fitting is L-BFGS in numpy, so there is no autograd state to leak and the fit is
deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCHEMA_VERSION = "constructive_policy_v1"


@dataclass(frozen=True)
class PolicyShape:
    site_features: int
    mode_features: int

    def __post_init__(self):
        if min(self.site_features, self.mode_features) < 1:
            raise ValueError("policy dimensions must be positive")

    @property
    def size(self) -> int:
        return self.site_features + self.mode_features + self.site_features * self.mode_features


def unpack(theta: np.ndarray, shape: PolicyShape):
    """Split the flat parameter vector into the pooled terms and the interaction."""
    ds, dm = shape.site_features, shape.mode_features
    if theta.shape != (shape.size,):
        raise ValueError(f"parameter vector must have length {shape.size}, got {theta.shape}")
    a = theta[:ds]
    b = theta[ds : ds + dm]
    w = theta[ds + dm :].reshape(ds, dm)
    return a, b, w


def pair_scores(theta, shape, site_matrix, mode_matrix) -> np.ndarray:
    """S(G, r, m) for every legal pair, shaped (sites, modes)."""
    a, b, w = unpack(theta, shape)
    general_site = site_matrix @ a
    general_mode = mode_matrix @ b
    interaction = site_matrix @ w @ mode_matrix.T
    return general_site[:, None] + general_mode[None, :] + interaction


def _example_loss(theta, shape, example, penalties):
    """Softmax cross-entropy at one decision, plus its share of the penalty."""
    scores = pair_scores(theta, shape, example["sites"], example["modes"])
    flat = scores.reshape(-1)
    target = example["site_index"] * example["modes"].shape[0] + example["mode_index"]
    shift = flat.max()
    exp = np.exp(flat - shift)
    total = exp.sum()
    loss = shift + np.log(total) - flat[target]
    probability = exp / total
    grad_flat = probability.copy()
    grad_flat[target] -= 1.0
    grad_scores = grad_flat.reshape(scores.shape)

    site_matrix, mode_matrix = example["sites"], example["modes"]
    grad_a = site_matrix.T @ grad_scores.sum(axis=1)
    grad_b = mode_matrix.T @ grad_scores.sum(axis=0)
    grad_w = site_matrix.T @ grad_scores @ mode_matrix
    return loss, np.concatenate([grad_a, grad_b, grad_w.reshape(-1)])


def objective(theta, shape, examples, penalties):
    """Mean cross-entropy with separate ridge penalties on the three terms."""
    total, grad = 0.0, np.zeros_like(theta)
    for example in examples:
        loss, piece = _example_loss(theta, shape, example, penalties)
        total += loss
        grad += piece
    total /= max(len(examples), 1)
    grad /= max(len(examples), 1)
    a, b, w = unpack(theta, shape)
    ridge = (
        penalties["site"] * float(a @ a)
        + penalties["mode"] * float(b @ b)
        + penalties["interaction"] * float((w * w).sum())
    )
    grad_ridge = np.concatenate(
        [
            2 * penalties["site"] * a,
            2 * penalties["mode"] * b,
            2 * penalties["interaction"] * w.reshape(-1),
        ]
    )
    return total + ridge, grad + grad_ridge


def fit(examples, shape, *, penalties=None, maxiter: int = 2000):
    """Fit the coupled law. Penalties default to a hard shrink on the interaction.

    The objective is convex in `(a, b, W)`, so the start point is the origin and there is
    no seed: two fits of the same examples agree exactly.
    """
    from scipy.optimize import minimize

    if not examples:
        raise ValueError("cannot fit a constructive policy without examples")
    penalties = penalties or {"site": 1e-3, "mode": 1e-3, "interaction": 1.0}
    if penalties["interaction"] < max(penalties["site"], penalties["mode"]):
        raise ValueError(
            "the interaction term must be shrunk at least as hard as the pooled terms; "
            "the joint data is far sparser than either marginal"
        )
    result = minimize(
        objective,
        np.zeros(shape.size),
        args=(shape, examples, penalties),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-10},
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "theta": result.x,
        "shape": shape,
        "penalties": penalties,
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "converged": bool(result.success),
        "examples": len(examples),
        "interaction_norm": float(np.linalg.norm(unpack(result.x, shape)[2])),
    }


def rank_of_truth(model, example) -> dict:
    """Held-out ranks: the joint pair, the site, and the mode given the true site."""
    scores = pair_scores(model["theta"], model["shape"], example["sites"], example["modes"])
    n_modes = example["modes"].shape[0]
    target = example["site_index"] * n_modes + example["mode_index"]
    flat = scores.reshape(-1)
    joint_rank = int((flat > flat[target]).sum())
    site_scores = scores.max(axis=1)
    site_rank = int((site_scores > site_scores[example["site_index"]]).sum())
    row = scores[example["site_index"]]
    mode_rank = int((row > row[example["mode_index"]]).sum())
    return {
        "joint_rank": joint_rank,
        "joint_candidates": flat.size,
        "site_rank": site_rank,
        "site_candidates": scores.shape[0],
        "mode_rank_given_site": mode_rank,
        "mode_candidates": n_modes,
    }
