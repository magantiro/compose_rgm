"""Behaviour-policy future value over controller lineages.

This is NOT the committor. The committor asks what is reachable from a state; this asks
what the historical Dynamic search actually FOUND below a state, under its own selection
policy and its own proposal law. Where the search never looked, the label is zero
improvement, and a state that was genuinely promising but abandoned early is labelled
exactly like a barren one. Call it what it is -- best observed descendant improvement
under the behaviour policy -- and read every number below as conditional on that.

The label at horizon `k` is

    Y_k(s) = max(0, max{ R(s) - R(d) : d a descendant of s, q(s) < q(d) <= q(s) + k })

with `R` the docking score (lower is better, so a positive `Y` is a real improvement) and
`q` the charged-call index. Only descendants strictly after `s` count, and only features
computable at `q(s)` are offered to the model, so nothing in the fit can see its own
future.

The fits are ridge-penalised and convex: logistic for "does this lineage improve within
`k` calls", linear for how much. Both are scored against the baselines a controller
already has for free -- current score, momentum, depth, ancestral edits, parent
improvement rate, and the soft allocator's own weight -- because a future-value model
that cannot beat "the best molecule so far is the best bet" has not earned a place in
the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCHEMA_VERSION = "prospective_value_v1"

HORIZONS = (10, 25, 50, 100)


@dataclass(frozen=True)
class Fit:
    weights: np.ndarray
    names: tuple[str, ...]
    kind: str
    horizon: int
    converged: bool
    objective: float

    def score(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights


def standardize(train: np.ndarray, *matrices):
    """Centre and scale by the TRAINING fold only; a constant column is left alone."""
    mean, scale = train.mean(axis=0), train.std(axis=0)
    constant = scale < 1e-12
    mean = np.where(constant, 0.0, mean)
    scale = np.where(constant, 1.0, scale)
    return tuple((m - mean) / scale for m in (train, *matrices))


def fit_logistic(features, labels, *, penalty: float = 1.0, maxiter: int = 500) -> Fit:
    """Ridge logistic regression by L-BFGS. Convex, so no seed and no restarts."""
    from scipy.optimize import minimize

    labels = np.asarray(labels, dtype=float)

    def objective(w):
        z = features @ w
        # log(1 + e^z) computed stably for either sign of z.
        log_partition = np.logaddexp(0.0, z)
        loss = float(np.mean(log_partition - labels * z)) + penalty * float(w[:-1] @ w[:-1])
        probability = 1.0 / (1.0 + np.exp(-z))
        grad = features.T @ (probability - labels) / len(labels)
        grad[:-1] += 2 * penalty * w[:-1]
        return loss, grad

    result = minimize(
        objective,
        np.zeros(features.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )
    return Fit(result.x, (), "logistic", 0, bool(result.success), float(result.fun))


def fit_ridge(features, labels, *, penalty: float = 1.0) -> Fit:
    """Closed-form ridge. The bias column is not penalised."""
    labels = np.asarray(labels, dtype=float)
    dimension = features.shape[1]
    ridge = penalty * np.eye(dimension)
    ridge[-1, -1] = 0.0
    weights = np.linalg.solve(
        features.T @ features / len(labels) + ridge, features.T @ labels / len(labels)
    )
    residual = features @ weights - labels
    return Fit(weights, (), "ridge", 0, True, float(residual @ residual / len(labels)))


def spearman(predicted, actual) -> float:
    """Rank correlation, tie-averaged. Undefined for a constant column, reported as 0."""
    predicted, actual = np.asarray(predicted, float), np.asarray(actual, float)
    if predicted.size < 3 or predicted.std() < 1e-12 or actual.std() < 1e-12:
        return 0.0

    def ranks(values):
        order = values.argsort()
        out = np.empty_like(values, dtype=float)
        out[order] = np.arange(len(values), dtype=float)
        # average ties so a column of mostly-equal values cannot look informative
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inverse, out)
        return (sums / counts)[inverse]

    a, b = ranks(predicted), ranks(actual)
    return float(np.corrcoef(a, b)[0, 1])


def roc_auc(scores, labels) -> float:
    """Probability a positive outranks a negative, ties counted as half."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    positive, negative = scores[labels == 1], scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    order = np.argsort(scores)
    rank = np.empty(len(scores), float)
    rank[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, rank)
    rank = (sums / counts)[inverse]
    n_pos, n_neg = positive.size, negative.size
    return float((rank[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def capture_at_fraction(scores, realized, fraction: float = 0.1) -> float:
    """Share of all realized improvement sitting in the top `fraction` by score.

    This is the operational question a budget forces: spend the next calls on the states
    this law likes, and how much of the improvement that actually existed do you get?
    """
    scores, realized = np.asarray(scores, float), np.asarray(realized, float)
    total = realized.sum()
    if total <= 0 or scores.size == 0:
        return float("nan")
    keep = max(1, round(fraction * scores.size))
    chosen = np.argsort(-scores)[:keep]
    return float(realized[chosen].sum() / total)


def fold_averaged(folds, *, weighted: bool = True) -> dict:
    """Average a metric INSIDE each fold rather than pooling scores across folds.

    Pooling is wrong here and the error is silent. Each fold standardises on its own
    training rows and fits its own coefficients, so two folds' scores are not on a
    common scale; concatenating them and ranking globally mixes those scales and
    depresses every fitted arm while leaving raw single-column arms untouched. That
    produced three separate readings where a fitted model scored below a feature it
    contains -- impossible for a rank metric, since fitting one feature is a monotone
    transform of it. `test_fitting_one_feature_reproduces_that_features_own_ranking`
    pins the invariant.

    `folds` is an iterable of `(size, metric)` pairs.
    """
    entries = [(float(size), float(value)) for size, value in folds if np.isfinite(value)]
    if not entries:
        return {"value": float("nan"), "folds": 0}
    sizes = np.asarray([size for size, _ in entries])
    values = np.asarray([value for _, value in entries])
    weights = sizes / sizes.sum() if weighted else np.full(len(values), 1 / len(values))
    return {
        "value": float(np.dot(weights, values)),
        "folds": len(values),
        "per_fold_min": float(values.min()),
        "per_fold_max": float(values.max()),
    }
