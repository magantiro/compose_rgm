"""PMO-native value over MACRO ACTIONS, learned from counted oracle outcomes.

WHY A DIFFERENT TARGET, measured rather than assumed.  The shared FiberControl value learns
realized improvement over the parent, which is the right local signal for T4.  PMO's blind
trajectories say that is not the quantity that governs progress here: a matched beam climbed
0.0496 -> 0.5172 through long stretches where the incumbent did not move at all while top-ten
quality and lineage DEPTH kept advancing, then a descendant paid off.  A one-step target cannot
express "this mediocre state is the gateway to a good one three decisions later", and the
benchmark itself scores the top-ten archive over the query trajectory rather than any single
parent-relative step.

So the reward here is ARCHIVE GAIN,

    r(x) = F_k(A + {x}) - F_k(A),      F_k = mean of the best k unique-endpoint utilities

which is aligned with what PMO actually rewards: a candidate is worth a call for what it adds to
the frontier, not for beating its own parent.  A child that improves a weak parent by a lot and
lands below the archive's tenth-best contributes nothing, and this target says so.

WHY THE ACTION, NOT THE MOLECULE.  COMPOSE acts through structured macros spanning roughly 8-18
primitive edits, and a 14-primitive ring replacement is not the same action as a 3-primitive
local growth.  Collapsing both into "a candidate molecule" throws away the structure the
proposal work just exposed.  Features therefore describe the (state, option) pair -- WHERE the
macro acted, WHAT family it was, HOW large and how retentive the compiled program is -- alongside
the search context the decision is actually made in: distance to the archive threshold, lineage
depth and remaining budget.

EXPLORATION COMES FROM UNCERTAINTY, NOT A CONSTANT.  The shipped allocator adds a fixed optimism
bonus; measured on real frozen pools, sweeping that constant from 0.25 to 0.0 moved selection by
less than a standard error, because in a freshly generated pool 95-97% of credit cells are
untried and a bonus applied to nearly everything is a constant offset that cannot reorder
anything.  Here the bonus is the model's own predictive spread, so it is large exactly where the
model has no evidence and vanishes where it does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from compose_v4.control.program_task import archive_top_k

#: The benchmark's archive width.  PMO scores the top ten.
DEFAULT_ARCHIVE_K = 10

#: Feature block names, in the order `macro_features` emits them.  Kept explicit so a model
#: fitted under one layout can never be scored under another.
FEATURE_NAMES: tuple[str, ...] = (
    "parent_score",
    "parent_minus_archive_threshold",
    "archive_headroom",
    "primitive_edits_scaled",
    "block_count_scaled",
    "heavy_atom_delta_scaled",
    "abs_heavy_atom_delta_scaled",
    "retention_realized",
    "retention_shortfall",
    "is_shallow_channel",
    "is_structured_channel",
    "is_jump_channel",
    "ring_ops_scaled",
    "insert_ops_scaled",
    "delete_ops_scaled",
    "lineage_depth_scaled",
    "budget_remaining_fraction",
    "bias",
)


def archive_gain(observations, candidate_utility: float, *, k: int = DEFAULT_ARCHIVE_K) -> float:
    """`F_k(A + {x}) - F_k(A)`: what this candidate adds to the scored frontier.

    ``observations`` is the counted archive as ``(endpoint, utility)`` pairs -- the same shape
    `archive_top_k` consumes, so the two can never disagree about what the archive is.  An empty
    archive gives the candidate its full value, which is correct: the first scored molecule
    defines the frontier.
    """
    if not math.isfinite(candidate_utility):
        raise ValueError("archive gain needs a finite candidate utility")
    rows = list(observations)
    before = archive_top_k(rows, k=k)
    after = archive_top_k([*rows, ("__candidate__", float(candidate_utility))], k=k)
    if before is None:
        return float(after)
    return float(after - before)


def archive_threshold(observations, *, k: int = DEFAULT_ARCHIVE_K) -> float:
    """The utility a candidate must exceed to enter the top-k at all.

    Below a full archive this is -inf: every candidate enters, so nothing is excluded yet.
    """
    labels: dict[str, list[float]] = {}
    for endpoint, utility in observations:
        labels.setdefault(endpoint, []).append(float(utility))
    if len(labels) < k:
        return -math.inf
    means = sorted((float(np.mean(v)) for v in labels.values()), reverse=True)
    return means[k - 1]


def macro_features(
    candidate,
    *,
    observations,
    lineage_depth: int = 0,
    budget_remaining_fraction: float = 1.0,
    k: int = DEFAULT_ARCHIVE_K,
) -> np.ndarray:
    """Describe the (state, MACRO) pair and the search context it is chosen in."""
    provenance = candidate.get("provenance", {})
    size = candidate.get("program_size", provenance.get("program_size", {})) or {}
    parent = provenance.get("parent_measured_score")
    parent_score = float(parent) if parent is not None else 0.0
    threshold = archive_threshold(observations, k=k)
    finite_threshold = threshold if math.isfinite(threshold) else parent_score
    best = archive_top_k(list(observations), k=1)
    headroom = (1.0 - float(best)) if best is not None else 1.0

    rules = candidate.get("program_rule_histogram") or {}
    if not rules:
        for action in (candidate.get("trace", {}) or {}).get("actions", ()) or ():
            name = action.get("rule") if isinstance(action, dict) else None
            if name:
                rules[name] = rules.get(name, 0) + 1
    ring = sum(
        rules.get(name, 0)
        for name in ("cycle_close", "cycle_open", "ring_system_restate")
    )
    channel = str(provenance.get("planner_channel", ""))
    retention = float(provenance.get("retention_realized", 1.0))
    target = float(provenance.get("retention_target", retention))
    primitives = float(size.get("primitive_edits", 0) or 0)
    heavy_delta = float(size.get("delta_from_measured_parent", 0) or 0)
    return np.asarray(
        [
            parent_score,
            parent_score - finite_threshold,
            headroom,
            primitives / 32.0,
            float(size.get("block_count", 0) or 0) / 8.0,
            heavy_delta / 16.0,
            abs(heavy_delta) / 16.0,
            retention,
            max(0.0, target - retention),
            float(channel == "shallow"),
            float(channel == "structured"),
            float(channel == "jump"),
            ring / 4.0,
            rules.get("atom_insert", 0) / 16.0,
            rules.get("atom_delete", 0) / 16.0,
            min(float(lineage_depth), 32.0) / 32.0,
            float(budget_remaining_fraction),
            1.0,
        ],
        dtype=float,
    )


@dataclass
class MacroValue:
    """Ridge over macro features with a predictive spread, fitted on counted outcomes only.

    The spread is the ridge posterior's, so it is wide where the design matrix has no support
    and narrows as evidence accumulates.  That is what replaces a fixed optimism constant: a
    bonus that is large only where the model genuinely does not know.
    """

    penalty: float = 1.0
    noise: float = 1.0
    weights: np.ndarray | None = None
    covariance: np.ndarray | None = None
    n: int = 0
    _mean: np.ndarray | None = field(default=None, repr=False)
    _scale: np.ndarray | None = field(default=None, repr=False)

    def fit(self, features, rewards) -> None:
        x = np.asarray(features, dtype=float)
        y = np.asarray(rewards, dtype=float)
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("macro value needs one reward per feature row")
        if x.shape[0] < 4:
            self.weights = self.covariance = None
            self.n = int(x.shape[0])
            return
        # Constant columns keep their value: centring an intercept to zero leaves a column the
        # unpenalised ridge entry cannot regularise and the solve goes singular. This exact bug
        # has appeared three times in this codebase.
        mean, scale = x.mean(axis=0), x.std(axis=0)
        constant = scale < 1e-12
        mean = np.where(constant, 0.0, mean)
        scale = np.where(constant, 1.0, scale)
        z = (x - mean) / scale
        ridge = self.penalty * np.eye(z.shape[1])
        ridge[-1, -1] = 0.0
        precision = z.T @ z + ridge * len(y)
        self.covariance = np.linalg.inv(precision) * self.noise
        self.weights = np.linalg.solve(precision, z.T @ y)
        self._mean, self._scale, self.n = mean, scale, len(y)

    def _design(self, features) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return (x - self._mean) / self._scale

    def predict(self, features) -> np.ndarray:
        if self.weights is None:
            return np.zeros(np.atleast_2d(np.asarray(features, dtype=float)).shape[0])
        return self._design(features) @ self.weights

    def spread(self, features) -> np.ndarray:
        """Predictive standard deviation; uniform and positive while unfitted."""
        rows = np.atleast_2d(np.asarray(features, dtype=float)).shape[0]
        if self.covariance is None:
            return np.ones(rows)
        z = self._design(features)
        return np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", z, self.covariance, z)))

    def acquire(self, features, *, beta: float = 1.0) -> np.ndarray:
        """`mu + beta * sigma`: exploration scaled by what the model does not know."""
        return self.predict(features) + float(beta) * self.spread(features)
