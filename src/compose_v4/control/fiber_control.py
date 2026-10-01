"""Budgeted selection over feasible executable molecular programs.

The online model predicts improvement relative to a parent. Acquisition restores
the parent offset, applies a batch-diversity preference, and reserves exploration
slots. Optional frozen-reference guidance acts only on those exploration slots.
This allocator is neither an exact trajectory-law sampler nor a calibrated
uncertainty model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from compose_v4.control.reference_selection import PanelGuide, draw_exploration

SCHEMA_VERSION = "fiber_control_v1"

STOP = "stop"

# Reproducibility of ONE docking call on this target, measured: the benchmark root was
# submitted twice under identical requests and returned -8.00 and -8.50, and later -8.10.
# Use this only to interpret single-call margins, not as an exploration scale.
SINGLE_CALL_REPRODUCIBILITY = 0.5


@dataclass
class SearchState:
    """Everything the controller may condition on, and nothing it may not."""

    archive: dict[str, float] = field(default_factory=dict)  # docked molecule -> score
    budget: int = 0
    rounds: int = 0
    history: list = field(default_factory=list)

    @property
    def incumbent(self) -> float:
        return min(self.archive.values()) if self.archive else float("inf")

    def parents(self, *, limit: int, rng, explore: float = 0.3) -> list[str]:
        """Select archive parents with incumbent retention and random exploration."""
        if not self.archive:
            return []
        ordered = sorted(self.archive, key=self.archive.get)
        keep = max(1, round(limit * (1 - explore)))
        chosen = ordered[:keep]
        rest = list(ordered[keep:])
        if rest and limit > len(chosen):
            extra = min(limit - len(chosen), len(rest))
            chosen += [rest[i] for i in rng.choice(len(rest), extra, replace=False)]
        return chosen


def program_features(record: dict, state: SearchState) -> np.ndarray:
    """Encode program structure, endpoint eligibility margins and parent context."""
    parent = record.get("parent_score", 0.0)
    incumbent = state.incumbent if math.isfinite(state.incumbent) else parent
    families = record.get("families") or []
    return np.asarray(
        [
            parent,
            parent - incumbent,
            record.get("similarity", 0.0) - record.get("delta", 0.6),
            record.get("qed", 0.0),
            record.get("sa", 0.0) / 4.0,
            record.get("regions", 1),
            1.0 if record.get("regions", 1) > 1 else 0.0,
            record.get("created", 0) / 6.0,
            record.get("deleted", 0) / 6.0,
            1.0 if "retained_element" in families else 0.0,
            1.0 if "retained_deletion" in families else 0.0,
            1.0 if "scale" in families else 0.0,
            1.0 if "element" in families else 0.0,
            1.0 if "bond_order" in families else 0.0,
            1.0,
        ],
        dtype=float,
    )


class ProgramValue:
    """Ridge prediction of parent-relative improvement from charged outcomes.

    Features are standardized on observed training rows. The final column is an
    unpenalized intercept. The penalty is relative to mean squared loss. Acquisition
    restores the parent score before comparing candidates from different parents."""

    def __init__(self, *, penalty: float = 1.0):
        self.penalty = penalty
        self.weights: np.ndarray | None = None
        self.n = 0

    def fit(self, features, improvements) -> None:
        x = np.asarray(features, dtype=float)
        y = np.asarray(improvements, dtype=float)
        if x.shape[0] < 4:
            self.weights = None
            return
        # Do NOT centre a constant column. Centring the intercept sends it to zero,
        # which leaves an all-zero column that the unpenalised ridge entry cannot
        # regularise, and the solve is singular. This is the third appearance of this
        # exact bug in this work -- it also silently destroyed the macro-goal model's
        # ability to represent even the marginal. Constant columns keep their value.
        mean, scale = x.mean(axis=0), x.std(axis=0)
        constant = scale < 1e-12
        mean = np.where(constant, 0.0, mean)
        scale = np.where(constant, 1.0, scale)
        z = (x - mean) / scale
        ridge = self.penalty * np.eye(z.shape[1])
        ridge[-1, -1] = 0.0
        self.weights = np.linalg.solve(z.T @ z / len(y) + ridge, z.T @ y / len(y))
        self._mean, self._scale, self.n = mean, scale, len(y)

    def predict(self, features) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if self.weights is None:
            return np.zeros(x.shape[0])
        return ((x - self._mean) / self._scale) @ self.weights


def endpoint_utility(candidates, value: ProgramValue) -> np.ndarray:
    """Return negative predicted endpoint docking score, so larger is better.

    The model predicts parent score minus endpoint score. Subtracting the parent
    score restores a common cross-parent objective scale."""
    x = np.asarray([c["features"] for c in candidates])
    parent = np.asarray([c.get("parent_score", 0.0) for c in candidates], dtype=float)
    return value.predict(x) - parent


def acquisition(
    candidates,
    value: ProgramValue,
    state: SearchState,
    rng,
    *,
    beta: float = 1.0,
    batch: int = 8,
    diversity: float = 0.5,
    exploration: int = 2,
    reference_guide: PanelGuide | None = None,
    reference_receipts: list[dict] | None = None,
) -> list[int]:
    """Select a fixed-size batch using value, diversity and exploration.

    A fitted model supplies at most ``batch - exploration`` ranked choices. Each
    ranked choice is penalized by its largest fingerprint overlap with previously
    selected choices. Remaining slots are sampled without replacement. With no
    fitted model, the entire batch uses the exploration rule.

    Reference guidance may reweight only the exploration draw and requires a receipt
    sink. Off and shadow retain the original random calls. No calibrated uncertainty
    or exact Doob control is implied by the online predictor."""
    if not candidates:
        return []
    room = min(batch, len(candidates))
    by_model = max(0, room - exploration) if value.weights is not None else 0
    utility = endpoint_utility(candidates, value) if by_model else None

    chosen: list[int] = []
    for _ in range(by_model):
        best, best_score = None, -np.inf
        for i, candidate in enumerate(candidates):
            if i in chosen:
                continue
            penalty = 0.0
            for j in chosen:
                overlap = len(candidate["fingerprint"] & candidates[j]["fingerprint"])
                union = len(candidate["fingerprint"] | candidates[j]["fingerprint"]) or 1
                penalty = max(penalty, overlap / union)
            score = beta * utility[i] - diversity * penalty
            if score > best_score:
                best, best_score = i, score
        if best is None:
            break
        chosen.append(best)

    remaining = [i for i in range(len(candidates)) if i not in chosen]
    if remaining:
        extra = min(room - len(chosen), len(remaining))
        chosen += [
            remaining[i]
            for i in draw_exploration(
                [candidates[i] for i in remaining],
                extra,
                rng,
                guide=reference_guide,
                receipts=reference_receipts,
            )
        ]
    return chosen


def should_stop(
    state: SearchState,
    candidates,
    value: ProgramValue,
    *,
    threshold: float = 0.05,
    minimum_observations: int = 40,
    patience: int = 3,
) -> bool:
    """Stop at budget exhaustion, empty support, or persistent modeled shortfall.

    Model-based stopping requires enough observations and the configured number of
    rounds without improvement. Predictions are compared with the incumbent, not
    with each candidate parent. An unfitted model cannot trigger early stopping."""
    if state.budget <= 0 or not candidates:
        return True
    if value.weights is None or value.n < minimum_observations:
        return False
    improvements = [h.get("improved") for h in state.history[-patience:]]
    if len(improvements) < patience or any(improvements):
        return False
    parent = np.asarray([c.get("parent_score", 0.0) for c in candidates], dtype=float)
    predicted = value.predict(np.asarray([c["features"] for c in candidates]))
    # the question is whether anything is expected to beat the INCUMBENT, not whether
    # anything is expected to beat its own parent
    return float((parent - predicted).min()) > state.incumbent - threshold
