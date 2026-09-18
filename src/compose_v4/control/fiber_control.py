"""FiberControl: finite-budget reward-tilted control over feasible COMPOSE programs.

The objective is one equation, not a stack of heuristics:

    P*(tau)  ∝  P0(tau) * 1[tau queryable] * exp(beta * J(tau))

`P0` is the production program process, the indicator is the exact T4 endpoint gate, and
`J` is the finite-budget objective -- the best docking score found within the remaining
oracle calls. Feasibility constrains the SUPPORT. Reward decides preference inside it.

Two design decisions are measurements rather than taste.

FEASIBILITY IS NOT A REWARD TERM. An earlier form multiplied the policy by a future
survival probability `psi`. Measured over 14 delta=0.6 feasible JAK2 molecules,
`corr(psi_1, docking score) = +0.424`, and because better docking is more negative that
means higher survival goes with WORSE binding: the best molecule had the lowest survival
(0.08, 92% dead on arrival) and the weakest had the highest (0.50). Multiplying by `psi`
would steer away from the good corner. Strong solutions sit on the constraint boundary --
median similarity margin +0.019 at delta=0.6, several at exactly 0.600 -- and spending
that margin is how a molecule got good.

THERE IS AN EXPLICIT STOP. Because of the above, a state can be excellent and nearly
terminal: 71% of profiled molecules have zero feasible continuations at depth three. A
controller without STOP is forced to treat a dead end as a failure rather than as a place
to finish, and would abandon the best molecule it has.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SCHEMA_VERSION = "fiber_control_v1"

STOP = "stop"

# Reproducibility of one docking call on this target, measured: the benchmark root was
# submitted twice under identical requests and returned -8.00 and -8.50.
NOISE_FLOOR = 0.5


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
        """Frontier to expand from: mostly the best, deliberately not only the best.

        Passive best-first allocation was measured to miss the productive branch on 8 of
        11 historical runs, where the winning branch ranked 3rd to 12th at decision time.
        So a share of the frontier is drawn away from the incumbent on purpose.
        """
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
    """Describe the DECISION, not the molecule.

    A generic whole-chemical-space docking surrogate was measured to fail here: ranking
    held-out JAK2 siblings it reached 0.498 against 0.404 for choosing at random. What
    this model has to answer is narrower and better matched to the data actually
    collected -- which structural programs work for THIS target from THIS context.
    """
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
    """Ridge model of realized improvement, fitted only on counted docking outcomes.

    The target is the improvement over the parent, not the endpoint score, because the
    parent term cancels: siblings of one parent share it exactly, so a contrast between
    them identifies the program's own effect with no confound. The parent offset is NOT
    lost -- `acquisition` adds it back to recover the endpoint score, which is what the
    objective is written over.
    """

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


def acquisition(
    candidates,
    value: ProgramValue,
    state: SearchState,
    rng,
    *,
    beta: float = 1.0,
    batch: int = 8,
    diversity: float = 0.5,
) -> list[int]:
    """Choose the next docking batch, or STOP.

    RANK BY THE OBJECTIVE, NOT BY THE GAIN. `J` is the best endpoint score found inside
    the budget, so a candidate is worth a call for its predicted ENDPOINT score
    `parent_score - gain`, never for its predicted gain alone. Ranking on gain alone is
    a real defect and it was caught mid-campaign: improvement over a weak parent is
    easier than improvement over a strong one, so the rule systematically bought
    candidates from the worst parents on the frontier. Round four of the first
    autonomous run docked endpoints at -6.8, -7.0 and -7.3 while the incumbent stood at
    -9.10, and the run gained 0.30 over its last 24 calls.

    Optimism is Thompson-style: a value model fitted on few observations is sampled
    rather than trusted. The noise floor is a measurement, not a taste -- the same root
    molecule submitted twice under identical requests came back -8.00 and -8.50, so a
    ranking margin under about half a unit is not a real ordering and exploration must
    still be able to overturn it. Diversity is enforced structurally, because a batch of
    near-identical candidates buys one observation at the price of several calls.
    """
    if not candidates:
        return []
    x = np.asarray([c["features"] for c in candidates])
    parent = np.asarray([c.get("parent_score", 0.0) for c in candidates], dtype=float)
    predicted = value.predict(x)
    spread = float(np.std(predicted)) if value.weights is not None else 1.0
    sampled = predicted + rng.normal(0.0, max(spread, NOISE_FLOOR), size=predicted.shape)
    # utility is minus the predicted endpoint score, so larger is better throughout
    sampled = sampled - parent

    chosen, taken = [], []
    for _ in range(min(batch, len(candidates))):
        best, best_score = None, -np.inf
        for i, candidate in enumerate(candidates):
            if i in chosen:
                continue
            penalty = 0.0
            for j in taken:
                overlap = len(candidate["fingerprint"] & candidates[j]["fingerprint"])
                union = len(candidate["fingerprint"] | candidates[j]["fingerprint"]) or 1
                penalty = max(penalty, overlap / union)
            score = beta * sampled[i] - diversity * penalty
            if score > best_score:
                best, best_score = i, score
        if best is None:
            break
        chosen.append(best)
        taken.append(best)
    return chosen


def should_stop(state: SearchState, candidates, value: ProgramValue, *,
                threshold: float = 0.05, minimum_observations: int = 40,
                patience: int = 3) -> bool:
    """Terminate only on evidence, never on an unfitted model's opinion.

    This is the decision that makes a boundary state usable: a molecule with no feasible
    continuation is not a failed branch, it may be where the run should end. But the first
    version stopped a campaign after 17 of 72 calls at -9.00, because a ridge model fitted
    on 16 observations predicted no candidate would improve. A model that thin has no
    opinion worth acting on, and -11.00 was known to exist.

    So stopping now requires a fitted model (`minimum_observations`), agreement that
    nothing is promising, AND `patience` consecutive rounds without a real improvement.
    Budget exhaustion and an empty candidate set remain immediate.
    """
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
