"""Reward-adaptive PMO control: counted outcomes change what gets PROPOSED, not only what is bought.

THE CAUSAL STRUCTURE THIS TRANSFERS FROM T4

    counted oracle outcome -> realized action reward -> update ProgramValue online
        -> change which actions are taken next round -> retain explicit random exploration

A predictor that filters molecules after COMPOSE has already decided what to generate is not that
loop.  Measured: substituting a strong contextual ranker into acquisition alone moved AUC by 0.001
at 250 calls, because the controller queried half of everything it generated -- there was nothing
to select.  The loop has to close upstream.

THE REWARD DEFINITION, which is the T4 lesson worth preserving most aggressively.

    LEARNING TARGET      delta = u(child) - u(parent)
    CROSS-PARENT VALUE   endpoint_hat = u(parent) + delta_hat

Learning on the delta gives a dense action-level signal: siblings of one parent share the parent
term exactly, so a contrast between them identifies the macro's own effect with no confounding.
But ranking DIFFERENT parents' candidates by predicted delta alone is a known defect -- improvement
over a weak parent is easier than over a strong one, so the rule systematically buys from the worst
parents on the frontier.  T4 caught that mid-campaign.  The parent offset is added back before any
cross-parent comparison, which recovers the quantity the objective is actually written over.

WHAT IS DELIBERATELY NOT THE TARGET.  Not headroom-normalised delta, not percentage improvement,
not pool-relative rescaling, and not top-10 archive gain.  Archive gain is a DECISION UTILITY
computed at acquisition time from the predicted endpoint against the live threshold; training on it
directly discards every sub-threshold observation, and once the archive is good that is most of
them.  Raw endpoint score, raw delta and archive gain are stored separately and never conflated.

Standardisation of the regression target uses the mean and standard deviation of PAST CHARGED
observations only -- never future rounds, never the current pool -- and predictions are returned to
raw delta units before acquisition sees them.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from compose_v4.control.pmo_contextual_macro import (
    POST_EXECUTION_BLOCKS,
    PRE_EXECUTION_BLOCKS,
    ContextualMacroValue,
)

#: T4's split: most of the batch to the model, a quarter held back for genuine exploration.
#: Every random call is still a legitimate reward observation and trains both heads afterwards.
MODEL_SHARE = 0.75

#: Below this many counted observations both heads are unfitted and selection is uniform. An
#: unfitted model has no opinion, and acting on one is how an early accident becomes permanent.
MINIMUM_OBSERVATIONS = 24

#: No live lineage may fall below this share of proposal compute, and no legal macro family may
#: fall to zero support. A previous PMO run put 99% of scored descendants on a single seed.
LINEAGE_FLOOR = 0.10
FAMILY_FLOOR = 0.02


@dataclass
class RunningScale:
    """Mean and standard deviation of past charged deltas, for conditioning only."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return max(math.sqrt(self.m2 / (self.n - 1)), 1e-6)

    def standardize(self, value):
        return (np.asarray(value, dtype=float) - self.mean) / self.std

    def restore(self, value):
        return np.asarray(value, dtype=float) * self.std + self.mean


@dataclass
class FamilyLedger:
    """Per-macro-family audit, logged rather than acted on.

    Kept deliberately OUTSIDE the value model so it cannot become a control surface.  No
    family is declared a winner in advance and none is floored or boosted from these
    numbers; they exist so that after a scored run we can read whether the controller
    discovered a preference -- region replacements in one phase, small growth in another
    -- instead of asserting one.  ``mass_history`` snapshots each family's proposal mass
    per round, so the change in its proposal probability AFTER reward is recoverable,
    which is what distinguishes a controller that learned from one that merely ran.
    """

    proposed: Counter = field(default_factory=Counter)
    queried: Counter = field(default_factory=Counter)
    deltas: dict = field(default_factory=dict)
    predicted: dict = field(default_factory=dict)
    mass_history: list = field(default_factory=list)

    def propose(self, rows) -> None:
        for row in rows:
            for family in row.get("families") or ():
                self.proposed[family] += 1

    def query(self, row, predicted_value=None) -> None:
        for family in row.get("families") or ():
            self.queried[family] += 1
            if predicted_value is not None:
                self.predicted.setdefault(family, []).append(float(predicted_value))

    def outcome(self, row, delta: float) -> None:
        for family in row.get("families") or ():
            self.deltas.setdefault(family, []).append(float(delta))

    def snapshot(self, rows, weights, label: str) -> None:
        """Record each family's total proposal mass under the CURRENT policy."""
        mass: dict = {}
        for row, weight in zip(rows, weights, strict=True):
            for family in row.get("families") or ():
                mass[family] = mass.get(family, 0.0) + float(weight)
        self.mass_history.append({"label": label, "mass": mass})

    def report(self) -> dict:
        families = sorted(set(self.proposed) | set(self.queried) | set(self.deltas))
        first = self.mass_history[0]["mass"] if self.mass_history else {}
        last = self.mass_history[-1]["mass"] if self.mass_history else {}
        rows = {}
        for family in families:
            deltas = self.deltas.get(family, [])
            predicted = self.predicted.get(family, [])
            rows[family] = {
                "n_proposed": int(self.proposed.get(family, 0)),
                "n_queried": int(self.queried.get(family, 0)),
                "expected_delta_u": float(np.mean(deltas)) if deltas else None,
                "p_delta_positive": (
                    float(np.mean([d > 0 for d in deltas])) if deltas else None
                ),
                "mean_predicted_value": float(np.mean(predicted)) if predicted else None,
                "proposal_mass_first": float(first.get(family, 0.0)),
                "proposal_mass_last": float(last.get(family, 0.0)),
            }
            rows[family]["proposal_mass_shift"] = (
                rows[family]["proposal_mass_last"] - rows[family]["proposal_mass_first"]
            )
        return {"families": rows, "rounds_snapshotted": len(self.mass_history)}


@dataclass
class RewardAdaptiveProgramController:
    """Online action-value control over COMPOSE macros, learned from charged PMO calls only."""

    model_share: float = MODEL_SHARE
    ledger: FamilyLedger = field(default_factory=FamilyLedger)
    lineage_floor: float = LINEAGE_FLOOR
    family_floor: float = FAMILY_FLOOR
    temperature: float = 0.05
    pre: ContextualMacroValue = field(
        default_factory=lambda: ContextualMacroValue(blocks=PRE_EXECUTION_BLOCKS)
    )
    post: ContextualMacroValue = field(
        default_factory=lambda: ContextualMacroValue(blocks=POST_EXECUTION_BLOCKS)
    )
    scale: RunningScale = field(default_factory=RunningScale)
    observations: list = field(default_factory=list)
    propensities: list = field(default_factory=list)
    policy_history: list = field(default_factory=list)

    @property
    def fitted(self) -> bool:
        return len(self.observations) >= MINIMUM_OBSERVATIONS

    # ---- Online learning -------------------------------------------------------------------
    def observe(self, edge, endpoint_score: float) -> None:
        """One CHARGED outcome. Trains on the DELTA; the parent offset is restored at decision."""
        parent_score = float(edge.get("parent_score", 0.0))
        delta = float(endpoint_score) - parent_score
        record = {
            **edge,
            "score": float(endpoint_score),      # raw endpoint utility
            "delta_raw": delta,                  # raw learning target
            "parent_score": parent_score,
        }
        self.observations.append(record)
        self.scale.update(delta)
        self.ledger.outcome(record, delta)
        if not self.fitted:
            return
        targets = self.scale.standardize([o["delta_raw"] for o in self.observations])
        self.pre.fit(self.observations, targets)
        self.post.fit(self.observations, targets)

    def _endpoint_hat(self, model, rows) -> np.ndarray:
        """Predicted ENDPOINT utility: parent offset restored, raw units, never delta alone."""
        parent = np.asarray([float(r.get("parent_score", 0.0)) for r in rows], dtype=float)
        if model.weights is None:
            return parent
        return parent + self.scale.restore(model.predict(rows))

    # ---- Upstream: reward changes what gets proposed -----------------------------------------
    def intent_policy(self, intents) -> np.ndarray:
        """Proposal mass over (parent, macro intent), floored per family and per lineage."""
        if not intents:
            return np.zeros(0)
        if not self.fitted:
            return np.full(len(intents), 1.0 / len(intents))
        value = self._endpoint_hat(self.pre, intents)
        weights = np.exp((value - value.max()) / self.temperature)
        weights = weights / weights.sum()
        families = {f for row in intents for f in (row.get("families") or ())}
        for family in families:
            members = [i for i, r in enumerate(intents) if family in (r.get("families") or ())]
            share = float(weights[members].sum())
            if share < self.family_floor:
                weights[members] += (self.family_floor - share) / len(members)
        parents = {r["parent"] for r in intents}
        for parent in parents:
            members = [i for i, r in enumerate(intents) if r["parent"] == parent]
            share = float(weights[members].sum())
            floor = self.lineage_floor / len(parents)
            if share < floor:
                weights[members] += (floor - share) / len(members)
        return weights / weights.sum()

    def record_policy(self, intents, label: str) -> dict:
        """Snapshot the proposal policy so a reward's effect on it is measurable, not asserted."""
        weights = self.intent_policy(intents)
        parents: dict[str, float] = {}
        for row, weight in zip(intents, weights, strict=True):
            parents[row["parent"]] = parents.get(row["parent"], 0.0) + float(weight)
        snapshot = {
            "label": label,
            "observations": len(self.observations),
            "fitted": self.fitted,
            "parent_mass": parents,
        }
        self.policy_history.append(snapshot)
        self.ledger.snapshot(intents, weights, label)
        return snapshot

    @staticmethod
    def policy_shift(before: dict, after: dict) -> float:
        """Total-variation distance between two parent allocations."""
        keys = set(before["parent_mass"]) | set(after["parent_mass"])
        return 0.5 * sum(
            abs(after["parent_mass"].get(k, 0.0) - before["parent_mass"].get(k, 0.0)) for k in keys
        )

    # ---- Downstream: which realized molecules deserve a call ---------------------------------
    def acquire(self, candidates, archive_threshold: float, *, batch: int, rng):
        """`max(endpoint_hat - tau, 0)` on the model share; the remainder drawn at random."""
        if not candidates:
            return [], []
        room = min(batch, len(candidates))
        by_model = int(room * self.model_share) if self.fitted else 0
        chosen, detail = [], []
        if by_model:
            predicted = self._endpoint_hat(self.post, candidates)
            utility = np.maximum(predicted - float(archive_threshold), 0.0)
            for index in np.argsort(-utility)[:by_model]:
                chosen.append(int(index))
                detail.append(
                    {
                        "index": int(index),
                        "reason": "model",
                        "predicted_endpoint": float(predicted[index]),
                        "propensity": float(by_model / len(candidates)),
                    }
                )
        remaining = [i for i in range(len(candidates)) if i not in set(chosen)]
        take = min(room - len(chosen), len(remaining))
        if take > 0:
            for raw in np.atleast_1d(rng.choice(len(remaining), take, replace=False)):
                index = remaining[int(raw)]
                chosen.append(index)
                detail.append(
                    {
                        "index": int(index),
                        "reason": "random" if self.fitted else "cold_start",
                        "propensity": float(take / max(len(remaining), 1)),
                    }
                )
        self.propensities.extend(detail)
        self.ledger.propose(candidates)
        for entry in detail:
            self.ledger.query(
                candidates[entry["index"]], entry.get("predicted_endpoint")
            )
        return chosen, detail
