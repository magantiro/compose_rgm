"""Online PMO proposal and acquisition control from charged observations.

Learning targets are raw endpoint-minus-parent rewards. Separate pre-execution
and post-execution feature models inform proposal allocation and query selection.
Cross-parent decisions restore the parent offset. Acquisition clips predicted
endpoint utility to the PMO range and compares it with the current archive
threshold. Target standardization uses past charged observations only.

Frozen-reference preferences are optional and affect only the existing
exploration allocation. Their scoring step changes no model parameters, does
not replace proposal construction, and does not increase the query count.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from compose_v4.control.pmo_contextual_macro import (
    MACRO_FAMILIES,
    POST_EXECUTION_BLOCKS,
    PRE_EXECUTION_BLOCKS,
    ContextualMacroValue,
)
from compose_v4.control.reference_selection import PanelGuide, draw_exploration

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
    """Record per-family proposals, queries, rewards and allocation history.

    These diagnostics do not themselves alter proposal probabilities or train the
    value model. Preserve their distinction from policy floors and fitted features."""

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

    def state(self) -> dict:
        """Serialize counters and histories for exact controller restoration."""
        return {
            "proposed": dict(self.proposed),
            "queried": dict(self.queried),
            "deltas": {k: list(v) for k, v in self.deltas.items()},
            "predicted": {k: list(v) for k, v in self.predicted.items()},
            "mass_history": list(self.mass_history),
        }

    @classmethod
    def from_state(cls, payload: dict) -> FamilyLedger:
        return cls(
            proposed=Counter(payload.get("proposed") or {}),
            queried=Counter(payload.get("queried") or {}),
            deltas={k: list(v) for k, v in (payload.get("deltas") or {}).items()},
            predicted={k: list(v) for k, v in (payload.get("predicted") or {}).items()},
            mass_history=list(payload.get("mass_history") or []),
        )

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
                "p_delta_positive": (float(np.mean([d > 0 for d in deltas])) if deltas else None),
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
        """Learn from one charged reward delta. Restore the parent offset at selection."""
        parent_score = float(edge.get("parent_score", 0.0))
        delta = float(endpoint_score) - parent_score
        record = {
            **edge,
            "score": float(endpoint_score),  # raw endpoint utility
            "delta_raw": delta,  # raw learning target
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

    def _decision_utility(self, model, rows) -> np.ndarray:
        """Predicted endpoint utility CLIPPED to the PMO range, for ranking only.

        PMO rewards live in [0, 1] and an early ridge fit on a handful of observations can
        predict well outside it. Clipping the DECISION keeps one wild extrapolation from
        owning a whole batch. The stored reward labels are never touched -- `observe` keeps
        the raw delta, so the regression still sees the truth and the out-of-range count
        stays visible in telemetry rather than being normalised away.
        """
        return np.clip(self._endpoint_hat(model, rows), 0.0, 1.0)

    def _endpoint_hat(self, model, rows) -> np.ndarray:
        """Predicted ENDPOINT utility: parent offset restored, raw units, never delta alone."""
        parent = np.asarray([float(r.get("parent_score", 0.0)) for r in rows], dtype=float)
        if model.weights is None:
            return parent
        return parent + self.scale.restore(model.predict(rows))

    # ---- Persistence: an interrupted campaign must resume, not restart cold -------------
    def state(self) -> dict:
        """Everything needed to continue the SAME trajectory after a container restart.

        The value heads are NOT serialized because they do not need to be: `observe` refits
        both on the entire observation log, so the fitted state is a pure function of that
        log and is reproduced exactly by replaying it. The running scale is likewise
        replayed in order rather than stored, since Welford is order-dependent and the order
        is the log's. What genuinely cannot be recomputed -- the family ledger's counters
        and the policy history -- is carried verbatim.

        A resume that silently restarts the model cold is indistinguishable from a working
        resume in every artifact, which is the failure this exists to prevent.
        """
        return {
            "schema_version": "pmo_reward_adaptive_state_v1",
            "observations": list(self.observations),
            "ledger": self.ledger.state(),
            "policy_history": list(self.policy_history),
            "propensities": list(self.propensities),
        }

    @classmethod
    def restore(cls, payload: dict, **kwargs) -> RewardAdaptiveProgramController:
        """Rebuild by REPLAYING the observation log, then refitting once."""
        if payload.get("schema_version") != "pmo_reward_adaptive_state_v1":
            raise ValueError(f"unknown controller state {payload.get('schema_version')!r}")
        brain = cls(**kwargs)
        brain.observations = list(payload.get("observations") or [])
        brain.policy_history = list(payload.get("policy_history") or [])
        brain.propensities = list(payload.get("propensities") or [])
        brain.ledger = FamilyLedger.from_state(payload.get("ledger") or {})
        for record in brain.observations:
            brain.scale.update(float(record["delta_raw"]))
        if brain.fitted:
            targets = brain.scale.standardize([o["delta_raw"] for o in brain.observations])
            brain.pre.fit(brain.observations, targets)
            brain.post.fit(brain.observations, targets)
        return brain

    # ---- Upstream: reward changes what gets proposed -----------------------------------------
    def intent_policy(self, intents) -> np.ndarray:
        """Proposal mass over (parent, macro intent), floored per family and per lineage."""
        if not intents:
            return np.zeros(0)
        if not self.fitted:
            return np.full(len(intents), 1.0 / len(intents))
        value = self._decision_utility(self.pre, intents)
        weights = np.exp((value - value.max()) / self.temperature)
        weights = weights / weights.sum()

        # Floor on VOCABULARY entries only. A region replacement reports three labels
        # (`region_replace`, `region_replace:fuse_ring`, `fuse_ring`) where a plain module
        # reports one, so flooring every label would rescue it from starvation three times
        # and quietly favour one family -- a winner declared by bookkeeping rather than by
        # data. The compound refinement is a feature, not an allocation group.
        def groups(row):
            return [f for f in (row.get("families") or ()) if f in MACRO_FAMILIES]

        # SORTED, because each floor application MUTATES `weights`, so the result depends
        # on the order they are applied in. Iterating a set of strings makes that order
        # depend on PYTHONHASHSEED, which differs per process -- two identical runs then
        # land ~3e-3 apart on parent mass, and at ten thousand calls that eventually flips a
        # selection. Deterministic order, identical semantics.
        families = {f for row in intents for f in groups(row)}
        for family in sorted(families):
            members = [i for i, r in enumerate(intents) if family in groups(r)]
            share = float(weights[members].sum())
            if share < self.family_floor:
                weights[members] += (self.family_floor - share) / len(members)
        parents = {r["parent"] for r in intents}
        for parent in sorted(parents):
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
    def acquire(
        self,
        candidates,
        archive_threshold: float,
        *,
        batch: int,
        rng,
        reference_guide: PanelGuide | None = None,
        reference_receipts: list[dict] | None = None,
    ):
        """Rank by `max(endpoint_hat - tau, 0)`. Sample the remaining exploration slots."""
        if not candidates:
            return [], []
        room = min(batch, len(candidates))
        by_model = int(room * self.model_share) if self.fitted else 0
        chosen, detail = [], []
        if by_model:
            raw = self._endpoint_hat(self.post, candidates)
            predicted = np.clip(raw, 0.0, 1.0)
            utility = np.maximum(predicted - float(archive_threshold), 0.0)
            for index in np.argsort(-utility)[:by_model]:
                chosen.append(int(index))
                detail.append(
                    {
                        "index": int(index),
                        "reason": "model",
                        # Record raw predictions to expose out-of-range values.
                        # The clipped value determines the ranking.
                        "predicted_endpoint": float(raw[index]),
                        "decision_utility": float(predicted[index]),
                        "propensity": float(by_model / len(candidates)),
                    }
                )
        remaining = [i for i in range(len(candidates)) if i not in set(chosen)]
        take = min(room - len(chosen), len(remaining))
        if take > 0:
            draws = draw_exploration(
                [candidates[i] for i in remaining],
                take,
                rng,
                guide=reference_guide,
                receipts=reference_receipts,
            )
            receipt = reference_receipts[-1] if reference_guide is not None else None
            guided = receipt is not None and receipt["probabilities_changed"]
            for position, raw in enumerate(draws):
                index = remaining[int(raw)]
                chosen.append(index)
                detail.append(
                    {
                        "index": int(index),
                        "reason": "random" if self.fitted else "cold_start",
                        "propensity": None if guided else float(take / max(len(remaining), 1)),
                        **(
                            {
                                "reference_guided": True,
                                "conditional_draw_probability": receipt[
                                    "conditional_draw_probabilities"
                                ][position],
                                "propensity_semantics": "marginal inclusion not computed",
                            }
                            if guided
                            else {}
                        ),
                    }
                )
        self.propensities.extend(detail)
        self.ledger.propose(candidates)
        for entry in detail:
            self.ledger.query(candidates[entry["index"]], entry.get("predicted_endpoint"))
        return chosen, detail
