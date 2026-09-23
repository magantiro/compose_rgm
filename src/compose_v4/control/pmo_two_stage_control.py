"""Two-stage reward-adaptive PMO control: allocate cheap generation, then spend scarce calls.

THE SPLIT IS FORCED BY WHAT IS KNOWABLE WHEN, not by taste.

    Q_pre(parent, macro intent)      decides where to spend PROPOSAL COMPUTE.  It may read only
                                     what exists before COMPOSE executes anything.
    Q_post(parent, macro, endpoint)  decides where to spend ORACLE CALLS.  It may read the
                                     realized molecule, because generation is cheap and has
                                     already happened.

Measured on the beam's own transitions, held out by start lineage: pre-execution information ranks
sibling macros at ~0.69 pairwise while the realized endpoint reaches ~0.85.  That gap is not a
defect to close -- it is the reason the stages exist.  Generate broadly under a modest Q_pre, then
let a strong Q_post decide what is worth paying for.

Q_pre IS ADDITIVE, and that is an evidence-led choice rather than a simplification.  A parent x
macro interaction model was fitted and LOST to the additive one on held-out lineages (R2 0.268 vs
0.308).  Parent context and macro intent each carry signal and combine; that they also interact is
not demonstrated at the sample size available, so the controller does not assume it.

THREE PROTECTIONS, each against a failure this work has already measured:

  lineage floor    a previous PMO run collapsed onto ONE initialization lineage -- 99% of scored
                   descendants from a single seed, and the best starting molecule got none.
  audit fraction   Q_post preferentially queries what it already likes, so a slice of calls is
                   drawn without reference to the model. Without it the training data is a
                   fixpoint of the model's own opinion.
  propensity log   every selection records the probability it was chosen with, so adaptive
                   sampling can be corrected for later instead of being discovered afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from compose_v4.control.pmo_contextual_macro import (
    POST_EXECUTION_BLOCKS,
    PRE_EXECUTION_BLOCKS,
    ContextualMacroValue,
)
from compose_v4.control.pmo_macro_value import (
    DEFAULT_ARCHIVE_K,
    archive_threshold,
    expected_archive_gain,
)

#: Share of oracle calls drawn WITHOUT consulting Q_post, so the training set is not a fixpoint
#: of the model's own preferences.
DEFAULT_AUDIT_FRACTION = 0.15

#: No live lineage may fall below this share of proposal compute, however unpromising it looks.
DEFAULT_LINEAGE_FLOOR = 0.10

#: Observations required before a head is trusted; below this the stage is deliberately uniform.
MINIMUM_OBSERVATIONS = 24


@dataclass
class TwoStageControl:
    """Q_pre allocates generation, Q_post allocates oracle calls, both updated from real scores."""

    audit_fraction: float = DEFAULT_AUDIT_FRACTION
    lineage_floor: float = DEFAULT_LINEAGE_FLOOR
    temperature: float = 0.10
    archive_k: int = DEFAULT_ARCHIVE_K
    pre: ContextualMacroValue = field(
        default_factory=lambda: ContextualMacroValue(blocks=PRE_EXECUTION_BLOCKS)
    )
    post: ContextualMacroValue = field(
        default_factory=lambda: ContextualMacroValue(blocks=POST_EXECUTION_BLOCKS)
    )
    observations: list = field(default_factory=list)
    propensities: list = field(default_factory=list)

    def __post_init__(self):
        if not 0.0 <= self.audit_fraction < 1.0:
            raise ValueError("audit fraction must lie in [0, 1)")
        if not 0.0 <= self.lineage_floor < 1.0:
            raise ValueError("lineage floor must lie in [0, 1)")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

    # ---- Learning -------------------------------------------------------------------------
    @property
    def fitted(self) -> bool:
        return len(self.observations) >= MINIMUM_OBSERVATIONS

    def observe(self, edge, score: float) -> None:
        """Record one CHARGED outcome and refit both heads on everything counted so far."""
        self.observations.append({**edge, "score": float(score)})
        if not self.fitted:
            return
        targets = [o["score"] for o in self.observations]
        self.pre.fit(self.observations, targets)
        self.post.fit(self.observations, targets)

    # ---- Stage one: where to spend cheap generation ----------------------------------------
    def intent_weights(self, intents) -> np.ndarray:
        """Proposal mass over (parent, macro intent) pairs, floored so nothing is eliminated.

        Below `MINIMUM_OBSERVATIONS` this is uniform on purpose: an unfitted model has no opinion
        and pretending otherwise is how an early accident becomes a permanent exclusion.
        """
        if not intents:
            return np.zeros(0)
        if not self.fitted:
            return np.full(len(intents), 1.0 / len(intents))
        value = self.pre.predict(intents)
        shifted = (value - value.max()) / self.temperature
        weights = np.exp(shifted)
        weights = weights / weights.sum()
        floor = self.lineage_floor / len(intents)
        weights = np.maximum(weights, floor)
        return weights / weights.sum()

    def parent_shares(self, intents, parents) -> dict:
        """Collapse intent mass onto parents, keeping every live lineage above the floor."""
        weights = self.intent_weights(intents)
        shares: dict[str, float] = {p: 0.0 for p in parents}
        for intent, weight in zip(intents, weights, strict=True):
            key = intent["parent"]
            if key in shares:
                shares[key] += float(weight)
        if not shares:
            return shares
        floor = self.lineage_floor / len(shares)
        adjusted = {k: max(v, floor) for k, v in shares.items()}
        total = sum(adjusted.values())
        return {k: v / total for k, v in adjusted.items()}

    # ---- Stage two: where to spend scarce oracle calls --------------------------------------
    def acquire(self, candidates, observations, *, batch: int, rng, budget_remaining=None):
        """Choose which realized molecules to pay for, by EXPECTED ARCHIVE GAIN.

        Candidates come from different parents, so they are never ranked by improvement over
        their own parent -- that systematically buys from the weakest parents on the frontier.
        The archive gain is derived from the predicted endpoint score against the archive we
        already know, which is the quantity the benchmark actually rewards.
        """
        if not candidates:
            return [], []
        room = min(batch, len(candidates))
        audit = round(self.audit_fraction * room)
        audit = min(max(audit, 1 if self.audit_fraction > 0 else 0), room)
        chosen, detail = [], []

        if self.fitted:
            threshold = archive_threshold(observations, k=self.archive_k)
            if not np.isfinite(threshold):
                # An unfilled archive admits everything; rank on the predicted score itself.
                utility = self.post.predict(candidates)
            else:
                mean = self.post.predict(candidates)
                spread = np.full(len(candidates), max(1e-6, float(np.std(mean))))
                utility = expected_archive_gain(
                    mean, spread, threshold=float(threshold), k=self.archive_k
                )
            order = list(np.argsort(-utility))
            total = float(np.sum(np.exp(utility / max(self.temperature, 1e-9)))) or 1.0
            for index in order[: room - audit]:
                chosen.append(index)
                detail.append(
                    {
                        "index": int(index),
                        "reason": "q_post",
                        "propensity": float(
                            np.exp(utility[index] / max(self.temperature, 1e-9)) / total
                        ),
                    }
                )
        remaining = [i for i in range(len(candidates)) if i not in set(chosen)]
        take = min(room - len(chosen), len(remaining))
        if take > 0:
            picked = rng.choice(len(remaining), take, replace=False)
            for raw in np.atleast_1d(picked):
                index = remaining[int(raw)]
                chosen.append(index)
                detail.append(
                    {
                        "index": int(index),
                        "reason": "audit" if self.fitted else "cold_start",
                        "propensity": float(take / max(len(remaining), 1)),
                    }
                )
        self.propensities.extend(detail)
        return chosen, detail
