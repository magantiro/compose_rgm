"""Inference-time guidance over executable molecular rewrite proposals.

This module deliberately wraps a frozen learned generator.  It never creates
an action outside the base model's legal sampler: a small Monte Carlo proposal
set is drawn from the learned marked CTMC, every proposal is committed by the
validity-closed executor, and one is resampled after a positive score tilt.

With finitely many proposals this is a self-normalized Monte Carlo
approximation to a score-tilted jump kernel, not an exact Doob transform.  The
proposal count and both requested and unique oracle evaluations are therefore
first-class reported quantities in matched-budget experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED

from compose_v4.chem.molecular_graph import MolecularGraph, molecular_graph_to_smiles
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.model.factorized_tracelet_rate_model import SampledRewriteMark
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system


StateScorer = Callable[[MolecularGraph], float]
ScorePotential = Callable[[float], float]
CONTROLLED_MARK_EQUATION = (
    "q_beta(m|x,t) proportional_to p_theta(m|x,t) "
    "exp(beta * U(QED(F_m(x)))); lambda_beta(x,t)=lambda_theta(x,t)"
)


def identity_score_potential(score: float) -> float:
    """Use an oracle score directly as the control potential."""

    value = float(score)
    if not np.isfinite(value):
        raise ValueError("oracle score must be finite")
    return value


@dataclass(frozen=True)
class QEDTargetDistancePotential:
    """Negative distance from a requested QED value.

    The controlled mark law favors successors close to ``target_qed`` rather
    than blindly maximizing QED.  ``distance_power=1`` gives
    ``potential(q) = -|q - target_qed|``.
    """

    target_qed: float
    distance_power: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.target_qed) or not 0.0 <= self.target_qed <= 1.0:
            raise ValueError("target QED must be finite and lie in [0, 1]")
        if not np.isfinite(self.distance_power) or self.distance_power <= 0.0:
            raise ValueError("distance power must be finite and positive")

    def __call__(self, qed: float) -> float:
        value = float(qed)
        if not np.isfinite(value):
            raise ValueError("QED score must be finite")
        return -abs(value - float(self.target_qed)) ** float(self.distance_power)


def kl_tilted_proposal_probabilities(
    scores: tuple[float, ...] | list[float] | np.ndarray,
    *,
    beta: float,
    score_potential: ScorePotential = identity_score_potential,
) -> np.ndarray:
    """Return the self-normalized finite-proposal KL tilt.

    For proposals sampled independently from the learned base mark law, this
    computes ``softmax(beta * potential(score))``.  Resampling with these
    weights is a Monte Carlo approximation to
    ``q(m|x,t) ∝ p_theta(m|x,t) exp(beta * potential(x_m))`` while leaving the
    learned total event hazard unchanged.
    """

    if not np.isfinite(beta):
        raise ValueError("guidance beta must be finite")
    numeric_scores = np.asarray(scores, dtype=np.float64)
    if numeric_scores.ndim != 1 or numeric_scores.size == 0:
        raise ValueError("at least one one-dimensional proposal score is required")
    if not np.isfinite(numeric_scores).all():
        raise ValueError("proposal scores must be finite")
    potentials = np.asarray(
        [float(score_potential(float(score))) for score in numeric_scores],
        dtype=np.float64,
    )
    if not np.isfinite(potentials).all():
        raise ValueError("proposal potentials must be finite")
    log_weights = float(beta) * potentials
    log_weights -= float(log_weights.max())
    weights = np.exp(log_weights)
    normalizer = float(weights.sum())
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise RuntimeError("controlled proposal weights failed to normalize")
    return weights / normalizer


class RewriteMarkSampler(Protocol):
    def sample_rewrite_mark(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        ...


@dataclass
class OracleAccounting:
    """Exact accounting for a cached deterministic molecular oracle."""

    requests: int = 0
    unique_evaluations: int = 0
    cache_hits: int = 0


@dataclass
class CachedStateScorer:
    """Cache a deterministic state scorer by canonical molecular identity."""

    scorer: StateScorer
    accounting: OracleAccounting = field(default_factory=OracleAccounting)
    _cache: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __call__(self, state: MolecularGraph) -> float:
        self.accounting.requests += 1
        key = canonical_state_key(state)
        cached = self._cache.get(key)
        if cached is not None:
            self.accounting.cache_hits += 1
            return cached
        value = float(self.scorer(state))
        if not np.isfinite(value):
            raise ValueError("molecular oracle returned a non-finite score")
        self._cache[key] = value
        self.accounting.unique_evaluations += 1
        return value


@dataclass(frozen=True)
class ScoredRewriteProposal:
    """One learned-policy mark, its executable successor, and oracle score."""

    mark: SampledRewriteMark
    successor: MolecularGraph
    successor_key: str
    score: float


def qed_state_score(state: MolecularGraph) -> float:
    """Return RDKit QED for a valid visible molecule (zero for formal null)."""

    smiles = molecular_graph_to_smiles(state)
    if smiles is None:
        return 0.0
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("valid molecular state failed RDKit QED conversion")
    return float(QED.qed(molecule))


@dataclass
class ProposalGuidedRewriteSampler:
    """Approximate a positive score tilt using only base-model legal marks.

    ``guidance_strength * time**tempering_power`` controls the score tilt.  A
    zero strength is an unguided Monte Carlo-resampling control.  Because all
    weights are strictly positive for finite scores, every action with positive
    base probability retains positive marginal probability.
    """

    base_sampler: RewriteMarkSampler
    scorer: CachedStateScorer
    proposals_per_event: int = 4
    guidance_strength: float = 1.0
    tempering_power: float = 1.0
    score_potential: ScorePotential = identity_score_potential
    guidance_start_event: int = 0
    max_guided_events: int | None = None
    _rollout_audit_events: list[dict[str, object]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _rollout_accounting_start: tuple[int, int, int] = field(
        default=(0, 0, 0),
        init=False,
        repr=False,
    )
    _event_opportunities_seen: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.proposals_per_event) <= 0:
            raise ValueError("proposals_per_event must be positive")
        if not np.isfinite(self.guidance_strength):
            raise ValueError("guidance_strength must be finite")
        if not np.isfinite(self.tempering_power) or self.tempering_power < 0.0:
            raise ValueError("tempering_power must be finite and nonnegative")
        probe = float(self.score_potential(0.0))
        if not np.isfinite(probe):
            raise ValueError("score potential must return finite values")
        if self.max_guided_events is not None and int(self.max_guided_events) <= 0:
            raise ValueError("max guided events must be positive when provided")
        if int(self.guidance_start_event) < 0:
            raise ValueError("guidance start event must be nonnegative")
        self._runtime = de_novo_rewrite_system()

    def begin_rollout_audit(self) -> None:
        """Reset trajectory-local proposal and oracle accounting."""

        self._rollout_audit_events.clear()
        self._event_opportunities_seen = 0
        accounting = self.scorer.accounting
        self._rollout_accounting_start = (
            int(accounting.requests),
            int(accounting.unique_evaluations),
            int(accounting.cache_hits),
        )

    def end_rollout_audit(self) -> dict[str, object]:
        """Return a JSON/pickle-safe audit of every oracle-bearing event."""

        accounting = self.scorer.accounting
        start_requests, start_unique, start_hits = self._rollout_accounting_start
        return {
            "controlled_events": len(self._rollout_audit_events),
            "max_guided_events": self.max_guided_events,
            "guidance_start_event": int(self.guidance_start_event),
            "event_opportunities_seen": self._event_opportunities_seen,
            "raw_proposal_marks": sum(
                len(event["proposals"]) for event in self._rollout_audit_events
            ),
            "canonical_group_requests": int(accounting.requests) - start_requests,
            "unique_oracle_evaluations": (
                int(accounting.unique_evaluations) - start_unique
            ),
            "cache_hits": int(accounting.cache_hits) - start_hits,
            "events": tuple(self._rollout_audit_events),
        }

    def sample_rewrite_mark(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        """Sample and score proposals immediately.

        This compatibility method is useful outside the ancestral sampler.  The
        ancestral CTMC calls :meth:`sample_hazard_probe` first and defers this
        oracle-bearing resampling to :meth:`resample_rewrite_mark_after_event`,
        so intervals containing no jump consume no oracle calls.
        """

        initial = self.sample_hazard_probe(state, time, rng)
        return self.resample_rewrite_mark_after_event(
            state,
            time,
            rng,
            initial,
        )

    def sample_hazard_probe(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        """Draw one unscored base mark carrying the unchanged CTMC hazard."""

        return self.base_sampler.sample_rewrite_mark(state, time, rng)

    def resample_rewrite_mark_after_event(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
        initial: SampledRewriteMark,
    ) -> SampledRewriteMark:
        """Tilt the conditional mark law after the base CTMC clock fires.

        ``initial`` is the unscored base mark used to obtain the waiting-time
        hazard.  It is retained as the first Monte Carlo proposal, so no base
        draw is wasted.  Positive-hazard virtual thinning marks are scored at
        the unchanged state and remain in the proposal set; guidance therefore
        cannot silently undo rate calibration by conditioning them away.
        """

        reference_hazard = float(initial.total_hazard)
        if reference_hazard <= 1e-12:
            return initial
        event_index = self._event_opportunities_seen
        self._event_opportunities_seen += 1
        if event_index < int(self.guidance_start_event):
            return initial
        if (
            self.max_guided_events is not None
            and len(self._rollout_audit_events) >= int(self.max_guided_events)
        ):
            return initial
        proposals = self.draw_scored_proposals_after_event(
            state,
            time,
            rng,
            initial,
        )
        if not proposals:
            return initial
        beta = float(self.guidance_strength) * float(time) ** float(
            self.tempering_power
        )
        probabilities = kl_tilted_proposal_probabilities(
            [proposal.score for proposal in proposals],
            beta=beta,
            score_potential=self.score_potential,
        )
        selected = int(rng.choice(len(proposals), p=probabilities))
        groups: dict[str, list[int]] = {}
        for index, proposal in enumerate(proposals):
            groups.setdefault(proposal.successor_key, []).append(index)
        self._rollout_audit_events.append(
            {
                "source_key": canonical_state_key(state),
                "zero_based_event_index": event_index,
                "zero_based_controlled_window_index": len(self._rollout_audit_events),
                "model_time": float(time),
                "total_hazard": reference_hazard,
                "beta": beta,
                "proposals": tuple(
                    {
                        "proposal_index": index,
                        "rule_name": proposal.mark.rule_name,
                        "successor_key": proposal.successor_key,
                        "score": proposal.score,
                        "valid": bool(is_valid_state(proposal.successor)),
                        "connected": bool(is_connected_or_null(proposal.successor)),
                        "selection_probability": float(probabilities[index]),
                    }
                    for index, proposal in enumerate(proposals)
                ),
                "canonical_groups": tuple(
                    {
                        "successor_key": key,
                        "proposal_indices": tuple(indices),
                        "multiplicity": len(indices),
                        "score": proposals[indices[0]].score,
                        "selection_mass": float(probabilities[indices].sum()),
                    }
                    for key, indices in groups.items()
                ),
                "selected_index": selected,
                "selected_rule_name": proposals[selected].mark.rule_name,
                "selected_successor_key": proposals[selected].successor_key,
                "selected_score": proposals[selected].score,
            }
        )
        return proposals[selected].mark

    def draw_scored_proposals_after_event(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
        initial: SampledRewriteMark,
    ) -> tuple[ScoredRewriteProposal, ...]:
        """Draw and execute one bounded proposal set without selecting from it.

        Exposing the common proposal set permits a matched beta-zero control,
        a stochastic target-distance tilt, and an offline best-of-K support
        diagnostic to share exactly the same oracle evaluations.
        """

        reference_hazard = float(initial.total_hazard)
        if reference_hazard <= 1e-12:
            return ()
        sampled_marks = [initial]
        for _ in range(int(self.proposals_per_event) - 1):
            sampled = self.base_sampler.sample_rewrite_mark(state, time, rng)
            hazard = float(sampled.total_hazard)
            if not np.isclose(hazard, reference_hazard, rtol=1e-5, atol=1e-8):
                raise RuntimeError("base sampler returned inconsistent state hazards")
            sampled_marks.append(sampled)

        unscored: list[tuple[SampledRewriteMark, MolecularGraph, str]] = []
        for sampled in sampled_marks:
            if sampled.action is None:
                # Positive-hazard virtual events are proposals in the
                # augmented thinning process and leave the molecule unchanged.
                if float(sampled.total_hazard) <= 1e-12:
                    return ()
                successor = state
            else:
                successor = self._runtime.apply(
                    state,
                    sampled.rule_name,
                    sampled.action,
                )
            unscored.append((sampled, successor, canonical_state_key(successor)))

        # Quotient only the deterministic oracle work.  The returned proposal
        # tuple keeps every sampled mark, so repeated marks/successors retain
        # their learned base mass under the finite-proposal KL tilt.
        scores_by_successor: dict[str, float] = {}
        for _sampled, successor, successor_key in unscored:
            if successor_key not in scores_by_successor:
                scores_by_successor[successor_key] = float(self.scorer(successor))

        candidates: list[ScoredRewriteProposal] = []
        for sampled, successor, successor_key in unscored:
            candidates.append(
                ScoredRewriteProposal(
                    mark=sampled,
                    successor=successor,
                    successor_key=successor_key,
                    score=scores_by_successor[successor_key],
                )
            )
        return tuple(candidates)


__all__ = [
    "CachedStateScorer",
    "CONTROLLED_MARK_EQUATION",
    "QEDTargetDistancePotential",
    "OracleAccounting",
    "ProposalGuidedRewriteSampler",
    "RewriteMarkSampler",
    "ScoredRewriteProposal",
    "ScorePotential",
    "StateScorer",
    "identity_score_potential",
    "kl_tilted_proposal_probabilities",
    "qed_state_score",
]
