from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    MolecularGraph,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import FixedMolecularStatePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.experiments.guided_rewrite_sampling import (
    CachedStateScorer,
    ProposalGuidedRewriteSampler,
    QEDTargetDistancePotential,
    kl_tilted_proposal_probabilities,
    qed_state_score,
)
from compose_v4.experiments.tracelet_conditional import sample_tracelet_ancestral
from compose_v4.model.factorized_tracelet_rate_model import SampledRewriteMark
from compose_v4.rewrite.operators import AtomRestate


@dataclass
class _AlternatingRestateSampler:
    calls: int = 0

    def sample_rewrite_mark(self, state, time, rng):
        del state, time, rng
        atom_type = ELEMENT_TO_IDX["N"] if self.calls % 2 == 0 else ELEMENT_TO_IDX["O"]
        implicit_h = 2 if atom_type == ELEMENT_TO_IDX["N"] else 1
        self.calls += 1
        return SampledRewriteMark(
            2.0,
            "atom_restate",
            AtomRestate(0, atom_type, 0, implicit_h),
        )


@dataclass
class _DuplicateRestateSampler:
    def sample_rewrite_mark(self, state, time, rng):
        del state, time, rng
        return SampledRewriteMark(
            1.0,
            "atom_restate",
            AtomRestate(0, ELEMENT_TO_IDX["N"], 0, 2),
        )


@dataclass
class _OneEventSampler:
    def sample_rewrite_mark(self, state, time, rng):
        del time, rng
        if int(state.atom_types[0]) == ELEMENT_TO_IDX["N"]:
            return SampledRewriteMark(0.0, "<TERMINAL>", None)
        return SampledRewriteMark(
            100.0,
            "atom_restate",
            AtomRestate(0, ELEMENT_TO_IDX["N"], 0, 2),
        )


@dataclass
class _RareEventSampler:
    def sample_rewrite_mark(self, state, time, rng):
        del state, time, rng
        return SampledRewriteMark(
            1e-12,
            "atom_restate",
            AtomRestate(0, ELEMENT_TO_IDX["N"], 0, 2),
        )


def _padded_ethane(n_slots: int = 4) -> MolecularGraph:
    return FixedMolecularStatePrior(smiles_to_molecular_graph("CC")).sample(
        np.random.default_rng(0),
        n_slots=n_slots,
    )


def test_fixed_state_prior_copies_and_pads_a_valid_source() -> None:
    original = smiles_to_molecular_graph("CC")
    prior = FixedMolecularStatePrior(original)
    first = prior.sample(np.random.default_rng(1), n_slots=5)
    second = prior.sample(np.random.default_rng(2), n_slots=5)
    assert first is not second
    assert first.n_atoms == 5
    assert molecular_graph_to_smiles(first) == "CC"
    first.atom_types[0] = 0
    assert molecular_graph_to_smiles(second) == "CC"


def test_proposal_guidance_selects_a_high_score_legal_successor() -> None:
    state = _padded_ethane()
    scorer = CachedStateScorer(
        lambda successor: float(successor.atom_types[0] == ELEMENT_TO_IDX["O"])
    )
    guided = ProposalGuidedRewriteSampler(
        _AlternatingRestateSampler(),
        scorer,
        proposals_per_event=2,
        guidance_strength=100.0,
        tempering_power=0.0,
    )
    sampled = guided.sample_rewrite_mark(state, 0.5, np.random.default_rng(4))
    successor = guided._runtime.apply(state, sampled.rule_name, sampled.action)
    assert int(successor.atom_types[0]) == ELEMENT_TO_IDX["O"]
    assert is_valid_state(successor)
    assert is_connected_or_null(successor)
    assert sampled.total_hazard == pytest.approx(2.0)


def test_qed_target_distance_potential_and_kl_tilt_are_target_specific() -> None:
    potential = QEDTargetDistancePotential(target_qed=0.8)
    assert potential(0.8) == pytest.approx(0.0)
    assert potential(0.5) == pytest.approx(-0.3)
    probabilities = kl_tilted_proposal_probabilities(
        [0.2, 0.75, 0.95],
        beta=8.0,
        score_potential=potential,
    )
    assert probabilities.sum() == pytest.approx(1.0)
    assert int(probabilities.argmax()) == 1
    assert np.allclose(
        kl_tilted_proposal_probabilities(
            [0.2, 0.75, 0.95],
            beta=0.0,
            score_potential=potential,
        ),
        np.full(3, 1.0 / 3.0),
    )


@pytest.mark.parametrize(
    ("target", "power"),
    [(-0.1, 1.0), (1.1, 1.0), (0.9, 0.0), (0.9, -1.0)],
)
def test_qed_target_distance_potential_rejects_invalid_parameters(
    target: float,
    power: float,
) -> None:
    with pytest.raises(ValueError):
        QEDTargetDistancePotential(target_qed=target, distance_power=power)


def test_common_scored_proposal_set_supports_multiple_controls() -> None:
    state = _padded_ethane()

    def atom_score(successor: MolecularGraph) -> float:
        return 0.8 if int(successor.atom_types[0]) == ELEMENT_TO_IDX["O"] else 0.2

    scorer = CachedStateScorer(atom_score)
    guided = ProposalGuidedRewriteSampler(
        _AlternatingRestateSampler(),
        scorer,
        proposals_per_event=4,
        guidance_strength=8.0,
        tempering_power=0.0,
        score_potential=QEDTargetDistancePotential(target_qed=0.8),
    )
    rng = np.random.default_rng(9)
    initial = guided.sample_hazard_probe(state, 0.5, rng)
    proposals = guided.draw_scored_proposals_after_event(
        state,
        0.5,
        rng,
        initial,
    )
    assert len(proposals) == 4
    assert scorer.accounting.requests == 2
    assert all(proposal.mark.total_hazard == pytest.approx(2.0) for proposal in proposals)
    assert all(is_valid_state(proposal.successor) for proposal in proposals)
    scores = [proposal.score for proposal in proposals]
    assert np.allclose(
        kl_tilted_proposal_probabilities(
            scores,
            beta=0.0,
            score_potential=guided.score_potential,
        ),
        np.full(4, 0.25),
    )
    assert kl_tilted_proposal_probabilities(
        scores,
        beta=8.0,
        score_potential=guided.score_potential,
    )[1] > 0.45


def test_oracle_accounting_distinguishes_requests_from_unique_evaluations() -> None:
    state = _padded_ethane()
    scorer = CachedStateScorer(lambda successor: float(successor.n_real_atoms))
    guided = ProposalGuidedRewriteSampler(
        _DuplicateRestateSampler(),
        scorer,
        proposals_per_event=4,
    )
    guided.begin_rollout_audit()
    guided.sample_rewrite_mark(state, 0.5, np.random.default_rng(5))
    assert scorer.accounting.requests == 1
    assert scorer.accounting.unique_evaluations == 1
    assert scorer.accounting.cache_hits == 0
    audit = guided.end_rollout_audit()
    assert audit["events"][0]["canonical_groups"][0]["multiplicity"] == 4
    assert len(audit["events"][0]["proposals"]) == 4


def test_max_guided_events_caps_oracle_bearing_resampling() -> None:
    state = _padded_ethane()
    scorer = CachedStateScorer(lambda successor: float(successor.n_real_atoms))
    guided = ProposalGuidedRewriteSampler(
        _AlternatingRestateSampler(),
        scorer,
        proposals_per_event=2,
        max_guided_events=1,
    )
    rng = np.random.default_rng(15)
    guided.begin_rollout_audit()
    first = guided.sample_hazard_probe(state, 0.5, rng)
    guided.resample_rewrite_mark_after_event(state, 0.5, rng, first)
    requests_after_first = scorer.accounting.requests
    second = guided.sample_hazard_probe(state, 0.5, rng)
    returned = guided.resample_rewrite_mark_after_event(state, 0.5, rng, second)
    assert returned is second
    assert scorer.accounting.requests == requests_after_first
    audit = guided.end_rollout_audit()
    assert audit["controlled_events"] == 1
    assert audit["raw_proposal_marks"] == 2


def test_guidance_start_event_skips_oracle_calls_before_the_window() -> None:
    state = _padded_ethane()
    scorer = CachedStateScorer(lambda successor: float(successor.n_real_atoms))
    guided = ProposalGuidedRewriteSampler(
        _AlternatingRestateSampler(),
        scorer,
        proposals_per_event=2,
        guidance_start_event=2,
        max_guided_events=1,
    )
    rng = np.random.default_rng(16)
    guided.begin_rollout_audit()
    for _ in range(2):
        initial = guided.sample_hazard_probe(state, 0.5, rng)
        returned = guided.resample_rewrite_mark_after_event(
            state,
            0.5,
            rng,
            initial,
        )
        assert returned is initial
    assert scorer.accounting.requests == 0
    initial = guided.sample_hazard_probe(state, 0.5, rng)
    guided.resample_rewrite_mark_after_event(state, 0.5, rng, initial)
    audit = guided.end_rollout_audit()
    assert audit["controlled_events"] == 1
    assert audit["events"][0]["zero_based_event_index"] == 2
    assert audit["events"][0]["zero_based_controlled_window_index"] == 0


def test_guided_sampler_runs_inside_ancestral_ctmc_from_a_fixed_molecule() -> None:
    source = _padded_ethane()
    scorer = CachedStateScorer(lambda successor: float(successor.n_real_atoms))
    guided = ProposalGuidedRewriteSampler(
        _OneEventSampler(),
        scorer,
        proposals_per_event=2,
    )
    rollout = sample_tracelet_ancestral(
        guided,
        rng=np.random.default_rng(7),
        n_slots=4,
        operational_horizon=1.0,
        time_step=0.1,
        max_events=4,
        source_prior=FixedMolecularStatePrior(source),
    )
    assert rollout.event_rules == ("atom_restate",)
    assert is_valid_state(rollout.final_state)
    assert is_connected_or_null(rollout.final_state)
    assert rollout.control_diagnostics is not None
    assert rollout.control_diagnostics["controlled_events"] == 1
    assert rollout.control_diagnostics["raw_proposal_marks"] == 2
    assert rollout.control_diagnostics["canonical_group_requests"] == 1
    assert len(rollout.control_diagnostics["events"]) == 1


def test_guidance_defers_oracle_calls_until_an_event_is_accepted() -> None:
    source = _padded_ethane()
    scorer = CachedStateScorer(lambda successor: float(successor.n_real_atoms))
    guided = ProposalGuidedRewriteSampler(
        _RareEventSampler(),
        scorer,
        proposals_per_event=4,
    )
    rollout = sample_tracelet_ancestral(
        guided,
        rng=np.random.default_rng(8),
        n_slots=4,
        operational_horizon=1.0,
        time_step=0.1,
        max_events=4,
        source_prior=FixedMolecularStatePrior(source),
    )
    assert rollout.event_rules == ()
    assert scorer.accounting.requests == 0


def test_qed_state_score_is_finite_for_a_valid_molecule() -> None:
    value = qed_state_score(smiles_to_molecular_graph("CCO"))
    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0
