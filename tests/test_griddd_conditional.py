from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    MolecularGraph,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import FixedMolecularStatePrior
from compose_v4.experiments.griddd_conditional import (
    ANALYTIC_BACKBONE_QUALIFICATION_FORMAT,
    BACKBONE_QUALIFICATION_FORMAT,
    CandidateGenerationResult,
    ConditionalArm,
    CorrectedBackboneContract,
    ExactBudgetLeadGuidedSampler,
    ExactOracleBudgetLedger,
    GridDDBenchmarkFairnessContract,
    GridDDConditionalEvaluator,
    GridDDNativeDirectEvaluator,
    GridDDProtocol,
    LeadRecord,
    RETAINED_PANCAKE_CHECKPOINT_SHA256,
    UnconditionalBackboneQualification,
    compare_conditional_ring_references,
    qed_state_oracle,
    standard_conditional_arms,
    tanimoto_similarity,
)
from compose_v4.experiments.tracelet_conditional import sample_tracelet_ancestral
from compose_v4.model.factorized_tracelet_rate_model import SampledRewriteMark
from compose_v4.rewrite.operators import AtomRestate


LEAD_SMILES = "Cc1occc1C(=O)NNC(=O)Nc1ccc(F)cc1F"
SUCCESS_SMILES = "COc1cccc(C(=O)Nc2ccc(F)cc2F)c1"


def _state(smiles: str) -> MolecularGraph:
    return smiles_to_molecular_graph(smiles)


def _contract(*, qualified: bool = False) -> CorrectedBackboneContract:
    return CorrectedBackboneContract(
        checkpoint_sha256="0" * 64,
        weight_source="pancake_derived",
        canonical_successor_execution=True,
        molecular_self_transitions_virtualized=True,
        unconditional_qualification_passed=qualified,
    )


def _qualification_payload() -> dict[str, object]:
    return {
        "format": BACKBONE_QUALIFICATION_FORMAT,
        "decision": {"qualification_passed": True},
        "checkpoint": {
            "path": "artifacts/canonical/checkpoint.pt",
            "sha256": "2" * 64,
            "weight_source": "pancake_derived",
            "source_checkpoint_sha256": RETAINED_PANCAKE_CHECKPOINT_SHA256,
            "property_condition_dim": 0,
        },
        "execution": {
            "canonical_successor_execution": True,
            "molecular_self_transitions_virtualized": True,
        },
        "model_modes": {
            "empirical_mark_prior_mode": "none",
            "ring_family_mass_mode": "boolean",
            "p1_p2_imported": False,
        },
        "evidence": {
            "multi_state_panel": {"passed": True, "state_count": 8},
            "heldout_rate_gate": {"passed": True},
            "rollout_gate": {"passed": True},
        },
    }


def _qualification() -> UnconditionalBackboneQualification:
    return UnconditionalBackboneQualification.from_payload(
        _qualification_payload(),
        manifest_path="qualification.json",
        manifest_sha256="3" * 64,
    )


def _conditioned_child_payload() -> dict[str, object]:
    return {
        "property_condition_dim": 1,
        "property_conditioning": {
            "names": ["qed"],
            "means": [0.5],
            "standard_deviations": [0.1],
        },
        "unconditional_backbone_qualification_manifest_sha256": "3" * 64,
        "unconditional_backbone_checkpoint_sha256": "2" * 64,
        "weight_source": "pancake_derived",
        "canonical_successor_execution": True,
        "molecular_self_transitions_virtualized": True,
        "empirical_mark_prior_mode": "none",
        "ring_family_mass_mode": "boolean",
        "p1_p2_imported": False,
    }


def test_corrected_backbone_contract_rejects_unvalidated_p1_p2_modes() -> None:
    with pytest.raises(ValueError, match="P1"):
        CorrectedBackboneContract.from_checkpoint_payload(
            {"empirical_mark_prior_mode": "corpus_residual_v1"},
            checkpoint_sha256="1" * 64,
            weight_source="pancake_derived",
            unconditional_qualification_passed=False,
        )
    with pytest.raises(ValueError, match="P2"):
        CorrectedBackboneContract.from_checkpoint_payload(
            {"ring_family_mass_mode": "catalog_topology_local_support"},
            checkpoint_sha256="1" * 64,
            weight_source="pancake_derived",
            unconditional_qualification_passed=False,
        )
    with pytest.raises(RuntimeError, match="qualification"):
        _contract().require_qualified()


def test_qualification_manifest_binds_canonical_pancake_child_without_p1_p2() -> None:
    qualification = _qualification()
    contract = CorrectedBackboneContract.from_qualification_manifest(
        qualification,
        _conditioned_child_payload(),
        conditional_checkpoint_sha256="4" * 64,
    )
    assert contract.unconditional_qualification_passed is True
    assert contract.weight_source == "pancake_derived"
    assert contract.canonical_successor_execution is True
    assert contract.molecular_self_transitions_virtualized is True
    assert contract.to_dict()["p1_p2_imported"] is False


def test_qualification_manifest_rejects_one_state_smoke_and_failed_evidence() -> None:
    payload = _qualification_payload()
    payload["format"] = "compose_v4_canonical_successor_distillation_smoke_v1"
    with pytest.raises(ValueError, match="not a canonical-successor"):
        UnconditionalBackboneQualification.from_payload(
            payload,
            manifest_path="one-state.json",
            manifest_sha256="5" * 64,
        )

    payload = _qualification_payload()
    payload["evidence"]["multi_state_panel"] = {  # type: ignore[index]
        "passed": False,
        "state_count": 1,
    }
    with pytest.raises(ValueError, match="has not passed"):
        UnconditionalBackboneQualification.from_payload(
            payload,
            manifest_path="failed.json",
            manifest_sha256="5" * 64,
        )


def test_conditional_child_must_bind_exact_manifest_parent_and_disable_p1_p2() -> None:
    qualification = _qualification()
    child = _conditioned_child_payload()
    child["unconditional_backbone_checkpoint_sha256"] = "6" * 64
    with pytest.raises(ValueError, match="qualified checkpoint"):
        qualification.assert_conditional_child(child, checkpoint_sha256="4" * 64)

    child = _conditioned_child_payload()
    child["empirical_mark_prior_mode"] = "corpus_residual_v1"
    with pytest.raises(ValueError, match="P1"):
        qualification.assert_conditional_child(child, checkpoint_sha256="4" * 64)

    child = _conditioned_child_payload()
    child["ring_family_mass_mode"] = "catalog_topology_local_support"
    with pytest.raises(ValueError, match="P2"):
        qualification.assert_conditional_child(child, checkpoint_sha256="4" * 64)


def test_analytic_qualification_binds_execution_and_frozen_sidecar_contract() -> None:
    payload = _qualification_payload()
    payload["format"] = ANALYTIC_BACKBONE_QUALIFICATION_FORMAT
    payload["checkpoint"]["sha256"] = RETAINED_PANCAKE_CHECKPOINT_SHA256  # type: ignore[index]
    payload["execution"] = {
        "canonical_successor_execution": True,
        "molecular_self_transitions_virtualized": True,
        "backbone_execution_kind": "analytic_pancake_quotient_adapter_v1",
        "analytic_adapter": {
            "source_path": "src/compose_v4/experiments/canonical_successor_distillation.py",
            "source_sha256": "7" * 64,
        },
        "calibration": {
            "atom_delete_log_rate_adjustment": -0.5,
            "small_ring_log_rate_adjustment": -1.5,
            "small_ring_maximum_size": 4,
        },
        "history_safety": "canonical_history_exact_thinning_v1",
    }
    qualification = UnconditionalBackboneQualification.from_payload(
        payload,
        manifest_path="analytic.json",
        manifest_sha256="8" * 64,
    )
    child = _conditioned_child_payload()
    child["unconditional_backbone_qualification_manifest_sha256"] = "8" * 64
    child["unconditional_backbone_checkpoint_sha256"] = (
        RETAINED_PANCAKE_CHECKPOINT_SHA256
    )
    child["unconditional_backbone_execution_kind"] = (
        "analytic_pancake_quotient_adapter_v1"
    )
    child["conditional_adapter"] = {
        "kind": "frozen_valid_state_qed_residual_v1",
        "base_parameters_frozen": True,
        "base_weights_in_sidecar_state_dict": False,
        "family_hazard_residual": True,
        "within_family_residual": True,
        "independent_total_hazard_residual": False,
        "missing_condition_identity": True,
    }
    contract = CorrectedBackboneContract.from_qualification_manifest(
        qualification,
        child,
        conditional_checkpoint_sha256="9" * 64,
    )
    assert contract.unconditional_execution_kind == (
        "analytic_pancake_quotient_adapter_v1"
    )

def test_griddd_similarity_and_qed_threshold_fixture_is_real() -> None:
    lead = _state(LEAD_SMILES)
    candidate = _state(SUCCESS_SMILES)
    assert 0.70 <= qed_state_oracle(lead) <= 0.80
    assert qed_state_oracle(candidate) >= 0.90
    assert tanimoto_similarity(lead, candidate) >= 0.40


def test_conditional_ring_reference_uses_high_qed_subset_not_global_marginal() -> None:
    report = compare_conditional_ring_references(
        (_state("CC"), _state("c1ccccc1"), _state("c1ccc2ccccc2c1")),
        (_state("c1ccccc1"), _state("c1ccc2ccccc2c1")),
        scorer=lambda state: 0.95,
    )
    assert report["unconditional_corpus"]["molecules"] == 3  # type: ignore[index]
    assert report["high_qed_target_subset"]["molecules"] == 2  # type: ignore[index]
    assert (
        report["high_qed_target_subset"]["stratum_fractions"][  # type: ignore[index]
            "fused_or_bridged"
        ]
        > report["unconditional_corpus"]["stratum_fractions"][  # type: ignore[index]
            "fused_or_bridged"
        ]
    )
    assert "conditional reference" in report["interpretation_boundary"]


def test_griddd_fairness_contract_keeps_native_and_internal_protocols_separate() -> None:
    contract = GridDDBenchmarkFairnessContract()
    payload = contract.to_dict()
    primary = payload["protocol_a_griddd_comparable"]
    internal = payload["protocol_b_compose_internal"]
    boundary = payload["claim_boundary"]
    assert primary["arm"] == "direct_conditioned_native_sampling"  # type: ignore[index]
    assert primary["beta_zero_shadow_rescoring"] is False  # type: ignore[index]
    assert primary["padding_calls"] is False  # type: ignore[index]
    assert internal["exact_oracle_calls_per_start_seed_arm"] == 981  # type: ignore[index]
    assert internal["arms"] == ["direct", "controller", "combined"]  # type: ignore[index]
    assert boundary["tables_must_remain_separate"] is True  # type: ignore[index]
    assert boundary["protocol_b_is_query_matched_to_griddd"] is False  # type: ignore[index]


@dataclass
class _RareEventSampler:
    def sample_rewrite_mark(self, state, time, rng):
        del state, time, rng
        return SampledRewriteMark(
            1e-12,
            "atom_restate",
            AtomRestate(0, ELEMENT_TO_IDX["N"], 0, 2),
        )


def test_exact_guidance_budget_pads_a_short_trajectory() -> None:
    lead = FixedMolecularStatePrior(_state("CC")).sample(
        np.random.default_rng(0),
        n_slots=4,
    )
    ledger = ExactOracleBudgetLedger(qed_state_oracle, budget=4)
    scope = ledger.scope(4, label="short")
    sampler = ExactBudgetLeadGuidedSampler(
        _RareEventSampler(),
        scope,
        lead,
        target_qed=0.9,
        minimum_similarity=0.4,
        proposals_per_event=4,
        beta=8.0,
    )
    rollout = sample_tracelet_ancestral(
        sampler,
        rng=np.random.default_rng(2),
        n_slots=4,
        operational_horizon=0.1,
        time_step=0.1,
        max_events=2,
        source_prior=FixedMolecularStatePrior(lead),
    )
    ledger.assert_exact()
    assert rollout.event_rules == ()
    assert rollout.control_diagnostics is not None
    assert rollout.control_diagnostics["oracle_calls"] == 4
    assert rollout.control_diagnostics["padding_calls"] == 4
    assert ledger.summary()["exact"] is True


@dataclass
class _ScriptedGenerator:
    scripts: dict[str, list[MolecularGraph | None]]
    productive_calls: dict[str, list[int]]
    counters: dict[str, int] = field(default_factory=dict)
    rng_draws: dict[str, list[int]] = field(default_factory=dict)

    def generate(
        self,
        *,
        lead_state,
        target_qed,
        arm: ConditionalArm,
        rng,
        oracle_scope,
    ):
        del target_qed
        index = self.counters.get(arm.name, 0)
        self.counters[arm.name] = index + 1
        self.rng_draws.setdefault(arm.name, []).append(
            int(rng.integers(0, np.iinfo(np.int64).max))
        )
        final = self.scripts[arm.name][index]
        for _ in range(self.productive_calls[arm.name][index]):
            oracle_scope.score(
                final or lead_state,
                purpose="scripted_productive",
                influences_selection=arm.controller,
            )
        valid = final is not None
        return CandidateGenerationResult(
            final_state=final,
            trajectory_states_audited=2 if valid else 1,
            all_trajectory_states_valid=valid,
            all_trajectory_states_connected=valid,
            events=1 if valid else 0,
            exhausted_event_budget=False,
            stalled=not valid,
            error=None if valid else "scripted failure",
        )


@dataclass
class _NativeScriptedGenerator:
    final: MolecularGraph
    calls: int = 0

    def generate_native(self, *, lead_state, target_qed, rng):
        del lead_state, target_qed, rng
        self.calls += 1
        return CandidateGenerationResult(
            final_state=self.final,
            trajectory_states_audited=2,
            all_trajectory_states_valid=True,
            all_trajectory_states_connected=True,
            events=1,
            exhausted_event_budget=False,
            metadata={
                "ring_rewrite_use_counts": {
                    "ring_system_grow": 1,
                    "ring_system_delete": 0,
                    "ring_system_restate": 0,
                },
                "control_diagnostics": {
                    "available_at_any_observation": {
                        "ring_system_grow": True,
                        "ring_system_delete": False,
                        "ring_system_restate": True,
                    }
                },
            },
        )


def test_protocol_a_native_evaluator_draws_20_without_padding_or_sampling_oracle() -> None:
    lead = _state(LEAD_SMILES)
    generator = _NativeScriptedGenerator(_state(SUCCESS_SMILES))
    scorer_calls = 0

    def scorer(state):
        nonlocal scorer_calls
        scorer_calls += 1
        return qed_state_oracle(state)

    report = GridDDNativeDirectEvaluator(
        GridDDProtocol(bootstrap_replicates=20),
        generator,
        _contract(qualified=True),
        scorer=scorer,
    ).evaluate((LeadRecord("lead", lead),), seeds=(11,))

    assert generator.calls == 20
    assert scorer_calls == 21
    assert report["candidate_generation"] == "direct_conditioned_native_sampling"
    assert report["native_sampling_oracle_calls"] == 0
    assert report["padding_calls"] == 0
    assert report["benchmark_evaluation_qed_calls"] == 21
    strata = report["ring_sensitivity"]["strata"]  # type: ignore[index]
    assert sum(item["candidate_count"] for item in strata.values()) == 20
    populated = next(item for item in strata.values() if item["candidate_count"])
    assert populated["ring_rewrite_use_counts"]["ring_system_grow"] == 20
    assert populated["ring_family_available_candidate_counts"][
        "ring_system_restate"
    ] == 20


def test_three_arms_have_exact_budgets_all_attempt_denominators_and_anytime() -> None:
    lead = _state(LEAD_SMILES)
    success = _state(SUCCESS_SMILES)
    generator = _ScriptedGenerator(
        scripts={
            "direct": [lead, lead],
            "controller": [success, lead],
            "combined": [success, success],
        },
        productive_calls={
            "direct": [0, 1],
            "controller": [2, 4],
            "combined": [4, 3],
        },
    )
    protocol = GridDDProtocol(
        candidates_per_start=2,
        guidance_oracle_calls_per_candidate=4,
        proposals_per_controlled_event=4,
        bootstrap_replicates=100,
    )
    report = GridDDConditionalEvaluator(
        protocol,
        generator,
        _contract(),
        allow_unqualified_smoke=True,
    ).evaluate(
        (LeadRecord("lead-0", lead),),
        seeds=(17,),
        arms=standard_conditional_arms(beta=8.0),
    )
    assert report["all_attempt_denominator"] == 1
    assert report["smoke_only"] is True
    assert report["large_benchmark_authorized"] is False
    for arm in ("direct", "controller", "combined"):
        arm_report = report["arms"][arm]
        assert arm_report["candidate_attempts"] == 2
        assert arm_report["total_oracle_calls"] == 11
        assert arm_report["exact_oracle_budget_every_start_seed"] is True
    assert report["arms"]["direct"]["successful_start_seeds"] == 0
    assert report["arms"]["controller"]["successful_start_seeds"] == 1
    assert report["arms"]["combined"]["successful_start_seeds"] == 1
    assert report["arms"]["controller"]["anytime_success_fraction_by_candidate"] == [
        1.0,
        1.0,
    ]
    combined_row = report["records"]["combined"][0]
    assert combined_row["oracle"]["budget"] == 11
    assert combined_row["oracle"]["used"] == 11
    assert combined_row["diversity"]["valid_candidates"] == 2
    assert combined_row["diversity"]["unique_candidates"] == 1
    assert generator.rng_draws["direct"] == generator.rng_draws["controller"]
    assert generator.rng_draws["direct"] == generator.rng_draws["combined"]
    assert report["seed_stream_identical_across_arms"] is True


def test_invalid_candidate_stays_in_denominator_and_budget_is_still_exact() -> None:
    lead = _state(LEAD_SMILES)
    generator = _ScriptedGenerator(
        scripts={name: [None] for name in ("direct", "controller", "combined")},
        productive_calls={name: [0] for name in ("direct", "controller", "combined")},
    )
    protocol = GridDDProtocol(
        candidates_per_start=1,
        guidance_oracle_calls_per_candidate=4,
        bootstrap_replicates=25,
    )
    report = GridDDConditionalEvaluator(
        protocol,
        generator,
        _contract(),
        allow_unqualified_smoke=True,
    ).evaluate((LeadRecord("lead-0", lead),), seeds=(3,))
    for arm in ("direct", "controller", "combined"):
        row = report["records"][arm][0]
        assert row["candidate_attempts"] == 1
        assert row["valid_candidates"] == 0
        assert row["failed_candidates"] == 1
        assert row["oracle"]["used"] == 6
        assert row["oracle"]["exact"] is True


def test_protocol_rejects_leads_outside_the_frozen_qed_window() -> None:
    lead = _state("CC")
    generator = _ScriptedGenerator(
        scripts={name: [lead] for name in ("direct", "controller", "combined")},
        productive_calls={name: [0] for name in ("direct", "controller", "combined")},
    )
    evaluator = GridDDConditionalEvaluator(
        GridDDProtocol(
            candidates_per_start=1,
            guidance_oracle_calls_per_candidate=4,
            bootstrap_replicates=25,
        ),
        generator,
        _contract(),
        allow_unqualified_smoke=True,
    )
    with pytest.raises(ValueError, match="outside"):
        evaluator.evaluate((LeadRecord("bad", lead),), seeds=(1,))
