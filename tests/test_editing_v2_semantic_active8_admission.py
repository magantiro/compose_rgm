"""Focused Action-V4 and exact-quotient semantic Active8 tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.editing_v2_semantic_active8_admission import (
    EXPLICITLY_DISABLED_RULES,
    SEMANTIC_ACTIVE8_EXECUTOR_RULES,
    SEMANTIC_ACTIVE8_FAMILIES,
    ProductionSemanticExactCandidateChecker,
    SemanticActive8AdmissionError,
    build_semantic_active8_admission_policy,
    classify_semantic_action,
    evaluate_semantic_active8_trace,
    semantic_migration_rejection_decision,
    validate_semantic_active8_admission_policy,
)
from compose_v4.data.packed_trace_store import (
    AddressedPackedTrace,
    PackedTraceAddress,
    PackedTraceProgress,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    BondInsert,
    BondReorder,
    BondReroute,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
)
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.tracelet_fiber import (
    enumerate_ring_system_restate_transitions,
)
from compose_v4.rewrite.tracelets import BondOrderChange, RingSystemRestate
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)


def _model(*, enable_cyclic_graft: bool = True):
    torch.manual_seed(19)
    return FactorizedTraceletRateModel(
        build_typed_ring_catalog(()),
        hidden_dim=8,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_cyclic_graft=enable_cyclic_graft,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        editing_process_semantics=SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
        atom_restate_action_semantics=SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
        ring_restate_scorer_mode=SEMANTIC_RING_RESTATE_SCORER_MODE,
        cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
        cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()


def _addressed(
    steps: tuple[RewriteStep, ...],
    states: tuple[object, ...],
    *,
    trace_id: str = "semantic-active8-fixture",
) -> AddressedPackedTrace:
    trace = RewriteTrace(states[0], states[-1], steps, {"fixture": True})
    path = PackedTraceProgress(trace, states)
    address = PackedTraceAddress(
        packed_shard_content_sha256="1" * 64,
        packed_shard_name="traces.jsonl.gz",
        entry_index=0,
        trace_id=trace_id,
        layer="reversible_synthetic_walk",
        partition="validation",
        source_key=canonical_state_key(states[0]),
        target_key=canonical_state_key(states[-1]),
        path_length=len(steps),
    )
    return AddressedPackedTrace(address=address, trace=trace, path=path)


def test_policy_is_exact_self_identifying_and_non_authorizing() -> None:
    policy = build_semantic_active8_admission_policy()
    assert validate_semantic_active8_admission_policy(policy) is policy
    assert policy.active_families == (
        "atom_insert",
        "atom_delete",
        "atom_restate",
        "bond_reorder",
        "bond_reroute",
        "cycle_insert",
        "cycle_attach",
        "ring_system_restate",
    )
    assert policy.executor_rule_to_family == (
        ("atom_delete", "atom_delete"),
        ("atom_insert", "atom_insert"),
        ("atom_restate_semantic", "atom_restate"),
        ("bond_reorder", "bond_reorder"),
        ("bond_reroute", "bond_reroute"),
        ("cycle_close", "cycle_insert"),
        ("cycle_open", "cycle_attach"),
        ("ring_system_restate", "ring_system_restate"),
    )
    assert policy.explicitly_disabled_rules == (
        "ring_system_delete",
        "ring_system_grow",
    )
    assert policy.training_authorized is False
    assert policy.active8_authorized is False
    assert len(policy.policy_sha256) == 64
    assert len(policy.implementation_file_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        policy.active8_authorized = True  # type: ignore[misc]
    with pytest.raises(SemanticActive8AdmissionError, match="stale, malformed"):
        validate_semantic_active8_admission_policy(replace(policy, training_authorized=True))


def test_action_v4_classifies_exactly_the_eight_semantic_rules() -> None:
    policy = build_semantic_active8_admission_policy()
    steps = (
        RewriteStep("atom_insert", AtomInsert(5, 6, 0, 3, ((1, 1),))),
        RewriteStep("atom_delete", AtomDelete(5)),
        RewriteStep("atom_restate_semantic", SemanticAtomRestate(1, 2)),
        RewriteStep("bond_reorder", BondReorder(0, 1, 2)),
        RewriteStep("bond_reroute", BondReroute(0, 1, 2, 3, 1)),
        RewriteStep("cycle_close", CycleCloseEdge(0, 5, 1)),
        RewriteStep("cycle_open", CycleOpenEdge(0, 5)),
        RewriteStep(
            "ring_system_restate",
            RingSystemRestate((BondOrderChange(0, 1, 2),)),
        ),
    )
    classifications = tuple(
        classify_semantic_action(step, step_index=index, policy=policy)
        for index, step in enumerate(steps)
    )
    assert tuple(item.executor_rule for item in classifications) == (
        "atom_insert",
        "atom_delete",
        "atom_restate_semantic",
        "bond_reorder",
        "bond_reroute",
        "cycle_close",
        "cycle_open",
        "ring_system_restate",
    )
    assert tuple(item.model_family for item in classifications) == (
        "atom_insert",
        "atom_delete",
        "atom_restate",
        "bond_reorder",
        "bond_reroute",
        "cycle_insert",
        "cycle_attach",
        "ring_system_restate",
    )
    assert all(item.policy_eligible for item in classifications)
    assert all(item.action_sha256 and len(item.action_sha256) == 64 for item in classifications)
    assert set(SEMANTIC_ACTIVE8_EXECUTOR_RULES) == {item.executor_rule for item in classifications}
    assert set(SEMANTIC_ACTIVE8_FAMILIES) == {str(item.model_family) for item in classifications}


def test_whole_trace_policy_excludes_disabled_and_multi_neighbor_actions() -> None:
    source = _state("CC")
    steps = (
        RewriteStep("ring_system_delete", object()),
        RewriteStep(
            "atom_insert",
            AtomInsert(2, 6, 0, 2, ((0, 1), (1, 1))),
        ),
    )
    addressed = _addressed(steps, (source, source, source))

    def must_not_run(_addressed, _step_index):
        raise AssertionError("static policy exclusion invoked the model checker")

    decision = evaluate_semantic_active8_trace(
        addressed,
        exact_candidate_checker=must_not_run,
    )
    assert decision.semantic_migration_status == "admitted"
    assert decision.semantic_migration_rejection is None
    assert decision.active8_status == "excluded"
    assert decision.emits_progress_rows is False
    assert len(decision.action_decisions) == 2
    assert {item.reason for item in decision.active8_exclusions} == {
        "explicitly_disabled_operator_family",
        "unsupported_multi_neighbor_atom_insert",
    }
    assert all(item.stage == "action_v4_policy" for item in decision.active8_exclusions)
    assert set(EXPLICITLY_DISABLED_RULES) == {
        "ring_system_delete",
        "ring_system_grow",
    }


def test_legacy_raw_cycle_action_is_outside_action_v4_policy() -> None:
    classification = classify_semantic_action(
        RewriteStep("bond_insert", BondInsert(0, 5, 1)),
        step_index=0,
        policy=build_semantic_active8_admission_policy(),
    )
    assert classification.policy_eligible is False
    assert classification.model_family is None
    assert classification.action_sha256 is None
    assert classification.exclusion_reason == "outside_action_codec_v4"


def test_production_checker_admits_exact_semantic_teacher_and_quotient() -> None:
    model = _model()
    source = _state("CCCCCC")
    step = RewriteStep("cycle_close", CycleCloseEdge(0, 5, 1))
    target = editing_v2_semantic_rewrite_system().apply(
        source,
        step.rule_name,
        step.action,
    )
    addressed = _addressed((step,), (source, target))
    checker = ProductionSemanticExactCandidateChecker(model, cache_size=2)

    decision = evaluate_semantic_active8_trace(
        addressed,
        exact_candidate_checker=checker,
        policy=checker.policy,
    )
    evidence = decision.action_decisions[0].candidate_evidence
    assert decision.active8_status == "accepted"
    assert decision.emits_progress_rows is True
    assert not decision.active8_exclusions
    assert evidence is not None and evidence.supported
    assert evidence.matching_mark_count == 1
    assert evidence.successor_alias_count >= 1
    assert evidence.raw_mark_count > evidence.canonical_successor_count > 0
    assert evidence.canonical_successor_key == canonical_state_key(target)


def test_production_checker_admits_symmetric_ring_restate_alias() -> None:
    model = _model()
    source = _state("c1ccccc1-c1ccccc1")
    runtime = editing_v2_semantic_rewrite_system()
    action, target = enumerate_ring_system_restate_transitions(
        source,
        system=runtime,
    )[0]
    addressed = _addressed(
        (RewriteStep("ring_system_restate", action),),
        (source, target),
        trace_id="symmetric-ring-restate",
    )
    checker = ProductionSemanticExactCandidateChecker(model, cache_size=2)

    decision = evaluate_semantic_active8_trace(
        addressed,
        exact_candidate_checker=checker,
        policy=checker.policy,
    )
    evidence = decision.action_decisions[0].candidate_evidence
    assert decision.active8_status == "accepted"
    assert decision.emits_progress_rows is True
    assert not decision.active8_exclusions
    assert evidence is not None and evidence.supported
    assert evidence.matching_mark_count == 1
    assert evidence.successor_alias_count == 2
    assert evidence.canonical_successor_key == canonical_state_key(target)


def test_exact_mark_with_wrong_persisted_target_excludes_whole_trace() -> None:
    model = _model()
    source = _state("CCCCCC")
    wrong_target = _state("CCCCC")
    step = RewriteStep("cycle_close", CycleCloseEdge(0, 5, 1))
    addressed = _addressed((step,), (source, wrong_target))
    checker = ProductionSemanticExactCandidateChecker(model, cache_size=2)

    decision = evaluate_semantic_active8_trace(
        addressed,
        exact_candidate_checker=checker,
        policy=checker.policy,
    )
    evidence = decision.action_decisions[0].candidate_evidence
    assert decision.active8_status == "excluded"
    assert decision.emits_progress_rows is False
    assert evidence is not None and evidence.supported is False
    assert evidence.exclusion_reason == "teacher_action_does_not_reproduce_exact_successor"
    assert decision.active8_exclusions[0].stage == "production_candidate_support"


def test_checker_rejects_model_outside_frozen_semantic_modes() -> None:
    model = _model(enable_cyclic_graft=False)
    with pytest.raises(
        SemanticActive8AdmissionError,
        match="does not implement the semantic Active8 modes",
    ):
        ProductionSemanticExactCandidateChecker(model)


def test_migration_rejection_is_not_relabelled_as_active8_exclusion() -> None:
    decision = semantic_migration_rejection_decision(
        trace_id="candidate-before-semantic-migration",
        reason_code="unsupported_legacy_action",
        detail="ActionCodecV4 conversion failed",
    )
    assert decision.semantic_migration_status == "rejected"
    assert decision.semantic_migration_rejection is not None
    assert decision.semantic_migration_rejection.reason_code == "unsupported_legacy_action"
    assert decision.active8_status == "not_evaluated"
    assert decision.active8_exclusions == ()
    assert decision.action_decisions == ()
    assert decision.emits_progress_rows is False
