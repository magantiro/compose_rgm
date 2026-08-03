"""The Process-V2 Active8 admission policy, against the real production model.

Acceptance tests 3, 4 and 5 live here.

Every teacher action below is obtained from the production marked law and then
replayed through the production executor, so no action is hand-constructed and
no family witness can be invented that the model could not actually propose.
The one deliberately synthetic object is the disabled-family trace in acceptance
5, which substitutes a rule name the executor has no rule for; that is a probe of
the classifier, which never executes anything, and it is labelled as such.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES, DISABLED_FAMILIES
from compose_v4.data.editing_v2_process_v2_active8_interfaces import (
    ACTIVE8_EXCLUDED,
    UPSTREAM_REJECTED,
)
from compose_v4.data.editing_v2_process_v2_active8_policy import (
    ACTIVE8_EXCLUSION_REASONS,
    DYNAMIC_EXCLUSION_STAGE,
    PROCESS_V2_ACTIVE8_EXECUTOR_RULES,
    PROCESS_V2_ACTIVE8_FAMILIES,
    REASON_EXPLICITLY_DISABLED,
    REASON_MARK_ABSENT,
    STATIC_EXCLUSION_STAGE,
    ProcessV2Active8PolicyError,
    ProcessV2ProductionCandidateChecker,
    build_process_v2_active8_policy,
    classify_process_v2_action,
    evaluate_process_v2_active8_trace,
    upstream_rejected_trace_decision,
    validate_process_v2_active8_policy,
)
from compose_v4.data.editing_v2_process_v2_active8_runtime import (
    build_process_v2_active8_runtime,
    load_process_v2_active8_model_config,
    process_v2_model_process_contract,
)
from compose_v4.data.packed_trace_store import (
    AddressedPackedTrace,
    PackedTraceAddress,
    PackedTraceProgress,
)
from compose_v4.experiments.editing_gate_zero_semantic_contract import (
    FrozenGateZeroSemanticContract,
    build_gate_zero_semantic_contract,
)
from compose_v4.experiments.editing_v2_semantic_runtime import (
    SemanticScratchModelConfig,
    build_semantic_scratch_runtime,
)
from compose_v4.experiments.production_successor_kernel import (
    canonical_successor_result,
)
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_process_v2_identity,
    editing_v2_process_identity,
)
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import AtomDelete, BondReorder, CycleCloseEdge
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace

ROOT = Path(__file__).resolve().parents[1]
SLOTS = 16
ENUMERATION_TIME = 0.5

#: A small panel that between them populate every one of the eight families in
#: the production marked law.  Real molecules, chosen for what the model can
#: propose on them rather than for chemistry interest.
_PANEL = ("CCCCCC", "C1CCCCC1", "CC1CCCCC1", "c1ccccc1", "Cc1ccccc1", "CCCCCCC")


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), SLOTS)


def _addressed(states: list[Any], steps: list[RewriteStep], *, trace_id: str):
    """Wrap an exact state path and its steps as one addressed packed trace."""

    trace = RewriteTrace(states[0], states[-1], tuple(steps), {"policy_fixture": trace_id})
    address = PackedTraceAddress(
        packed_shard_content_sha256="0" * 64,
        packed_shard_name="policy_fixture.jsonl.gz",
        entry_index=0,
        trace_id=trace_id,
        layer="observed_local_analogue",
        partition="train",
        source_key=canonical_state_key(states[0]),
        target_key=canonical_state_key(states[-1]),
        path_length=len(steps),
    )
    return AddressedPackedTrace(
        address=address, trace=trace, path=PackedTraceProgress(trace, tuple(states))
    )


def _small_config(**overrides: Any) -> SemanticScratchModelConfig:
    frozen = load_process_v2_active8_model_config(repo_root=ROOT)
    fields = {
        "initialization_seed": frozen.initialization_seed,
        "max_atoms": SLOTS,
        "hidden_dim": 32,
        "message_passing_steps": 2,
        "mark_dim": frozen.mark_dim,
        "dtype": frozen.dtype,
        "atom_vocabulary_class_count": frozen.atom_vocabulary_class_count,
        "catalog_fingerprint": frozen.catalog_fingerprint,
    }
    fields.update(overrides)
    return SemanticScratchModelConfig(**fields)


@pytest.fixture(name="runtime", scope="module")
def _runtime():
    """The Process-V2 model, built through the frozen model/process contract."""

    return build_process_v2_active8_runtime(
        model_config=_small_config(), repo_root=ROOT, candidate_time=ENUMERATION_TIME
    )


@pytest.fixture(name="v1_control", scope="module")
def _v1_control():
    """An explicitly injected Process-V1 control model, built the same way.

    Not a mock and not a narrowed V2 model: it is constructed from the frozen
    *V1* model/process contract, so the only thing separating it from the V2
    model is the process the contract declares.
    """

    contract = build_gate_zero_semantic_contract()
    return build_semantic_scratch_runtime(
        _small_config(),
        FrozenGateZeroSemanticContract(
            source=Path("configs/editing_gate_zero_semantic_model_process_v1.json"),
            payload=contract,
            file_sha256="0" * 64,
        ),
    )


def _marks_by_family(model, state) -> dict[str, list[Any]]:
    system = editing_v2_semantic_rewrite_system()
    result = canonical_successor_result(model, state, ENUMERATION_TIME, system=system)
    grouped: dict[str, list[Any]] = {}
    for mark in result.marked_law.marks:
        grouped.setdefault(mark.family_name, []).append(mark)
    return grouped


def _one_step_trace(model, smiles: str, mark) -> Any:
    system = editing_v2_semantic_rewrite_system()
    source = _state(smiles)
    successor = system.apply(source, mark.executor_rule_name, mark.action)
    return _addressed(
        [source, successor],
        [RewriteStep(mark.executor_rule_name, mark.action)],
        trace_id=f"{smiles}:{mark.executor_rule_name}",
    )


# ---- Acceptance test 3 --------------------------------------------------------


def test_a_connected_nonleaf_delete_is_admitted_under_v2_and_absent_under_v1(
    runtime, v1_control
) -> None:
    """Acceptance 3. Without this the V2 expansion is asserted, not measured.

    Cyclohexane has no leaf: every atom sits on the ring, so ``AtomDelete(0)``
    is a connected-nonleaf deletion.  Process V2 exists to make exactly that
    deletion learnable, so it must be in the V2 marked law, must classify as
    ``atom_delete``, and must produce supported candidate evidence.  The V1
    control is the same construction against the frozen V1 contract, and it must
    not carry the mark at all -- which is what makes the V2 admission an
    expansion rather than a relabelling.
    """

    smiles = "C1CCCCC1"
    source = _state(smiles)
    system = editing_v2_semantic_rewrite_system()
    successor = system.apply(source, "atom_delete", AtomDelete(0))
    addressed = _addressed(
        [source, successor],
        [RewriteStep("atom_delete", AtomDelete(0))],
        trace_id="connected_nonleaf_delete",
    )

    policy = runtime.policy
    classification = classify_process_v2_action(
        addressed.trace.steps[0], step_index=0, policy=policy
    )
    assert classification.policy_eligible is True
    assert classification.model_family == "atom_delete"

    decision = evaluate_process_v2_active8_trace(
        addressed, candidate_check=runtime.checker, policy=policy
    )
    assert decision.active8_status == "accepted"
    assert decision.category is None
    evidence = decision.action_decisions[0].candidate_evidence
    assert evidence is not None and evidence.supported is True
    assert evidence.matching_mark_count >= 1
    assert evidence.exact_successor_mark_count >= 1
    assert evidence.successor_alias_count >= 1

    # Measured on the V2 model: cyclohexane really has no leaf, so every
    # admitted atom_delete here is a connected-nonleaf deletion.
    v2_deletes = _marks_by_family(runtime.model, source).get("atom_delete", [])
    assert len(v2_deletes) == 6
    assert {mark.action.v for mark in v2_deletes} == set(range(6))

    # The explicitly injected V1 control does not carry the mark at all.
    v1_deletes = _marks_by_family(v1_control.model, source).get("atom_delete", [])
    assert v1_deletes == []
    assert (
        v1_control.semantic_model_identity["editing_process_semantics"]
        != runtime.scratch.semantic_model_identity["editing_process_semantics"]
    )

    # And the V2 policy refuses the V1 model outright, so a V1 control can never
    # be used to produce a V2 decision by accident.
    with pytest.raises(ProcessV2Active8PolicyError, match="Process-V2 Active8 modes"):
        ProcessV2ProductionCandidateChecker(v1_control.model, policy=policy)


# ---- Acceptance test 4 --------------------------------------------------------


def test_the_policy_binds_the_v2_process_identity_and_the_declared_eight(
    runtime,
) -> None:
    """Acceptance 4, the binding. Without this the policy could bind V1."""

    policy = validate_process_v2_active8_policy(runtime.policy)
    live_v2 = editing_process_v2_identity()["process_identity_sha256"]
    live_v1 = editing_v2_process_identity()["process_identity_sha256"]
    assert policy.process_identity_sha256 == live_v2
    assert policy.process_identity_sha256 != live_v1

    # The eight are read from their frozen registries, not restated here.
    assert policy.active_families == tuple(ACTIVE8_FAMILIES)
    assert len(policy.active_families) == 8
    assert PROCESS_V2_ACTIVE8_FAMILIES == tuple(ACTIVE8_FAMILIES)
    assert PROCESS_V2_ACTIVE8_EXECUTOR_RULES == tuple(
        action_codec_v4.ACTIVE8_EXECUTOR_RULES
    )
    assert len(PROCESS_V2_ACTIVE8_EXECUTOR_RULES) == 8
    assert dict(policy.executor_rule_to_family) == {
        rule: action_codec_v4.canonical_family(rule)
        for rule in action_codec_v4.ACTIVE8_EXECUTOR_RULES
    }

    # The runtime the decisions are made against is the same process.
    assert runtime.scratch.process_identity_sha256 == live_v2
    assert (
        runtime.scratch.semantic_model_process_contract_sha256
        == process_v2_model_process_contract(repo_root=ROOT).payload["contract_sha256"]
    )


def test_every_one_of_the_eight_families_is_witnessed_and_admitted(runtime) -> None:
    """Acceptance 4, the families. Without this "eight" is a list, not a fiber.

    Each witness is a mark the production model actually proposed, replayed
    through the production executor into a one-step trace and then decided by
    the policy.  A family that the model cannot propose anywhere on the panel
    fails here rather than being quietly absent from the evidence.
    """

    witnesses: dict[str, tuple[str, Any]] = {}
    for smiles in _PANEL:
        grouped = _marks_by_family(runtime.model, _state(smiles))
        for family, marks in grouped.items():
            witnesses.setdefault(family, (smiles, marks[0]))
    assert set(witnesses) == set(ACTIVE8_FAMILIES), sorted(witnesses)

    for family in ACTIVE8_FAMILIES:
        smiles, mark = witnesses[family]
        addressed = _one_step_trace(runtime.model, smiles, mark)
        decision = evaluate_process_v2_active8_trace(
            addressed, candidate_check=runtime.checker, policy=runtime.policy
        )
        assert decision.active8_status == "accepted", (family, smiles)
        assert decision.policy_sha256 == runtime.policy.policy_sha256
        action = decision.action_decisions[0]
        assert action.classification.model_family == family
        assert action.candidate_evidence is not None
        assert action.candidate_evidence.supported is True
        assert action.candidate_evidence.raw_mark_count > 0
        assert action.candidate_evidence.canonical_successor_count > 0


def test_ring_system_delete_and_grow_stay_disabled_everywhere(runtime) -> None:
    """Acceptance 4, the disabled pair. Without this the pilot could widen quietly.

    Three independent places: the model's capabilities, the marked law the model
    actually emits over the panel, and the policy's own refusal of the two rule
    names.  A capability flag alone would not prove the fiber is empty, and an
    empty fiber alone would not prove a future model could not fill it.
    """

    capabilities = runtime.model.operator_capabilities
    assert capabilities.compute_ring_grow_support is False
    assert capabilities.compute_ring_system_delete is False
    assert runtime.model.enable_ring_grow_macro is False
    assert runtime.model.enable_ring_system_delete is False

    for smiles in _PANEL:
        grouped = _marks_by_family(runtime.model, _state(smiles))
        assert set(grouped) <= set(ACTIVE8_FAMILIES), smiles
        for disabled in DISABLED_FAMILIES:
            assert disabled not in grouped, (smiles, disabled)

    assert runtime.policy.explicitly_disabled_rules == tuple(DISABLED_FAMILIES)
    assert set(DISABLED_FAMILIES) == {"ring_system_delete", "ring_system_grow"}
    for disabled in DISABLED_FAMILIES:
        classification = classify_process_v2_action(
            RewriteStep(disabled, AtomDelete(0)), step_index=0, policy=runtime.policy
        )
        assert classification.policy_eligible is False
        assert classification.exclusion_reason == REASON_EXPLICITLY_DISABLED
        assert classification.action_sha256 is None


# ---- Acceptance test 5 --------------------------------------------------------


def _three_step_trace(*, middle_rule: str | None = None):
    """A real three-step trace, optionally with its middle rule substituted.

    Heptane, cyclized to a methylcyclohexane, then one RING bond reordered, then
    the pendant methyl deleted.  Every step executes through the production
    executor; only the middle one is outside the model's peripheral bond-reorder
    mask.
    """

    system = editing_v2_semantic_rewrite_system()
    plan = [
        RewriteStep("cycle_close", CycleCloseEdge(0, 5, 1)),
        RewriteStep("bond_reorder", BondReorder(1, 2, 2)),
        RewriteStep("atom_delete", AtomDelete(6)),
    ]
    state = _state("CCCCCCC")
    states = [state]
    steps: list[RewriteStep] = []
    for step in plan:
        state = system.apply(state, step.rule_name, step.action)
        states.append(state)
        steps.append(step)
    if middle_rule is not None:
        steps[1] = RewriteStep(middle_rule, steps[1].action)
    return _addressed(states, steps, trace_id=f"middle:{middle_rule or 'real'}")


def test_a_disallowed_middle_action_excludes_the_entire_trace(runtime) -> None:
    """Acceptance 5, dynamic. Without this a trace could be admitted in part.

    The middle action reorders a ring bond, which the production model's mask
    does not carry, so it is unsupported.  The outer two actions ARE supported,
    which is the point: a stage that admitted the parts it could would hand a
    later consumer a progress row whose predecessor transition was never
    learnable.
    """

    addressed = _three_step_trace()
    decision = evaluate_process_v2_active8_trace(
        addressed, candidate_check=runtime.checker, policy=runtime.policy
    )
    assert decision.active8_status == "excluded"
    assert decision.category == ACTIVE8_EXCLUDED
    assert decision.emits_progress_rows is False

    # Exactly one exclusion, addressed to the middle step.
    assert len(decision.active8_exclusions) == 1
    exclusion = decision.active8_exclusions[0]
    assert exclusion.step_index == 1
    assert exclusion.stage == DYNAMIC_EXCLUSION_STAGE
    assert exclusion.reason == REASON_MARK_ABSENT
    assert exclusion.reason in ACTIVE8_EXCLUSION_REASONS

    # The outer two really were supported: the whole trace is excluded because
    # of one action, not because nothing in it was admissible.
    supported = [
        item.candidate_evidence.supported
        for item in decision.action_decisions
        if item.candidate_evidence is not None
    ]
    assert supported == [True, False, True]


def test_a_disallowed_middle_action_never_reaches_the_model(runtime) -> None:
    """Acceptance 5, static. Without this a refused trace would still cost a fiber.

    The middle rule is substituted for a disabled family, which the classifier
    refuses before the model is consulted.  The whole trace is excluded, no
    action carries production candidate evidence, and the checker is never
    called -- proven by a checker that fails if it is.

    Deliberately synthetic: the executor has no ``ring_system_delete`` rule in
    the Active8 surface, so a trace carrying one cannot be produced by replay.
    What is under test is the classifier, which never executes anything.
    """

    calls: list[int] = []

    def refusing_checker(addressed: Any, step_index: int):
        calls.append(step_index)
        raise AssertionError("the model was consulted for a statically excluded trace")

    addressed = _three_step_trace(middle_rule="ring_system_delete")
    decision = evaluate_process_v2_active8_trace(
        addressed, candidate_check=refusing_checker, policy=runtime.policy
    )
    assert calls == []
    assert decision.active8_status == "excluded"
    assert decision.category == ACTIVE8_EXCLUDED
    assert decision.emits_progress_rows is False
    assert len(decision.action_decisions) == 3
    assert all(item.candidate_evidence is None for item in decision.action_decisions)
    assert [item.step_index for item in decision.active8_exclusions] == [1]
    assert decision.active8_exclusions[0].stage == STATIC_EXCLUSION_STAGE
    assert decision.active8_exclusions[0].reason == REASON_EXPLICITLY_DISABLED


# ---- The policy's own contract ------------------------------------------------


def test_the_policy_publishes_every_authority_field_false(runtime) -> None:
    """Deciding what is admissible is not permission to train on it."""

    payload = runtime.policy.as_payload()
    for field in (
        "training_authorized",
        "gate_zero_authorized",
        "t1_authorized",
        "bounded_p50_authorized",
        "long_training_authorized",
        "checkpoint_selection_authorized",
        "final_test_selection_authorized",
        "active8_authorized",
    ):
        assert payload[field] is False, field
    assert "p50_authorized" not in payload


def test_the_policy_is_self_hashed_and_refuses_a_v1_policy_object() -> None:
    """A policy object of another type is not this policy, however alike."""

    from compose_v4.data.editing_v2_semantic_active8_admission import (
        build_semantic_active8_admission_policy,
    )

    policy = build_process_v2_active8_policy()
    assert validate_process_v2_active8_policy(policy) is policy
    with pytest.raises(ProcessV2Active8PolicyError, match="another type"):
        validate_process_v2_active8_policy(build_semantic_active8_admission_policy())
    assert (
        build_semantic_active8_admission_policy().policy_sha256 != policy.policy_sha256
    )


def test_an_upstream_rejection_is_carried_without_being_evaluated() -> None:
    """The two categories are different statements about the same trace."""

    policy = build_process_v2_active8_policy()
    decision = upstream_rejected_trace_decision(
        trace_id="upstream-fixture",
        rejection_code="atom_delete_outside_process_v2_mask",
        policy=policy,
    )
    assert decision.category == UPSTREAM_REJECTED
    assert decision.category != ACTIVE8_EXCLUDED
    assert decision.active8_status == "not_evaluated"
    assert decision.emits_progress_rows is False
    assert decision.action_decisions == ()
    assert decision.active8_exclusions == ()
    assert decision.upstream_rejection_code == "atom_delete_outside_process_v2_mask"
