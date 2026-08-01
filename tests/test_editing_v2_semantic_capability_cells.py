"""Focused tests for compact Editing-V2 semantic capability cells."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.editing_semantic_sampling_sidecar import (
    EndpointDescriptor,
    MappingDescriptor,
    PathDescriptor,
    SamplingCoefficients,
    SemanticGroups,
    SemanticSamplingProgress,
)
from compose_v4.data.editing_v2_semantic_active8_admission import (
    SemanticActive8ActionDecision,
    SemanticActive8TraceDecision,
    SemanticExactCandidateEvidence,
    build_semantic_active8_admission_policy,
    classify_semantic_action,
)
from compose_v4.data.editing_v2_semantic_capability_cells import (
    SemanticCapabilityCellError,
    classify_accepted_semantic_progress,
    classify_action_family_context,
    load_semantic_capability_cell_registry,
    validate_semantic_capability_cell_registry,
)
from compose_v4.data.packed_trace_store import (
    AddressedPackedTrace,
    PackedTraceAddress,
    PackedTraceProgress,
)
from compose_v4.rewrite.factorized_fiber import enumerate_pendant_graft_actions
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    BondInsert,
    BondReorder,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
    enumerate_cycle_open_edges,
)
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.tracelets import BondOrderChange, RingSystemRestate

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "editing_v2_semantic_capability_cells_v1.json"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _addressed(
    source,
    successor,
    step: RewriteStep,
    *,
    lane: str = "operator_aware_real_endpoint",
    partition: str = "train",
) -> AddressedPackedTrace:
    trace = RewriteTrace(source, successor, (step,), {"fixture": True})
    return AddressedPackedTrace(
        address=PackedTraceAddress(
            packed_shard_content_sha256=SHA_A,
            packed_shard_name="semantic_cells.jsonl.gz",
            entry_index=3,
            trace_id="semantic-cell-fixture",
            layer=lane,
            partition=partition,
            source_key=canonical_state_key(source),
            target_key=canonical_state_key(successor),
            path_length=1,
        ),
        trace=trace,
        path=PackedTraceProgress(trace, (source, successor)),
    )


def _progress(addressed: AddressedPackedTrace, *, progress_index: int = 0):
    address = addressed.address
    return SemanticSamplingProgress(
        address=address,
        progress_index=progress_index,
        candidate_ledger_row_sha256=SHA_A,
        partition_resolution={
            "status": "assigned",
            "assigned_role": address.partition,
            "source_endpoint_role": address.partition,
            "target_endpoint_role": address.partition,
            "split_component_id": "semantic-cell-split-component",
            "split_assignment_manifest_sha256": SHA_B,
        },
        admitted_record_envelope_sha256=SHA_B,
        groups=SemanticGroups(
            source_group_id="source-group",
            scaffold_group_id="scaffold-group",
            series_group_id=None,
            document_or_source_group_id="document-group",
            transformation_signature="fixture-transformation",
        ),
        record_membership_cells=("cardinality_shrinkage",),
        record_membership_classifier_identity_sha256=SHA_A,
        endpoint_descriptor=EndpointDescriptor(
            source_canonical_key=address.source_key,
            target_canonical_key=address.target_key,
            atom_count_delta=-1,
            graph_cycle_rank_delta=0,
            source_endpoint_evidence_kind="observed_real_endpoint",
            target_endpoint_evidence_kind="observed_real_endpoint",
            pair_relationship_evidence_kind="observed_real_endpoint_pair",
        ),
        path_descriptor=PathDescriptor(
            path_length=1,
            operator_family_set=("atom_delete",),
            path_evidence_kind="compiled_path_between_real_endpoints",
            intermediate_evidence_kind="executor_generated_intermediates",
            action_sequence_evidence_kind="compiler_generated_action_sequence",
            maximum_intermediate_source_distance=1.0,
            immediate_reversal_count=0,
            repeated_state_count=0,
            metric_policy_sha256=SHA_A,
        ),
        relationship_group_ids=(),
        mappings=MappingDescriptor(
            constant_core_mapping=None,
            protected_mapping=None,
            mapping_policy_sha256=SHA_A,
        ),
        sampling_coefficients=SamplingCoefficients(
            path_sampling_coefficient=1.0,
            progress_sampling_coefficient=1.0,
            effective_teacher_coefficient=1.0,
            coefficient_policy_sha256=SHA_A,
        ),
    )


def _accepted_decision(
    addressed: AddressedPackedTrace,
    *,
    with_candidate_evidence: bool = True,
) -> SemanticActive8TraceDecision:
    policy = build_semantic_active8_admission_policy()
    classification = classify_semantic_action(
        addressed.trace.steps[0],
        step_index=0,
        policy=policy,
    )
    source = addressed.path.state_at(0)
    successor = addressed.path.state_at(1)
    evidence = (
        SemanticExactCandidateEvidence(
            supported=True,
            action_sha256=str(classification.action_sha256),
            source_state_sha256=persistent_slot_state_sha256(source),
            target_state_sha256=persistent_slot_state_sha256(successor),
            canonical_successor_key=canonical_state_key(successor),
            raw_mark_count=20,
            canonical_successor_count=10,
            matching_mark_count=1,
            successor_alias_count=3,
            exclusion_reason=None,
        )
        if with_candidate_evidence
        else None
    )
    return SemanticActive8TraceDecision(
        trace_id=addressed.address.trace_id,
        policy_sha256=policy.policy_sha256,
        semantic_migration_status="admitted",
        semantic_migration_rejection=None,
        active8_status="accepted",
        emits_progress_rows=True,
        action_decisions=(
            SemanticActive8ActionDecision(
                classification=classification,
                candidate_evidence=evidence,
            ),
        ),
        active8_exclusions=(),
    )


def test_registry_is_self_hashed_compact_and_non_authorizing() -> None:
    registry = load_semantic_capability_cell_registry()
    assert len(registry.registry_sha256) == 64
    assert tuple(family for family, _ in registry.family_contexts) == (
        "atom_insert",
        "atom_delete",
        "atom_restate",
        "bond_reorder",
        "bond_reroute",
        "cycle_insert",
        "cycle_attach",
        "ring_system_restate",
    )
    payload = json.loads(REGISTRY_PATH.read_text())
    policy = payload["cell_identity_policy"]
    assert policy["balancing_dimensions"] == [
        "model_family",
        "family_specific_context",
    ]
    assert "data_lane" in policy["separate_nonbalancing_dimensions"]
    assert "raw_mark_count_stratum" in policy["separate_nonbalancing_dimensions"]
    assert payload["training_authorized"] is False
    assert payload["gate_zero_authorized"] is False
    assert payload["t1_authorized"] is False
    assert payload["bounded_p50_authorized"] is False


def test_registry_tamper_and_unknown_context_fail_closed() -> None:
    payload = json.loads(REGISTRY_PATH.read_text())
    tampered = copy.deepcopy(payload)
    tampered["family_contexts"]["atom_insert"].append("invented_birth")
    with pytest.raises(SemanticCapabilityCellError, match="self-hash"):
        validate_semantic_capability_cell_registry(tampered)

    self_consistent_but_unknown = copy.deepcopy(tampered)
    self_consistent_but_unknown["registry_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in self_consistent_but_unknown.items()
            if key != "registry_sha256"
        }
    )
    with pytest.raises(SemanticCapabilityCellError, match="contexts have drifted"):
        validate_semantic_capability_cell_registry(self_consistent_but_unknown)

    unauthorized = copy.deepcopy(payload)
    unauthorized["training_authorized"] = True
    unauthorized["registry_sha256"] = _canonical_sha256(
        {key: value for key, value in unauthorized.items() if key != "registry_sha256"}
    )
    with pytest.raises(SemanticCapabilityCellError, match="must remain false"):
        validate_semantic_capability_cell_registry(unauthorized)

    for binding in (
        "semantic_process_identity_sha256",
        "corpus_contract_file_sha256",
        "classifier_implementation_sha256",
    ):
        stale = copy.deepcopy(payload)
        stale["bindings"][binding] = SHA_A
        stale["registry_sha256"] = _canonical_sha256(
            {key: value for key, value in stale.items() if key != "registry_sha256"}
        )
        with pytest.raises(SemanticCapabilityCellError, match="bindings have drifted"):
            validate_semantic_capability_cell_registry(stale)


def test_family_contexts_cover_birth_death_graft_and_cycle_direction() -> None:
    system = editing_v2_semantic_rewrite_system()

    carbon = _state("C")
    null = system.apply(carbon, "atom_delete", AtomDelete(0))
    root_birth = AtomInsert(
        slot=0,
        atom_type=int(carbon.atom_types[0]),
        formal_charge=int(carbon.formal_charges[0]),
        implicit_h_count=int(carbon.implicit_h_counts[0]),
        neighbors=(),
    )
    assert classify_action_family_context(
        null,
        carbon,
        RewriteStep("atom_insert", root_birth),
    ) == ("atom_insert", "root_birth")

    chain = _state("CC")
    shortened = system.apply(chain, "atom_delete", AtomDelete(1))
    assert classify_action_family_context(
        chain,
        shortened,
        RewriteStep("atom_delete", AtomDelete(1)),
    ) == ("atom_delete", "leaf_death")

    ring = _state("C1CCCCC1")
    opened_by_atom_death = system.apply(ring, "atom_delete", AtomDelete(0))
    assert classify_action_family_context(
        ring,
        opened_by_atom_death,
        RewriteStep("atom_delete", AtomDelete(0)),
    ) == ("atom_delete", "connected_nonleaf_death")

    graft_source = _state("c1ccccc1CC")
    graft_action = enumerate_pendant_graft_actions(graft_source)[0]
    graft_target = system.apply(graft_source, "bond_reroute", graft_action)
    graft_family, graft_context = classify_action_family_context(
        graft_source,
        graft_target,
        RewriteStep("bond_reroute", graft_action),
    )
    assert graft_family == "bond_reroute"
    assert graft_context.endswith("_cyclic_source")

    open_chain = _state("CCCCCC")
    close = CycleCloseEdge(0, 5, 1)
    closed_ring = system.apply(open_chain, "cycle_close", close)
    assert classify_action_family_context(
        open_chain,
        closed_ring,
        RewriteStep("cycle_close", close),
    ) == ("cycle_insert", "close_to_monocyclic_ring_system")
    opening = CycleOpenEdge(0, 5)
    reopened = system.apply(closed_ring, "cycle_open", opening)
    assert classify_action_family_context(
        closed_ring,
        reopened,
        RewriteStep("cycle_open", opening),
    ) == ("cycle_attach", "open_from_monocyclic_ring_system")


def test_family_contexts_cover_state_change_families() -> None:
    system = editing_v2_semantic_rewrite_system()
    ethane = _state("CC")
    double = BondReorder(0, 1, 2)
    ethene = system.apply(ethane, "bond_reorder", double)
    assert classify_action_family_context(
        ethane,
        ethene,
        RewriteStep("bond_reorder", double),
    ) == ("bond_reorder", "bond_order_increase")

    nitrogen_class = next(
        index
        for index, (element, _valence) in enumerate(ORGANIC_VOCABULARY.classes)
        if element == ELEMENT_TO_IDX["N"]
    )
    restate = SemanticAtomRestate(0, nitrogen_class)
    assert classify_action_family_context(
        _state("C"),
        _state("N"),
        RewriteStep("atom_restate_semantic", restate),
    ) == ("atom_restate", "element_identity_change")

    saturated_ring = _state("C1CCCCC1")
    aromatic_ring = _state("c1ccccc1")
    ring_restate = RingSystemRestate(
        (
            BondOrderChange(0, 1, 2),
            BondOrderChange(2, 3, 2),
            BondOrderChange(4, 5, 2),
        )
    )
    executed_aromatic_ring = system.apply(
        saturated_ring,
        "ring_system_restate",
        ring_restate,
    )
    assert canonical_state_key(executed_aromatic_ring) == canonical_state_key(
        aromatic_ring
    )
    assert classify_action_family_context(
        saturated_ring,
        executed_aromatic_ring,
        RewriteStep("ring_system_restate", ring_restate),
    ) == ("ring_system_restate", "aromatization")


@pytest.mark.parametrize(
    ("smiles", "expected_context"),
    (
        ("C1CCC2(CC1)CCCC2", "open_from_articulated_polycyclic_ring_system"),
        ("C1CCC2CCCCC2C1", "open_from_nonarticulated_polycyclic_ring_system"),
    ),
)
def test_cycle_context_uses_graph_topology_without_template_labels(
    smiles: str,
    expected_context: str,
) -> None:
    source = _state(smiles)
    action = enumerate_cycle_open_edges(source)[0]
    successor = editing_v2_semantic_rewrite_system().apply(source, "cycle_open", action)
    assert classify_action_family_context(
        source,
        successor,
        RewriteStep("cycle_open", action),
    ) == ("cycle_attach", expected_context)


def test_accepted_progress_assignment_is_deterministic_and_separates_audit_axes() -> (
    None
):
    source = _state("CC")
    step = RewriteStep("atom_delete", AtomDelete(1))
    successor = editing_v2_semantic_rewrite_system().apply(
        source,
        "atom_delete",
        step.action,
    )
    addressed = _addressed(source, successor, step)
    progress = _progress(addressed)
    decision = _accepted_decision(addressed)

    first = classify_accepted_semantic_progress(addressed, progress, decision)
    second = classify_accepted_semantic_progress(addressed, progress, decision)
    assert first == second
    assert first.capability_cell_id == "editing_v2_active8_v1:atom_delete:leaf_death"
    assert first.data_lane == "operator_aware_real_endpoint"
    assert first.partition_role == "train"
    assert first.raw_mark_count_stratum == "marks_017_064"
    assert first.canonical_successor_count_stratum == "successors_005_016"
    assert first.successor_alias_multiplicity_stratum == "aliases_002_004"
    assert first.matching_mark_count == 1
    assert first.atom_element_transition == (
        ("source_element", "C"),
        ("successor_element", "ABSENT"),
    )
    assert first.endpoint_evidence_roles[0] == (
        "source_endpoint",
        "observed_real_endpoint",
    )
    assert first.as_payload()["training_authorized"] is False
    assert first.as_payload()["assignment_sha256"] == first.assignment_sha256


def test_assignment_binds_groups_mappings_and_effective_coefficients() -> None:
    source = _state("CC")
    step = RewriteStep("atom_delete", AtomDelete(1))
    successor = editing_v2_semantic_rewrite_system().apply(
        source,
        "atom_delete",
        step.action,
    )
    addressed = _addressed(source, successor, step)
    progress = _progress(addressed)
    decision = _accepted_decision(addressed)
    baseline = classify_accepted_semantic_progress(addressed, progress, decision)

    changed_groups = replace(
        progress,
        groups=replace(progress.groups, source_group_id="source-group-2"),
    )
    changed_coefficients = replace(
        progress,
        sampling_coefficients=replace(
            progress.sampling_coefficients,
            effective_teacher_coefficient=2.0,
        ),
    )
    changed_mapping = replace(
        progress,
        mappings=replace(progress.mappings, constant_core_mapping=((0, 0),)),
    )
    assignments = tuple(
        classify_accepted_semantic_progress(addressed, item, decision)
        for item in (changed_groups, changed_coefficients, changed_mapping)
    )
    assert all(
        item.assignment_sha256 != baseline.assignment_sha256 for item in assignments
    )
    assert all(
        item.semantic_progress_evidence_sha256
        != baseline.semantic_progress_evidence_sha256
        for item in assignments
    )


def test_duplicate_teacher_direction_and_lane_evidence_mismatch_fail_closed() -> None:
    source = _state("CC")
    step = RewriteStep("atom_delete", AtomDelete(1))
    successor = editing_v2_semantic_rewrite_system().apply(
        source,
        "atom_delete",
        step.action,
    )
    addressed = _addressed(source, successor, step)
    progress = _progress(addressed)
    decision = _accepted_decision(addressed)
    action_decision = decision.action_decisions[0]
    assert action_decision.candidate_evidence is not None
    duplicate = replace(
        decision,
        action_decisions=(
            replace(
                action_decision,
                candidate_evidence=replace(
                    action_decision.candidate_evidence,
                    matching_mark_count=2,
                ),
            ),
        ),
    )
    with pytest.raises(SemanticCapabilityCellError, match="exactly once"):
        classify_accepted_semantic_progress(addressed, progress, duplicate)

    wrong_direction = replace(
        progress,
        record_membership_cells=("cardinality_growth",),
    )
    with pytest.raises(SemanticCapabilityCellError, match="endpoint deltas"):
        classify_accepted_semantic_progress(addressed, wrong_direction, decision)

    synthetic_addressed = _addressed(
        source,
        successor,
        step,
        lane="reversible_synthetic_walk",
    )
    synthetic_progress = _progress(synthetic_addressed)
    synthetic_decision = _accepted_decision(synthetic_addressed)
    with pytest.raises(SemanticCapabilityCellError, match="lane and evidence"):
        classify_accepted_semantic_progress(
            synthetic_addressed,
            synthetic_progress,
            synthetic_decision,
        )


def test_unknown_action_missing_evidence_and_terminal_row_fail_closed() -> None:
    source = _state("CC")
    with pytest.raises(SemanticCapabilityCellError, match="outside Action V4"):
        classify_action_family_context(
            source,
            source,
            RewriteStep("bond_insert", BondInsert(0, 1, 1)),
        )

    step = RewriteStep("atom_delete", AtomDelete(1))
    successor = editing_v2_semantic_rewrite_system().apply(
        source,
        "atom_delete",
        step.action,
    )
    addressed = _addressed(source, successor, step)
    progress = _progress(addressed)
    decision = _accepted_decision(addressed, with_candidate_evidence=False)
    with pytest.raises(
        SemanticCapabilityCellError, match="lacks exact candidate evidence"
    ):
        classify_accepted_semantic_progress(addressed, progress, decision)

    complete_decision = _accepted_decision(addressed)
    terminal = replace(progress, progress_index=1)
    with pytest.raises(SemanticCapabilityCellError, match="terminal or out-of-range"):
        classify_accepted_semantic_progress(addressed, terminal, complete_decision)
