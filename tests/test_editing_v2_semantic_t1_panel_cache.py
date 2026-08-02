from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_semantic_active8_admission import (
    build_semantic_active8_admission_policy,
)
from compose_v4.data.editing_v2_semantic_capability_cells import (
    load_semantic_capability_cell_registry,
)
from compose_v4.data.immutable_artifact import ImmutableArtifactError
from compose_v4.experiments.editing_v2_semantic_t1_panel_cache import (
    CACHE_HANDOFF,
    GPU_CACHE_POLICY,
    PANEL_SCHEMA,
    PANEL_STATUS,
    REPEATED_PANEL_KIND,
    UNIQUE_PANEL_KIND,
    SemanticT1EmpiricalMultiplicityReceipt,
    SemanticT1GateZeroBinding,
    SemanticT1PanelError,
    SemanticT1PanelRequest,
    SemanticT1TeacherOccurrence,
    build_semantic_t1_panel_from_occurrence_factory,
    build_semantic_t1_repeated_panel_from_occurrence_factory,
    deserialize_semantic_t1_panel,
    read_semantic_t1_panel,
    serialize_semantic_t1_panel,
    write_semantic_t1_panel,
)

_REGISTRY = load_semantic_capability_cell_registry()
_ACTIVE8_POLICY = build_semantic_active8_admission_policy()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


_CONTEXT = {
    "atom_insert": "one_neighbor_birth",
    "atom_delete": "leaf_death",
    "atom_restate": "element_identity_change",
    "bond_reorder": "bond_order_increase",
    "bond_reroute": "single_atom_pendant_acyclic_source",
    "cycle_insert": "close_to_monocyclic_ring_system",
    "cycle_attach": "open_from_monocyclic_ring_system",
    "ring_system_restate": "aromatization",
}


def _request(*, atom_insert_limit: int = 2) -> SemanticT1PanelRequest:
    limits = {family: 2 for family in ACTIVE8_FAMILIES}
    limits["atom_insert"] = atom_insert_limit
    return SemanticT1PanelRequest.create(
        request_id="editing_v2_semantic_t1_test",
        source_revision_sha256=_digest("source-revision"),
        support_time=0.5,
        minimum_entries_by_family={family: 1 for family in ACTIVE8_FAMILIES},
        maximum_entries_by_family=limits,
    )


def test_panel_fails_when_any_family_is_below_its_request_minimum() -> None:
    source_identity = _digest("below-minimum")
    request = SemanticT1PanelRequest.create(
        request_id="minimum_two",
        source_revision_sha256=_digest("source-revision"),
        support_time=0.5,
        minimum_entries_by_family={family: 2 for family in ACTIVE8_FAMILIES},
        maximum_entries_by_family={family: 2 for family in ACTIVE8_FAMILIES},
    )
    with pytest.raises(SemanticT1PanelError, match="below the frozen minimum"):
        _build(_corpus(source_identity), request=request)


def _binding(source_identity: str) -> SemanticT1GateZeroBinding:
    return SemanticT1GateZeroBinding(
        contract_sha256=_digest("contract"),
        evidence_sha256=_digest("evidence"),
        evidence_file_sha256=_digest("evidence-file"),
        decision_sha256=_digest("decision"),
        decision_file_sha256=_digest("decision-file"),
        completion_sha256=_digest("completion"),
        decision_source_inventory_sha256=source_identity,
        model_runtime_identity_sha256=_digest("model-runtime"),
        model_source_revision_sha256=_digest("source-revision"),
        initial_model_state_sha256=_digest("initial-model-state"),
    )


def _occurrence(
    family: str,
    tag: str,
    *,
    source_identity: str,
    source_tag: str | None = None,
    successor_tag: str | None = None,
    action_tag: str | None = None,
    raw_stratum: str = "raw_small",
    canonical_successor_stratum: str = "successor_small",
    alias_count: int = 1,
    partition_role: str = "train",
) -> SemanticT1TeacherOccurrence:
    context = _CONTEXT[family]
    source_key_tag = source_tag or tag
    successor_key_tag = successor_tag or tag
    return SemanticT1TeacherOccurrence(
        decision_source_inventory_sha256=source_identity,
        decision_sha256=_digest(f"decision:{tag}"),
        trace_address_sha256=_digest(f"trace-address:{tag}"),
        source_progress_address_sha256=_digest(f"source-progress:{tag}"),
        successor_progress_address_sha256=_digest(f"target-progress:{tag}"),
        trace_id=f"trace:{tag}",
        packed_shard_content_sha256=_digest(f"shard:{tag}"),
        packed_shard_name=f"shard_{tag}.jsonl.gz",
        packed_entry_index=int(_digest(f"entry:{tag}")[:6], 16),
        trace_source_key=f"trace-source:{tag}",
        trace_target_key=f"trace-target:{tag}",
        path_length=2,
        progress_index=0,
        data_lane="reversible_synthetic_walk",
        partition_role=partition_role,
        action_sha256=_digest(f"action:{action_tag or tag}"),
        source_state_sha256=_digest(f"source-state:{source_key_tag}"),
        target_state_sha256=_digest(f"target-state:{tag}"),
        source_canonical_key=f"source:{source_key_tag}",
        successor_canonical_key=f"successor:{successor_key_tag}",
        model_family=family,
        family_context=context,
        capability_cell_id=f"editing_v2_active8_v1:{family}:{context}",
        raw_mark_count=8,
        raw_mark_count_stratum=raw_stratum,
        canonical_successor_count=5,
        canonical_successor_count_stratum=canonical_successor_stratum,
        successor_alias_multiplicity=alias_count,
        successor_alias_multiplicity_stratum=(
            "alias_single" if alias_count == 1 else "alias_multiple"
        ),
        matching_mark_count=1,
        registry_sha256=_REGISTRY.registry_sha256,
        process_identity_sha256=_REGISTRY.process_identity_sha256,
        corpus_contract_file_sha256=_REGISTRY.corpus_contract_file_sha256,
        classifier_implementation_sha256=(_REGISTRY.classifier_implementation_sha256),
        active8_policy_sha256=_ACTIVE8_POLICY.policy_sha256,
        assignment_sha256=_digest(f"assignment:{tag}"),
    )


def _corpus(source_identity: str) -> tuple[SemanticT1TeacherOccurrence, ...]:
    rows = [
        _occurrence(family, family, source_identity=source_identity) for family in ACTIVE8_FAMILIES
    ]
    # A second atom-insert difficulty stratum must be represented.
    rows.append(
        _occurrence(
            "atom_insert",
            "atom_insert_large",
            source_identity=source_identity,
            raw_stratum="raw_large",
            canonical_successor_stratum="successor_large",
        )
    )
    # Two distinct teacher marks and exact slot targets in one canonical
    # successor fiber collapse to one unit-weight molecular target.
    rows[0] = _occurrence(
        "atom_insert",
        "atom_insert_alias_a",
        source_identity=source_identity,
        source_tag="atom_insert_shared_source",
        successor_tag="atom_insert_shared_target",
        action_tag="atom_insert_alias_a",
        alias_count=2,
    )
    rows.append(
        _occurrence(
            "atom_insert",
            "atom_insert_alias_b",
            source_identity=source_identity,
            source_tag="atom_insert_shared_source",
            successor_tag="atom_insert_shared_target",
            action_tag="atom_insert_alias_b",
            alias_count=2,
        )
    )
    return tuple(rows)


def _build(
    rows: tuple[SemanticT1TeacherOccurrence, ...],
    *,
    request: SemanticT1PanelRequest,
    reverse: bool = False,
):
    source_identity = rows[0].decision_source_inventory_sha256
    ordered = tuple(reversed(rows)) if reverse else rows
    registry = _REGISTRY
    return build_semantic_t1_panel_from_occurrence_factory(
        lambda: iter(ordered),
        request=request,
        gate_zero_binding=_binding(source_identity),
        decision_source_identity={
            "inventory_sha256": source_identity,
            "status": "VERIFIED_ACTIVE8_DECISION_SOURCE_NO_DOWNSTREAM_AUTHORITY",
            "process_identity_sha256": registry.process_identity_sha256,
            "policy_sha256": _ACTIVE8_POLICY.policy_sha256,
            "counts": {"traces": len(rows)},
        },
        registry=registry,
    )


def test_train_panel_collapses_aliases_and_is_order_deterministic() -> None:
    source_identity = _digest("decision-source")
    rows = _corpus(source_identity)

    forward = _build(rows, request=_request())
    reverse = _build(rows, request=_request(), reverse=True)

    assert forward.entries == reverse.entries
    assert forward.cache_trace_inputs == reverse.cache_trace_inputs
    assert forward.coverage_receipts == reverse.coverage_receipts
    assert (
        forward.first_pass_train_teacher_stream_sha256
        != reverse.first_pass_train_teacher_stream_sha256
    )
    assert len(forward.entries) == len(ACTIVE8_FAMILIES) + 1
    assert {entry.model_family for entry in forward.entries} == set(ACTIVE8_FAMILIES)
    assert all(entry.objective_coefficient == 1 for entry in forward.entries)
    collapsed = next(
        entry
        for entry in forward.entries
        if entry.source_canonical_key == "source:atom_insert_shared_source"
    )
    assert collapsed.teacher_occurrence_count == 2
    assert collapsed.teacher_action_identity_count == 2
    assert collapsed.production_successor_alias_multiplicity == 2
    assert len(collapsed.target_state_occurrence_counts) == 2
    assert sum(count for _, count in collapsed.teacher_action_occurrence_counts) == 2
    assert len(
        {
            (
                entry.source_state_sha256,
                entry.support_time_hex,
                entry.successor_canonical_key,
            )
            for entry in forward.entries
        }
    ) == len(forward.entries)
    assert all(
        entry.panel_entry_sha256
        for cache_input in forward.cache_trace_inputs
        for entry in forward.entries
        if entry.panel_entry_sha256 in cache_input.panel_entry_sha256s
    )
    assert forward.request.as_payload()["panel_kind"] == UNIQUE_PANEL_KIND
    assert forward.identity_body()["objective"]["panel_kind"] == UNIQUE_PANEL_KIND
    assert forward.single_target_source_state_count == 9
    assert forward.repeated_source_state_count == 0
    assert len(forward.single_target_source_inventory_sha256) == 64
    assert len(forward.repeated_source_inventory_sha256) == 64


def test_unique_capacity_panel_excludes_multi_target_exact_sources() -> None:
    source_identity = _digest("multi-target-capacity-source")
    rows = list(_corpus(source_identity))
    rows.append(
        _occurrence(
            "atom_insert",
            "atom_insert_distinct_target",
            source_identity=source_identity,
            source_tag="atom_insert_shared_source",
            successor_tag="a_second_canonical_target",
        )
    )

    artifact = _build(tuple(rows), request=_request())

    assert all(
        entry.source_canonical_key != "source:atom_insert_shared_source"
        for entry in artifact.entries
    )
    assert len(
        {(entry.source_state_sha256, entry.support_time_hex) for entry in artifact.entries}
    ) == len(artifact.entries)
    assert {entry.model_family for entry in artifact.entries} == set(ACTIVE8_FAMILIES)
    assert artifact.single_target_source_state_count == 8
    assert artifact.repeated_source_state_count == 1


def test_repeated_panel_uses_proven_empirical_units_not_raw_rows() -> None:
    source_identity = _digest("repeated-empirical-source")
    rows = (
        _occurrence(
            "atom_insert",
            "target_a_encoding_1",
            source_identity=source_identity,
            source_tag="shared_repeated_source",
            successor_tag="target_a",
        ),
        _occurrence(
            "atom_insert",
            "target_a_encoding_2",
            source_identity=source_identity,
            source_tag="shared_repeated_source",
            successor_tag="target_a",
        ),
        _occurrence(
            "atom_insert",
            "target_b_observation_1",
            source_identity=source_identity,
            source_tag="shared_repeated_source",
            successor_tag="target_b",
        ),
        _occurrence(
            "atom_insert",
            "target_b_observation_2",
            source_identity=source_identity,
            source_tag="shared_repeated_source",
            successor_tag="target_b",
        ),
    )
    observation_units = (
        _digest("observed-pair-a"),
        _digest("observed-pair-a"),
        _digest("observed-pair-b1"),
        _digest("observed-pair-b2"),
    )
    receipts = {
        occurrence.occurrence_sha256: SemanticT1EmpiricalMultiplicityReceipt.create(
            occurrence,
            observation_unit_sha256=observation_unit,
            provenance_artifact_sha256=_digest("empirical-provenance"),
        )
        for occurrence, observation_unit in zip(
            rows,
            observation_units,
            strict=True,
        )
    }
    selection = build_semantic_t1_repeated_panel_from_occurrence_factory(
        lambda: iter(rows),
        request=_request(),
        gate_zero_binding=_binding(source_identity),
        decision_source_identity={
            "inventory_sha256": source_identity,
            "status": "VERIFIED_ACTIVE8_DECISION_SOURCE_NO_DOWNSTREAM_AUTHORITY",
            "process_identity_sha256": _REGISTRY.process_identity_sha256,
            "policy_sha256": _ACTIVE8_POLICY.policy_sha256,
        },
        registry=_REGISTRY,
        empirical_receipts=receipts,
        empirical_provenance_artifact_sha256=_digest("empirical-provenance"),
        maximum_source_states=1,
    )

    assert selection.identity_body()["panel_kind"] == REPEATED_PANEL_KIND
    assert len(selection.groups) == 1
    group = selection.groups[0]
    assert group.raw_teacher_occurrence_count == 4
    assert group.independent_observation_count == 3
    probabilities = {
        target.successor_canonical_key: (
            target.probability_numerator,
            target.probability_denominator,
        )
        for target in group.targets
    }
    assert probabilities == {
        "successor:target_a": (1, 3),
        "successor:target_b": (2, 3),
    }


def test_repeated_panel_fails_without_complete_multiplicity_provenance() -> None:
    source_identity = _digest("missing-repeated-provenance")
    rows = (
        _occurrence(
            "cycle_attach",
            "open_target_a",
            source_identity=source_identity,
            source_tag="shared_ring_source",
            successor_tag="opened_a",
        ),
        _occurrence(
            "cycle_attach",
            "open_target_b",
            source_identity=source_identity,
            source_tag="shared_ring_source",
            successor_tag="opened_b",
        ),
    )
    receipt = SemanticT1EmpiricalMultiplicityReceipt.create(
        rows[0],
        observation_unit_sha256=_digest("only-one-provenance-unit"),
        provenance_artifact_sha256=_digest("partial-provenance"),
    )
    with pytest.raises(
        SemanticT1PanelError,
        match="empirical multiplicity provenance is incomplete",
    ):
        build_semantic_t1_repeated_panel_from_occurrence_factory(
            lambda: iter(rows),
            request=_request(),
            gate_zero_binding=_binding(source_identity),
            decision_source_identity={
                "inventory_sha256": source_identity,
                "status": "VERIFIED_ACTIVE8_DECISION_SOURCE_NO_DOWNSTREAM_AUTHORITY",
                "process_identity_sha256": _REGISTRY.process_identity_sha256,
                "policy_sha256": _ACTIVE8_POLICY.policy_sha256,
            },
            registry=_REGISTRY,
            empirical_receipts={rows[0].occurrence_sha256: receipt},
            empirical_provenance_artifact_sha256=_digest("partial-provenance"),
            maximum_source_states=1,
        )

    shared_unit = _digest("conflicting-observation-unit")
    complete_but_conflicting = {
        occurrence.occurrence_sha256: SemanticT1EmpiricalMultiplicityReceipt.create(
            occurrence,
            observation_unit_sha256=shared_unit,
            provenance_artifact_sha256=_digest("conflicting-provenance"),
        )
        for occurrence in rows
    }
    with pytest.raises(
        SemanticT1PanelError,
        match="observation unit maps to multiple canonical successors",
    ):
        build_semantic_t1_repeated_panel_from_occurrence_factory(
            lambda: iter(rows),
            request=_request(),
            gate_zero_binding=_binding(source_identity),
            decision_source_identity={
                "inventory_sha256": source_identity,
                "status": "VERIFIED_ACTIVE8_DECISION_SOURCE_NO_DOWNSTREAM_AUTHORITY",
                "process_identity_sha256": _REGISTRY.process_identity_sha256,
                "policy_sha256": _ACTIVE8_POLICY.policy_sha256,
            },
            registry=_REGISTRY,
            empirical_receipts=complete_but_conflicting,
            empirical_provenance_artifact_sha256=_digest("conflicting-provenance"),
            maximum_source_states=1,
        )


def test_panel_covers_every_observed_difficulty_stratum_before_repetition() -> None:
    source_identity = _digest("coverage-source")
    artifact = _build(_corpus(source_identity), request=_request())

    atom_insert_receipts = [
        receipt
        for receipt in artifact.coverage_receipts
        if receipt.stratum.model_family == "atom_insert"
    ]
    assert len(atom_insert_receipts) == 2
    assert all(receipt.selected_candidate_count == 1 for receipt in atom_insert_receipts)
    assert sum(receipt.train_teacher_occurrence_count for receipt in atom_insert_receipts) == 3

    with pytest.raises(SemanticT1PanelError, match="cannot cover all observed strata"):
        _build(
            _corpus(source_identity),
            request=_request(atom_insert_limit=1),
        )


def test_request_label_does_not_reroll_selection_and_revision_is_bound() -> None:
    source_identity = _digest("request-policy-source")
    rows = _corpus(source_identity)
    first = _build(rows, request=_request())
    relabeled = _build(
        rows,
        request=SemanticT1PanelRequest.create(
            request_id="descriptive_label_only",
            source_revision_sha256=_digest("source-revision"),
            support_time=0.5,
            minimum_entries_by_family={family: 1 for family in ACTIVE8_FAMILIES},
            maximum_entries_by_family={family: 2 for family in ACTIVE8_FAMILIES},
        ),
    )
    assert first.request.request_sha256 != relabeled.request.request_sha256
    assert first.request.selection_seed_sha256 == relabeled.request.selection_seed_sha256
    assert first.entries == relabeled.entries

    with pytest.raises(SemanticT1PanelError, match="source revision differs"):
        _build(
            rows,
            request=SemanticT1PanelRequest.create(
                request_id="wrong_revision",
                source_revision_sha256=_digest("another-revision"),
                support_time=0.5,
                minimum_entries_by_family={family: 1 for family in ACTIVE8_FAMILIES},
                maximum_entries_by_family={family: 2 for family in ACTIVE8_FAMILIES},
            ),
        )


def test_cross_family_alias_fails_closed_instead_of_double_counting() -> None:
    source_identity = _digest("cross-family-source")
    rows = list(_corpus(source_identity))
    rows.append(
        _occurrence(
            "atom_delete",
            "cross_family_alias",
            source_identity=source_identity,
            source_tag="atom_insert_shared_source",
            successor_tag="atom_insert_shared_target",
        )
    )
    with pytest.raises(
        SemanticT1PanelError,
        match="molecular objective unit was selected in multiple families",
    ):
        _build(tuple(rows), request=_request())


def test_occurrence_rejects_heldout_role_and_request_requires_all_families() -> None:
    source_identity = _digest("heldout-source")
    with pytest.raises(ValueError, match="train-only"):
        _occurrence(
            "cycle_attach",
            "heldout",
            source_identity=source_identity,
            partition_role="validation",
        )
    with pytest.raises(ValueError, match="every Active8 family"):
        SemanticT1PanelRequest.create(
            request_id="missing_family",
            source_revision_sha256=_digest("source-revision"),
            support_time=0.5,
            minimum_entries_by_family={"atom_insert": 1},
            maximum_entries_by_family={"atom_insert": 2},
        )


def test_panel_rejects_source_drift_between_its_two_passes() -> None:
    source_identity = _digest("drifting-source")
    rows = _corpus(source_identity)
    calls = 0

    def drifting_factory():
        nonlocal calls
        calls += 1
        return iter(rows if calls == 1 else rows[:-1])

    with pytest.raises(
        SemanticT1PanelError,
        match="source changed|did not reproduce",
    ):
        build_semantic_t1_panel_from_occurrence_factory(
            drifting_factory,
            request=_request(),
            gate_zero_binding=_binding(source_identity),
            decision_source_identity={
                "inventory_sha256": source_identity,
                "status": ("VERIFIED_ACTIVE8_DECISION_SOURCE_NO_DOWNSTREAM_AUTHORITY"),
                "process_identity_sha256": _REGISTRY.process_identity_sha256,
                "policy_sha256": _ACTIVE8_POLICY.policy_sha256,
            },
            registry=_REGISTRY,
        )


def test_panel_rejects_same_count_drift_outside_selected_groups() -> None:
    source_identity = _digest("same-count-drifting-source")
    rows = list(_corpus(source_identity))
    rows.extend(
        _occurrence(
            "atom_delete",
            f"unselected_atom_delete_{index}",
            source_identity=source_identity,
        )
        for index in range(4)
    )
    rows = list(rows)
    reference = _build(tuple(rows), request=_request())
    selected_units = {
        (entry.source_state_sha256, entry.successor_canonical_key) for entry in reference.entries
    }
    unselected_index = next(
        index
        for index, occurrence in enumerate(rows)
        if (
            occurrence.source_state_sha256,
            occurrence.successor_canonical_key,
        )
        not in selected_units
    )
    changed = list(rows)
    changed[unselected_index] = _occurrence(
        "atom_delete",
        "same_unselected_group_new_mark",
        source_identity=source_identity,
        source_tag=rows[unselected_index].source_canonical_key.removeprefix("source:"),
        successor_tag=rows[unselected_index].successor_canonical_key.removeprefix("successor:"),
        action_tag="changed_unselected_action",
    )
    calls = 0

    def drifting_factory():
        nonlocal calls
        calls += 1
        return iter(rows if calls == 1 else changed)

    with pytest.raises(SemanticT1PanelError, match="source changed"):
        build_semantic_t1_panel_from_occurrence_factory(
            drifting_factory,
            request=_request(),
            gate_zero_binding=_binding(source_identity),
            decision_source_identity={
                "inventory_sha256": source_identity,
                "status": "VERIFIED_ACTIVE8_DECISION_SOURCE_NO_DOWNSTREAM_AUTHORITY",
                "process_identity_sha256": _REGISTRY.process_identity_sha256,
                "policy_sha256": _ACTIVE8_POLICY.policy_sha256,
            },
            registry=_REGISTRY,
        )


def test_artifact_is_canonical_nonauthorizing_and_immutable(tmp_path: Path) -> None:
    source_identity = _digest("artifact-source")
    artifact = _build(_corpus(source_identity), request=_request())
    encoded = serialize_semantic_t1_panel(artifact)
    payload = json.loads(encoded)

    assert encoded.endswith(b"\n")
    assert payload["schema"] == PANEL_SCHEMA
    assert payload["status"] == PANEL_STATUS
    assert payload["training_authorized"] is False
    assert payload["t1_authorized"] is False
    assert payload["bounded_p50_authorized"] is False
    assert payload["objective"]["hazard_included"] is False
    assert payload["cache_handoff"]["kind"] == CACHE_HANDOFF
    assert payload["cache_handoff"]["gpu_policy"] == GPU_CACHE_POLICY
    assert payload["counts"]["single_target_source_state_count"] == 9
    assert payload["counts"]["repeated_source_state_count"] == 0
    assert len(payload["counts"]["single_target_source_inventory_sha256"]) == 64
    assert len(payload["counts"]["repeated_source_inventory_sha256"]) == 64
    body = dict(payload)
    supplied = body.pop("artifact_sha256")
    assert (
        supplied
        == hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert (
        deserialize_semantic_t1_panel(
            encoded,
            expected_artifact_sha256=artifact.artifact_sha256,
            expected_panel_implementation_sha256=(artifact.panel_implementation_sha256),
        )
        == artifact
    )

    destination = tmp_path / "panel.json"
    assert write_semantic_t1_panel(destination, artifact) is True
    assert write_semantic_t1_panel(destination, artifact) is False
    assert (
        read_semantic_t1_panel(
            destination,
            expected_artifact_sha256=artifact.artifact_sha256,
            expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
            expected_file_bytes=len(encoded),
            expected_panel_implementation_sha256=(artifact.panel_implementation_sha256),
        )
        == artifact
    )
    with pytest.raises(SemanticT1PanelError, match="not canonical JSON"):
        deserialize_semantic_t1_panel(json.dumps(payload, indent=2).encode("utf-8"))

    nested_counts = dict(artifact.decision_source_identity)["counts"]
    nested_counts["traces"] += 1
    with pytest.raises(SemanticT1PanelError, match="stale self-hash"):
        serialize_semantic_t1_panel(artifact)

    changed = _build(
        _corpus(source_identity),
        request=SemanticT1PanelRequest.create(
            request_id="another_request",
            source_revision_sha256=_digest("source-revision"),
            support_time=0.5,
            minimum_entries_by_family={family: 1 for family in ACTIVE8_FAMILIES},
            maximum_entries_by_family={family: 2 for family in ACTIVE8_FAMILIES},
        ),
    )
    with pytest.raises(ImmutableArtifactError, match="different bytes"):
        write_semantic_t1_panel(destination, changed)


def test_artifact_rejects_duplicate_coverage_and_cache_trace_keys() -> None:
    source_identity = _digest("duplicate-artifact-source")
    artifact = _build(_corpus(source_identity), request=_request())
    first_receipt = artifact.coverage_receipts[0]
    with pytest.raises(ValueError, match="receipt strata are not unique"):
        replace(
            artifact,
            coverage_receipts=(
                first_receipt,
                first_receipt,
                *artifact.coverage_receipts[1:],
            ),
            artifact_sha256="",
        )

    first_cache = artifact.cache_trace_inputs[0]
    with pytest.raises(ValueError, match="cache trace keys are not unique"):
        replace(
            artifact,
            cache_trace_inputs=(
                first_cache,
                first_cache,
                *artifact.cache_trace_inputs[1:],
            ),
            artifact_sha256="",
        )
