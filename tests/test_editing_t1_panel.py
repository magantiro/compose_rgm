"""The editing T1 panels are exact, deterministic, and decision-free."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.active8_trace_inventory import (
    Active8TraceAdmission,
    ExactCandidateEvidence,
    inventory_record_for_trace,
)
from compose_v4.data.packed_trace_store import (
    AddressedPackedTrace,
    PackedTraceAddress,
    PackedTraceProgress,
)
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records
from compose_v4.experiments.editing_gate_zero_runtime import (
    FrozenValidationSource,
    ValidationShardBinding,
)
from compose_v4.experiments.editing_t1_panel import (
    EDITING_T1_GLOBAL_FAMILY_SELECTOR,
    EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
    EDITING_T1_UNIQUE_PANEL_KIND,
    EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
    EditingT1PanelError,
    active8_t1_identity,
    build_editing_t1_panel,
    filter_active8_forensics_rows,
    load_editing_t1_panel,
    selected_source_rows,
    validate_charge_policy_exclusions,
    validate_editing_t1_panel,
)
from compose_v4.experiments.editing_t1_successor_runtime import (
    EDITING_T1_LOCAL_ADAPTER_FAMILIES,
    EDITING_T1_RESULT_SCHEMA,
    EDITING_T1_RESULT_STATUS,
    EDITING_T1_RESULT_VERSION,
    EditingT1RuntimeContract,
    EditingT1RuntimeError,
    T1CacheShardReceipt,
    editing_t1_implementation_sha256,
    editing_t1_implementation_sources,
    load_editing_t1_result,
    load_editing_t1_runtime_contract,
    materialize_t1_panel,
    require_editing_t1_family_scope_applicable,
    run_editing_t1_arm,
    validate_editing_t1_result,
    validate_t1_active8_runtime_binding,
    validate_t1_cache_shard_receipt,
)
from compose_v4.experiments.factorized_successor_training import (
    rewrite_action_codec_sha256,
)
from compose_v4.experiments.ringcore_semantic_sidecar import (
    LoadedSemanticCellSidecar,
    SemanticSidecarRow,
    SemanticSidecarSourceShard,
    read_semantic_cell_sidecar,
)
from compose_v4.experiments.ringcore_successor_leaderboard import (
    load_json_object,
)
from compose_v4.experiments.ringcore_validation_panel import (
    validate_validation_panel_artifact,
)
from compose_v4.experiments.successor_micro_overfit import (
    RINGCORE_EDITING_FAMILIES,
    SuccessorMicroOverfitError,
    configure_micro_overfit_parameters,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.action_codec import canonical_family
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import AtomDelete
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.tracelets import RingSystemDelete
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

ROOT = Path(__file__).resolve().parents[1]
FORENSICS = (
    ROOT / "diagnostics" / "coherence" / "ringcore_v1_family_forensics_initial_2026-07-30.json.gz"
)
SIDECAR = (
    ROOT
    / "diagnostics"
    / "coherence"
    / "ringcore_v1_validation_semantic_sidecar_2026-07-30.jsonl.gz"
)
SIDECAR_MANIFEST = (
    ROOT
    / "diagnostics"
    / "coherence"
    / "ringcore_v1_validation_semantic_sidecar_2026-07-30.manifest.json"
)
CHARGE_POLICY_AUDIT = (
    ROOT / "diagnostics" / "coherence" / "packed_charge_policy_audit_v1_2026-07-30.json"
)
CHARGE_POLICY_EXCLUSIONS = (
    ROOT / "diagnostics" / "coherence" / "packed_charge_policy_exclusions_v1_2026-07-30.json"
)
FROZEN_PANEL = (
    ROOT
    / "diagnostics"
    / "coherence"
    / "editing_t1_successor_panel_v3_active8_2026-07-30.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _active8_admission_for_forensics(
    forensics: dict,
    *,
    excluded: set[tuple[str, int]] | None = None,
    unified_packed_manifest_sha256: str = "3" * 64,
) -> Active8TraceAdmission:
    excluded = set() if excluded is None else set(excluded)
    decisions: dict[str, list[tuple[str, bool]]] = {}
    lanes: dict[tuple[str, str, str], str] = {}
    metadata: dict[str, MappingProxyType] = {}
    for row in forensics["rows"]:
        state_ref = row["exact_state_ref"]
        digest = state_ref["shard_sha256"]
        entry_index = state_ref["record_index"]
        shard_path = Path(state_ref["shard_name"])
        prefix = f"{state_ref['shard_name']}:{entry_index}:"
        assert row["record_key"].startswith(prefix)
        trace_id = row["record_key"][len(prefix) :]
        current = decisions.setdefault(digest, [])
        while len(current) <= entry_index:
            current.append((f"unobserved-{len(current)}", True))
        prior_id, prior_decision = current[entry_index]
        accepted = (digest, entry_index) not in excluded
        if not prior_id.startswith("unobserved-"):
            assert (prior_id, prior_decision) == (trace_id, accepted)
        current[entry_index] = (trace_id, accepted)
        lane = (shard_path.parts[-3], row["partition"], shard_path.name)
        prior_digest = lanes.setdefault(lane, digest)
        assert prior_digest == digest
        metadata.setdefault(digest, MappingProxyType({}))
    return Active8TraceAdmission(
        manifest_path=Path("/fixture/ACTIVE8_TRACE_INVENTORY.json"),
        manifest_file_sha256="1" * 64,
        inventory_sha256="2" * 64,
        unified_packed_manifest_sha256=unified_packed_manifest_sha256,
        support_contract_sha256="4" * 64,
        effective_source_corpus_cache_sha256="5" * 64,
        decisions_by_digest=MappingProxyType(
            {digest: tuple(values) for digest, values in decisions.items()}
        ),
        shard_digest_by_lane=MappingProxyType(lanes),
        shard_metadata_by_digest=MappingProxyType(metadata),
        counts=MappingProxyType({}),
    )


def _active8_admission_for_record(
    record: PathRecord,
    *,
    accepted: bool,
    support_contract_sha256: str = "4" * 64,
    unified_packed_manifest_sha256: str = "4" * 64,
) -> Active8TraceAdmission:
    address = record.corpus_address
    assert address is not None
    return Active8TraceAdmission(
        manifest_path=Path("/fixture/ACTIVE8_TRACE_INVENTORY.json"),
        manifest_file_sha256="1" * 64,
        inventory_sha256="2" * 64,
        unified_packed_manifest_sha256=unified_packed_manifest_sha256,
        support_contract_sha256=support_contract_sha256,
        effective_source_corpus_cache_sha256="5" * 64,
        decisions_by_digest=MappingProxyType(
            {
                address.packed_shard_content_sha256: (
                    (address.trace_id, accepted),
                )
            }
        ),
        shard_digest_by_lane=MappingProxyType(
            {
                (
                    address.layer,
                    address.partition,
                    address.packed_shard_name,
                ): address.packed_shard_content_sha256
            }
        ),
        shard_metadata_by_digest=MappingProxyType(
            {address.packed_shard_content_sha256: MappingProxyType({})}
        ),
        counts=MappingProxyType({}),
    )


@pytest.fixture(scope="module")
def frozen_inputs():
    with gzip.open(FORENSICS, "rt") as handle:
        forensics = json.load(handle)
    validate_validation_panel_artifact(
        forensics,
        config=load_json_object(ROOT / "configs" / "ringcore_v1_successor_leaderboard_v1.json"),
        inventory=load_json_object(
            ROOT
            / "diagnostics"
            / "coherence"
            / "ringcore_v1_scientific_a7546e2_frozen_inventory.json"
        ),
    )
    manifest = load_json_object(SIDECAR_MANIFEST)
    sidecar = read_semantic_cell_sidecar(
        SIDECAR,
        SIDECAR_MANIFEST,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_config_sha256=manifest["config_sha256"],
        expected_provenance_sha256=manifest["provenance_sha256"],
    )
    unified_packed_manifest_sha256 = manifest["provenance"]["unified_packed_manifest_sha256"]
    kwargs = {
        "forensics": forensics,
        "forensics_file_sha256": _sha256(FORENSICS),
        "sidecar": sidecar,
        "semantic_sidecar_file_sha256": _sha256(SIDECAR),
        "semantic_sidecar_manifest_file_sha256": _sha256(SIDECAR_MANIFEST),
        "charge_policy_audit": load_json_object(CHARGE_POLICY_AUDIT),
        "charge_policy_audit_file_sha256": _sha256(CHARGE_POLICY_AUDIT),
        "charge_policy_exclusions": load_json_object(CHARGE_POLICY_EXCLUSIONS),
        "charge_policy_exclusions_file_sha256": _sha256(CHARGE_POLICY_EXCLUSIONS),
        "active8_admission": _active8_admission_for_forensics(
            forensics,
            unified_packed_manifest_sha256=(unified_packed_manifest_sha256),
        ),
        "gate_zero_unified_packed_manifest_sha256": (unified_packed_manifest_sha256),
    }
    return forensics, kwargs


def test_active8_t1_panel_is_complete_deterministic_rederivation(
    frozen_inputs,
    tmp_path,
):
    forensics, kwargs = frozen_inputs
    expected = build_editing_t1_panel(**kwargs)
    stale_pre_inventory_panel = json.loads(FROZEN_PANEL.read_text())
    assert stale_pre_inventory_panel != expected
    with pytest.raises(EditingT1PanelError, match="T1 source fields disagree"):
        validate_editing_t1_panel(
            stale_pre_inventory_panel,
            **kwargs,
        )
    panel_path = tmp_path / "editing-t1-active8-panel.json"
    panel_path.write_text(json.dumps(expected))
    observed = expected

    assert observed["training_authorized"] is False
    assert all(value is None for value in observed["thresholds"].values())
    assert set(observed["census"]["unique_rows_by_family"].values()) == {64}
    assert observed["census"]["unique_rows_total"] == 512
    assert observed["census"]["source_forensics_rows_total"] == 2304
    assert observed["census"]["active8_eligible_forensics_rows_total"] == 2304
    assert observed["census"]["active8_excluded_forensics_rows_total"] == 0
    assert (
        observed["census"][
            "active8_excluded_unique_forensics_trace_addresses"
        ]
        == 0
    )
    assert observed["census"]["charge_policy_eligible_forensics_rows_total"] == 2292
    assert observed["census"]["charge_policy_excluded_forensics_rows_total"] == 12
    assert observed["census"]["charge_policy_excluded_unique_forensics_trace_addresses"] == 12
    assert observed["census"]["pilot_family_eligible_forensics_rows_total"] == 2036
    assert observed["census"]["operator_freeze_excluded_forensics_rows_total"] == 256
    assert observed["census"]["global_repeated_multitarget_groups_total"] == 81
    assert observed["census"]["global_repeated_rows_total"] == 173
    assert observed["census"]["global_repeated_cross_family_groups_total"] == 60
    assert observed["census"]["global_repeated_cross_family_rows_total"] == 131
    assert observed["census"]["global_repeated_panel_meets_64_example_minimum"] is True
    assert observed["census"]["within_family_repeated_multitarget_groups_total"] == 24
    assert observed["census"]["within_family_repeated_rows_total"] == 48
    assert observed["selection"]["operator_freeze_excluded_families"] == [
        "ring_system_delete"
    ]
    assert "ring_system_delete" not in observed["selection"]["active_families"]
    assert observed["architecture"]["enable_ring_system_delete"] is False
    assert (
        observed["selection"]["conditional_repeated_state_relationship"]
        == "every within-family repeated row is an exact conditional subset/view "
        "of the primary global repeated evidence; these panels deliberately "
        "reuse observations and must not be treated as statistically independent"
    )
    overlap = observed["census"]["panel_overlap"]
    assert overlap == {
        "unique_state_vs_global_repeated": {
            "shared_source_states": 0,
            "shared_progress_addresses": 0,
            "shared_trace_addresses": 7,
            "shared_source_row_references": 0,
        },
        "unique_state_vs_within_family_repeated": {
            "shared_source_states": 0,
            "shared_progress_addresses": 0,
            "shared_trace_addresses": 1,
            "shared_source_row_references": 0,
        },
        "global_repeated_vs_within_family_repeated": {
            "shared_source_states": 24,
            "shared_progress_addresses": 48,
            "shared_trace_addresses": 48,
            "shared_source_row_references": 48,
        },
        "within_family_repeated_is_exact_row_subset_of_global_repeated": True,
        "statistically_independent_panel_claim": False,
    }
    assert (
        observed["source"]["charge_policy_exclusion_payload_sha256"]
        == "ad18e751b2423ebfe47e6d2684f0932eec0fbe8008ca28342f5c45d59889fc55"
    )
    excluded_trace_ids = validate_charge_policy_exclusions(
        audit=kwargs["charge_policy_audit"],
        audit_file_sha256=kwargs["charge_policy_audit_file_sha256"],
        exclusions=kwargs["charge_policy_exclusions"],
        exclusions_file_sha256=kwargs["charge_policy_exclusions_file_sha256"],
    )
    selected_references = [
        reference
        for references in observed["unique_state_panels"].values()
        for reference in references
    ]
    selected_references.extend(
        reference
        for group in observed["global_repeated_state_panel"]
        for reference in group["rows"]
    )
    selected_references.extend(
        reference
        for groups in observed["within_family_repeated_state_panels"].values()
        for group in groups
        for reference in group["rows"]
    )
    for reference in selected_references:
        row = forensics["rows"][reference["source_row_index"]]
        state_ref = row["exact_state_ref"]
        assert (
            state_ref["shard_sha256"],
            state_ref["record_index"],
        ) not in excluded_trace_ids

    loaded = load_editing_t1_panel(
        panel_path,
        expected_artifact_sha256=observed["artifact_sha256"],
        **kwargs,
    )
    all_targets: dict[str, set[str]] = {}
    active_families = set(observed["selection"]["active_families"])
    for row in forensics["rows"]:
        state_ref = row["exact_state_ref"]
        if row["teacher_family"] not in active_families or (
            state_ref["shard_sha256"],
            state_ref["record_index"],
        ) in excluded_trace_ids:
            continue
        all_targets.setdefault(row["source_state_sha256"], set()).add(row["teacher_successor_key"])
    unique_state_union: set[str] = set()
    for family in observed["selection"]["active_families"]:
        selected = selected_source_rows(
            loaded,
            forensics,
            family=family,
            panel_kind=EDITING_T1_UNIQUE_PANEL_KIND,
        )
        assert len(selected) == 64
        family_states = {row["source_state_sha256"] for row in selected}
        assert len(family_states) == 64
        assert unique_state_union.isdisjoint(family_states)
        unique_state_union.update(family_states)
        assert all(len(all_targets[row["source_state_sha256"]]) == 1 for row in selected)
    assert len(unique_state_union) == 512


def test_repeated_panel_is_empirical_and_never_synthesized(frozen_inputs):
    forensics, kwargs = frozen_inputs
    panel = build_editing_t1_panel(**kwargs)

    global_repeated = selected_source_rows(
        panel,
        forensics,
        family=EDITING_T1_GLOBAL_FAMILY_SELECTOR,
        panel_kind=EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
    )
    assert len(global_repeated) == 173
    global_by_source: dict[str, set[str]] = {}
    global_families_by_source: dict[str, set[str]] = {}
    for row in global_repeated:
        global_by_source.setdefault(row["source_state_sha256"], set()).add(
            row["teacher_successor_key"]
        )
        global_families_by_source.setdefault(
            row["source_state_sha256"],
            set(),
        ).add(row["teacher_family"])
    assert len(global_by_source) == 81
    assert all(len(targets) > 1 for targets in global_by_source.values())
    assert sum(len(families) > 1 for families in global_families_by_source.values()) == 60
    with pytest.raises(ValueError, match="requires family='all_families'"):
        selected_source_rows(
            panel,
            forensics,
            family="cycle_attach",
            panel_kind=EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
        )

    cycle_open = selected_source_rows(
        panel,
        forensics,
        family="cycle_attach",
        panel_kind=EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
    )
    assert len(cycle_open) == 28
    by_source: dict[str, set[str]] = {}
    for row in cycle_open:
        by_source.setdefault(row["source_state_sha256"], set()).add(row["teacher_successor_key"])
    assert len(by_source) == 14
    assert all(len(targets) == 2 for targets in by_source.values())
    unique_cycle_open = selected_source_rows(
        panel,
        forensics,
        family="cycle_attach",
        panel_kind=EDITING_T1_UNIQUE_PANEL_KIND,
    )
    assert not ({row["source_state_sha256"] for row in unique_cycle_open} & set(global_by_source))

    with pytest.raises(EditingT1PanelError, match="no observed repeated"):
        selected_source_rows(
            panel,
            forensics,
            family="atom_delete",
            panel_kind=EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
        )


def test_panel_build_rejects_active8_for_another_gate_zero_unified_manifest(
    frozen_inputs,
):
    _forensics, kwargs = frozen_inputs
    mismatched = {
        **kwargs,
        "active8_admission": replace(
            kwargs["active8_admission"],
            unified_packed_manifest_sha256="0" * 64,
        ),
    }
    with pytest.raises(
        EditingT1PanelError,
        match="name different unified packed manifests",
    ):
        build_editing_t1_panel(**mismatched)


def test_t1_panel_rejects_row_reference_and_threshold_tampering(
    frozen_inputs,
):
    _forensics, kwargs = frozen_inputs
    panel = build_editing_t1_panel(**kwargs)

    tampered = copy.deepcopy(panel)
    tampered["unique_state_panels"]["atom_insert"][0]["source_row_sha256"] = "0" * 64
    body = {key: value for key, value in tampered.items() if key != "artifact_sha256"}
    tampered["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(EditingT1PanelError, match="does not resolve"):
        validate_editing_t1_panel(tampered, **kwargs)

    thresholded = copy.deepcopy(panel)
    thresholded["thresholds"]["minimum_unique_state_teacher_successor_top1"] = 0.9
    threshold_body = {key: value for key, value in thresholded.items() if key != "artifact_sha256"}
    thresholded["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            threshold_body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(EditingT1PanelError, match="must remain"):
        validate_editing_t1_panel(thresholded, **kwargs)


def test_charge_policy_exclusion_index_is_self_hashed_and_exactly_addressed():
    body = {
        "schema": "compose.data.packed_charge_policy_exclusions",
        "schema_version": 1,
        "selection": ("exclude_whole_trace_if_any_consecutive_state_pair_violates"),
        "source_input_inventory_sha256": "1" * 64,
        "charge_policy_audit_implementation_sha256": "2" * 64,
        "excluded_unique_trace_addresses": 1,
        "violation_type_bit_order": [
            "formal_charge_array_changed",
            "charged_center_element_changed",
        ],
        "shards": [
            {
                "manifest_layer": "mmp_analogue",
                "envelope_layer": "mmp_analogue",
                "partition": "validation",
                "relative_path": "mmp/validation/shard_0000.jsonl.gz",
                "packed_shard_content_sha256": "3" * 64,
                "excluded_entry_count": 1,
                "violating_entry_indices": [7],
                "entries": [
                    {
                        "entry_index": 7,
                        "trace_id": "trace-7",
                        "violation_type_mask": 1,
                    }
                ],
            }
        ],
    }
    payload = {
        **body,
        "payload_sha256": hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }
    audit = {
        "schema": "compose.data.packed_charge_policy_audit",
        "schema_version": 1,
        "input_inventory_sha256": "1" * 64,
        "implementation": {"implementation_sha256": "2" * 64},
        "counts": {"traces_with_violations": 1},
        "exclusion_payload": payload,
    }

    index = validate_charge_policy_exclusions(
        audit=audit,
        audit_file_sha256="4" * 64,
        exclusions=payload,
        exclusions_file_sha256="5" * 64,
    )

    assert index == {("3" * 64, 7): "trace-7"}

    tampered = copy.deepcopy(payload)
    tampered["shards"][0]["entries"][0]["trace_id"] = "other"
    with pytest.raises(EditingT1PanelError):
        validate_charge_policy_exclusions(
            audit=audit,
            audit_file_sha256="4" * 64,
            exclusions=tampered,
            exclusions_file_sha256="5" * 64,
        )


def test_disallowed_middle_action_removes_neighboring_rows_before_t1_selection():
    state = pad_molecular_graph(smiles_to_molecular_graph("CC"), 8)
    steps = (
        RewriteStep("atom_delete", AtomDelete(1)),
        RewriteStep(
            "ring_system_delete",
            RingSystemDelete(
                system_atoms=(0, 1),
                retained_system_atoms=(0,),
                bond_deletions=(),
                atom_deletions=(1,),
                atom_payloads=(),
                bond_reorders=(),
                source_aromatic_edges=(),
                aromatic_edges=(),
                topology_class="fixture",
            ),
        ),
        RewriteStep("atom_delete", AtomDelete(0)),
    )
    trace = RewriteTrace(
        source=state,
        target=state,
        steps=steps,
        metadata={"fixture": "T1 whole-trace admission"},
    )
    address = PackedTraceAddress(
        packed_shard_content_sha256="a" * 64,
        packed_shard_name="fixture.jsonl.gz",
        entry_index=0,
        trace_id="middle-disallowed",
        layer="corruption",
        partition="validation",
        source_key="CC",
        target_key="CC",
        path_length=len(steps),
    )
    addressed = AddressedPackedTrace(
        address=address,
        trace=trace,
        path=PackedTraceProgress(
            trace,
            tuple(state for _ in range(len(steps) + 1)),
        ),
    )

    decision = inventory_record_for_trace(
        addressed,
        exact_candidate_checker=lambda *_: ExactCandidateEvidence(
            supported=True,
            action_sha256="b" * 64,
        ),
    )
    assert decision["decision"] == "excluded"
    assert decision["progress_rows"] == []
    assert decision["exclusions"][0]["step_index"] == 1

    admission = Active8TraceAdmission(
        manifest_path=Path("/fixture/ACTIVE8_TRACE_INVENTORY.json"),
        manifest_file_sha256="1" * 64,
        inventory_sha256="2" * 64,
        unified_packed_manifest_sha256="3" * 64,
        support_contract_sha256="4" * 64,
        effective_source_corpus_cache_sha256="5" * 64,
        decisions_by_digest=MappingProxyType(
            {"a" * 64: (("middle-disallowed", False),)}
        ),
        shard_digest_by_lane=MappingProxyType(
            {
                (
                    "corruption",
                    "validation",
                    "fixture.jsonl.gz",
                ): "a" * 64
            }
        ),
        shard_metadata_by_digest=MappingProxyType(
            {"a" * 64: MappingProxyType({})}
        ),
        counts=MappingProxyType({}),
    )
    neighboring_rows = (
        {
            "record_key": (
                "corruption/validation/fixture.jsonl.gz:0:middle-disallowed"
            ),
            "partition": "validation",
            "exact_state_ref": {
                "shard_sha256": "a" * 64,
                "record_index": 0,
                "progress_index": 0,
                "shard_name": "corruption/validation/fixture.jsonl.gz",
            },
        },
        {
            "record_key": (
                "corruption/validation/fixture.jsonl.gz:0:middle-disallowed"
            ),
            "partition": "validation",
            "exact_state_ref": {
                "shard_sha256": "a" * 64,
                "record_index": 0,
                "progress_index": 2,
                "shard_name": "corruption/validation/fixture.jsonl.gz",
            },
        },
    )

    accepted_rows, excluded_rows, excluded_traces = (
        filter_active8_forensics_rows(neighboring_rows, admission)
    )
    assert accepted_rows == ()
    assert excluded_rows == 2
    assert excluded_traces == 1


def test_shared_t1_family_scope_guard_defines_global_and_local_semantics():
    assert EDITING_T1_LOCAL_ADAPTER_FAMILIES == (
        "bond_reorder",
        "cycle_insert",
        "cycle_attach",
        "ring_system_restate",
    )
    assert (
        require_editing_t1_family_scope_applicable(
            EDITING_T1_GLOBAL_FAMILY_SELECTOR,
            "heads_plus_local_adapter",
        )
        == RINGCORE_EDITING_FAMILIES
    )
    assert require_editing_t1_family_scope_applicable(
        "atom_insert",
        "heads_only",
    ) == ("atom_insert",)
    with pytest.raises(
        EditingT1RuntimeError,
        match="changes no parameters beyond heads_only",
    ):
        require_editing_t1_family_scope_applicable(
            "atom_insert",
            "heads_plus_local_adapter",
        )


def test_direct_t1_runtime_rejects_duplicate_local_adapter_arm_before_inputs():
    with pytest.raises(
        EditingT1RuntimeError,
        match="changes no parameters beyond heads_only",
    ):
        run_editing_t1_arm(
            contract=None,
            gate_zero_contract=None,
            source=None,
            panel={},
            forensics={},
            family="atom_insert",
            panel_kind=EDITING_T1_UNIQUE_PANEL_KIND,
            scope="heads_plus_local_adapter",
            device=torch.device("cpu"),
            excluded_trace_ids={},
            active8_admission=None,
        )


def test_direct_t1_cli_rejects_duplicate_local_adapter_arm(
    monkeypatch,
    tmp_path,
):
    from scripts import run_editing_t1_successor_gate as t1_cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_editing_t1_successor_gate.py",
            "--transfer-root",
            str(tmp_path),
            "--active8-inventory",
            str(tmp_path / "ACTIVE8_TRACE_INVENTORY.json"),
            "--active8-inventory-file-sha256",
            "a" * 64,
            "--family",
            "atom_insert",
            "--scope",
            "heads_plus_local_adapter",
        ],
    )
    with pytest.raises(
        SystemExit,
        match="changes no parameters beyond heads_only",
    ):
        t1_cli.main()


def test_t1_contract_and_ring_family_scope_ladder_are_explicit():
    contract_path = ROOT / "configs" / "editing_t1_successor_gate_v3.json"
    with pytest.raises(EditingT1RuntimeError):
        load_editing_t1_runtime_contract(contract_path)
    payload = json.loads(contract_path.read_text())
    payload.update(
        {
            "active8_inventory_manifest_file_sha256": "1" * 64,
            "active8_inventory_sha256": "2" * 64,
            "active8_effective_source_corpus_cache_sha256": "5" * 64,
            "active8_unified_packed_manifest_sha256": "3" * 64,
            "active8_support_contract_sha256": payload[
                "gate_zero_runtime_contract_sha256"
            ],
            "implementation_sha256": editing_t1_implementation_sha256(),
        }
    )
    payload["optimization"] = {
        **payload["optimization"],
        "scopes": [
            "heads_only",
            "heads_plus_local_adapter",
            "all",
        ],
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "contract_sha256"
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    contract = EditingT1RuntimeContract(payload)
    assert contract.training_authorized is False
    assert contract.optimization["steps"] == 500
    assert tuple(contract.optimization["scopes"]) == (
        "heads_only",
        "heads_plus_local_adapter",
        "all",
    )
    assert tuple(contract.payload["panel_kinds"]) == (
        EDITING_T1_UNIQUE_PANEL_KIND,
        EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
        EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
    )
    assert contract.payload["implementation_sha256"] == editing_t1_implementation_sha256()
    implementation_sources = set(editing_t1_implementation_sources())
    assert {
        "modal_apps/run_editing_t1_successor_gate.py",
        "scripts/build_editing_t1_successor_panel.py",
        "scripts/run_editing_t1_successor_gate.py",
        "src/compose_v4/experiments/editing_gate_zero_runtime.py",
        "src/compose_v4/experiments/editing_step_zero_probe.py",
        "src/compose_v4/experiments/factorized_mark_conditional.py",
        "src/compose_v4/experiments/production_successor_kernel.py",
        "src/compose_v4/experiments/successor_fiber_cache_builder.py",
        "src/compose_v4/experiments/successor_micro_overfit.py",
        "src/compose_v4/data/successor_fiber_cache.py",
        "src/compose_v4/model/factorized_tracelet_rate_model.py",
    } <= implementation_sources

    torch.manual_seed(7)
    model = FactorizedTraceletRateModel(
        build_typed_ring_catalog(()),
        hidden_dim=12,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    )
    selected = configure_micro_overfit_parameters(
        model,
        ("ring_system_restate",),
        scope="heads_plus_local_adapter",
    )
    assert any(name.startswith("ring_restate_head.") for name in selected)
    assert any(name.startswith("pair_project.") for name in selected)
    assert any(name.startswith("restate_order_embedding.") for name in selected)
    assert any(name.startswith("family_head.") for name in selected)
    with pytest.raises(
        SuccessorMicroOverfitError,
        match="changes no parameters beyond heads_only",
    ):
        configure_micro_overfit_parameters(
            model,
            ("atom_insert",),
            scope="heads_plus_local_adapter",
        )


def _runtime_fixture():
    records, attempted = build_cycle_op_records(
        ("CC1CCCCC1O",),
        n_slots=12,
        seed=4,
        max_bonds_per_molecule=1,
    )
    assert attempted > 0 and records
    original = records[0]
    path = original.path
    step = path.trace.steps[0]
    family = canonical_family(step.rule_name)
    shard_digest = "1" * 64
    address = PackedTraceAddress(
        packed_shard_content_sha256=shard_digest,
        packed_shard_name="shard_0000.jsonl.gz",
        entry_index=0,
        trace_id="fixture-trace",
        layer="cycle_ops",
        partition="validation",
        source_key=canonical_state_key(path.trace.source),
        target_key=canonical_state_key(path.trace.target),
        path_length=path.path_length,
    )
    record = PathRecord(
        target_key=address.target_key,
        path=path,
        corpus_address=address,
    )
    state = path.state_at(0)
    target = path.state_at(1)
    cell = "fixture-cell"
    sidecar = LoadedSemanticCellSidecar(
        rows=(
            SemanticSidecarRow(
                packed_shard_content_sha256=shard_digest,
                entry_index=0,
                progress_index=0,
                semantic_cell_id=cell,
            ),
        ),
        manifest={},
    )
    spec = SemanticSidecarSourceShard(
        packed_shard_content_sha256=shard_digest,
        packed_shard_name=address.packed_shard_name,
        layer="cycle_ops",
        partition="validation",
        packed_manifest_sha256="2" * 64,
        provenance_overlay_sha256="3" * 64,
        packed_entry_count=1,
        effective_trace_count=1,
        progress_row_count=path.path_length + 1,
    )
    source = FrozenValidationSource(
        records=(record,),
        sidecar=sidecar,
        bindings=(
            ValidationShardBinding(
                spec=spec,
                packed_path=Path("/fixture/shard"),
                manifest_path=Path("/fixture/manifest"),
                overlay_path=Path("/fixture/overlay"),
            ),
        ),
        unified_packed_manifest_sha256="4" * 64,
        representability_overlay_sha256="5" * 64,
    )
    row = {
        "row_index": 0,
        "row_sha256": "6" * 64,
        "exact_state_ref": {
            "shard_sha256": shard_digest,
            "record_index": 0,
            "progress_index": 0,
            "shard_name": f"cycle_ops/validation/{address.packed_shard_name}",
        },
        "progress_index": 0,
        "path_length": path.path_length,
        "semantic_cell_id": cell,
        "source_state_sha256": persistent_slot_state_sha256(state),
        "source_state_key": canonical_state_key(state),
        "teacher_successor_key": canonical_state_key(target),
        "teacher_rule_name": step.rule_name,
        "teacher_family": family,
        "teacher_action_sha256": rewrite_action_codec_sha256(
            step.rule_name,
            step.action,
        ),
        "importance_weight": 1.0,
    }
    panel = {
        "artifact_sha256": "7" * 64,
        "unique_state_panels": {
            family: [
                {
                    "source_row_index": 0,
                    "source_row_sha256": row["row_sha256"],
                }
            ]
        },
        "global_repeated_state_panel": [],
        "within_family_repeated_state_panels": {family: []},
    }
    return source, {"rows": [row]}, panel, family


def test_t1_runtime_fails_on_gate_zero_active8_identity_mismatch():
    source, _forensics, _panel, _family = _runtime_fixture()
    admission = _active8_admission_for_record(
        source.records[0],
        accepted=True,
        support_contract_sha256="4" * 64,
        unified_packed_manifest_sha256=(
            source.unified_packed_manifest_sha256
        ),
    )
    identity = dict(active8_t1_identity(admission))
    panel = {"source": dict(identity)}
    contract = SimpleNamespace(payload=dict(identity))

    observed = validate_t1_active8_runtime_binding(
        contract=contract,
        gate_zero_contract=SimpleNamespace(sha256="4" * 64),
        source=source,
        panel=panel,
        active8_admission=admission,
    )
    assert dict(observed) == identity

    with pytest.raises(
        EditingT1RuntimeError,
        match="Gate0 runtime contract and Active8 support contract disagree",
    ):
        validate_t1_active8_runtime_binding(
            contract=contract,
            gate_zero_contract=SimpleNamespace(sha256="6" * 64),
            source=source,
            panel=panel,
            active8_admission=admission,
        )


def test_runtime_fixture_round_trips_exact_production_cache():
    source, forensics, panel, family = _runtime_fixture()
    active8_admission = _active8_admission_for_record(
        source.records[0],
        accepted=True,
    )
    torch.manual_seed(7)
    model = FactorizedTraceletRateModel(
        build_typed_ring_catalog(()),
        hidden_dim=12,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    )
    materialized = materialize_t1_panel(
        model,
        source=source,
        panel=panel,
        forensics=forensics,
        family=family,
        panel_kind=EDITING_T1_UNIQUE_PANEL_KIND,
        max_atoms=12,
        excluded_trace_ids={},
        active8_admission=active8_admission,
    )

    assert len(materialized.examples) == 1
    assert materialized.unique_progress_address_count == 1
    assert len(materialized.cache_receipts) == 1
    assert materialized.cache_receipts[0].record_count == (source.records[0].path.path_length + 1)
    assert materialized.prepared.fibers[0].target_key == (materialized.examples[0].target_key)

    exact_trace_key = (
        source.records[0].corpus_address.packed_shard_content_sha256,
        source.records[0].corpus_address.entry_index,
    )
    with pytest.raises(
        Exception,
        match="selected a charge-policy-excluded",
    ):
        materialize_t1_panel(
            model,
            source=source,
            panel=panel,
            forensics=forensics,
            family=family,
            panel_kind=EDITING_T1_UNIQUE_PANEL_KIND,
            max_atoms=12,
            excluded_trace_ids={exact_trace_key: source.records[0].corpus_address.trace_id},
            active8_admission=active8_admission,
        )
    with pytest.raises(Exception, match="trace_id disagrees"):
        materialize_t1_panel(
            model,
            source=source,
            panel=panel,
            forensics=forensics,
            family=family,
            panel_kind=EDITING_T1_UNIQUE_PANEL_KIND,
            max_atoms=12,
            excluded_trace_ids={exact_trace_key: "wrong-trace-id"},
            active8_admission=active8_admission,
        )

    repeated_panel = copy.deepcopy(panel)
    row_reference = repeated_panel["unique_state_panels"][family][0]
    repeated_panel["global_repeated_state_panel"] = [
        {
            "source_state_sha256": forensics["rows"][0]["source_state_sha256"],
            "distinct_teacher_successor_count": 2,
            "distinct_teacher_family_count": 1,
            "rows": [row_reference, row_reference],
        }
    ]
    repeated = materialize_t1_panel(
        model,
        source=source,
        panel=repeated_panel,
        forensics=forensics,
        family=EDITING_T1_GLOBAL_FAMILY_SELECTOR,
        panel_kind=EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
        max_atoms=12,
        excluded_trace_ids={},
        active8_admission=active8_admission,
    )
    assert len(repeated.examples) == 2
    assert repeated.unique_progress_address_count == 1
    assert repeated.cache_receipts[0].record_count == (source.records[0].path.path_length + 1)

    tampered = copy.deepcopy(forensics)
    tampered["rows"][0]["teacher_action_sha256"] = "0" * 64
    with pytest.raises(
        Exception,
        match="teacher/source/target identity",
    ):
        materialize_t1_panel(
            model,
            source=source,
            panel=panel,
            forensics=tampered,
            family=family,
            panel_kind=EDITING_T1_UNIQUE_PANEL_KIND,
            max_atoms=12,
            excluded_trace_ids={},
            active8_admission=active8_admission,
        )

    with pytest.raises(
        EditingT1RuntimeError,
        match="whole-trace Active8-excluded",
    ):
        materialize_t1_panel(
            model,
            source=source,
            panel=panel,
            forensics=forensics,
            family=family,
            panel_kind=EDITING_T1_UNIQUE_PANEL_KIND,
            max_atoms=12,
            excluded_trace_ids={},
            active8_admission=_active8_admission_for_record(
                source.records[0],
                accepted=False,
            ),
        )


def _durable_result_fixture():
    receipt = T1CacheShardReceipt(
        packed_shard_content_sha256="1" * 64,
        cache_content_sha256="2" * 64,
        encoded_sha256="3" * 64,
        record_count=2,
    )
    body = {
        "schema": EDITING_T1_RESULT_SCHEMA,
        "schema_version": EDITING_T1_RESULT_VERSION,
        "status": EDITING_T1_RESULT_STATUS,
        "training_authorized": False,
        "gate_decision": None,
        "numeric_thresholds_frozen": False,
        "contract_sha256": "4" * 64,
        "gate_zero_runtime_contract_sha256": "5" * 64,
        "panel_artifact_sha256": "6" * 64,
        "charge_policy_exclusion_payload_sha256": "7" * 64,
        "charge_policy_source_input_inventory_sha256": "8" * 64,
        "active8_inventory_manifest_file_sha256": "d" * 64,
        "active8_inventory_sha256": "e" * 64,
        "active8_effective_source_corpus_cache_sha256": "f" * 64,
        "active8_unified_packed_manifest_sha256": "0" * 64,
        "active8_support_contract_sha256": "5" * 64,
        "family": "atom_insert",
        "family_status": "CURRENT_ACTIVE_FAMILY",
        "panel_kind": EDITING_T1_UNIQUE_PANEL_KIND,
        "panel_role": "DETERMINISTIC_FAMILY_CAPACITY_DIAGNOSTIC",
        "scope": "heads_only",
        "operator_identity": {
            "operator_capability_fingerprint": "fixture-capability",
            "enable_ring_system_delete": False,
            "enable_ring_grow_macro": False,
            "enable_cycle_ops": True,
        },
        "example_count": 1,
        "unique_progress_address_count": 1,
        "repeated_progress_observation_count": 0,
        "source_row_sha256s": ["9" * 64],
        "scratch_initialization": {
            "regime": "scratch",
            "transfer_plan_sha256": "a" * 64,
            "initial_model_state_sha256": "b" * 64,
        },
        "cache_receipts": [receipt.__dict__],
        "training_report": {
            "scope": "heads_only",
            "steps": 1,
        },
        "final_model_state_sha256": "c" * 64,
    }
    result = {
        **body,
        "result_sha256": hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }
    return result, receipt


def test_t1_durable_result_and_cache_receipts_fail_closed(tmp_path):
    result, receipt = _durable_result_fixture()
    expected = {
        "expected_result_sha256": result["result_sha256"],
        "expected_contract_sha256": result["contract_sha256"],
        "expected_panel_artifact_sha256": result["panel_artifact_sha256"],
        "expected_active8_identity": {
            "active8_inventory_manifest_file_sha256": (
                result["active8_inventory_manifest_file_sha256"]
            ),
            "active8_inventory_sha256": result["active8_inventory_sha256"],
            "active8_effective_source_corpus_cache_sha256": (
                result["active8_effective_source_corpus_cache_sha256"]
            ),
            "active8_unified_packed_manifest_sha256": (
                result["active8_unified_packed_manifest_sha256"]
            ),
            "active8_support_contract_sha256": (
                result["active8_support_contract_sha256"]
            ),
        },
        "expected_cache_receipts": (receipt,),
    }
    observed = validate_editing_t1_result(result, **expected)
    assert observed["training_authorized"] is False
    assert validate_t1_cache_shard_receipt(receipt.__dict__, expected=receipt) == receipt

    result_path = tmp_path / "t1-result.json"
    result_path.write_text(json.dumps(result))
    loaded = load_editing_t1_result(result_path, **expected)
    assert loaded["result_sha256"] == result["result_sha256"]

    with pytest.raises(EditingT1RuntimeError, match="another runtime contract"):
        validate_editing_t1_result(
            result,
            **{
                **expected,
                "expected_contract_sha256": "d" * 64,
            },
        )
    with pytest.raises(EditingT1RuntimeError, match="another panel artifact"):
        validate_editing_t1_result(
            result,
            **{
                **expected,
                "expected_panel_artifact_sha256": "e" * 64,
            },
        )
    with pytest.raises(
        EditingT1RuntimeError,
        match="another physical or logical Active8 corpus",
    ):
        validate_editing_t1_result(
            result,
            **{
                **expected,
                "expected_active8_identity": {
                    **expected["expected_active8_identity"],
                    "active8_inventory_sha256": "a" * 64,
                },
            },
        )

    schema_tamper = copy.deepcopy(result)
    schema_tamper["schema"] = "compose.editing.other_result"
    schema_body = {key: value for key, value in schema_tamper.items() if key != "result_sha256"}
    schema_tamper["result_sha256"] = hashlib.sha256(
        json.dumps(
            schema_body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(EditingT1RuntimeError, match="identity, status, or self-hash"):
        validate_editing_t1_result(
            schema_tamper,
            **{
                **expected,
                "expected_result_sha256": schema_tamper["result_sha256"],
            },
        )

    self_hash_tamper = copy.deepcopy(result)
    self_hash_tamper["cache_receipts"][0]["cache_content_sha256"] = "d" * 64
    with pytest.raises(EditingT1RuntimeError, match="self-hash"):
        validate_editing_t1_result(self_hash_tamper, **expected)

    retained_identity_tamper = copy.deepcopy(result)
    retained_identity_tamper["cache_receipts"][0]["cache_content_sha256"] = "d" * 64
    retained_body = {
        key: value for key, value in retained_identity_tamper.items() if key != "result_sha256"
    }
    retained_identity_tamper["result_sha256"] = hashlib.sha256(
        json.dumps(
            retained_body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(EditingT1RuntimeError, match="cache receipts disagree"):
        validate_editing_t1_result(
            retained_identity_tamper,
            **{
                **expected,
                "expected_result_sha256": retained_identity_tamper["result_sha256"],
            },
        )

    count_tamper = copy.deepcopy(result)
    count_tamper["cache_receipts"][0]["record_count"] = 3
    count_body = {key: value for key, value in count_tamper.items() if key != "result_sha256"}
    count_tamper["result_sha256"] = hashlib.sha256(
        json.dumps(
            count_body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(EditingT1RuntimeError, match="cache receipts disagree"):
        validate_editing_t1_result(
            count_tamper,
            **{
                **expected,
                "expected_result_sha256": count_tamper["result_sha256"],
            },
        )

    malformed_receipt = copy.deepcopy(receipt.__dict__)
    malformed_receipt["encoded_sha256"] = "not-a-sha"
    with pytest.raises(EditingT1RuntimeError, match="encoded_sha256"):
        validate_t1_cache_shard_receipt(malformed_receipt)
