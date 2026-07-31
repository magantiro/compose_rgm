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
    EDITING_T1_CAPACITY_CENSUS_SCHEMA,
    EDITING_T1_CAPACITY_CENSUS_SCHEMA_VERSION,
    EDITING_T1_CAPACITY_CENSUS_STATUS,
    EDITING_T1_GLOBAL_FAMILY_SELECTOR,
    EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
    EDITING_T1_PANEL_SCHEMA,
    EDITING_T1_PANEL_SCHEMA_VERSION,
    EDITING_T1_PANEL_STATUS,
    EDITING_T1_UNIQUE_PANEL_KIND,
    EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
    EditingT1PanelError,
    active8_t1_identity,
    build_editing_t1_capacity_census,
    build_editing_t1_panel,
    filter_active8_forensics_rows,
    load_editing_t1_panel,
    selected_source_rows,
    validate_charge_policy_exclusions,
    validate_editing_t1_panel,
    validate_editing_t1_capacity_census,
)
import compose_v4.experiments.editing_t1_panel as editing_t1_panel_module
from compose_v4.experiments.editing_t1_successor_runtime import (
    EDITING_T1_LOCAL_ADAPTER_FAMILIES,
    EDITING_T1_RESULT_SCHEMA,
    EDITING_T1_RESULT_STATUS,
    EDITING_T1_RESULT_VERSION,
    EDITING_T1_RUNTIME_CONTRACT_VERSION,
    EDITING_T1_V4_CAPACITY_CENSUS_RELATIVE_PATH,
    EDITING_T1_V4_PANEL_RELATIVE_PATH,
    EDITING_T1_V8_CONTRACT_RELATIVE_PATH,
    EditingT1RuntimeContract,
    EditingT1RuntimeError,
    T1CacheShardReceipt,
    build_editing_t1_runtime_contract,
    editing_t1_implementation_sha256,
    editing_t1_implementation_sources,
    editing_t1_numeric_thresholds_sha256,
    load_editing_t1_launch_authority,
    load_editing_t1_result,
    load_editing_t1_runtime_contract,
    materialize_t1_panel,
    require_editing_t1_family_scope_applicable,
    run_editing_t1_arm,
    validate_editing_t1_result,
    validate_editing_t1_launch_authority,
    validate_t1_active8_runtime_binding,
    validate_t1_cache_shard_receipt,
    write_editing_t1_runtime_contract,
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
    ROOT / "diagnostics" / "coherence" / "editing_t1_successor_panel_v3_active8_2026-07-30.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _synthetic_t1_launch_authority():
    active8_identity = {
        "active8_inventory_manifest_file_sha256": "1" * 64,
        "active8_inventory_sha256": "2" * 64,
        "active8_effective_source_corpus_cache_sha256": "3" * 64,
        "active8_unified_packed_manifest_sha256": "4" * 64,
        "active8_support_contract_sha256": "5" * 64,
    }
    capacity_body = {
        "schema": EDITING_T1_CAPACITY_CENSUS_SCHEMA,
        "schema_version": EDITING_T1_CAPACITY_CENSUS_SCHEMA_VERSION,
        "status": EDITING_T1_CAPACITY_CENSUS_STATUS,
        "training_authorized": False,
        "forensics_file_sha256": "6" * 64,
        "forensics_artifact_sha256": "7" * 64,
        "gate_zero_runtime_contract_sha256": "5" * 64,
        "unified_packed_manifest_sha256": "4" * 64,
        "representability_overlay_sha256": "8" * 64,
        **active8_identity,
        "rows": [{"source_row_sha256": "9" * 64}],
    }
    capacity_census = {
        **capacity_body,
        "census_sha256": _stable_sha256(capacity_body),
    }
    capacity_content = (
        json.dumps(
            capacity_census,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    capacity_file_sha256 = hashlib.sha256(capacity_content).hexdigest()
    repeated_families = {
        "atom_insert",
        "atom_restate",
        "bond_reroute",
        "cycle_attach",
        "ring_system_restate",
    }
    selection = {
        "required_high_candidate_families": list(RINGCORE_EDITING_FAMILIES),
        "required_aliased_teacher_families": [
            "atom_insert",
            "bond_reroute",
            "cycle_attach",
        ],
    }
    panel_census = {
        "unique_high_candidate_stratum_available_by_family": {
            family: 1 for family in RINGCORE_EDITING_FAMILIES
        },
        "unique_high_candidate_stratum_selected_by_family": {
            family: 1 for family in RINGCORE_EDITING_FAMILIES
        },
        "unique_aliased_teacher_stratum_available_by_family": {
            family: int(family in selection["required_aliased_teacher_families"])
            for family in RINGCORE_EDITING_FAMILIES
        },
        "unique_aliased_teacher_stratum_selected_by_family": {
            family: int(family in selection["required_aliased_teacher_families"])
            for family in RINGCORE_EDITING_FAMILIES
        },
        "registered_high_candidate_families_all_covered": True,
        "registered_aliased_teacher_families_all_covered": True,
        "families_without_observed_within_family_multitarget_repeats": [
            family for family in RINGCORE_EDITING_FAMILIES if family not in repeated_families
        ],
    }
    panel_body = {
        "schema": EDITING_T1_PANEL_SCHEMA,
        "schema_version": EDITING_T1_PANEL_SCHEMA_VERSION,
        "status": EDITING_T1_PANEL_STATUS,
        "training_authorized": False,
        "source": {
            "forensics_file_sha256": "6" * 64,
            "charge_policy_audit_file_sha256": "a" * 64,
            "charge_policy_exclusions_file_sha256": "b" * 64,
            "charge_policy_exclusion_payload_sha256": "c" * 64,
            "charge_policy_source_input_inventory_sha256": "d" * 64,
            "capacity_census_file_sha256": capacity_file_sha256,
            "capacity_census_sha256": capacity_census["census_sha256"],
            **active8_identity,
        },
        "selection": selection,
        "census": panel_census,
        "thresholds": {
            "minimum_unique_state_teacher_successor_top1": None,
            "minimum_unique_state_teacher_successor_probability": None,
            "maximum_unique_state_teacher_successor_nll": None,
            "maximum_global_repeated_state_excess_nll_over_empirical_entropy": None,
        },
        "within_family_repeated_state_panels": {
            family: ([{"rows": [{"source_row_index": 0}]}] if family in repeated_families else [])
            for family in RINGCORE_EDITING_FAMILIES
        },
    }
    panel = {
        **panel_body,
        "artifact_sha256": _stable_sha256(panel_body),
    }
    optimization = {
        "steps": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "scopes": [
            "heads_only",
            "heads_plus_local_adapter",
            "all",
        ],
        "report_points": [1, 2],
    }
    thresholds = {
        "minimum_unique_state_teacher_successor_top1": 0.8,
        "minimum_unique_state_teacher_successor_probability": 0.6,
        "maximum_unique_state_teacher_successor_nll": 0.7,
        "maximum_global_repeated_state_excess_nll_over_empirical_entropy": 0.2,
    }
    contract = build_editing_t1_runtime_contract(
        gate_zero_runtime_contract_sha256="5" * 64,
        panel=panel,
        capacity_census=capacity_census,
        capacity_census_file_sha256=capacity_file_sha256,
        active8_identity=active8_identity,
        optimization=optimization,
        thresholds=thresholds,
    )
    authority = validate_editing_t1_launch_authority(
        contract=contract,
        panel=panel,
        capacity_census=capacity_census,
        capacity_census_file_sha256=capacity_file_sha256,
        expected_active8_inventory_manifest_file_sha256=(
            active8_identity["active8_inventory_manifest_file_sha256"]
        ),
    )
    return authority, capacity_content, optimization, thresholds


def _capacity_census(
    forensics: dict,
    admission: Active8TraceAdmission,
) -> dict[str, object]:
    admitted, _excluded_rows, _excluded_traces = filter_active8_forensics_rows(
        forensics["rows"], admission
    )
    rows = []
    for row in admitted:
        raw_count = 64 + int(row["row_index"]) % 97
        rows.append(
            {
                "source_row_sha256": row["row_sha256"],
                "raw_legal_mark_count": raw_count,
                "canonical_successor_count": raw_count - 7,
                # Canonical fibers may contain persistent-slot aliases whose
                # exact successor tensors differ. T1 must cover quotient-level
                # aliasing rather than requiring duplicate exact tensors.
                "teacher_exact_successor_alias_count": 1,
                "teacher_canonical_successor_alias_count": 2,
            }
        )
    body = {
        "schema": EDITING_T1_CAPACITY_CENSUS_SCHEMA,
        "schema_version": EDITING_T1_CAPACITY_CENSUS_SCHEMA_VERSION,
        "status": EDITING_T1_CAPACITY_CENSUS_STATUS,
        "forensics_artifact_sha256": forensics["artifact_sha256"],
        **active8_t1_identity(admission),
        "rows": rows,
    }
    return {**body, "census_sha256": _stable_sha256(body)}


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
            {address.packed_shard_content_sha256: ((address.trace_id, accepted),)}
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


@pytest.fixture
def frozen_inputs(monkeypatch):
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
    admission = _active8_admission_for_forensics(
        forensics,
        unified_packed_manifest_sha256=(unified_packed_manifest_sha256),
    )
    capacity_census = _capacity_census(forensics, admission)
    capacity_index = {
        row["source_row_sha256"]: MappingProxyType(dict(row)) for row in capacity_census["rows"]
    }
    monkeypatch.setattr(
        editing_t1_panel_module,
        "validate_editing_t1_capacity_census",
        lambda *_args, **_kwargs: MappingProxyType(capacity_index),
    )
    kwargs = {
        "model": object(),
        "source": SimpleNamespace(),
        "forensics": forensics,
        "forensics_file_sha256": _sha256(FORENSICS),
        "sidecar": sidecar,
        "semantic_sidecar_file_sha256": _sha256(SIDECAR),
        "semantic_sidecar_manifest_file_sha256": _sha256(SIDECAR_MANIFEST),
        "charge_policy_audit": load_json_object(CHARGE_POLICY_AUDIT),
        "charge_policy_audit_file_sha256": _sha256(CHARGE_POLICY_AUDIT),
        "charge_policy_exclusions": load_json_object(CHARGE_POLICY_EXCLUSIONS),
        "charge_policy_exclusions_file_sha256": _sha256(CHARGE_POLICY_EXCLUSIONS),
        "capacity_census": capacity_census,
        "capacity_census_file_sha256": "a" * 64,
        "active8_admission": admission,
        "gate_zero_runtime_contract_sha256": admission.support_contract_sha256,
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
    with pytest.raises(
        EditingT1PanelError,
        match="schema, status, or self-hash",
    ):
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
    assert observed["census"]["registered_high_candidate_families_all_covered"] is True
    assert observed["census"]["registered_aliased_teacher_families_all_covered"] is True
    assert all(
        value >= 1
        for value in observed["census"]["unique_high_candidate_stratum_selected_by_family"].values()
    )
    assert all(
        value >= 1
        for value in observed["census"][
            "unique_aliased_teacher_stratum_selected_by_family"
        ].values()
    )
    assert observed["census"]["source_forensics_rows_total"] == 2304
    assert observed["census"]["active8_eligible_forensics_rows_total"] == 2304
    assert observed["census"]["active8_excluded_forensics_rows_total"] == 0
    assert observed["census"]["active8_excluded_unique_forensics_trace_addresses"] == 0
    assert observed["census"]["charge_policy_eligible_forensics_rows_total"] == 2292
    assert observed["census"]["charge_policy_excluded_forensics_rows_total"] == 12
    assert observed["census"]["charge_policy_excluded_unique_forensics_trace_addresses"] == 12
    assert observed["census"]["pilot_family_eligible_forensics_rows_total"] == 2036
    assert observed["census"]["operator_freeze_excluded_forensics_rows_total"] == 256
    assert observed["census"]["global_repeated_available_multitarget_groups_total"] == 81
    assert observed["census"]["global_repeated_available_rows_total"] == 173
    assert 64 <= observed["census"]["global_repeated_rows_total"] <= 128
    assert (
        observed["census"]["global_repeated_multitarget_groups_total"]
        <= observed["census"]["global_repeated_available_multitarget_groups_total"]
    )
    assert observed["census"]["global_repeated_cross_family_groups_total"] >= 1
    assert observed["census"]["global_repeated_cross_family_rows_total"] >= 2
    assert observed["census"]["global_repeated_panel_meets_64_example_minimum"] is True
    assert observed["census"]["global_repeated_panel_meets_128_example_maximum"] is True
    assert observed["census"]["within_family_repeated_multitarget_groups_total"] == 24
    assert observed["census"]["within_family_repeated_rows_total"] == 48
    assert (
        sum(
            observed["census"][
                "within_family_repeated_available_multitarget_groups_by_family"
            ].values()
        )
        == 24
    )
    assert sum(observed["census"]["within_family_repeated_available_rows_by_family"].values()) == 48
    assert observed["selection"]["operator_freeze_excluded_families"] == ["ring_system_delete"]
    assert "ring_system_delete" not in observed["selection"]["active_families"]
    assert observed["architecture"]["enable_ring_system_delete"] is False
    assert (
        observed["selection"]["conditional_repeated_state_relationship"]
        == "global and within-family panels are bounded deterministic views of the "
        "same frozen observation pool; they may overlap but must not be treated "
        "as statistically independent"
    )
    overlap = observed["census"]["panel_overlap"]
    for relationship in (
        "unique_state_vs_global_repeated",
        "unique_state_vs_within_family_repeated",
    ):
        assert overlap[relationship]["shared_source_states"] == 0
        assert overlap[relationship]["shared_progress_addresses"] == 0
        assert overlap[relationship]["shared_source_row_references"] == 0
    assert (
        overlap["global_repeated_vs_within_family_repeated"]["shared_source_row_references"]
        <= observed["census"]["within_family_repeated_rows_total"]
    )
    assert overlap["statistically_independent_panel_claim"] is False
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
        if (
            row["teacher_family"] not in active_families
            or (
                state_ref["shard_sha256"],
                state_ref["record_index"],
            )
            in excluded_trace_ids
        ):
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


def test_t1_panel_capacity_worker_count_does_not_change_artifact(
    frozen_inputs,
    monkeypatch,
):
    _forensics, kwargs = frozen_inputs
    validate_capacity = editing_t1_panel_module.validate_editing_t1_capacity_census
    observed_workers: list[int] = []

    def recording_validator(*args, workers=1, **validator_kwargs):
        observed_workers.append(workers)
        return validate_capacity(*args, workers=workers, **validator_kwargs)

    monkeypatch.setattr(
        editing_t1_panel_module,
        "validate_editing_t1_capacity_census",
        recording_validator,
    )
    serial = build_editing_t1_panel(**kwargs)
    parallel = build_editing_t1_panel(
        **kwargs,
        capacity_workers=2,
    )

    assert parallel == serial
    assert observed_workers == [1, 2]


def test_repeated_panel_is_empirical_and_never_synthesized(frozen_inputs):
    forensics, kwargs = frozen_inputs
    panel = build_editing_t1_panel(**kwargs)

    global_repeated = selected_source_rows(
        panel,
        forensics,
        family=EDITING_T1_GLOBAL_FAMILY_SELECTOR,
        panel_kind=EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
    )
    assert 64 <= len(global_repeated) <= 128
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
    assert len(global_by_source) == panel["census"]["global_repeated_multitarget_groups_total"]
    assert all(len(targets) > 1 for targets in global_by_source.values())
    assert (
        sum(len(families) > 1 for families in global_families_by_source.values())
        == panel["census"]["global_repeated_cross_family_groups_total"]
    )
    assert panel["census"]["global_repeated_available_rows_total"] > len(global_repeated)
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


def test_t1_panel_derives_alias_strata_without_caller_omission(
    frozen_inputs,
):
    _forensics, kwargs = frozen_inputs
    panel = build_editing_t1_panel(**kwargs)
    assert tuple(panel["selection"]["required_high_candidate_families"]) == (
        RINGCORE_EDITING_FAMILIES
    )
    assert tuple(panel["selection"]["required_aliased_teacher_families"]) == (
        RINGCORE_EDITING_FAMILIES
    )
    with pytest.raises(TypeError, match="unexpected keyword"):
        build_editing_t1_panel(
            **kwargs,
            required_aliased_teacher_families=(),
        )


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
        decisions_by_digest=MappingProxyType({"a" * 64: (("middle-disallowed", False),)}),
        shard_digest_by_lane=MappingProxyType(
            {
                (
                    "corruption",
                    "validation",
                    "fixture.jsonl.gz",
                ): "a" * 64
            }
        ),
        shard_metadata_by_digest=MappingProxyType({"a" * 64: MappingProxyType({})}),
        counts=MappingProxyType({}),
    )
    neighboring_rows = (
        {
            "record_key": ("corruption/validation/fixture.jsonl.gz:0:middle-disallowed"),
            "partition": "validation",
            "exact_state_ref": {
                "shard_sha256": "a" * 64,
                "record_index": 0,
                "progress_index": 0,
                "shard_name": "corruption/validation/fixture.jsonl.gz",
            },
        },
        {
            "record_key": ("corruption/validation/fixture.jsonl.gz:0:middle-disallowed"),
            "partition": "validation",
            "exact_state_ref": {
                "shard_sha256": "a" * 64,
                "record_index": 0,
                "progress_index": 2,
                "shard_name": "corruption/validation/fixture.jsonl.gz",
            },
        },
    )

    accepted_rows, excluded_rows, excluded_traces = filter_active8_forensics_rows(
        neighboring_rows, admission
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


def test_direct_t1_cli_defaults_to_one_consistent_current_authority():
    from scripts import run_editing_t1_successor_gate as t1_cli

    args = t1_cli._parser().parse_args(
        [
            "--transfer-root",
            "/tmp/t1-fixture",
            "--active8-inventory",
            "/tmp/t1-fixture/ACTIVE8_TRACE_INVENTORY.json",
            "--active8-inventory-file-sha256",
            "a" * 64,
            "--family",
            "cycle_attach",
        ]
    )
    assert args.t1_contract == ROOT / EDITING_T1_V8_CONTRACT_RELATIVE_PATH
    assert args.panel == ROOT / EDITING_T1_V4_PANEL_RELATIVE_PATH
    assert args.capacity_census == ROOT / EDITING_T1_V4_CAPACITY_CENSUS_RELATIVE_PATH


def test_t1_v4_contract_freeze_is_atomic_and_cross_validated(tmp_path):
    authority, capacity_content, optimization, thresholds = _synthetic_t1_launch_authority()
    contract_path = tmp_path / "editing_t1_successor_gate_v4.json"
    panel_path = tmp_path / "editing_t1_successor_panel_v4.json"
    capacity_path = tmp_path / "editing_t1_capacity_census_v1.json"
    panel_path.write_text(
        json.dumps(
            dict(authority.panel),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    capacity_path.write_bytes(capacity_content)

    write_editing_t1_runtime_contract(authority.contract, contract_path)
    first_bytes = contract_path.read_bytes()
    write_editing_t1_runtime_contract(authority.contract, contract_path)
    assert contract_path.read_bytes() == first_bytes

    observed = load_editing_t1_launch_authority(
        contract_path=contract_path,
        panel_path=panel_path,
        capacity_census_path=capacity_path,
        expected_active8_inventory_manifest_file_sha256="1" * 64,
    )
    assert observed.contract.sha256 == authority.contract.sha256
    assert observed.capacity_census_file_sha256 == hashlib.sha256(capacity_content).hexdigest()

    conflicting_optimization = {
        **optimization,
        "steps": 3,
        "report_points": [1, 2, 3],
    }
    conflicting = build_editing_t1_runtime_contract(
        gate_zero_runtime_contract_sha256="5" * 64,
        panel=authority.panel,
        capacity_census=authority.capacity_census,
        capacity_census_file_sha256=authority.capacity_census_file_sha256,
        active8_identity=authority.active8_identity,
        optimization=conflicting_optimization,
        thresholds=thresholds,
    )
    with pytest.raises(EditingT1RuntimeError, match="already exists"):
        write_editing_t1_runtime_contract(conflicting, contract_path)


def test_t1_v4_contract_refuses_unfrozen_thresholds_and_identity_drift():
    authority, _capacity_content, optimization, thresholds = _synthetic_t1_launch_authority()
    unfrozen = {**thresholds, "minimum_unique_state_teacher_successor_top1": None}
    with pytest.raises(EditingT1RuntimeError, match="must be finite"):
        build_editing_t1_runtime_contract(
            gate_zero_runtime_contract_sha256="5" * 64,
            panel=authority.panel,
            capacity_census=authority.capacity_census,
            capacity_census_file_sha256=authority.capacity_census_file_sha256,
            active8_identity=authority.active8_identity,
            optimization=optimization,
            thresholds=unfrozen,
        )
    nonfinite = {
        **thresholds,
        "minimum_unique_state_teacher_successor_probability": float("nan"),
    }
    with pytest.raises(EditingT1RuntimeError, match="must be finite"):
        build_editing_t1_runtime_contract(
            gate_zero_runtime_contract_sha256="5" * 64,
            panel=authority.panel,
            capacity_census=authority.capacity_census,
            capacity_census_file_sha256=authority.capacity_census_file_sha256,
            active8_identity=authority.active8_identity,
            optimization=optimization,
            thresholds=nonfinite,
        )
    with pytest.raises(EditingT1RuntimeError, match="another physical Active8"):
        validate_editing_t1_launch_authority(
            contract=authority.contract,
            panel=authority.panel,
            capacity_census=authority.capacity_census,
            capacity_census_file_sha256=authority.capacity_census_file_sha256,
            expected_active8_inventory_manifest_file_sha256="f" * 64,
        )
    with pytest.raises(EditingT1RuntimeError, match="contract disagrees"):
        validate_editing_t1_launch_authority(
            contract=authority.contract,
            panel=authority.panel,
            capacity_census=authority.capacity_census,
            capacity_census_file_sha256="e" * 64,
            expected_active8_inventory_manifest_file_sha256="1" * 64,
        )


def test_t1_contract_and_ring_family_scope_ladder_are_explicit():
    contract_path = ROOT / "configs" / "editing_t1_successor_gate_v3.json"
    with pytest.raises(EditingT1RuntimeError):
        load_editing_t1_runtime_contract(contract_path)
    payload = json.loads(contract_path.read_text())
    payload.update(
        {
            "schema_version": EDITING_T1_RUNTIME_CONTRACT_VERSION,
            "active8_inventory_manifest_file_sha256": "1" * 64,
            "active8_inventory_sha256": "2" * 64,
            "active8_effective_source_corpus_cache_sha256": "5" * 64,
            "active8_unified_packed_manifest_sha256": "3" * 64,
            "active8_support_contract_sha256": payload["gate_zero_runtime_contract_sha256"],
            "capacity_census_file_sha256": "6" * 64,
            "capacity_census_sha256": "7" * 64,
            "panel_selection_sha256": "8" * 64,
            "panel_census_sha256": "9" * 64,
            "panel_capacity_strata_sha256": "a" * 64,
            "required_high_candidate_families": list(RINGCORE_EDITING_FAMILIES),
            "required_aliased_teacher_families": [
                "atom_insert",
                "bond_reroute",
                "cycle_attach",
            ],
            "within_family_repeated_families": [
                "bond_reroute",
                "cycle_attach",
            ],
            "implementation_sha256": editing_t1_implementation_sha256(),
        }
    )
    payload["thresholds"] = {
        "minimum_unique_state_teacher_successor_top1": 0.8,
        "minimum_unique_state_teacher_successor_probability": 0.6,
        "maximum_unique_state_teacher_successor_nll": 0.7,
        "maximum_global_repeated_state_excess_nll_over_empirical_entropy": (0.2),
    }
    payload["thresholds_sha256"] = editing_t1_numeric_thresholds_sha256(payload["thresholds"])
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
            {key: value for key, value in payload.items() if key != "contract_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    contract = EditingT1RuntimeContract(payload)
    assert contract.training_authorized is False
    assert contract.numeric_thresholds_sha256 == payload["thresholds_sha256"]
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

    unfrozen = copy.deepcopy(payload)
    unfrozen["thresholds"]["minimum_unique_state_teacher_successor_top1"] = None
    unfrozen["thresholds_sha256"] = "0" * 64
    unfrozen["contract_sha256"] = _stable_sha256(
        {key: value for key, value in unfrozen.items() if key != "contract_sha256"}
    )
    with pytest.raises(
        EditingT1RuntimeError,
        match="must be finite",
    ):
        EditingT1RuntimeContract(unfrozen)

    implementation_sources = set(editing_t1_implementation_sources())
    assert {
        "modal_apps/run_editing_t1_successor_gate.py",
        "scripts/build_editing_t1_successor_panel.py",
        "scripts/freeze_editing_t1_successor_contract.py",
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


def _runtime_fixture(
    *,
    smiles: str = "CC1CCCCC1O",
    shard_digest: str = "1" * 64,
    shard_name: str = "shard_0000.jsonl.gz",
    row_sha256: str = "6" * 64,
):
    records, attempted = build_cycle_op_records(
        (smiles,),
        n_slots=12,
        seed=4,
        max_bonds_per_molecule=1,
    )
    assert attempted > 0 and records
    original = records[0]
    path = original.path
    step = path.trace.steps[0]
    family = canonical_family(step.rule_name)
    address = PackedTraceAddress(
        packed_shard_content_sha256=shard_digest,
        packed_shard_name=shard_name,
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
        "row_sha256": row_sha256,
        "record_key": (f"cycle_ops/validation/{address.packed_shard_name}:0:{address.trace_id}"),
        "partition": "validation",
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


def test_capacity_census_recomputes_exact_production_support_and_rejects_forgery(
    monkeypatch,
):
    source, forensics, _panel, _family = _runtime_fixture()
    forensics = {
        "artifact_sha256": "7" * 64,
        "rows": forensics["rows"],
    }
    monkeypatch.setattr(
        editing_t1_panel_module,
        "_validate_forensics_rows",
        lambda payload, _sidecar: tuple(payload["rows"]),
    )
    admission = _active8_admission_for_record(
        source.records[0],
        accepted=True,
        support_contract_sha256="8" * 64,
        unified_packed_manifest_sha256=(source.unified_packed_manifest_sha256),
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
    kwargs = {
        "model": model,
        "source": source,
        "forensics": forensics,
        "forensics_file_sha256": "9" * 64,
        "active8_admission": admission,
        "gate_zero_runtime_contract_sha256": "8" * 64,
    }
    census = build_editing_t1_capacity_census(**kwargs)
    row = census["rows"][0]
    assert row["raw_legal_mark_count"] >= row["canonical_successor_count"] > 0
    assert (
        row["teacher_canonical_successor_alias_count"]
        >= row["teacher_exact_successor_alias_count"]
        > 0
    )
    validate_editing_t1_capacity_census(census, **kwargs)

    forged = copy.deepcopy(census)
    forged["rows"][0]["raw_legal_mark_count"] += 1
    forged["census_sha256"] = _stable_sha256(
        {key: value for key, value in forged.items() if key != "census_sha256"}
    )
    with pytest.raises(
        EditingT1PanelError,
        match="production re-enumeration",
    ):
        validate_editing_t1_capacity_census(forged, **kwargs)


def test_capacity_census_parallel_compilation_is_byte_identical_and_bounded(
    monkeypatch,
):
    first_source, first_forensics, _panel, _family = _runtime_fixture()
    second_source, second_forensics, _panel, _family = _runtime_fixture(
        smiles="CC1CCCC1O",
        shard_digest="a" * 64,
        shard_name="shard_0001.jsonl.gz",
        row_sha256="b" * 64,
    )
    second_row = {
        **second_forensics["rows"][0],
        "row_index": 1,
    }
    source = FrozenValidationSource(
        records=(*first_source.records, *second_source.records),
        sidecar=LoadedSemanticCellSidecar(
            rows=(*first_source.sidecar.rows, *second_source.sidecar.rows),
            manifest={},
        ),
        bindings=(*first_source.bindings, *second_source.bindings),
        unified_packed_manifest_sha256=(first_source.unified_packed_manifest_sha256),
        representability_overlay_sha256=(first_source.representability_overlay_sha256),
    )
    forensics = {
        "artifact_sha256": "7" * 64,
        "rows": [first_forensics["rows"][0], second_row],
    }
    monkeypatch.setattr(
        editing_t1_panel_module,
        "_validate_forensics_rows",
        lambda payload, _sidecar: tuple(payload["rows"]),
    )
    admission = _active8_admission_for_forensics(
        forensics,
        unified_packed_manifest_sha256=(source.unified_packed_manifest_sha256),
    )
    torch.manual_seed(7)
    serial_model = FactorizedTraceletRateModel(
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
    parallel_model = copy.deepcopy(serial_model)
    kwargs = {
        "source": source,
        "forensics": forensics,
        "forensics_file_sha256": "9" * 64,
        "active8_admission": admission,
        "gate_zero_runtime_contract_sha256": admission.support_contract_sha256,
    }

    serial = build_editing_t1_capacity_census(
        serial_model,
        **kwargs,
        workers=1,
    )
    parallel = build_editing_t1_capacity_census(
        parallel_model,
        **kwargs,
        workers=2,
    )

    assert parallel == serial
    assert (
        json.dumps(parallel, indent=2, sort_keys=True, allow_nan=False).encode()
        == json.dumps(serial, indent=2, sort_keys=True, allow_nan=False).encode()
    )
    with pytest.raises(ValueError, match="integer from 1 through 8"):
        build_editing_t1_capacity_census(
            parallel_model,
            **kwargs,
            workers=9,
        )
    unsupported_model = copy.deepcopy(serial_model)
    unsupported_model.enable_cycle_ops = False
    with pytest.raises(
        EditingT1PanelError,
        match="could not enumerate production successor support",
    ):
        build_editing_t1_capacity_census(
            unsupported_model,
            **kwargs,
            workers=2,
        )


def test_t1_panel_builder_cli_has_explicit_bounded_worker_count():
    from scripts import build_editing_t1_successor_panel as panel_cli

    required = [
        "--transfer-root",
        "/tmp/t1-fixture",
        "--active8-inventory",
        "/tmp/t1-fixture/ACTIVE8_TRACE_INVENTORY.json",
        "--active8-inventory-file-sha256",
        "a" * 64,
        "--capacity-census-output",
        "/tmp/t1-fixture/census.json",
        "--output",
        "/tmp/t1-fixture/panel.json",
    ]
    assert panel_cli._parser().parse_args(required).workers == 1
    assert panel_cli._parser().parse_args([*required, "--workers", "8"]).workers == 8
    with pytest.raises(SystemExit):
        panel_cli._parser().parse_args([*required, "--workers", "0"])


def test_t1_runtime_fails_on_gate_zero_active8_identity_mismatch():
    source, _forensics, _panel, _family = _runtime_fixture()
    admission = _active8_admission_for_record(
        source.records[0],
        accepted=True,
        support_contract_sha256="4" * 64,
        unified_packed_manifest_sha256=(source.unified_packed_manifest_sha256),
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


def test_run_arm_rejects_panel_census_or_strata_identity_drift_before_training():
    panel = {
        "artifact_sha256": "1" * 64,
        "selection": {
            "required_high_candidate_families": list(RINGCORE_EDITING_FAMILIES),
            "required_aliased_teacher_families": [],
        },
        "census": {
            "unique_high_candidate_stratum_available_by_family": {},
            "unique_high_candidate_stratum_selected_by_family": {},
            "unique_aliased_teacher_stratum_available_by_family": {},
            "unique_aliased_teacher_stratum_selected_by_family": {},
            "registered_high_candidate_families_all_covered": True,
            "registered_aliased_teacher_families_all_covered": True,
            "families_without_observed_within_family_multitarget_repeats": [],
        },
    }
    contract = SimpleNamespace(
        payload={
            "gate_zero_runtime_contract_sha256": "2" * 64,
            "panel_artifact_sha256": panel["artifact_sha256"],
            "panel_selection_sha256": _stable_sha256(panel["selection"]),
            "panel_census_sha256": _stable_sha256(panel["census"]),
            "panel_capacity_strata_sha256": "3" * 64,
        }
    )
    with pytest.raises(
        EditingT1RuntimeError,
        match="panel identity: panel_capacity_strata_sha256",
    ):
        run_editing_t1_arm(
            contract=contract,
            gate_zero_contract=SimpleNamespace(
                sha256="2" * 64,
                model={"enable_ring_system_delete": False},
            ),
            source=None,
            panel=panel,
            forensics={},
            family="atom_insert",
            panel_kind=EDITING_T1_UNIQUE_PANEL_KIND,
            scope="all",
            device=torch.device("cpu"),
            excluded_trace_ids={},
            active8_admission=None,
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

    with pytest.raises(
        EditingT1RuntimeError,
        match="another exact panel trace set",
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
            active8_admission=active8_admission,
            successor_cache=SimpleNamespace(
                receipt=SimpleNamespace(selected_trace_set_sha256="0" * 64)
            ),
        )

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
    metrics = {
        "n_examples": 64,
        "canonical_successor_nll": 0.2,
        "teacher_successor_probability": 0.8,
        "teacher_successor_top1_recall": 1.0,
        "teacher_family_probability": 0.9,
        "teacher_family_nll": 0.1,
        "within_teacher_family_successor_probability": 0.9,
        "within_teacher_family_successor_nll": 0.1,
        "mean_raw_mark_count": 4.0,
        "mean_canonical_successor_count": 3.0,
        "mean_alias_multiplicity": 1.0,
        "repeated_state_excess_nll_over_empirical_entropy": 0.0,
    }
    body = {
        "schema": EDITING_T1_RESULT_SCHEMA,
        "schema_version": EDITING_T1_RESULT_VERSION,
        "status": EDITING_T1_RESULT_STATUS,
        "training_authorized": False,
        "gate_decision": None,
        "numeric_thresholds_frozen": True,
        "numeric_thresholds_sha256": "0" * 64,
        "contract_sha256": "4" * 64,
        "gate_zero_runtime_contract_sha256": "5" * 64,
        "panel_artifact_sha256": "6" * 64,
        "panel_selection_sha256": "1" * 64,
        "panel_census_sha256": "2" * 64,
        "panel_capacity_strata_sha256": "3" * 64,
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
        "example_count": 64,
        "unique_progress_address_count": 64,
        "repeated_progress_observation_count": 0,
        "source_row_sha256s": [f"{index + 1:064x}" for index in range(64)],
        "scratch_initialization": {
            "regime": "scratch",
            "transfer_plan_sha256": "a" * 64,
            "initial_model_state_sha256": "b" * 64,
        },
        "cache_receipts": [receipt.__dict__],
        "training_report": {
            "scope": "heads_only",
            "steps": 1,
            "families": ["atom_insert"],
            "optimizer_steps_with_nonzero_gradient": 1,
            "required_components_without_gradient": [],
            "component_gradient_update_counts": {"family_head": 1},
            "history": [{"step": 1.0, "loss": 1.0, "gradient_norm": 1.0}],
            "initial": dict(metrics),
            "final": dict(metrics),
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
        "expected_numeric_thresholds_sha256": result["numeric_thresholds_sha256"],
        "expected_panel_artifact_sha256": result["panel_artifact_sha256"],
        "expected_panel_selection_sha256": result["panel_selection_sha256"],
        "expected_panel_census_sha256": result["panel_census_sha256"],
        "expected_panel_capacity_strata_sha256": result["panel_capacity_strata_sha256"],
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
            "active8_support_contract_sha256": (result["active8_support_contract_sha256"]),
        },
        "expected_cache_receipts": (receipt,),
    }
    observed = validate_editing_t1_result(result, **expected)
    assert observed["training_authorized"] is False
    assert validate_t1_cache_shard_receipt(receipt.__dict__, expected=receipt) == receipt

    mapping_proxy_expected = {
        **expected,
        "expected_active8_identity": MappingProxyType(
            dict(expected["expected_active8_identity"])
        ),
    }
    assert (
        validate_editing_t1_result(result, **mapping_proxy_expected)["result_sha256"]
        == result["result_sha256"]
    )

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
    with pytest.raises(
        EditingT1RuntimeError,
        match="another prospective threshold map",
    ):
        validate_editing_t1_result(
            result,
            **{
                **expected,
                "expected_numeric_thresholds_sha256": "d" * 64,
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
