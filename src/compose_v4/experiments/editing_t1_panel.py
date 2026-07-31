"""Immutable exact-address panels for the editing T1 capacity gate.

T1 is a bounded *development* diagnostic.  It asks whether the configured
scratch editor can memorize exact canonical molecular successors before any
mixed-corpus pilot is allowed.  This module derives two disjoint panel views
from the already-frozen RingCore validation family-forensics panel:

* exactly 64 globally deterministic, unique persistent-slot source states for
  every active family;
* a 64-to-128-observation complete-group panel of exact states having more than
  one teacher successor under the full mixed-family editing law; and
* explicitly secondary, at-most-128-observation within-family repeated-state
  views.

The deterministic panels exclude any source that has multiple target
successors anywhere in the active eight-family projection of the complete
frozen forensics panel, including targets recorded under another active
operator family. The frozen ``ring_system_delete`` rows are explicitly counted
but excluded before grouping under the bounded-pilot operator freeze. Thus an
operator-specific capacity arm cannot turn a genuinely stochastic active-family
choice into an arbitrary one-hot label. No example is synthesized or duplicated
to satisfy a count. The artifact contains immutable source-row references and is
re-derived during loading, so changing the source panel, semantic sidecar, or
selection implementation fails closed.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import tempfile
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from compose_v4.data.active8_trace_inventory import (
    Active8TraceAdmission,
    Active8TraceInventoryError,
)
from compose_v4.data.successor_fiber_cache import (
    fiber_compiler_implementation_hash,
)
from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.experiments.editing_gate_zero_runtime import (
    FrozenValidationSource,
)
from compose_v4.experiments.factorized_successor_training import (
    SuccessorTrainingError,
    compile_state_successor_map,
    require_exact_successor_action_identity,
    rewrite_action_codec_sha256,
)
from compose_v4.experiments.ringcore_semantic_sidecar import (
    LoadedSemanticCellSidecar,
)
from compose_v4.experiments.successor_micro_overfit import (
    RINGCORE_EDITING_FAMILIES,
)
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.rewrite.action_codec import canonical_family
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    de_novo_rewrite_system,
)

EDITING_T1_PANEL_SCHEMA = "compose.editing.t1_successor_panel"
EDITING_T1_PANEL_SCHEMA_VERSION = 4
EDITING_T1_PANEL_STATUS = "FROZEN_DEVELOPMENT_CAPACITY_PANEL_NO_DECISION"
EDITING_T1_CAPACITY_CENSUS_SCHEMA = "compose.editing.t1_capacity_census"
EDITING_T1_CAPACITY_CENSUS_SCHEMA_VERSION = 1
EDITING_T1_CAPACITY_CENSUS_STATUS = "FROZEN_CHECKPOINT_INDEPENDENT_ACTIVE8_SUCCESSOR_SUPPORT_CENSUS"
EDITING_T1_UNIQUE_EXAMPLES_PER_FAMILY = 64
EDITING_T1_REPEATED_MIN_EXAMPLES = 64
EDITING_T1_REPEATED_MAX_EXAMPLES = 128
EDITING_T1_SUPPORT_TIME = 0.5
EDITING_T1_MAX_CAPACITY_WORKERS = 8
EDITING_T1_SELECTION_ALGORITHM = (
    "physical_active8_whole_trace_admission_then_forensics_stream_first_globally_"
    "deterministic_unique_exact_state_64_per_family_plus_bounded_complete_global_"
    "and_within_family_multitarget_exact_state_groups_with_registered_capacity_strata_v4"
)
EDITING_T1_GLOBAL_FAMILY_SELECTOR = "all_families"
EDITING_T1_UNIQUE_PANEL_KIND = "unique_state"
EDITING_T1_GLOBAL_REPEATED_PANEL_KIND = "global_repeated_state_distribution"
EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND = "within_family_repeated_state_distribution"
EDITING_T1_DIAGNOSTIC_ONLY_FAMILIES: tuple[str, ...] = ()
EDITING_T1_OPERATOR_FREEZE_EXCLUDED_FAMILIES = ("ring_system_delete",)
EDITING_T1_REQUIRED_SCOPES = (
    "heads_only",
    "heads_plus_local_adapter",
    "all",
)

_TOP_LEVEL_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "source",
    "selection",
    "architecture",
    "thresholds",
    "unique_state_panels",
    "global_repeated_state_panel",
    "within_family_repeated_state_panels",
    "census",
    "artifact_sha256",
}
_SOURCE_FIELDS = {
    "forensics_file_sha256",
    "forensics_artifact_sha256",
    "leaderboard_protocol_sha256",
    "frozen_inventory_sha256",
    "semantic_sidecar_file_sha256",
    "semantic_sidecar_manifest_file_sha256",
    "semantic_sidecar_manifest_sha256",
    "semantic_sidecar_config_sha256",
    "semantic_sidecar_provenance_sha256",
    "charge_policy_audit_file_sha256",
    "charge_policy_exclusions_file_sha256",
    "charge_policy_exclusion_payload_sha256",
    "charge_policy_source_input_inventory_sha256",
    "charge_policy_audit_implementation_sha256",
    "charge_policy_excluded_unique_trace_addresses",
    "capacity_census_file_sha256",
    "capacity_census_sha256",
    "active8_inventory_manifest_file_sha256",
    "active8_inventory_sha256",
    "active8_effective_source_corpus_cache_sha256",
    "active8_unified_packed_manifest_sha256",
    "active8_support_contract_sha256",
}
_SELECTION_FIELDS = {
    "algorithm",
    "active_families",
    "diagnostic_only_families",
    "operator_freeze_excluded_families",
    "unique_examples_per_family",
    "required_high_candidate_families",
    "required_aliased_teacher_families",
    "high_candidate_definition",
    "aliased_teacher_definition",
    "global_determinism_policy",
    "primary_repeated_state_policy",
    "conditional_repeated_state_policy",
    "conditional_repeated_state_relationship",
    "operator_freeze_exclusion_policy",
    "charge_policy_exclusion_policy",
    "active8_whole_trace_admission_policy",
    "support_time",
}
_ARCHITECTURE_FIELDS = {
    "hidden_dim",
    "message_passing_steps",
    "mark_dim",
    "atom_vocabulary_class_count",
    "enable_ring_restates",
    "enable_cyclic_graft",
    "enable_heteroatom_scan",
    "enable_ring_opening",
    "enable_cycle_ops",
    "enable_ring_grow_macro",
    "enable_ring_system_delete",
    "required_scope_ladder",
}
_THRESHOLD_FIELDS = {
    "minimum_unique_state_teacher_successor_top1",
    "minimum_unique_state_teacher_successor_probability",
    "maximum_unique_state_teacher_successor_nll",
    "maximum_global_repeated_state_excess_nll_over_empirical_entropy",
}
_CAPACITY_CENSUS_ROW_FIELDS = {
    "source_row_index",
    "source_row_sha256",
    "packed_shard_content_sha256",
    "packed_entry_index",
    "progress_index",
    "teacher_family",
    "teacher_action_sha256",
    "teacher_successor_key",
    "teacher_successor_state_sha256",
    "raw_legal_mark_count",
    "canonical_successor_count",
    "teacher_canonical_successor_alias_count",
    "teacher_exact_successor_alias_count",
}
_ROW_REF_FIELDS = {"source_row_index", "source_row_sha256"}
_WITHIN_FAMILY_REPEATED_GROUP_FIELDS = {
    "source_state_sha256",
    "distinct_teacher_successor_count",
    "rows",
}
_GLOBAL_REPEATED_GROUP_FIELDS = {
    *_WITHIN_FAMILY_REPEATED_GROUP_FIELDS,
    "distinct_teacher_family_count",
}
_CHARGE_EXCLUSION_FIELDS = {
    "schema",
    "schema_version",
    "selection",
    "source_input_inventory_sha256",
    "charge_policy_audit_implementation_sha256",
    "excluded_unique_trace_addresses",
    "violation_type_bit_order",
    "shards",
    "payload_sha256",
}
_CHARGE_EXCLUSION_SHARD_FIELDS = {
    "manifest_layer",
    "envelope_layer",
    "partition",
    "relative_path",
    "packed_shard_content_sha256",
    "excluded_entry_count",
    "violating_entry_indices",
    "entries",
}
_CHARGE_EXCLUSION_ENTRY_FIELDS = {
    "entry_index",
    "trace_id",
    "violation_type_mask",
}


class EditingT1PanelError(RuntimeError):
    """The T1 panel or one of its frozen sources is invalid."""


ACTIVE8_T1_IDENTITY_FIELDS = (
    "active8_inventory_manifest_file_sha256",
    "active8_inventory_sha256",
    "active8_effective_source_corpus_cache_sha256",
    "active8_unified_packed_manifest_sha256",
    "active8_support_contract_sha256",
)
_CAPACITY_CENSUS_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "forensics_file_sha256",
    "forensics_artifact_sha256",
    "gate_zero_runtime_contract_sha256",
    "unified_packed_manifest_sha256",
    "representability_overlay_sha256",
    *ACTIVE8_T1_IDENTITY_FIELDS,
    "operator_capability_fingerprint",
    "support_time",
    "implementation",
    "rows",
    "census_sha256",
}
_CAPACITY_CENSUS_IMPLEMENTATION_FIELDS = {
    "builder_implementation_sha256",
    "fiber_compiler_implementation_sha256",
}
_CAPACITY_CENSUS_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/experiments/editing_t1_panel.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/experiments/production_successor_kernel.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "src/compose_v4/rewrite/action_codec.py",
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/chem/persistent_state_identity.py",
)


@dataclass(frozen=True)
class _ForensicsTraceAddress:
    packed_shard_content_sha256: str
    entry_index: int
    trace_id: str


@dataclass(frozen=True)
class _CapacityTeacherRequest:
    row_position: int
    target_state_sha256: str
    action_sha256: str
    teacher_successor_key: str


@dataclass(frozen=True)
class _CapacitySourceRequest:
    source_state_sha256: str
    state: MolecularGraph
    teachers: tuple[_CapacityTeacherRequest, ...]


@dataclass(frozen=True)
class _CapacityRowMetrics:
    row_position: int
    raw_legal_mark_count: int
    canonical_successor_count: int
    teacher_canonical_successor_alias_count: int
    teacher_exact_successor_alias_count: int


_CAPACITY_WORKER_MODEL: Any | None = None


def active8_t1_identity(
    admission: Active8TraceAdmission,
) -> Mapping[str, str]:
    """Project the five non-interchangeable Active8 identities used by T1."""

    if not isinstance(admission, Active8TraceAdmission):
        raise TypeError("T1 requires a verified Active8TraceAdmission")
    return MappingProxyType(
        {
            "active8_inventory_manifest_file_sha256": (admission.manifest_file_sha256),
            "active8_inventory_sha256": admission.inventory_sha256,
            "active8_effective_source_corpus_cache_sha256": (
                admission.effective_source_corpus_cache_sha256
            ),
            "active8_unified_packed_manifest_sha256": (admission.unified_packed_manifest_sha256),
            "active8_support_contract_sha256": (admission.support_contract_sha256),
        }
    )


def _forensics_trace_address(
    row: Mapping[str, Any],
    admission: Active8TraceAdmission,
) -> _ForensicsTraceAddress:
    state_ref = row.get("exact_state_ref")
    if not isinstance(state_ref, Mapping):
        raise EditingT1PanelError("forensics row lacks an exact packed-state reference")
    shard_digest = state_ref.get("shard_sha256")
    record_index = state_ref.get("record_index")
    shard_name = state_ref.get("shard_name")
    record_key = row.get("record_key")
    partition = row.get("partition")
    if (
        not _is_sha256(shard_digest)
        or type(record_index) is not int
        or record_index < 0
        or not isinstance(shard_name, str)
        or not shard_name
        or not isinstance(record_key, str)
        or not record_key
        or not isinstance(partition, str)
        or not partition
    ):
        raise EditingT1PanelError("forensics row lacks a complete Active8 trace address")
    shard_path = PurePosixPath(shard_name)
    if (
        shard_path.is_absolute()
        or ".." in shard_path.parts
        or len(shard_path.parts) < 3
        or shard_path.parts[-2] != partition
    ):
        raise EditingT1PanelError("forensics packed shard path disagrees with its partition")
    prefix = f"{shard_name}:{record_index}:"
    if not record_key.startswith(prefix) or len(record_key) == len(prefix):
        raise EditingT1PanelError("forensics record key disagrees with its exact packed address")
    trace_id = record_key[len(prefix) :]
    try:
        expected_digest = admission.expected_source_digest(
            packed_shard_name=shard_path.name,
            layer=shard_path.parts[-3],
            partition=partition,
        )
    except Active8TraceInventoryError as error:
        raise EditingT1PanelError(
            "forensics row names a shard absent from the physical Active8 inventory"
        ) from error
    if expected_digest != shard_digest:
        raise EditingT1PanelError(
            "forensics row shard digest disagrees with the physical Active8 inventory"
        )
    return _ForensicsTraceAddress(
        packed_shard_content_sha256=shard_digest,
        entry_index=record_index,
        trace_id=trace_id,
    )


def filter_active8_forensics_rows(
    rows: Sequence[Mapping[str, Any]],
    admission: Active8TraceAdmission,
) -> tuple[tuple[Mapping[str, Any], ...], int, int]:
    """Apply the immutable whole-trace decision before any T1 row selection.

    Returns ``(accepted_rows, excluded_row_count, excluded_trace_count)``.
    One excluded middle action therefore removes every otherwise usable
    prefix, suffix, and terminal observation from the trace.
    """

    if not isinstance(admission, Active8TraceAdmission):
        raise TypeError("T1 requires a verified Active8TraceAdmission")
    accepted_rows: list[Mapping[str, Any]] = []
    excluded_traces: set[tuple[str, int, str]] = set()
    for row in rows:
        address = _forensics_trace_address(row, admission)
        try:
            accepted = admission.is_accepted(address)
        except Active8TraceInventoryError as error:
            raise EditingT1PanelError(
                "forensics row disagrees with its whole-trace Active8 decision"
            ) from error
        if accepted:
            accepted_rows.append(row)
        else:
            excluded_traces.add(
                (
                    address.packed_shard_content_sha256,
                    address.entry_index,
                    address.trace_id,
                )
            )
    return (
        tuple(accepted_rows),
        len(rows) - len(accepted_rows),
        len(excluded_traces),
    )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EditingT1PanelError("T1 panel metadata is not finite canonical JSON") from error


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EditingT1PanelError(f"{name} must be an object")
    if set(value) != expected:
        raise EditingT1PanelError(
            f"{name} fields disagree; "
            f"missing={sorted(expected - set(value))!r}, "
            f"unexpected={sorted(set(value) - expected)!r}"
        )
    return value


def _validate_registered_family_subset(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or (not value and not allow_empty)
        or any(not isinstance(family, str) for family in value)
    ):
        qualifier = "" if allow_empty else "nonempty "
        raise EditingT1PanelError(f"{name} must be a {qualifier}family sequence")
    observed = tuple(value)
    if len(observed) != len(set(observed)):
        raise EditingT1PanelError(f"{name} contains duplicate families")
    expected_order = tuple(family for family in RINGCORE_EDITING_FAMILIES if family in observed)
    if observed != expected_order:
        raise EditingT1PanelError(f"{name} must be an ordered subset of the Active8 families")
    return observed


def editing_t1_capacity_census_implementation_sha256() -> str:
    """Hash the code that resolves packed rows and enumerates production support."""

    repository = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for relative in _CAPACITY_CENSUS_IMPLEMENTATION_SOURCES:
        source = repository / relative
        digest.update(relative.encode("utf-8"))
        digest.update(source.read_bytes() if source.exists() else b"<MISSING>")
    return digest.hexdigest()


def _record_index(
    source: FrozenValidationSource,
) -> Mapping[tuple[str, int], Any]:
    index: dict[tuple[str, int], Any] = {}
    for record in source.records:
        address = record.corpus_address
        if address is None:
            raise EditingT1PanelError("T1 capacity census cannot use an unaddressed packed trace")
        key = (
            address.packed_shard_content_sha256,
            address.entry_index,
        )
        if key in index:
            raise EditingT1PanelError(
                "T1 capacity census source repeats an exact packed trace address"
            )
        index[key] = record
    return MappingProxyType(index)


def _require_capacity_workers(workers: int) -> int:
    if type(workers) is not int or workers < 1 or workers > EDITING_T1_MAX_CAPACITY_WORKERS:
        raise ValueError(
            "T1 capacity census workers must be an integer from "
            f"1 through {EDITING_T1_MAX_CAPACITY_WORKERS}"
        )
    return workers


def _compile_capacity_source(
    request: _CapacitySourceRequest,
    *,
    model: Any,
) -> tuple[_CapacityRowMetrics, ...]:
    runtime = de_novo_rewrite_system()
    try:
        compiled = compile_state_successor_map(
            model,
            request.state,
            time=EDITING_T1_SUPPORT_TIME,
            system=runtime,
        )
    except SuccessorTrainingError as error:
        raise EditingT1PanelError(
            "T1 capacity census could not enumerate production successor support"
        ) from error
    if compiled.source_state_sha256 != request.source_state_sha256:
        raise EditingT1PanelError(
            "T1 capacity census compilation returned another exact source state"
        )
    raw_mark_count = sum(len(group.marks) for group in compiled.successor_groups) + len(
        compiled.virtual_marks
    )
    if raw_mark_count <= 0:
        raise EditingT1PanelError("T1 capacity census encountered empty production support")

    metrics: list[_CapacityRowMetrics] = []
    for teacher in request.teachers:
        try:
            require_exact_successor_action_identity(
                compiled,
                target_state_sha256=teacher.target_state_sha256,
                action_sha256=teacher.action_sha256,
            )
        except SuccessorTrainingError as error:
            raise EditingT1PanelError(
                "T1 capacity census teacher is absent from exact production support"
            ) from error
        teacher_group = next(
            (
                group
                for group in compiled.successor_groups
                if group.target_key == teacher.teacher_successor_key
            ),
            None,
        )
        if teacher_group is None:
            raise EditingT1PanelError("T1 capacity census teacher canonical successor is absent")
        exact_alias_count = sum(
            mark.successor_state_sha256 == teacher.target_state_sha256
            for mark in teacher_group.marks
        )
        if exact_alias_count <= 0:
            raise EditingT1PanelError("T1 capacity census encountered empty production support")
        metrics.append(
            _CapacityRowMetrics(
                row_position=teacher.row_position,
                raw_legal_mark_count=raw_mark_count,
                canonical_successor_count=len(compiled.successor_groups),
                teacher_canonical_successor_alias_count=len(teacher_group.marks),
                teacher_exact_successor_alias_count=exact_alias_count,
            )
        )
    return tuple(metrics)


def _initialize_capacity_worker(model: Any) -> None:
    global _CAPACITY_WORKER_MODEL

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _CAPACITY_WORKER_MODEL = model


def _compile_capacity_source_in_worker(
    request: _CapacitySourceRequest,
) -> tuple[_CapacityRowMetrics, ...]:
    if _CAPACITY_WORKER_MODEL is None:
        raise EditingT1PanelError("T1 capacity worker was not initialized")
    return _compile_capacity_source(
        request,
        model=_CAPACITY_WORKER_MODEL,
    )


def _compile_capacity_sources(
    requests: Sequence[_CapacitySourceRequest],
    *,
    model: Any,
    workers: int,
) -> tuple[tuple[_CapacityRowMetrics, ...], ...]:
    if not requests:
        return ()
    if workers == 1:
        return tuple(_compile_capacity_source(request, model=model) for request in requests)

    results: list[tuple[_CapacityRowMetrics, ...]] = []
    request_iterator = iter(requests)
    pending = deque()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_capacity_worker,
        initargs=(model,),
    ) as executor:
        for _ in range(min(workers, len(requests))):
            pending.append(
                executor.submit(
                    _compile_capacity_source_in_worker,
                    next(request_iterator),
                )
            )
        while pending:
            results.append(pending.popleft().result())
            try:
                request = next(request_iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(_compile_capacity_source_in_worker, request))
    return tuple(results)


def build_editing_t1_capacity_census(
    model: Any,
    *,
    source: FrozenValidationSource,
    forensics: Mapping[str, Any],
    forensics_file_sha256: str,
    active8_admission: Active8TraceAdmission,
    gate_zero_runtime_contract_sha256: str,
    workers: int = 1,
) -> dict[str, Any]:
    """Recompute exact Active8 successor-support counts from production primitives."""

    workers = _require_capacity_workers(workers)
    for name, value in (
        ("forensics_file_sha256", forensics_file_sha256),
        (
            "gate_zero_runtime_contract_sha256",
            gate_zero_runtime_contract_sha256,
        ),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if not isinstance(source, FrozenValidationSource):
        raise TypeError("T1 capacity census requires a FrozenValidationSource")
    if not isinstance(active8_admission, Active8TraceAdmission):
        raise TypeError("T1 capacity census requires a verified Active8 admission")
    if source.unified_packed_manifest_sha256 != active8_admission.unified_packed_manifest_sha256:
        raise EditingT1PanelError(
            "T1 capacity source and Active8 admission name different unified corpora"
        )
    if active8_admission.support_contract_sha256 != gate_zero_runtime_contract_sha256:
        raise EditingT1PanelError(
            "T1 capacity census Gate0 and Active8 support identities disagree"
        )
    if getattr(model, "enable_ring_system_delete", None) is not False:
        raise EditingT1PanelError(
            "T1 capacity census requires the exact Active8 model with ring_system_delete disabled"
        )
    if getattr(model, "enable_ring_grow_macro", None) is not False:
        raise EditingT1PanelError(
            "T1 capacity census requires the exact Active8 model with ring growth disabled"
        )
    capabilities = getattr(model, "operator_capabilities", None)
    if capabilities is None or not callable(getattr(capabilities, "fingerprint", None)):
        raise EditingT1PanelError("T1 capacity census model lacks an operator-capability identity")

    rows = _validate_forensics_rows(forensics, source.sidecar)
    admitted_rows, _excluded_rows, _excluded_traces = filter_active8_forensics_rows(
        rows,
        active8_admission,
    )
    records = _record_index(source)
    runtime = de_novo_rewrite_system()
    base_rows: list[dict[str, Any]] = []
    states_by_source: dict[str, MolecularGraph] = {}
    teachers_by_source: dict[str, list[_CapacityTeacherRequest]] = {}
    for row in admitted_rows:
        address = _forensics_trace_address(row, active8_admission)
        exact_trace_key = (
            address.packed_shard_content_sha256,
            address.entry_index,
        )
        try:
            record = records[exact_trace_key]
        except KeyError:
            raise EditingT1PanelError(
                "T1 capacity census forensics row is absent from the exact packed source"
            ) from None
        packed_address = record.corpus_address
        assert packed_address is not None
        state_ref = row["exact_state_ref"]
        progress_index = state_ref["progress_index"]
        if (
            type(progress_index) is not int
            or progress_index < 0
            or progress_index >= record.path.path_length
            or progress_index != row["progress_index"]
            or record.path.path_length != row["path_length"]
            or packed_address.trace_id != address.trace_id
            or Path(str(state_ref["shard_name"])).name != packed_address.packed_shard_name
        ):
            raise EditingT1PanelError(
                "T1 capacity census row disagrees with its exact packed progress address"
            )
        semantic_key = (*exact_trace_key, progress_index)
        if (
            semantic_key not in source.sidecar.semantic_cell_ids
            or source.sidecar.semantic_cell_ids[semantic_key] != row["semantic_cell_id"]
        ):
            raise EditingT1PanelError(
                "T1 capacity census row disagrees with its exact semantic cell"
            )

        state = record.path.state_at(progress_index)
        target = record.path.state_at(progress_index + 1)
        step = record.path.trace.steps[progress_index]
        source_state_sha256 = persistent_slot_state_sha256(state)
        target_state_sha256 = persistent_slot_state_sha256(target)
        action_sha256 = rewrite_action_codec_sha256(
            step.rule_name,
            step.action,
        )
        family = canonical_family(step.rule_name)
        if (
            source_state_sha256 != row["source_state_sha256"]
            or canonical_state_key(state) != row["source_state_key"]
            or canonical_state_key(target) != row["teacher_successor_key"]
            or step.rule_name != row["teacher_rule_name"]
            or family != row["teacher_family"]
            or family not in RINGCORE_EDITING_FAMILIES
            or action_sha256 != row["teacher_action_sha256"]
        ):
            raise EditingT1PanelError(
                "T1 capacity census teacher/source identity disagrees with packed data"
            )
        try:
            replayed = runtime.apply(state, step.rule_name, step.action)
        except Exception as error:
            raise EditingT1PanelError(
                "T1 capacity census teacher action is not executable"
            ) from error
        if persistent_slot_state_sha256(replayed) != target_state_sha256:
            raise EditingT1PanelError(
                "T1 capacity census teacher does not execute to the exact stored successor"
            )

        row_position = len(base_rows)
        states_by_source.setdefault(source_state_sha256, state)
        teachers_by_source.setdefault(source_state_sha256, []).append(
            _CapacityTeacherRequest(
                row_position=row_position,
                target_state_sha256=target_state_sha256,
                action_sha256=action_sha256,
                teacher_successor_key=str(row["teacher_successor_key"]),
            )
        )
        base_rows.append(
            {
                "source_row_index": int(row["row_index"]),
                "source_row_sha256": str(row["row_sha256"]),
                "packed_shard_content_sha256": (address.packed_shard_content_sha256),
                "packed_entry_index": address.entry_index,
                "progress_index": progress_index,
                "teacher_family": family,
                "teacher_action_sha256": action_sha256,
                "teacher_successor_key": str(row["teacher_successor_key"]),
                "teacher_successor_state_sha256": target_state_sha256,
            }
        )

    requests = tuple(
        _CapacitySourceRequest(
            source_state_sha256=source_state_sha256,
            state=state,
            teachers=tuple(teachers_by_source[source_state_sha256]),
        )
        for source_state_sha256, state in states_by_source.items()
    )
    metrics_by_row: list[_CapacityRowMetrics | None] = [None] * len(base_rows)
    for source_metrics in _compile_capacity_sources(
        requests,
        model=model,
        workers=workers,
    ):
        for metrics in source_metrics:
            if metrics_by_row[metrics.row_position] is not None:
                raise EditingT1PanelError(
                    "T1 capacity census compilation repeated a source-row result"
                )
            metrics_by_row[metrics.row_position] = metrics
    if any(metrics is None for metrics in metrics_by_row):
        raise EditingT1PanelError("T1 capacity census compilation omitted a source-row result")
    census_rows: list[dict[str, Any]] = []
    for base_row, metrics in zip(base_rows, metrics_by_row, strict=True):
        assert metrics is not None
        census_rows.append(
            {
                **base_row,
                "raw_legal_mark_count": metrics.raw_legal_mark_count,
                "canonical_successor_count": metrics.canonical_successor_count,
                "teacher_canonical_successor_alias_count": (
                    metrics.teacher_canonical_successor_alias_count
                ),
                "teacher_exact_successor_alias_count": (
                    metrics.teacher_exact_successor_alias_count
                ),
            }
        )

    forensics_artifact_sha256 = forensics.get("artifact_sha256")
    if not _is_sha256(forensics_artifact_sha256):
        raise EditingT1PanelError("T1 capacity census forensics lacks an artifact identity")
    body = {
        "schema": EDITING_T1_CAPACITY_CENSUS_SCHEMA,
        "schema_version": EDITING_T1_CAPACITY_CENSUS_SCHEMA_VERSION,
        "status": EDITING_T1_CAPACITY_CENSUS_STATUS,
        "training_authorized": False,
        "forensics_file_sha256": forensics_file_sha256,
        "forensics_artifact_sha256": forensics_artifact_sha256,
        "gate_zero_runtime_contract_sha256": (gate_zero_runtime_contract_sha256),
        "unified_packed_manifest_sha256": (source.unified_packed_manifest_sha256),
        "representability_overlay_sha256": (source.representability_overlay_sha256),
        **active8_t1_identity(active8_admission),
        "operator_capability_fingerprint": capabilities.fingerprint(),
        "support_time": EDITING_T1_SUPPORT_TIME,
        "implementation": {
            "builder_implementation_sha256": (editing_t1_capacity_census_implementation_sha256()),
            "fiber_compiler_implementation_sha256": (fiber_compiler_implementation_hash()),
        },
        "rows": census_rows,
    }
    return {**body, "census_sha256": _stable_sha256(body)}


def _validate_editing_t1_capacity_census_structure(
    census: Mapping[str, Any],
    *,
    forensics: Mapping[str, Any],
    forensics_file_sha256: str,
    source: FrozenValidationSource,
    active8_admission: Active8TraceAdmission,
    gate_zero_runtime_contract_sha256: str,
) -> Mapping[str, Mapping[str, int | str]]:
    """Validate census structure and all retained immutable identities."""

    payload = _require_exact_fields(
        census,
        _CAPACITY_CENSUS_FIELDS,
        name="T1 capacity census",
    )
    body = {key: value for key, value in payload.items() if key != "census_sha256"}
    if (
        payload["schema"] != EDITING_T1_CAPACITY_CENSUS_SCHEMA
        or payload["schema_version"] != EDITING_T1_CAPACITY_CENSUS_SCHEMA_VERSION
        or payload["status"] != EDITING_T1_CAPACITY_CENSUS_STATUS
        or payload["training_authorized"] is not False
        or payload["forensics_file_sha256"] != forensics_file_sha256
        or payload["forensics_artifact_sha256"] != forensics.get("artifact_sha256")
        or payload["gate_zero_runtime_contract_sha256"] != gate_zero_runtime_contract_sha256
        or payload["unified_packed_manifest_sha256"] != source.unified_packed_manifest_sha256
        or payload["representability_overlay_sha256"] != source.representability_overlay_sha256
        or payload["support_time"] != EDITING_T1_SUPPORT_TIME
        or not _is_sha256(payload["census_sha256"])
        or payload["census_sha256"] != _stable_sha256(body)
    ):
        raise EditingT1PanelError("T1 capacity census identity, source, or self-hash is invalid")
    expected_active8 = dict(active8_t1_identity(active8_admission))
    observed_active8 = {field: payload[field] for field in ACTIVE8_T1_IDENTITY_FIELDS}
    if observed_active8 != expected_active8:
        raise EditingT1PanelError(
            "T1 capacity census is bound to another physical or logical Active8 corpus"
        )
    implementation = _require_exact_fields(
        payload["implementation"],
        _CAPACITY_CENSUS_IMPLEMENTATION_FIELDS,
        name="T1 capacity census implementation",
    )
    if (
        implementation["builder_implementation_sha256"]
        != editing_t1_capacity_census_implementation_sha256()
        or implementation["fiber_compiler_implementation_sha256"]
        != fiber_compiler_implementation_hash()
        or not isinstance(payload["operator_capability_fingerprint"], str)
        or not payload["operator_capability_fingerprint"]
    ):
        raise EditingT1PanelError("T1 capacity census implementation or operator identity drifted")
    source_rows = forensics.get("rows")
    if not isinstance(source_rows, list):
        raise EditingT1PanelError("T1 capacity census source forensics lacks rows")
    admitted_rows, _excluded_rows, _excluded_traces = filter_active8_forensics_rows(
        source_rows, active8_admission
    )
    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != len(admitted_rows):
        raise EditingT1PanelError(
            "T1 capacity census must cover every Active8-admitted forensics row exactly once"
        )
    index: dict[str, Mapping[str, int | str]] = {}
    for expected_row, raw_capacity in zip(
        admitted_rows,
        rows,
        strict=True,
    ):
        capacity = _require_exact_fields(
            raw_capacity,
            _CAPACITY_CENSUS_ROW_FIELDS,
            name="T1 capacity census row",
        )
        row_sha256 = capacity["source_row_sha256"]
        raw_mark_count = capacity["raw_legal_mark_count"]
        successor_count = capacity["canonical_successor_count"]
        canonical_alias_count = capacity["teacher_canonical_successor_alias_count"]
        exact_alias_count = capacity["teacher_exact_successor_alias_count"]
        state_ref = expected_row["exact_state_ref"]
        if (
            capacity["source_row_index"] != expected_row.get("row_index")
            or row_sha256 != expected_row.get("row_sha256")
            or not _is_sha256(row_sha256)
            or capacity["packed_shard_content_sha256"] != state_ref.get("shard_sha256")
            or capacity["packed_entry_index"] != state_ref.get("record_index")
            or capacity["progress_index"] != state_ref.get("progress_index")
            or capacity["teacher_family"] != expected_row.get("teacher_family")
            or capacity["teacher_action_sha256"] != expected_row.get("teacher_action_sha256")
            or capacity["teacher_successor_key"] != expected_row.get("teacher_successor_key")
            or not _is_sha256(capacity["teacher_successor_state_sha256"])
            or type(raw_mark_count) is not int
            or type(successor_count) is not int
            or type(canonical_alias_count) is not int
            or type(exact_alias_count) is not int
            or raw_mark_count <= 0
            or successor_count <= 0
            or successor_count > raw_mark_count
            or canonical_alias_count <= 0
            or canonical_alias_count > raw_mark_count
            or exact_alias_count <= 0
            or exact_alias_count > canonical_alias_count
            or row_sha256 in index
        ):
            raise EditingT1PanelError(
                "T1 capacity census row is off-order, duplicated, or has invalid support counts"
            )
        index[str(row_sha256)] = MappingProxyType(dict(capacity))
    return MappingProxyType(index)


def validate_editing_t1_capacity_census(
    census: Mapping[str, Any],
    *,
    model: Any,
    source: FrozenValidationSource,
    forensics: Mapping[str, Any],
    forensics_file_sha256: str,
    active8_admission: Active8TraceAdmission,
    gate_zero_runtime_contract_sha256: str,
    workers: int = 1,
) -> Mapping[str, Mapping[str, int | str]]:
    """Recompute the production census and reject self-consistent fabrication."""

    index = _validate_editing_t1_capacity_census_structure(
        census,
        forensics=forensics,
        forensics_file_sha256=forensics_file_sha256,
        source=source,
        active8_admission=active8_admission,
        gate_zero_runtime_contract_sha256=(gate_zero_runtime_contract_sha256),
    )
    expected = build_editing_t1_capacity_census(
        model,
        source=source,
        forensics=forensics,
        forensics_file_sha256=forensics_file_sha256,
        active8_admission=active8_admission,
        gate_zero_runtime_contract_sha256=(gate_zero_runtime_contract_sha256),
        workers=workers,
    )
    if dict(census) != expected:
        raise EditingT1PanelError("T1 capacity census disagrees with production re-enumeration")
    return index


def validate_charge_policy_exclusions(
    *,
    audit: Mapping[str, Any],
    audit_file_sha256: str,
    exclusions: Mapping[str, Any],
    exclusions_file_sha256: str,
) -> Mapping[tuple[str, int], str]:
    """Validate the frozen whole-trace exclusion index and return trace IDs."""

    for name, value in (
        ("audit_file_sha256", audit_file_sha256),
        ("exclusions_file_sha256", exclusions_file_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    payload = _require_exact_fields(
        exclusions,
        _CHARGE_EXCLUSION_FIELDS,
        name="charge-policy exclusion payload",
    )
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if (
        payload["schema"] != "compose.data.packed_charge_policy_exclusions"
        or payload["schema_version"] != 1
        or payload["selection"] != "exclude_whole_trace_if_any_consecutive_state_pair_violates"
        or not _is_sha256(payload["source_input_inventory_sha256"])
        or not _is_sha256(payload["charge_policy_audit_implementation_sha256"])
        or not isinstance(payload["violation_type_bit_order"], list)
        or not payload["violation_type_bit_order"]
        or any(
            not isinstance(value, str) or not value for value in payload["violation_type_bit_order"]
        )
        or len(payload["violation_type_bit_order"]) != len(set(payload["violation_type_bit_order"]))
        or payload["payload_sha256"] != _stable_sha256(body)
    ):
        raise EditingT1PanelError(
            "charge-policy exclusion schema, selection, or self-hash is invalid"
        )
    expected_count = payload["excluded_unique_trace_addresses"]
    if type(expected_count) is not int or expected_count < 0:
        raise EditingT1PanelError("charge-policy excluded trace count must be nonnegative")
    shards = payload["shards"]
    if not isinstance(shards, list):
        raise EditingT1PanelError("charge-policy exclusion shards must be a list")
    index: dict[tuple[str, int], str] = {}
    for shard in shards:
        shard = _require_exact_fields(
            shard,
            _CHARGE_EXCLUSION_SHARD_FIELDS,
            name="charge-policy exclusion shard",
        )
        shard_digest = shard["packed_shard_content_sha256"]
        entries = shard["entries"]
        indices = shard["violating_entry_indices"]
        if (
            not _is_sha256(shard_digest)
            or type(shard["excluded_entry_count"]) is not int
            or not isinstance(entries, list)
            or not isinstance(indices, list)
            or shard["excluded_entry_count"] != len(entries)
            or any(type(index_value) is not int or index_value < 0 for index_value in indices)
            or indices != sorted(set(indices))
        ):
            raise EditingT1PanelError("charge-policy exclusion shard is malformed")
        observed_indices: list[int] = []
        for entry in entries:
            entry = _require_exact_fields(
                entry,
                _CHARGE_EXCLUSION_ENTRY_FIELDS,
                name="charge-policy exclusion entry",
            )
            entry_index = entry["entry_index"]
            trace_id = entry["trace_id"]
            violation_type_mask = entry["violation_type_mask"]
            if (
                type(entry_index) is not int
                or entry_index < 0
                or not isinstance(trace_id, str)
                or not trace_id
                or type(violation_type_mask) is not int
                or violation_type_mask <= 0
                or violation_type_mask >= (1 << len(payload["violation_type_bit_order"]))
            ):
                raise EditingT1PanelError(
                    "charge-policy exclusion entry lacks an exact trace identity"
                )
            key = (str(shard_digest), entry_index)
            if key in index:
                raise EditingT1PanelError(
                    "charge-policy exclusion index repeats an exact trace address"
                )
            index[key] = trace_id
            observed_indices.append(entry_index)
        if observed_indices != indices:
            raise EditingT1PanelError(
                "charge-policy exclusion entry identities disagree with their shard index"
            )
    if len(index) != expected_count:
        raise EditingT1PanelError(
            "charge-policy exclusion count disagrees with exact indexed traces"
        )

    audit_exclusions = audit.get("exclusion_payload")
    audit_counts = audit.get("counts")
    audit_implementation = audit.get("implementation")
    if (
        audit.get("schema") != "compose.data.packed_charge_policy_audit"
        or audit.get("schema_version") != 1
        or audit.get("input_inventory_sha256") != payload["source_input_inventory_sha256"]
        or not isinstance(audit_counts, Mapping)
        or audit_counts.get("traces_with_violations") != expected_count
        or not isinstance(audit_implementation, Mapping)
        or audit_implementation.get("implementation_sha256")
        != payload["charge_policy_audit_implementation_sha256"]
        or audit_exclusions != dict(payload)
    ):
        raise EditingT1PanelError("charge-policy audit and standalone exclusion index disagree")
    return MappingProxyType(index)


def _source_row_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    index = row.get("row_index")
    digest = row.get("row_sha256")
    if type(index) is not int or index < 0 or not _is_sha256(digest):
        raise EditingT1PanelError("forensics row lacks a valid row index or self-hash")
    return {
        "source_row_index": index,
        "source_row_sha256": digest,
    }


def _validate_forensics_rows(
    forensics: Mapping[str, Any],
    sidecar: LoadedSemanticCellSidecar,
) -> tuple[Mapping[str, Any], ...]:
    rows = forensics.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise EditingT1PanelError("forensics artifact lacks a row sequence")
    materialized = tuple(rows)
    if not materialized:
        raise EditingT1PanelError("forensics artifact is empty")
    source_families = set(
        (*RINGCORE_EDITING_FAMILIES, *EDITING_T1_OPERATOR_FREEZE_EXCLUDED_FAMILIES)
    )
    sidecar_cells = sidecar.semantic_cell_ids
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise EditingT1PanelError("forensics row is not an object")
        if row.get("row_index") != index:
            raise EditingT1PanelError("forensics rows are not in their frozen index order")
        family = row.get("teacher_family")
        if family not in source_families:
            raise EditingT1PanelError(f"forensics row references an unknown family: {family!r}")
        if row.get("terminal") is not False:
            raise EditingT1PanelError("T1 family forensics may contain nonterminal rows only")
        source_digest = row.get("source_state_sha256")
        source_key = row.get("source_state_key")
        target_key = row.get("teacher_successor_key")
        if (
            not _is_sha256(source_digest)
            or not isinstance(source_key, str)
            or not source_key
            or not isinstance(target_key, str)
            or not target_key
            or source_key == target_key
        ):
            raise EditingT1PanelError(
                "forensics row lacks a productive exact source/target identity"
            )
        state_ref = row.get("exact_state_ref")
        if not isinstance(state_ref, Mapping):
            raise EditingT1PanelError("forensics row lacks an exact packed-state reference")
        exact_key = (
            state_ref.get("shard_sha256"),
            state_ref.get("record_index"),
            state_ref.get("progress_index"),
        )
        if exact_key not in sidecar_cells:
            raise EditingT1PanelError(f"semantic sidecar is missing forensics row {index}")
        if sidecar_cells[exact_key] != row.get("semantic_cell_id"):
            raise EditingT1PanelError(f"semantic sidecar cell disagrees for forensics row {index}")
    return materialized


def _select_unique_panels(
    rows: tuple[Mapping[str, Any], ...],
    *,
    capacity_by_row_sha256: Mapping[str, Mapping[str, int | str]],
    required_high_candidate_families: tuple[str, ...],
    required_aliased_teacher_families: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    targets_by_state: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        targets_by_state[str(row["source_state_sha256"])].add(str(row["teacher_successor_key"]))

    eligible_by_family: dict[str, list[Mapping[str, Any]]] = {
        family: [] for family in RINGCORE_EDITING_FAMILIES
    }
    seen_eligible_states = {family: set() for family in RINGCORE_EDITING_FAMILIES}
    for row in rows:
        family = str(row["teacher_family"])
        source_digest = str(row["source_state_sha256"])
        if (
            source_digest not in seen_eligible_states[family]
            and len(targets_by_state[source_digest]) == 1
        ):
            row_sha256 = str(row["row_sha256"])
            if row_sha256 not in capacity_by_row_sha256:
                raise EditingT1PanelError(
                    "T1 capacity census is missing an eligible unique-state row"
                )
            eligible_by_family[family].append(row)
            seen_eligible_states[family].add(source_digest)

    selected: dict[str, list[dict[str, Any]]] = {family: [] for family in RINGCORE_EDITING_FAMILIES}
    selected_row_sha256s: dict[str, set[str]] = {
        family: set() for family in RINGCORE_EDITING_FAMILIES
    }

    def add(family: str, row: Mapping[str, Any]) -> None:
        row_sha256 = str(row["row_sha256"])
        if row_sha256 not in selected_row_sha256s[family]:
            selected[family].append(_source_row_ref(row))
            selected_row_sha256s[family].add(row_sha256)

    for family in RINGCORE_EDITING_FAMILIES:
        eligible = eligible_by_family[family]
        if family in required_high_candidate_families:
            if not eligible:
                raise EditingT1PanelError(
                    f"registered high-candidate stratum is empty for {family}"
                )
            maximum = max(
                (
                    int(
                        capacity_by_row_sha256[str(row["row_sha256"])]["canonical_successor_count"]
                    ),
                    int(capacity_by_row_sha256[str(row["row_sha256"])]["raw_legal_mark_count"]),
                )
                for row in eligible
            )
            high_candidate = next(
                row
                for row in eligible
                if (
                    int(
                        capacity_by_row_sha256[str(row["row_sha256"])]["canonical_successor_count"]
                    ),
                    int(capacity_by_row_sha256[str(row["row_sha256"])]["raw_legal_mark_count"]),
                )
                == maximum
            )
            add(family, high_candidate)
        if family in required_aliased_teacher_families:
            aliased = next(
                (
                    row
                    for row in eligible
                    if int(
                        capacity_by_row_sha256[str(row["row_sha256"])][
                            "teacher_canonical_successor_alias_count"
                        ]
                    )
                    > 1
                ),
                None,
            )
            if aliased is None:
                raise EditingT1PanelError(
                    f"registered aliased-teacher stratum is empty for {family}"
                )
            add(family, aliased)
        for row in eligible:
            if len(selected[family]) >= EDITING_T1_UNIQUE_EXAMPLES_PER_FAMILY:
                break
            add(family, row)

    shortfalls = {
        family: EDITING_T1_UNIQUE_EXAMPLES_PER_FAMILY - len(panel)
        for family, panel in selected.items()
        if len(panel) != EDITING_T1_UNIQUE_EXAMPLES_PER_FAMILY
    }
    if shortfalls:
        raise EditingT1PanelError(
            f"forensics artifact cannot fill exact unique-state panels: {shortfalls}"
        )
    return selected


def _derived_capacity_strata(
    rows: tuple[Mapping[str, Any], ...],
    *,
    capacity_by_row_sha256: Mapping[str, Mapping[str, int | str]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive mandatory strata from measured pretraining availability."""

    targets_by_state: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        targets_by_state[str(row["source_state_sha256"])].add(str(row["teacher_successor_key"]))
    alias_available: set[str] = set()
    seen_states = {family: set() for family in RINGCORE_EDITING_FAMILIES}
    unique_available = {family: 0 for family in RINGCORE_EDITING_FAMILIES}
    for row in rows:
        family = str(row["teacher_family"])
        source_digest = str(row["source_state_sha256"])
        if len(targets_by_state[source_digest]) != 1 or source_digest in seen_states[family]:
            continue
        seen_states[family].add(source_digest)
        unique_available[family] += 1
        capacity = capacity_by_row_sha256.get(str(row["row_sha256"]))
        if capacity is None:
            raise EditingT1PanelError("T1 capacity census is missing a pilot-eligible exact row")
        if int(capacity["teacher_canonical_successor_alias_count"]) > 1:
            alias_available.add(family)
    missing = [family for family in RINGCORE_EDITING_FAMILIES if unique_available[family] == 0]
    if missing:
        raise EditingT1PanelError(
            "T1 high-candidate strata require an eligible exact state for every "
            f"Active8 family: {missing}"
        )
    return (
        RINGCORE_EDITING_FAMILIES,
        tuple(family for family in RINGCORE_EDITING_FAMILIES if family in alias_available),
    )


def _group_repeated_rows(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    include_family_count: bool,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for source_digest, occurrences in grouped.items():
        target_count = len({str(row["teacher_successor_key"]) for row in occurrences})
        if target_count <= 1:
            continue
        ordered = sorted(
            occurrences,
            key=lambda row: int(row["row_index"]),
        )
        group = {
            "source_state_sha256": source_digest,
            "distinct_teacher_successor_count": target_count,
            "rows": [_source_row_ref(row) for row in ordered],
        }
        if include_family_count:
            group["distinct_teacher_family_count"] = len(
                {str(row["teacher_family"]) for row in occurrences}
            )
        groups.append(group)
    return sorted(
        groups,
        key=lambda group: (
            group["rows"][0]["source_row_index"],
            group["source_state_sha256"],
        ),
    )


def _derive_repeated_panels(
    rows: tuple[Mapping[str, Any], ...],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    global_grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    within_family_grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        family: defaultdict(list) for family in RINGCORE_EDITING_FAMILIES
    }
    for row in rows:
        source_digest = str(row["source_state_sha256"])
        global_grouped[source_digest].append(row)
        within_family_grouped[str(row["teacher_family"])][source_digest].append(row)

    global_panel = _group_repeated_rows(
        global_grouped,
        include_family_count=True,
    )
    conditional_panels: dict[str, list[dict[str, Any]]] = {}
    for family in RINGCORE_EDITING_FAMILIES:
        conditional_panels[family] = _group_repeated_rows(
            within_family_grouped[family],
            include_family_count=False,
        )
    return global_panel, conditional_panels


def _bounded_complete_repeated_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    minimum_rows: int,
    prefer_cross_family: bool,
) -> list[dict[str, Any]]:
    """Select complete empirical state groups within the frozen T1 row bound."""

    eligible = [group for group in groups if len(group["rows"]) <= EDITING_T1_REPEATED_MAX_EXAMPLES]
    selected_keys: set[str] = set()
    selected: list[Mapping[str, Any]] = []
    selected_rows = 0

    def add(group: Mapping[str, Any]) -> None:
        nonlocal selected_rows
        key = str(group["source_state_sha256"])
        row_count = len(group["rows"])
        if (
            key not in selected_keys
            and selected_rows + row_count <= EDITING_T1_REPEATED_MAX_EXAMPLES
        ):
            selected.append(group)
            selected_keys.add(key)
            selected_rows += row_count

    if prefer_cross_family:
        cross_family = next(
            (group for group in eligible if int(group.get("distinct_teacher_family_count", 0)) > 1),
            None,
        )
        if cross_family is None:
            raise EditingT1PanelError(
                "global repeated-state T1 panel lacks a complete cross-family group"
            )
        add(cross_family)
    for group in eligible:
        add(group)
    if selected_rows < minimum_rows:
        raise EditingT1PanelError(
            "complete repeated-state groups cannot satisfy the bounded T1 "
            f"minimum of {minimum_rows} rows"
        )
    return [
        dict(group)
        for group in sorted(
            selected,
            key=lambda group: (
                int(group["rows"][0]["source_row_index"]),
                str(group["source_state_sha256"]),
            ),
        )
    ]


def _select_repeated_panels(
    rows: tuple[Mapping[str, Any], ...],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    available_global, available_conditional = _derive_repeated_panels(rows)
    global_panel = _bounded_complete_repeated_groups(
        available_global,
        minimum_rows=EDITING_T1_REPEATED_MIN_EXAMPLES,
        prefer_cross_family=True,
    )
    conditional_panels = {
        family: _bounded_complete_repeated_groups(
            available_conditional[family],
            minimum_rows=0,
            prefer_cross_family=False,
        )
        for family in RINGCORE_EDITING_FAMILIES
    }
    return global_panel, conditional_panels


def _panel_census(
    unique: Mapping[str, Sequence[Mapping[str, Any]]],
    global_repeated: Sequence[Mapping[str, Any]],
    conditional_repeated: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    pilot_eligible_rows: Sequence[Mapping[str, Any]],
    capacity_by_row_sha256: Mapping[str, Mapping[str, int | str]],
    required_high_candidate_families: tuple[str, ...],
    required_aliased_teacher_families: tuple[str, ...],
    source_row_count: int,
    active8_eligible_row_count: int,
    active8_excluded_row_count: int,
    active8_excluded_trace_count: int,
    charge_eligible_row_count: int,
    charge_excluded_row_count: int,
    pilot_eligible_row_count: int,
    operator_freeze_excluded_row_count: int,
    charge_excluded_forensics_trace_count: int,
) -> dict[str, Any]:
    available_global, available_conditional = _derive_repeated_panels(tuple(pilot_eligible_rows))
    available_global_rows = sum(len(group["rows"]) for group in available_global)
    available_conditional_group_counts = {
        family: len(available_conditional[family]) for family in RINGCORE_EDITING_FAMILIES
    }
    available_conditional_row_counts = {
        family: sum(len(group["rows"]) for group in available_conditional[family])
        for family in RINGCORE_EDITING_FAMILIES
    }
    conditional_group_counts = {
        family: len(conditional_repeated[family]) for family in RINGCORE_EDITING_FAMILIES
    }
    conditional_row_counts = {
        family: sum(len(group["rows"]) for group in conditional_repeated[family])
        for family in RINGCORE_EDITING_FAMILIES
    }
    global_row_total = sum(len(group["rows"]) for group in global_repeated)
    cross_family_groups = [
        group for group in global_repeated if int(group["distinct_teacher_family_count"]) > 1
    ]
    unique_references = [
        reference for family in RINGCORE_EDITING_FAMILIES for reference in unique[family]
    ]
    global_references = [reference for group in global_repeated for reference in group["rows"]]
    conditional_references = [
        reference
        for family in RINGCORE_EDITING_FAMILIES
        for group in conditional_repeated[family]
        for reference in group["rows"]
    ]

    def identity_sets(
        references: Sequence[Mapping[str, Any]],
    ) -> tuple[set[str], set[tuple[str, int, int]], set[tuple[str, int]], set[tuple[int, str]]]:
        selected_rows = [
            source_rows[int(reference["source_row_index"])] for reference in references
        ]
        return (
            {str(row["source_state_sha256"]) for row in selected_rows},
            {
                (
                    str(row["exact_state_ref"]["shard_sha256"]),
                    int(row["exact_state_ref"]["record_index"]),
                    int(row["exact_state_ref"]["progress_index"]),
                )
                for row in selected_rows
            },
            {
                (
                    str(row["exact_state_ref"]["shard_sha256"]),
                    int(row["exact_state_ref"]["record_index"]),
                )
                for row in selected_rows
            },
            {
                (
                    int(reference["source_row_index"]),
                    str(reference["source_row_sha256"]),
                )
                for reference in references
            },
        )

    unique_identities = identity_sets(unique_references)
    global_identities = identity_sets(global_references)
    conditional_identities = identity_sets(conditional_references)

    targets_by_state: dict[str, set[str]] = defaultdict(set)
    for row in pilot_eligible_rows:
        targets_by_state[str(row["source_state_sha256"])].add(str(row["teacher_successor_key"]))
    eligible_unique_by_family: dict[str, list[Mapping[str, Any]]] = {
        family: [] for family in RINGCORE_EDITING_FAMILIES
    }
    seen_states: dict[str, set[str]] = {family: set() for family in RINGCORE_EDITING_FAMILIES}
    for row in pilot_eligible_rows:
        family = str(row["teacher_family"])
        source_digest = str(row["source_state_sha256"])
        if len(targets_by_state[source_digest]) == 1 and source_digest not in seen_states[family]:
            eligible_unique_by_family[family].append(row)
            seen_states[family].add(source_digest)
    selected_unique_rows = {
        family: [source_rows[int(reference["source_row_index"])] for reference in unique[family]]
        for family in RINGCORE_EDITING_FAMILIES
    }

    def capacity_key(row: Mapping[str, Any]) -> tuple[int, int]:
        capacity = capacity_by_row_sha256[str(row["row_sha256"])]
        return (
            int(capacity["canonical_successor_count"]),
            int(capacity["raw_legal_mark_count"]),
        )

    high_candidate_available: dict[str, int] = {}
    high_candidate_selected: dict[str, int] = {}
    aliased_available: dict[str, int] = {}
    aliased_selected: dict[str, int] = {}
    for family in RINGCORE_EDITING_FAMILIES:
        eligible = eligible_unique_by_family[family]
        maximum = max((capacity_key(row) for row in eligible), default=None)
        high_candidate_available[family] = sum(capacity_key(row) == maximum for row in eligible)
        high_candidate_selected[family] = sum(
            capacity_key(row) == maximum for row in selected_unique_rows[family]
        )
        aliased_available[family] = sum(
            int(
                capacity_by_row_sha256[str(row["row_sha256"])][
                    "teacher_canonical_successor_alias_count"
                ]
            )
            > 1
            for row in eligible
        )
        aliased_selected[family] = sum(
            int(
                capacity_by_row_sha256[str(row["row_sha256"])][
                    "teacher_canonical_successor_alias_count"
                ]
            )
            > 1
            for row in selected_unique_rows[family]
        )

    def intersection_counts(
        left: tuple[
            set[str],
            set[tuple[str, int, int]],
            set[tuple[str, int]],
            set[tuple[int, str]],
        ],
        right: tuple[
            set[str],
            set[tuple[str, int, int]],
            set[tuple[str, int]],
            set[tuple[int, str]],
        ],
    ) -> dict[str, int]:
        return {
            "shared_source_states": len(left[0] & right[0]),
            "shared_progress_addresses": len(left[1] & right[1]),
            "shared_trace_addresses": len(left[2] & right[2]),
            "shared_source_row_references": len(left[3] & right[3]),
        }

    return {
        "source_forensics_rows_total": source_row_count,
        "active8_eligible_forensics_rows_total": active8_eligible_row_count,
        "active8_excluded_forensics_rows_total": active8_excluded_row_count,
        "active8_excluded_unique_forensics_trace_addresses": (active8_excluded_trace_count),
        "charge_policy_eligible_forensics_rows_total": charge_eligible_row_count,
        "charge_policy_excluded_forensics_rows_total": charge_excluded_row_count,
        "charge_policy_excluded_unique_forensics_trace_addresses": (
            charge_excluded_forensics_trace_count
        ),
        "pilot_family_eligible_forensics_rows_total": pilot_eligible_row_count,
        "operator_freeze_excluded_forensics_rows_total": operator_freeze_excluded_row_count,
        "unique_rows_by_family": {
            family: len(unique[family]) for family in RINGCORE_EDITING_FAMILIES
        },
        "unique_rows_total": sum(len(unique[family]) for family in unique),
        "unique_high_candidate_stratum_available_by_family": (high_candidate_available),
        "unique_high_candidate_stratum_selected_by_family": (high_candidate_selected),
        "unique_aliased_teacher_stratum_available_by_family": (aliased_available),
        "unique_aliased_teacher_stratum_selected_by_family": (aliased_selected),
        "registered_high_candidate_families_all_covered": all(
            high_candidate_selected[family] >= 1 for family in required_high_candidate_families
        ),
        "registered_aliased_teacher_families_all_covered": all(
            aliased_selected[family] >= 1 for family in required_aliased_teacher_families
        ),
        "global_repeated_multitarget_groups_total": len(global_repeated),
        "global_repeated_rows_total": global_row_total,
        "global_repeated_available_multitarget_groups_total": len(available_global),
        "global_repeated_available_rows_total": available_global_rows,
        "global_repeated_cross_family_groups_total": len(cross_family_groups),
        "global_repeated_cross_family_rows_total": sum(
            len(group["rows"]) for group in cross_family_groups
        ),
        "global_repeated_panel_meets_64_example_minimum": (
            global_row_total >= EDITING_T1_REPEATED_MIN_EXAMPLES
        ),
        "global_repeated_panel_meets_128_example_maximum": (
            global_row_total <= EDITING_T1_REPEATED_MAX_EXAMPLES
        ),
        "within_family_repeated_multitarget_groups_by_family": (conditional_group_counts),
        "within_family_repeated_rows_by_family": conditional_row_counts,
        "within_family_repeated_available_multitarget_groups_by_family": (
            available_conditional_group_counts
        ),
        "within_family_repeated_available_rows_by_family": (available_conditional_row_counts),
        "within_family_repeated_multitarget_groups_total": sum(conditional_group_counts.values()),
        "within_family_repeated_rows_total": sum(conditional_row_counts.values()),
        "families_without_observed_within_family_multitarget_repeats": [
            family for family in RINGCORE_EDITING_FAMILIES if conditional_group_counts[family] == 0
        ],
        "panel_overlap": {
            "unique_state_vs_global_repeated": intersection_counts(
                unique_identities,
                global_identities,
            ),
            "unique_state_vs_within_family_repeated": intersection_counts(
                unique_identities,
                conditional_identities,
            ),
            "global_repeated_vs_within_family_repeated": intersection_counts(
                global_identities,
                conditional_identities,
            ),
            "within_family_repeated_is_exact_row_subset_of_global_repeated": (
                conditional_identities[3] <= global_identities[3]
            ),
            "statistically_independent_panel_claim": False,
        },
    }


def _artifact_body(
    *,
    source: Mapping[str, Any],
    unique: Mapping[str, Sequence[Mapping[str, Any]]],
    global_repeated: Sequence[Mapping[str, Any]],
    conditional_repeated: Mapping[str, Sequence[Mapping[str, Any]]],
    source_rows: Sequence[Mapping[str, Any]],
    pilot_eligible_rows: Sequence[Mapping[str, Any]],
    capacity_by_row_sha256: Mapping[str, Mapping[str, int | str]],
    required_high_candidate_families: tuple[str, ...],
    required_aliased_teacher_families: tuple[str, ...],
    source_row_count: int,
    active8_eligible_row_count: int,
    active8_excluded_row_count: int,
    active8_excluded_trace_count: int,
    charge_eligible_row_count: int,
    charge_excluded_row_count: int,
    pilot_eligible_row_count: int,
    operator_freeze_excluded_row_count: int,
    charge_excluded_forensics_trace_count: int,
) -> dict[str, Any]:
    return {
        "schema": EDITING_T1_PANEL_SCHEMA,
        "schema_version": EDITING_T1_PANEL_SCHEMA_VERSION,
        "status": EDITING_T1_PANEL_STATUS,
        "training_authorized": False,
        "source": dict(source),
        "selection": {
            "algorithm": EDITING_T1_SELECTION_ALGORITHM,
            "active_families": list(RINGCORE_EDITING_FAMILIES),
            "diagnostic_only_families": list(EDITING_T1_DIAGNOSTIC_ONLY_FAMILIES),
            "operator_freeze_excluded_families": list(EDITING_T1_OPERATOR_FREEZE_EXCLUDED_FAMILIES),
            "unique_examples_per_family": (EDITING_T1_UNIQUE_EXAMPLES_PER_FAMILY),
            "required_high_candidate_families": list(required_high_candidate_families),
            "required_aliased_teacher_families": list(required_aliased_teacher_families),
            "high_candidate_definition": (
                "per-family lexicographic maximum of "
                "(canonical_successor_count, raw_legal_mark_count) among "
                "globally deterministic unique exact-state candidates after "
                "Active8, charge-policy, and operator-freeze admission"
            ),
            "aliased_teacher_definition": (
                "teacher_canonical_successor_alias_count greater than one among "
                "measured globally deterministic exact-state candidates"
            ),
            "global_determinism_policy": (
                "after whole-trace Active8 admission, operator-freeze, and "
                "charge-policy exclusions, unique family panels exclude an exact "
                "source whenever the full active eight-family pilot projection "
                "records more than one distinct canonical teacher successor"
            ),
            "primary_repeated_state_policy": (
                "deterministic complete-group selection from exact persistent-slot "
                "sources with more than one distinct canonical teacher successor under "
                "the active eight-family pilot law; at least one cross-family group and "
                "64 to 128 observations are required; "
                "operator-freeze-excluded families are omitted before grouping; every "
                "observation and importance weight in a selected state group is preserved; "
                "groups are never truncated and no observation is synthesized"
            ),
            "conditional_repeated_state_policy": (
                "secondary diagnostic views select complete within-family state groups "
                "under the same 128-observation maximum; they are not the primary T1 "
                "successor law"
            ),
            "conditional_repeated_state_relationship": (
                "global and within-family panels are bounded deterministic views of the "
                "same frozen observation pool; they may overlap but must not be treated "
                "as statistically independent"
            ),
            "operator_freeze_exclusion_policy": (
                "exclude every row whose recorded teacher family is ring_system_delete "
                "before unique or repeated-state selection; the superseded source "
                "forensics remains immutable and is counted explicitly"
            ),
            "charge_policy_exclusion_policy": (
                "exclude a complete packed trace before panel selection whenever "
                "the frozen exact-state charge-policy index contains its "
                "(packed_shard_content_sha256, entry_index); selected addresses "
                "are checked again against indexed trace_id at runtime"
            ),
            "active8_whole_trace_admission_policy": (
                "load and physically hash the immutable Active8 inventory and every "
                "decision shard, then require the exact "
                "(packed_shard_content_sha256, entry_index, trace_id) decision before "
                "any T1 selection; if any middle action is excluded, no neighboring "
                "row from that trace may enter any T1 panel"
            ),
            "support_time": EDITING_T1_SUPPORT_TIME,
        },
        "architecture": {
            "hidden_dim": 256,
            "message_passing_steps": 6,
            "mark_dim": 32,
            "atom_vocabulary_class_count": 15,
            "enable_ring_restates": True,
            "enable_cyclic_graft": True,
            "enable_heteroatom_scan": True,
            "enable_ring_opening": True,
            "enable_cycle_ops": True,
            "enable_ring_grow_macro": False,
            "enable_ring_system_delete": False,
            "required_scope_ladder": list(EDITING_T1_REQUIRED_SCOPES),
        },
        "thresholds": {
            "minimum_unique_state_teacher_successor_top1": None,
            "minimum_unique_state_teacher_successor_probability": None,
            "maximum_unique_state_teacher_successor_nll": None,
            "maximum_global_repeated_state_excess_nll_over_empirical_entropy": None,
        },
        "unique_state_panels": {
            family: list(unique[family]) for family in RINGCORE_EDITING_FAMILIES
        },
        "global_repeated_state_panel": list(global_repeated),
        "within_family_repeated_state_panels": {
            family: list(conditional_repeated[family]) for family in RINGCORE_EDITING_FAMILIES
        },
        "census": _panel_census(
            unique,
            global_repeated,
            conditional_repeated,
            source_rows=source_rows,
            pilot_eligible_rows=pilot_eligible_rows,
            capacity_by_row_sha256=capacity_by_row_sha256,
            required_high_candidate_families=(required_high_candidate_families),
            required_aliased_teacher_families=(required_aliased_teacher_families),
            source_row_count=source_row_count,
            active8_eligible_row_count=active8_eligible_row_count,
            active8_excluded_row_count=active8_excluded_row_count,
            active8_excluded_trace_count=active8_excluded_trace_count,
            charge_eligible_row_count=charge_eligible_row_count,
            charge_excluded_row_count=charge_excluded_row_count,
            pilot_eligible_row_count=pilot_eligible_row_count,
            operator_freeze_excluded_row_count=operator_freeze_excluded_row_count,
            charge_excluded_forensics_trace_count=(charge_excluded_forensics_trace_count),
        ),
    }


def build_editing_t1_panel(
    *,
    model: Any,
    source: FrozenValidationSource,
    forensics: Mapping[str, Any],
    forensics_file_sha256: str,
    sidecar: LoadedSemanticCellSidecar,
    semantic_sidecar_file_sha256: str,
    semantic_sidecar_manifest_file_sha256: str,
    charge_policy_audit: Mapping[str, Any],
    charge_policy_audit_file_sha256: str,
    charge_policy_exclusions: Mapping[str, Any],
    charge_policy_exclusions_file_sha256: str,
    capacity_census: Mapping[str, Any],
    capacity_census_file_sha256: str,
    active8_admission: Active8TraceAdmission,
    gate_zero_runtime_contract_sha256: str,
    gate_zero_unified_packed_manifest_sha256: str,
    capacity_workers: int = 1,
) -> dict[str, Any]:
    """Derive the exact T1 panels from validated frozen inputs."""

    for name, value in (
        ("forensics_file_sha256", forensics_file_sha256),
        ("semantic_sidecar_file_sha256", semantic_sidecar_file_sha256),
        (
            "semantic_sidecar_manifest_file_sha256",
            semantic_sidecar_manifest_file_sha256,
        ),
        ("charge_policy_audit_file_sha256", charge_policy_audit_file_sha256),
        (
            "charge_policy_exclusions_file_sha256",
            charge_policy_exclusions_file_sha256,
        ),
        (
            "gate_zero_runtime_contract_sha256",
            gate_zero_runtime_contract_sha256,
        ),
        (
            "gate_zero_unified_packed_manifest_sha256",
            gate_zero_unified_packed_manifest_sha256,
        ),
        ("capacity_census_file_sha256", capacity_census_file_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    forensics_artifact_sha256 = forensics.get("artifact_sha256")
    leaderboard_protocol_sha256 = forensics.get("leaderboard_protocol_sha256")
    frozen_inventory_sha256 = forensics.get("frozen_inventory_sha256")
    for name, value in (
        ("forensics_artifact_sha256", forensics_artifact_sha256),
        ("leaderboard_protocol_sha256", leaderboard_protocol_sha256),
        ("frozen_inventory_sha256", frozen_inventory_sha256),
    ):
        if not _is_sha256(value):
            raise EditingT1PanelError(f"forensics source lacks {name}")
    manifest = sidecar.manifest
    provenance = manifest.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("unified_packed_manifest_sha256")
        != gate_zero_unified_packed_manifest_sha256
        or active8_admission.unified_packed_manifest_sha256
        != gate_zero_unified_packed_manifest_sha256
    ):
        raise EditingT1PanelError(
            "Gate0 validation source, semantic sidecar, and Active8 inventory "
            "name different unified packed manifests"
        )
    rows = _validate_forensics_rows(forensics, sidecar)
    (
        active8_eligible_rows,
        active8_excluded_row_count,
        active8_excluded_trace_count,
    ) = filter_active8_forensics_rows(rows, active8_admission)
    capacity_by_row_sha256 = validate_editing_t1_capacity_census(
        capacity_census,
        model=model,
        source=source,
        forensics=forensics,
        forensics_file_sha256=forensics_file_sha256,
        active8_admission=active8_admission,
        gate_zero_runtime_contract_sha256=(gate_zero_runtime_contract_sha256),
        workers=capacity_workers,
    )
    excluded_trace_ids = validate_charge_policy_exclusions(
        audit=charge_policy_audit,
        audit_file_sha256=charge_policy_audit_file_sha256,
        exclusions=charge_policy_exclusions,
        exclusions_file_sha256=charge_policy_exclusions_file_sha256,
    )
    charge_eligible_rows = tuple(
        row
        for row in active8_eligible_rows
        if (
            str(row["exact_state_ref"]["shard_sha256"]),
            int(row["exact_state_ref"]["record_index"]),
        )
        not in excluded_trace_ids
    )
    active_families = set(RINGCORE_EDITING_FAMILIES)
    eligible_rows = tuple(
        row for row in charge_eligible_rows if str(row["teacher_family"]) in active_families
    )
    operator_freeze_excluded_row_count = sum(
        str(row["teacher_family"]) in EDITING_T1_OPERATOR_FREEZE_EXCLUDED_FAMILIES
        for row in charge_eligible_rows
    )
    excluded_forensics_trace_count = len(
        {
            (
                str(row["exact_state_ref"]["shard_sha256"]),
                int(row["exact_state_ref"]["record_index"]),
            )
            for row in active8_eligible_rows
            if (
                str(row["exact_state_ref"]["shard_sha256"]),
                int(row["exact_state_ref"]["record_index"]),
            )
            in excluded_trace_ids
        }
    )
    high_candidate_families, aliased_teacher_families = _derived_capacity_strata(
        eligible_rows,
        capacity_by_row_sha256=capacity_by_row_sha256,
    )
    unique = _select_unique_panels(
        eligible_rows,
        capacity_by_row_sha256=capacity_by_row_sha256,
        required_high_candidate_families=high_candidate_families,
        required_aliased_teacher_families=aliased_teacher_families,
    )
    global_repeated, conditional_repeated = _select_repeated_panels(eligible_rows)
    source = {
        "forensics_file_sha256": forensics_file_sha256,
        "forensics_artifact_sha256": forensics_artifact_sha256,
        "leaderboard_protocol_sha256": leaderboard_protocol_sha256,
        "frozen_inventory_sha256": frozen_inventory_sha256,
        "semantic_sidecar_file_sha256": semantic_sidecar_file_sha256,
        "semantic_sidecar_manifest_file_sha256": (semantic_sidecar_manifest_file_sha256),
        "semantic_sidecar_manifest_sha256": manifest["manifest_sha256"],
        "semantic_sidecar_config_sha256": manifest["config_sha256"],
        "semantic_sidecar_provenance_sha256": (manifest["provenance_sha256"]),
        "charge_policy_audit_file_sha256": charge_policy_audit_file_sha256,
        "charge_policy_exclusions_file_sha256": (charge_policy_exclusions_file_sha256),
        "charge_policy_exclusion_payload_sha256": charge_policy_exclusions["payload_sha256"],
        "charge_policy_source_input_inventory_sha256": (
            charge_policy_exclusions["source_input_inventory_sha256"]
        ),
        "charge_policy_audit_implementation_sha256": (
            charge_policy_exclusions["charge_policy_audit_implementation_sha256"]
        ),
        "charge_policy_excluded_unique_trace_addresses": (
            charge_policy_exclusions["excluded_unique_trace_addresses"]
        ),
        "capacity_census_file_sha256": capacity_census_file_sha256,
        "capacity_census_sha256": capacity_census["census_sha256"],
        **active8_t1_identity(active8_admission),
    }
    body = _artifact_body(
        source=source,
        unique=unique,
        global_repeated=global_repeated,
        conditional_repeated=conditional_repeated,
        source_rows=rows,
        pilot_eligible_rows=eligible_rows,
        capacity_by_row_sha256=capacity_by_row_sha256,
        required_high_candidate_families=high_candidate_families,
        required_aliased_teacher_families=aliased_teacher_families,
        source_row_count=len(rows),
        active8_eligible_row_count=len(active8_eligible_rows),
        active8_excluded_row_count=active8_excluded_row_count,
        active8_excluded_trace_count=active8_excluded_trace_count,
        charge_eligible_row_count=len(charge_eligible_rows),
        charge_excluded_row_count=(len(active8_eligible_rows) - len(charge_eligible_rows)),
        pilot_eligible_row_count=len(eligible_rows),
        operator_freeze_excluded_row_count=operator_freeze_excluded_row_count,
        charge_excluded_forensics_trace_count=excluded_forensics_trace_count,
    )
    return {**body, "artifact_sha256": _stable_sha256(body)}


def _validate_row_ref(
    value: object,
    *,
    source_rows: tuple[Mapping[str, Any], ...],
) -> None:
    payload = _require_exact_fields(
        value,
        _ROW_REF_FIELDS,
        name="T1 source-row reference",
    )
    index = payload["source_row_index"]
    digest = payload["source_row_sha256"]
    if (
        type(index) is not int
        or not 0 <= index < len(source_rows)
        or not _is_sha256(digest)
        or source_rows[index].get("row_sha256") != digest
    ):
        raise EditingT1PanelError("T1 source-row reference does not resolve exactly")


def validate_editing_t1_panel(
    panel: Mapping[str, Any],
    *,
    model: Any,
    source: FrozenValidationSource,
    forensics: Mapping[str, Any],
    forensics_file_sha256: str,
    sidecar: LoadedSemanticCellSidecar,
    semantic_sidecar_file_sha256: str,
    semantic_sidecar_manifest_file_sha256: str,
    charge_policy_audit: Mapping[str, Any],
    charge_policy_audit_file_sha256: str,
    charge_policy_exclusions: Mapping[str, Any],
    charge_policy_exclusions_file_sha256: str,
    capacity_census: Mapping[str, Any],
    capacity_census_file_sha256: str,
    active8_admission: Active8TraceAdmission,
    gate_zero_runtime_contract_sha256: str,
    gate_zero_unified_packed_manifest_sha256: str,
) -> None:
    """Validate the artifact and re-derive its complete selection."""

    payload = _require_exact_fields(
        panel,
        _TOP_LEVEL_FIELDS,
        name="T1 panel",
    )
    body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if (
        payload["schema"] != EDITING_T1_PANEL_SCHEMA
        or payload["schema_version"] != EDITING_T1_PANEL_SCHEMA_VERSION
        or payload["status"] != EDITING_T1_PANEL_STATUS
        or payload["training_authorized"] is not False
        or not _is_sha256(payload["artifact_sha256"])
        or payload["artifact_sha256"] != _stable_sha256(body)
    ):
        raise EditingT1PanelError("T1 panel schema, status, or self-hash is invalid")
    _require_exact_fields(payload["source"], _SOURCE_FIELDS, name="T1 source")
    selection = _require_exact_fields(
        payload["selection"],
        _SELECTION_FIELDS,
        name="T1 selection",
    )
    if tuple(selection["required_high_candidate_families"]) != (RINGCORE_EDITING_FAMILIES):
        raise EditingT1PanelError(
            "T1 panel must cover high-candidate strata for every Active8 family"
        )
    _validate_registered_family_subset(
        selection["required_aliased_teacher_families"],
        name="required_aliased_teacher_families",
        allow_empty=True,
    )
    _require_exact_fields(
        payload["architecture"],
        _ARCHITECTURE_FIELDS,
        name="T1 architecture",
    )
    thresholds = _require_exact_fields(
        payload["thresholds"],
        _THRESHOLD_FIELDS,
        name="T1 thresholds",
    )
    if any(value is not None for value in thresholds.values()):
        raise EditingT1PanelError("T1 numeric thresholds must remain explicitly unfrozen")
    source_rows = _validate_forensics_rows(forensics, sidecar)
    for family in RINGCORE_EDITING_FAMILIES:
        unique = payload["unique_state_panels"].get(family)
        conditional = payload["within_family_repeated_state_panels"].get(family)
        if not isinstance(unique, list) or not isinstance(conditional, list):
            raise EditingT1PanelError(f"T1 panel is missing family {family!r}")
        for row_ref in unique:
            _validate_row_ref(row_ref, source_rows=source_rows)
        for group in conditional:
            group = _require_exact_fields(
                group,
                _WITHIN_FAMILY_REPEATED_GROUP_FIELDS,
                name="T1 within-family repeated-state group",
            )
            if not _is_sha256(group["source_state_sha256"]):
                raise EditingT1PanelError("T1 repeated group source digest is invalid")
            if (
                type(group["distinct_teacher_successor_count"]) is not int
                or group["distinct_teacher_successor_count"] <= 1
                or not isinstance(group["rows"], list)
                or len(group["rows"]) < 2
            ):
                raise EditingT1PanelError("T1 repeated-state group is not multi-successor")
            for row_ref in group["rows"]:
                _validate_row_ref(row_ref, source_rows=source_rows)
        if sum(len(group["rows"]) for group in conditional) > (EDITING_T1_REPEATED_MAX_EXAMPLES):
            raise EditingT1PanelError("T1 within-family repeated-state panel exceeds its row bound")
    global_repeated = payload["global_repeated_state_panel"]
    if not isinstance(global_repeated, list):
        raise EditingT1PanelError("T1 global repeated-state panel must be a list")
    for group in global_repeated:
        group = _require_exact_fields(
            group,
            _GLOBAL_REPEATED_GROUP_FIELDS,
            name="T1 global repeated-state group",
        )
        if (
            not _is_sha256(group["source_state_sha256"])
            or type(group["distinct_teacher_successor_count"]) is not int
            or group["distinct_teacher_successor_count"] <= 1
            or type(group["distinct_teacher_family_count"]) is not int
            or group["distinct_teacher_family_count"] <= 0
            or not isinstance(group["rows"], list)
            or len(group["rows"]) < 2
        ):
            raise EditingT1PanelError("T1 global repeated-state group is invalid")
        for row_ref in group["rows"]:
            _validate_row_ref(row_ref, source_rows=source_rows)
    global_repeated_row_count = sum(len(group["rows"]) for group in global_repeated)
    if not (
        EDITING_T1_REPEATED_MIN_EXAMPLES
        <= global_repeated_row_count
        <= EDITING_T1_REPEATED_MAX_EXAMPLES
    ):
        raise EditingT1PanelError("T1 global repeated-state panel violates its 64-to-128 row bound")
    if set(payload["unique_state_panels"]) != set(RINGCORE_EDITING_FAMILIES) or set(
        payload["within_family_repeated_state_panels"]
    ) != set(RINGCORE_EDITING_FAMILIES):
        raise EditingT1PanelError("T1 panel family maps have extra or missing families")

    expected = build_editing_t1_panel(
        model=model,
        source=source,
        forensics=forensics,
        forensics_file_sha256=forensics_file_sha256,
        sidecar=sidecar,
        semantic_sidecar_file_sha256=semantic_sidecar_file_sha256,
        semantic_sidecar_manifest_file_sha256=(semantic_sidecar_manifest_file_sha256),
        charge_policy_audit=charge_policy_audit,
        charge_policy_audit_file_sha256=charge_policy_audit_file_sha256,
        charge_policy_exclusions=charge_policy_exclusions,
        charge_policy_exclusions_file_sha256=(charge_policy_exclusions_file_sha256),
        capacity_census=capacity_census,
        capacity_census_file_sha256=capacity_census_file_sha256,
        active8_admission=active8_admission,
        gate_zero_runtime_contract_sha256=(gate_zero_runtime_contract_sha256),
        gate_zero_unified_packed_manifest_sha256=(gate_zero_unified_packed_manifest_sha256),
    )
    if dict(payload) != expected:
        raise EditingT1PanelError("T1 panel does not equal the deterministic source re-derivation")


def load_editing_t1_panel(
    path: Path,
    *,
    expected_artifact_sha256: str,
    model: Any,
    source: FrozenValidationSource,
    forensics: Mapping[str, Any],
    forensics_file_sha256: str,
    sidecar: LoadedSemanticCellSidecar,
    semantic_sidecar_file_sha256: str,
    semantic_sidecar_manifest_file_sha256: str,
    charge_policy_audit: Mapping[str, Any],
    charge_policy_audit_file_sha256: str,
    charge_policy_exclusions: Mapping[str, Any],
    charge_policy_exclusions_file_sha256: str,
    capacity_census: Mapping[str, Any],
    capacity_census_file_sha256: str,
    active8_admission: Active8TraceAdmission,
    gate_zero_runtime_contract_sha256: str,
    gate_zero_unified_packed_manifest_sha256: str,
    max_file_bytes: int = 1 << 24,
) -> Mapping[str, Any]:
    """Load a self-hashed T1 artifact and bind it to every exact source."""

    if not _is_sha256(expected_artifact_sha256):
        raise ValueError("expected_artifact_sha256 must be a lowercase SHA-256")
    panel_path = Path(path)
    if not panel_path.is_file() or panel_path.stat().st_size > max_file_bytes:
        raise EditingT1PanelError("T1 panel is absent or exceeds its bound")
    try:
        panel = json.loads(panel_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingT1PanelError("T1 panel is invalid JSON") from error
    validate_editing_t1_panel(
        panel,
        model=model,
        source=source,
        forensics=forensics,
        forensics_file_sha256=forensics_file_sha256,
        sidecar=sidecar,
        semantic_sidecar_file_sha256=semantic_sidecar_file_sha256,
        semantic_sidecar_manifest_file_sha256=(semantic_sidecar_manifest_file_sha256),
        charge_policy_audit=charge_policy_audit,
        charge_policy_audit_file_sha256=charge_policy_audit_file_sha256,
        charge_policy_exclusions=charge_policy_exclusions,
        charge_policy_exclusions_file_sha256=(charge_policy_exclusions_file_sha256),
        capacity_census=capacity_census,
        capacity_census_file_sha256=capacity_census_file_sha256,
        active8_admission=active8_admission,
        gate_zero_runtime_contract_sha256=(gate_zero_runtime_contract_sha256),
        gate_zero_unified_packed_manifest_sha256=(gate_zero_unified_packed_manifest_sha256),
    )
    if panel["artifact_sha256"] != expected_artifact_sha256:
        raise EditingT1PanelError("T1 panel artifact hash mismatch")
    return MappingProxyType(panel)


def within_family_repeated_panel_families(
    panel: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return families whose rebuilt panel has observed repeated-state rows."""

    panels = panel.get("within_family_repeated_state_panels")
    if not isinstance(panels, Mapping) or set(panels) != set(RINGCORE_EDITING_FAMILIES):
        raise EditingT1PanelError("T1 panel repeated-state family map is missing or off-contract")
    available: list[str] = []
    for family in RINGCORE_EDITING_FAMILIES:
        groups = panels[family]
        if not isinstance(groups, list):
            raise EditingT1PanelError(f"T1 repeated-state panel for {family} is not a list")
        if groups:
            available.append(family)
    return tuple(available)


def editing_t1_panel_selection_sha256(panel: Mapping[str, Any]) -> str:
    selection = panel.get("selection")
    if not isinstance(selection, Mapping):
        raise EditingT1PanelError("T1 panel lacks a selection object")
    return _stable_sha256(dict(selection))


def editing_t1_panel_census_sha256(panel: Mapping[str, Any]) -> str:
    census = panel.get("census")
    if not isinstance(census, Mapping):
        raise EditingT1PanelError("T1 panel lacks a census object")
    return _stable_sha256(dict(census))


def editing_t1_panel_capacity_strata_sha256(
    panel: Mapping[str, Any],
) -> str:
    """Bind measured capacity availability and mandatory selected coverage."""

    selection = panel.get("selection")
    census = panel.get("census")
    if not isinstance(selection, Mapping) or not isinstance(census, Mapping):
        raise EditingT1PanelError("T1 panel lacks capacity-stratum selection or census")
    fields = (
        "unique_high_candidate_stratum_available_by_family",
        "unique_high_candidate_stratum_selected_by_family",
        "unique_aliased_teacher_stratum_available_by_family",
        "unique_aliased_teacher_stratum_selected_by_family",
        "registered_high_candidate_families_all_covered",
        "registered_aliased_teacher_families_all_covered",
        "families_without_observed_within_family_multitarget_repeats",
    )
    if any(field not in census for field in fields):
        raise EditingT1PanelError("T1 panel capacity-stratum census is incomplete")
    return _stable_sha256(
        {
            "required_high_candidate_families": selection.get("required_high_candidate_families"),
            "required_aliased_teacher_families": selection.get("required_aliased_teacher_families"),
            **{field: census[field] for field in fields},
        }
    )


def selected_source_rows(
    panel: Mapping[str, Any],
    forensics: Mapping[str, Any],
    *,
    family: str,
    panel_kind: str,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve one exact family/role panel after artifact validation."""

    rows = tuple(forensics["rows"])
    if panel_kind == EDITING_T1_GLOBAL_REPEATED_PANEL_KIND:
        if family != EDITING_T1_GLOBAL_FAMILY_SELECTOR:
            raise ValueError(
                "the primary global repeated-state panel requires family='all_families'"
            )
        references = [
            reference
            for group in panel["global_repeated_state_panel"]
            for reference in group["rows"]
        ]
        if not references:
            raise EditingT1PanelError("primary global repeated-state panel is empty")
    elif family not in RINGCORE_EDITING_FAMILIES:
        raise ValueError(f"unknown T1 family {family!r}")
    elif panel_kind == EDITING_T1_UNIQUE_PANEL_KIND:
        references = panel["unique_state_panels"][family]
    elif panel_kind == EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND:
        references = [
            reference
            for group in panel["within_family_repeated_state_panels"][family]
            for reference in group["rows"]
        ]
        if not references:
            raise EditingT1PanelError(
                f"family {family!r} has no observed repeated multi-successor state"
            )
    else:
        raise ValueError(f"unknown T1 panel kind {panel_kind!r}")
    selected = tuple(rows[int(reference["source_row_index"])] for reference in references)
    if panel_kind == EDITING_T1_UNIQUE_PANEL_KIND and len(
        {row["source_state_sha256"] for row in selected}
    ) != len(selected):
        raise EditingT1PanelError("resolved unique-state panel contains an exact-state duplicate")
    return selected


def write_editing_t1_panel(
    panel: Mapping[str, Any],
    path: Path,
) -> None:
    """Atomically freeze canonical JSON without overwriting a different file."""

    content = (
        json.dumps(
            panel,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise EditingT1PanelError(
                    "immutable T1 panel already exists with different bytes"
                ) from None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def write_editing_t1_capacity_census(
    census: Mapping[str, Any],
    path: Path,
) -> None:
    """Atomically freeze a production-recomputed capacity census."""

    write_editing_t1_panel(census, path)


__all__ = [
    "ACTIVE8_T1_IDENTITY_FIELDS",
    "EDITING_T1_DIAGNOSTIC_ONLY_FAMILIES",
    "EDITING_T1_CAPACITY_CENSUS_SCHEMA",
    "EDITING_T1_CAPACITY_CENSUS_SCHEMA_VERSION",
    "EDITING_T1_CAPACITY_CENSUS_STATUS",
    "EDITING_T1_GLOBAL_FAMILY_SELECTOR",
    "EDITING_T1_GLOBAL_REPEATED_PANEL_KIND",
    "EDITING_T1_MAX_CAPACITY_WORKERS",
    "EDITING_T1_OPERATOR_FREEZE_EXCLUDED_FAMILIES",
    "EDITING_T1_PANEL_SCHEMA",
    "EDITING_T1_PANEL_SCHEMA_VERSION",
    "EDITING_T1_PANEL_STATUS",
    "EDITING_T1_REQUIRED_SCOPES",
    "EDITING_T1_REPEATED_MAX_EXAMPLES",
    "EDITING_T1_REPEATED_MIN_EXAMPLES",
    "EDITING_T1_SELECTION_ALGORITHM",
    "EDITING_T1_SUPPORT_TIME",
    "EDITING_T1_UNIQUE_EXAMPLES_PER_FAMILY",
    "EDITING_T1_UNIQUE_PANEL_KIND",
    "EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND",
    "EditingT1PanelError",
    "active8_t1_identity",
    "build_editing_t1_capacity_census",
    "build_editing_t1_panel",
    "editing_t1_capacity_census_implementation_sha256",
    "editing_t1_panel_capacity_strata_sha256",
    "editing_t1_panel_census_sha256",
    "editing_t1_panel_selection_sha256",
    "filter_active8_forensics_rows",
    "load_editing_t1_panel",
    "selected_source_rows",
    "validate_charge_policy_exclusions",
    "validate_editing_t1_capacity_census",
    "validate_editing_t1_panel",
    "within_family_repeated_panel_families",
    "write_editing_t1_capacity_census",
    "write_editing_t1_panel",
]
