"""Strict score-independent successor cache for the semantic Editing-V2 P50 stream.

The prepared P50 recipe commits both the unique nonterminal addresses used by
optimization and the complete trace closure needed by the production
successor-fiber validator.  This module turns that commitment into immutable,
content-addressed CPU artifacts.  It stores executor coordinates only, never
model scores, probabilities, hazards, checkpoints, or training authority.

The module intentionally stops at the physical cache boundary.  A separate
producer must reopen semantic source records and compile each planned leaf with
the production executor.  A separate P50 authority layer must then reopen this
cache together with the other required domain artifacts.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.immutable_artifact import write_bytes_if_absent
from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_semantic_capability_cells import (
    SemanticCapabilityCellRegistry,
    classify_verified_structural_transition,
    load_semantic_capability_cell_registry,
)
from compose_v4.data.successor_fiber_cache import (
    DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheError,
    SuccessorFiberCacheRecord,
    canonical_successor_fiber_records_for_shard,
    successor_fiber_cache_record_from_payload,
    successor_fiber_cache_record_payload,
)
from compose_v4.experiments.editing_v2_semantic_p50_recipe_stream import (
    PREPARED_SCHEMA,
    PREPARED_STATUS,
    REQUIRED_BINDING_PURPOSES,
    SCHEMA_VERSION as PREPARED_SCHEMA_VERSION,
    SemanticP50Prerequisites,
)
from compose_v4.experiments.editing_v2_semantic_t1_decision import (
    DECISION_GO_STATUS,
    validate_semantic_gate_zero_evidence_receipt,
    validate_semantic_t1_capacity_decision,
)
from compose_v4.experiments.editing_v2_semantic_p50_source_inventory import (
    SemanticP50SourceInventoryBinding,
    VerifiedSemanticP50SourceInventory,
    load_semantic_p50_source_inventory,
)
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_v2_semantic_runtime import SemanticScratchRuntime
from compose_v4.experiments.successor_fiber_cache_builder import (
    SuccessorFiberCacheBuildError,
    compile_successor_fiber_trace_union,
)
from compose_v4.rewrite.kernel import canonical_state_key

PLAN_SCHEMA = "compose.editing_v2.semantic_p50_successor_cache_plan"
LEAF_SCHEMA = "compose.editing_v2.semantic_p50_successor_cache_leaf"
MANIFEST_SCHEMA = "compose.editing_v2.semantic_p50_successor_cache_manifest"
COMPLETION_SCHEMA = "compose.editing_v2.semantic_p50_successor_cache_completion"
SCHEMA_VERSION = 1

PLAN_STATUS = "FROZEN_EXACT_STREAM_UNION_CACHE_PLAN_NO_AUTHORITY"
LEAF_STATUS = "COMPLETE_EXACT_TRACE_CLOSURE_COORDINATES_NO_AUTHORITY"
MANIFEST_STATUS = "COMPLETE_EXACT_STREAM_UNION_CACHE_MANIFEST_NO_AUTHORITY"
COMPLETION_STATUS = "COMPLETE_PHYSICALLY_REOPENABLE_STREAM_UNION_CACHE_NO_AUTHORITY"

PLAN_FILENAME = "semantic_p50_successor_cache_plan.json"
LEAF_FILENAME = "semantic_p50_successor_cache_leaf.json"
MANIFEST_FILENAME = "semantic_p50_successor_cache_manifest.json"
COMPLETION_FILENAME = "semantic_p50_successor_cache_completion.json"
DEFAULT_OUTPUT_PREFIX = "/artifacts/editing_v2/semantic_p50_successor_cache"
VALIDATION_PARTITION_ROLE = "validation"
VALIDATION_DERIVATION_ALGORITHM = (
    "all_physically_reopened_active8_admitted_validation_transitions_v1"
)
SUPPORT_COMPILATION_TIME = 0.5

MAX_PREPARED_BYTES = 256 << 20
MAX_PLAN_BYTES = 256 << 20
MAX_LEAF_BYTES = 1 << 30
MAX_MANIFEST_BYTES = 64 << 20
MAX_COMPLETION_BYTES = 2 << 20

NO_DOWNSTREAM_AUTHORITY = {
    "training_authorized": False,
    "bounded_p50_authorized": False,
    "p500_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}

_HEX = frozenset("0123456789abcdef")
_ADDRESS_FIELDS = {
    "packed_shard_content_sha256",
    "packed_shard_name",
    "entry_index",
    "layer",
    "partition",
    "trace_id",
    "trace_source_key",
    "trace_target_key",
    "progress_index",
    "path_length",
}
_CLOSURE_FIELDS = {
    "address",
    "requested_for_training",
    "closure_only",
    "schedulable",
    "terminal",
}
_NORMALIZED_CLOSURE_FIELDS = {
    "address",
    "selected_for_consumer",
    "closure_only",
    "schedulable",
    "terminal",
    "population_role",
}
_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/experiments/editing_v2_semantic_p50_successor_cache.py",
    "src/compose_v4/experiments/editing_v2_semantic_p50_recipe_stream.py",
    "src/compose_v4/experiments/editing_v2_semantic_p50_source_inventory.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_decision.py",
    "src/compose_v4/experiments/editing_v2_semantic_runtime.py",
    "src/compose_v4/experiments/successor_fiber_cache_builder.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/experiments/production_successor_kernel.py",
    "src/compose_v4/experiments/cnof_conditional.py",
    "src/compose_v4/experiments/editing_p50_gate.py",
    "src/compose_v4/data/editing_corpus_contract.py",
    "src/compose_v4/data/editing_v2_semantic_capability_cells.py",
    "src/compose_v4/data/editing_v2_semantic_active8_decision_source.py",
    "src/compose_v4/data/semantic_packed_trace_store.py",
    "src/compose_v4/data/successor_fiber_cache.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "src/compose_v4/chem/persistent_state_identity.py",
    "src/compose_v4/rewrite/action_codec_v4.py",
    "src/compose_v4/rewrite/kernel.py",
)


class SemanticP50SuccessorCacheError(RuntimeError):
    """P50 cache bytes, address coverage, or semantic lineage disagree."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache metadata is not finite canonical JSON"
        ) from error
    return raw + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _require_sha(value: object, *, field_name: str) -> str:
    if not _is_sha(value):
        raise SemanticP50SuccessorCacheError(f"{field_name} must be a lowercase SHA-256")
    return str(value)


def _require_no_authority(value: Mapping[str, Any], *, field_name: str) -> None:
    if any(value.get(name) is not expected for name, expected in NO_DOWNSTREAM_AUTHORITY.items()):
        raise SemanticP50SuccessorCacheError(
            f"{field_name} crosses the downstream authority boundary"
        )


def _load_canonical_object(
    path: Path, *, field_name: str, maximum_bytes: int
) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    try:
        size = source.stat().st_size
        if not 0 < size <= maximum_bytes:
            raise SemanticP50SuccessorCacheError(f"{field_name} exceeds its byte bound")
        raw = source.read_bytes()
        payload = json.loads(raw)
    except SemanticP50SuccessorCacheError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticP50SuccessorCacheError(
            f"{field_name} is absent or invalid: {source}"
        ) from error
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, newline=True):
        raise SemanticP50SuccessorCacheError(f"{field_name} is not a canonical JSON object")
    return payload, raw


def _artifact_path(value: object, *, artifact_root: Path, field_name: str) -> Path:
    if not isinstance(value, str):
        raise SemanticP50SuccessorCacheError(f"{field_name} must be a normalized /artifacts path")
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or "\\" in value
        or str(pure) != value
        or value.endswith("/")
    ):
        raise SemanticP50SuccessorCacheError(f"{field_name} must be a normalized /artifacts path")
    root = Path(artifact_root).resolve()
    resolved = (root / Path(*pure.parts[2:])).resolve()
    if not resolved.is_relative_to(root):
        raise SemanticP50SuccessorCacheError(f"{field_name} resolves outside artifact_root")
    return resolved


def _artifact_address(path: Path, *, artifact_root: Path) -> str:
    root = Path(artifact_root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache artifact resolves outside artifact_root"
        )
    relative = resolved.relative_to(root)
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def _address_payload(address: SuccessorFiberCacheAddress) -> dict[str, Any]:
    return {
        "packed_shard_content_sha256": address.packed_shard_content_sha256,
        "packed_shard_name": address.packed_shard_name,
        "entry_index": address.entry_index,
        "layer": address.layer,
        "partition": address.partition,
        "trace_id": address.trace_id,
        "trace_source_key": address.trace_source_key,
        "trace_target_key": address.trace_target_key,
        "progress_index": address.progress_index,
        "path_length": address.path_length,
    }


def _parse_address(value: object, *, field_name: str) -> SuccessorFiberCacheAddress:
    if not isinstance(value, Mapping) or set(value) != _ADDRESS_FIELDS:
        raise SemanticP50SuccessorCacheError(f"{field_name} has missing or unknown address fields")
    try:
        address = SuccessorFiberCacheAddress(**dict(value))
    except (TypeError, ValueError) as error:
        raise SemanticP50SuccessorCacheError(
            f"{field_name} is not a valid cache address"
        ) from error
    if _address_payload(address) != dict(value):
        raise SemanticP50SuccessorCacheError(f"{field_name} does not round-trip exactly")
    return address


def semantic_p50_successor_cache_implementation_sha256(*, repo_root: Path) -> str:
    """Hash the complete importable COMPOSE tree fixing cache semantics."""

    root = Path(repo_root).resolve()
    source_root = root / "src" / "compose_v4"
    transitive_sources = {
        path.relative_to(root).as_posix() for path in source_root.rglob("*.py") if path.is_file()
    }
    relative_sources = tuple(sorted(set(_IMPLEMENTATION_SOURCES) | transitive_sources))
    digest = hashlib.sha256()
    for relative in relative_sources:
        path = root / relative
        if not path.is_file():
            raise SemanticP50SuccessorCacheError(
                f"semantic P50 cache implementation source is absent: {relative}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _policy() -> dict[str, Any]:
    body = {
        "schema": "compose.editing_v2.semantic_p50_successor_cache_policy",
        "schema_version": 1,
        "status": "FROZEN_CPU_SCORE_INDEPENDENT_TRACE_CLOSURE_CACHE",
        **NO_DOWNSTREAM_AUTHORITY,
        "compile_device": "cpu",
        "compile_dtype": "torch.float32",
        "support_compilation_time": SUPPORT_COMPILATION_TIME,
        "coverage": "exact_prepared_recipe_complete_trace_closure",
        "scheduled_coverage": "exact_prepared_recipe_requested_address_union",
        "model_scores_or_probabilities_stored": False,
        "hazard_coordinates_included": False,
    }
    return {**body, "policy_sha256": _sha(body)}


_PREPARED_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *NO_DOWNSTREAM_AUTHORITY,
    "scientific_scope",
    "recipe_policy_file_sha256",
    "recipe_policy_sha256",
    "prerequisites",
    "candidate_inventory_sha256",
    "candidate_count",
    "required_families",
    "declared_nonempty_semantic_cells",
    "recipe",
    "stream_contract",
    "ordered_stream_rows",
    "ordered_address_stream_sha256",
    "ordered_training_stream_sha256",
    "requested_address_union",
    "requested_address_union_sha256",
    "complete_trace_closure_inventory",
    "complete_trace_closure_inventory_sha256",
    "cache_contract",
    "planned_family_exposure",
    "planned_semantic_cell_exposure",
    "minimum_nonzero_gradient_updates_by_family",
    "minimum_nonzero_gradient_updates_by_semantic_cell",
    "scheduled_lane_counts",
    "validation_contract",
    "required_physical_binding_purposes",
    "unresolved_physical_bindings",
    "prepared_recipe_sha256",
}


def validate_semantic_p50_prepared_recipe(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact scheduled and closure address contracts."""

    prepared = dict(value)
    body = dict(prepared)
    supplied_sha = body.pop("prepared_recipe_sha256", None)
    if (
        set(prepared) != _PREPARED_FIELDS
        or prepared.get("schema") != PREPARED_SCHEMA
        or prepared.get("schema_version") != PREPARED_SCHEMA_VERSION
        or prepared.get("status") != PREPARED_STATUS
        or supplied_sha != _sha(body)
        or tuple(prepared.get("required_physical_binding_purposes", ()))
        != REQUIRED_BINDING_PURPOSES
        or tuple(prepared.get("unresolved_physical_bindings", ())) != REQUIRED_BINDING_PURPOSES
    ):
        raise SemanticP50SuccessorCacheError(
            "prepared semantic P50 recipe schema or identity disagrees"
        )
    _require_no_authority(prepared, field_name="prepared semantic P50 recipe")
    try:
        prerequisites = SemanticP50Prerequisites(**dict(prepared["prerequisites"]))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise SemanticP50SuccessorCacheError(
            "prepared semantic P50 prerequisite identity is invalid"
        ) from error
    if prerequisites.as_payload() != prepared["prerequisites"]:
        raise SemanticP50SuccessorCacheError("prepared semantic P50 prerequisite fields disagree")

    requested_payloads = prepared.get("requested_address_union")
    closure_rows = prepared.get("complete_trace_closure_inventory")
    stream_rows = prepared.get("ordered_stream_rows")
    if not all(
        isinstance(items, list) for items in (requested_payloads, closure_rows, stream_rows)
    ):
        raise SemanticP50SuccessorCacheError(
            "prepared semantic P50 address inventories must be arrays"
        )
    requested = tuple(
        _parse_address(item, field_name=f"requested_address_union[{index}]")
        for index, item in enumerate(requested_payloads)
    )
    if (
        not requested
        or requested != tuple(sorted(requested))
        or len(set(requested)) != len(requested)
        or any(address.partition != "train" or address.is_terminal for address in requested)
        or prepared.get("requested_address_union_sha256") != _sha(requested_payloads)
    ):
        raise SemanticP50SuccessorCacheError(
            "prepared semantic P50 requested-address union is invalid"
        )
    requested_set = set(requested)

    closure_addresses: list[SuccessorFiberCacheAddress] = []
    for index, row in enumerate(closure_rows):
        if not isinstance(row, Mapping) or set(row) != _CLOSURE_FIELDS:
            raise SemanticP50SuccessorCacheError(
                f"complete_trace_closure_inventory[{index}] fields disagree"
            )
        address = _parse_address(
            row["address"],
            field_name=f"complete_trace_closure_inventory[{index}].address",
        )
        requested_here = address in requested_set
        expected = {
            "address": _address_payload(address),
            "requested_for_training": requested_here,
            "closure_only": not requested_here,
            "schedulable": requested_here and not address.is_terminal,
            "terminal": address.is_terminal,
        }
        if dict(row) != expected:
            raise SemanticP50SuccessorCacheError(
                "prepared semantic P50 closure scheduling flags disagree"
            )
        closure_addresses.append(address)
    closure = tuple(closure_addresses)
    if (
        not closure
        or closure != tuple(sorted(closure))
        or len(set(closure)) != len(closure)
        or not requested_set.issubset(closure)
        or any(address.partition != "train" for address in closure)
        or prepared.get("complete_trace_closure_inventory_sha256") != _sha(closure_rows)
    ):
        raise SemanticP50SuccessorCacheError(
            "prepared semantic P50 complete-trace closure is invalid"
        )
    grouped: defaultdict[tuple[Any, ...], list[SuccessorFiberCacheAddress]] = defaultdict(list)
    for address in closure:
        grouped[address.trace_key].append(address)
    if any(
        [address.progress_index for address in addresses]
        != list(range(addresses[0].path_length + 1))
        or not any(address in requested_set for address in addresses)
        for addresses in grouped.values()
    ):
        raise SemanticP50SuccessorCacheError(
            "prepared semantic P50 closure omits a trace row or has an untouched trace"
        )

    scheduled_addresses: list[SuccessorFiberCacheAddress] = []
    for index, row in enumerate(stream_rows):
        if not isinstance(row, Mapping):
            raise SemanticP50SuccessorCacheError(f"ordered_stream_rows[{index}] must be an object")
        row_body = dict(row)
        supplied_row_sha = row_body.pop("row_sha256", None)
        if supplied_row_sha != _sha(row_body) or row.get("stream_index") != index:
            raise SemanticP50SuccessorCacheError(
                "prepared semantic P50 ordered stream row identity disagrees"
            )
        address = _parse_address(
            row.get("address"), field_name=f"ordered_stream_rows[{index}].address"
        )
        if address not in requested_set or address.is_terminal:
            raise SemanticP50SuccessorCacheError(
                "prepared semantic P50 stream schedules an unrequested or terminal row"
            )
        scheduled_addresses.append(address)
    if (
        set(scheduled_addresses) != requested_set
        or prepared.get("ordered_address_stream_sha256")
        != _sha([_address_payload(address) for address in scheduled_addresses])
        or prepared.get("ordered_training_stream_sha256") != _sha(stream_rows)
    ):
        raise SemanticP50SuccessorCacheError(
            "prepared semantic P50 ordered stream differs from the requested union"
        )
    cache_contract = prepared.get("cache_contract")
    if not isinstance(cache_contract, Mapping) or (
        cache_contract.get("coverage_mode") != "complete_trace_closure_of_planned_address_union"
        or cache_contract.get("closure_only_rows_schedulable") is not False
        or cache_contract.get("terminal_rows_schedulable") is not False
        or cache_contract.get("touched_trace_count") != len(grouped)
        or cache_contract.get("requested_unique_nonterminal_address_count") != len(requested)
        or cache_contract.get("closure_record_count") != len(closure)
        or cache_contract.get("closure_only_record_count") != len(closure) - len(requested)
    ):
        raise SemanticP50SuccessorCacheError(
            "prepared semantic P50 cache census or unschedulability contract disagrees"
        )
    return prepared


def load_semantic_p50_prepared_recipe(
    path: Path, *, expected_prepared_recipe_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    """Physically reopen and validate one canonical prepared recipe."""

    payload, raw = _load_canonical_object(
        path,
        field_name="semantic P50 prepared recipe",
        maximum_bytes=MAX_PREPARED_BYTES,
    )
    validated = validate_semantic_p50_prepared_recipe(payload)
    if (
        expected_prepared_recipe_sha256 is not None
        and validated["prepared_recipe_sha256"] != expected_prepared_recipe_sha256
    ):
        raise SemanticP50SuccessorCacheError(
            "prepared semantic P50 recipe differs from the expected identity"
        )
    return validated, hashlib.sha256(raw).hexdigest()


def _prepared_binding(
    prepared: Mapping[str, Any], *, path: Path, file_sha256: str, artifact_root: Path
) -> dict[str, Any]:
    return {
        "artifact_path": _artifact_address(path, artifact_root=artifact_root),
        "file_sha256": file_sha256,
        "prepared_recipe_sha256": prepared["prepared_recipe_sha256"],
        "recipe_policy_sha256": prepared["recipe_policy_sha256"],
        "ordered_address_stream_sha256": prepared["ordered_address_stream_sha256"],
        "ordered_training_stream_sha256": prepared["ordered_training_stream_sha256"],
        "requested_address_union_sha256": prepared["requested_address_union_sha256"],
        "complete_trace_closure_inventory_sha256": prepared[
            "complete_trace_closure_inventory_sha256"
        ],
        "prerequisites": dict(prepared["prerequisites"]),
    }


@dataclass(frozen=True, slots=True)
class SemanticP50SourceReopenPaths:
    """Physical inputs needed to reconstruct the Active8 decision source."""

    source_inventory_path: Path
    migration_completion_path: Path
    chunk_cache_plan_path: Path
    chunk_cache_global_completion_path: Path
    decision_plan_path: Path
    decision_completion_path: Path


def _source_binding_from_prerequisites(
    prerequisites: Mapping[str, Any],
) -> SemanticP50SourceInventoryBinding:
    try:
        return SemanticP50SourceInventoryBinding(
            source_inventory_file_sha256=prerequisites["source_inventory_file_sha256"],
            source_inventory_sha256=prerequisites["source_inventory_sha256"],
            process_identity_sha256=prerequisites["process_identity_sha256"],
            model_runtime_identity_sha256=prerequisites["model_runtime_identity_sha256"],
            active8_policy_sha256=prerequisites["active8_policy_sha256"],
            operator_capability_fingerprint=prerequisites["operator_capability_fingerprint"],
            decision_source_implementation_sha256=prerequisites[
                "decision_source_implementation_sha256"
            ],
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise SemanticP50SuccessorCacheError(
            "semantic P50 prerequisites cannot bind the physical source"
        ) from error


def _validate_verified_source_binding(
    source: VerifiedSemanticP50SourceInventory,
    *,
    prerequisites: Mapping[str, Any],
) -> None:
    if not isinstance(source, VerifiedSemanticP50SourceInventory):
        raise TypeError("semantic P50 validation cache requires VerifiedSemanticP50SourceInventory")
    if source.binding != _source_binding_from_prerequisites(prerequisites):
        raise SemanticP50SuccessorCacheError(
            "physically verified semantic source differs from prepared prerequisites"
        )


def _open_verified_source(
    paths: SemanticP50SourceReopenPaths,
    *,
    prerequisites: Mapping[str, Any],
    artifact_root: Path,
    repo_root: Path,
) -> VerifiedSemanticP50SourceInventory:
    if not isinstance(paths, SemanticP50SourceReopenPaths):
        raise TypeError("source_reopen_paths must use the typed physical schema")
    expected = _source_binding_from_prerequisites(prerequisites)
    try:
        source = load_semantic_p50_source_inventory(
            paths.source_inventory_path,
            expected_binding=expected,
            migration_completion_path=paths.migration_completion_path,
            chunk_cache_plan_path=paths.chunk_cache_plan_path,
            chunk_cache_global_completion_path=(paths.chunk_cache_global_completion_path),
            decision_plan_path=paths.decision_plan_path,
            decision_completion_path=paths.decision_completion_path,
            artifact_root=artifact_root,
            repo_root=repo_root,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise SemanticP50SuccessorCacheError(
            "semantic P50 validation source failed complete physical reopening"
        ) from error
    _validate_verified_source_binding(source, prerequisites=prerequisites)
    return source


def _normalized_closure_row(
    address: SuccessorFiberCacheAddress,
    *,
    selected: bool,
    population_role: str,
) -> dict[str, Any]:
    return {
        "address": _address_payload(address),
        "selected_for_consumer": selected,
        "closure_only": not selected,
        "schedulable": selected and not address.is_terminal,
        "terminal": address.is_terminal,
        "population_role": population_role,
    }


def _derive_authenticated_validation_contract(
    source: VerifiedSemanticP50SourceInventory,
    *,
    prepared: Mapping[str, Any],
    registry: SemanticCapabilityCellRegistry,
) -> dict[str, Any]:
    """Derive all validation candidates from the physically reopened source."""

    _validate_verified_source_binding(source, prerequisites=prepared["prerequisites"])
    if not isinstance(registry, SemanticCapabilityCellRegistry):
        raise TypeError("semantic P50 validation derivation requires the typed capability registry")
    prerequisites = prepared["prerequisites"]
    if (
        registry.registry_sha256 != prerequisites["capability_registry_sha256"]
        or registry.classifier_implementation_sha256
        != prerequisites["classifier_implementation_sha256"]
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 validation classifier differs from prepared recipe"
        )

    candidates: list[dict[str, Any]] = []
    closure_addresses: list[SuccessorFiberCacheAddress] = []
    for trace in source.index.iter_accepted_traces_for_partition(VALIDATION_PARTITION_ROLE):
        trace_candidates: list[dict[str, Any]] = []
        for transition in source.index.accepted_transitions_for(trace):
            source.index.validate_accepted_transition(transition)
            assignment = classify_verified_structural_transition(
                source.index,
                transition,
                registry=registry,
            )
            address = SuccessorFiberCacheAddress.from_packed_trace(
                transition.addressed_trace.address,
                progress_index=transition.step_index,
            )
            if (
                address.partition != VALIDATION_PARTITION_ROLE
                or address.is_terminal
                or assignment.model_family not in ACTIVE8_FAMILIES
                or assignment.data_lane != address.layer
            ):
                raise SemanticP50SuccessorCacheError(
                    "authenticated validation transition has another role or support"
                )
            candidate_body = {
                "address": _address_payload(address),
                "family": assignment.model_family,
                "semantic_cell_id": assignment.capability_cell_id,
                "data_lane": assignment.data_lane,
                "assignment_sha256": assignment.assignment_sha256,
            }
            trace_candidates.append({**candidate_body, "candidate_sha256": _sha(candidate_body)})
        if not trace_candidates:
            continue
        trace_addresses = tuple(
            SuccessorFiberCacheAddress(**item["address"]) for item in trace_candidates
        )
        representative = trace_addresses[0]
        if any(
            address.trace_key != representative.trace_key for address in trace_addresses
        ) or tuple(address.progress_index for address in trace_addresses) != tuple(
            range(representative.path_length)
        ):
            raise SemanticP50SuccessorCacheError(
                "authenticated validation trace does not expose every admitted transition"
            )
        candidates.extend(trace_candidates)
        closure_addresses.extend(
            SuccessorFiberCacheAddress(
                **{
                    **_address_payload(representative),
                    "progress_index": progress_index,
                }
            )
            for progress_index in range(representative.path_length + 1)
        )
    candidates.sort(key=lambda item: SuccessorFiberCacheAddress(**item["address"]))
    candidate_addresses = tuple(
        SuccessorFiberCacheAddress(**item["address"]) for item in candidates
    )
    if (
        not candidates
        or len(set(candidate_addresses)) != len(candidate_addresses)
        or any(address.partition != VALIDATION_PARTITION_ROLE for address in candidate_addresses)
    ):
        raise SemanticP50SuccessorCacheError(
            "authenticated validation candidate inventory is empty or duplicated"
        )
    closure = tuple(sorted(closure_addresses))
    if len(set(closure)) != len(closure):
        raise SemanticP50SuccessorCacheError(
            "authenticated validation trace closure repeats an address"
        )
    selected = set(candidate_addresses)
    closure_rows = [
        _normalized_closure_row(
            address,
            selected=address in selected,
            population_role="validation_baseline",
        )
        for address in closure
    ]
    family_counts: defaultdict[str, int] = defaultdict(int)
    cell_counts: defaultdict[str, int] = defaultdict(int)
    for candidate in candidates:
        family_counts[candidate["family"]] += 1
        cell_counts[candidate["semantic_cell_id"]] += 1
    required_cells = tuple(prepared["validation_contract"]["required_nonempty_semantic_cells"])
    missing_families = sorted(set(ACTIVE8_FAMILIES) - set(family_counts))
    missing_required_cells = sorted(set(required_cells) - set(cell_counts))
    body = {
        "algorithm": VALIDATION_DERIVATION_ALGORITHM,
        "partition_role": VALIDATION_PARTITION_ROLE,
        "source_inventory_sha256": source.binding.source_inventory_sha256,
        "decision_source_implementation_sha256": (
            source.binding.decision_source_implementation_sha256
        ),
        "capability_registry_sha256": registry.registry_sha256,
        "classifier_implementation_sha256": (registry.classifier_implementation_sha256),
        "candidate_rows": candidates,
        "candidate_inventory_sha256": _sha(candidates),
        "candidate_count": len(candidates),
        "closure_rows": closure_rows,
        "closure_inventory_sha256": _sha(closure_rows),
        "closure_record_count": len(closure_rows),
        "family_counts": dict(sorted(family_counts.items())),
        "semantic_cell_counts": dict(sorted(cell_counts.items())),
        "required_families": list(ACTIVE8_FAMILIES),
        "required_semantic_cells": list(required_cells),
        "missing_required_families": missing_families,
        "missing_required_semantic_cells": missing_required_cells,
        "complete_required_coverage": not missing_families and not missing_required_cells,
    }
    return {**body, "validation_contract_sha256": _sha(body)}


def _validate_authenticated_validation_contract(
    value: object,
    *,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "algorithm",
        "partition_role",
        "source_inventory_sha256",
        "decision_source_implementation_sha256",
        "capability_registry_sha256",
        "classifier_implementation_sha256",
        "candidate_rows",
        "candidate_inventory_sha256",
        "candidate_count",
        "closure_rows",
        "closure_inventory_sha256",
        "closure_record_count",
        "family_counts",
        "semantic_cell_counts",
        "required_families",
        "required_semantic_cells",
        "missing_required_families",
        "missing_required_semantic_cells",
        "complete_required_coverage",
        "validation_contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise SemanticP50SuccessorCacheError(
            "semantic P50 authenticated validation contract fields disagree"
        )
    contract = dict(value)
    body = dict(contract)
    supplied_sha = body.pop("validation_contract_sha256")
    candidates = contract["candidate_rows"]
    closure_rows = contract["closure_rows"]
    if (
        contract["algorithm"] != VALIDATION_DERIVATION_ALGORITHM
        or contract["partition_role"] != VALIDATION_PARTITION_ROLE
        or supplied_sha != _sha(body)
        or not isinstance(candidates, list)
        or not isinstance(closure_rows, list)
        or not candidates
        or contract["candidate_count"] != len(candidates)
        or contract["candidate_inventory_sha256"] != _sha(candidates)
        or contract["closure_record_count"] != len(closure_rows)
        or contract["closure_inventory_sha256"] != _sha(closure_rows)
        or contract["source_inventory_sha256"]
        != prepared["prerequisites"]["source_inventory_sha256"]
        or contract["decision_source_implementation_sha256"]
        != prepared["prerequisites"]["decision_source_implementation_sha256"]
        or contract["capability_registry_sha256"]
        != prepared["prerequisites"]["capability_registry_sha256"]
        or contract["classifier_implementation_sha256"]
        != prepared["prerequisites"]["classifier_implementation_sha256"]
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 authenticated validation contract identity disagrees"
        )
    candidate_addresses: list[SuccessorFiberCacheAddress] = []
    family_counts: defaultdict[str, int] = defaultdict(int)
    cell_counts: defaultdict[str, int] = defaultdict(int)
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise SemanticP50SuccessorCacheError(
                "semantic P50 validation candidate must be an object"
            )
        candidate_body = dict(candidate)
        candidate_sha = candidate_body.pop("candidate_sha256", None)
        if set(candidate_body) != {
            "address",
            "family",
            "semantic_cell_id",
            "data_lane",
            "assignment_sha256",
        } or candidate_sha != _sha(candidate_body):
            raise SemanticP50SuccessorCacheError(
                "semantic P50 validation candidate identity disagrees"
            )
        address = _parse_address(candidate["address"], field_name="validation candidate address")
        if (
            address.partition != VALIDATION_PARTITION_ROLE
            or address.is_terminal
            or candidate["family"] not in ACTIVE8_FAMILIES
            or candidate["data_lane"] != address.layer
        ):
            raise SemanticP50SuccessorCacheError(
                "semantic P50 validation candidate role or support disagrees"
            )
        _require_sha(candidate["assignment_sha256"], field_name="candidate.assignment_sha256")
        candidate_addresses.append(address)
        family_counts[candidate["family"]] += 1
        cell_counts[candidate["semantic_cell_id"]] += 1
    if candidate_addresses != sorted(candidate_addresses) or len(set(candidate_addresses)) != len(
        candidate_addresses
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 validation candidates are not a unique canonical inventory"
        )
    selected = set(candidate_addresses)
    closure_addresses: list[SuccessorFiberCacheAddress] = []
    for row in closure_rows:
        if not isinstance(row, Mapping):
            raise SemanticP50SuccessorCacheError(
                "semantic P50 validation closure row must be an object"
            )
        address = _parse_address(row.get("address"), field_name="validation closure address")
        expected = _normalized_closure_row(
            address,
            selected=address in selected,
            population_role="validation_baseline",
        )
        if dict(row) != expected or address.partition != VALIDATION_PARTITION_ROLE:
            raise SemanticP50SuccessorCacheError(
                "semantic P50 validation closure role or flags disagree"
            )
        closure_addresses.append(address)
    if (
        closure_addresses != sorted(closure_addresses)
        or len(set(closure_addresses)) != len(closure_addresses)
        or not selected.issubset(closure_addresses)
        or selected != {address for address in closure_addresses if not address.is_terminal}
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 validation closure is not the complete transition union"
        )
    grouped: defaultdict[tuple[Any, ...], list[SuccessorFiberCacheAddress]] = defaultdict(list)
    for address in closure_addresses:
        grouped[address.trace_key].append(address)
    if any(
        [address.progress_index for address in addresses]
        != list(range(addresses[0].path_length + 1))
        or not any(address in selected for address in addresses)
        for addresses in grouped.values()
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 validation closure omits an authenticated trace row"
        )
    required_cells = list(prepared["validation_contract"]["required_nonempty_semantic_cells"])
    missing_families = sorted(set(ACTIVE8_FAMILIES) - set(family_counts))
    missing_cells = sorted(set(required_cells) - set(cell_counts))
    if (
        contract["family_counts"] != dict(sorted(family_counts.items()))
        or contract["semantic_cell_counts"] != dict(sorted(cell_counts.items()))
        or contract["required_families"] != list(ACTIVE8_FAMILIES)
        or contract["required_semantic_cells"] != required_cells
        or contract["missing_required_families"] != missing_families
        or contract["missing_required_semantic_cells"] != missing_cells
        or contract["complete_required_coverage"]
        is not (not missing_families and not missing_cells)
    ):
        raise SemanticP50SuccessorCacheError("semantic P50 validation coverage census disagrees")
    return contract


def _authenticate_cache_records_against_source(
    *,
    source: VerifiedSemanticP50SourceInventory,
    plan: Mapping[str, Any],
    records_by_address_sha256: Mapping[str, SuccessorFiberCacheRecord],
    scratch_runtime: SemanticScratchRuntime,
) -> None:
    """Fresh-compile and compare every coordinate on every exact trace."""

    prerequisites = plan["prepared_recipe_binding"]["prerequisites"]
    if not isinstance(scratch_runtime, SemanticScratchRuntime):
        raise TypeError("strict semantic P50 cache opening requires SemanticScratchRuntime")
    if (
        scratch_runtime.initial_model_state_sha256
        != prerequisites["scratch_initial_model_state_sha256"]
        or state_dict_semantic_sha256(scratch_runtime.model.state_dict())
        != scratch_runtime.initial_model_state_sha256
        or scratch_runtime.process_identity_sha256 != prerequisites["process_identity_sha256"]
        or scratch_runtime.architecture.operator_capability_fingerprint
        != prerequisites["operator_capability_fingerprint"]
    ):
        raise SemanticP50SuccessorCacheError(
            "strict cache compiler runtime differs from the frozen scratch process"
        )

    expected_by_trace: defaultdict[tuple[Any, ...], set[SuccessorFiberCacheAddress]] = defaultdict(
        set
    )
    for task in plan["tasks"]:
        for row in task["closure_rows"]:
            address = _parse_address(
                row["address"], field_name="source-authenticated closure address"
            )
            expected_by_trace[address.trace_key].add(address)
    if not expected_by_trace:
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache has no trace closure to authenticate"
        )

    observed_trace_keys: set[tuple[Any, ...]] = set()
    physical_path_records: list[PathRecord] = []
    for partition_role in ("train", VALIDATION_PARTITION_ROLE):
        for trace in source.index.iter_accepted_traces_for_partition(partition_role):
            packed_address = trace.addressed_trace.address
            first = SuccessorFiberCacheAddress.from_packed_trace(packed_address, progress_index=0)
            trace_key = first.trace_key
            if trace_key not in expected_by_trace:
                continue
            if trace_key in observed_trace_keys:
                raise SemanticP50SuccessorCacheError(
                    "physical semantic source repeats a planned cache trace"
                )
            transitions = tuple(source.index.accepted_transitions_for(trace))
            if len(transitions) != packed_address.path_length or tuple(
                item.step_index for item in transitions
            ) != tuple(range(packed_address.path_length)):
                raise SemanticP50SuccessorCacheError(
                    "physical semantic source omits a planned teacher transition"
                )
            expected_addresses = expected_by_trace[trace_key]
            physical_addresses = {
                SuccessorFiberCacheAddress.from_packed_trace(
                    packed_address, progress_index=progress_index
                )
                for progress_index in range(packed_address.path_length + 1)
            }
            if expected_addresses != physical_addresses:
                raise SemanticP50SuccessorCacheError(
                    "planned cache closure differs from the physical trace states"
                )
            for address in sorted(physical_addresses):
                address_sha = _sha(_address_payload(address))
                record = records_by_address_sha256.get(address_sha)
                if record is None:
                    raise SemanticP50SuccessorCacheError(
                        "physical trace state is absent from the loaded cache"
                    )
                state = trace.addressed_trace.path.state_at(address.progress_index)
                source_key = canonical_state_key(state)
                source_state_sha256 = persistent_slot_state_sha256(state)
                if (
                    record.source_key != source_key
                    or record.source_state_sha256 != source_state_sha256
                ):
                    raise SemanticP50SuccessorCacheError(
                        "cached source state differs from the physical trace state"
                    )
                if address.is_terminal:
                    if record.teacher_fiber is not None:
                        raise SemanticP50SuccessorCacheError(
                            "cached terminal row carries a teacher successor"
                        )
                    continue
                successor_state = trace.addressed_trace.path.state_at(address.progress_index + 1)
                if (
                    record.teacher_fiber is None
                    or record.target_key != canonical_state_key(successor_state)
                    or record.target_state_sha256 != persistent_slot_state_sha256(successor_state)
                ):
                    raise SemanticP50SuccessorCacheError(
                        "cached teacher successor differs from the physical trace"
                    )
            physical_path_records.append(
                PathRecord(
                    target_key=packed_address.target_key,
                    path=trace.addressed_trace.path,
                    corpus_address=packed_address,
                )
            )
            observed_trace_keys.add(trace_key)
    if observed_trace_keys != set(expected_by_trace):
        raise SemanticP50SuccessorCacheError(
            "physical semantic source does not contain every planned cache trace"
        )
    scratch_runtime.model.eval()
    try:
        freshly_compiled = compile_successor_fiber_trace_union(
            scratch_runtime.model,
            physical_path_records,
            time=SUPPORT_COMPILATION_TIME,
        )
    except (SuccessorFiberCacheBuildError, TypeError, ValueError) as error:
        raise SemanticP50SuccessorCacheError(
            "fresh production successor compilation failed during strict opening"
        ) from error
    fresh_by_address = {
        _sha(_address_payload(record.address)): record for record in freshly_compiled
    }
    if (
        len(fresh_by_address) != len(freshly_compiled)
        or set(fresh_by_address) != set(records_by_address_sha256)
        or any(
            fresh_by_address[address_sha] != cached
            for address_sha, cached in records_by_address_sha256.items()
        )
    ):
        raise SemanticP50SuccessorCacheError(
            "cached productive successor coordinates differ from fresh production compilation"
        )


def build_semantic_p50_successor_cache_plan(
    *,
    prepared_recipe_path: Path,
    verified_source: VerifiedSemanticP50SourceInventory,
    source_revision_sha256: str,
    artifact_root: Path,
    repo_root: Path,
    registry: SemanticCapabilityCellRegistry | None = None,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Freeze exact train and source-derived validation cache tasks."""

    source_revision_sha256 = _require_sha(
        source_revision_sha256, field_name="source_revision_sha256"
    )
    root = Path(artifact_root).resolve()
    prepared_path = Path(prepared_recipe_path).resolve()
    if not prepared_path.is_relative_to(root):
        raise SemanticP50SuccessorCacheError("prepared recipe must resolve inside artifact_root")
    prepared, prepared_file_sha = load_semantic_p50_prepared_recipe(prepared_path)
    selected_registry = registry or load_semantic_capability_cell_registry()
    validation_contract = _derive_authenticated_validation_contract(
        verified_source,
        prepared=prepared,
        registry=selected_registry,
    )
    implementation_sha = semantic_p50_successor_cache_implementation_sha256(repo_root=repo_root)
    policy = _policy()
    output_root = _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=root,
        field_name="output_prefix",
    ).parent
    prepared_binding = _prepared_binding(
        prepared,
        path=prepared_path,
        file_sha256=prepared_file_sha,
        artifact_root=root,
    )
    base_identity = {
        "source_revision_sha256": source_revision_sha256,
        "implementation_sha256": implementation_sha,
        "policy": policy,
        "prepared_recipe_binding": prepared_binding,
        "validation_contract": validation_contract,
        "output_prefix": output_prefix,
    }
    build_identity_sha = _sha(base_identity)

    requested_by_group: defaultdict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for payload in prepared["requested_address_union"]:
        address = _parse_address(payload, field_name="requested address")
        requested_by_group[
            (
                "train_optimization",
                address.packed_shard_content_sha256,
                address.packed_shard_name,
                address.layer,
            )
        ].append(dict(payload))
    closure_by_group: defaultdict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in prepared["complete_trace_closure_inventory"]:
        address = _parse_address(row["address"], field_name="closure address")
        closure_by_group[
            (
                "train_optimization",
                address.packed_shard_content_sha256,
                address.packed_shard_name,
                address.layer,
            )
        ].append(
            _normalized_closure_row(
                address,
                selected=bool(row["requested_for_training"]),
                population_role="train_optimization",
            )
        )
    for candidate in validation_contract["candidate_rows"]:
        address = _parse_address(candidate["address"], field_name="validation candidate address")
        requested_by_group[
            (
                "validation_baseline",
                address.packed_shard_content_sha256,
                address.packed_shard_name,
                address.layer,
            )
        ].append(dict(candidate["address"]))
    for row in validation_contract["closure_rows"]:
        address = _parse_address(row["address"], field_name="validation closure address")
        closure_by_group[
            (
                "validation_baseline",
                address.packed_shard_content_sha256,
                address.packed_shard_name,
                address.layer,
            )
        ].append(dict(row))
    if set(requested_by_group) - set(closure_by_group):
        raise SemanticP50SuccessorCacheError(
            "prepared requested-address group is absent from the closure"
        )
    tasks: list[dict[str, Any]] = []
    for task_index, group in enumerate(sorted(closure_by_group)):
        population_role, digest, shard_name, data_lane = group
        closure_rows = closure_by_group[group]
        requested_rows = requested_by_group.get(group, [])
        if not requested_rows:
            raise SemanticP50SuccessorCacheError(
                "prepared closure contains a shard with no scheduled address"
            )
        task_body = {
            "task_index": task_index,
            "population_role": population_role,
            "partition_role": (
                "train" if population_role == "train_optimization" else VALIDATION_PARTITION_ROLE
            ),
            "packed_shard_content_sha256": digest,
            "packed_shard_name": shard_name,
            "data_lane": data_lane,
            "requested_addresses": requested_rows,
            "requested_address_inventory_sha256": _sha(requested_rows),
            "closure_rows": closure_rows,
            "closure_inventory_sha256": _sha(closure_rows),
            "requested_address_count": len(requested_rows),
            "closure_record_count": len(closure_rows),
        }
        task_sha = _sha({"build_identity_sha256": build_identity_sha, "task": task_body})
        tasks.append({**task_body, "task_identity_sha256": task_sha})
    plan_body = {
        "schema": PLAN_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": PLAN_STATUS,
        **NO_DOWNSTREAM_AUTHORITY,
        **base_identity,
        "build_identity_sha256": build_identity_sha,
        "run_artifact_root": _artifact_address(
            output_root / build_identity_sha, artifact_root=root
        ),
        "tasks": tasks,
        "task_inventory_sha256": _sha(tasks),
        "task_count": len(tasks),
        "train_requested_address_count": len(prepared["requested_address_union"]),
        "train_closure_record_count": len(prepared["complete_trace_closure_inventory"]),
        "validation_requested_address_count": validation_contract["candidate_count"],
        "validation_closure_record_count": validation_contract["closure_record_count"],
        "requested_address_count": sum(len(values) for values in requested_by_group.values()),
        "closure_record_count": sum(len(values) for values in closure_by_group.values()),
    }
    run_identity_sha = _sha(plan_body)
    with_run = {**plan_body, "run_identity_sha256": run_identity_sha}
    return {**with_run, "plan_sha256": _sha(with_run)}


_PLAN_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *NO_DOWNSTREAM_AUTHORITY,
    "source_revision_sha256",
    "implementation_sha256",
    "policy",
    "prepared_recipe_binding",
    "validation_contract",
    "output_prefix",
    "build_identity_sha256",
    "run_artifact_root",
    "tasks",
    "task_inventory_sha256",
    "task_count",
    "train_requested_address_count",
    "train_closure_record_count",
    "validation_requested_address_count",
    "validation_closure_record_count",
    "requested_address_count",
    "closure_record_count",
    "run_identity_sha256",
    "plan_sha256",
}


def validate_semantic_p50_successor_cache_plan(
    value: Mapping[str, Any], *, artifact_root: Path, repo_root: Path
) -> dict[str, Any]:
    """Validate the plan and physically reopen its prepared recipe."""

    plan = dict(value)
    body = dict(plan)
    supplied_plan_sha = body.pop("plan_sha256", None)
    run_body = dict(body)
    supplied_run_sha = run_body.pop("run_identity_sha256", None)
    if (
        set(plan) != _PLAN_FIELDS
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or supplied_plan_sha != _sha(body)
        or supplied_run_sha != _sha(run_body)
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 successor-cache plan schema or self-hash disagrees"
        )
    _require_no_authority(plan, field_name="semantic P50 successor-cache plan")
    _require_sha(plan.get("source_revision_sha256"), field_name="source_revision_sha256")
    current_implementation = semantic_p50_successor_cache_implementation_sha256(repo_root=repo_root)
    if (
        plan.get("implementation_sha256") != current_implementation
        or plan.get("policy") != _policy()
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 successor-cache plan implementation or policy is stale"
        )
    base_identity = {
        "source_revision_sha256": plan["source_revision_sha256"],
        "implementation_sha256": plan["implementation_sha256"],
        "policy": plan["policy"],
        "prepared_recipe_binding": plan["prepared_recipe_binding"],
        "validation_contract": plan["validation_contract"],
        "output_prefix": plan["output_prefix"],
    }
    if plan.get("build_identity_sha256") != _sha(base_identity):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 successor-cache build identity disagrees"
        )
    binding = plan.get("prepared_recipe_binding")
    expected_binding_fields = {
        "artifact_path",
        "file_sha256",
        "prepared_recipe_sha256",
        "recipe_policy_sha256",
        "ordered_address_stream_sha256",
        "ordered_training_stream_sha256",
        "requested_address_union_sha256",
        "complete_trace_closure_inventory_sha256",
        "prerequisites",
    }
    if not isinstance(binding, Mapping) or set(binding) != expected_binding_fields:
        raise SemanticP50SuccessorCacheError(
            "semantic P50 successor-cache prepared binding fields disagree"
        )
    for name in expected_binding_fields - {"artifact_path", "prerequisites"}:
        _require_sha(binding.get(name), field_name=f"prepared_recipe_binding.{name}")
    prepared_path = _artifact_path(
        binding["artifact_path"],
        artifact_root=artifact_root,
        field_name="prepared_recipe_binding.artifact_path",
    )
    prepared, file_sha = load_semantic_p50_prepared_recipe(
        prepared_path,
        expected_prepared_recipe_sha256=binding["prepared_recipe_sha256"],
    )
    expected_binding = _prepared_binding(
        prepared,
        path=prepared_path,
        file_sha256=file_sha,
        artifact_root=artifact_root,
    )
    if dict(binding) != expected_binding:
        raise SemanticP50SuccessorCacheError(
            "semantic P50 successor-cache prepared physical binding disagrees"
        )
    validation_contract = _validate_authenticated_validation_contract(
        plan.get("validation_contract"),
        prepared=prepared,
    )

    tasks = plan.get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or plan.get("task_count") != len(tasks)
        or plan.get("task_inventory_sha256") != _sha(tasks)
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 successor-cache task inventory disagrees"
        )
    observed_requested: dict[str, list[dict[str, Any]]] = {
        "train_optimization": [],
        "validation_baseline": [],
    }
    observed_closure: dict[str, list[dict[str, Any]]] = {
        "train_optimization": [],
        "validation_baseline": [],
    }
    previous_group: tuple[str, str, str, str] | None = None
    for task_index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise SemanticP50SuccessorCacheError("cache task must be an object")
        task_body = dict(task)
        task_identity_sha = task_body.pop("task_identity_sha256", None)
        expected_task_fields = {
            "task_index",
            "population_role",
            "partition_role",
            "packed_shard_content_sha256",
            "packed_shard_name",
            "data_lane",
            "requested_addresses",
            "requested_address_inventory_sha256",
            "closure_rows",
            "closure_inventory_sha256",
            "requested_address_count",
            "closure_record_count",
        }
        population_role = task.get("population_role")
        partition_role = task.get("partition_role")
        group = (
            population_role,
            task.get("packed_shard_content_sha256"),
            task.get("packed_shard_name"),
            task.get("data_lane"),
        )
        requested_rows = task.get("requested_addresses")
        closure_rows = task.get("closure_rows")
        if (
            set(task_body) != expected_task_fields
            or task.get("task_index") != task_index
            or not isinstance(requested_rows, list)
            or not isinstance(closure_rows, list)
            or not requested_rows
            or not closure_rows
            or population_role not in {"train_optimization", "validation_baseline"}
            or partition_role
            != ("train" if population_role == "train_optimization" else VALIDATION_PARTITION_ROLE)
            or task.get("requested_address_count") != len(requested_rows)
            or task.get("closure_record_count") != len(closure_rows)
            or task.get("requested_address_inventory_sha256") != _sha(requested_rows)
            or task.get("closure_inventory_sha256") != _sha(closure_rows)
            or not all(isinstance(item, str) and item for item in group)
            or task_identity_sha
            != _sha(
                {
                    "build_identity_sha256": plan["build_identity_sha256"],
                    "task": task_body,
                }
            )
        ):
            raise SemanticP50SuccessorCacheError(
                "semantic P50 successor-cache task identity or ordering disagrees"
            )
        if previous_group is not None and group <= previous_group:
            raise SemanticP50SuccessorCacheError(
                "semantic P50 successor-cache task identity or ordering disagrees"
            )
        previous_group = group
        _require_sha(group[1], field_name="task.packed_shard_content_sha256")
        for address_payload in requested_rows:
            address = _parse_address(address_payload, field_name="task requested address")
            if (
                address.packed_shard_content_sha256,
                address.packed_shard_name,
                address.layer,
            ) != group[1:] or address.partition != partition_role:
                raise SemanticP50SuccessorCacheError(
                    "semantic P50 task requested address differs from its shard"
                )
        task_requested_set = {SuccessorFiberCacheAddress(**item) for item in requested_rows}
        for row in closure_rows:
            if not isinstance(row, Mapping) or set(row) != _NORMALIZED_CLOSURE_FIELDS:
                raise SemanticP50SuccessorCacheError(
                    "semantic P50 task closure row fields disagree"
                )
            address = _parse_address(row["address"], field_name="task closure address")
            if (
                address.packed_shard_content_sha256,
                address.packed_shard_name,
                address.layer,
            ) != group[1:] or address.partition != partition_role:
                raise SemanticP50SuccessorCacheError(
                    "semantic P50 task closure address differs from its shard"
                )
            selected = address in task_requested_set
            if dict(row) != _normalized_closure_row(
                address,
                selected=selected,
                population_role=population_role,
            ):
                raise SemanticP50SuccessorCacheError(
                    "semantic P50 task closure flags differ from its role"
                )
        observed_requested[population_role].extend(requested_rows)
        observed_closure[population_role].extend(closure_rows)
    for rows in observed_requested.values():
        rows.sort(key=lambda item: SuccessorFiberCacheAddress(**item))
    for rows in observed_closure.values():
        rows.sort(key=lambda item: SuccessorFiberCacheAddress(**item["address"]))
    expected_train_closure = [
        _normalized_closure_row(
            SuccessorFiberCacheAddress(**row["address"]),
            selected=bool(row["requested_for_training"]),
            population_role="train_optimization",
        )
        for row in prepared["complete_trace_closure_inventory"]
    ]
    expected_validation_addresses = [
        row["address"] for row in validation_contract["candidate_rows"]
    ]
    if (
        observed_requested["train_optimization"] != prepared["requested_address_union"]
        or observed_closure["train_optimization"] != expected_train_closure
        or observed_requested["validation_baseline"] != expected_validation_addresses
        or observed_closure["validation_baseline"] != validation_contract["closure_rows"]
        or plan.get("train_requested_address_count")
        != len(observed_requested["train_optimization"])
        or plan.get("train_closure_record_count") != len(observed_closure["train_optimization"])
        or plan.get("validation_requested_address_count")
        != len(observed_requested["validation_baseline"])
        or plan.get("validation_closure_record_count")
        != len(observed_closure["validation_baseline"])
        or plan.get("requested_address_count")
        != sum(len(rows) for rows in observed_requested.values())
        or plan.get("closure_record_count") != sum(len(rows) for rows in observed_closure.values())
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache plan differs from the exact prepared address sets"
        )
    run_root = _artifact_path(
        plan.get("run_artifact_root"),
        artifact_root=artifact_root,
        field_name="run_artifact_root",
    )
    expected_output_root = _artifact_path(
        f"{plan.get('output_prefix')}/placeholder",
        artifact_root=artifact_root,
        field_name="output_prefix",
    ).parent
    if run_root != expected_output_root / plan["build_identity_sha256"]:
        raise SemanticP50SuccessorCacheError(
            "semantic P50 successor-cache content-addressed run path disagrees"
        )
    return plan


def _task_for_identity(plan: Mapping[str, Any], task_identity_sha256: str) -> Mapping[str, Any]:
    task = next(
        (item for item in plan["tasks"] if item["task_identity_sha256"] == task_identity_sha256),
        None,
    )
    if task is None:
        raise SemanticP50SuccessorCacheError("semantic P50 cache task is absent from the plan")
    return task


def _validate_compiler_runtime(value: object, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "device",
        "dtype",
        "python_version",
        "torch_version",
        "rdkit_version",
        "implementation_sha256",
        "source_revision_sha256",
        "runtime_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise SemanticP50SuccessorCacheError("semantic P50 cache compiler runtime fields disagree")
    runtime = dict(value)
    body = dict(runtime)
    supplied_sha = body.pop("runtime_sha256")
    if (
        runtime.get("device") != "cpu"
        or runtime.get("dtype") != "torch.float32"
        or runtime.get("implementation_sha256") != plan["implementation_sha256"]
        or runtime.get("source_revision_sha256") != plan["source_revision_sha256"]
        or supplied_sha != _sha(body)
        or any(
            not isinstance(runtime.get(name), str) or not runtime[name]
            for name in ("python_version", "torch_version", "rdkit_version")
        )
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache compiler runtime differs from the plan"
        )
    return runtime


def build_semantic_p50_successor_cache_leaf(
    plan: Mapping[str, Any],
    *,
    task_identity_sha256: str,
    records: Iterable[SuccessorFiberCacheRecord],
    compiler_runtime: Mapping[str, Any],
    artifact_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Bind precompiled production coordinates to one exact plan task."""

    validated_plan = validate_semantic_p50_successor_cache_plan(
        plan, artifact_root=artifact_root, repo_root=repo_root
    )
    task = _task_for_identity(validated_plan, task_identity_sha256)
    runtime = _validate_compiler_runtime(compiler_runtime, plan=validated_plan)
    try:
        ordered = canonical_successor_fiber_records_for_shard(
            tuple(records),
            expected_packed_shard_content_sha256=task["packed_shard_content_sha256"],
            limits=DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
        )
    except (SuccessorFiberCacheError, TypeError, ValueError) as error:
        raise SemanticP50SuccessorCacheError(
            "semantic P50 production successor coordinates are invalid"
        ) from error
    record_payloads = [successor_fiber_cache_record_payload(record) for record in ordered]
    leaf_body = {
        "schema": LEAF_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": LEAF_STATUS,
        **NO_DOWNSTREAM_AUTHORITY,
        "run_identity_sha256": validated_plan["run_identity_sha256"],
        "build_identity_sha256": validated_plan["build_identity_sha256"],
        "task_identity_sha256": task["task_identity_sha256"],
        "prepared_recipe_sha256": validated_plan["prepared_recipe_binding"][
            "prepared_recipe_sha256"
        ],
        "compiler_runtime": runtime,
        "population_role": task["population_role"],
        "partition_role": task["partition_role"],
        "packed_shard_content_sha256": task["packed_shard_content_sha256"],
        "packed_shard_name": task["packed_shard_name"],
        "data_lane": task["data_lane"],
        "requested_addresses": task["requested_addresses"],
        "requested_address_inventory_sha256": task["requested_address_inventory_sha256"],
        "closure_rows": task["closure_rows"],
        "closure_inventory_sha256": task["closure_inventory_sha256"],
        "record_count": len(record_payloads),
        "record_inventory_sha256": _sha(record_payloads),
        "records": record_payloads,
        "model_scores_or_probabilities_stored": False,
        "hazard_coordinates_included": False,
    }
    leaf = {**leaf_body, "leaf_sha256": _sha(leaf_body)}
    return validate_semantic_p50_successor_cache_leaf_for_plan(leaf, plan=validated_plan)


_LEAF_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *NO_DOWNSTREAM_AUTHORITY,
    "run_identity_sha256",
    "build_identity_sha256",
    "task_identity_sha256",
    "prepared_recipe_sha256",
    "compiler_runtime",
    "population_role",
    "partition_role",
    "packed_shard_content_sha256",
    "packed_shard_name",
    "data_lane",
    "requested_addresses",
    "requested_address_inventory_sha256",
    "closure_rows",
    "closure_inventory_sha256",
    "record_count",
    "record_inventory_sha256",
    "records",
    "model_scores_or_probabilities_stored",
    "hazard_coordinates_included",
    "leaf_sha256",
}


def validate_semantic_p50_successor_cache_leaf(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one leaf independently before binding it to a plan task."""

    leaf = dict(value)
    body = dict(leaf)
    supplied_sha = body.pop("leaf_sha256", None)
    records_payload = leaf.get("records")
    if (
        set(leaf) != _LEAF_FIELDS
        or leaf.get("schema") != LEAF_SCHEMA
        or leaf.get("schema_version") != SCHEMA_VERSION
        or leaf.get("status") != LEAF_STATUS
        or supplied_sha != _sha(body)
        or not isinstance(records_payload, list)
        or leaf.get("record_count") != len(records_payload)
        or leaf.get("record_inventory_sha256") != _sha(records_payload)
        or leaf.get("model_scores_or_probabilities_stored") is not False
        or leaf.get("hazard_coordinates_included") is not False
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache leaf schema or self-hash disagrees"
        )
    _require_no_authority(leaf, field_name="semantic P50 cache leaf")
    try:
        records = tuple(successor_fiber_cache_record_from_payload(item) for item in records_payload)
        canonical = canonical_successor_fiber_records_for_shard(
            records,
            expected_packed_shard_content_sha256=leaf["packed_shard_content_sha256"],
            limits=DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
        )
    except (KeyError, TypeError, ValueError, SuccessorFiberCacheError) as error:
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache leaf coordinate records are invalid"
        ) from error
    if records != canonical:
        raise SemanticP50SuccessorCacheError("semantic P50 cache leaf records are not canonical")
    if any(
        record.address.partition != leaf.get("partition_role")
        or record.address.layer != leaf.get("data_lane")
        or record.address.packed_shard_name != leaf.get("packed_shard_name")
        for record in records
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache leaf contains another role, lane, or shard"
        )
    return leaf


def _leaf_address_maps(
    leaf: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    closure = {_sha(row["address"]): dict(row["address"]) for row in leaf["closure_rows"]}
    requested = {_sha(address): dict(address) for address in leaf["requested_addresses"]}
    return closure, requested


def validate_semantic_p50_successor_cache_leaf_for_plan(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Require exact, not merely equal-count, task address coverage."""

    leaf = validate_semantic_p50_successor_cache_leaf(value)
    task = _task_for_identity(plan, leaf["task_identity_sha256"])
    runtime = _validate_compiler_runtime(leaf.get("compiler_runtime"), plan=plan)
    if (
        leaf.get("run_identity_sha256") != plan["run_identity_sha256"]
        or leaf.get("build_identity_sha256") != plan["build_identity_sha256"]
        or leaf.get("prepared_recipe_sha256")
        != plan["prepared_recipe_binding"]["prepared_recipe_sha256"]
        or leaf.get("population_role") != task["population_role"]
        or leaf.get("partition_role") != task["partition_role"]
        or leaf.get("packed_shard_content_sha256") != task["packed_shard_content_sha256"]
        or leaf.get("packed_shard_name") != task["packed_shard_name"]
        or leaf.get("data_lane") != task["data_lane"]
        or leaf.get("requested_addresses") != task["requested_addresses"]
        or leaf.get("requested_address_inventory_sha256")
        != task["requested_address_inventory_sha256"]
        or leaf.get("closure_rows") != task["closure_rows"]
        or leaf.get("closure_inventory_sha256") != task["closure_inventory_sha256"]
        or leaf.get("record_count") != task["closure_record_count"]
        or runtime != leaf["compiler_runtime"]
    ):
        raise SemanticP50SuccessorCacheError("semantic P50 cache leaf differs from its frozen task")
    expected_closure, expected_requested = _leaf_address_maps(leaf)
    observed_records: dict[str, SuccessorFiberCacheRecord] = {}
    for payload in leaf["records"]:
        record = successor_fiber_cache_record_from_payload(payload)
        address_payload = _address_payload(record.address)
        address_sha = _sha(address_payload)
        if address_sha in observed_records:
            raise SemanticP50SuccessorCacheError(
                "semantic P50 cache leaf repeats a progress address"
            )
        observed_records[address_sha] = record
    if set(observed_records) != set(expected_closure):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache leaf address set differs from the exact trace closure"
        )
    if not set(expected_requested).issubset(observed_records):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache leaf misses a requested consumer address"
        )
    if any(
        observed_records[address_sha].teacher_fiber is None
        or observed_records[address_sha].address.is_terminal
        for address_sha in expected_requested
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 requested address is terminal or lacks a teacher fiber"
        )
    return leaf


def _leaf_path(plan: Mapping[str, Any], task: Mapping[str, Any], *, artifact_root: Path) -> Path:
    run_root = _artifact_path(
        plan["run_artifact_root"],
        artifact_root=artifact_root,
        field_name="run_artifact_root",
    )
    return run_root / "tasks" / task["task_identity_sha256"] / LEAF_FILENAME


def write_semantic_p50_successor_cache_artifact(path: Path, value: Mapping[str, Any]) -> bool:
    """Publish canonical bytes immutably, permitting byte-identical reuse."""

    return write_bytes_if_absent(Path(path), _canonical_bytes(dict(value), newline=True))


def build_semantic_p50_successor_cache_manifest(
    plan: Mapping[str, Any],
    *,
    plan_path: Path,
    artifact_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Reduce all immutable leaves and prove both exact address unions."""

    validated_plan = validate_semantic_p50_successor_cache_plan(
        plan, artifact_root=artifact_root, repo_root=repo_root
    )
    run_root = _artifact_path(
        validated_plan["run_artifact_root"],
        artifact_root=artifact_root,
        field_name="run_artifact_root",
    )
    resolved_plan_path = Path(plan_path).resolve()
    if resolved_plan_path != run_root / PLAN_FILENAME:
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache plan is not at its content-addressed path"
        )
    physical_plan, plan_bytes = _load_canonical_object(
        resolved_plan_path,
        field_name="semantic P50 cache plan",
        maximum_bytes=MAX_PLAN_BYTES,
    )
    if physical_plan != validated_plan:
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache plan bytes differ from the reducer input"
        )
    leaves: list[dict[str, Any]] = []
    observed_closure: set[str] = set()
    observed_requested: set[str] = set()
    observed_closure_by_role: dict[str, set[str]] = {
        "train_optimization": set(),
        "validation_baseline": set(),
    }
    observed_requested_by_role: dict[str, set[str]] = {
        "train_optimization": set(),
        "validation_baseline": set(),
    }
    compiler_runtime: dict[str, Any] | None = None
    for task in validated_plan["tasks"]:
        leaf_path = _leaf_path(validated_plan, task, artifact_root=artifact_root)
        leaf, leaf_bytes = _load_canonical_object(
            leaf_path,
            field_name="semantic P50 cache leaf",
            maximum_bytes=MAX_LEAF_BYTES,
        )
        validated_leaf = validate_semantic_p50_successor_cache_leaf_for_plan(
            leaf, plan=validated_plan
        )
        if compiler_runtime is None:
            compiler_runtime = dict(validated_leaf["compiler_runtime"])
        elif compiler_runtime != validated_leaf["compiler_runtime"]:
            raise SemanticP50SuccessorCacheError(
                "semantic P50 cache leaves were compiled in different runtimes"
            )
        closure, requested = _leaf_address_maps(validated_leaf)
        if observed_closure.intersection(closure) or observed_requested.intersection(requested):
            raise SemanticP50SuccessorCacheError(
                "semantic P50 cache repeats an address across leaves"
            )
        observed_closure.update(closure)
        observed_requested.update(requested)
        observed_closure_by_role[task["population_role"]].update(closure)
        observed_requested_by_role[task["population_role"]].update(requested)
        leaves.append(
            {
                "task_identity_sha256": task["task_identity_sha256"],
                "leaf_artifact_path": _artifact_address(leaf_path, artifact_root=artifact_root),
                "leaf_file_sha256": hashlib.sha256(leaf_bytes).hexdigest(),
                "leaf_file_bytes": len(leaf_bytes),
                "leaf_sha256": validated_leaf["leaf_sha256"],
                "record_count": validated_leaf["record_count"],
                "record_inventory_sha256": validated_leaf["record_inventory_sha256"],
                "requested_address_inventory_sha256": validated_leaf[
                    "requested_address_inventory_sha256"
                ],
                "closure_inventory_sha256": validated_leaf["closure_inventory_sha256"],
            }
        )
    expected_closure = {
        _sha(row["address"]) for task in validated_plan["tasks"] for row in task["closure_rows"]
    }
    expected_requested = {
        _sha(address) for task in validated_plan["tasks"] for address in task["requested_addresses"]
    }
    if (
        observed_closure != expected_closure
        or observed_requested != expected_requested
        or len(observed_closure) != validated_plan["closure_record_count"]
        or len(observed_requested) != validated_plan["requested_address_count"]
        or compiler_runtime is None
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache leaves do not exactly cover the planned address sets"
        )
    manifest_body = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": MANIFEST_STATUS,
        **NO_DOWNSTREAM_AUTHORITY,
        "run_identity_sha256": validated_plan["run_identity_sha256"],
        "build_identity_sha256": validated_plan["build_identity_sha256"],
        "source_revision_sha256": validated_plan["source_revision_sha256"],
        "implementation_sha256": validated_plan["implementation_sha256"],
        "policy_sha256": validated_plan["policy"]["policy_sha256"],
        "prepared_recipe_binding": validated_plan["prepared_recipe_binding"],
        "validation_contract": validated_plan["validation_contract"],
        "plan_artifact_path": _artifact_address(resolved_plan_path, artifact_root=artifact_root),
        "plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "plan_file_bytes": len(plan_bytes),
        "plan_sha256": validated_plan["plan_sha256"],
        "compiler_runtime": compiler_runtime,
        "leaf_count": len(leaves),
        "leaf_inventory_sha256": _sha(leaves),
        "leaves": leaves,
        "requested_address_count": len(observed_requested),
        "requested_address_set_sha256": _sha(sorted(observed_requested)),
        "closure_record_count": len(observed_closure),
        "closure_address_set_sha256": _sha(sorted(observed_closure)),
        "train_requested_address_set_sha256": _sha(
            sorted(observed_requested_by_role["train_optimization"])
        ),
        "train_closure_address_set_sha256": _sha(
            sorted(observed_closure_by_role["train_optimization"])
        ),
        "validation_requested_address_set_sha256": _sha(
            sorted(observed_requested_by_role["validation_baseline"])
        ),
        "validation_closure_address_set_sha256": _sha(
            sorted(observed_closure_by_role["validation_baseline"])
        ),
        "model_scores_or_probabilities_stored": False,
        "hazard_coordinates_included": False,
    }
    return {**manifest_body, "manifest_sha256": _sha(manifest_body)}


_MANIFEST_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *NO_DOWNSTREAM_AUTHORITY,
    "run_identity_sha256",
    "build_identity_sha256",
    "source_revision_sha256",
    "implementation_sha256",
    "policy_sha256",
    "prepared_recipe_binding",
    "validation_contract",
    "plan_artifact_path",
    "plan_file_sha256",
    "plan_file_bytes",
    "plan_sha256",
    "compiler_runtime",
    "leaf_count",
    "leaf_inventory_sha256",
    "leaves",
    "requested_address_count",
    "requested_address_set_sha256",
    "closure_record_count",
    "closure_address_set_sha256",
    "train_requested_address_set_sha256",
    "train_closure_address_set_sha256",
    "validation_requested_address_set_sha256",
    "validation_closure_address_set_sha256",
    "model_scores_or_probabilities_stored",
    "hazard_coordinates_included",
    "manifest_sha256",
}


def validate_semantic_p50_successor_cache_manifest(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a manifest against one exact plan."""

    manifest = dict(value)
    body = dict(manifest)
    supplied_sha = body.pop("manifest_sha256", None)
    leaves = manifest.get("leaves")
    if (
        set(manifest) != _MANIFEST_FIELDS
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != MANIFEST_STATUS
        or supplied_sha != _sha(body)
        or manifest.get("run_identity_sha256") != plan["run_identity_sha256"]
        or manifest.get("build_identity_sha256") != plan["build_identity_sha256"]
        or manifest.get("source_revision_sha256") != plan["source_revision_sha256"]
        or manifest.get("implementation_sha256") != plan["implementation_sha256"]
        or manifest.get("policy_sha256") != plan["policy"]["policy_sha256"]
        or manifest.get("prepared_recipe_binding") != plan["prepared_recipe_binding"]
        or manifest.get("validation_contract") != plan["validation_contract"]
        or manifest.get("plan_sha256") != plan["plan_sha256"]
        or not isinstance(leaves, list)
        or manifest.get("leaf_count") != len(leaves)
        or manifest.get("leaf_count") != plan["task_count"]
        or manifest.get("leaf_inventory_sha256") != _sha(leaves)
        or manifest.get("requested_address_count") != plan["requested_address_count"]
        or manifest.get("closure_record_count") != plan["closure_record_count"]
        or manifest.get("model_scores_or_probabilities_stored") is not False
        or manifest.get("hazard_coordinates_included") is not False
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache manifest schema or lineage disagrees"
        )
    _require_no_authority(manifest, field_name="semantic P50 cache manifest")
    _validate_compiler_runtime(manifest.get("compiler_runtime"), plan=plan)
    expected_closure = sorted(
        _sha(row["address"]) for task in plan["tasks"] for row in task["closure_rows"]
    )
    expected_requested = sorted(
        _sha(address) for task in plan["tasks"] for address in task["requested_addresses"]
    )
    train_requested = sorted(
        _sha(address)
        for task in plan["tasks"]
        if task["population_role"] == "train_optimization"
        for address in task["requested_addresses"]
    )
    train_closure = sorted(
        _sha(row["address"])
        for task in plan["tasks"]
        if task["population_role"] == "train_optimization"
        for row in task["closure_rows"]
    )
    validation_requested = sorted(
        _sha(address)
        for task in plan["tasks"]
        if task["population_role"] == "validation_baseline"
        for address in task["requested_addresses"]
    )
    validation_closure = sorted(
        _sha(row["address"])
        for task in plan["tasks"]
        if task["population_role"] == "validation_baseline"
        for row in task["closure_rows"]
    )
    if (
        manifest.get("closure_address_set_sha256") != _sha(expected_closure)
        or manifest.get("requested_address_set_sha256") != _sha(expected_requested)
        or manifest.get("train_requested_address_set_sha256") != _sha(train_requested)
        or manifest.get("train_closure_address_set_sha256") != _sha(train_closure)
        or manifest.get("validation_requested_address_set_sha256") != _sha(validation_requested)
        or manifest.get("validation_closure_address_set_sha256") != _sha(validation_closure)
        or [leaf.get("task_identity_sha256") for leaf in leaves]
        != [task["task_identity_sha256"] for task in plan["tasks"]]
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache manifest exact address identity disagrees"
        )
    expected_leaf_fields = {
        "task_identity_sha256",
        "leaf_artifact_path",
        "leaf_file_sha256",
        "leaf_file_bytes",
        "leaf_sha256",
        "record_count",
        "record_inventory_sha256",
        "requested_address_inventory_sha256",
        "closure_inventory_sha256",
    }
    for task, leaf in zip(plan["tasks"], leaves, strict=True):
        if (
            not isinstance(leaf, Mapping)
            or set(leaf) != expected_leaf_fields
            or leaf.get("record_count") != task["closure_record_count"]
            or leaf.get("requested_address_inventory_sha256")
            != task["requested_address_inventory_sha256"]
            or leaf.get("closure_inventory_sha256") != task["closure_inventory_sha256"]
            or type(leaf.get("leaf_file_bytes")) is not int
            or leaf["leaf_file_bytes"] <= 0
        ):
            raise SemanticP50SuccessorCacheError(
                "semantic P50 cache manifest leaf differs from its task"
            )
        for name in expected_leaf_fields - {
            "leaf_artifact_path",
            "leaf_file_bytes",
            "record_count",
        }:
            _require_sha(leaf.get(name), field_name=f"manifest leaf {name}")
    return manifest


def build_semantic_p50_successor_cache_completion(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    plan_path: Path,
    manifest_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    """Build a no-authority receipt for a physically complete cache."""

    validated_manifest = validate_semantic_p50_successor_cache_manifest(manifest, plan=plan)
    run_root = _artifact_path(
        plan["run_artifact_root"],
        artifact_root=artifact_root,
        field_name="run_artifact_root",
    )
    plan_source = Path(plan_path).resolve()
    manifest_source = Path(manifest_path).resolve()
    if plan_source != run_root / PLAN_FILENAME or manifest_source != run_root / MANIFEST_FILENAME:
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache completion inputs are not exact siblings"
        )
    plan_bytes = plan_source.read_bytes()
    manifest_bytes = manifest_source.read_bytes()
    if plan_bytes != _canonical_bytes(
        dict(plan), newline=True
    ) or manifest_bytes != _canonical_bytes(validated_manifest, newline=True):
        raise SemanticP50SuccessorCacheError("semantic P50 cache completion input bytes disagree")
    prerequisites = plan["prepared_recipe_binding"]["prerequisites"]
    body = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        **NO_DOWNSTREAM_AUTHORITY,
        "run_identity_sha256": plan["run_identity_sha256"],
        "build_identity_sha256": plan["build_identity_sha256"],
        "source_revision_sha256": plan["source_revision_sha256"],
        "implementation_sha256": plan["implementation_sha256"],
        "prepared_recipe_sha256": plan["prepared_recipe_binding"]["prepared_recipe_sha256"],
        "validation_contract_sha256": plan["validation_contract"]["validation_contract_sha256"],
        "source_inventory_sha256": prerequisites["source_inventory_sha256"],
        "process_identity_sha256": prerequisites["process_identity_sha256"],
        "model_runtime_identity_sha256": prerequisites["model_runtime_identity_sha256"],
        "active8_policy_sha256": prerequisites["active8_policy_sha256"],
        "gate_zero_evidence_sha256": prerequisites["gate_zero_evidence_sha256"],
        "t1_decision_sha256": prerequisites["t1_decision_sha256"],
        "scratch_initial_model_state_sha256": prerequisites["scratch_initial_model_state_sha256"],
        "plan_artifact_path": _artifact_address(plan_source, artifact_root=artifact_root),
        "plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "plan_sha256": plan["plan_sha256"],
        "manifest_artifact_path": _artifact_address(manifest_source, artifact_root=artifact_root),
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_sha256": validated_manifest["manifest_sha256"],
        "compiler_runtime_sha256": validated_manifest["compiler_runtime"]["runtime_sha256"],
        "leaf_count": validated_manifest["leaf_count"],
        "requested_address_count": validated_manifest["requested_address_count"],
        "requested_address_set_sha256": validated_manifest["requested_address_set_sha256"],
        "closure_record_count": validated_manifest["closure_record_count"],
        "closure_address_set_sha256": validated_manifest["closure_address_set_sha256"],
        "train_requested_address_count": plan["train_requested_address_count"],
        "train_closure_record_count": plan["train_closure_record_count"],
        "validation_requested_address_count": plan["validation_requested_address_count"],
        "validation_closure_record_count": plan["validation_closure_record_count"],
        "train_requested_address_set_sha256": validated_manifest[
            "train_requested_address_set_sha256"
        ],
        "train_closure_address_set_sha256": validated_manifest["train_closure_address_set_sha256"],
        "validation_requested_address_set_sha256": validated_manifest[
            "validation_requested_address_set_sha256"
        ],
        "validation_closure_address_set_sha256": validated_manifest[
            "validation_closure_address_set_sha256"
        ],
        "stream_union_successor_cache_compiled": True,
        "training_launched": False,
        "next_stage_authorized": None,
    }
    return {**body, "completion_sha256": _sha(body)}


_COMPLETION_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *NO_DOWNSTREAM_AUTHORITY,
    "run_identity_sha256",
    "build_identity_sha256",
    "source_revision_sha256",
    "implementation_sha256",
    "prepared_recipe_sha256",
    "validation_contract_sha256",
    "source_inventory_sha256",
    "process_identity_sha256",
    "model_runtime_identity_sha256",
    "active8_policy_sha256",
    "gate_zero_evidence_sha256",
    "t1_decision_sha256",
    "scratch_initial_model_state_sha256",
    "plan_artifact_path",
    "plan_file_sha256",
    "plan_sha256",
    "manifest_artifact_path",
    "manifest_file_sha256",
    "manifest_sha256",
    "compiler_runtime_sha256",
    "leaf_count",
    "requested_address_count",
    "requested_address_set_sha256",
    "closure_record_count",
    "closure_address_set_sha256",
    "train_requested_address_count",
    "train_closure_record_count",
    "validation_requested_address_count",
    "validation_closure_record_count",
    "train_requested_address_set_sha256",
    "train_closure_address_set_sha256",
    "validation_requested_address_set_sha256",
    "validation_closure_address_set_sha256",
    "stream_union_successor_cache_compiled",
    "training_launched",
    "next_stage_authorized",
    "completion_sha256",
}


def validate_semantic_p50_successor_cache_completion(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact no-authority cache completion receipt."""

    completion = dict(value)
    body = dict(completion)
    supplied_sha = body.pop("completion_sha256", None)
    if (
        set(completion) != _COMPLETION_FIELDS
        or completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("status") != COMPLETION_STATUS
        or supplied_sha != _sha(body)
        or completion.get("stream_union_successor_cache_compiled") is not True
        or completion.get("training_launched") is not False
        or completion.get("next_stage_authorized") is not None
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache completion schema or identity disagrees"
        )
    _require_no_authority(completion, field_name="semantic P50 cache completion")
    for name in _COMPLETION_FIELDS - {
        "schema",
        "schema_version",
        "status",
        *NO_DOWNSTREAM_AUTHORITY,
        "plan_artifact_path",
        "manifest_artifact_path",
        "leaf_count",
        "requested_address_count",
        "closure_record_count",
        "train_requested_address_count",
        "train_closure_record_count",
        "validation_requested_address_count",
        "validation_closure_record_count",
        "stream_union_successor_cache_compiled",
        "training_launched",
        "next_stage_authorized",
    }:
        _require_sha(completion.get(name), field_name=f"completion.{name}")
    count_fields = (
        "leaf_count",
        "requested_address_count",
        "closure_record_count",
        "train_requested_address_count",
        "train_closure_record_count",
        "validation_requested_address_count",
        "validation_closure_record_count",
    )
    if (
        any(type(completion.get(name)) is not int or completion[name] <= 0 for name in count_fields)
        or completion["requested_address_count"]
        != completion["train_requested_address_count"]
        + completion["validation_requested_address_count"]
        or completion["closure_record_count"]
        != completion["train_closure_record_count"] + completion["validation_closure_record_count"]
    ):
        raise SemanticP50SuccessorCacheError("semantic P50 cache completion role census disagrees")
    return completion


@dataclass(frozen=True, slots=True)
class SemanticP50SuccessorCacheExpectedIdentity:
    """Explicit cross-run identities required by the strict opener."""

    prepared_recipe_sha256: str
    source_inventory_sha256: str
    process_identity_sha256: str
    model_runtime_identity_sha256: str
    active8_policy_sha256: str
    gate_zero_evidence_sha256: str
    t1_decision_sha256: str
    scratch_initial_model_state_sha256: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_sha(getattr(self, name), field_name=f"expected.{name}")

    def as_mapping(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def expected_semantic_p50_successor_cache_identity_from_reopened_prerequisites(
    *,
    prepared_recipe_sha256: str,
    source: VerifiedSemanticP50SourceInventory,
    gate_zero_evidence_path: Path,
    t1_decision_path: Path,
    repo_root: Path,
) -> SemanticP50SuccessorCacheExpectedIdentity:
    """Construct opener expectations from independent physical prerequisites."""

    if not isinstance(source, VerifiedSemanticP50SourceInventory):
        raise TypeError("expected cache identity requires a verified semantic source")
    prepared_recipe_sha256 = _require_sha(
        prepared_recipe_sha256, field_name="prepared_recipe_sha256"
    )
    gate_path = Path(gate_zero_evidence_path).resolve()
    t1_path = Path(t1_decision_path).resolve()
    gate_zero = validate_semantic_gate_zero_evidence_receipt(gate_path)
    gate_file_sha = _file_sha(gate_path)
    try:
        t1_payload = json.loads(t1_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticP50SuccessorCacheError(
            "semantic T1 decision is unreadable while constructing cache expectations"
        ) from error
    t1 = validate_semantic_t1_capacity_decision(
        t1_payload,
        decision_path=t1_path,
        repo_root=repo_root,
        expected_gate_zero_evidence_file_sha256=gate_file_sha,
        require_p50_go=True,
    )
    runtime = gate_zero.get("model_runtime_identity")
    policy = gate_zero.get("active8_policy")
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(policy, Mapping)
        or gate_zero.get("structural_result") != "PASS"
        or gate_zero.get("decision_source_inventory_sha256")
        != source.binding.source_inventory_sha256
        or runtime.get("identity_sha256") != source.binding.model_runtime_identity_sha256
        or runtime.get("process_identity_sha256") != source.binding.process_identity_sha256
        or policy.get("policy_sha256") != source.binding.active8_policy_sha256
        or t1.get("status") != DECISION_GO_STATUS
        or t1.get("bounded_p50_authorized") is not True
        or t1.get("failed_checks") != []
        or t1.get("decision_source_inventory_sha256") != source.binding.source_inventory_sha256
        or t1.get("gate_zero_evidence_file_sha256") != gate_file_sha
        or t1.get("initial_model_state_sha256") != runtime.get("initial_model_state_sha256")
    ):
        raise SemanticP50SuccessorCacheError(
            "independently reopened source, Gate0, T1, or scratch identity disagrees"
        )
    return SemanticP50SuccessorCacheExpectedIdentity(
        prepared_recipe_sha256=prepared_recipe_sha256,
        source_inventory_sha256=source.binding.source_inventory_sha256,
        process_identity_sha256=source.binding.process_identity_sha256,
        model_runtime_identity_sha256=source.binding.model_runtime_identity_sha256,
        active8_policy_sha256=source.binding.active8_policy_sha256,
        gate_zero_evidence_sha256=gate_zero["evidence_sha256"],
        t1_decision_sha256=t1["decision_sha256"],
        scratch_initial_model_state_sha256=t1["initial_model_state_sha256"],
    )


@dataclass(frozen=True, slots=True)
class SemanticP50SuccessorCache:
    """Strict load-only view of one physically complete stream-union cache."""

    completion: Mapping[str, Any]
    manifest: Mapping[str, Any]
    records: tuple[SuccessorFiberCacheRecord, ...]
    requested_records: tuple[SuccessorFiberCacheRecord, ...]
    train_requested_records: tuple[SuccessorFiberCacheRecord, ...]
    validation_requested_records: tuple[SuccessorFiberCacheRecord, ...]
    records_by_address_sha256: Mapping[str, SuccessorFiberCacheRecord] = field(repr=False)

    def record_for_address(self, address: SuccessorFiberCacheAddress) -> SuccessorFiberCacheRecord:
        try:
            return self.records_by_address_sha256[_sha(_address_payload(address))]
        except KeyError as error:
            raise KeyError(f"address is absent from semantic P50 cache: {address}") from error


def open_semantic_p50_successor_cache(
    completion_path: Path,
    *,
    artifact_root: Path,
    repo_root: Path,
    expected_identity: SemanticP50SuccessorCacheExpectedIdentity,
    source_reopen_paths: SemanticP50SourceReopenPaths,
    scratch_runtime: SemanticScratchRuntime,
    registry: SemanticCapabilityCellRegistry | None = None,
) -> SemanticP50SuccessorCache:
    """Physically reopen every artifact and independently rederive validation."""

    if not isinstance(expected_identity, SemanticP50SuccessorCacheExpectedIdentity):
        raise TypeError("expected_identity must be SemanticP50SuccessorCacheExpectedIdentity")
    root = Path(artifact_root).resolve()
    completion_source = Path(completion_path).resolve()
    if completion_source.name != COMPLETION_FILENAME or not completion_source.is_relative_to(root):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache completion is outside its artifact root or misnamed"
        )
    completion, _ = _load_canonical_object(
        completion_source,
        field_name="semantic P50 cache completion",
        maximum_bytes=MAX_COMPLETION_BYTES,
    )
    validated_completion = validate_semantic_p50_successor_cache_completion(completion)
    if any(
        validated_completion[name] != value
        for name, value in expected_identity.as_mapping().items()
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache differs from an explicit expected cross-run identity"
        )
    plan_path = _artifact_path(
        validated_completion["plan_artifact_path"],
        artifact_root=root,
        field_name="completion.plan_artifact_path",
    )
    manifest_path = _artifact_path(
        validated_completion["manifest_artifact_path"],
        artifact_root=root,
        field_name="completion.manifest_artifact_path",
    )
    if (
        plan_path.parent != completion_source.parent
        or manifest_path.parent != completion_source.parent
        or plan_path.name != PLAN_FILENAME
        or manifest_path.name != MANIFEST_FILENAME
    ):
        raise SemanticP50SuccessorCacheError("semantic P50 cache completion siblings disagree")
    plan, plan_bytes = _load_canonical_object(
        plan_path,
        field_name="semantic P50 cache plan",
        maximum_bytes=MAX_PLAN_BYTES,
    )
    validated_plan = validate_semantic_p50_successor_cache_plan(
        plan, artifact_root=root, repo_root=repo_root
    )
    physically_reopened_source = _open_verified_source(
        source_reopen_paths,
        prerequisites=validated_plan["prepared_recipe_binding"]["prerequisites"],
        artifact_root=root,
        repo_root=repo_root,
    )
    selected_registry = registry or load_semantic_capability_cell_registry()
    rederived_validation_contract = _derive_authenticated_validation_contract(
        physically_reopened_source,
        prepared=load_semantic_p50_prepared_recipe(
            _artifact_path(
                validated_plan["prepared_recipe_binding"]["artifact_path"],
                artifact_root=root,
                field_name="prepared_recipe_binding.artifact_path",
            ),
            expected_prepared_recipe_sha256=validated_plan["prepared_recipe_binding"][
                "prepared_recipe_sha256"
            ],
        )[0],
        registry=selected_registry,
    )
    if rederived_validation_contract != validated_plan["validation_contract"]:
        raise SemanticP50SuccessorCacheError(
            "physical validation source differs from the frozen cache plan"
        )
    manifest, manifest_bytes = _load_canonical_object(
        manifest_path,
        field_name="semantic P50 cache manifest",
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    validated_manifest = validate_semantic_p50_successor_cache_manifest(
        manifest, plan=validated_plan
    )
    if (
        hashlib.sha256(plan_bytes).hexdigest() != validated_completion["plan_file_sha256"]
        or validated_plan["plan_sha256"] != validated_completion["plan_sha256"]
        or hashlib.sha256(manifest_bytes).hexdigest()
        != validated_completion["manifest_file_sha256"]
        or validated_manifest["manifest_sha256"] != validated_completion["manifest_sha256"]
        or validated_manifest["compiler_runtime"]["runtime_sha256"]
        != validated_completion["compiler_runtime_sha256"]
        or validated_manifest["leaf_count"] != validated_completion["leaf_count"]
        or validated_manifest["requested_address_count"]
        != validated_completion["requested_address_count"]
        or validated_manifest["requested_address_set_sha256"]
        != validated_completion["requested_address_set_sha256"]
        or validated_manifest["closure_record_count"]
        != validated_completion["closure_record_count"]
        or validated_manifest["closure_address_set_sha256"]
        != validated_completion["closure_address_set_sha256"]
        or validated_completion["validation_contract_sha256"]
        != validated_plan["validation_contract"]["validation_contract_sha256"]
        or validated_manifest["train_requested_address_set_sha256"]
        != validated_completion["train_requested_address_set_sha256"]
        or validated_manifest["train_closure_address_set_sha256"]
        != validated_completion["train_closure_address_set_sha256"]
        or validated_manifest["validation_requested_address_set_sha256"]
        != validated_completion["validation_requested_address_set_sha256"]
        or validated_manifest["validation_closure_address_set_sha256"]
        != validated_completion["validation_closure_address_set_sha256"]
        or validated_plan["train_requested_address_count"]
        != validated_completion["train_requested_address_count"]
        or validated_plan["train_closure_record_count"]
        != validated_completion["train_closure_record_count"]
        or validated_plan["validation_requested_address_count"]
        != validated_completion["validation_requested_address_count"]
        or validated_plan["validation_closure_record_count"]
        != validated_completion["validation_closure_record_count"]
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache completion disagrees with physical plan or manifest"
        )

    all_records: list[SuccessorFiberCacheRecord] = []
    by_address: dict[str, SuccessorFiberCacheRecord] = {}
    requested_hashes: set[str] = set()
    requested_hashes_by_role: dict[str, set[str]] = {
        "train_optimization": set(),
        "validation_baseline": set(),
    }
    for task, leaf_spec in zip(validated_plan["tasks"], validated_manifest["leaves"], strict=True):
        leaf_path = _artifact_path(
            leaf_spec["leaf_artifact_path"],
            artifact_root=root,
            field_name="manifest.leaf_artifact_path",
        )
        if leaf_path != _leaf_path(validated_plan, task, artifact_root=root):
            raise SemanticP50SuccessorCacheError(
                "semantic P50 cache leaf is not at its content-addressed path"
            )
        leaf, leaf_bytes = _load_canonical_object(
            leaf_path,
            field_name="semantic P50 cache leaf",
            maximum_bytes=MAX_LEAF_BYTES,
        )
        validated_leaf = validate_semantic_p50_successor_cache_leaf_for_plan(
            leaf, plan=validated_plan
        )
        if (
            hashlib.sha256(leaf_bytes).hexdigest() != leaf_spec["leaf_file_sha256"]
            or len(leaf_bytes) != leaf_spec["leaf_file_bytes"]
            or validated_leaf["leaf_sha256"] != leaf_spec["leaf_sha256"]
            or validated_leaf["record_inventory_sha256"] != leaf_spec["record_inventory_sha256"]
        ):
            raise SemanticP50SuccessorCacheError(
                "semantic P50 cache leaf differs from the physical manifest"
            )
        for payload in validated_leaf["records"]:
            record = successor_fiber_cache_record_from_payload(payload)
            address_sha = _sha(_address_payload(record.address))
            if address_sha in by_address:
                raise SemanticP50SuccessorCacheError(
                    "semantic P50 cache repeats an address across leaves"
                )
            by_address[address_sha] = record
            all_records.append(record)
        leaf_requested_hashes = {_sha(address) for address in validated_leaf["requested_addresses"]}
        requested_hashes.update(leaf_requested_hashes)
        requested_hashes_by_role[task["population_role"]].update(leaf_requested_hashes)
    if (
        len(all_records) != validated_manifest["closure_record_count"]
        or len(requested_hashes) != validated_manifest["requested_address_count"]
        or not requested_hashes.issubset(by_address)
        or len(requested_hashes_by_role["train_optimization"])
        != validated_completion["train_requested_address_count"]
        or len(requested_hashes_by_role["validation_baseline"])
        != validated_completion["validation_requested_address_count"]
    ):
        raise SemanticP50SuccessorCacheError(
            "semantic P50 cache loaded address census differs from the manifest"
        )
    _authenticate_cache_records_against_source(
        source=physically_reopened_source,
        plan=validated_plan,
        records_by_address_sha256=by_address,
        scratch_runtime=scratch_runtime,
    )
    requested_records = tuple(by_address[address_sha] for address_sha in sorted(requested_hashes))
    train_requested_records = tuple(
        by_address[address_sha]
        for address_sha in sorted(requested_hashes_by_role["train_optimization"])
    )
    validation_requested_records = tuple(
        by_address[address_sha]
        for address_sha in sorted(requested_hashes_by_role["validation_baseline"])
    )
    return SemanticP50SuccessorCache(
        completion=MappingProxyType(dict(validated_completion)),
        manifest=MappingProxyType(dict(validated_manifest)),
        records=tuple(all_records),
        requested_records=requested_records,
        train_requested_records=train_requested_records,
        validation_requested_records=validation_requested_records,
        records_by_address_sha256=MappingProxyType(by_address),
    )


__all__ = [
    "COMPLETION_FILENAME",
    "DEFAULT_OUTPUT_PREFIX",
    "LEAF_FILENAME",
    "MANIFEST_FILENAME",
    "PLAN_FILENAME",
    "SUPPORT_COMPILATION_TIME",
    "SemanticP50SuccessorCache",
    "SemanticP50SuccessorCacheError",
    "SemanticP50SuccessorCacheExpectedIdentity",
    "SemanticP50SourceReopenPaths",
    "build_semantic_p50_successor_cache_completion",
    "build_semantic_p50_successor_cache_leaf",
    "build_semantic_p50_successor_cache_manifest",
    "build_semantic_p50_successor_cache_plan",
    "expected_semantic_p50_successor_cache_identity_from_reopened_prerequisites",
    "load_semantic_p50_prepared_recipe",
    "open_semantic_p50_successor_cache",
    "semantic_p50_successor_cache_implementation_sha256",
    "validate_semantic_p50_prepared_recipe",
    "validate_semantic_p50_successor_cache_completion",
    "validate_semantic_p50_successor_cache_leaf",
    "validate_semantic_p50_successor_cache_leaf_for_plan",
    "validate_semantic_p50_successor_cache_manifest",
    "validate_semantic_p50_successor_cache_plan",
    "write_semantic_p50_successor_cache_artifact",
]
