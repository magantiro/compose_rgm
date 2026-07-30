"""Fail-closed runtime for one real T1 canonical-successor capacity arm.

The runtime consumes the exact frozen validation source already qualified by
Gate 0, resolves one immutable family panel, replays its recorded teacher
actions, compiles and round-trips production successor-cache records, and only
then performs bounded scratch-model micro-overfit.  Its result is diagnostic:
there are no frozen numeric thresholds and no path to a GO decision.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheRecord,
    deserialize_successor_fiber_cache,
    serialize_successor_fiber_cache,
)
from compose_v4.experiments.editing_gate_zero_runtime import (
    FrozenValidationSource,
    GateZeroRuntimeContract,
    build_exact_cache_provenance,
    build_scratch_ringcore_model,
)
from compose_v4.experiments.editing_p50_gate import (
    state_dict_semantic_sha256,
)
from compose_v4.experiments.editing_t1_panel import (
    EDITING_T1_DIAGNOSTIC_ONLY_FAMILIES,
    EDITING_T1_GLOBAL_FAMILY_SELECTOR,
    EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
    EDITING_T1_REQUIRED_SCOPES,
    EDITING_T1_SUPPORT_TIME,
    EDITING_T1_UNIQUE_PANEL_KIND,
    EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
    EditingT1PanelError,
    selected_source_rows,
)
from compose_v4.experiments.factorized_successor_training import (
    rewrite_action_codec_sha256,
)
from compose_v4.experiments.successor_fiber_cache_builder import (
    SuccessorFiberCacheBuildError,
    compile_successor_fiber_trace,
)
from compose_v4.experiments.successor_micro_overfit import (
    RINGCORE_EDITING_FAMILIES,
    PreparedSuccessorPanel,
    SuccessorSupervisionExample,
    prepare_cached_successor_panel,
    train_successor_micro_panel,
)
from compose_v4.rewrite.action_codec import canonical_family
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    de_novo_rewrite_system,
)

EDITING_T1_RUNTIME_CONTRACT_SCHEMA = "compose.editing.t1_successor_runtime_contract"
EDITING_T1_RUNTIME_CONTRACT_VERSION = 3
EDITING_T1_RUNTIME_CONTRACT_STATUS = "FROZEN_BOUNDED_CAPACITY_DIAGNOSTIC_NO_TRAINING_AUTHORITY"
EDITING_T1_RESULT_SCHEMA = "compose.editing.t1_successor_capacity_result"
EDITING_T1_RESULT_VERSION = 3
EDITING_T1_RESULT_STATUS = "CAPACITY_DIAGNOSTIC_COMPLETE_NO_GATE_DECISION"

_CONTRACT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "gate_zero_runtime_contract_sha256",
    "panel_artifact_sha256",
    "forensics_file_sha256",
    "charge_policy_audit_file_sha256",
    "charge_policy_exclusions_file_sha256",
    "charge_policy_exclusion_payload_sha256",
    "charge_policy_source_input_inventory_sha256",
    "implementation_sha256",
    "families",
    "panel_kinds",
    "optimization",
    "thresholds",
    "contract_sha256",
}
_OPTIMIZATION_FIELDS = {
    "steps",
    "learning_rate",
    "weight_decay",
    "scopes",
    "report_points",
}
_THRESHOLD_FIELDS = {
    "minimum_unique_state_teacher_successor_top1",
    "minimum_unique_state_teacher_successor_probability",
    "maximum_unique_state_teacher_successor_nll",
    "maximum_global_repeated_state_excess_nll_over_empirical_entropy",
}
_RESULT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "gate_decision",
    "numeric_thresholds_frozen",
    "contract_sha256",
    "gate_zero_runtime_contract_sha256",
    "panel_artifact_sha256",
    "charge_policy_exclusion_payload_sha256",
    "charge_policy_source_input_inventory_sha256",
    "family",
    "family_status",
    "panel_kind",
    "panel_role",
    "scope",
    "operator_identity",
    "example_count",
    "unique_progress_address_count",
    "repeated_progress_observation_count",
    "source_row_sha256s",
    "scratch_initialization",
    "cache_receipts",
    "training_report",
    "final_model_state_sha256",
    "result_sha256",
}
_CACHE_RECEIPT_FIELDS = {
    "packed_shard_content_sha256",
    "cache_content_sha256",
    "encoded_sha256",
    "record_count",
}


class EditingT1RuntimeError(RuntimeError):
    """A real T1 arm is off-contract or scientifically incomplete."""


_IMPLEMENTATION_ENTRYPOINT_SOURCES = (
    "modal_apps/run_editing_t1_successor_gate.py",
    "scripts/build_editing_t1_successor_panel.py",
    "scripts/run_editing_t1_successor_gate.py",
    "src/compose_v4/experiments/editing_gate_zero_runtime.py",
    "src/compose_v4/experiments/editing_t1_panel.py",
    "src/compose_v4/experiments/editing_t1_successor_runtime.py",
    "src/compose_v4/experiments/successor_micro_overfit.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/experiments/production_successor_kernel.py",
    "src/compose_v4/experiments/successor_fiber_cache_builder.py",
    "src/compose_v4/data/successor_fiber_cache.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
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
        raise EditingT1RuntimeError("T1 runtime metadata is not finite canonical JSON") from error


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _module_source_path(repository: Path, module_name: str) -> Path | None:
    if not module_name.startswith("compose_v4"):
        return None
    module_path = repository / "src" / Path(*module_name.split("."))
    source_path = module_path.with_suffix(".py")
    if source_path.is_file():
        return source_path
    package_path = module_path / "__init__.py"
    return package_path if package_path.is_file() else None


def _module_name_for_source(repository: Path, source: Path) -> tuple[str | None, bool]:
    try:
        relative = source.relative_to(repository / "src")
    except ValueError:
        return None, False
    parts = list(relative.parts)
    is_package = parts[-1] == "__init__.py"
    if is_package:
        parts.pop()
    else:
        parts[-1] = Path(parts[-1]).stem
    return ".".join(parts), is_package


def _imported_local_sources(repository: Path, source: Path) -> set[Path]:
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise EditingT1RuntimeError(
            f"T1 implementation source cannot be parsed: {source}"
        ) from error
    current_module, is_package = _module_name_for_source(repository, source)
    current_package = None
    if current_module is not None:
        current_package = current_module if is_package else current_module.rpartition(".")[0]
    discovered: set[Path] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if not current_package:
                    raise EditingT1RuntimeError(
                        f"T1 implementation has an unresolved relative import: {source}"
                    )
                package_parts = current_package.split(".")
                retained = len(package_parts) - node.level + 1
                if retained < 0:
                    raise EditingT1RuntimeError(
                        f"T1 implementation relative import escapes its package: {source}"
                    )
                base_parts = package_parts[:retained]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                candidates.append(base)
                candidates.extend(f"{base}.{alias.name}" for alias in node.names)
        for module_name in candidates:
            imported = _module_source_path(repository, module_name)
            if imported is not None:
                discovered.add(imported)
    return discovered


def _package_initializers(repository: Path, source: Path) -> set[Path]:
    try:
        relative = source.relative_to(repository / "src")
    except ValueError:
        return set()
    initializers: set[Path] = set()
    parent = relative.parent
    while parent.parts:
        candidate = repository / "src" / parent / "__init__.py"
        if candidate.is_file():
            initializers.add(candidate)
        parent = parent.parent
    return initializers


def editing_t1_implementation_sources() -> tuple[str, ...]:
    """Return the complete static local-import closure for a T1 arm.

    The roots cover panel derivation, Gate-0 loading/scratch construction,
    production cache compilation, model forward/loss, and both local and Modal
    entrypoints. Package initializers and every statically imported
    ``compose_v4`` module are included recursively so a behavior dependency
    cannot drift while retaining the same T1 contract identity.
    """

    repository = Path(__file__).resolve().parents[3]
    pending = [repository / relative for relative in _IMPLEMENTATION_ENTRYPOINT_SOURCES]
    sources: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in sources:
            continue
        if not source.is_file():
            relative = source.relative_to(repository)
            raise EditingT1RuntimeError(f"T1 implementation source is absent: {relative}")
        sources.add(source)
        dependencies = _imported_local_sources(repository, source)
        dependencies.update(_package_initializers(repository, source))
        pending.extend(dependencies - sources)
    return tuple(sorted(path.relative_to(repository).as_posix() for path in sources))


def editing_t1_implementation_sha256() -> str:
    """Hash the complete local behavior closure that defines one arm."""

    repository = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for relative in editing_t1_implementation_sources():
        source = repository / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EditingT1RuntimeError(f"{name} must be an object")
    if set(value) != expected:
        raise EditingT1RuntimeError(
            f"{name} fields disagree; "
            f"missing={sorted(expected - set(value))!r}, "
            f"unexpected={sorted(set(value) - expected)!r}"
        )
    return value


@dataclass(frozen=True)
class EditingT1RuntimeContract:
    """Exact inputs and bounded optimizer settings for T1."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        frozen = dict(self.payload)
        _require_exact_fields(
            frozen,
            _CONTRACT_FIELDS,
            name="T1 runtime contract",
        )
        body = {key: value for key, value in frozen.items() if key != "contract_sha256"}
        if (
            frozen["schema"] != EDITING_T1_RUNTIME_CONTRACT_SCHEMA
            or frozen["schema_version"] != EDITING_T1_RUNTIME_CONTRACT_VERSION
            or frozen["status"] != EDITING_T1_RUNTIME_CONTRACT_STATUS
            or frozen["training_authorized"] is not False
            or not _is_sha256(frozen["contract_sha256"])
            or frozen["contract_sha256"] != _stable_sha256(body)
        ):
            raise EditingT1RuntimeError("T1 runtime contract identity or self-hash is invalid")
        for field in (
            "gate_zero_runtime_contract_sha256",
            "panel_artifact_sha256",
            "forensics_file_sha256",
            "charge_policy_audit_file_sha256",
            "charge_policy_exclusions_file_sha256",
            "charge_policy_exclusion_payload_sha256",
            "charge_policy_source_input_inventory_sha256",
            "implementation_sha256",
        ):
            if not _is_sha256(frozen[field]):
                raise EditingT1RuntimeError(f"T1 contract {field} must be a SHA-256")
        if frozen["implementation_sha256"] != (editing_t1_implementation_sha256()):
            raise EditingT1RuntimeError("T1 implementation source hash drifted")
        if tuple(frozen["families"]) != RINGCORE_EDITING_FAMILIES:
            raise EditingT1RuntimeError(
                "T1 contract must name the ordered eight active pilot families"
            )
        if tuple(frozen["panel_kinds"]) != (
            EDITING_T1_UNIQUE_PANEL_KIND,
            EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
            EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
        ):
            raise EditingT1RuntimeError("T1 contract panel-kind ladder drifted")
        optimization = _require_exact_fields(
            frozen["optimization"],
            _OPTIMIZATION_FIELDS,
            name="T1 optimization",
        )
        if (
            type(optimization["steps"]) is not int
            or not 0 < optimization["steps"] <= 500
            or not isinstance(optimization["learning_rate"], (int, float))
            or isinstance(optimization["learning_rate"], bool)
            or not math.isfinite(float(optimization["learning_rate"]))
            or float(optimization["learning_rate"]) <= 0.0
            or not isinstance(optimization["weight_decay"], (int, float))
            or isinstance(optimization["weight_decay"], bool)
            or not math.isfinite(float(optimization["weight_decay"]))
            or float(optimization["weight_decay"]) < 0.0
            or tuple(optimization["scopes"]) != EDITING_T1_REQUIRED_SCOPES
        ):
            raise EditingT1RuntimeError("T1 optimization settings are invalid or off-ladder")
        points = optimization["report_points"]
        if (
            not isinstance(points, list)
            or any(type(point) is not int for point in points)
            or points != sorted(set(points))
            or any(point <= 0 or point > optimization["steps"] for point in points)
        ):
            raise EditingT1RuntimeError("T1 report points must be sorted unique in-range integers")
        thresholds = _require_exact_fields(
            frozen["thresholds"],
            _THRESHOLD_FIELDS,
            name="T1 thresholds",
        )
        if any(value is not None for value in thresholds.values()):
            raise EditingT1RuntimeError("T1 thresholds must remain explicitly null")
        object.__setattr__(self, "payload", MappingProxyType(frozen))

    @property
    def sha256(self) -> str:
        return str(self.payload["contract_sha256"])

    @property
    def optimization(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.payload["optimization"]))

    @property
    def training_authorized(self) -> bool:
        return False

    def assert_training_authorized(self) -> None:
        raise EditingT1RuntimeError(
            "T1 is a bounded development diagnostic and never authorizes training"
        )


def load_editing_t1_runtime_contract(
    path: Path,
) -> EditingT1RuntimeContract:
    source = Path(path)
    if not source.is_file():
        raise EditingT1RuntimeError(f"T1 runtime contract is absent: {source}")
    try:
        payload = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingT1RuntimeError("T1 runtime contract is invalid JSON") from error
    return EditingT1RuntimeContract(payload)


@dataclass(frozen=True)
class T1CacheShardReceipt:
    """Exact serialized cache identity for one packed source shard."""

    packed_shard_content_sha256: str
    cache_content_sha256: str
    encoded_sha256: str
    record_count: int

    def __post_init__(self) -> None:
        for name in (
            "packed_shard_content_sha256",
            "cache_content_sha256",
            "encoded_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise EditingT1RuntimeError(f"T1 cache receipt {name} must be a SHA-256")
        if type(self.record_count) is not int or self.record_count <= 0:
            raise EditingT1RuntimeError("T1 cache receipt record_count must be positive")


def validate_t1_cache_shard_receipt(
    value: Mapping[str, Any] | T1CacheShardReceipt,
    *,
    expected: Mapping[str, Any] | T1CacheShardReceipt | None = None,
) -> T1CacheShardReceipt:
    """Validate one durable cache receipt and, when supplied, its exact identity."""

    if isinstance(value, T1CacheShardReceipt):
        receipt = value
    else:
        payload = _require_exact_fields(
            value,
            _CACHE_RECEIPT_FIELDS,
            name="T1 cache shard receipt",
        )
        receipt = T1CacheShardReceipt(**payload)
    if expected is not None:
        expected_receipt = validate_t1_cache_shard_receipt(expected)
        if receipt != expected_receipt:
            raise EditingT1RuntimeError(
                "T1 cache receipt disagrees with its expected shard/content/encoding/count identity"
            )
    return receipt


@dataclass(frozen=True)
class MaterializedT1Panel:
    """Exact examples and round-tripped production cache fibers."""

    family: str
    panel_kind: str
    examples: tuple[SuccessorSupervisionExample, ...]
    prepared: PreparedSuccessorPanel
    cache_receipts: tuple[T1CacheShardReceipt, ...]
    source_row_sha256s: tuple[str, ...]
    unique_progress_address_count: int


def _validated_receipt_sequence(
    value: object,
    *,
    name: str,
) -> tuple[T1CacheShardReceipt, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EditingT1RuntimeError(f"{name} must be a receipt sequence")
    receipts = tuple(validate_t1_cache_shard_receipt(item) for item in value)
    if not receipts:
        raise EditingT1RuntimeError(f"{name} cannot be empty")
    shard_digests = tuple(receipt.packed_shard_content_sha256 for receipt in receipts)
    if shard_digests != tuple(sorted(shard_digests)) or len(shard_digests) != len(
        set(shard_digests)
    ):
        raise EditingT1RuntimeError(
            f"{name} must contain one receipt per shard in canonical digest order"
        )
    return receipts


def validate_editing_t1_result(
    result: Mapping[str, Any],
    *,
    expected_result_sha256: str,
    expected_contract_sha256: str,
    expected_panel_artifact_sha256: str,
    expected_cache_receipts: Sequence[Mapping[str, Any] | T1CacheShardReceipt],
) -> Mapping[str, Any]:
    """Validate one durable T1 result against independently retained identities."""

    for name, value in (
        ("expected_result_sha256", expected_result_sha256),
        ("expected_contract_sha256", expected_contract_sha256),
        ("expected_panel_artifact_sha256", expected_panel_artifact_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    payload = _require_exact_fields(result, _RESULT_FIELDS, name="T1 durable result")
    body = {key: value for key, value in payload.items() if key != "result_sha256"}
    if (
        payload["schema"] != EDITING_T1_RESULT_SCHEMA
        or payload["schema_version"] != EDITING_T1_RESULT_VERSION
        or payload["status"] != EDITING_T1_RESULT_STATUS
        or payload["training_authorized"] is not False
        or payload["gate_decision"] is not None
        or payload["numeric_thresholds_frozen"] is not False
        or not _is_sha256(payload["result_sha256"])
        or payload["result_sha256"] != _stable_sha256(body)
        or payload["result_sha256"] != expected_result_sha256
    ):
        raise EditingT1RuntimeError("T1 durable result identity, status, or self-hash is invalid")
    if payload["contract_sha256"] != expected_contract_sha256:
        raise EditingT1RuntimeError("T1 durable result is bound to another runtime contract")
    if payload["panel_artifact_sha256"] != expected_panel_artifact_sha256:
        raise EditingT1RuntimeError("T1 durable result is bound to another panel artifact")
    for field in (
        "gate_zero_runtime_contract_sha256",
        "charge_policy_exclusion_payload_sha256",
        "charge_policy_source_input_inventory_sha256",
        "final_model_state_sha256",
    ):
        if not _is_sha256(payload[field]):
            raise EditingT1RuntimeError(f"T1 durable result {field} must be a SHA-256")

    family = payload["family"]
    panel_kind = payload["panel_kind"]
    scope = payload["scope"]
    if family != EDITING_T1_GLOBAL_FAMILY_SELECTOR and family not in RINGCORE_EDITING_FAMILIES:
        raise EditingT1RuntimeError("T1 durable result family is off-contract")
    if panel_kind not in (
        EDITING_T1_UNIQUE_PANEL_KIND,
        EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
        EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
    ):
        raise EditingT1RuntimeError("T1 durable result panel kind is off-contract")
    if (family == EDITING_T1_GLOBAL_FAMILY_SELECTOR) != (
        panel_kind == EDITING_T1_GLOBAL_REPEATED_PANEL_KIND
    ):
        raise EditingT1RuntimeError(
            "T1 durable result global family/panel relationship is invalid"
        )
    if scope not in EDITING_T1_REQUIRED_SCOPES:
        raise EditingT1RuntimeError("T1 durable result scope is off-contract")

    example_count = payload["example_count"]
    unique_address_count = payload["unique_progress_address_count"]
    repeated_observation_count = payload["repeated_progress_observation_count"]
    if (
        type(example_count) is not int
        or example_count <= 0
        or type(unique_address_count) is not int
        or not 0 < unique_address_count <= example_count
        or type(repeated_observation_count) is not int
        or repeated_observation_count != example_count - unique_address_count
    ):
        raise EditingT1RuntimeError("T1 durable result example/address counts are invalid")
    source_row_sha256s = payload["source_row_sha256s"]
    if (
        not isinstance(source_row_sha256s, list)
        or len(source_row_sha256s) != example_count
        or any(not _is_sha256(value) for value in source_row_sha256s)
    ):
        raise EditingT1RuntimeError("T1 durable result source-row identities are invalid")
    if panel_kind == EDITING_T1_UNIQUE_PANEL_KIND and repeated_observation_count != 0:
        raise EditingT1RuntimeError("T1 unique-state result repeats a progress address")

    operator_identity = payload["operator_identity"]
    if (
        not isinstance(operator_identity, Mapping)
        or set(operator_identity)
        != {
            "operator_capability_fingerprint",
            "enable_ring_system_delete",
            "enable_ring_grow_macro",
            "enable_cycle_ops",
        }
        or not isinstance(operator_identity["operator_capability_fingerprint"], str)
        or not operator_identity["operator_capability_fingerprint"]
        or operator_identity["enable_ring_system_delete"] is not False
        or operator_identity["enable_ring_grow_macro"] is not False
        or operator_identity["enable_cycle_ops"] is not True
    ):
        raise EditingT1RuntimeError("T1 durable result operator identity is invalid")
    scratch = payload["scratch_initialization"]
    if (
        not isinstance(scratch, Mapping)
        or scratch.get("regime") != "scratch"
        or not _is_sha256(scratch.get("transfer_plan_sha256"))
        or not _is_sha256(scratch.get("initial_model_state_sha256"))
    ):
        raise EditingT1RuntimeError("T1 durable result scratch initialization is invalid")
    training_report = payload["training_report"]
    if (
        not isinstance(training_report, Mapping)
        or training_report.get("scope") != scope
        or training_report.get("steps") is None
        or type(training_report["steps"]) is not int
        or not 0 < training_report["steps"] <= 500
    ):
        raise EditingT1RuntimeError("T1 durable result training report is invalid")

    observed_receipts = _validated_receipt_sequence(
        payload["cache_receipts"],
        name="T1 durable result cache_receipts",
    )
    retained_receipts = _validated_receipt_sequence(
        expected_cache_receipts,
        name="expected T1 cache receipts",
    )
    if observed_receipts != retained_receipts:
        raise EditingT1RuntimeError(
            "T1 durable result cache receipts disagree with expected "
            "shard/content/encoding/record-count identities"
        )
    return MappingProxyType(dict(payload))


def load_editing_t1_result(
    path: Path,
    *,
    expected_result_sha256: str,
    expected_contract_sha256: str,
    expected_panel_artifact_sha256: str,
    expected_cache_receipts: Sequence[Mapping[str, Any] | T1CacheShardReceipt],
    max_file_bytes: int = 1 << 27,
) -> Mapping[str, Any]:
    """Load a bounded durable result and validate every externally retained identity."""

    source = Path(path)
    if (
        type(max_file_bytes) is not int
        or max_file_bytes <= 0
        or not source.is_file()
        or source.stat().st_size > max_file_bytes
    ):
        raise EditingT1RuntimeError("T1 durable result is absent or exceeds its bound")
    try:
        result = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingT1RuntimeError("T1 durable result is invalid JSON") from error
    return validate_editing_t1_result(
        result,
        expected_result_sha256=expected_result_sha256,
        expected_contract_sha256=expected_contract_sha256,
        expected_panel_artifact_sha256=expected_panel_artifact_sha256,
        expected_cache_receipts=expected_cache_receipts,
    )


def _record_index(
    source: FrozenValidationSource,
) -> Mapping[tuple[str, int], Any]:
    result: dict[tuple[str, int], Any] = {}
    for record in source.records:
        address = record.corpus_address
        if address is None:
            raise EditingT1RuntimeError("frozen validation source contains an unaddressed record")
        key = (
            address.packed_shard_content_sha256,
            address.entry_index,
        )
        if key in result:
            raise EditingT1RuntimeError("frozen validation source repeats an exact packed entry")
        result[key] = record
    return MappingProxyType(result)


def _resolve_example(
    row: Mapping[str, Any],
    *,
    record_by_address: Mapping[tuple[str, int], Any],
    semantic_cells: Mapping[tuple[str, int, int], str | None],
    family_selector: str,
    panel_kind: str,
) -> tuple[SuccessorSupervisionExample, SuccessorFiberCacheAddress]:
    state_ref = row["exact_state_ref"]
    exact_trace_key = (
        str(state_ref["shard_sha256"]),
        int(state_ref["record_index"]),
    )
    try:
        record = record_by_address[exact_trace_key]
    except KeyError:
        raise EditingT1RuntimeError(
            "T1 row addresses a trace outside the frozen validation source"
        ) from None
    address = record.corpus_address
    assert address is not None
    progress = int(state_ref["progress_index"])
    if (
        progress != int(row["progress_index"])
        or progress >= record.path.path_length
        or address.path_length != int(row["path_length"])
        or Path(str(state_ref["shard_name"])).name != address.packed_shard_name
    ):
        raise EditingT1RuntimeError(
            "T1 row progress/path/shard reference disagrees with packed data"
        )
    exact_key = (*exact_trace_key, progress)
    if exact_key not in semantic_cells or semantic_cells[exact_key] != row["semantic_cell_id"]:
        raise EditingT1RuntimeError("T1 row semantic cell disagrees with the exact sidecar")
    state = record.path.state_at(progress)
    target = record.path.state_at(progress + 1)
    step = record.path.trace.steps[progress]
    recorded_family = canonical_family(step.rule_name)
    expected_family = (
        None if family_selector == EDITING_T1_GLOBAL_FAMILY_SELECTOR else family_selector
    )
    source_digest = persistent_slot_state_sha256(state)
    if (
        source_digest != row["source_state_sha256"]
        or canonical_state_key(state) != row["source_state_key"]
        or canonical_state_key(target) != row["teacher_successor_key"]
        or step.rule_name != row["teacher_rule_name"]
        or row["teacher_family"] != recorded_family
        or (expected_family is not None and recorded_family != expected_family)
        or rewrite_action_codec_sha256(step.rule_name, step.action) != row["teacher_action_sha256"]
    ):
        raise EditingT1RuntimeError(
            "T1 row teacher/source/target identity disagrees with packed data"
        )
    runtime = de_novo_rewrite_system()
    try:
        replayed = runtime.apply(state, step.rule_name, step.action)
    except Exception as error:
        raise EditingT1RuntimeError("T1 teacher action is not executable") from error
    if persistent_slot_state_sha256(replayed) != persistent_slot_state_sha256(target):
        raise EditingT1RuntimeError(
            "T1 teacher action does not execute to the exact stored successor"
        )
    importance_weight = float(row["importance_weight"])
    if not math.isfinite(importance_weight) or importance_weight <= 0.0:
        raise EditingT1RuntimeError("T1 row importance weight is not finite and positive")
    return (
        SuccessorSupervisionExample(
            family_name=recorded_family,
            state=state,
            target=target,
            teacher_rule_name=step.rule_name,
            teacher_action=step.action,
            time=EDITING_T1_SUPPORT_TIME,
            teacher_rate=1.0,
            importance_weight=importance_weight,
            data_lane=f"frozen_validation_forensics:{panel_kind}",
        ),
        SuccessorFiberCacheAddress.from_packed_trace(
            address,
            progress_index=progress,
        ),
    )


def materialize_t1_panel(
    model: Any,
    *,
    source: FrozenValidationSource,
    panel: Mapping[str, Any],
    forensics: Mapping[str, Any],
    family: str,
    panel_kind: str,
    max_atoms: int,
    excluded_trace_ids: Mapping[tuple[str, int], str],
) -> MaterializedT1Panel:
    """Resolve, compile, serialize, reload, and tensorize one exact T1 panel."""

    if panel_kind == EDITING_T1_GLOBAL_REPEATED_PANEL_KIND:
        if family != EDITING_T1_GLOBAL_FAMILY_SELECTOR:
            raise ValueError("global repeated-state T1 requires family='all_families'")
    elif family not in RINGCORE_EDITING_FAMILIES:
        raise ValueError(f"unknown T1 family {family!r}")
    if getattr(model, "enable_ring_system_delete", None) is not False:
        raise EditingT1RuntimeError(
            "eight-family T1 pilot support must explicitly disable ring_system_delete"
        )
    try:
        source_rows = selected_source_rows(
            panel,
            forensics,
            family=family,
            panel_kind=panel_kind,
        )
    except EditingT1PanelError as error:
        raise EditingT1RuntimeError("requested T1 panel is absent or invalid") from error
    record_by_address = _record_index(source)
    for row in source_rows:
        state_ref = row["exact_state_ref"]
        trace_key = (
            str(state_ref["shard_sha256"]),
            int(state_ref["record_index"]),
        )
        excluded_trace_id = excluded_trace_ids.get(trace_key)
        if excluded_trace_id is None:
            continue
        try:
            excluded_record = record_by_address[trace_key]
        except KeyError:
            raise EditingT1RuntimeError(
                "charge-policy-excluded T1 row is absent from the frozen source"
            ) from None
        address = excluded_record.corpus_address
        assert address is not None
        if address.trace_id != excluded_trace_id:
            raise EditingT1RuntimeError(
                "charge-policy exclusion trace_id disagrees at an exact shard address"
            )
        raise EditingT1RuntimeError("T1 panel selected a charge-policy-excluded exact trace")
    resolved = tuple(
        _resolve_example(
            row,
            record_by_address=record_by_address,
            semantic_cells=source.sidecar.semantic_cell_ids,
            family_selector=family,
            panel_kind=panel_kind,
        )
        for row in source_rows
    )
    examples = tuple(item[0] for item in resolved)
    addresses = tuple(item[1] for item in resolved)
    unique_address_count = len(set(addresses))
    if panel_kind == EDITING_T1_UNIQUE_PANEL_KIND and len(addresses) != unique_address_count:
        raise EditingT1RuntimeError("T1 panel repeats an exact packed progress address")

    # The production cache schema deliberately accepts complete trace chains,
    # never sparse progress rows.  Compile each selected immutable trace once,
    # retain every progress (including terminal), and later project only the
    # exact T1 addresses.  A sparse cache that happened to contain the teacher
    # rows would bypass the cache's trace-completeness invariant.
    selected_trace_keys = {
        (address.packed_shard_content_sha256, address.entry_index) for address in addresses
    }
    cache_records: list[SuccessorFiberCacheRecord] = []
    for trace_key in sorted(selected_trace_keys):
        record = record_by_address[trace_key]
        try:
            cache_records.extend(
                compile_successor_fiber_trace(
                    model,
                    record,
                    time=EDITING_T1_SUPPORT_TIME,
                )
            )
        except SuccessorFiberCacheBuildError as error:
            raise EditingT1RuntimeError(
                "T1 complete-trace successor-cache compilation failed"
            ) from error

    selected_shards = {record.address.packed_shard_content_sha256 for record in cache_records}
    provenance = build_exact_cache_provenance(
        model,
        source,
        selected_shard_digests=selected_shards,
        max_atoms=max_atoms,
    )
    records_by_shard: dict[str, list[SuccessorFiberCacheRecord]] = defaultdict(list)
    for record in cache_records:
        records_by_shard[record.address.packed_shard_content_sha256].append(record)

    decoded_by_address: dict[
        SuccessorFiberCacheAddress,
        SuccessorFiberCacheRecord,
    ] = {}
    receipts: list[T1CacheShardReceipt] = []
    for shard_digest in sorted(records_by_shard):
        encoded, cache = serialize_successor_fiber_cache(
            records_by_shard[shard_digest],
            provenance=provenance[shard_digest],
        )
        decoded = deserialize_successor_fiber_cache(
            encoded,
            expected_provenance=provenance[shard_digest],
            expected_content_sha256=cache.content_sha256,
        )
        if decoded != cache:
            raise EditingT1RuntimeError("T1 cache round trip changed production fibers")
        receipts.append(
            T1CacheShardReceipt(
                packed_shard_content_sha256=shard_digest,
                cache_content_sha256=cache.content_sha256,
                encoded_sha256=hashlib.sha256(encoded).hexdigest(),
                record_count=len(cache.records),
            )
        )
        for record in decoded.records:
            if record.address in decoded_by_address:
                raise EditingT1RuntimeError("T1 cache repeats a progress address across shards")
            decoded_by_address[record.address] = record

    if not set(addresses).issubset(decoded_by_address):
        raise EditingT1RuntimeError("T1 cache does not cover every selected progress address")
    cached_fibers = tuple(decoded_by_address[address].teacher_fiber for address in addresses)
    if any(fiber is None for fiber in cached_fibers):
        raise EditingT1RuntimeError("T1 cache returned a terminal row for nonterminal supervision")
    prepared = prepare_cached_successor_panel(
        model,
        examples,
        tuple(fiber for fiber in cached_fibers if fiber is not None),
    )
    return MaterializedT1Panel(
        family=family,
        panel_kind=panel_kind,
        examples=examples,
        prepared=prepared,
        cache_receipts=tuple(receipts),
        source_row_sha256s=tuple(str(row["row_sha256"]) for row in source_rows),
        unique_progress_address_count=unique_address_count,
    )


def run_editing_t1_arm(
    *,
    contract: EditingT1RuntimeContract,
    gate_zero_contract: GateZeroRuntimeContract,
    source: FrozenValidationSource,
    panel: Mapping[str, Any],
    forensics: Mapping[str, Any],
    family: str,
    panel_kind: str,
    scope: str,
    device: torch.device,
    excluded_trace_ids: Mapping[tuple[str, int], str],
) -> dict[str, Any]:
    """Run one family/panel/scope arm from the common exact scratch state."""

    if gate_zero_contract.sha256 != contract.payload["gate_zero_runtime_contract_sha256"]:
        raise EditingT1RuntimeError("T1 contract is bound to another Gate-0 runtime contract")
    if gate_zero_contract.model.get("enable_ring_system_delete") is not False:
        raise EditingT1RuntimeError(
            "eight-family T1 pilot requires explicit enable_ring_system_delete=false"
        )
    if panel["artifact_sha256"] != contract.payload["panel_artifact_sha256"]:
        raise EditingT1RuntimeError("T1 contract is bound to another panel artifact")
    for contract_field, source_field in (
        ("charge_policy_audit_file_sha256", "charge_policy_audit_file_sha256"),
        (
            "charge_policy_exclusions_file_sha256",
            "charge_policy_exclusions_file_sha256",
        ),
        (
            "charge_policy_exclusion_payload_sha256",
            "charge_policy_exclusion_payload_sha256",
        ),
        (
            "charge_policy_source_input_inventory_sha256",
            "charge_policy_source_input_inventory_sha256",
        ),
    ):
        if contract.payload[contract_field] != panel["source"][source_field]:
            raise EditingT1RuntimeError(
                f"T1 contract/panel charge provenance disagrees for {contract_field}"
            )
    if family != EDITING_T1_GLOBAL_FAMILY_SELECTOR and family not in tuple(
        contract.payload["families"]
    ):
        raise EditingT1RuntimeError("requested T1 family is off-contract")
    if panel_kind not in tuple(contract.payload["panel_kinds"]):
        raise EditingT1RuntimeError("requested T1 panel kind is off-contract")
    if (family == EDITING_T1_GLOBAL_FAMILY_SELECTOR) != (
        panel_kind == EDITING_T1_GLOBAL_REPEATED_PANEL_KIND
    ):
        raise EditingT1RuntimeError(
            "all_families is reserved for the primary global repeated-state panel"
        )
    if scope not in tuple(contract.optimization["scopes"]):
        raise EditingT1RuntimeError("requested T1 scope is off-contract")

    model, parity = build_scratch_ringcore_model(gate_zero_contract)
    model = model.to(device)
    initial_state_sha256 = state_dict_semantic_sha256(model.state_dict())
    materialized = materialize_t1_panel(
        model,
        source=source,
        panel=panel,
        forensics=forensics,
        family=family,
        panel_kind=panel_kind,
        max_atoms=int(gate_zero_contract.model["max_atoms"]),
        excluded_trace_ids=excluded_trace_ids,
    )
    optimization = contract.optimization
    training_report = train_successor_micro_panel(
        model,
        materialized.prepared,
        steps=int(optimization["steps"]),
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        scope=scope,
        seed=int(gate_zero_contract.model["seed"]),
        report_points=tuple(optimization["report_points"]),
    )
    result = {
        "schema": EDITING_T1_RESULT_SCHEMA,
        "schema_version": EDITING_T1_RESULT_VERSION,
        "status": EDITING_T1_RESULT_STATUS,
        "training_authorized": False,
        "gate_decision": None,
        "numeric_thresholds_frozen": False,
        "contract_sha256": contract.sha256,
        "gate_zero_runtime_contract_sha256": gate_zero_contract.sha256,
        "panel_artifact_sha256": panel["artifact_sha256"],
        "charge_policy_exclusion_payload_sha256": contract.payload[
            "charge_policy_exclusion_payload_sha256"
        ],
        "charge_policy_source_input_inventory_sha256": contract.payload[
            "charge_policy_source_input_inventory_sha256"
        ],
        "family": family,
        "family_status": (
            "PRIMARY_MIXED_FAMILY_SUCCESSOR_LAW"
            if family == EDITING_T1_GLOBAL_FAMILY_SELECTOR
            else (
                "DIAGNOSTIC_PENDING_OPERATOR_DECISION"
                if family in EDITING_T1_DIAGNOSTIC_ONLY_FAMILIES
                else "CURRENT_ACTIVE_FAMILY"
            )
        ),
        "panel_kind": panel_kind,
        "panel_role": (
            "PRIMARY_MIXED_FAMILY_EMPIRICAL_SUCCESSOR_LAW"
            if panel_kind == EDITING_T1_GLOBAL_REPEATED_PANEL_KIND
            else (
                "SECONDARY_WITHIN_FAMILY_CONDITIONAL_DIAGNOSTIC"
                if panel_kind == EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND
                else "DETERMINISTIC_FAMILY_CAPACITY_DIAGNOSTIC"
            )
        ),
        "scope": scope,
        "operator_identity": {
            "operator_capability_fingerprint": (model.operator_capabilities.fingerprint()),
            "enable_ring_system_delete": model.enable_ring_system_delete,
            "enable_ring_grow_macro": model.enable_ring_grow_macro,
            "enable_cycle_ops": model.enable_cycle_ops,
        },
        "example_count": len(materialized.examples),
        "unique_progress_address_count": (materialized.unique_progress_address_count),
        "repeated_progress_observation_count": (
            len(materialized.examples) - materialized.unique_progress_address_count
        ),
        "source_row_sha256s": list(materialized.source_row_sha256s),
        "scratch_initialization": {
            **asdict(parity),
            "initial_model_state_sha256": initial_state_sha256,
        },
        "cache_receipts": [asdict(receipt) for receipt in materialized.cache_receipts],
        "training_report": training_report,
        "final_model_state_sha256": state_dict_semantic_sha256(model.state_dict()),
    }
    sealed = {**result, "result_sha256": _stable_sha256(result)}
    validate_editing_t1_result(
        sealed,
        expected_result_sha256=sealed["result_sha256"],
        expected_contract_sha256=contract.sha256,
        expected_panel_artifact_sha256=str(panel["artifact_sha256"]),
        expected_cache_receipts=materialized.cache_receipts,
    )
    return sealed


__all__ = [
    "EDITING_T1_RESULT_SCHEMA",
    "EDITING_T1_RESULT_STATUS",
    "EDITING_T1_RESULT_VERSION",
    "EDITING_T1_RUNTIME_CONTRACT_SCHEMA",
    "EDITING_T1_RUNTIME_CONTRACT_STATUS",
    "EDITING_T1_RUNTIME_CONTRACT_VERSION",
    "EditingT1RuntimeContract",
    "EditingT1RuntimeError",
    "MaterializedT1Panel",
    "T1CacheShardReceipt",
    "editing_t1_implementation_sha256",
    "editing_t1_implementation_sources",
    "load_editing_t1_runtime_contract",
    "load_editing_t1_result",
    "materialize_t1_panel",
    "run_editing_t1_arm",
    "validate_editing_t1_result",
    "validate_t1_cache_shard_receipt",
]
