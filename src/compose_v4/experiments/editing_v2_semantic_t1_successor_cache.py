"""Semantic-lineage CPU successor-fiber cache for bounded Editing-V2 T1.

The semantic T1 panel names a bounded union of complete admitted train traces.
This module compiles those traces once with the production semantic executor and
canonical-successor evaluator, stores only score-independent table coordinates,
and exposes a strict load-only lookup for GPU capacity work.

This cache owns a semantic source/provenance schema.  It deliberately does not
reuse the legacy :class:`T1SuccessorCacheIdentity`, whose fields describe an
older corpus, panel, and Active8 lineage.  No artifact in this module authorizes
T1 optimization, P50, checkpoint selection, or training.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from compose_v4.data.editing_corpus_contract import REQUIRED_DATA_LANES
from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
    EditingV2SemanticActive8SourceInventory,
)
from compose_v4.data.immutable_artifact import write_bytes_if_absent
from compose_v4.data.successor_fiber_cache import (
    DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheError,
    SuccessorFiberCacheRecord,
    canonical_successor_fiber_records_for_shard,
    successor_fiber_cache_record_from_payload,
    successor_fiber_cache_record_payload,
)
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.editing_v2_semantic_t1_artifact_contracts import (
    CACHE_COMPLETION_FILENAME,
    CACHE_COMPLETION_SCHEMA,
    CACHE_COMPLETION_SCHEMA_VERSION,
    CACHE_COMPLETION_STATUS,
    CACHE_LEAF_FILENAME,
    CACHE_LEAF_SCHEMA,
    CACHE_LEAF_SCHEMA_VERSION,
    CACHE_LEAF_STATUS,
    CACHE_MANIFEST_FILENAME,
    CACHE_MANIFEST_SCHEMA,
    CACHE_MANIFEST_SCHEMA_VERSION,
    CACHE_MANIFEST_STATUS,
    CACHE_PLAN_FILENAME,
    CACHE_PLAN_SCHEMA,
    CACHE_PLAN_SCHEMA_VERSION,
    CACHE_PLAN_STATUS,
    NO_DOWNSTREAM_AUTHORITY,
    PANEL_ARTIFACT_FILENAME,
    PANEL_COMPLETION_FILENAME,
    PANEL_COMPLETION_SCHEMA,
    PANEL_COMPLETION_SCHEMA_VERSION,
    PANEL_COMPLETION_STATUS,
    PANEL_RUN_REQUEST_FILENAME,
    PANEL_RUN_REQUEST_SCHEMA,
    PANEL_RUN_REQUEST_SCHEMA_VERSION,
    PANEL_RUN_REQUEST_STATUS,
)
from compose_v4.experiments.editing_v2_semantic_t1_panel_cache import (
    SemanticT1CacheTraceInput,
    SemanticT1PanelArtifact,
    deserialize_semantic_t1_panel,
    semantic_t1_panel_implementation_sha256,
)
from compose_v4.experiments.successor_fiber_cache_builder import (
    SuccessorFiberCacheBuildError,
    compile_successor_fiber_trace_union,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.action_codec_v4 import (
    SCHEMA_VERSION as ACTION_CODEC_V4_SCHEMA_VERSION,
)
from compose_v4.rewrite.typed_ring_catalog import ring_catalog_fingerprint

CACHE_POLICY_RELATIVE_PATH = "configs/editing_v2_semantic_t1_successor_cache_v1.json"
CACHE_OUTPUT_PREFIX = "/artifacts/editing_v2/semantic_t1_successor_cache"

MAX_PLAN_BYTES = 32 << 20
MAX_LEAF_BYTES = 1 << 30
MAX_MANIFEST_BYTES = 32 << 20
MAX_COMPLETION_BYTES = 1 << 20

_SHA256_HEX = frozenset("0123456789abcdef")
_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/experiments/editing_v2_semantic_t1_successor_cache.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_artifact_contracts.py",
    "src/compose_v4/experiments/editing_v2_semantic_runtime.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_panel_cache.py",
    "src/compose_v4/experiments/successor_fiber_cache_builder.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/experiments/production_successor_kernel.py",
    "src/compose_v4/data/editing_corpus_contract.py",
    "src/compose_v4/data/editing_v2_semantic_active8_source_adapter.py",
    "src/compose_v4/data/successor_fiber_cache.py",
    "src/compose_v4/data/semantic_packed_trace_store.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "src/compose_v4/chem/persistent_state_identity.py",
    "src/compose_v4/rewrite/action_codec_v4.py",
    "src/compose_v4/rewrite/kernel.py",
)


class SemanticT1SuccessorCacheError(RuntimeError):
    """Semantic T1 cache bytes or lineage are absent, stale, or inconsistent."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache metadata is not finite canonical JSON"
        ) from error
    return encoded + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _require_sha(value: object, *, field_name: str) -> str:
    if not _is_sha(value):
        raise SemanticT1SuccessorCacheError(f"{field_name} must be a lowercase SHA-256")
    return str(value)


def _require_nonauthorizing(value: Mapping[str, Any], *, field_name: str) -> None:
    if any(
        value.get(name) is not expected
        for name, expected in NO_DOWNSTREAM_AUTHORITY.items()
    ):
        raise SemanticT1SuccessorCacheError(
            f"{field_name} crosses the downstream authority boundary"
        )


def _load_canonical_object(
    path: Path,
    *,
    field_name: str,
    maximum_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    try:
        size = source.stat().st_size
        if not 0 < size <= maximum_bytes:
            raise SemanticT1SuccessorCacheError(f"{field_name} exceeds its byte bound")
        raw = source.read_bytes()
        value = json.loads(raw)
    except SemanticT1SuccessorCacheError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticT1SuccessorCacheError(
            f"{field_name} is absent or invalid: {source}"
        ) from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value, newline=True):
        raise SemanticT1SuccessorCacheError(
            f"{field_name} is not a canonical JSON object"
        )
    return value, raw


def _load_pretty_object(
    path: Path,
    *,
    field_name: str,
    maximum_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    """Load the panel launcher's frozen indented-JSON request encoding."""

    source = Path(path)
    try:
        size = source.stat().st_size
        if not 0 < size <= maximum_bytes:
            raise SemanticT1SuccessorCacheError(f"{field_name} exceeds its byte bound")
        raw = source.read_bytes()
        value = json.loads(raw)
    except SemanticT1SuccessorCacheError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticT1SuccessorCacheError(
            f"{field_name} is absent or invalid: {source}"
        ) from error
    expected = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if not isinstance(value, dict) or raw != expected:
        raise SemanticT1SuccessorCacheError(
            f"{field_name} is not the frozen indented JSON object"
        )
    return value, raw


def _artifact_path(
    value: object,
    *,
    artifact_root: Path,
    field_name: str,
) -> Path:
    if not isinstance(value, str):
        raise SemanticT1SuccessorCacheError(
            f"{field_name} must be a normalized /artifacts path"
        )
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
        raise SemanticT1SuccessorCacheError(
            f"{field_name} must be a normalized /artifacts path"
        )
    root = Path(artifact_root).resolve()
    resolved = (root / Path(*pure.parts[2:])).resolve()
    if not resolved.is_relative_to(root):
        raise SemanticT1SuccessorCacheError(
            f"{field_name} resolves outside artifact_root"
        )
    return resolved


def _artifact_address(path: Path, *, artifact_root: Path) -> str:
    root = Path(artifact_root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache artifact resolves outside artifact_root"
        )
    relative = resolved.relative_to(root)
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def semantic_t1_successor_cache_implementation_sha256(*, repo_root: Path) -> str:
    """Hash every source that fixes cache compilation and coordinate meaning."""

    root = Path(repo_root).resolve()
    digest = hashlib.sha256()
    for relative in _IMPLEMENTATION_SOURCES:
        path = root / relative
        if not path.is_file():
            raise SemanticT1SuccessorCacheError(
                f"semantic T1 cache implementation source is absent: {relative}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedSemanticT1PanelBundle:
    """Physically and semantically verified panel-launcher handoff."""

    completion_path: Path
    completion_file_sha256: str
    completion: Mapping[str, Any]
    request_path: Path
    request_file_sha256: str
    request: Mapping[str, Any]
    panel_path: Path
    panel_file_sha256: str
    panel: SemanticT1PanelArtifact


def load_verified_semantic_t1_panel_bundle(
    completion_path: Path,
    *,
    artifact_root: Path,
    repo_root: Path,
) -> VerifiedSemanticT1PanelBundle:
    """Revalidate the exact nonauthorizing panel completion and payload bytes."""

    completion_source = Path(completion_path).resolve()
    if completion_source.name != PANEL_COMPLETION_FILENAME:
        raise SemanticT1SuccessorCacheError(
            f"panel completion must name {PANEL_COMPLETION_FILENAME}"
        )
    root = Path(artifact_root).resolve()
    if not completion_source.is_relative_to(root):
        raise SemanticT1SuccessorCacheError(
            "panel completion resolves outside artifact_root"
        )
    completion, completion_bytes = _load_canonical_object(
        completion_source,
        field_name="semantic T1 panel completion",
        maximum_bytes=MAX_COMPLETION_BYTES,
    )
    completion_body = dict(completion)
    supplied_completion_sha256 = completion_body.pop("completion_sha256", None)
    required_completion_fields = {
        "schema",
        "schema_version",
        "status",
        *NO_DOWNSTREAM_AUTHORITY,
        "run_identity_sha256",
        "source_revision_sha256",
        "input_inventory_sha256",
        "panel_policy",
        "request_artifact_path",
        "request_file_sha256",
        "panel_artifact_path",
        "panel_artifact_sha256",
        "panel_file_sha256",
        "panel_file_bytes",
        "panel_implementation_sha256",
        "decision_source_inventory_sha256",
        "gate_zero_completion_sha256",
        "counts",
        "cache_trace_inputs_reopened",
        "resolved_cache_trace_count",
        "unique_state_panel_prepared",
        "repeated_state_panel_prepared",
        "empirical_multiplicity_receipts_consumed",
        "training_launched",
        "successor_cache_compiled",
        "next_stage_authorized",
        "completion_sha256",
    }
    if (
        set(completion) != required_completion_fields
        or completion.get("schema") != PANEL_COMPLETION_SCHEMA
        or completion.get("schema_version") != PANEL_COMPLETION_SCHEMA_VERSION
        or completion.get("status") != PANEL_COMPLETION_STATUS
        or supplied_completion_sha256 != _sha(completion_body)
        or completion.get("cache_trace_inputs_reopened") is not True
        or completion.get("unique_state_panel_prepared") is not True
        or completion.get("repeated_state_panel_prepared") is not False
        or completion.get("empirical_multiplicity_receipts_consumed") is not False
        or completion.get("training_launched") is not False
        or completion.get("successor_cache_compiled") is not False
        or completion.get("next_stage_authorized") is not None
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 panel completion schema or identity disagrees"
        )
    _require_nonauthorizing(completion, field_name="semantic T1 panel completion")

    request_path = _artifact_path(
        completion.get("request_artifact_path"),
        artifact_root=root,
        field_name="panel completion request_artifact_path",
    )
    panel_path = _artifact_path(
        completion.get("panel_artifact_path"),
        artifact_root=root,
        field_name="panel completion panel_artifact_path",
    )
    if (
        request_path.parent != completion_source.parent
        or panel_path.parent != completion_source.parent
        or request_path.name != PANEL_RUN_REQUEST_FILENAME
        or panel_path.name != PANEL_ARTIFACT_FILENAME
    ):
        raise SemanticT1SuccessorCacheError(
            "panel request, panel, and completion are not exact sibling artifacts"
        )
    request, request_bytes = _load_pretty_object(
        request_path,
        field_name="semantic T1 panel request",
        maximum_bytes=MAX_PLAN_BYTES,
    )
    request_body = dict(request)
    supplied_run_sha256 = request_body.pop("run_identity_sha256", None)
    required_request_fields = {
        "schema",
        "schema_version",
        "status",
        *NO_DOWNSTREAM_AUTHORITY,
        "source_revision",
        "python_runtime",
        "inputs",
        "panel_request",
        "panel_policy",
        "panel_implementation_sha256",
        "output_prefix",
        "training_launched",
        "successor_cache_compiled",
        "run_identity_sha256",
    }
    if (
        set(request) != required_request_fields
        or request.get("schema") != PANEL_RUN_REQUEST_SCHEMA
        or request.get("schema_version") != PANEL_RUN_REQUEST_SCHEMA_VERSION
        or request.get("status") != PANEL_RUN_REQUEST_STATUS
        or supplied_run_sha256 != _sha(request_body)
        or completion.get("run_identity_sha256") != supplied_run_sha256
        or completion.get("request_file_sha256")
        != hashlib.sha256(request_bytes).hexdigest()
        or request.get("training_launched") is not False
        or request.get("successor_cache_compiled") is not False
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 panel request identity disagrees"
        )
    _require_nonauthorizing(request, field_name="semantic T1 panel request")
    panel_bytes = panel_path.read_bytes()
    panel_file_sha256 = hashlib.sha256(panel_bytes).hexdigest()
    current_panel_implementation = semantic_t1_panel_implementation_sha256(
        repo_root=repo_root
    )
    if (
        panel_file_sha256 != completion.get("panel_file_sha256")
        or len(panel_bytes) != completion.get("panel_file_bytes")
        or current_panel_implementation != completion.get("panel_implementation_sha256")
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 panel physical or implementation identity disagrees"
        )
    try:
        panel = deserialize_semantic_t1_panel(
            panel_bytes,
            expected_artifact_sha256=completion.get("panel_artifact_sha256"),
            expected_panel_implementation_sha256=current_panel_implementation,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 panel artifact failed strict decoding"
        ) from error
    if (
        request.get("panel_request") != panel.request.as_payload()
        or request.get("panel_implementation_sha256")
        != panel.panel_implementation_sha256
        or completion.get("decision_source_inventory_sha256")
        != panel.gate_zero_binding.decision_source_inventory_sha256
        or completion.get("gate_zero_completion_sha256")
        != panel.gate_zero_binding.completion_sha256
        or completion.get("counts") != panel.identity_body()["counts"]
        or completion.get("resolved_cache_trace_count") != len(panel.cache_trace_inputs)
    ):
        raise SemanticT1SuccessorCacheError("semantic T1 panel handoff fields disagree")
    return VerifiedSemanticT1PanelBundle(
        completion_path=completion_source,
        completion_file_sha256=hashlib.sha256(completion_bytes).hexdigest(),
        completion=MappingProxyType(dict(completion)),
        request_path=request_path,
        request_file_sha256=hashlib.sha256(request_bytes).hexdigest(),
        request=MappingProxyType(dict(request)),
        panel_path=panel_path,
        panel_file_sha256=panel_file_sha256,
        panel=panel,
    )


def load_semantic_t1_successor_cache_policy(*, repo_root: Path) -> dict[str, Any]:
    """Load the prospective, self-hashed CPU cache-build policy."""

    path = Path(repo_root).resolve() / CACHE_POLICY_RELATIVE_PATH
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticT1SuccessorCacheError(
            f"cannot load semantic T1 cache policy: {path}"
        ) from error
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *NO_DOWNSTREAM_AUTHORITY,
        "compile_device_role",
        "compiler_device",
        "compiler_dtype",
        "complete_trace_rows_required",
        "model_scores_or_probabilities_stored",
        "hazard_coordinates_included",
        "repeated_state_panel_included",
        "maximum_leaf_count",
        "maximum_selected_trace_count",
        "maximum_progress_record_count",
        "maximum_panel_entry_count",
        "maximum_concurrent_volume_writers",
        "policy_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_fields
        or raw != _canonical_bytes(payload, newline=True)
    ):
        raise SemanticT1SuccessorCacheError("semantic T1 cache policy fields disagree")
    body = dict(payload)
    supplied_sha256 = body.pop("policy_sha256", None)
    _require_nonauthorizing(payload, field_name="semantic T1 cache policy")
    if (
        payload.get("schema") != "compose.editing_v2.semantic_t1_successor_cache_policy"
        or payload.get("schema_version") != 1
        or payload.get("status")
        != "FROZEN_CPU_COORDINATE_CACHE_NO_DOWNSTREAM_AUTHORITY"
        or payload.get("compile_device_role") != "cpu_once"
        or payload.get("compiler_device") != "cpu"
        or payload.get("compiler_dtype") != "torch.float32"
        or payload.get("complete_trace_rows_required") is not True
        or payload.get("model_scores_or_probabilities_stored") is not False
        or payload.get("hazard_coordinates_included") is not False
        or payload.get("repeated_state_panel_included") is not False
        or supplied_sha256 != _sha(body)
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache policy identity disagrees"
        )
    for name in (
        "maximum_leaf_count",
        "maximum_selected_trace_count",
        "maximum_progress_record_count",
        "maximum_panel_entry_count",
        "maximum_concurrent_volume_writers",
    ):
        if type(payload.get(name)) is not int or payload[name] <= 0:
            raise SemanticT1SuccessorCacheError(
                f"semantic T1 cache policy {name} must be positive"
            )
    return payload


def _validate_model_runtime_identity(
    value: Mapping[str, Any],
    *,
    panel: SemanticT1PanelArtifact,
) -> dict[str, Any]:
    runtime = dict(value)
    body = dict(runtime)
    supplied_sha256 = body.pop("identity_sha256", None)
    architecture = runtime.get("architecture")
    if (
        supplied_sha256 != _sha(body)
        or supplied_sha256 != panel.gate_zero_binding.model_runtime_identity_sha256
        or runtime.get("initial_model_state_sha256")
        != panel.gate_zero_binding.initial_model_state_sha256
        or runtime.get("source_revision_sha256")
        != panel.gate_zero_binding.model_source_revision_sha256
        or runtime.get("process_identity_sha256") != panel.process_identity_sha256
        or not isinstance(architecture, Mapping)
        or architecture.get("dtype") != "torch.float32"
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache model runtime differs from Gate 0 or the panel"
        )
    return runtime


def _source_payload(
    source: Any,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    return {
        "data_lane": source.data_lane,
        "partition_role": source.partition_role,
        "task_identity_sha256": source.task_identity_sha256,
        "semantic_artifact_path": _artifact_address(
            source.semantic_artifact_directory,
            artifact_root=artifact_root,
        ),
        "semantic_shard_sha256": source.semantic_shard_sha256,
        "semantic_manifest_file_sha256": source.semantic_manifest_file_sha256,
        "semantic_manifest_sha256": source.semantic_manifest_sha256,
        "semantic_completion_file_sha256": (source.semantic_completion_file_sha256),
        "semantic_completion_sha256": source.semantic_completion_sha256,
        "semantic_record_stream_sha256": source.semantic_record_stream_sha256,
        "process_identity_sha256": source.process_identity_sha256,
        "action_codec_schema_version": source.action_codec_schema_version,
        "action_codec_implementation_hash": (source.action_codec_implementation_hash),
        "source_binding": source.source_binding.as_mapping(),
        "semantic_entry_count": source.counts.semantic_entries,
    }


def _panel_binding(bundle: VerifiedSemanticT1PanelBundle) -> dict[str, Any]:
    panel = bundle.panel
    completion_artifact_path = str(
        PurePosixPath(str(bundle.completion["panel_artifact_path"])).with_name(
            PANEL_COMPLETION_FILENAME
        )
    )
    return {
        "completion_artifact_path": completion_artifact_path,
        "completion_file_sha256": bundle.completion_file_sha256,
        "completion_sha256": bundle.completion["completion_sha256"],
        "request_artifact_path": bundle.completion["request_artifact_path"],
        "request_file_sha256": bundle.request_file_sha256,
        "run_identity_sha256": bundle.request["run_identity_sha256"],
        "panel_artifact_path": bundle.completion["panel_artifact_path"],
        "panel_file_sha256": bundle.panel_file_sha256,
        "panel_file_bytes": bundle.completion["panel_file_bytes"],
        "panel_artifact_sha256": panel.artifact_sha256,
        "panel_implementation_sha256": panel.panel_implementation_sha256,
        "panel_request_sha256": panel.request.request_sha256,
        "support_time_hex": panel.request.support_time_hex,
        "panel_entry_sha256s": [entry.panel_entry_sha256 for entry in panel.entries],
        "panel_entry_inventory_sha256": panel.identity_body()[
            "panel_entry_inventory_sha256"
        ],
        "cache_input_inventory_sha256": panel.identity_body()[
            "cache_input_inventory_sha256"
        ],
        "decision_source_inventory_sha256": (
            panel.gate_zero_binding.decision_source_inventory_sha256
        ),
        "process_identity_sha256": panel.process_identity_sha256,
        "gate_zero_binding": panel.gate_zero_binding.as_payload(),
    }


def _unique_train_sources_by_digest(
    sources: Iterable[Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Require one bound train source for every semantic shard digest."""

    train_sources = tuple(
        source for source in sources if source.partition_role == "train"
    )
    sources_by_digest: dict[str, Any] = {}
    for source in train_sources:
        if source.semantic_shard_sha256 in sources_by_digest:
            other = sources_by_digest[source.semantic_shard_sha256]
            raise SemanticT1SuccessorCacheError(
                "semantic train source shard digest is reused across bound sources: "
                f"{other.data_lane!r} and {source.data_lane!r}"
            )
        sources_by_digest[source.semantic_shard_sha256] = source
    return train_sources, sources_by_digest


def build_semantic_t1_successor_cache_plan(
    bundle: VerifiedSemanticT1PanelBundle,
    *,
    source_inventory: EditingV2SemanticActive8SourceInventory,
    model_runtime_identity: Mapping[str, Any],
    semantic_model_process_contract: Mapping[str, str],
    source_revision: Mapping[str, Any],
    artifact_root: Path,
    repo_root: Path,
    output_prefix: str = CACHE_OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Freeze one task per selected semantic train shard before compilation."""

    if not isinstance(bundle, VerifiedSemanticT1PanelBundle):
        raise TypeError("bundle must be VerifiedSemanticT1PanelBundle")
    if not isinstance(source_inventory, EditingV2SemanticActive8SourceInventory):
        raise TypeError("source_inventory has another lineage")
    panel = bundle.panel
    runtime = _validate_model_runtime_identity(model_runtime_identity, panel=panel)
    policy = load_semantic_t1_successor_cache_policy(repo_root=repo_root)
    implementation_sha256 = semantic_t1_successor_cache_implementation_sha256(
        repo_root=repo_root
    )
    semantic_contract = dict(semantic_model_process_contract)
    if (
        set(semantic_contract) != {"path", "file_sha256", "contract_sha256"}
        or semantic_contract.get("contract_sha256")
        != runtime.get("semantic_model_process_contract_sha256")
        or not _is_sha(semantic_contract.get("file_sha256"))
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic model/process contract differs from Gate 0"
        )
    if source_inventory.process_identity_sha256 != panel.process_identity_sha256:
        raise SemanticT1SuccessorCacheError(
            "semantic source inventory differs from the panel process"
        )
    # The request stores the physical completion hash.  The inventory exposes
    # that hash and the completion's semantic self-hash separately.
    if (
        source_inventory.migration_completion_file_sha256
        != bundle.request["inputs"]["migration_completion"]["file_sha256"]
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic source inventory differs from the panel migration input"
        )
    train_sources, sources_by_digest = _unique_train_sources_by_digest(
        source_inventory.sources
    )
    selected_digests = tuple(
        sorted({item.packed_shard_content_sha256 for item in panel.cache_trace_inputs})
    )
    if set(selected_digests) - set(sources_by_digest):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 panel selects a shard outside the train source inventory"
        )
    if len(selected_digests) > policy["maximum_leaf_count"]:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 selected shard count exceeds the frozen cache bound"
        )
    if (
        len(panel.cache_trace_inputs) > policy["maximum_selected_trace_count"]
        or len(panel.entries) > policy["maximum_panel_entry_count"]
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 panel exceeds the prospective cache bounds"
        )
    progress_count = sum(item.path_length + 1 for item in panel.cache_trace_inputs)
    if progress_count > policy["maximum_progress_record_count"]:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 complete-trace union exceeds the cache record bound"
        )
    root = Path(artifact_root).resolve()
    output_placeholder = _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=root,
        field_name="cache output_prefix",
    )
    panel_binding = _panel_binding(bundle)
    codec_schema_versions = {
        source.action_codec_schema_version for source in train_sources
    }
    codec_implementation_hashes = {
        source.action_codec_implementation_hash for source in train_sources
    }
    if len(codec_schema_versions) != 1 or len(codec_implementation_hashes) != 1:
        raise SemanticT1SuccessorCacheError(
            "semantic train sources do not share one ActionCodec identity"
        )
    source_identity = {
        "migration_completion_file_sha256": (
            source_inventory.migration_completion_file_sha256
        ),
        "migration_completion_sha256": source_inventory.migration_completion_sha256,
        "migration_plan_sha256": source_inventory.migration_plan_sha256,
        "source_inventory_sha256": source_inventory.source_inventory_sha256,
        "task_inventory_sha256": source_inventory.task_inventory_sha256,
        "result_inventory_sha256": source_inventory.result_inventory_sha256,
        "process_identity_sha256": source_inventory.process_identity_sha256,
        "builder_identity_sha256": source_inventory.builder_identity_sha256,
        "action_codec_schema_version": next(iter(codec_schema_versions)),
        "action_codec_implementation_hash": next(iter(codec_implementation_hashes)),
    }
    base_identity = {
        "source_revision": dict(source_revision),
        "policy": policy,
        "implementation_sha256": implementation_sha256,
        "panel_binding": panel_binding,
        "model_runtime_identity": runtime,
        "semantic_model_process_contract": semantic_contract,
        "semantic_source_inventory": source_identity,
        "output_prefix": output_prefix,
    }
    build_identity_sha256 = _sha(base_identity)
    tasks: list[dict[str, Any]] = []
    for task_index, digest in enumerate(selected_digests):
        source = sources_by_digest[digest]
        inputs = [
            item.as_payload()
            for item in panel.cache_trace_inputs
            if item.packed_shard_content_sha256 == digest
        ]
        if any(item["data_lane"] != source.data_lane for item in inputs):
            raise SemanticT1SuccessorCacheError(
                "semantic T1 panel input lane differs from its bound shard"
            )
        source_payload = _source_payload(source, artifact_root=root)
        task_body = {
            "task_index": task_index,
            "semantic_source": source_payload,
            "cache_trace_inputs": inputs,
            "cache_input_inventory_sha256": _sha(inputs),
            "selected_trace_count": len(inputs),
            "complete_progress_record_count": sum(
                int(item["path_length"]) + 1 for item in inputs
            ),
            "selected_panel_entry_count": sum(
                len(item["panel_entry_sha256s"]) for item in inputs
            ),
        }
        task_identity_sha256 = _sha(
            {
                "build_identity_sha256": build_identity_sha256,
                "task": task_body,
            }
        )
        tasks.append({"task_identity_sha256": task_identity_sha256, **task_body})
    plan_body = {
        "schema": CACHE_PLAN_SCHEMA,
        "schema_version": CACHE_PLAN_SCHEMA_VERSION,
        "status": CACHE_PLAN_STATUS,
        **NO_DOWNSTREAM_AUTHORITY,
        **base_identity,
        "build_identity_sha256": build_identity_sha256,
        "run_artifact_root": _artifact_address(
            output_placeholder.parent / build_identity_sha256,
            artifact_root=root,
        ),
        "tasks": tasks,
        "task_inventory_sha256": _sha(tasks),
        "task_count": len(tasks),
        "selected_trace_count": len(panel.cache_trace_inputs),
        "complete_progress_record_count": progress_count,
        "selected_panel_entry_count": len(panel.entries),
    }
    run_identity_sha256 = _sha(plan_body)
    body = {**plan_body, "run_identity_sha256": run_identity_sha256}
    return {**body, "plan_sha256": _sha(body)}


def validate_semantic_t1_successor_cache_plan(
    value: Mapping[str, Any],
    *,
    repo_root: Path,
    artifact_root: Path,
    expected_source_revision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a frozen build plan against the current compiler sources."""

    plan = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *NO_DOWNSTREAM_AUTHORITY,
        "source_revision",
        "policy",
        "implementation_sha256",
        "panel_binding",
        "model_runtime_identity",
        "semantic_model_process_contract",
        "semantic_source_inventory",
        "output_prefix",
        "build_identity_sha256",
        "run_artifact_root",
        "tasks",
        "task_inventory_sha256",
        "task_count",
        "selected_trace_count",
        "complete_progress_record_count",
        "selected_panel_entry_count",
        "run_identity_sha256",
        "plan_sha256",
    }
    body = dict(plan)
    supplied_plan_sha256 = body.pop("plan_sha256", None)
    run_body = dict(body)
    supplied_run_identity = run_body.pop("run_identity_sha256", None)
    if (
        set(plan) != expected_fields
        or plan.get("schema") != CACHE_PLAN_SCHEMA
        or plan.get("schema_version") != CACHE_PLAN_SCHEMA_VERSION
        or plan.get("status") != CACHE_PLAN_STATUS
        or supplied_plan_sha256 != _sha(body)
        or supplied_run_identity != _sha(run_body)
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 successor-cache plan schema or self-hash disagrees"
        )
    _require_nonauthorizing(plan, field_name="semantic T1 successor-cache plan")
    current_policy = load_semantic_t1_successor_cache_policy(repo_root=repo_root)
    current_implementation = semantic_t1_successor_cache_implementation_sha256(
        repo_root=repo_root
    )
    if (
        plan.get("policy") != current_policy
        or plan.get("implementation_sha256") != current_implementation
        or (
            expected_source_revision is not None
            and plan.get("source_revision") != dict(expected_source_revision)
        )
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 successor-cache plan source or policy is stale"
        )
    tasks = plan.get("tasks")
    if (
        not isinstance(tasks, list)
        or plan.get("task_count") != len(tasks)
        or plan.get("task_inventory_sha256") != _sha(tasks)
        or not 0 < len(tasks) <= current_policy["maximum_leaf_count"]
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 successor-cache task inventory disagrees"
        )
    base_identity = {
        "source_revision": plan["source_revision"],
        "policy": plan["policy"],
        "implementation_sha256": plan["implementation_sha256"],
        "panel_binding": plan["panel_binding"],
        "model_runtime_identity": plan["model_runtime_identity"],
        "semantic_model_process_contract": plan["semantic_model_process_contract"],
        "semantic_source_inventory": plan["semantic_source_inventory"],
        "output_prefix": plan["output_prefix"],
    }
    if plan.get("build_identity_sha256") != _sha(base_identity):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 successor-cache build identity disagrees"
        )
    panel_binding = plan.get("panel_binding")
    expected_panel_binding_fields = {
        "completion_artifact_path",
        "completion_file_sha256",
        "completion_sha256",
        "request_artifact_path",
        "request_file_sha256",
        "run_identity_sha256",
        "panel_artifact_path",
        "panel_file_sha256",
        "panel_file_bytes",
        "panel_artifact_sha256",
        "panel_implementation_sha256",
        "panel_request_sha256",
        "support_time_hex",
        "panel_entry_sha256s",
        "panel_entry_inventory_sha256",
        "cache_input_inventory_sha256",
        "decision_source_inventory_sha256",
        "process_identity_sha256",
        "gate_zero_binding",
    }
    if not isinstance(panel_binding, dict) or set(panel_binding) != (
        expected_panel_binding_fields
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 successor-cache panel binding fields disagree"
        )
    panel_entry_sha256s = panel_binding.get("panel_entry_sha256s")
    gate_zero_binding = panel_binding.get("gate_zero_binding")
    expected_gate_zero_binding_fields = {
        "contract_sha256",
        "evidence_sha256",
        "evidence_file_sha256",
        "decision_sha256",
        "decision_file_sha256",
        "completion_sha256",
        "decision_source_inventory_sha256",
        "model_runtime_identity_sha256",
        "model_source_revision_sha256",
        "initial_model_state_sha256",
    }
    try:
        support_time = float.fromhex(panel_binding.get("support_time_hex"))
    except (TypeError, ValueError) as error:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 successor-cache support time is invalid"
        ) from error
    if (
        not isinstance(panel_entry_sha256s, list)
        or not panel_entry_sha256s
        or len(set(panel_entry_sha256s)) != len(panel_entry_sha256s)
        or any(not _is_sha(item) for item in panel_entry_sha256s)
        or panel_binding.get("panel_entry_inventory_sha256")
        != _sha(panel_entry_sha256s)
        or support_time.hex() != panel_binding.get("support_time_hex")
        or type(panel_binding.get("panel_file_bytes")) is not int
        or panel_binding["panel_file_bytes"] <= 0
        or not isinstance(gate_zero_binding, dict)
        or set(gate_zero_binding) != expected_gate_zero_binding_fields
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 successor-cache panel inventory or time disagrees"
        )
    for field_name in expected_gate_zero_binding_fields:
        _require_sha(
            gate_zero_binding.get(field_name),
            field_name=f"panel_binding.gate_zero_binding.{field_name}",
        )
    completion_path = _artifact_path(
        panel_binding.get("completion_artifact_path"),
        artifact_root=artifact_root,
        field_name="panel_binding.completion_artifact_path",
    )
    request_path = _artifact_path(
        panel_binding.get("request_artifact_path"),
        artifact_root=artifact_root,
        field_name="panel_binding.request_artifact_path",
    )
    panel_path = _artifact_path(
        panel_binding.get("panel_artifact_path"),
        artifact_root=artifact_root,
        field_name="panel_binding.panel_artifact_path",
    )
    if (
        len({completion_path.parent, request_path.parent, panel_path.parent}) != 1
        or completion_path.name != PANEL_COMPLETION_FILENAME
        or request_path.name != PANEL_RUN_REQUEST_FILENAME
        or panel_path.name != PANEL_ARTIFACT_FILENAME
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 successor-cache panel artifacts are not exact siblings"
        )
    model_runtime = plan.get("model_runtime_identity")
    semantic_contract = plan.get("semantic_model_process_contract")
    if (
        not isinstance(model_runtime, dict)
        or not isinstance(semantic_contract, dict)
        or set(semantic_contract) != {"path", "file_sha256", "contract_sha256"}
        or model_runtime.get("identity_sha256")
        != gate_zero_binding["model_runtime_identity_sha256"]
        or model_runtime.get("initial_model_state_sha256")
        != gate_zero_binding["initial_model_state_sha256"]
        or model_runtime.get("source_revision_sha256")
        != gate_zero_binding["model_source_revision_sha256"]
        or model_runtime.get("process_identity_sha256")
        != panel_binding["process_identity_sha256"]
        or semantic_contract.get("contract_sha256")
        != model_runtime.get("semantic_model_process_contract_sha256")
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 successor-cache model/process binding disagrees"
        )
    _require_sha(
        semantic_contract.get("file_sha256"),
        field_name="semantic_model_process_contract.file_sha256",
    )
    _require_sha(
        semantic_contract.get("contract_sha256"),
        field_name="semantic_model_process_contract.contract_sha256",
    )
    for field_name in (
        "completion_file_sha256",
        "completion_sha256",
        "request_file_sha256",
        "run_identity_sha256",
        "panel_file_sha256",
        "panel_artifact_sha256",
        "panel_implementation_sha256",
        "panel_request_sha256",
        "panel_entry_inventory_sha256",
        "cache_input_inventory_sha256",
        "decision_source_inventory_sha256",
        "process_identity_sha256",
    ):
        _require_sha(
            panel_binding.get(field_name),
            field_name=f"panel_binding.{field_name}",
        )
    source_inventory = plan.get("semantic_source_inventory")
    expected_source_inventory_fields = {
        "migration_completion_file_sha256",
        "migration_completion_sha256",
        "migration_plan_sha256",
        "source_inventory_sha256",
        "task_inventory_sha256",
        "result_inventory_sha256",
        "process_identity_sha256",
        "builder_identity_sha256",
        "action_codec_schema_version",
        "action_codec_implementation_hash",
    }
    if (
        not isinstance(source_inventory, dict)
        or set(source_inventory) != expected_source_inventory_fields
        or source_inventory.get("process_identity_sha256")
        != model_runtime.get("process_identity_sha256")
        or source_inventory.get("process_identity_sha256")
        != plan["panel_binding"].get("process_identity_sha256")
        or source_inventory.get("action_codec_schema_version")
        != ACTION_CODEC_V4_SCHEMA_VERSION
        or not isinstance(source_inventory.get("action_codec_implementation_hash"), str)
        or len(source_inventory["action_codec_implementation_hash"]) != 16
        or any(
            character not in _SHA256_HEX
            for character in source_inventory["action_codec_implementation_hash"]
        )
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 successor-cache semantic-v4 source identity disagrees"
        )
    for field_name in expected_source_inventory_fields - {
        "action_codec_schema_version",
        "action_codec_implementation_hash",
    }:
        _require_sha(
            source_inventory.get(field_name),
            field_name=f"semantic_source_inventory.{field_name}",
        )
    expected_source_fields = {
        "data_lane",
        "partition_role",
        "task_identity_sha256",
        "semantic_artifact_path",
        "semantic_shard_sha256",
        "semantic_manifest_file_sha256",
        "semantic_manifest_sha256",
        "semantic_completion_file_sha256",
        "semantic_completion_sha256",
        "semantic_record_stream_sha256",
        "process_identity_sha256",
        "action_codec_schema_version",
        "action_codec_implementation_hash",
        "source_binding",
        "semantic_entry_count",
    }
    observed_traces = 0
    observed_progress = 0
    observed_entries = 0
    observed_panel_entry_sha256s: list[str] = []
    observed_cache_input_sha256s: set[str] = set()
    observed_source_task_identity_sha256s: set[str] = set()
    observed_data_lanes: set[str] = set()
    previous_digest = ""
    for task_index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise SemanticT1SuccessorCacheError("cache task must be an object")
        task_body = {
            key: item for key, item in task.items() if key != "task_identity_sha256"
        }
        source = task.get("semantic_source")
        inputs = task.get("cache_trace_inputs")
        if (
            task.get("task_index") != task_index
            or not isinstance(source, dict)
            or set(source) != expected_source_fields
            or source.get("partition_role") != "train"
            or source.get("data_lane") not in REQUIRED_DATA_LANES
            or source.get("data_lane") in observed_data_lanes
            or source.get("process_identity_sha256")
            != source_inventory["process_identity_sha256"]
            or source.get("action_codec_schema_version")
            != source_inventory["action_codec_schema_version"]
            or source.get("action_codec_implementation_hash")
            != source_inventory["action_codec_implementation_hash"]
            or not isinstance(inputs, list)
            or task.get("cache_input_inventory_sha256") != _sha(inputs)
            or task.get("task_identity_sha256")
            != _sha(
                {
                    "build_identity_sha256": plan["build_identity_sha256"],
                    "task": task_body,
                }
            )
        ):
            raise SemanticT1SuccessorCacheError(
                f"semantic T1 successor-cache task {task_index} identity disagrees"
            )
        digest = _require_sha(
            source.get("semantic_shard_sha256"),
            field_name=f"tasks[{task_index}].semantic_shard_sha256",
        )
        if digest <= previous_digest:
            raise SemanticT1SuccessorCacheError(
                "semantic T1 successor-cache tasks are not in shard order"
            )
        previous_digest = digest
        source_task_identity_sha256 = _require_sha(
            source.get("task_identity_sha256"),
            field_name=f"tasks[{task_index}].source.task_identity_sha256",
        )
        if source_task_identity_sha256 in observed_source_task_identity_sha256s:
            raise SemanticT1SuccessorCacheError(
                "semantic T1 successor-cache reuses one bound semantic source"
            )
        observed_source_task_identity_sha256s.add(source_task_identity_sha256)
        observed_data_lanes.add(source["data_lane"])
        for field_name in (
            "semantic_shard_sha256",
            "semantic_manifest_file_sha256",
            "semantic_manifest_sha256",
            "semantic_completion_file_sha256",
            "semantic_completion_sha256",
            "semantic_record_stream_sha256",
            "process_identity_sha256",
        ):
            _require_sha(
                source.get(field_name),
                field_name=f"tasks[{task_index}].source.{field_name}",
            )
        if (
            not isinstance(source.get("source_binding"), dict)
            or type(source.get("semantic_entry_count")) is not int
            or source["semantic_entry_count"] <= 0
        ):
            raise SemanticT1SuccessorCacheError(
                "semantic T1 successor-cache source binding or census disagrees"
            )
        _artifact_path(
            source.get("semantic_artifact_path"),
            artifact_root=artifact_root,
            field_name=f"tasks[{task_index}].semantic_artifact_path",
        )
        typed_inputs: list[SemanticT1CacheTraceInput] = []
        for item in inputs:
            if not isinstance(item, dict):
                raise SemanticT1SuccessorCacheError(
                    "cache trace input must be an object"
                )
            values = dict(item)
            if values.pop("partition_role", None) != "train":
                raise SemanticT1SuccessorCacheError(
                    "cache trace input must be train-only"
                )
            for field_name in ("selected_progress_indices", "panel_entry_sha256s"):
                if not isinstance(values.get(field_name), list):
                    raise SemanticT1SuccessorCacheError(
                        f"cache trace input {field_name} must be an array"
                    )
                values[field_name] = tuple(values[field_name])
            try:
                typed = SemanticT1CacheTraceInput(**values)
            except (TypeError, ValueError) as error:
                raise SemanticT1SuccessorCacheError(
                    "semantic T1 cache trace input is invalid"
                ) from error
            if (
                typed.as_payload() != item
                or typed.packed_shard_content_sha256 != digest
                or typed.data_lane != source.get("data_lane")
                or typed.cache_input_sha256 in observed_cache_input_sha256s
            ):
                raise SemanticT1SuccessorCacheError(
                    "semantic T1 cache trace input differs from its source"
                )
            typed_inputs.append(typed)
            observed_cache_input_sha256s.add(typed.cache_input_sha256)
            observed_panel_entry_sha256s.extend(typed.panel_entry_sha256s)
        selected_progress = sum(len(item.panel_entry_sha256s) for item in typed_inputs)
        complete_progress = sum(item.path_length + 1 for item in typed_inputs)
        if (
            task.get("selected_trace_count") != len(typed_inputs)
            or task.get("complete_progress_record_count") != complete_progress
            or task.get("selected_panel_entry_count") != selected_progress
            or not typed_inputs
        ):
            raise SemanticT1SuccessorCacheError(
                "semantic T1 successor-cache task census disagrees"
            )
        observed_traces += len(typed_inputs)
        observed_progress += complete_progress
        observed_entries += selected_progress
    if (
        observed_traces != plan.get("selected_trace_count")
        or observed_progress != plan.get("complete_progress_record_count")
        or observed_entries != plan.get("selected_panel_entry_count")
        or observed_entries != len(panel_entry_sha256s)
        or len(set(observed_panel_entry_sha256s)) != len(observed_panel_entry_sha256s)
        or set(observed_panel_entry_sha256s) != set(panel_entry_sha256s)
        or observed_traces > current_policy["maximum_selected_trace_count"]
        or observed_progress > current_policy["maximum_progress_record_count"]
        or observed_entries > current_policy["maximum_panel_entry_count"]
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 successor-cache plan aggregate census disagrees"
        )
    expected_run_root = _artifact_path(
        plan.get("run_artifact_root"),
        artifact_root=artifact_root,
        field_name="cache plan run_artifact_root",
    )
    expected_prefix = _artifact_path(
        f"{plan.get('output_prefix')}/placeholder",
        artifact_root=artifact_root,
        field_name="cache plan output_prefix",
    ).parent
    if expected_run_root != expected_prefix / plan["build_identity_sha256"]:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 successor-cache run path disagrees"
        )
    return plan


def _record_address_payload(address: SuccessorFiberCacheAddress) -> dict[str, Any]:
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


def _model_matches_plan(
    model: FactorizedTraceletRateModel,
    plan: Mapping[str, Any],
) -> None:
    from compose_v4.experiments.editing_p50_gate import (
        state_dict_semantic_sha256,
    )

    if not isinstance(model, FactorizedTraceletRateModel):
        raise TypeError(
            "semantic T1 cache compiler requires FactorizedTraceletRateModel"
        )
    runtime = plan["model_runtime_identity"]
    architecture = runtime["architecture"]
    state = model.state_dict()
    devices = {str(tensor.device) for tensor in state.values()}
    dtypes = sorted({str(tensor.dtype) for tensor in state.values()})
    observed_architecture = {
        "max_atoms": architecture["max_atoms"],
        "hidden_dim": model.hidden_dim,
        "message_passing_steps": model.message_passing_steps,
        "mark_dim": model.mark_dim,
        "dtype": str(next(model.parameters()).dtype),
        "parameter_dtypes": dtypes,
        "atom_vocabulary_class_count": len(model.atom_vocabulary.classes),
        "catalog_fingerprint": ring_catalog_fingerprint(model.ring_catalog),
        "operator_capability_fingerprint": (model.operator_capabilities.fingerprint()),
    }
    if (
        devices != {"cpu"}
        or observed_architecture != architecture
        or state_dict_semantic_sha256(state) != runtime["initial_model_state_sha256"]
        or model.training
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache compiler model differs from Gate 0 scratch state"
        )


def compile_semantic_t1_successor_cache_leaf(
    plan: Mapping[str, Any],
    *,
    task_identity_sha256: str,
    panel: SemanticT1PanelArtifact,
    records: Iterable[PathRecord],
    model: FactorizedTraceletRateModel,
    compiler_runtime: Mapping[str, Any],
    repo_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    """Compile one selected semantic shard into score-independent coordinates."""

    validated = validate_semantic_t1_successor_cache_plan(
        plan,
        repo_root=repo_root,
        artifact_root=artifact_root,
    )
    task = next(
        (
            item
            for item in validated["tasks"]
            if item["task_identity_sha256"] == task_identity_sha256
        ),
        None,
    )
    if task is None:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache task is absent from the plan"
        )
    if (
        not isinstance(panel, SemanticT1PanelArtifact)
        or panel.artifact_sha256 != validated["panel_binding"]["panel_artifact_sha256"]
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache compiler panel differs from the plan"
        )
    runtime = dict(compiler_runtime)
    required_runtime_fields = {
        "device",
        "dtype",
        "torch_version",
        "cuda_version",
        "rdkit_version",
        "implementation_sha256",
        "source_revision_sha256",
    }
    if (
        set(runtime) != required_runtime_fields
        or runtime.get("device") != "cpu"
        or runtime.get("dtype") != "torch.float32"
        or runtime.get("implementation_sha256") != validated["implementation_sha256"]
        or runtime.get("source_revision_sha256")
        != validated["source_revision"].get("source_revision_sha256")
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache compiler runtime differs from the plan"
        )
    _model_matches_plan(model, validated)
    materialized = tuple(records)
    expected_inputs = [
        SemanticT1CacheTraceInput(
            **{
                **{
                    key: value for key, value in item.items() if key != "partition_role"
                },
                "selected_progress_indices": tuple(item["selected_progress_indices"]),
                "panel_entry_sha256s": tuple(item["panel_entry_sha256s"]),
            }
        )
        for item in task["cache_trace_inputs"]
    ]
    expected_keys = {
        (
            item.packed_shard_content_sha256,
            item.packed_entry_index,
            item.trace_id,
        ): item
        for item in expected_inputs
    }
    observed_keys = {
        (
            record.corpus_address.packed_shard_content_sha256,
            record.corpus_address.entry_index,
            record.corpus_address.trace_id,
        )
        for record in materialized
        if record.corpus_address is not None
    }
    if (
        len(materialized) != len(observed_keys)
        or observed_keys != set(expected_keys)
        or any(record.corpus_address is None for record in materialized)
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache compiler trace union differs from the task"
        )
    for record in materialized:
        address = record.corpus_address
        assert address is not None
        expected = expected_keys[
            (
                address.packed_shard_content_sha256,
                address.entry_index,
                address.trace_id,
            )
        ]
        if (
            address.packed_shard_name != expected.packed_shard_name
            or address.layer != expected.data_lane
            or address.partition != "train"
            or address.source_key != expected.trace_source_key
            or address.target_key != expected.trace_target_key
            or address.path_length != expected.path_length
        ):
            raise SemanticT1SuccessorCacheError(
                "semantic T1 cache compiler trace envelope differs from the task"
            )
    try:
        compiled = compile_successor_fiber_trace_union(
            model,
            materialized,
            time=panel.request.support_time,
        )
        ordered = canonical_successor_fiber_records_for_shard(
            compiled,
            expected_packed_shard_content_sha256=task["semantic_source"][
                "semantic_shard_sha256"
            ],
            limits=DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
        )
    except (
        SuccessorFiberCacheBuildError,
        SuccessorFiberCacheError,
        ValueError,
    ) as error:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 production successor-fiber compilation failed"
        ) from error
    if len(ordered) != task["complete_progress_record_count"]:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache leaf does not contain every complete-trace row"
        )
    record_by_key = {
        (
            record.address.packed_shard_content_sha256,
            record.address.entry_index,
            record.address.trace_id,
            record.address.progress_index,
        ): record
        for record in ordered
    }
    panel_entries = {entry.panel_entry_sha256: entry for entry in panel.entries}
    bindings: list[dict[str, Any]] = []
    for cache_input in expected_inputs:
        for progress_index, panel_entry_sha256 in zip(
            cache_input.selected_progress_indices,
            cache_input.panel_entry_sha256s,
            strict=True,
        ):
            try:
                cache_record = record_by_key[
                    (
                        cache_input.packed_shard_content_sha256,
                        cache_input.packed_entry_index,
                        cache_input.trace_id,
                        progress_index,
                    )
                ]
                entry = panel_entries[panel_entry_sha256]
            except KeyError as error:
                raise SemanticT1SuccessorCacheError(
                    "semantic T1 panel entry is absent from its compiled leaf"
                ) from error
            if (
                cache_record.teacher_fiber is None
                or cache_record.source_state_sha256 != entry.source_state_sha256
                or cache_record.target_state_sha256
                != entry.representative_target_state_sha256
                or cache_record.target_key != entry.successor_canonical_key
                or entry.progress_index != progress_index
            ):
                raise SemanticT1SuccessorCacheError(
                    "semantic T1 cache coordinates differ from the panel target"
                )
            address_payload = _record_address_payload(cache_record.address)
            binding_body = {
                "panel_entry_sha256": panel_entry_sha256,
                "cache_address": address_payload,
                "cache_address_sha256": _sha(address_payload),
                "source_state_sha256": cache_record.source_state_sha256,
                "target_state_sha256": cache_record.target_state_sha256,
                "successor_canonical_key": cache_record.target_key,
            }
            bindings.append({**binding_body, "binding_sha256": _sha(binding_body)})
    bindings.sort(key=lambda item: item["panel_entry_sha256"])
    record_payloads = [successor_fiber_cache_record_payload(item) for item in ordered]
    leaf_body = {
        "schema": CACHE_LEAF_SCHEMA,
        "schema_version": CACHE_LEAF_SCHEMA_VERSION,
        "status": CACHE_LEAF_STATUS,
        **NO_DOWNSTREAM_AUTHORITY,
        "run_identity_sha256": validated["run_identity_sha256"],
        "build_identity_sha256": validated["build_identity_sha256"],
        "task_identity_sha256": task["task_identity_sha256"],
        "panel_artifact_sha256": panel.artifact_sha256,
        "decision_source_inventory_sha256": validated["panel_binding"][
            "decision_source_inventory_sha256"
        ],
        "semantic_source": task["semantic_source"],
        "compiler_runtime": runtime,
        "support_time_hex": panel.request.support_time_hex,
        "cache_trace_inputs": task["cache_trace_inputs"],
        "cache_input_inventory_sha256": task["cache_input_inventory_sha256"],
        "record_count": len(record_payloads),
        "record_inventory_sha256": _sha(record_payloads),
        "records": record_payloads,
        "panel_entry_binding_count": len(bindings),
        "panel_entry_binding_inventory_sha256": _sha(bindings),
        "panel_entry_bindings": bindings,
        "model_scores_or_probabilities_stored": False,
        "hazard_coordinates_included": False,
    }
    leaf = {**leaf_body, "leaf_sha256": _sha(leaf_body)}
    validate_semantic_t1_successor_cache_leaf_for_plan(
        leaf,
        plan=validated,
    )
    return leaf


def write_semantic_t1_successor_cache_leaf(path: Path, leaf: Mapping[str, Any]) -> bool:
    """Immutably publish one canonical leaf, allowing byte-identical reuse."""

    validated = validate_semantic_t1_successor_cache_leaf(leaf)
    return write_bytes_if_absent(Path(path), _canonical_bytes(validated, newline=True))


def validate_semantic_t1_successor_cache_leaf(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a leaf's self-hash, complete traces, and entry bindings."""

    leaf = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *NO_DOWNSTREAM_AUTHORITY,
        "run_identity_sha256",
        "build_identity_sha256",
        "task_identity_sha256",
        "panel_artifact_sha256",
        "decision_source_inventory_sha256",
        "semantic_source",
        "compiler_runtime",
        "support_time_hex",
        "cache_trace_inputs",
        "cache_input_inventory_sha256",
        "record_count",
        "record_inventory_sha256",
        "records",
        "panel_entry_binding_count",
        "panel_entry_binding_inventory_sha256",
        "panel_entry_bindings",
        "model_scores_or_probabilities_stored",
        "hazard_coordinates_included",
        "leaf_sha256",
    }
    body = dict(leaf)
    supplied_sha256 = body.pop("leaf_sha256", None)
    if (
        set(leaf) != expected_fields
        or leaf.get("schema") != CACHE_LEAF_SCHEMA
        or leaf.get("schema_version") != CACHE_LEAF_SCHEMA_VERSION
        or leaf.get("status") != CACHE_LEAF_STATUS
        or supplied_sha256 != _sha(body)
        or leaf.get("model_scores_or_probabilities_stored") is not False
        or leaf.get("hazard_coordinates_included") is not False
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache leaf schema or self-hash disagrees"
        )
    _require_nonauthorizing(leaf, field_name="semantic T1 cache leaf")
    records_payload = leaf.get("records")
    bindings = leaf.get("panel_entry_bindings")
    source = leaf.get("semantic_source")
    if (
        not isinstance(records_payload, list)
        or leaf.get("record_count") != len(records_payload)
        or leaf.get("record_inventory_sha256") != _sha(records_payload)
        or not isinstance(bindings, list)
        or leaf.get("panel_entry_binding_count") != len(bindings)
        or leaf.get("panel_entry_binding_inventory_sha256") != _sha(bindings)
        or not isinstance(source, dict)
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache leaf inventories disagree"
        )
    try:
        records = tuple(
            successor_fiber_cache_record_from_payload(item) for item in records_payload
        )
        ordered = canonical_successor_fiber_records_for_shard(
            records,
            expected_packed_shard_content_sha256=source["semantic_shard_sha256"],
            limits=DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
        )
    except (KeyError, TypeError, ValueError, SuccessorFiberCacheError) as error:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache leaf coordinate records are invalid"
        ) from error
    if records != ordered:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache leaf records are not canonical"
        )
    if any(
        record.address.partition != "train"
        or record.address.layer != source.get("data_lane")
        for record in records
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache leaf contains a wrong role or lane"
        )
    record_keys = {
        _sha(_record_address_payload(record.address)): record for record in records
    }
    if len(record_keys) != len(records):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache leaf repeats a progress address"
        )
    panel_ids: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, dict):
            raise SemanticT1SuccessorCacheError(
                "semantic T1 cache binding must be an object"
            )
        binding_body = dict(binding)
        binding_sha256 = binding_body.pop("binding_sha256", None)
        address_sha256 = binding.get("cache_address_sha256")
        try:
            record = record_keys[address_sha256]
        except KeyError as error:
            raise SemanticT1SuccessorCacheError(
                "semantic T1 cache binding names an absent coordinate record"
            ) from error
        panel_id = binding.get("panel_entry_sha256")
        if (
            binding_sha256 != _sha(binding_body)
            or binding.get("cache_address") != _record_address_payload(record.address)
            or address_sha256 != _sha(binding["cache_address"])
            or panel_id in panel_ids
            or not _is_sha(panel_id)
            or record.teacher_fiber is None
            or binding.get("source_state_sha256") != record.source_state_sha256
            or binding.get("target_state_sha256") != record.target_state_sha256
            or binding.get("successor_canonical_key") != record.target_key
        ):
            raise SemanticT1SuccessorCacheError(
                "semantic T1 cache panel-entry binding disagrees"
            )
        panel_ids.add(str(panel_id))
    if [item["panel_entry_sha256"] for item in bindings] != sorted(panel_ids):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache panel-entry bindings are not canonical"
        )
    return leaf


def _expected_task_address_contract(
    task: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Derive the exact complete-row and selected-entry address union."""

    expected_records: dict[str, dict[str, Any]] = {}
    expected_panel_entries: dict[str, dict[str, Any]] = {}
    for cache_input in task["cache_trace_inputs"]:
        selected = dict(
            zip(
                cache_input["selected_progress_indices"],
                cache_input["panel_entry_sha256s"],
                strict=True,
            )
        )
        for progress_index in range(cache_input["path_length"] + 1):
            address = {
                "packed_shard_content_sha256": cache_input[
                    "packed_shard_content_sha256"
                ],
                "packed_shard_name": cache_input["packed_shard_name"],
                "entry_index": cache_input["packed_entry_index"],
                "layer": cache_input["data_lane"],
                "partition": "train",
                "trace_id": cache_input["trace_id"],
                "trace_source_key": cache_input["trace_source_key"],
                "trace_target_key": cache_input["trace_target_key"],
                "progress_index": progress_index,
                "path_length": cache_input["path_length"],
            }
            address_sha256 = _sha(address)
            if address_sha256 in expected_records:
                raise SemanticT1SuccessorCacheError(
                    "semantic T1 cache plan repeats one complete-trace address"
                )
            expected_records[address_sha256] = address
            panel_entry_sha256 = selected.get(progress_index)
            if panel_entry_sha256 is not None:
                if panel_entry_sha256 in expected_panel_entries:
                    raise SemanticT1SuccessorCacheError(
                        "semantic T1 cache plan repeats one selected panel entry"
                    )
                expected_panel_entries[panel_entry_sha256] = address
    return expected_records, expected_panel_entries


def _task_for_leaf(
    plan: Mapping[str, Any], leaf: Mapping[str, Any]
) -> Mapping[str, Any]:
    task = next(
        (
            item
            for item in plan["tasks"]
            if item["task_identity_sha256"] == leaf.get("task_identity_sha256")
        ),
        None,
    )
    if task is None:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache leaf has no task in the frozen plan"
        )
    runtime = leaf.get("compiler_runtime")
    if (
        leaf.get("run_identity_sha256") != plan["run_identity_sha256"]
        or leaf.get("build_identity_sha256") != plan["build_identity_sha256"]
        or leaf.get("panel_artifact_sha256")
        != plan["panel_binding"]["panel_artifact_sha256"]
        or leaf.get("decision_source_inventory_sha256")
        != plan["panel_binding"]["decision_source_inventory_sha256"]
        or leaf.get("semantic_source") != task["semantic_source"]
        or leaf.get("cache_trace_inputs") != task["cache_trace_inputs"]
        or leaf.get("cache_input_inventory_sha256")
        != task["cache_input_inventory_sha256"]
        or leaf.get("record_count") != task["complete_progress_record_count"]
        or leaf.get("panel_entry_binding_count") != task["selected_panel_entry_count"]
        or leaf.get("support_time_hex") != plan["panel_binding"]["support_time_hex"]
        or not isinstance(runtime, dict)
        or set(runtime)
        != {
            "device",
            "dtype",
            "torch_version",
            "cuda_version",
            "rdkit_version",
            "implementation_sha256",
            "source_revision_sha256",
        }
        or runtime.get("device") != "cpu"
        or runtime.get("dtype") != "torch.float32"
        or runtime.get("implementation_sha256") != plan["implementation_sha256"]
        or runtime.get("source_revision_sha256")
        != plan["source_revision"]["source_revision_sha256"]
        or not isinstance(runtime.get("torch_version"), str)
        or not runtime["torch_version"]
        or not isinstance(runtime.get("rdkit_version"), str)
        or not runtime["rdkit_version"]
        or (
            runtime.get("cuda_version") is not None
            and (
                not isinstance(runtime["cuda_version"], str)
                or not runtime["cuda_version"]
            )
        )
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache leaf differs from its frozen task"
        )
    expected_records, expected_panel_entries = _expected_task_address_contract(task)
    observed_records = {
        _sha(_record_address_payload(record.address)): _record_address_payload(
            record.address
        )
        for record in (
            successor_fiber_cache_record_from_payload(item) for item in leaf["records"]
        )
    }
    observed_panel_entries = {
        binding["panel_entry_sha256"]: binding["cache_address"]
        for binding in leaf["panel_entry_bindings"]
    }
    if (
        observed_records != expected_records
        or observed_panel_entries != expected_panel_entries
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache leaf address union differs from its exact "
            "complete-trace and selected-progress contract"
        )
    return task


def validate_semantic_t1_successor_cache_leaf_for_plan(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one leaf and bind it to an exact frozen plan task."""

    leaf = validate_semantic_t1_successor_cache_leaf(value)
    _task_for_leaf(plan, leaf)
    return leaf


def _leaf_artifact_path(
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> Path:
    run_root = _artifact_path(
        plan["run_artifact_root"],
        artifact_root=artifact_root,
        field_name="semantic T1 cache run_artifact_root",
    )
    return run_root / "tasks" / str(task["task_identity_sha256"]) / CACHE_LEAF_FILENAME


def build_semantic_t1_successor_cache_manifest(
    plan: Mapping[str, Any],
    *,
    plan_path: Path,
    artifact_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Reduce every immutable leaf into one complete semantic cache manifest."""

    validated = validate_semantic_t1_successor_cache_plan(
        plan,
        repo_root=repo_root,
        artifact_root=artifact_root,
    )
    run_root = _artifact_path(
        validated["run_artifact_root"],
        artifact_root=artifact_root,
        field_name="semantic T1 cache run_artifact_root",
    )
    resolved_plan_path = Path(plan_path).resolve()
    if resolved_plan_path != run_root / CACHE_PLAN_FILENAME:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache plan is not at its content-addressed path"
        )
    plan_value, plan_bytes = _load_canonical_object(
        resolved_plan_path,
        field_name="semantic T1 cache plan",
        maximum_bytes=MAX_PLAN_BYTES,
    )
    if plan_value != validated:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache plan bytes differ from the reducer input"
        )

    leaves: list[dict[str, Any]] = []
    all_panel_bindings: list[dict[str, Any]] = []
    record_count = 0
    for task in validated["tasks"]:
        leaf_path = _leaf_artifact_path(
            validated,
            task,
            artifact_root=artifact_root,
        )
        leaf, leaf_bytes = _load_canonical_object(
            leaf_path,
            field_name="semantic T1 cache leaf",
            maximum_bytes=MAX_LEAF_BYTES,
        )
        validated_leaf = validate_semantic_t1_successor_cache_leaf(leaf)
        _task_for_leaf(validated, validated_leaf)
        leaf_spec = {
            "task_identity_sha256": task["task_identity_sha256"],
            "semantic_shard_sha256": task["semantic_source"]["semantic_shard_sha256"],
            "data_lane": task["semantic_source"]["data_lane"],
            "leaf_artifact_path": _artifact_address(
                leaf_path,
                artifact_root=artifact_root,
            ),
            "leaf_file_sha256": hashlib.sha256(leaf_bytes).hexdigest(),
            "leaf_file_bytes": len(leaf_bytes),
            "leaf_sha256": leaf["leaf_sha256"],
            "cache_input_inventory_sha256": leaf["cache_input_inventory_sha256"],
            "record_count": leaf["record_count"],
            "record_inventory_sha256": leaf["record_inventory_sha256"],
            "panel_entry_binding_count": leaf["panel_entry_binding_count"],
            "panel_entry_binding_inventory_sha256": leaf[
                "panel_entry_binding_inventory_sha256"
            ],
        }
        leaves.append(leaf_spec)
        all_panel_bindings.extend(leaf["panel_entry_bindings"])
        record_count += int(leaf["record_count"])

    all_panel_bindings.sort(key=lambda item: item["panel_entry_sha256"])
    panel_ids = [item["panel_entry_sha256"] for item in all_panel_bindings]
    if (
        len(panel_ids) != validated["selected_panel_entry_count"]
        or len(set(panel_ids)) != len(panel_ids)
        or set(panel_ids) != set(validated["panel_binding"]["panel_entry_sha256s"])
        or record_count != validated["complete_progress_record_count"]
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache leaves do not exactly cover the committed panel"
        )
    manifest_body = {
        "schema": CACHE_MANIFEST_SCHEMA,
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "status": CACHE_MANIFEST_STATUS,
        **NO_DOWNSTREAM_AUTHORITY,
        "run_identity_sha256": validated["run_identity_sha256"],
        "build_identity_sha256": validated["build_identity_sha256"],
        "source_revision": validated["source_revision"],
        "policy_sha256": validated["policy"]["policy_sha256"],
        "implementation_sha256": validated["implementation_sha256"],
        "plan_artifact_path": _artifact_address(
            resolved_plan_path,
            artifact_root=artifact_root,
        ),
        "plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "plan_file_bytes": len(plan_bytes),
        "plan_sha256": validated["plan_sha256"],
        "panel_binding": validated["panel_binding"],
        "model_runtime_identity": validated["model_runtime_identity"],
        "semantic_model_process_contract": validated["semantic_model_process_contract"],
        "semantic_source_inventory": validated["semantic_source_inventory"],
        "leaf_count": len(leaves),
        "leaf_inventory_sha256": _sha(leaves),
        "leaves": leaves,
        "selected_trace_count": validated["selected_trace_count"],
        "record_count": record_count,
        "panel_entry_binding_count": len(all_panel_bindings),
        "panel_entry_inventory_sha256": validated["panel_binding"][
            "panel_entry_inventory_sha256"
        ],
        "panel_entry_binding_inventory_sha256": _sha(all_panel_bindings),
        "model_scores_or_probabilities_stored": False,
        "hazard_coordinates_included": False,
        "repeated_state_panel_included": False,
    }
    return {
        **manifest_body,
        "manifest_sha256": _sha(manifest_body),
    }


def validate_semantic_t1_successor_cache_manifest(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a complete manifest against its exact plan identity."""

    manifest = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *NO_DOWNSTREAM_AUTHORITY,
        "run_identity_sha256",
        "build_identity_sha256",
        "source_revision",
        "policy_sha256",
        "implementation_sha256",
        "plan_artifact_path",
        "plan_file_sha256",
        "plan_file_bytes",
        "plan_sha256",
        "panel_binding",
        "model_runtime_identity",
        "semantic_model_process_contract",
        "semantic_source_inventory",
        "leaf_count",
        "leaf_inventory_sha256",
        "leaves",
        "selected_trace_count",
        "record_count",
        "panel_entry_binding_count",
        "panel_entry_inventory_sha256",
        "panel_entry_binding_inventory_sha256",
        "model_scores_or_probabilities_stored",
        "hazard_coordinates_included",
        "repeated_state_panel_included",
        "manifest_sha256",
    }
    body = dict(manifest)
    supplied_sha256 = body.pop("manifest_sha256", None)
    leaves = manifest.get("leaves")
    if (
        set(manifest) != expected_fields
        or manifest.get("schema") != CACHE_MANIFEST_SCHEMA
        or manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != CACHE_MANIFEST_STATUS
        or supplied_sha256 != _sha(body)
        or manifest.get("run_identity_sha256") != plan["run_identity_sha256"]
        or manifest.get("build_identity_sha256") != plan["build_identity_sha256"]
        or manifest.get("source_revision") != plan["source_revision"]
        or manifest.get("policy_sha256") != plan["policy"]["policy_sha256"]
        or manifest.get("implementation_sha256") != plan["implementation_sha256"]
        or manifest.get("plan_sha256") != plan["plan_sha256"]
        or manifest.get("panel_binding") != plan["panel_binding"]
        or manifest.get("model_runtime_identity") != plan["model_runtime_identity"]
        or manifest.get("semantic_model_process_contract")
        != plan["semantic_model_process_contract"]
        or manifest.get("semantic_source_inventory")
        != plan["semantic_source_inventory"]
        or not isinstance(leaves, list)
        or manifest.get("leaf_count") != len(leaves)
        or manifest.get("leaf_inventory_sha256") != _sha(leaves)
        or manifest.get("leaf_count") != plan["task_count"]
        or manifest.get("selected_trace_count") != plan["selected_trace_count"]
        or manifest.get("record_count") != plan["complete_progress_record_count"]
        or manifest.get("panel_entry_binding_count")
        != plan["selected_panel_entry_count"]
        or manifest.get("panel_entry_inventory_sha256")
        != plan["panel_binding"]["panel_entry_inventory_sha256"]
        or manifest.get("model_scores_or_probabilities_stored") is not False
        or manifest.get("hazard_coordinates_included") is not False
        or manifest.get("repeated_state_panel_included") is not False
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache manifest schema or lineage disagrees"
        )
    _require_nonauthorizing(manifest, field_name="semantic T1 cache manifest")
    if [leaf.get("task_identity_sha256") for leaf in leaves] != [
        task["task_identity_sha256"] for task in plan["tasks"]
    ]:
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache manifest leaf order differs from the plan"
        )
    expected_leaf_fields = {
        "task_identity_sha256",
        "semantic_shard_sha256",
        "data_lane",
        "leaf_artifact_path",
        "leaf_file_sha256",
        "leaf_file_bytes",
        "leaf_sha256",
        "cache_input_inventory_sha256",
        "record_count",
        "record_inventory_sha256",
        "panel_entry_binding_count",
        "panel_entry_binding_inventory_sha256",
    }
    for task, leaf in zip(plan["tasks"], leaves, strict=True):
        leaf_address = leaf.get("leaf_artifact_path")
        expected_suffix = PurePosixPath(
            "tasks", task["task_identity_sha256"], CACHE_LEAF_FILENAME
        )
        if (
            set(leaf) != expected_leaf_fields
            or leaf.get("semantic_shard_sha256")
            != task["semantic_source"]["semantic_shard_sha256"]
            or leaf.get("data_lane") != task["semantic_source"]["data_lane"]
            or leaf.get("cache_input_inventory_sha256")
            != task["cache_input_inventory_sha256"]
            or leaf.get("record_count") != task["complete_progress_record_count"]
            or leaf.get("panel_entry_binding_count")
            != task["selected_panel_entry_count"]
            or not isinstance(leaf_address, str)
            or not PurePosixPath(leaf_address).is_absolute()
            or PurePosixPath(leaf_address).parts[-3:] != expected_suffix.parts
            or type(leaf.get("leaf_file_bytes")) is not int
            or leaf["leaf_file_bytes"] <= 0
        ):
            raise SemanticT1SuccessorCacheError(
                "semantic T1 cache manifest leaf differs from its task"
            )
        for name in (
            "task_identity_sha256",
            "semantic_shard_sha256",
            "leaf_file_sha256",
            "leaf_sha256",
            "cache_input_inventory_sha256",
            "record_inventory_sha256",
            "panel_entry_binding_inventory_sha256",
        ):
            _require_sha(leaf.get(name), field_name=f"manifest leaf {name}")
    return manifest


def write_semantic_t1_successor_cache_artifact(
    path: Path, value: Mapping[str, Any]
) -> bool:
    """Immutably publish a canonical plan, manifest, or completion object."""

    return write_bytes_if_absent(
        Path(path), _canonical_bytes(dict(value), newline=True)
    )


def build_semantic_t1_successor_cache_completion(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    plan_path: Path,
    manifest_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    """Build a no-authority receipt for one complete immutable CPU cache."""

    validated_manifest = validate_semantic_t1_successor_cache_manifest(
        manifest,
        plan=plan,
    )
    plan_source = Path(plan_path).resolve()
    manifest_source = Path(manifest_path).resolve()
    expected_root = _artifact_path(
        plan["run_artifact_root"],
        artifact_root=artifact_root,
        field_name="semantic T1 cache run_artifact_root",
    )
    if (
        plan_source != expected_root / CACHE_PLAN_FILENAME
        or manifest_source != expected_root / CACHE_MANIFEST_FILENAME
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache completion inputs are not exact siblings"
        )
    plan_bytes = plan_source.read_bytes()
    manifest_bytes = manifest_source.read_bytes()
    if plan_bytes != _canonical_bytes(
        dict(plan), newline=True
    ) or manifest_bytes != _canonical_bytes(dict(validated_manifest), newline=True):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache completion input bytes disagree"
        )
    body = {
        "schema": CACHE_COMPLETION_SCHEMA,
        "schema_version": CACHE_COMPLETION_SCHEMA_VERSION,
        "status": CACHE_COMPLETION_STATUS,
        **NO_DOWNSTREAM_AUTHORITY,
        "run_identity_sha256": plan["run_identity_sha256"],
        "build_identity_sha256": plan["build_identity_sha256"],
        "source_revision_sha256": plan["source_revision"]["source_revision_sha256"],
        "panel_completion_sha256": plan["panel_binding"]["completion_sha256"],
        "panel_artifact_sha256": plan["panel_binding"]["panel_artifact_sha256"],
        "decision_source_inventory_sha256": plan["panel_binding"][
            "decision_source_inventory_sha256"
        ],
        "initial_model_state_sha256": plan["model_runtime_identity"][
            "initial_model_state_sha256"
        ],
        "plan_artifact_path": _artifact_address(
            plan_source,
            artifact_root=artifact_root,
        ),
        "plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "plan_sha256": plan["plan_sha256"],
        "manifest_artifact_path": _artifact_address(
            manifest_source,
            artifact_root=artifact_root,
        ),
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_sha256": validated_manifest["manifest_sha256"],
        "leaf_count": validated_manifest["leaf_count"],
        "record_count": validated_manifest["record_count"],
        "panel_entry_binding_count": validated_manifest["panel_entry_binding_count"],
        "unique_state_successor_cache_compiled": True,
        "repeated_state_empirical_law_compiled": False,
        "training_launched": False,
        "next_stage_authorized": None,
    }
    return {**body, "completion_sha256": _sha(body)}


def validate_semantic_t1_successor_cache_completion(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact no-authority cache completion receipt."""

    completion = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *NO_DOWNSTREAM_AUTHORITY,
        "run_identity_sha256",
        "build_identity_sha256",
        "source_revision_sha256",
        "panel_completion_sha256",
        "panel_artifact_sha256",
        "decision_source_inventory_sha256",
        "initial_model_state_sha256",
        "plan_artifact_path",
        "plan_file_sha256",
        "plan_sha256",
        "manifest_artifact_path",
        "manifest_file_sha256",
        "manifest_sha256",
        "leaf_count",
        "record_count",
        "panel_entry_binding_count",
        "unique_state_successor_cache_compiled",
        "repeated_state_empirical_law_compiled",
        "training_launched",
        "next_stage_authorized",
        "completion_sha256",
    }
    body = dict(completion)
    supplied_sha256 = body.pop("completion_sha256", None)
    if (
        set(completion) != expected_fields
        or completion.get("schema") != CACHE_COMPLETION_SCHEMA
        or completion.get("schema_version") != CACHE_COMPLETION_SCHEMA_VERSION
        or completion.get("status") != CACHE_COMPLETION_STATUS
        or supplied_sha256 != _sha(body)
        or completion.get("unique_state_successor_cache_compiled") is not True
        or completion.get("repeated_state_empirical_law_compiled") is not False
        or completion.get("training_launched") is not False
        or completion.get("next_stage_authorized") is not None
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache completion schema or identity disagrees"
        )
    _require_nonauthorizing(completion, field_name="semantic T1 cache completion")
    for name in (
        "run_identity_sha256",
        "build_identity_sha256",
        "source_revision_sha256",
        "panel_completion_sha256",
        "panel_artifact_sha256",
        "decision_source_inventory_sha256",
        "initial_model_state_sha256",
        "plan_file_sha256",
        "plan_sha256",
        "manifest_file_sha256",
        "manifest_sha256",
        "completion_sha256",
    ):
        _require_sha(completion.get(name), field_name=f"completion.{name}")
    return completion


@dataclass(frozen=True, slots=True)
class SemanticT1SuccessorCache:
    """Strict load-only view of one complete unique-state coordinate cache."""

    completion: Mapping[str, Any]
    manifest: Mapping[str, Any]
    records: tuple[SuccessorFiberCacheRecord, ...]
    records_by_address_sha256: Mapping[str, SuccessorFiberCacheRecord] = field(
        repr=False
    )
    records_by_panel_entry_sha256: Mapping[str, SuccessorFiberCacheRecord] = field(
        repr=False
    )

    def record_for_panel_entry_sha256(
        self, panel_entry_sha256: str
    ) -> SuccessorFiberCacheRecord:
        try:
            return self.records_by_panel_entry_sha256[panel_entry_sha256]
        except KeyError as error:
            raise KeyError(
                f"panel entry is absent from semantic T1 cache: {panel_entry_sha256}"
            ) from error


def open_semantic_t1_successor_cache(
    completion_path: Path,
    *,
    artifact_root: Path,
    repo_root: Path,
    expected_panel_completion_sha256: str,
    expected_panel_artifact_sha256: str,
    expected_decision_source_inventory_sha256: str,
    expected_initial_model_state_sha256: str,
) -> SemanticT1SuccessorCache:
    """Open a complete cache only under explicit expected semantic identities."""

    expected = {
        "panel_completion_sha256": expected_panel_completion_sha256,
        "panel_artifact_sha256": expected_panel_artifact_sha256,
        "decision_source_inventory_sha256": (expected_decision_source_inventory_sha256),
        "initial_model_state_sha256": expected_initial_model_state_sha256,
    }
    for name, digest in expected.items():
        _require_sha(digest, field_name=f"expected.{name}")
    completion_source = Path(completion_path).resolve()
    if completion_source.name != CACHE_COMPLETION_FILENAME:
        raise SemanticT1SuccessorCacheError(
            f"cache completion must name {CACHE_COMPLETION_FILENAME}"
        )
    completion, _ = _load_canonical_object(
        completion_source,
        field_name="semantic T1 cache completion",
        maximum_bytes=MAX_COMPLETION_BYTES,
    )
    validated_completion = validate_semantic_t1_successor_cache_completion(completion)
    if any(validated_completion[name] != digest for name, digest in expected.items()):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache differs from an explicit expected identity"
        )
    plan_path = _artifact_path(
        validated_completion["plan_artifact_path"],
        artifact_root=artifact_root,
        field_name="cache completion plan_artifact_path",
    )
    manifest_path = _artifact_path(
        validated_completion["manifest_artifact_path"],
        artifact_root=artifact_root,
        field_name="cache completion manifest_artifact_path",
    )
    if (
        plan_path.parent != completion_source.parent
        or manifest_path.parent != completion_source.parent
        or plan_path.name != CACHE_PLAN_FILENAME
        or manifest_path.name != CACHE_MANIFEST_FILENAME
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache completion inputs are not exact siblings"
        )
    plan, plan_bytes = _load_canonical_object(
        plan_path,
        field_name="semantic T1 cache plan",
        maximum_bytes=MAX_PLAN_BYTES,
    )
    validated_plan = validate_semantic_t1_successor_cache_plan(
        plan,
        repo_root=repo_root,
        artifact_root=artifact_root,
    )
    manifest, manifest_bytes = _load_canonical_object(
        manifest_path,
        field_name="semantic T1 cache manifest",
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    validated_manifest = validate_semantic_t1_successor_cache_manifest(
        manifest,
        plan=validated_plan,
    )
    if (
        hashlib.sha256(plan_bytes).hexdigest()
        != validated_completion["plan_file_sha256"]
        or validated_plan["plan_sha256"] != validated_completion["plan_sha256"]
        or hashlib.sha256(manifest_bytes).hexdigest()
        != validated_completion["manifest_file_sha256"]
        or validated_manifest["manifest_sha256"]
        != validated_completion["manifest_sha256"]
        or validated_manifest["leaf_count"] != validated_completion["leaf_count"]
        or validated_manifest["record_count"] != validated_completion["record_count"]
        or validated_manifest["panel_entry_binding_count"]
        != validated_completion["panel_entry_binding_count"]
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache completion disagrees with plan or manifest bytes"
        )

    all_records: list[SuccessorFiberCacheRecord] = []
    records_by_address: dict[str, SuccessorFiberCacheRecord] = {}
    records_by_panel: dict[str, SuccessorFiberCacheRecord] = {}
    binding_payloads: list[dict[str, Any]] = []
    for task, leaf_spec in zip(
        validated_plan["tasks"], validated_manifest["leaves"], strict=True
    ):
        leaf_path = _artifact_path(
            leaf_spec["leaf_artifact_path"],
            artifact_root=artifact_root,
            field_name="cache manifest leaf_artifact_path",
        )
        if leaf_path != _leaf_artifact_path(
            validated_plan,
            task,
            artifact_root=artifact_root,
        ):
            raise SemanticT1SuccessorCacheError(
                "semantic T1 cache leaf is not at its content-addressed path"
            )
        leaf, leaf_bytes = _load_canonical_object(
            leaf_path,
            field_name="semantic T1 cache leaf",
            maximum_bytes=MAX_LEAF_BYTES,
        )
        validated_leaf = validate_semantic_t1_successor_cache_leaf(leaf)
        _task_for_leaf(validated_plan, validated_leaf)
        if (
            hashlib.sha256(leaf_bytes).hexdigest() != leaf_spec["leaf_file_sha256"]
            or len(leaf_bytes) != leaf_spec["leaf_file_bytes"]
            or validated_leaf["leaf_sha256"] != leaf_spec["leaf_sha256"]
            or validated_leaf["record_inventory_sha256"]
            != leaf_spec["record_inventory_sha256"]
            or validated_leaf["panel_entry_binding_inventory_sha256"]
            != leaf_spec["panel_entry_binding_inventory_sha256"]
        ):
            raise SemanticT1SuccessorCacheError(
                "semantic T1 cache leaf differs from the manifest"
            )
        leaf_records = tuple(
            successor_fiber_cache_record_from_payload(item)
            for item in validated_leaf["records"]
        )
        for record in leaf_records:
            address_sha256 = _sha(_record_address_payload(record.address))
            if address_sha256 in records_by_address:
                raise SemanticT1SuccessorCacheError(
                    "semantic T1 cache repeats a progress address across leaves"
                )
            records_by_address[address_sha256] = record
        for binding in validated_leaf["panel_entry_bindings"]:
            panel_id = binding["panel_entry_sha256"]
            if panel_id in records_by_panel:
                raise SemanticT1SuccessorCacheError(
                    "semantic T1 cache repeats a panel entry across leaves"
                )
            records_by_panel[panel_id] = records_by_address[
                binding["cache_address_sha256"]
            ]
        all_records.extend(leaf_records)
        binding_payloads.extend(validated_leaf["panel_entry_bindings"])
    binding_payloads.sort(key=lambda item: item["panel_entry_sha256"])
    if (
        len(all_records) != validated_manifest["record_count"]
        or len(records_by_panel) != validated_manifest["panel_entry_binding_count"]
        or _sha(binding_payloads)
        != validated_manifest["panel_entry_binding_inventory_sha256"]
    ):
        raise SemanticT1SuccessorCacheError(
            "semantic T1 cache loaded census differs from the manifest"
        )
    return SemanticT1SuccessorCache(
        completion=MappingProxyType(dict(validated_completion)),
        manifest=MappingProxyType(dict(validated_manifest)),
        records=tuple(all_records),
        records_by_address_sha256=MappingProxyType(records_by_address),
        records_by_panel_entry_sha256=MappingProxyType(records_by_panel),
    )
