"""Immutable semantic Editing-V2 source identity for a future P50 recipe.

The artifact written here is exactly the canonical JSON representation of
``EditingV2SemanticActive8DecisionIndex.identity_payload()``.  It is a
content-addressed source receipt, not a training authorization.  Loading it
reconstructs the complete migration, chunk-cache, and Active8 decision index
from physical artifacts before accepting either its physical or semantic
identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_semantic_active8_decision_source import (
    INDEX_ENCODING,
    INDEX_SCHEMA,
    INDEX_SCHEMA_VERSION,
    INDEX_STATUS,
    EditingV2SemanticActive8DecisionIndex,
    SemanticActive8DecisionSourceError,
    resolve_editing_v2_semantic_active8_decision_source,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    EditingV2ProcessIdentityError,
    require_editing_v2_process_identity,
)

SEMANTIC_P50_SOURCE_INVENTORY_FILENAME = "SEMANTIC_P50_SOURCE_INVENTORY.json"

_NO_AUTHORITY = {
    "training_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_HEX = frozenset("0123456789abcdef")


class SemanticP50SourceInventoryError(RuntimeError):
    """A semantic P50 source snapshot is incomplete, stale, or mutable."""


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
        raise SemanticP50SourceInventoryError(
            "semantic P50 source identity is not finite canonical JSON"
        ) from error
    return encoded + (b"\n" if newline else b"")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise SemanticP50SourceInventoryError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_fingerprint(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 16
        or any(character not in _HEX for character in value)
    ):
        raise SemanticP50SourceInventoryError(
            f"{field} must be a 16-character lowercase capability fingerprint"
        )
    return value


def _require_inside(path: Path, *, root: Path, field: str) -> Path:
    resolved_root = Path(root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise SemanticP50SourceInventoryError(f"{field} resolves outside artifact_root")
    return resolved


def _load_canonical(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if not source.is_file():
        raise SemanticP50SourceInventoryError(f"{field} is absent: {source}")
    try:
        raw = source.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticP50SourceInventoryError(f"{field} is unreadable: {source}") from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value, newline=True):
        raise SemanticP50SourceInventoryError(f"{field} must be canonical newline-terminated JSON")
    return value, raw


@dataclass(frozen=True, slots=True)
class SemanticP50SourceInventoryBinding:
    """Exact values a later prospective P50 recipe must bind."""

    source_inventory_file_sha256: str
    source_inventory_sha256: str
    process_identity_sha256: str
    model_runtime_identity_sha256: str
    active8_policy_sha256: str
    operator_capability_fingerprint: str
    decision_source_implementation_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "source_inventory_file_sha256",
            "source_inventory_sha256",
            "process_identity_sha256",
            "model_runtime_identity_sha256",
            "active8_policy_sha256",
            "decision_source_implementation_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        _require_fingerprint(
            self.operator_capability_fingerprint,
            field="operator_capability_fingerprint",
        )

    def as_payload(self) -> dict[str, str]:
        """Return a stable mapping suitable for a future recipe binding."""

        return {
            "source_inventory_file_sha256": self.source_inventory_file_sha256,
            "source_inventory_sha256": self.source_inventory_sha256,
            "process_identity_sha256": self.process_identity_sha256,
            "model_runtime_identity_sha256": self.model_runtime_identity_sha256,
            "active8_policy_sha256": self.active8_policy_sha256,
            "operator_capability_fingerprint": self.operator_capability_fingerprint,
            "decision_source_implementation_sha256": (self.decision_source_implementation_sha256),
        }


@dataclass(frozen=True, slots=True)
class PublishedSemanticP50SourceInventory:
    """Receipt for one immutable source-identity publication."""

    path: Path
    binding: SemanticP50SourceInventoryBinding


@dataclass(frozen=True, slots=True)
class VerifiedSemanticP50SourceInventory:
    """A persisted snapshot and its fully reconstructed decision index."""

    path: Path
    binding: SemanticP50SourceInventoryBinding
    index: EditingV2SemanticActive8DecisionIndex


def _decision_plan_identities(
    path: Path,
    *,
    index: EditingV2SemanticActive8DecisionIndex,
) -> tuple[str, str, str]:
    plan, raw = _load_canonical(path, field="semantic Active8 decision plan")
    if (
        hashlib.sha256(raw).hexdigest() != index.decision_plan_file_sha256
        or plan.get("plan_sha256") != index.decision_plan_sha256
        or plan.get("run_identity_sha256") != index.decision_run_identity_sha256
    ):
        raise SemanticP50SourceInventoryError(
            "semantic Active8 decision plan differs from the reconstructed index"
        )
    runtime = plan.get("model_runtime_identity")
    policy = plan.get("policy")
    if not isinstance(runtime, Mapping) or not isinstance(policy, Mapping):
        raise SemanticP50SourceInventoryError(
            "semantic Active8 decision plan lacks model or operator identity"
        )
    architecture = runtime.get("architecture")
    semantic_model = runtime.get("semantic_model_identity")
    if not isinstance(architecture, Mapping) or not isinstance(semantic_model, Mapping):
        raise SemanticP50SourceInventoryError(
            "semantic Active8 model runtime identity is incomplete"
        )
    fingerprint = _require_fingerprint(
        architecture.get("operator_capability_fingerprint"),
        field="decision plan operator_capability_fingerprint",
    )
    if (
        runtime.get("identity_sha256") != index.model_runtime_identity_sha256
        or runtime.get("process_identity_sha256") != index.process_identity_sha256
        or semantic_model.get("operator_capability_fingerprint") != fingerprint
        or policy.get("policy_sha256") != index.policy_sha256
        or policy.get("process_identity_sha256") != index.process_identity_sha256
        or tuple(policy.get("active_families", ())) != ACTIVE8_FAMILIES
    ):
        raise SemanticP50SourceInventoryError(
            "semantic Active8 process, model, or operator identity disagrees"
        )
    return (
        str(runtime["identity_sha256"]),
        str(policy["policy_sha256"]),
        fingerprint,
    )


def _binding(
    *,
    source_inventory_file_sha256: str,
    index: EditingV2SemanticActive8DecisionIndex,
    decision_plan_path: Path,
) -> SemanticP50SourceInventoryBinding:
    try:
        require_editing_v2_process_identity(index.process_identity_sha256)
    except EditingV2ProcessIdentityError as error:
        raise SemanticP50SourceInventoryError(
            "semantic P50 source names another Editing-V2 process"
        ) from error
    runtime_sha256, policy_sha256, fingerprint = _decision_plan_identities(
        decision_plan_path,
        index=index,
    )
    return SemanticP50SourceInventoryBinding(
        source_inventory_file_sha256=source_inventory_file_sha256,
        source_inventory_sha256=index.inventory_sha256,
        process_identity_sha256=index.process_identity_sha256,
        model_runtime_identity_sha256=runtime_sha256,
        active8_policy_sha256=policy_sha256,
        operator_capability_fingerprint=fingerprint,
        decision_source_implementation_sha256=(index.decision_source_implementation_sha256),
    )


def publish_semantic_p50_source_inventory(
    index: EditingV2SemanticActive8DecisionIndex,
    *,
    decision_plan_path: Path,
    output_directory: Path,
) -> PublishedSemanticP50SourceInventory:
    """Publish the exact non-authorizing index identity without overwriting."""

    if not isinstance(index, EditingV2SemanticActive8DecisionIndex):
        raise TypeError("semantic P50 source publication requires a verified decision index")
    payload = index.identity_payload()
    if (
        payload.get("schema") != INDEX_SCHEMA
        or payload.get("schema_version") != INDEX_SCHEMA_VERSION
        or payload.get("status") != INDEX_STATUS
        or payload.get("encoding") != INDEX_ENCODING
        or any(payload.get(name) is not expected for name, expected in _NO_AUTHORITY.items())
    ):
        raise SemanticP50SourceInventoryError(
            "semantic P50 source identity crosses its no-authority boundary"
        )
    content = _canonical_bytes(payload, newline=True)
    file_sha256 = hashlib.sha256(content).hexdigest()
    binding = _binding(
        source_inventory_file_sha256=file_sha256,
        index=index,
        decision_plan_path=decision_plan_path,
    )
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / SEMANTIC_P50_SOURCE_INVENTORY_FILENAME
    if target.exists():
        if not target.is_file() or target.read_bytes() != content:
            raise SemanticP50SourceInventoryError(
                f"immutable semantic P50 source-inventory collision at {target}"
            )
        return PublishedSemanticP50SourceInventory(target, binding)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return PublishedSemanticP50SourceInventory(target, binding)


def load_semantic_p50_source_inventory(
    source_inventory_path: Path,
    *,
    expected_binding: SemanticP50SourceInventoryBinding,
    migration_completion_path: Path,
    chunk_cache_plan_path: Path,
    chunk_cache_global_completion_path: Path,
    decision_plan_path: Path,
    decision_completion_path: Path,
    artifact_root: Path,
    repo_root: Path,
) -> VerifiedSemanticP50SourceInventory:
    """Rebuild the complete lineage, then accept the persisted identity."""

    if not isinstance(expected_binding, SemanticP50SourceInventoryBinding):
        raise TypeError("expected_binding must be a semantic P50 source binding")
    root = Path(artifact_root).resolve()
    source_path = _require_inside(
        source_inventory_path,
        root=root,
        field="source_inventory_path",
    )
    if source_path.name != SEMANTIC_P50_SOURCE_INVENTORY_FILENAME:
        raise SemanticP50SourceInventoryError("semantic P50 source inventory has another filename")
    payload, raw = _load_canonical(source_path, field="semantic P50 source inventory")
    observed_file_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_file_sha256 != expected_binding.source_inventory_file_sha256:
        raise SemanticP50SourceInventoryError(
            "semantic P50 source inventory physical SHA-256 disagrees"
        )
    try:
        index = resolve_editing_v2_semantic_active8_decision_source(
            migration_completion_path=migration_completion_path,
            chunk_cache_plan_path=chunk_cache_plan_path,
            chunk_cache_global_completion_path=chunk_cache_global_completion_path,
            decision_plan_path=decision_plan_path,
            decision_completion_path=decision_completion_path,
            artifact_root=root,
            repo_root=repo_root,
        )
    except SemanticActive8DecisionSourceError as error:
        raise SemanticP50SourceInventoryError(
            "semantic P50 source lineage failed complete physical revalidation"
        ) from error
    if payload != index.identity_payload():
        raise SemanticP50SourceInventoryError(
            "persisted semantic P50 source differs from the reconstructed index"
        )
    observed_binding = _binding(
        source_inventory_file_sha256=observed_file_sha256,
        index=index,
        decision_plan_path=decision_plan_path,
    )
    if observed_binding != expected_binding:
        raise SemanticP50SourceInventoryError(
            "semantic P50 source physical, process, model, or operator binding disagrees"
        )
    return VerifiedSemanticP50SourceInventory(source_path, observed_binding, index)


__all__ = [
    "SEMANTIC_P50_SOURCE_INVENTORY_FILENAME",
    "PublishedSemanticP50SourceInventory",
    "SemanticP50SourceInventoryBinding",
    "SemanticP50SourceInventoryError",
    "VerifiedSemanticP50SourceInventory",
    "load_semantic_p50_source_inventory",
    "publish_semantic_p50_source_inventory",
]
