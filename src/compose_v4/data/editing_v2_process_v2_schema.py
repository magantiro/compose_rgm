"""Frozen shared interfaces for the Process-V2 runnable chain.

Every Process-V2 layer consumes this module.  It is deliberately small and
deliberately first: the previous branch let four layers grow behind an
unreviewed interface, and the rework that caused is the reason this one is
frozen before any implementation fans out.

Six interfaces are frozen here.

**Prospective contract versus resolved evidence binding.**  A *prospective
contract* is an immutable design declaration whose deterministic builder emits
the same bytes every time.  It therefore cannot carry a slot that a later run
fills: the builder always emits the null body, the validator rebuilds the null
body and requires equality, so a correctly resealed body with measured hashes
fails validation.  That is not a bug to work around, it is the definition of a
frozen artifact.  Measured provenance lives in a **resolved evidence binding**,
a separate versioned artifact that cites the prospective contract and pins
measured identities.  Neither ever carries authority.

**Authority is false, and is spelled one way.**  `bounded_p50_authorized` is the
name.  `p50_authorized` was a second spelling for the same concept and is now a
hard incompatibility rather than an accepted alias, because a consumer grepping
one spelling silently misses the other.

**Physical versus semantic identity.**  `*_file_sha256` is the SHA-256 of the
target's bytes.  `*_semantic_sha256` is the target's own declared self-hash.
`*_process_identity_sha256` is a computed process identity and addresses no
file at all.  Conflating the three is how a V1 artifact comes to be read as a
V2 one, so the pointer type carries which is meant rather than leaving it to a
naming convention.

**Typed pointers.**  A pointer declares its kind, provider, target schema, and
hash algorithm.  The previous verifier treated any string that happened to name
an existing file as an edge, which meant a deleted or misspelled target simply
vanished from the graph and its missing-target diagnostic became unreachable.
A declared pointer cannot vanish.

**Exact completion and chunk manifest**, and **admission completion and
rejection ledger**: the schema names and required field sets the cache, rebind,
and downstream stages share.

**Structural decision-index protocol.**  The minimum a later layer needs from a
decision index, expressed as a protocol so a V2 index never subclasses a V1
loader that would revalidate V1 schemas and identities.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Protocol, runtime_checkable

# ---- Schema identity ----

SCHEMA_NAMESPACE = "compose.editing_v2.process_v2"

RESOLVED_EVIDENCE_BINDING_SCHEMA = f"{SCHEMA_NAMESPACE}.resolved_evidence_binding"
RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION = 1
RESOLVED_EVIDENCE_BINDING_STATUS = (
    "RESOLVED_EVIDENCE_BINDING_PROVENANCE_ONLY_NO_DOWNSTREAM_AUTHORITY"
)

EXACT_COMPLETION_BINDING_SCHEMA = f"{SCHEMA_NAMESPACE}.exact_completion_binding"
EXACT_COMPLETION_BINDING_SCHEMA_VERSION = 1

CHUNK_MANIFEST_SCHEMA = f"{SCHEMA_NAMESPACE}.chunk_manifest"
CHUNK_MANIFEST_SCHEMA_VERSION = 1

CHUNK_CACHE_COMPLETION_SCHEMA = f"{SCHEMA_NAMESPACE}.chunk_cache_completion"
CHUNK_CACHE_COMPLETION_SCHEMA_VERSION = 1

ADMISSION_COMPLETION_SCHEMA = f"{SCHEMA_NAMESPACE}.admission_completion"
ADMISSION_COMPLETION_SCHEMA_VERSION = 1

REJECTION_LEDGER_SCHEMA = f"{SCHEMA_NAMESPACE}.rejection_ledger"
REJECTION_LEDGER_SCHEMA_VERSION = 1

# The marker published last, after a complete generation validates. A reader
# that does not find it must treat the generation as absent, not as partial.
GENERATION_COMMITTED_MARKER = "COMMITTED.json"
GENERATION_COMMITTED_SCHEMA = f"{SCHEMA_NAMESPACE}.generation_committed"
GENERATION_COMMITTED_SCHEMA_VERSION = 1


class ProcessV2SchemaError(ValueError):
    """A shared Process-V2 interface was violated."""


# ---- Authority ----

# The complete authority vocabulary. Every Process-V2 artifact carries all of
# these, and a prospective contract or resolved evidence binding carries them
# all false. Only a stage decision or permit may set one true, and only when
# its frozen policy explicitly grants it.
AUTHORITY_FIELDS: tuple[str, ...] = (
    "training_authorized",
    "gate_zero_authorized",
    "t1_authorized",
    "bounded_p50_authorized",
    "long_training_authorized",
    "checkpoint_selection_authorized",
    "final_test_selection_authorized",
)

# Retired spellings. Accepting these indefinitely is what let two names for one
# concept coexist; naming them here turns a silent miss into a loud refusal.
RETIRED_AUTHORITY_FIELDS: Mapping[str, str] = {
    "p50_authorized": "bounded_p50_authorized",
}


def authority_false_block() -> dict[str, bool]:
    """Every authority field, explicitly false."""

    return dict.fromkeys(AUTHORITY_FIELDS, False)


def require_authority_false(payload: Mapping[str, Any], *, label: str) -> None:
    """Refuse anything that grants, omits, or misspells an authority field."""

    for retired, replacement in RETIRED_AUTHORITY_FIELDS.items():
        if retired in payload:
            raise ProcessV2SchemaError(
                f"{label} uses the retired authority field {retired!r}; it was "
                f"renamed to {replacement!r}. Two spellings for one concept mean a "
                "consumer grepping one silently misses the other, so this is an "
                "incompatibility rather than an accepted alias."
            )
    missing = [field for field in AUTHORITY_FIELDS if field not in payload]
    if missing:
        raise ProcessV2SchemaError(f"{label} omits authority fields {sorted(missing)}")
    granted = [field for field in AUTHORITY_FIELDS if payload[field] is not False]
    if granted:
        raise ProcessV2SchemaError(
            f"{label} grants authority {sorted(granted)}; a prospective contract and "
            "a resolved evidence binding both prove identity only and never permit "
            "training or launch"
        )


# ---- Identity roles and typed pointers ----


class IdentityRole:
    """Which hash a pin carries. Named, never inferred from a field name."""

    PHYSICAL = "physical_sha256"
    """SHA-256 of the target's bytes. Spelled `*_file_sha256` in a payload."""

    SEMANTIC = "semantic_sha256"
    """The target's own declared self-hash. Spelled `*_semantic_sha256`."""

    PROCESS_IDENTITY = "process_identity_sha256"
    """A computed process identity. Addresses no file."""


IDENTITY_ROLES: tuple[str, ...] = (
    IdentityRole.PHYSICAL,
    IdentityRole.SEMANTIC,
    IdentityRole.PROCESS_IDENTITY,
)


class PointerKind:
    """What a pointer addresses, so a verifier knows how to check it."""

    REPOSITORY_CONFIG = "repository_config"
    """A file tracked in this repository. A missing target is a FAILURE."""

    REMOTE_ARTIFACT = "remote_artifact"
    """A volume-relative artifact. Verified by its stage loader, never by the
    local filesystem, and therefore never reported as a missing local file."""

    LINEAGE_REFERENCE = "lineage_reference"
    """A deliberately historical value. Carries no currency claim."""

    EXTERNAL_ASSET = "external_asset"
    """A source asset outside the repository, pinned by bytes."""


POINTER_KINDS: tuple[str, ...] = (
    PointerKind.REPOSITORY_CONFIG,
    PointerKind.REMOTE_ARTIFACT,
    PointerKind.LINEAGE_REFERENCE,
    PointerKind.EXTERNAL_ASSET,
)

# The canonical-body rule a `semantic_sha256` is computed under. Declared in the
# pointer so a verifier never has to guess which algorithm produced a value.
SEMANTIC_HASH_ALGORITHMS: tuple[str, ...] = (
    "self_hash_field_v1",
    """the target declares a `*_sha256` field equal to the canonical hash of its
    body minus that field""",
    "whole_canonical_body_v1",
    """the target declares no self-hash, so the canonical hash of its entire
    body is used""",
)[::2]

_POINTER_FIELDS = frozenset(
    {"kind", "provider", "target", "target_schema", "identity_role", "hash_algorithm", "sha256"}
)


def typed_pointer(
    *,
    kind: str,
    provider: str,
    target: str,
    target_schema: str | None,
    identity_role: str,
    sha256: str,
    hash_algorithm: str | None = None,
) -> dict[str, Any]:
    """Build one declared pointer.

    Declared, not inferred: the previous verifier treated any string that named
    an existing file as an edge, so a deleted or misspelled target dropped out
    of the graph entirely and could never be reported missing.
    """

    if kind not in POINTER_KINDS:
        raise ProcessV2SchemaError(f"unknown pointer kind {kind!r}; expected {POINTER_KINDS}")
    if identity_role not in IDENTITY_ROLES:
        raise ProcessV2SchemaError(
            f"unknown identity role {identity_role!r}; expected {IDENTITY_ROLES}"
        )
    if identity_role == IdentityRole.SEMANTIC and hash_algorithm not in (
        SEMANTIC_HASH_ALGORITHMS
    ):
        raise ProcessV2SchemaError(
            "a semantic pointer must declare its hash algorithm; expected one of "
            f"{SEMANTIC_HASH_ALGORITHMS}"
        )
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ProcessV2SchemaError(f"pointer to {target!r} carries a malformed SHA-256")
    return {
        "kind": kind,
        "provider": provider,
        "target": target,
        "target_schema": target_schema,
        "identity_role": identity_role,
        "hash_algorithm": hash_algorithm,
        "sha256": sha256,
    }


def validate_typed_pointer(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _POINTER_FIELDS:
        raise ProcessV2SchemaError(f"{label} is not a typed pointer")
    return typed_pointer(
        kind=str(value["kind"]),
        provider=str(value["provider"]),
        target=str(value["target"]),
        target_schema=value["target_schema"],
        identity_role=str(value["identity_role"]),
        sha256=str(value["sha256"]),
        hash_algorithm=value["hash_algorithm"],
    )


# ---- Canonical serialization ----


def canonical_bytes(value: object) -> bytes:
    """The one canonical encoding every Process-V2 hash is taken over."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    """Seal a body with its self-hash under `field`."""

    if field in body:
        raise ProcessV2SchemaError(f"body already carries its self-hash field {field!r}")
    return {**dict(body), field: canonical_sha256(dict(body))}


def verify_self_hash(payload: Mapping[str, Any], *, field: str, label: str) -> None:
    declared = payload.get(field)
    body = {key: item for key, item in payload.items() if key != field}
    if declared != canonical_sha256(body):
        raise ProcessV2SchemaError(f"{label} self-hash {field!r} disagrees with its body")


# ---- The structural decision-index protocol ----


@runtime_checkable
class StructuralDecisionIndex(Protocol):
    """The minimum a downstream stage needs from a decision index.

    Expressed as a protocol so a Process-V2 index satisfies it structurally.
    Subclassing a V1 loader would drag in V1 schema and live-identity
    revalidation, which is exactly what makes a V1 loader reject the historical
    payload by construction.
    """

    @property
    def process_identity_sha256(self) -> str:
        """The process identity the decisions were computed under."""

    @property
    def completion_sha256(self) -> str:
        """The self-hash of the completion that sealed this index."""

    def counts(self) -> Mapping[str, int]:
        """Census, including `source_entries`, `admitted_entries`,
        `rejected_entries`. `source == admitted + rejected` must hold."""

    def rejected_traces_by_code(self) -> Mapping[str, int]:
        """Reason-coded rejection census."""

    def identity(self) -> Mapping[str, Any]:
        """The deterministic descriptor a consumer records as provenance."""


def require_census_reconciles(counts: Mapping[str, int], *, label: str) -> None:
    """`source == admitted + rejected`, checked wherever a census is carried."""

    missing = [
        field
        for field in ("source_entries", "admitted_entries", "rejected_entries")
        if field not in counts
    ]
    if missing:
        raise ProcessV2SchemaError(f"{label} census omits {sorted(missing)}")
    source = int(counts["source_entries"])
    admitted = int(counts["admitted_entries"])
    rejected = int(counts["rejected_entries"])
    if admitted + rejected != source:
        raise ProcessV2SchemaError(
            f"{label} census does not reconcile: {admitted} admitted + {rejected} "
            f"rejected != {source} source"
        )


__all__ = [
    "ADMISSION_COMPLETION_SCHEMA",
    "ADMISSION_COMPLETION_SCHEMA_VERSION",
    "AUTHORITY_FIELDS",
    "CHUNK_CACHE_COMPLETION_SCHEMA",
    "CHUNK_CACHE_COMPLETION_SCHEMA_VERSION",
    "CHUNK_MANIFEST_SCHEMA",
    "CHUNK_MANIFEST_SCHEMA_VERSION",
    "EXACT_COMPLETION_BINDING_SCHEMA",
    "EXACT_COMPLETION_BINDING_SCHEMA_VERSION",
    "GENERATION_COMMITTED_MARKER",
    "GENERATION_COMMITTED_SCHEMA",
    "GENERATION_COMMITTED_SCHEMA_VERSION",
    "IDENTITY_ROLES",
    "POINTER_KINDS",
    "REJECTION_LEDGER_SCHEMA",
    "REJECTION_LEDGER_SCHEMA_VERSION",
    "RESOLVED_EVIDENCE_BINDING_SCHEMA",
    "RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION",
    "RESOLVED_EVIDENCE_BINDING_STATUS",
    "RETIRED_AUTHORITY_FIELDS",
    "SCHEMA_NAMESPACE",
    "SEMANTIC_HASH_ALGORITHMS",
    "IdentityRole",
    "PointerKind",
    "ProcessV2SchemaError",
    "StructuralDecisionIndex",
    "authority_false_block",
    "canonical_bytes",
    "canonical_sha256",
    "require_authority_false",
    "require_census_reconciles",
    "self_hashed",
    "typed_pointer",
    "validate_typed_pointer",
    "verify_self_hash",
]
