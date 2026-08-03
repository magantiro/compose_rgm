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
import re
from typing import Any, Mapping, Protocol, runtime_checkable

# ---- Schema identity ----

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

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


def _walk(node: object, path: str = "") -> Any:
    """Yield every `(path, key, value)` mapping entry at any depth."""

    if isinstance(node, Mapping):
        for key, value in node.items():
            yield path, key, value
            yield from _walk(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")


def require_authority_false(payload: Mapping[str, Any], *, label: str) -> None:
    """Refuse anything that grants, omits, or misspells an authority field.

    Checked at EVERY depth, not just the top level.  Every authority block in
    this repository is nested -- `build_editing_process_v2_contract()` puts one
    under `.authority`, the census module under `.decisions` -- so a top-level
    scan cannot see the place authority actually lives, and the one function
    whose job is to refuse a granted field would pass a nested grant.

    Required top-level fields are still required at the top level: a nested
    block does not satisfy the artifact's own declaration.
    """

    for observed_path, key, value in _walk(payload):
        location = f"{observed_path}.{key}" if observed_path else key
        if key in RETIRED_AUTHORITY_FIELDS:
            raise ProcessV2SchemaError(
                f"{label} uses the retired authority field {key!r} at {location!r}; it "
                f"was renamed to {RETIRED_AUTHORITY_FIELDS[key]!r}. Two spellings for "
                "one concept mean a consumer grepping one silently misses the other, "
                "so this is an incompatibility rather than an accepted alias."
            )
        if key in AUTHORITY_FIELDS and value is not False:
            raise ProcessV2SchemaError(
                f"{label} grants authority at {location!r}; a prospective contract and "
                "a resolved evidence binding both prove identity only and never permit "
                "training or launch"
            )
    missing = [field for field in AUTHORITY_FIELDS if field not in payload]
    if missing:
        raise ProcessV2SchemaError(f"{label} omits authority fields {sorted(missing)}")


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
    # `kind` x `identity_role` coherence. Without it the three roles the module
    # docstring calls load-bearing stay constructible in nonsensical
    # combinations: a process identity, which addresses no file, could be
    # pinned to a config path under a semantic algorithm.
    if identity_role == IdentityRole.PROCESS_IDENTITY and kind != (
        PointerKind.LINEAGE_REFERENCE
    ):
        raise ProcessV2SchemaError(
            "a process-identity pin addresses no file, so it is not a "
            f"{kind!r} pointer; record it as an identity pin, or as a "
            f"{PointerKind.LINEAGE_REFERENCE!r} when it is deliberately historical"
        )
    if identity_role == IdentityRole.SEMANTIC:
        if kind not in (PointerKind.REPOSITORY_CONFIG, PointerKind.REMOTE_ARTIFACT):
            raise ProcessV2SchemaError(
                f"a {kind!r} pointer has no declared self-hash to address; only a "
                "repository config or a remote artifact carries a semantic hash"
            )
        if hash_algorithm not in SEMANTIC_HASH_ALGORITHMS:
            raise ProcessV2SchemaError(
                "a semantic pointer must declare its hash algorithm; expected one of "
                f"{SEMANTIC_HASH_ALGORITHMS}"
            )
    elif hash_algorithm is not None:
        raise ProcessV2SchemaError(
            f"a {identity_role!r} pointer must not declare a semantic hash algorithm; "
            f"{hash_algorithm!r} would not be applied and would misdescribe the pin"
        )
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise ProcessV2SchemaError(
            f"pointer to {target!r} carries a malformed SHA-256; it must be exactly "
            "64 lowercase hex characters, because an uppercase variant of the same "
            "digest hashes to a different pointer and breaks byte stability"
        )
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
    # Type-checked, never `str()`-coerced: an object whose `__str__` returns 64
    # characters would be accepted here while failing `typed_pointer` itself,
    # so coercion would make the validator weaker than the builder.
    for field in ("kind", "provider", "target", "identity_role", "sha256"):
        if not isinstance(value[field], str):
            raise ProcessV2SchemaError(
                f"{label} field {field!r} is {type(value[field]).__name__}, not str"
            )
    for field in ("target_schema", "hash_algorithm"):
        if value[field] is not None and not isinstance(value[field], str):
            raise ProcessV2SchemaError(
                f"{label} field {field!r} is {type(value[field]).__name__}, not str or None"
            )
    return typed_pointer(
        kind=value["kind"],
        provider=value["provider"],
        target=value["target"],
        target_schema=value["target_schema"],
        identity_role=value["identity_role"],
        sha256=value["sha256"],
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
    if not isinstance(declared, str) or _SHA256_RE.fullmatch(declared) is None:
        # An object with a permissive `__eq__` would otherwise verify any body.
        raise ProcessV2SchemaError(f"{label} self-hash {field!r} is not a SHA-256 string")
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


CENSUS_FIELDS: tuple[str, ...] = (
    "source_entries",
    "admitted_entries",
    "rejected_entries",
)


def require_census_reconciles(counts: Mapping[str, int], *, label: str) -> None:
    """`source == admitted + rejected`, with exact non-negative integer counts.

    The counts are type-checked rather than coerced.  `int()` accepts a string,
    a float, a bool and a bytes object, so `{"admitted": "646779"}` and
    `{"source": 3.9}` both reconciled while meaning something else entirely; and
    a negative `rejected_entries` reconciling against an inflated
    `admitted_entries` is precisely the silent support inflation this census
    exists to stop.  `bool` is excluded explicitly because it is an `int`
    subclass and `True + True == 2` is never a census.
    """

    missing = [field for field in CENSUS_FIELDS if field not in counts]
    if missing:
        raise ProcessV2SchemaError(f"{label} census omits {sorted(missing)}")
    resolved: dict[str, int] = {}
    for field in CENSUS_FIELDS:
        value = counts[field]
        if type(value) is not int:
            raise ProcessV2SchemaError(
                f"{label} census field {field!r} is {type(value).__name__}, not an "
                "exact int; a coerced count is not a count"
            )
        if value < 0:
            raise ProcessV2SchemaError(
                f"{label} census field {field!r} is negative ({value}); a negative "
                "count can reconcile an inflated total and hide lost support"
            )
        resolved[field] = value
    if resolved["admitted_entries"] + resolved["rejected_entries"] != (
        resolved["source_entries"]
    ):
        raise ProcessV2SchemaError(
            f"{label} census does not reconcile: {resolved['admitted_entries']} "
            f"admitted + {resolved['rejected_entries']} rejected != "
            f"{resolved['source_entries']} source"
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
    "CENSUS_FIELDS",
    "canonical_bytes",
    "canonical_sha256",
    "require_authority_false",
    "require_census_reconciles",
    "self_hashed",
    "typed_pointer",
    "validate_typed_pointer",
    "verify_self_hash",
]
