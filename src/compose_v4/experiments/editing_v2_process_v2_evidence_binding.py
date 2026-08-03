"""The resolved evidence binding: measured provenance, separated from frozen policy.

WHY THIS IS A SEPARATE ARTIFACT
-------------------------------
The seven Process-V2 prospective contracts used to carry an ``admitted_source``
block whose two hashes were null and were advertised as "a later rebind run may
fill them".  That was impossible, and the impossibility is structural rather
than a bug:

* the deterministic builder always emits the null body;
* the validator rebuilds the null body and requires exact equality;
* so a correctly resealed body carrying measured hashes fails validation.

Confirmed against the real code before the slot was removed: filling the two
hashes, resealing ``contract_sha256`` so it agrees with its own body, and
revalidating raises ``body differs from the deterministic rebuild at
admitted_source.completion_sha256``.  A frozen artifact having no fillable slot
is the definition of frozen, not a defect to work around.

Measured provenance therefore lives here, in a **separate versioned artifact**
that cites the prospective contract and pins what a run actually measured.  One
binding resolves one stage.

WHAT A BINDING PINS
-------------------
* the immutable prospective contract: path, physical hash, semantic hash, and
  which field carries that self-hash;
* ``ADMITTED_SOURCE_SCHEMA`` and ``ADMITTED_SOURCE_SCHEMA_VERSION`` imported from
  the owning adapter, never restated -- a hand-copied schema literal in the
  contract module had already drifted from the adapter's own constant, and a
  consumer matching the adapter would not have matched;
* the admitted-source completion, run, semantic-evidence, physical-inventory,
  adapter-implementation and process identities, with a census that reconciles;
* every other measured prerequisite the stage consumes, as a declared typed
  pointer carrying its kind, provider, target schema and hash algorithm.

WHAT IT NEVER CARRIES
---------------------
Authority.  Every field in the frozen authority vocabulary is present and false,
checked by ``require_authority_false`` at build and again at validate.  A
resolved evidence binding proves identity and provenance; it never permits
training, a Gate-0 run, a T1 run, or a bounded P50 launch.  Only a stage decision
or permit may grant authority, and only when its own frozen policy says so.

THE ADAPTER'S AUTHORITY BLOCK IS NOT INGESTED
---------------------------------------------
``ProcessV2AdmittedSource.identity()`` carries its own authority fields under a
vocabulary that spells the bounded-P50 field differently from the frozen one.
That spelling is NOT translated here and NOT copied into a binding: renaming it
at its source would move the scientific Process-V2 identity, which is an owner
decision, and translating it here would quietly create a second authority
vocabulary in a third place.

A binding therefore carries exactly one authority block, its own, in the frozen
spelling, every field false.  What it takes from the adapter's descriptor is the
measured identities and the census.  A descriptor that GRANTS anything is still
refused, by a spelling-agnostic guard: any ``*_authorized`` field that is not
exactly ``False`` is a refusal, whatever it is called.

INVARIANTS MAINTAINED (and tested)
----------------------------------
* build -> serialize -> validate -> load round-trips, and rebuilding the same
  inputs is byte-identical;
* the cited prospective contract must exist and must still hash to its pinned
  physical and semantic values; a repository target that is absent is a failure,
  never a silently dropped edge;
* a granted authority field anywhere in the binding or in the supplied descriptor,
  a census that does not reconcile, a non-integer or negative count, a drifted
  adapter schema, and a malformed typed pointer each raise with a message naming
  that defect;
* a remote-artifact prerequisite is never checked against the local filesystem.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Imported from the owning adapter, never restated.
from compose_v4.data.editing_process_v2_admitted_source import (
    ADMITTED_SOURCE_SCHEMA,
    ADMITTED_SOURCE_SCHEMA_VERSION,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    RESOLVED_EVIDENCE_BINDING_SCHEMA,
    RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION,
    RESOLVED_EVIDENCE_BINDING_STATUS,
    IdentityRole,
    PointerKind,
    ProcessV2SchemaError,
    authority_false_block,
    canonical_sha256,
    require_authority_false,
    require_census_reconciles,
    self_hashed,
    typed_pointer,
    validate_typed_pointer,
    verify_self_hash,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    GATE_ZERO_STRUCTURAL,
    P50_RECIPE_POLICY,
    SELF_HASH_FIELD,
    T1_CAPACITY_POLICY,
    T1_PANEL_POLICY,
    semantic_pointer_algorithm,
    semantic_sha256_of,
)

# ---- Errors ----


class ProcessV2EvidenceBindingError(ValueError):
    """A resolved evidence binding is unbuildable, stale, or claims authority."""


# ---- Stage vocabulary ----

#: ``stage -> the immutable prospective contract that governs it``.  A binding
#: cannot claim a stage whose contract it does not cite: the stage name alone is
#: not evidence that the right policy was read.
EVIDENCE_BINDING_STAGES: Mapping[str, str] = {
    "gate_zero": GATE_ZERO_STRUCTURAL,
    "t1_panel": T1_PANEL_POLICY,
    "t1_capacity": T1_CAPACITY_POLICY,
    "bounded_p50": P50_RECIPE_POLICY,
}

#: The self-hash field a resolved evidence binding carries.  It is deliberately
#: NOT ``contract_sha256``: a binding is not a contract, and a reader that saw
#: the same field name on both would have to know which artifact it held.
BINDING_SELF_HASH_FIELD = "evidence_binding_sha256"

#: The measured identities of the admitted source, all required.  Naming them
#: explicitly is what stops a binding from pinning whichever subset a particular
#: run happened to produce.
ADMITTED_SOURCE_IDENTITY_FIELDS: tuple[str, ...] = (
    "adapter_implementation_sha256",
    "completion_sha256",
    "physical_inventory_sha256",
    "process_identity_sha256",
    "run_identity_sha256",
    "semantic_evidence_sha256",
)

_HEX64_LENGTH = 64

#: The census fields a binding requires and re-checks.
_CENSUS_FIELDS: tuple[str, ...] = (
    "admitted_entries",
    "rejected_entries",
    "source_entries",
)


# ---- Helpers ----


def _fail(detail: str) -> None:
    raise ProcessV2EvidenceBindingError(detail)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_count(value: object, *, label: str) -> int:
    """A count must be an exact non-negative int.

    ``bool`` is excluded explicitly: it is a subclass of ``int``, so ``True``
    would otherwise be accepted as the count ``1``.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer count, got {value!r}")
    assert isinstance(value, int)  # narrowed by the guard above
    if value < 0:
        _fail(f"{label} must not be negative, got {value!r}")
    return value


def _require_no_granted_authority(payload: Mapping[str, Any], *, label: str) -> None:
    """Refuse any granted authority field, whatever it is spelled.

    Deliberately spelling-agnostic. The adapter's descriptor uses a different
    vocabulary from the frozen one, and reconciling the two is an owner decision
    because the retired spelling lives inside a file whose bytes are hashed into
    the scientific Process-V2 identity. Refusing a grant needs no such decision:
    a field named ``*_authorized`` that is not exactly ``False`` is a refusal.
    """

    granted = sorted(
        field
        for field, value in payload.items()
        if field.endswith("_authorized") and value is not False
    )
    if granted:
        _fail(
            f"{label} grants authority {granted}; a resolved evidence binding proves "
            "identity only and never permits training or launch"
        )


def _require_hex64(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != _HEX64_LENGTH:
        _fail(f"{label} must be a 64-character SHA-256, got {value!r}")
    assert isinstance(value, str)  # narrowed by the guard above
    if any(character not in "0123456789abcdef" for character in value):
        _fail(f"{label} must be lowercase hexadecimal, got {value!r}")
    return value


def _contract_pointers(contract_name: str, *, repo_root: Path) -> dict[str, Any]:
    """Pin the cited prospective contract by path, physical, and semantic hash."""

    path = Path(repo_root) / contract_name
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ProcessV2EvidenceBindingError(
            f"the prospective contract {contract_name} is not readable at {repo_root}; "
            "a resolved evidence binding may only cite a contract that exists"
        ) from error
    payload = json.loads(raw)
    if not isinstance(payload, dict) or SELF_HASH_FIELD not in payload:
        _fail(
            f"the prospective contract {contract_name} declares no {SELF_HASH_FIELD!r}, "
            "so its semantic identity cannot be pinned"
        )
    return {
        "physical": typed_pointer(
            kind=PointerKind.REPOSITORY_CONFIG,
            provider="editing_v2_process_v2_contract_chain",
            target=contract_name,
            target_schema=str(payload["schema"]),
            identity_role=IdentityRole.PHYSICAL,
            sha256=_sha256_bytes(raw),
        ),
        "self_hash_field": SELF_HASH_FIELD,
        "semantic": typed_pointer(
            kind=PointerKind.REPOSITORY_CONFIG,
            provider="editing_v2_process_v2_contract_chain",
            target=contract_name,
            target_schema=str(payload["schema"]),
            identity_role=IdentityRole.SEMANTIC,
            hash_algorithm=semantic_pointer_algorithm(contract_name, raw),
            sha256=semantic_sha256_of(contract_name, raw),
        ),
    }


def _census_block(counts: Mapping[str, Any]) -> dict[str, int]:
    """Validate and normalise the census, then re-check it through the frozen rule."""

    missing = [field for field in _CENSUS_FIELDS if field not in counts]
    if missing:
        _fail(f"admitted_source.counts omits {sorted(missing)}")
    validated = {
        str(field): _require_count(value, label=f"admitted_source.counts.{field}")
        for field, value in sorted(counts.items())
    }
    try:
        require_census_reconciles(validated, label="admitted_source.counts")
    except ProcessV2SchemaError as error:
        raise ProcessV2EvidenceBindingError(str(error)) from error
    return validated


def _admitted_source_block(admitted_source: Mapping[str, Any]) -> dict[str, Any]:
    missing = [
        field for field in ADMITTED_SOURCE_IDENTITY_FIELDS if field not in admitted_source
    ]
    if missing:
        _fail(
            f"the admitted-source identity omits {sorted(missing)}; a resolved evidence "
            f"binding pins exactly {list(ADMITTED_SOURCE_IDENTITY_FIELDS)}"
        )
    identities = {
        field: _require_hex64(admitted_source[field], label=f"admitted_source.{field}")
        for field in ADMITTED_SOURCE_IDENTITY_FIELDS
    }
    # The descriptor's own authority block is checked but never copied: a binding
    # carries exactly one authority block, its own, in the frozen spelling.
    _require_no_granted_authority(
        admitted_source, label="the admitted-source identity"
    )
    counts = admitted_source.get("counts")
    if not isinstance(counts, Mapping):
        _fail("the admitted-source identity carries no counts census")
    assert isinstance(counts, Mapping)  # narrowed by the guard above
    census = _census_block(counts)
    rejected = admitted_source.get("rejected_traces_by_code")
    if not isinstance(rejected, Mapping):
        _fail("the admitted-source identity carries no reason-coded rejection census")
    assert isinstance(rejected, Mapping)  # narrowed by the guard above
    by_code = {
        str(code): _require_count(
            value, label=f"admitted_source.rejected_traces_by_code.{code}"
        )
        for code, value in sorted(rejected.items())
    }
    rejected_total = sum(by_code.values())
    if rejected_total != census["rejected_entries"]:
        _fail(
            f"the reason-coded rejection census sums to {rejected_total}, which "
            f"disagrees with rejected_entries={census['rejected_entries']}"
        )
    return {
        **identities,
        "counts": census,
        "rejected_traces_by_code": by_code,
        "schema": ADMITTED_SOURCE_SCHEMA,
        "schema_version": int(ADMITTED_SOURCE_SCHEMA_VERSION),
    }


def _measured_prerequisites_block(
    prerequisites: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for role in sorted(prerequisites):
        try:
            block[role] = validate_typed_pointer(
                prerequisites[role], label=f"measured_prerequisites.{role}"
            )
        except ProcessV2SchemaError as error:
            raise ProcessV2EvidenceBindingError(str(error)) from error
    return block


# ---- Build ----


def build_resolved_evidence_binding(
    *,
    stage: str,
    admitted_source: Mapping[str, Any],
    measured_prerequisites: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    """Build one resolved evidence binding, sealed with its own self-hash.

    Args:
        stage: one of :data:`EVIDENCE_BINDING_STAGES`.
        admitted_source: the adapter's ``identity()`` descriptor. Its measured
            identities and census are pinned; its own authority block is checked
            for a grant but is never copied into the binding.
        measured_prerequisites: ``role -> typed pointer`` for every other
            measured input the stage consumes.
        repo_root: the checkout whose prospective contract is cited.

    Raises:
        ProcessV2EvidenceBindingError: on an unknown stage, an unreadable or
            unsealed contract, an incomplete admitted-source identity, a census
            that does not reconcile, a malformed pointer, or any granted
            authority.
    """

    if stage not in EVIDENCE_BINDING_STAGES:
        _fail(
            f"{stage!r} is not a resolved-evidence-binding stage; expected one of "
            f"{sorted(EVIDENCE_BINDING_STAGES)}"
        )
    contract_name = EVIDENCE_BINDING_STAGES[stage]
    body: dict[str, Any] = {
        "admitted_source": _admitted_source_block(admitted_source),
        "measured_prerequisites": _measured_prerequisites_block(measured_prerequisites),
        "prospective_contract": _contract_pointers(contract_name, repo_root=repo_root),
        "schema": RESOLVED_EVIDENCE_BINDING_SCHEMA,
        "schema_version": RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION,
        "stage": stage,
        "status": RESOLVED_EVIDENCE_BINDING_STATUS,
        **authority_false_block(),
    }
    require_authority_false(body, label=f"the {stage} resolved evidence binding")
    return self_hashed(body, field=BINDING_SELF_HASH_FIELD)


def serialize_resolved_evidence_binding(payload: Mapping[str, Any]) -> bytes:
    """Serialize to the byte-stable form: ``indent=2, sort_keys=True`` + ``\\n``."""

    text = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False)
    return (text + "\n").encode()


# ---- Validate ----


def validate_resolved_evidence_binding(
    value: object, *, repo_root: Path
) -> dict[str, Any]:
    """Validate one resolved evidence binding against the live repository.

    Raises:
        ProcessV2EvidenceBindingError: on any disagreement, each named
            specifically rather than collapsed into a self-hash mismatch.
    """

    if not isinstance(value, Mapping):
        _fail(f"a resolved evidence binding must be a JSON object, got {type(value).__name__}")
    assert isinstance(value, Mapping)  # narrowed by the guard above
    payload: dict[str, Any] = dict(value)

    if payload.get("schema") != RESOLVED_EVIDENCE_BINDING_SCHEMA:
        _fail(
            f"schema is {payload.get('schema')!r}, not "
            f"{RESOLVED_EVIDENCE_BINDING_SCHEMA!r}"
        )
    if payload.get("schema_version") != RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION:
        _fail(
            f"schema_version is {payload.get('schema_version')!r}, not "
            f"{RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION}"
        )
    if payload.get("status") != RESOLVED_EVIDENCE_BINDING_STATUS:
        _fail(f"status is {payload.get('status')!r}, not {RESOLVED_EVIDENCE_BINDING_STATUS!r}")
    stage = payload.get("stage")
    if stage not in EVIDENCE_BINDING_STAGES:
        _fail(f"stage is {stage!r}, not one of {sorted(EVIDENCE_BINDING_STAGES)}")
    assert isinstance(stage, str)  # narrowed by the guard above

    try:
        require_authority_false(payload, label=f"the {stage} resolved evidence binding")
    except ProcessV2SchemaError as error:
        raise ProcessV2EvidenceBindingError(str(error)) from error

    contract_name = EVIDENCE_BINDING_STAGES[stage]
    block = payload.get("prospective_contract")
    if not isinstance(block, Mapping) or set(block) != {
        "physical",
        "self_hash_field",
        "semantic",
    }:
        _fail(
            "prospective_contract must carry exactly "
            "{physical, self_hash_field, semantic}"
        )
    assert isinstance(block, Mapping)  # narrowed by the guard above
    live = _contract_pointers(contract_name, repo_root=repo_root)
    for role in ("physical", "semantic"):
        try:
            pinned = validate_typed_pointer(
                block[role], label=f"prospective_contract.{role}"
            )
        except ProcessV2SchemaError as error:
            raise ProcessV2EvidenceBindingError(str(error)) from error
        if pinned["target"] != contract_name:
            _fail(
                f"prospective_contract.{role} targets {pinned['target']!r}, but the "
                f"{stage!r} stage is governed by {contract_name!r}"
            )
        if pinned != live[role]:
            _fail(
                f"prospective_contract.{role} pins {pinned['sha256']}, but the live "
                f"value is {live[role]['sha256']}"
            )
    if block["self_hash_field"] != live["self_hash_field"]:
        _fail(
            f"prospective_contract.self_hash_field is {block['self_hash_field']!r}, "
            f"not {live['self_hash_field']!r}"
        )

    source = payload.get("admitted_source")
    if not isinstance(source, Mapping):
        _fail("admitted_source is missing or is not an object")
    assert isinstance(source, Mapping)  # narrowed by the guard above
    if source.get("schema") != ADMITTED_SOURCE_SCHEMA:
        _fail(
            f"admitted_source.schema is {source.get('schema')!r}, but the owning "
            f"adapter declares {ADMITTED_SOURCE_SCHEMA!r}"
        )
    if source.get("schema_version") != ADMITTED_SOURCE_SCHEMA_VERSION:
        _fail(
            f"admitted_source.schema_version is {source.get('schema_version')!r}, but "
            f"the owning adapter declares {ADMITTED_SOURCE_SCHEMA_VERSION}"
        )
    missing = [
        field for field in ADMITTED_SOURCE_IDENTITY_FIELDS if field not in source
    ]
    if missing:
        _fail(f"admitted_source omits {sorted(missing)}")
    _require_no_granted_authority(source, label="admitted_source")
    counts = source.get("counts")
    if not isinstance(counts, Mapping):
        _fail("admitted_source.counts is missing or is not an object")
    assert isinstance(counts, Mapping)  # narrowed by the guard above
    _census_block(counts)

    prerequisites = payload.get("measured_prerequisites")
    if not isinstance(prerequisites, Mapping):
        _fail("measured_prerequisites is missing or is not an object")
    assert isinstance(prerequisites, Mapping)  # narrowed by the guard above
    for role in sorted(prerequisites):
        try:
            pointer = validate_typed_pointer(
                prerequisites[role], label=f"measured_prerequisites.{role}"
            )
        except ProcessV2SchemaError as error:
            raise ProcessV2EvidenceBindingError(str(error)) from error
        # A repository target that is absent is a failure. A remote target is
        # verified by its stage loader and is deliberately NOT looked for on the
        # local filesystem, so it can never be misreported as a missing file.
        if pointer["kind"] == PointerKind.REPOSITORY_CONFIG:
            target = Path(repo_root) / pointer["target"]
            if not target.is_file():
                _fail(
                    f"measured_prerequisites.{role} declares the repository target "
                    f"{pointer['target']!r}, which is not present"
                )
            if pointer["identity_role"] == IdentityRole.PHYSICAL:
                observed = _sha256_bytes(target.read_bytes())
                if observed != pointer["sha256"]:
                    _fail(
                        f"measured_prerequisites.{role} pins {pointer['target']} at "
                        f"{pointer['sha256']}, but the live physical hash is {observed}"
                    )

    try:
        verify_self_hash(
            payload,
            field=BINDING_SELF_HASH_FIELD,
            label=f"the {stage} resolved evidence binding",
        )
    except ProcessV2SchemaError as error:
        raise ProcessV2EvidenceBindingError(str(error)) from error
    return payload


def resolved_evidence_binding_self_hash(payload: Mapping[str, Any]) -> str:
    """The canonical SHA-256 of ``payload`` MINUS its self-hash field."""

    return canonical_sha256(
        {key: value for key, value in payload.items() if key != BINDING_SELF_HASH_FIELD}
    )


# ---- Publication ----


def write_resolved_evidence_binding(payload: Mapping[str, Any], *, path: Path) -> Path:
    """Write one binding atomically: compute into a temporary file, then rename.

    A binding is a single file, so a rename within the same directory is the
    whole transaction. Multi-file publication is the chain publisher's problem,
    not this one's.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = serialize_resolved_evidence_binding(payload)
    handle, staged = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(raw)
        os.replace(staged, path)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
    return path


def load_resolved_evidence_binding(path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Read one binding from disk and validate it.

    Raises:
        ProcessV2EvidenceBindingError: if it is absent, unparseable, or
            inconsistent.
    """

    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ProcessV2EvidenceBindingError(
            f"the resolved evidence binding {path} is not readable"
        ) from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProcessV2EvidenceBindingError(
            f"the resolved evidence binding {path} is not valid JSON"
        ) from error
    return validate_resolved_evidence_binding(payload, repo_root=repo_root)


__all__ = [
    "ADMITTED_SOURCE_IDENTITY_FIELDS",
    "BINDING_SELF_HASH_FIELD",
    "EVIDENCE_BINDING_STAGES",
    "ProcessV2EvidenceBindingError",
    "build_resolved_evidence_binding",
    "load_resolved_evidence_binding",
    "resolved_evidence_binding_self_hash",
    "serialize_resolved_evidence_binding",
    "validate_resolved_evidence_binding",
    "write_resolved_evidence_binding",
]
