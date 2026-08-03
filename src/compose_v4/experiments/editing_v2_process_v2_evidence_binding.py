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
* the admitted-source identity descriptor, **embedded verbatim** and validated
  through its owning validator;
* every other measured prerequisite the stage consumes, as a declared typed
  pointer carrying its kind, provider, target schema and hash algorithm.

VERSION 2 DELETES A FABRICATED PROJECTION
-----------------------------------------
Version 1 did not embed the descriptor.  It projected six named fields out of it
and required all six, and **three of those six names have never existed**:
``physical_inventory_sha256``, ``process_identity_sha256`` and
``semantic_evidence_sha256`` are emitted by no version of the adapter.  A binding
built from a real ``ProcessV2AdmittedSource.identity()`` was therefore
unbuildable, and the defect survived every one of the module's tests because
``build_resolved_evidence_binding`` had no production caller and every test drove
it through a hand-written fixture that invented exactly those three names.  A
fixture shaped to the code rather than to the artifact cannot fail.

The three names are not translated to anything.  There is no migration path from
version 1, because a converter would have to fabricate three values rather than
rename them, so version 1 is refused explicitly rather than upgraded.

Embedding the descriptor verbatim, rather than projecting a subset of it, is what
makes the divergence impossible to reintroduce: there is no list of field names
here to drift from the adapter's, and the shape is checked by the adapter's own
``validate_process_v2_admitted_source_identity`` rather than by a second opinion
about it.

WHAT IT NEVER CARRIES
---------------------
Authority.  Every field in the frozen authority vocabulary is present and false,
checked by ``require_authority_false`` and by the vocabulary-free
``require_no_granted_authority`` at build and again at validate, both of which
walk the whole payload -- so the embedded descriptor's authority block is checked
at its own depth rather than trusted.  A resolved evidence binding proves
identity and provenance; it never permits training, a Gate-0 run, a T1 run, or a
bounded P50 launch.  Only a stage decision or permit may grant authority, and
only when its own frozen policy says so.

Since admitted-source schema 3 there is ONE authority vocabulary: the descriptor
carries the same seven frozen fields the binding does, so nothing is translated
and nothing is dropped.

A CORRECTION TO WHAT THIS DOCSTRING USED TO SAY
-----------------------------------------------
It previously stated that renaming the adapter's bounded-P50 field "would move
the scientific Process-V2 identity", and refused to translate the spelling on
that basis.  **That was measured and is false.**  The V2 identity hashes exactly
nineteen implementation files -- the eighteen V1 files plus
``process_v2_atom_delete.py`` -- together with the process contract and the action
codec.  ``editing_process_v2_admitted_source.py`` is not among them, so renaming
its authority field leaves both scientific identities exactly where they were.
What it does move is ``adapter_implementation_sha256``, which hashes the
adapter's own bytes and is supposed to move when the adapter changes.

The frozen field at ``editing_v2_process_identity.py`` and in
``configs/editing_v2_semantic_process_v2.json`` is a different matter and does
stay: those bytes ARE hashed into the identity.  The distinction the old claim
collapsed is between the frozen contract and the adapters around it.

INVARIANTS MAINTAINED (and tested)
----------------------------------
* build -> serialize -> validate -> load round-trips against a REAL resolved
  admitted source, and rebuilding the same inputs is byte-identical;
* the cited prospective contract must exist, must still hash to its pinned
  physical and semantic values, and must declare that it is cited by this
  binding schema at this version; a repository target that is absent is a
  failure, never a silently dropped edge;
* the field set is exactly the declared one, so a block a run "also measured"
  cannot be sealed into the artifact whose whole job is to carry measured
  provenance downstream -- failure mode 1 above, arriving through the binding
  rather than through the frozen contract;
* the binding and the embedded descriptor self-hash independently, so tampering
  with the descriptor is caught even when the binding is resealed around it;
* the Process-V2 identity the descriptor was resolved under must equal the one
  the cited contract binds -- otherwise a binding would join evidence from one
  process to a policy frozen for another;
* a granted authority field at any depth, a census that does not reconcile, a
  coerced count, a drifted adapter schema, and a malformed typed pointer each
  raise with a message naming that defect;
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

# The owning adapter validates its own descriptor. Nothing about its shape --
# not its schema string, not its version, not its field list -- is restated here,
# because every restatement is a copy that can drift, and the one this replaces
# had drifted to three field names that never existed.
from compose_v4.data.editing_process_v2_admitted_source import (
    ProcessV2AdmittedSourceIdentityError,
    validate_process_v2_admitted_source_identity,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    RESOLVED_EVIDENCE_BINDING_SCHEMA,
    RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION,
    RESOLVED_EVIDENCE_BINDING_STATUS,
    IdentityRole,
    PointerKind,
    ProcessV2SchemaError,
    authority_false_block,
    canonical_sha256,
    require_authority_false,
    require_no_granted_authority,
    self_hashed,
    typed_pointer,
    validate_typed_pointer,
    verify_self_hash,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    GATE_ZERO_STRUCTURAL,
    P50_RECIPE_POLICY,
    PROCESS_IDENTITY_PIN_FIELD,
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

#: The exact field set :func:`build_resolved_evidence_binding` emits, and the
#: complete set a valid binding may carry.  Declared next to the builder that
#: produces it so the two cannot drift.
RESOLVED_EVIDENCE_BINDING_FIELDS: tuple[str, ...] = tuple(
    sorted(
        (
            "admitted_source",
            "measured_prerequisites",
            "prospective_contract",
            "schema",
            "schema_version",
            "stage",
            "status",
            *AUTHORITY_FIELDS,
            BINDING_SELF_HASH_FIELD,
        )
    )
)

# ---- Helpers ----


def _fail(detail: str) -> None:
    raise ProcessV2EvidenceBindingError(detail)


def _require_exact_fields(payload: Mapping[str, Any], *, label: str) -> None:
    """The exact field set, and only that.

    A binding is the artifact that carries measured provenance downstream, so an
    unknown field sealed into one is carried by every consumer that reads it -- the
    module docstring's failure mode 1, arriving through the artifact built to
    prevent it rather than through the frozen contract.  Both sibling validators
    (``_require_exact_fields`` in the admitted-source adapter, ``_exact_fields`` in
    the Active8 source) check this; this one did not, so
    ``binding + {"measured_later_evidence": ...}``, resealed, validated clean.

    Key ORDER is deliberately not checked here: unlike the two descriptors, a
    binding is not embedded verbatim in another artifact, and its serializer sorts.
    """

    observed = set(payload)
    if observed != set(RESOLVED_EVIDENCE_BINDING_FIELDS):
        missing = sorted(set(RESOLVED_EVIDENCE_BINDING_FIELDS) - observed)
        unexpected = sorted(observed - set(RESOLVED_EVIDENCE_BINDING_FIELDS))
        _fail(
            f"{label} field set differs from the declared shape; missing={missing} "
            f"unexpected={unexpected}"
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_prospective_contract(
    contract_name: str, *, repo_root: Path
) -> tuple[bytes, dict[str, Any]]:
    """Read and minimally shape-check the cited contract. Read once, used twice."""

    path = Path(repo_root) / contract_name
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ProcessV2EvidenceBindingError(
            f"the prospective contract {contract_name} is not readable at {repo_root}; "
            "a resolved evidence binding may only cite a contract that exists"
        ) from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProcessV2EvidenceBindingError(
            f"the prospective contract {contract_name} is not valid JSON"
        ) from error
    if not isinstance(payload, dict) or SELF_HASH_FIELD not in payload:
        _fail(
            f"the prospective contract {contract_name} declares no {SELF_HASH_FIELD!r}, "
            "so its semantic identity cannot be pinned"
        )
    return raw, payload


def _contract_pointers(contract_name: str, *, repo_root: Path) -> dict[str, Any]:
    """Pin the cited prospective contract by path, physical, and semantic hash."""

    raw, payload = _read_prospective_contract(contract_name, repo_root=repo_root)
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


def _admitted_source_block(
    admitted_source: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    """Validate the descriptor through its OWNING validator and embed it verbatim.

    Nothing is projected, renamed, subset or normalised.  Version 1 projected six
    names out of the descriptor, three of which the adapter has never emitted, and
    the mismatch was invisible because no production caller existed and the tests
    supplied a fixture carrying those three names.  A binding that embeds the
    artifact has no field list of its own to drift.
    """

    try:
        return validate_process_v2_admitted_source_identity(
            admitted_source, repo_root=Path(repo_root)
        )
    except ProcessV2AdmittedSourceIdentityError as error:
        raise ProcessV2EvidenceBindingError(
            f"the admitted-source identity is not one this binding can pin: {error}"
        ) from error


def _check_contract_declarations(
    contract_name: str,
    *,
    repo_root: Path,
    admitted_source: Mapping[str, Any],
) -> None:
    """Prove the cited contract expects THIS binding and binds THIS process.

    Two joins that would otherwise be assumed:

    * a contract states which schema, at which version, is allowed to cite it with
      measured evidence.  A version-1 binding citing a contract that expects
      version 2 is a stale consumer, and the contract is the side that says so.
    * a contract binds a process identity.  Evidence resolved under a different
      Process-V2 identity joined to a policy frozen for this one is exactly the
      cross-process read the whole chain exists to prevent, and neither side
      catches it alone: the descriptor only knows it matches the LIVE identity,
      and the contract only knows what it pinned.
    """

    _, payload = _read_prospective_contract(contract_name, repo_root=repo_root)
    declaration = payload.get("resolved_evidence_binding")
    if not isinstance(declaration, Mapping):
        _fail(
            f"the prospective contract {contract_name} declares no "
            "resolved_evidence_binding block, so it does not state which schema may "
            "cite it with measured evidence"
        )
    assert isinstance(declaration, Mapping)  # narrowed by the guard above
    if declaration.get("cited_by_schema") != RESOLVED_EVIDENCE_BINDING_SCHEMA:
        _fail(
            f"{contract_name} expects to be cited by "
            f"{declaration.get('cited_by_schema')!r}, not "
            f"{RESOLVED_EVIDENCE_BINDING_SCHEMA!r}"
        )
    if declaration.get("cited_by_schema_version") != RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION:
        _fail(
            f"{contract_name} expects resolved-evidence-binding schema version "
            f"{declaration.get('cited_by_schema_version')!r}, but this binding is "
            f"version {RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION}; regenerate the "
            "prospective contract chain rather than resealing the binding"
        )

    process_identity = payload.get("process_identity")
    if not isinstance(process_identity, Mapping):
        _fail(f"the prospective contract {contract_name} declares no process_identity")
    assert isinstance(process_identity, Mapping)  # narrowed by the guard above
    declared = process_identity.get(PROCESS_IDENTITY_PIN_FIELD)
    resolved_under = admitted_source.get("process_v2_identity_sha256")
    if declared != resolved_under:
        _fail(
            f"the admitted source was resolved under Process-V2 identity "
            f"{resolved_under!r}, but {contract_name} binds {declared!r}; a binding may "
            "not join evidence from one process to a policy frozen for another"
        )


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
        admitted_source: the adapter's ``identity()`` descriptor, embedded
            verbatim after validating it through its owning validator.
        measured_prerequisites: ``role -> typed pointer`` for every other
            measured input the stage consumes.
        repo_root: the checkout whose prospective contract is cited, and against
            which the descriptor's live implementation hash is checked.

    Raises:
        ProcessV2EvidenceBindingError: on an unknown stage, an unreadable or
            unsealed contract, a descriptor that is not a valid admitted-source
            identity, a contract that expects a different binding version or
            binds a different process, or any granted authority.
    """

    if stage not in EVIDENCE_BINDING_STAGES:
        _fail(
            f"{stage!r} is not a resolved-evidence-binding stage; expected one of "
            f"{sorted(EVIDENCE_BINDING_STAGES)}"
        )
    contract_name = EVIDENCE_BINDING_STAGES[stage]
    source = _admitted_source_block(admitted_source, repo_root=repo_root)
    _check_contract_declarations(
        contract_name, repo_root=repo_root, admitted_source=source
    )
    body: dict[str, Any] = {
        "admitted_source": source,
        "measured_prerequisites": _measured_prerequisites_block(measured_prerequisites),
        "prospective_contract": _contract_pointers(contract_name, repo_root=repo_root),
        "schema": RESOLVED_EVIDENCE_BINDING_SCHEMA,
        "schema_version": RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION,
        "stage": stage,
        "status": RESOLVED_EVIDENCE_BINDING_STATUS,
        **authority_false_block(),
    }
    _check_authority(body, label=f"the {stage} resolved evidence binding")
    return self_hashed(body, field=BINDING_SELF_HASH_FIELD)


def _check_authority(payload: Mapping[str, Any], *, label: str) -> None:
    """Both guards, over the WHOLE payload including the embedded descriptor.

    They are complementary rather than redundant: one knows the vocabulary and
    refuses a retired spelling or a missing field, the other knows no vocabulary
    and refuses a grant under a name nobody has registered.
    """

    try:
        require_no_granted_authority(payload, label=label)
        require_authority_false(payload, label=label)
    except ProcessV2SchemaError as error:
        raise ProcessV2EvidenceBindingError(str(error)) from error


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
            f"{RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION}; version 1 required three "
            "admitted-source fields the adapter has never emitted "
            "(physical_inventory_sha256, process_identity_sha256, "
            "semantic_evidence_sha256), so there is no migration path from it and a "
            "converter would have to fabricate those three values rather than rename "
            "them"
        )
    if payload.get("status") != RESOLVED_EVIDENCE_BINDING_STATUS:
        _fail(f"status is {payload.get('status')!r}, not {RESOLVED_EVIDENCE_BINDING_STATUS!r}")
    stage = payload.get("stage")
    if stage not in EVIDENCE_BINDING_STAGES:
        _fail(f"stage is {stage!r}, not one of {sorted(EVIDENCE_BINDING_STAGES)}")
    assert isinstance(stage, str)  # narrowed by the guard above

    _check_authority(payload, label=f"the {stage} resolved evidence binding")

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

    # The embedded descriptor is validated through its owning validator, which
    # verifies its OWN self-hash. That independence is the property: resealing the
    # binding around a tampered descriptor does not make the descriptor valid, and
    # tampering with the descriptor without resealing the binding fails twice.
    source = _admitted_source_block(payload.get("admitted_source"), repo_root=repo_root)
    _check_contract_declarations(
        contract_name, repo_root=repo_root, admitted_source=source
    )

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

    # Last, so that every block whose own validator names a missing field
    # specifically keeps that diagnostic; what this adds is the field NO other
    # check looks at, which is exactly the shape a smuggled measured-evidence
    # block has.
    _require_exact_fields(payload, label=f"the {stage} resolved evidence binding")

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
    "BINDING_SELF_HASH_FIELD",
    "EVIDENCE_BINDING_STAGES",
    "RESOLVED_EVIDENCE_BINDING_FIELDS",
    "ProcessV2EvidenceBindingError",
    "build_resolved_evidence_binding",
    "load_resolved_evidence_binding",
    "resolved_evidence_binding_self_hash",
    "serialize_resolved_evidence_binding",
    "validate_resolved_evidence_binding",
    "write_resolved_evidence_binding",
]
