#!/usr/bin/env python
"""Sound generic verifier for a declared Process-V2 pointer graph.

WHAT WAS UNSOUND BEFORE
-----------------------
``scripts/verify_process_v2_hash_chain.py`` can report ``AGREES`` while failing to
prove the graph it claims to have checked.  Three mechanisms, each fixed here:

1. **Identity edges fanned out to both nodes.**  It accepted *either* live process
   identity for every artifact and recorded each pin as addressing both, so a V2
   artifact carrying the V1 value passed.  This verifier resolves an identity edge
   **contextually** -- from the declared provider, identity schema, identity schema
   version, implementing module, process semantics and value together -- to
   **exactly one** node.  A declaration that resolves to no node is a failure, and
   nothing ever resolves to two.  Cardinality is required **per required
   artifact**: each declared root declares exactly one identity edge.

2. **A pointer existed only if its target did.**  Edges were discovered by
   scanning for strings that named an *existing* file, so a deleted or misspelled
   target simply dropped out of the graph and the missing-target diagnostic was
   unreachable.  This verifier discovers **declared typed pointers** structurally,
   through ``validate_typed_pointer``.  A declared pointer cannot vanish, so a
   missing repository target is a reportable failure.

3. **Warnings coexisted with agreement.**  An unverifiable semantic derivation was
   a warning and the run still printed ``AGREES``.  Here a required edge that
   cannot be decided yields ``INCONCLUSIVE``, which is not success.

WHAT WAS STILL UNSOUND AFTER THAT, AND IS FIXED HERE
----------------------------------------------------
*Discovery by exact field-set match is not validation.*  A declared pointer or a
declared identity edge that GAINED, LOST or MISSPELLED one key stopped being
discovered, so it did not become malformed, it became **absent** -- the same
"a declared thing must not vanish" defect, one level up.  Both shapes are now
recognised as NEAR MISSES and fail (:data:`_NEAR_MISS_KEY_EDITS`).

*The graph walk proved the edges between artifacts, never the artifacts.*  A
governed root could be edited anywhere its own body is not addressed by some other
artifact's pointer and the walk still agreed -- and the graph leaf is addressed by
nothing at all.  Every known prospective root is now handed to its **owning
deterministic artifact validator** BEFORE any edge is traversed, which proves the
root's schema, exact field set, canonical bytes, self-hash, declared pointer
inventory and deterministic rebuild in one step.

*A bare hash is a pin too.*  A ``*_sha256`` value that is not part of a declared
typed pointer is invisible to a structural pointer scan, so it looks checked and is
not.  Every hash literal in every traversed artifact is now accounted for: covered
by a declared pointer, by a declared identity edge, by the artifact's own self-hash,
or by the artifact having been rebuilt deterministically.  Whatever remains is
listed by name under ``unchecked_pins`` and rendered under its own heading.  The
report states the gap rather than implying coverage.

*Deferral was indistinguishable from agreement.*  A ``remote_artifact`` pin, or an
``external_asset`` that resolves outside the checkout, is not verifiable here; the
run used to print ``AGREES`` anyway.  An unresolved deferral is now ``UNVERIFIED``
and blocks agreement.  A **versioned stage receipt** (``--stage-receipt``) is the
only thing that resolves one.  It is therefore the only permit-shaped object here
and is held to that standard: read only from a normalized path inside the checkout,
schema- and version-checked, self-hash checked, required to declare the one
disposition under which it resolves anything, refused if any field spelled
``*_authorized`` at any depth is not ``False``, and required to attribute every
resolution to its own declared stage.

*A validator is evidence, so it may be withheld but never substituted.*  The
``validators`` seam exists for control-arm measurements -- ``validators={}``
degrades a run to the edge graph that existed before roots were proven -- but it
accepted any callable, so a mapping of no-ops made a tampered governed root report
``AGREES`` and absorbed its unchecked pins.  Only the validator this module
registered for that exact artifact is ever invoked; anything else is ``UNVERIFIED``.

WHAT IT CHECKS
--------------
Roots first, then edges.

* every known prospective root is validated by its owning deterministic validator;
  a root with no registered validator is ``UNVERIFIED``, never assumed sound;
* ``repository_config`` -- the normalized target must stay inside the checkout and
  must exist.  A ``physical_sha256`` pointer is compared to the file's bytes; a
  ``semantic_sha256`` pointer is compared under the algorithm the pointer itself
  declares (``self_hash_field_v1`` or ``whole_canonical_body_v1``), so the verifier
  never has to guess.  An algorithm the target cannot satisfy is a failure, not a
  warning.  A JSON target's own ``schema`` must equal the pointer's declared
  ``target_schema``;
* ``remote_artifact`` -- deferred to its stage loader.  Never looked for on the
  local filesystem and therefore never misreported as a missing local file, but
  ``UNVERIFIED`` until a stage receipt resolves it;
* ``lineage_reference`` -- deliberately historical.  Its value is required NOT to
  equal the target's live value: a "superseded" pin that equals what is live is a
  false claim, not lineage.  Its ``target_schema`` is deliberately NOT compared to
  the target's live schema, because a lineage pin makes no currency claim at all;
* ``external_asset`` -- verified physically when it resolves inside the checkout,
  ``UNVERIFIED`` otherwise unless a stage receipt resolves it.

STATUS
------
``AGREES``       every required root was validated, every required edge was decided
                 and every one agreed.
``INCONCLUSIVE`` no disagreement found, but at least one required root or edge could
                 not be decided.  Not success.
``DISAGREES``    at least one proven disagreement.

Exit status is ``0`` for ``AGREES``, ``1`` for ``INCONCLUSIVE`` or ``DISAGREES``,
and ``2`` on a usage or I/O failure.  It writes nothing.

Usage::

    .venv/bin/python scripts/verify_process_v2_chain.py
    .venv/bin/python scripts/verify_process_v2_chain.py --root configs/other.json
    .venv/bin/python scripts/verify_process_v2_chain.py --stage-receipt receipts/p50.json
    .venv/bin/python scripts/verify_process_v2_chain.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from compose_v4.data.editing_v2_process_v2_schema import (  # noqa: E402
    POINTER_FIELDS,
    IdentityRole,
    PointerKind,
    ProcessV2SchemaError,
    canonical_sha256,
    require_authority_false,
    validate_typed_pointer,
    verify_self_hash,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (  # noqa: E402
    PROCESS_V2_CHAIN_ARTIFACTS,
    SELF_HASH_FIELD_ALGORITHM,
    WHOLE_CANONICAL_BODY_ALGORITHM,
    ProcessV2ChainError,
    load_process_v2_chain_artifact,
)
from compose_v4.rewrite.editing_v2_process_identity import (  # noqa: E402
    editing_process_v2_identity,
    editing_v2_process_identity,
)

# ---- Statuses and severities ----

AGREES = "AGREES"
INCONCLUSIVE = "INCONCLUSIVE"
DISAGREES = "DISAGREES"

FAIL = "FAIL"
"""A proven disagreement."""

UNVERIFIED = "UNVERIFIED"
"""A required root or edge the verifier could not decide. Blocks ``AGREES``."""

DEFERRED = "DEFERRED"
"""Delegated to a stage loader AND resolved by a versioned stage receipt.

An UNresolved delegation is ``UNVERIFIED``, not this: "a stage loader owns it" is
a statement about who checks it, not evidence that anyone did.
"""

# The exact field set a declared process-identity edge carries. Discovery is
# structural: an object with exactly these keys is an identity edge.
_IDENTITY_PIN_FIELD = "process_identity_sha256"
_IDENTITY_EDGE_FIELDS = frozenset(
    {
        "identity_schema",
        "identity_schema_version",
        "module",
        _IDENTITY_PIN_FIELD,
        "process_semantics",
        "provider",
    }
)

#: How far a mapping may differ from a declared field set and still be recognised
#: as a NEAR MISS of it rather than as an unrelated object.  One unit is one key
#: inserted or one key deleted, so ``2`` covers a dropped key, an added key, and a
#: single misspelled key (a deletion plus an insertion).
#:
#: The bound is not free.  Measured on the committed graph, the closest non-pointer
#: mapping shares ONE key with :data:`POINTER_FIELDS` (symmetric difference >= 6),
#: but ``configs/editing_v2_semantic_process_v2.json:.process_identity`` -- a frozen
#: description of the identity PROVIDER that deliberately pins no value -- sits at
#: symmetric difference 3 from the identity-edge field set.  Widening this to 3
#: would make that frozen artifact fail.  ``tests/test_verify_process_v2_chain_root_
#: validation.py`` pins that margin so a future artifact cannot silently move into
#: it, and the residual is stated plainly: a declaration that differs by three or
#: more key edits is not recognised, and for a governed root the identity-edge
#: cardinality rule is what still refuses it.
_NEAR_MISS_KEY_EDITS = 2

# ---- Stage receipts ----

STAGE_RECEIPT_SCHEMA = "compose.editing_v2.process_v2.stage_receipt"
STAGE_RECEIPT_SCHEMA_VERSION = 1
STAGE_RECEIPT_SELF_HASH_FIELD = "receipt_sha256"
STAGE_RECEIPT_ENTRY_FIELDS = frozenset({"kind", "target", "sha256", "verified_by"})
STAGE_RECEIPT_REQUIRED_FIELDS = frozenset(
    {"schema", "schema_version", "stage", "status", "resolved", STAGE_RECEIPT_SELF_HASH_FIELD}
)

STAGE_RECEIPT_STATUS = "STAGE_RECEIPT_PROVENANCE_ONLY_NO_DOWNSTREAM_AUTHORITY"
"""The one disposition under which a receipt resolves anything.

``status`` was a required field whose VALUE was never read, so a receipt saying
``STAGE_RECEIPT_REFUSED_NOTHING_VERIFIED`` resolved its entries exactly as one
saying it had verified them.  A field that is required and never compared is worse
than an absent one: it reads as a check.  One spelling is required rather than a
set of accepted ones, for the same reason the schema and version are: two spellings
of one disposition mean a reader grepping for one silently misses the other.
"""

_AUTHORITY_FIELD_SUFFIX = "_authorized"
"""How an authority grant is recognised without a vocabulary.

:func:`require_authority_false` is vocabulary-BOUND -- it refuses the seven known
fields and the retired spellings, and requires all seven to be present -- which is
exactly right for an artifact that must carry the block.  It cannot see a field
outside that vocabulary, and this repository has one: ``p500_authorized``.  A stage
receipt is a NON-granting artifact, so the guard it needs is the vocabulary-free
one: any field spelled ``*_authorized`` that is not exactly ``False`` is a grant.

Defined here rather than imported because the shared schema module exposes only the
vocabulary-bound guard; the one vocabulary-free implementation in this repository
(``editing_v2_process_v2_evidence_binding``) is module-private, raises that module's
error type, and scans only the top level, so a nested grant would pass it.
"""


# ---- Identity nodes ----


@dataclass(frozen=True)
class IdentityNode:
    """One computed process identity, addressed by its declaration, not its value."""

    provider: str
    identity_schema: str
    identity_schema_version: int
    module: str
    process_semantics: str
    value: str

    @property
    def name(self) -> str:
        return f"identity:{self.provider}"

    @property
    def key(self) -> tuple[str, str, int, str, str]:
        return (
            self.provider,
            self.identity_schema,
            self.identity_schema_version,
            self.module,
            self.process_semantics,
        )

    def declaration(self) -> dict[str, Any]:
        return {
            "identity_schema": self.identity_schema,
            "identity_schema_version": self.identity_schema_version,
            "module": self.module,
            "process_semantics": self.process_semantics,
            "provider": self.provider,
        }


def _providing_module(provider: Callable[[], Mapping[str, Any]]) -> str:
    """The checkout-relative source file that computes an identity.

    Derived from the live provider rather than transcribed.  A transcribed path
    keeps matching after the provider moves, which is precisely the class of stale
    binding this verifier exists to refuse, and the providers here are
    ``lru_cache`` wrappers, so the module is read from ``__module__`` rather than
    from the wrapper object.
    """

    module = sys.modules[provider.__module__]
    return Path(str(module.__file__)).resolve().relative_to(REPO_ROOT).as_posix()


def live_identity_nodes() -> tuple[IdentityNode, ...]:
    """The identity nodes this checkout currently computes.

    Each node is keyed by provider, identity schema, identity schema VERSION,
    implementing MODULE and process semantics.  Digest and role alone are not a
    key: two processes can share a role, a value can be copied between them, and a
    schema can be revised in place, so a declaration that names an outdated schema
    version or a module that no longer computes the identity must resolve to no
    node instead of matching on the strength of the fields that did not move.
    """

    nodes = []
    for provider, identity in (
        (editing_v2_process_identity, editing_v2_process_identity()),
        (editing_process_v2_identity, editing_process_v2_identity()),
    ):
        nodes.append(
            IdentityNode(
                provider=provider.__name__,
                identity_schema=str(identity["schema"]),
                identity_schema_version=int(identity["schema_version"]),  # type: ignore[call-overload]
                module=_providing_module(provider),
                process_semantics=str(identity["process_semantics"]),
                value=str(identity[_IDENTITY_PIN_FIELD]),
            )
        )
    return tuple(nodes)


# ---- Owning artifact validators ----

RootValidator = Callable[[Path], None]

_VALIDATOR_REGISTRATION = object()
"""The stamp only :func:`_chain_artifact_validator` applies.

Module-private and compared by identity, so it cannot be reproduced by naming it.
A caller that genuinely wants to forge one has to reach into this module and say
so, which is the difference between a hole and a deliberate act.
"""

_VALIDATOR_ARTIFACT_ATTRIBUTE = "chain_artifact"
_VALIDATOR_REGISTRATION_ATTRIBUTE = "registration"


def _chain_artifact_validator(name: str) -> RootValidator:
    def validate(repo_root: Path) -> None:
        load_process_v2_chain_artifact(name, repo_root=repo_root)

    # Stamped with the artifact it owns AND with the registration identity, so a
    # substituted callable cannot present itself as either.
    setattr(validate, _VALIDATOR_ARTIFACT_ATTRIBUTE, name)
    setattr(validate, _VALIDATOR_REGISTRATION_ATTRIBUTE, _VALIDATOR_REGISTRATION)
    return validate


def _is_registered_validator(name: str, validator: RootValidator) -> bool:
    """Whether ``validator`` is the registry's own validator FOR ``name``.

    Both halves are load-bearing.  The registration identity refuses a substituted
    callable outright; the artifact name refuses a registered validator that has
    been moved onto a different root, which would prove the wrong artifact while
    reporting the right one.
    """

    return (
        getattr(validator, _VALIDATOR_REGISTRATION_ATTRIBUTE, None) is _VALIDATOR_REGISTRATION
        and getattr(validator, _VALIDATOR_ARTIFACT_ATTRIBUTE, None) == name
    )


def owning_root_validators() -> dict[str, RootValidator]:
    """``root -> the deterministic validator that owns it``.

    Derived from :data:`PROCESS_V2_CHAIN_ARTIFACTS` rather than transcribed, so a
    contract added to the chain is validated the moment it is added instead of
    silently joining the set of roots nothing proves.
    """

    return {name: _chain_artifact_validator(name) for name in PROCESS_V2_CHAIN_ARTIFACTS}


# ---- Findings ----


@dataclass(frozen=True)
class Finding:
    severity: str
    category: str
    location: str
    detail: str

    def render(self) -> str:
        return f"[{self.severity}] {self.category}: {self.location}\n        {self.detail}"


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    identity_edges: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    roots: list[dict[str, Any]] = field(default_factory=list)
    unchecked_pins: list[dict[str, Any]] = field(default_factory=list)
    stage_receipts: list[dict[str, Any]] = field(default_factory=list)

    def add(self, severity: str, category: str, location: str, detail: str) -> None:
        self.findings.append(Finding(severity, category, location, detail))

    def blocks(self, location: str) -> bool:
        """Whether some finding already refuses ``location`` or something inside it."""

        return any(
            finding.severity in (FAIL, UNVERIFIED)
            and (finding.location == location or finding.location.startswith(f"{location}:"))
            for finding in self.findings
        )

    @property
    def status(self) -> str:
        if any(finding.severity == FAIL for finding in self.findings):
            return DISAGREES
        if any(finding.severity == UNVERIFIED for finding in self.findings):
            return INCONCLUSIVE
        return AGREES

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifacts": list(self.artifacts),
            "edges": list(self.edges),
            "findings": [
                {
                    "category": finding.category,
                    "detail": finding.detail,
                    "location": finding.location,
                    "severity": finding.severity,
                }
                for finding in self.findings
            ],
            "identity_edges": list(self.identity_edges),
            "roots": list(self.roots),
            "stage_receipts": list(self.stage_receipts),
            "status": self.status,
            "unchecked_pins": list(self.unchecked_pins),
        }


# ---- Reading ----

_HEX64_DIGITS = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64_DIGITS


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _walk(node: object, path: str = "") -> Iterator[tuple[str, object]]:
    yield path or ".", node
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield from _walk(value, f"{path}.{key}")
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")


def _declared_self_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    """Every top-level field that satisfies the self-hash equation.

    Identified by the equation, never by a field-name list, so an artifact that
    renames its self-hash field still self-identifies and one that grows a second
    satisfying field is reported as ambiguous rather than silently resolved.
    """

    return {
        key: value
        for key, value in payload.items()
        if key.endswith("_sha256")
        and _is_sha256(value)
        and value == canonical_sha256({k: v for k, v in payload.items() if k != key})
    }


# ---- Repository path normalization ----


def normalized_repository_path(repo_root: Path, target: str) -> Path | None:
    """``target`` as an absolute path inside ``repo_root``, or ``None`` if it is not.

    A repository pointer addresses a file tracked in THIS checkout, by its
    normalized checkout-relative path.  Two things are refused, and they are not the
    same thing:

    * an **escape** -- an absolute target, or a ``..`` component or symlink whose
      real path lands outside the tree.  Such a pin reads, hashes and agrees against
      a file the checkout does not govern, so it looks checked and is checking
      something else;
    * a **non-normalized** target that does resolve inside, such as
      ``configs/../configs/x.json``.  A target is an identity, and the chain
      compares declared targets by string, so two spellings of one file are two
      pointers that no longer compare equal and one of them silently stops
      satisfying the edge it was written for.

    The second rule is enforced GENERALLY, against the target string, not against
    a list of known bad spellings.  ``pathlib`` folds ``.``, ``//`` and a trailing
    ``/`` away before any component scan can see them, so a ``..``-only check
    accepted ``configs/./x.json``, ``configs//x.json``, ``./configs/x.json`` and
    ``configs/x.json/`` -- four more spellings of one file, each of which the walk
    enqueues and re-checks under a second graph name.  A target is therefore
    required to be in normal form ALREADY: it must be exactly what ``pathlib``
    would fold it to.  ``..`` survives that folding, so it keeps its own check.

    Normalization is decided before any read, so a correct digest is no defence.
    """

    if not target or target.startswith("/") or Path(target).is_absolute():
        return None
    candidate = Path(target)
    if any(part == ".." for part in candidate.parts):
        return None
    if candidate.as_posix() != target:
        return None
    root = repo_root.resolve()
    joined = root / candidate
    try:
        real = joined.resolve()
    except OSError:
        return None
    if real != root and not real.is_relative_to(root):
        return None
    return joined


# ---- Stage receipts ----


def _receipt_index(
    repo_root: Path, paths: Sequence[str], report: Report
) -> dict[tuple[str, str], set[str]]:
    """Load, validate and index every declared stage receipt.

    A receipt is the ONLY thing that resolves a deferred edge, so it is held to the
    same standard as the artifacts it resolves: a declared schema and version, a
    verified self-hash, a disposition that says it verified something, and no
    granted authority under any spelling.  A receipt that fails any of those
    resolves nothing at all rather than resolving what it happens to name.

    Its PATH is held to the same containment rule as a pointer target, and for the
    same reason.  ``repo_root / relative_path`` lets an absolute operand win
    outright, so a receipt named by absolute path was read from anywhere on the
    filesystem and its resolutions were accepted -- an ungoverned file deciding
    which deferred edges the checkout agrees about.
    """

    index: dict[tuple[str, str], set[str]] = {}
    for relative_path in paths:
        row: dict[str, Any] = {"path": relative_path, "resolved": 0, "result": "refused"}
        receipt_path = normalized_repository_path(repo_root, relative_path)
        if receipt_path is None:
            report.add(
                FAIL,
                "stage_receipt_path_escapes_the_checkout",
                relative_path,
                "is not a normalized path inside this checkout. A stage receipt is "
                "the only thing that turns an undecided edge into agreement, so one "
                "read from outside the tree would let an ungoverned file decide what "
                "this checkout agrees about",
            )
            report.stage_receipts.append(row)
            continue
        try:
            raw = receipt_path.read_bytes()
        except OSError as error:
            report.add(FAIL, "unreadable_stage_receipt", relative_path, str(error))
            report.stage_receipts.append(row)
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            report.add(FAIL, "malformed_stage_receipt", relative_path, str(error))
            report.stage_receipts.append(row)
            continue
        detail = _stage_receipt_defect(payload)
        if detail is not None:
            report.add(FAIL, "malformed_stage_receipt", relative_path, detail)
            report.stage_receipts.append(row)
            continue
        assert isinstance(payload, dict)  # narrowed by _stage_receipt_defect
        row["stage"] = str(payload["stage"])
        row["schema_version"] = int(payload["schema_version"])
        for entry in payload["resolved"]:
            index.setdefault((str(entry["kind"]), str(entry["target"])), set()).add(
                str(entry["sha256"])
            )
        row["resolved"] = len(payload["resolved"])
        row["result"] = "accepted"
        report.stage_receipts.append(row)
    return index


def _granted_authority_fields(payload: object) -> list[str]:
    """Every field at any depth that grants authority, whatever it is spelled.

    Deliberately vocabulary-free: the check is the SUFFIX, so a field this
    repository's authority vocabulary does not list -- ``p500_authorized`` is the
    one it actually has -- cannot slip past by not being on a list.  Depth matters
    for the same reason it does in :func:`require_authority_false`: every authority
    block in this repository is nested, so a top-level scan looks in the one place
    authority does not live.
    """

    return sorted(
        location
        for location, node in _walk(payload)
        if location.rsplit(".", 1)[-1].endswith(_AUTHORITY_FIELD_SUFFIX) and node is not False
    )


def _stage_receipt_defect(payload: object) -> str | None:
    """Why this receipt may not resolve anything, or ``None`` when it may."""

    if not isinstance(payload, dict):
        return "a stage receipt must be a JSON object"
    missing = sorted(STAGE_RECEIPT_REQUIRED_FIELDS - set(payload))
    if missing:
        return f"omits {missing}"
    if payload["schema"] != STAGE_RECEIPT_SCHEMA:
        return f"declares schema {payload['schema']!r}, not {STAGE_RECEIPT_SCHEMA!r}"
    if payload["schema_version"] != STAGE_RECEIPT_SCHEMA_VERSION:
        return (
            f"declares schema_version {payload['schema_version']!r}, not "
            f"{STAGE_RECEIPT_SCHEMA_VERSION}; an unversioned or differently versioned "
            "receipt cannot be read under this contract"
        )
    if payload["status"] != STAGE_RECEIPT_STATUS:
        return (
            f"declares status {payload['status']!r}, not {STAGE_RECEIPT_STATUS!r}; a "
            "receipt that does not state that disposition resolves nothing, whatever "
            "its entries say"
        )
    if not isinstance(payload["stage"], str) or not payload["stage"]:
        return "stage must be a non-empty string naming the stage that verified these"
    try:
        verify_self_hash(payload, field=STAGE_RECEIPT_SELF_HASH_FIELD, label="the stage receipt")
        require_authority_false(payload, label="the stage receipt")
    except ProcessV2SchemaError as error:
        return str(error)
    granted = _granted_authority_fields(payload)
    if granted:
        return (
            f"grants authority at {granted}; a stage receipt records that a stage "
            "verified some bytes and is the only thing that turns an undecided edge "
            "into agreement, so it never permits anything itself"
        )
    entries = payload["resolved"]
    if not isinstance(entries, list):
        return "resolved must be a list of resolution entries"
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or set(entry) != STAGE_RECEIPT_ENTRY_FIELDS:
            return f"resolved[{index}] must carry exactly {sorted(STAGE_RECEIPT_ENTRY_FIELDS)}"
        if not _is_sha256(entry["sha256"]):
            return f"resolved[{index}] carries a malformed SHA-256"
        for name in ("kind", "target", "verified_by"):
            if not isinstance(entry[name], str) or not entry[name]:
                return f"resolved[{index}] field {name!r} must be a non-empty string"
        # `verified_by` used to be free text nothing ever compared -- "nobody at
        # all" was accepted. There is no registry of stage loaders here to decide
        # it against, and inventing one would be inventing policy, so it is bound
        # to the receipt's own `stage` instead: one receipt carries one stage's
        # evidence, and an entry attributed to someone else belongs in that
        # someone's receipt. Both fields are now compared against something, which
        # is what the alternative -- deleting them -- would also achieve, without
        # losing the attribution a permit-shaped artifact should carry.
        if entry["verified_by"] != payload["stage"]:
            return (
                f"resolved[{index}] is attributed to {entry['verified_by']!r} but this "
                f"receipt is the {payload['stage']!r} receipt; one receipt carries one "
                "stage's evidence"
            )
    return None


def _receipt_resolves(
    receipts: Mapping[tuple[str, str], set[str]], pointer: Mapping[str, Any]
) -> bool:
    return str(pointer["sha256"]) in receipts.get(
        (str(pointer["kind"]), str(pointer["target"])), set()
    )


# ---- Edge checking ----


def _check_target_schema(
    pointer: Mapping[str, Any], raw: bytes, *, location: str, report: Report
) -> bool:
    """The target's own ``schema`` must be the schema the pointer says it binds.

    A pin that agrees on bytes while naming the wrong schema is a pin whose reader
    will validate the target under the wrong contract, and the hash comparison
    cannot see it: both sides move together.
    """

    declared = pointer["target_schema"]
    if not str(pointer["target"]).endswith(".json"):
        observed: Any = None
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return True  # reported by the caller's own parse
        observed = payload.get("schema") if isinstance(payload, dict) else None
        if observed is not None:
            observed = str(observed)
    if observed == declared:
        return True
    report.add(
        FAIL,
        "target_schema_disagrees",
        location,
        f"declares target_schema {declared!r} for {pointer['target']}, whose own "
        f"schema is {observed!r}",
    )
    return False


def _check_repository_pointer(
    pointer: Mapping[str, Any],
    *,
    location: str,
    repo_root: Path,
    report: Report,
    compare_target_schema: bool = True,
) -> str:
    """Check one repository pointer and return its result token."""

    target_path = normalized_repository_path(repo_root, str(pointer["target"]))
    if target_path is None:
        report.add(
            FAIL,
            "pointer_target_escapes_the_checkout",
            location,
            f"declares the repository target {pointer['target']!r}, which is not a "
            "normalized path inside this checkout. An absolute target, or a '..' "
            "component or symlink leaving the tree, makes a pin address a file the "
            "checkout does not govern while still hashing and agreeing; a target "
            "that is merely spelled differently -- '..' resolving back inside, a "
            "'.' component, a doubled or trailing slash -- makes a second spelling "
            "of one file, which no longer compares equal to the target the edge "
            "declares",
        )
        return "escapes_checkout"
    if not target_path.is_file():
        report.add(
            FAIL,
            "missing_repository_target",
            location,
            f"declares the repository target {pointer['target']!r}, which is not "
            "present. A declared pointer cannot be satisfied by deleting its target",
        )
        return "missing_target"
    raw = target_path.read_bytes()

    # Role coherence FIRST. A mislabelled role is more fundamental than a hash
    # mismatch, and checking the hash first masks it: the specific diagnostic
    # "this file is pinned as a process identity" degrades into a generic stale
    # pointer, which tells a reader to re-pin when the real defect is that the
    # pin means the wrong thing.
    if pointer["identity_role"] == IdentityRole.PROCESS_IDENTITY:
        report.add(
            FAIL,
            "file_pinned_as_a_process_identity",
            location,
            f"pins the file {pointer['target']} with identity role "
            f"{IdentityRole.PROCESS_IDENTITY!r}; a process identity addresses no file",
        )
        return "disagrees"

    schema_agrees = True
    if compare_target_schema:
        schema_agrees = _check_target_schema(pointer, raw, location=location, report=report)

    if pointer["identity_role"] == IdentityRole.PHYSICAL:
        observed = _sha256_bytes(raw)
        if observed == pointer["sha256"]:
            return "agrees" if schema_agrees else "disagrees"
        report.add(
            FAIL,
            "stale_physical_pointer",
            location,
            f"pins {pointer['target']} at {pointer['sha256']}, but its bytes hash to "
            f"{observed}",
        )
        return "disagrees"

    # Semantic. The algorithm is declared, so nothing is guessed.
    algorithm = pointer["hash_algorithm"]
    if not str(pointer["target"]).endswith(".json"):
        report.add(
            FAIL,
            "semantic_algorithm_unsatisfiable",
            location,
            f"pins the non-JSON target {pointer['target']} semantically under "
            f"{algorithm!r}; a target with no separable semantic body must be pinned "
            "by its physical hash",
        )
        return "disagrees"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        report.add(
            FAIL,
            "unparseable_target",
            location,
            f"{pointer['target']} is not valid JSON: {error}",
        )
        return "disagrees"
    if not isinstance(payload, dict):
        report.add(
            FAIL,
            "unparseable_target",
            location,
            f"{pointer['target']} is not a JSON object",
        )
        return "disagrees"

    if algorithm == SELF_HASH_FIELD_ALGORITHM:
        declared = _declared_self_hashes(payload)
        if not declared:
            report.add(
                FAIL,
                "semantic_algorithm_unsatisfiable",
                location,
                f"declares {algorithm!r}, but {pointer['target']} carries no field "
                "equal to the canonical hash of its body minus that field",
            )
            return "disagrees"
        if len(declared) > 1:
            report.add(
                FAIL,
                "ambiguous_self_hash",
                location,
                f"{pointer['target']} declares {sorted(declared)}, so a "
                f"{algorithm!r} pin cannot select one",
            )
            return "disagrees"
        observed = next(iter(declared.values()))
    elif algorithm == WHOLE_CANONICAL_BODY_ALGORITHM:
        if _declared_self_hashes(payload):
            report.add(
                FAIL,
                "wrong_semantic_algorithm",
                location,
                f"declares {algorithm!r}, but {pointer['target']} does carry a "
                f"self-hash field {sorted(_declared_self_hashes(payload))}, so its "
                f"semantic identity is {SELF_HASH_FIELD_ALGORITHM!r}",
            )
            return "disagrees"
        observed = canonical_sha256(payload)
    else:  # pragma: no cover - validate_typed_pointer rejects unknown algorithms
        report.add(
            UNVERIFIED,
            "unknown_semantic_algorithm",
            location,
            f"declares the semantic algorithm {algorithm!r}, which this verifier "
            "cannot apply",
        )
        return "unverified"

    if observed == pointer["sha256"]:
        return "agrees" if schema_agrees else "disagrees"
    report.add(
        FAIL,
        "stale_semantic_pointer",
        location,
        f"pins {pointer['target']} at {pointer['sha256']} under {algorithm!r}, but "
        f"that rule yields {observed}",
    )
    return "disagrees"


def _check_lineage_pointer(
    pointer: Mapping[str, Any], *, location: str, repo_root: Path, report: Report
) -> str:
    """A lineage value must be historical: equal to a live value it is a lie."""

    target_path = normalized_repository_path(repo_root, str(pointer["target"]))
    if target_path is None:
        report.add(
            FAIL,
            "pointer_target_escapes_the_checkout",
            location,
            f"records lineage for {pointer['target']!r}, which is not a normalized "
            "path inside this checkout",
        )
        return "escapes_checkout"
    if not target_path.is_file():
        # A lineage reference may outlive its target; that is what historical
        # means. There is nothing to disagree with.
        return "historical_target_absent"
    raw = target_path.read_bytes()
    if pointer["identity_role"] == IdentityRole.PHYSICAL:
        live = _sha256_bytes(raw)
    elif str(pointer["target"]).endswith(".json"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return "historical"
        declared = _declared_self_hashes(payload) if isinstance(payload, dict) else {}
        live = next(iter(declared.values())) if len(declared) == 1 else ""
    else:
        return "historical"
    if live and live == pointer["sha256"]:
        report.add(
            FAIL,
            "lineage_value_is_live",
            location,
            f"records {pointer['sha256']} as superseded, but that is the live "
            f"{pointer['identity_role']} of {pointer['target']}",
        )
        return "disagrees"
    return "historical"


def _check_pointer(
    pointer: Mapping[str, Any],
    *,
    location: str,
    repo_root: Path,
    report: Report,
    receipts: Mapping[tuple[str, str], set[str]],
) -> str:
    kind = pointer["kind"]
    if kind == PointerKind.REPOSITORY_CONFIG:
        return _check_repository_pointer(
            pointer, location=location, repo_root=repo_root, report=report
        )
    if kind == PointerKind.LINEAGE_REFERENCE:
        # A lineage pin is deliberately historical, so its declared target_schema
        # describes the SUPERSEDED artifact and must not be compared to the live
        # one. Comparing it would turn a correct lineage record into a failure the
        # moment the live artifact's schema is renamed, which is exactly when
        # lineage matters most.
        return _check_lineage_pointer(
            pointer, location=location, repo_root=repo_root, report=report
        )
    if kind == PointerKind.REMOTE_ARTIFACT:
        if _receipt_resolves(receipts, pointer):
            report.add(
                DEFERRED,
                "remote_artifact_resolved_by_stage_receipt",
                location,
                f"pins the volume-relative artifact {pointer['target']} at "
                f"{pointer['sha256']}, which a versioned stage receipt resolves",
            )
            return "resolved_by_stage_receipt"
        report.add(
            UNVERIFIED,
            "remote_artifact_unresolved_by_any_stage_receipt",
            location,
            f"pins the volume-relative artifact {pointer['target']} at "
            f"{pointer['sha256']}; it is deliberately not looked for on the local "
            "filesystem, and no versioned stage receipt resolves it, so this edge is "
            "undecided rather than agreed",
        )
        return "deferred"
    # external_asset
    inside = normalized_repository_path(repo_root, str(pointer["target"]))
    if inside is not None and inside.is_file():
        # An external asset carries no schema declaration of its own, so its
        # target_schema is not comparable even when the bytes happen to live here.
        return _check_repository_pointer(
            pointer,
            location=location,
            repo_root=repo_root,
            report=report,
            compare_target_schema=False,
        )
    if _receipt_resolves(receipts, pointer):
        report.add(
            DEFERRED,
            "external_asset_resolved_by_stage_receipt",
            location,
            f"pins the external asset {pointer['target']} at {pointer['sha256']}, "
            "which a versioned stage receipt resolves",
        )
        return "resolved_by_stage_receipt"
    report.add(
        UNVERIFIED,
        "external_asset_unresolved_outside_the_checkout",
        location,
        f"pins the external asset {pointer['target']} at {pointer['sha256']}, which "
        "does not resolve inside this checkout and which no versioned stage receipt "
        "resolves",
    )
    return "deferred"


# ---- Identity edge checking ----


def _check_identity_edge(
    edge: Mapping[str, Any],
    *,
    location: str,
    nodes: tuple[IdentityNode, ...],
    report: Report,
) -> dict[str, Any]:
    """Resolve one declared identity edge to exactly one node, then compare.

    The declaration is matched on all five naming fields, not on the digest and the
    role: provider, identity schema, identity schema VERSION, implementing MODULE
    and process semantics.  A value is not a name, so a declaration whose schema
    version or module no longer describes anything this checkout computes resolves
    to no node even when its digest is currently correct.
    """

    declared = (
        str(edge.get("provider")),
        str(edge.get("identity_schema")),
        edge.get("identity_schema_version"),
        str(edge.get("module")),
        str(edge.get("process_semantics")),
    )
    resolved = [node for node in nodes if node.key == declared]
    row: dict[str, Any] = {
        "declared": {
            "identity_schema": declared[1],
            "identity_schema_version": declared[2],
            "module": declared[3],
            "process_semantics": declared[4],
            "provider": declared[0],
        },
        "location": location,
        "resolved_nodes": [node.name for node in resolved],
        "sha256": str(edge.get(_IDENTITY_PIN_FIELD)),
    }
    if not resolved:
        report.add(
            FAIL,
            "unclassified_identity_edge",
            location,
            f"declares provider {declared[0]!r}, schema {declared[1]!r} version "
            f"{declared[2]!r}, module {declared[3]!r} and process semantics "
            f"{declared[4]!r}, which name no identity this checkout computes "
            f"({[node.key for node in nodes]}). An unclassified identity edge resolves "
            "to no node; it must never fan out to every candidate",
        )
        row["result"] = "unclassified"
        return row
    if len(resolved) > 1:  # pragma: no cover - the live table is keyed uniquely
        report.add(
            FAIL,
            "ambiguous_identity_edge",
            location,
            f"resolves to {[node.name for node in resolved]}; an identity edge must "
            "select exactly one node",
        )
        row["result"] = "ambiguous"
        return row
    node = resolved[0]
    if row["sha256"] == node.value:
        row["result"] = "agrees"
        return row
    others = [other.name for other in nodes if other.value == row["sha256"]]
    detail = (
        f"declares {node.name} but pins {row['sha256']}, and {node.name} is currently "
        f"{node.value}"
    )
    if others:
        detail += (
            f". That value is currently {', '.join(others)}, which this declaration "
            "does not name"
        )
    report.add(FAIL, "stale_identity_pin", location, detail)
    row["result"] = "disagrees"
    return row


# ---- Structural discovery ----


_Located = list[tuple[str, Mapping[str, Any]]]


def _key_edit_distance(observed: frozenset[str], declared: frozenset[str]) -> int:
    """Insertions plus deletions between two key sets."""

    return len(observed ^ declared)


@dataclass
class _Discovery:
    """What one artifact structurally declares, before anything is checked."""

    relative_path: str
    payload: dict[str, Any] | None = None
    pointers: _Located = field(default_factory=list)
    identities: _Located = field(default_factory=list)
    malformed_pointers: list[tuple[str, str]] = field(default_factory=list)
    near_miss_pointers: list[tuple[str, str]] = field(default_factory=list)
    near_miss_identities: list[tuple[str, str]] = field(default_factory=list)
    covered: set[str] = field(default_factory=set)


def _discover(payload: object, discovery: _Discovery) -> None:
    """Record the declared typed pointers and identity edges inside one artifact.

    Discovery is structural.  A mapping is a pointer if and only if it validates as
    one, and an identity edge if and only if it carries exactly the identity edge
    field set.  Nothing is inferred from a string that happens to look like a path,
    which is what let a misspelled target disappear from the old graph.

    An exact field-set match is a DISCOVERY rule, not a validation rule, so on its
    own it reopens the vanishing defect one level up: an object one key away from
    either declared shape stops being discovered, which makes it absent rather than
    malformed.  A near miss is therefore recorded and refused, never skipped.
    """

    for location, node in _walk(payload):
        if not isinstance(node, Mapping):
            continue
        keys = frozenset(node)
        if keys == _IDENTITY_EDGE_FIELDS:
            discovery.identities.append((location, node))
            discovery.covered.add(f"{location}.{_IDENTITY_PIN_FIELD}")
            continue
        if keys == POINTER_FIELDS:
            discovery.covered.add(f"{location}.sha256")
            try:
                discovery.pointers.append(
                    (location, validate_typed_pointer(node, label=location))
                )
            except ProcessV2SchemaError as error:
                # A node carrying the pointer key set IS a declared pointer.
                # Skipping one that fails validation drops it from the graph
                # exactly as a missing target used to be dropped, so its
                # diagnostic becomes unreachable and a mislabelled role degrades
                # into a generic stale pointer somewhere else. It fails here.
                discovery.malformed_pointers.append((location, str(error)))
            continue
        pointer_distance = _key_edit_distance(keys, POINTER_FIELDS)
        if pointer_distance <= _NEAR_MISS_KEY_EDITS:
            discovery.near_miss_pointers.append(
                (location, _near_miss_detail(keys, POINTER_FIELDS, "typed pointer"))
            )
            discovery.covered.update(
                f"{location}.{key}" for key in keys if key.endswith("sha256")
            )
            continue
        identity_distance = _key_edit_distance(keys, _IDENTITY_EDGE_FIELDS)
        if identity_distance <= _NEAR_MISS_KEY_EDITS:
            discovery.near_miss_identities.append(
                (location, _near_miss_detail(keys, _IDENTITY_EDGE_FIELDS, "identity edge"))
            )
            discovery.covered.update(
                f"{location}.{key}" for key in keys if key.endswith("sha256")
            )


def _near_miss_detail(observed: frozenset[str], declared: frozenset[str], shape: str) -> str:
    missing = sorted(declared - observed)
    extra = sorted(observed - declared)
    return (
        f"is one key edit away from a declared {shape} (missing={missing} "
        f"extra={extra}), so it is not an unrelated object: it is a declaration that "
        "would VANISH under an exact field-set match instead of being refused. A "
        f"declared {shape} must fail loudly, never disappear"
    )


# ---- Traversal ----


def _read_artifact(relative_path: str, *, repo_root: Path, report: Report) -> _Discovery:
    """Read one artifact and structurally discover what it declares."""

    discovery = _Discovery(relative_path)
    path = repo_root / relative_path
    if not path.is_file():
        report.add(
            FAIL,
            "missing_graph_artifact",
            relative_path,
            "is declared as a member of the graph but is not present",
        )
        return discovery
    if not relative_path.endswith(".json"):
        return discovery
    try:
        payload = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        report.add(FAIL, "unparseable_artifact", relative_path, str(error))
        return discovery
    if not isinstance(payload, dict):
        report.add(
            FAIL, "unparseable_artifact", relative_path, "artifact is not a JSON object"
        )
        return discovery
    discovery.payload = payload
    discovery.covered.update(f".{key}" for key in _declared_self_hashes(payload))
    _discover(payload, discovery)
    return discovery


def _report_structural_defects(discovery: _Discovery, report: Report) -> None:
    name = discovery.relative_path
    for location, detail in discovery.malformed_pointers:
        report.add(
            FAIL,
            "malformed_declared_pointer",
            f"{name}:{location}",
            f"carries the declared pointer key set but is not a valid pointer: {detail}",
        )
    for location, detail in discovery.near_miss_pointers:
        report.add(FAIL, "near_miss_declared_pointer", f"{name}:{location}", detail)
    for location, detail in discovery.near_miss_identities:
        report.add(FAIL, "near_miss_declared_identity_edge", f"{name}:{location}", detail)


def _check_identity_edges(
    discovery: _Discovery, *, nodes: tuple[IdentityNode, ...], report: Report
) -> int:
    for location, edge in discovery.identities:
        report.identity_edges.append(
            _check_identity_edge(
                edge,
                location=f"{discovery.relative_path}:{location}",
                nodes=nodes,
                report=report,
            )
        )
    return len(discovery.identities)


def _check_identity_cardinality(name: str, declared: int, report: Report) -> bool:
    """Whether ``name`` declares exactly one identity edge, reporting if it does not.

    Cardinality is checked PER REQUIRED ARTIFACT, not once per graph. The declared
    roots are the required artifacts, and a graph-global "at least one identity
    edge" cannot see a missing edge in one of them while the others carry theirs --
    the rest of the graph hides its absence. That matters here more than anywhere
    else because discovery is an exact field-set match: an edge that GAINS or LOSES
    a key stops being discovered, so it does not become malformed, it becomes
    absent. Counting per artifact is what makes that reportable, and it is the same
    "a declared thing must not vanish" rule typed pointers already enforce.
    """

    if declared == 1:
        return True
    if declared == 0:
        report.add(
            UNVERIFIED,
            "no_identity_edge_declared",
            name,
            "declares no process-identity edge, so nothing proves which process it "
            "describes. An identity edge is discovered by its exact declared field "
            "set, so one that gained or lost a key has vanished rather than merely "
            "been malformed, and no other artifact's edge stands in for it",
        )
        return False
    report.add(
        FAIL,
        "multiple_identity_edges_declared",
        name,
        f"declares {declared} process-identity edges; a required artifact declares "
        "exactly one, because two declarations leave 'which process does this "
        "artifact describe' unanswerable from the artifact itself",
    )
    return False


def _validate_roots(
    required: Sequence[str],
    *,
    repo_root: Path,
    identified: Mapping[str, bool],
    validators: Mapping[str, RootValidator],
    report: Report,
) -> set[str]:
    """Hand every known prospective root to its owning deterministic validator.

    This runs BEFORE any edge is traversed, because an edge between two artifacts
    proves nothing about either artifact's own body: the graph leaf is addressed by
    no pointer at all, so every value inside it -- policy thresholds, bare hash pins
    with no adjacent path, the whole projected body -- was previously unchecked
    while the run reported agreement.  The owning validator proves schema, exact
    field set, canonical bytes, self-hash, declared pointer inventory and
    deterministic rebuild in one step.

    ``validators`` may only WITHHOLD: an entry that is not the validator this
    module registered for that exact artifact is refused unrun.  A validator that
    a caller supplies is a validator whose verdict the caller wrote, and a no-op
    one is indistinguishable from a proof, so the mapping is a selection of which
    owned roots to prove rather than a substitution of what proves them.

    Returns the roots whose bodies were proven by a deterministic rebuild.
    """

    proven: set[str] = set()
    for name in required:
        validator = validators.get(name)
        if validator is None:
            report.roots.append(
                {"artifact": name, "validator": None, "result": "no_validator"}
            )
            report.add(
                UNVERIFIED,
                "root_validator_unknown",
                name,
                "is declared as a required root but no deterministic artifact "
                "validator owns it, so its schema, exact field set, canonical bytes, "
                "self-hash, pointer inventory and deterministic rebuild are undecided. "
                "An unowned root is not a sound root",
            )
            continue
        if not _is_registered_validator(name, validator):
            # A validator is EVIDENCE, so it may only be withheld, never
            # substituted. Withholding one (`validators={}`, or a subset) is the
            # control arm this parameter exists for and leaves the run strictly
            # weaker; supplying a callable of one's own would let the caller decide
            # the outcome, which is the one thing a verifier must not delegate.
            report.roots.append(
                {
                    "artifact": name,
                    "validator": _validator_name(validator),
                    "result": "substituted_validator",
                    "detail": (
                        "is not the registered validator for this artifact, so it was "
                        "not run"
                    ),
                }
            )
            report.add(
                UNVERIFIED,
                "root_validator_not_registered",
                name,
                "was offered a validator this module did not register for it, so it "
                "was NOT invoked and this root is undecided. A substituted validator "
                "returns whatever its author chooses -- including nothing at all, "
                "which is indistinguishable from a proof -- so it could only ever "
                "manufacture agreement. Withholding a validator is supported and "
                "makes a run weaker; replacing one is not",
            )
            continue
        if not identified.get(name, False):
            # The identity-edge rule has already refused this artifact, and it
            # refuses it MORE specifically than a rebuild can: "declares no
            # process-identity edge" versus "field set differs from the
            # deterministic rebuild". This is the same precedence the pointer
            # checks already apply, where role coherence is checked before the
            # hash so the specific diagnostic is not degraded into a generic one.
            # The skip is recorded, and the guard below proves it never coexists
            # with agreement.
            report.roots.append(
                {
                    "artifact": name,
                    "validator": _validator_name(validator),
                    "result": "not_validated",
                    "detail": (
                        "the identity-edge rule already refused this artifact, which "
                        "blocks agreement on its own"
                    ),
                }
            )
            continue
        try:
            validator(repo_root)
        except (ProcessV2ChainError, ProcessV2SchemaError, OSError) as error:
            report.roots.append(
                {
                    "artifact": name,
                    "validator": _validator_name(validator),
                    "result": "refused",
                    "detail": str(error),
                }
            )
            report.add(
                FAIL,
                "root_artifact_refused_by_its_validator",
                name,
                f"was refused by its owning deterministic validator: {error}",
            )
            continue
        report.roots.append(
            {
                "artifact": name,
                "validator": _validator_name(validator),
                "result": "validated",
            }
        )
        proven.add(name)
    return proven


def _validator_name(validator: RootValidator) -> str:
    closure = getattr(validator, "__qualname__", "")
    return str(closure) or repr(validator)


def _record_unchecked_pins(
    discovery: _Discovery, *, proven_by_rebuild: bool, report: Report
) -> None:
    """List every hash literal nothing in this run actually checked.

    A bare ``*_sha256`` value with no adjacent path sibling is still a pin, and a
    structural pointer scan cannot see it, so it reads as covered while nothing
    compares it to anything.  That is the worst failure mode available to a
    verifier, so the residue is stated by name instead of being left implicit.

    An artifact the run rebuilt deterministically needs no entry: every byte of it,
    bare pins included, was reproduced from its sources.
    """

    if discovery.payload is None or proven_by_rebuild:
        return
    for location, node in _walk(discovery.payload):
        if not _is_sha256(node) or location in discovery.covered:
            continue
        report.unchecked_pins.append(
            {
                "artifact": discovery.relative_path,
                "location": location,
                "sha256": str(node),
            }
        )


def verify_process_v2_chain(
    *,
    repo_root: Path,
    roots: Sequence[str] = PROCESS_V2_CHAIN_ARTIFACTS,
    stage_receipts: Sequence[str] = (),
    validators: Mapping[str, RootValidator] | None = None,
) -> dict[str, Any]:
    """Verify the declared roots and the pointer graph reachable from them.

    Roots are proven first, by their owning deterministic validators; only then are
    edges traversed.  Returns the report. Its ``status`` is ``AGREES`` only when
    every required root was validated and every required edge was decided and
    agreed; ``INCONCLUSIVE`` when something required could not be decided;
    ``DISAGREES`` on any proven disagreement.
    """

    repo_root = Path(repo_root)
    nodes = live_identity_nodes()
    report = Report()
    owning = dict(owning_root_validators() if validators is None else validators)
    receipts = _receipt_index(repo_root, tuple(stage_receipts), report)
    required = tuple(dict.fromkeys(roots))
    if not required:
        report.add(
            UNVERIFIED,
            "no_identity_edge_declared",
            "",
            "no root artifact was named, so nothing proves which process this graph "
            "describes",
        )

    # ---- Phase 1: read the roots and discover what they declare ----
    discovered: dict[str, _Discovery] = {}
    for name in required:
        discovered[name] = _read_artifact(name, repo_root=repo_root, report=report)
        report.artifacts.append(name)
        _report_structural_defects(discovered[name], report)

    # ---- Phase 2: identity edges and per-root cardinality ----
    identified: dict[str, bool] = {}
    for name in required:
        declared = _check_identity_edges(discovered[name], nodes=nodes, report=report)
        identified[name] = _check_identity_cardinality(name, declared, report)

    # ---- Phase 3: root validation, BEFORE any edge is traversed ----
    proven = _validate_roots(
        required,
        repo_root=repo_root,
        identified=identified,
        validators=owning,
        report=report,
    )

    # ---- Phase 4: traverse the declared pointer graph ----
    seen: set[str] = set()
    queue: list[str] = list(required)
    while queue:
        relative_path = queue.pop(0)
        if relative_path in seen:
            continue
        seen.add(relative_path)
        if relative_path not in discovered:
            discovered[relative_path] = _read_artifact(
                relative_path, repo_root=repo_root, report=report
            )
            report.artifacts.append(relative_path)
            _report_structural_defects(discovered[relative_path], report)
            _check_identity_edges(discovered[relative_path], nodes=nodes, report=report)
        discovery = discovered[relative_path]
        for location, pointer in discovery.pointers:
            where = f"{relative_path}:{location}"
            result = _check_pointer(
                pointer,
                location=where,
                repo_root=repo_root,
                report=report,
                receipts=receipts,
            )
            report.edges.append(
                {
                    "hash_algorithm": pointer["hash_algorithm"],
                    "identity_role": pointer["identity_role"],
                    "kind": pointer["kind"],
                    "location": where,
                    "provider": pointer["provider"],
                    "result": result,
                    "sha256": pointer["sha256"],
                    "target": pointer["target"],
                    "target_schema": pointer["target_schema"],
                }
            )
            if (
                pointer["kind"] == PointerKind.REPOSITORY_CONFIG
                and result != "escapes_checkout"
                and pointer["target"] not in seen
                and pointer["target"] not in queue
            ):
                queue.append(str(pointer["target"]))

    # ---- The residue, stated rather than implied ----
    for name in report.artifacts:
        _record_unchecked_pins(
            discovered[name], proven_by_rebuild=name in proven, report=report
        )

    # A root whose validation was skipped must always be refused by something else.
    # Without this guard the skip is a silent hole exactly where the verifier is
    # supposed to be loudest.
    for row in report.roots:
        if row["result"] != "not_validated" or report.blocks(str(row["artifact"])):
            continue
        report.add(  # pragma: no cover - the precondition already reports
            UNVERIFIED,
            "root_not_validated",
            str(row["artifact"]),
            "was not handed to its owning validator and nothing else refuses it",
        )
    return report.as_dict()


# ---- Entry point ----


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a declared Process-V2 pointer graph. Validates every known root "
            "through its owning deterministic validator, resolves every identity edge "
            "to exactly one node, checks every declared typed pointer, and refuses to "
            "report agreement when a required root or edge could not be decided. "
            "Reads only."
        )
    )
    parser.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT, help="repository root (default: this checkout)"
    )
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        default=None,
        help="a root artifact to start from (repeatable; default: the seven chain configs)",
    )
    parser.add_argument(
        "--stage-receipt",
        action="append",
        dest="stage_receipts",
        default=None,
        help=(
            "a versioned stage receipt resolving remote or external pointers "
            "(repeatable; repository-relative)"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON on stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root: Path = args.repo_root.resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    roots = tuple(args.roots) if args.roots else PROCESS_V2_CHAIN_ARTIFACTS
    report = verify_process_v2_chain(
        repo_root=root,
        roots=roots,
        stage_receipts=tuple(args.stage_receipts or ()),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _render(report)
    return 0 if report["status"] == AGREES else 1


def _render(report: Mapping[str, Any]) -> None:
    print(f"== graph ({len(report['artifacts'])} artifacts) ==")
    for name in report["artifacts"]:
        print(f"  {name}")
    print()
    print(f"== roots validated by their owning validator ({len(report['roots'])}) ==")
    for row in report["roots"]:
        print(f"  {row['result']:16s} {row['artifact']}")
        if row.get("detail"):
            print(f"      {row['detail']}")
    print()
    if report["stage_receipts"]:
        print(f"== stage receipts ({len(report['stage_receipts'])}) ==")
        for row in report["stage_receipts"]:
            print(f"  {row['result']:10s} {row['resolved']:4d} resolved  {row['path']}")
        print()
    print(f"== identity edges ({len(report['identity_edges'])}) ==")
    for row in report["identity_edges"]:
        resolved = ", ".join(row["resolved_nodes"]) or "<none>"
        print(f"  {row['location']}")
        print(f"      declares {row['declared']['provider']} -> {resolved} [{row['result']}]")
    print()
    print(f"== declared pointers ({len(report['edges'])}) ==")
    for edge in report["edges"]:
        print(
            f"  {edge['result']:26s} {edge['kind']:18s} {edge['identity_role']:22s} "
            f"{edge['target']}"
        )
    print()
    print(f"== unchecked pins -- STATED, NOT VERIFIED ({len(report['unchecked_pins'])}) ==")
    if not report["unchecked_pins"]:
        print("  none")
    for pin in report["unchecked_pins"]:
        print(f"  {pin['artifact']}:{pin['location']}  {pin['sha256']}")
    print()
    print(f"== findings ({len(report['findings'])}) ==")
    if not report["findings"]:
        print("  none")
    for finding in report["findings"]:
        print(
            f"  [{finding['severity']}] {finding['category']}: {finding['location']}\n"
            f"        {finding['detail']}"
        )
    print()
    print(f"status: {report['status']}")


if __name__ == "__main__":
    raise SystemExit(main())
