#!/usr/bin/env python
"""Read-only verifier for the Process-V2 identity and its transitive re-pin chain.

The Process-V2 semantic process identity hashes implementation sources.  Editing
any bound source therefore moves the V1 identity value, the V2 identity value,
and the V2 contract hashes, and every artifact that content-addresses one of
those artifacts has to be re-pinned in dependency order.  Stopping partway leaves
the chain internally inconsistent, and at least one link compares a config
against a source constant rather than against the live loader, so a partial
re-pin there is silently stale rather than loud.

This script is the tool to run after every source edit and last of all before a
handoff.  It answers four questions and writes nothing:

1. **Live identities.**  What are the V1 identity, the V2 identity, and the V2
   contract self and physical hashes right now, and does the committed contract
   still equal a fresh deterministic rebuild?
2. **Chain agreement.**  For every artifact reachable from those four values,
   does each pinned pointer agree with the live value it addresses?  Pins are
   DISCOVERED by scanning, not enumerated here, so a newly added pin cannot hide:
   see :func:`_scan_json_literals`, :func:`_scan_python_literals`, and
   :func:`_pointer_edges`.
3. **Obsoleted values.**  Which hash values did the current edits obsolete, and
   does anything still reference one?  The sweep covers the old physical hash of
   every changed file plus every 64-hex literal that the base version of a
   changed file contained and the current version does not.
4. **V1 boundary.**  Is every V1-named config modification hash-pointer-only, and
   is no V1-named artifact treated as Process-V2 authority?

Exit status is ``0`` when everything agrees, ``1`` on any disagreement, and ``2``
on a usage or I/O failure.  Documentation findings are reported but never fail
the run: ``docs/EDITING_PROCESS_V2_DECISION.md`` and
``docs/PROCESS_V2_IMPLEMENTATION_REPORT.md`` are historical ledgers that record
superseded values on purpose.

Usage::

    .venv/bin/python scripts/verify_process_v2_hash_chain.py
    .venv/bin/python scripts/verify_process_v2_hash_chain.py --base-revision HEAD~1
    .venv/bin/python scripts/verify_process_v2_hash_chain.py --json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from compose_v4.rewrite.editing_v2_process_identity import (  # noqa: E402
    PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_SEMANTICS,
    REJECTED_PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    REJECTED_PROCESS_V2_CANDIDATE_IDENTITY_SHA256,
    SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
    build_editing_process_v2_contract,
    editing_process_v2_identity,
    editing_v2_process_identity,
    serialize_editing_process_v2_contract,
)

# The base of the Process-V2 correction round.  Override with --base-revision to
# audit a different edit window; the script never assumes this commit exists.
PROCESS_V2_CORRECTION_BASE_REVISION = "5368c46"

V1_CONTRACT_RELATIVE_PATH = "configs/editing_v2_semantic_process_v1.json"
V2_CONTRACT_RELATIVE_PATH = "configs/editing_v2_semantic_process_v2.json"
# The V1 semantic process contract is frozen by the Process-V2 decision.
V1_CONTRACT_FROZEN_SELF_SHA256 = (
    "f928f6adaf22ba7520dd28839655c93bc523ce77317b32050d3dd1b02bbcf288"
)
PROCESS_V2_RESOLVER_RELATIVE_PATH = "src/compose_v4/rewrite/process_v2_atom_delete.py"
RATE_MODEL_RELATIVE_PATH = "src/compose_v4/model/factorized_tracelet_rate_model.py"

SCAN_GLOBS = (
    "configs/**/*.json",
    "src/**/*.py",
    "scripts/**/*.py",
    "modal_apps/**/*.py",
    "tests/**/*.py",
    "recipes/**/*.json",
)
DOCUMENTATION_GLOBS = ("docs/**/*.md",)

HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
IS_HEX64 = re.compile(r"^[0-9a-f]{64}$")
# ``NAME = "<hex>"`` and the parenthesised continuation form ruff's line length
# forces on 64-character literals.
PY_CONSTANT = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=\n]+)?=\s*\(?\s*$\n?"
    r"|^(?P<inline>[A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=\n]+)?=\s*\(?\s*[\"'](?P<value>[0-9a-f]{64})[\"']",
    re.MULTILINE,
)
# A location whose key path admits a deliberately historical value.
LINEAGE_PATH_TOKENS = ("lineage", "rejected", "superseded", "historical", "was_")

# ---- Result records ----


@dataclass(frozen=True)
class Finding:
    """One reportable disagreement or observation.

    ``artifact`` scopes the finding.  A finding about an artifact outside the
    Process-V2 chain is still reported, but it is downgraded to a warning: this
    repository content-addresses many unrelated artifacts, and a verifier that
    fails on all of them stops being usable for the chain it exists to check.
    """

    severity: str  # FAIL | WARN
    category: str
    location: str
    detail: str
    artifact: str | None = None

    def scoped(self, chain: frozenset[str]) -> Finding:
        if self.severity != "FAIL" or self.artifact is None or self.artifact in chain:
            return self
        return Finding("WARN", self.category, self.location, self.detail, self.artifact)

    def render(self) -> str:
        return f"[{self.severity}] {self.category}: {self.location}\n        {self.detail}"


@dataclass
class ValueIndex:
    """Which artifact each known hash value belongs to, and in which role."""

    by_hash: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    by_artifact: dict[str, dict[str, str]] = field(default_factory=dict)

    def add(self, artifact: str, role: str, value: str) -> None:
        self.by_hash.setdefault(value, []).append((artifact, role))
        self.by_artifact.setdefault(artifact, {})[role] = value

    def owners(self, value: str) -> list[tuple[str, str]]:
        return self.by_hash.get(value, [])


@dataclass(frozen=True)
class Literal:
    """One 64-hex literal found at a precise location.

    ``context`` is the surrounding text (the JSON key path, or the source line),
    which is what decides whether a deliberately historical value is allowed here.
    """

    relative_path: str
    location: str
    value: str
    context: str = ""


@dataclass(frozen=True)
class PointerEdge:
    """A pinned pointer: ``source`` addresses ``target`` with ``value``.

    ``target`` is either a repository-relative path or one of the two identity
    pseudo-artifacts, which are computed rather than stored.
    """

    source: str
    location: str
    target: str
    value: str


V1_IDENTITY_ARTIFACT = "identity:editing_v2_process_identity"
V2_IDENTITY_ARTIFACT = "identity:editing_process_v2_identity"
IDENTITY_ARTIFACTS = (V1_IDENTITY_ARTIFACT, V2_IDENTITY_ARTIFACT)


# ---- Deterministic hashing, matching the production derivation ----


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash_fields(payload: object) -> dict[str, str]:
    """Discover which top-level fields are self-hashes, and their live values.

    A self-hash is definitionally the canonical hash of everything else in the
    object, so no field-name list is needed: the equation identifies the field.
    """

    if not isinstance(payload, dict):
        return {}
    found: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(value, str) or not IS_HEX64.match(value):
            continue
        if not key.endswith("_sha256"):
            continue
        if value == _canonical_sha256({k: v for k, v in payload.items() if k != key}):
            found[key] = value
    return found


def _self_hash_candidates(payload: object) -> dict[str, str]:
    """Every top-level ``*_sha256`` field, whether or not it currently agrees."""

    if not isinstance(payload, dict):
        return {}
    return {
        key: value
        for key, value in payload.items()
        if key.endswith("_sha256") and isinstance(value, str) and IS_HEX64.match(value)
    }


# ---- Git access to the base revision ----


class GitUnavailable(RuntimeError):
    """The base revision could not be read, so base-relative checks are skipped."""


def _git_show(root: Path, revision: str, relative_path: str) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relative_path}"],
            capture_output=True,
            check=False,
        )
    except (FileNotFoundError, OSError) as error:  # no git binary in the image
        raise GitUnavailable(str(error)) from error
    if completed.returncode != 0:
        return None
    return completed.stdout


def _git_changed_paths(root: Path, revision: str) -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only", revision],
            capture_output=True,
            check=False,
            text=True,
        )
    except (FileNotFoundError, OSError) as error:
        raise GitUnavailable(str(error)) from error
    if completed.returncode != 0:
        raise GitUnavailable(completed.stderr.strip() or f"cannot diff against {revision}")
    return tuple(sorted(line for line in completed.stdout.splitlines() if line))


# ---- Scanning ----


def _tracked_paths(root: Path, globs: tuple[str, ...]) -> tuple[Path, ...]:
    seen: dict[str, Path] = {}
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                seen[str(path.relative_to(root))] = path
    return tuple(seen[key] for key in sorted(seen))


def _walk_json(node: object, path: str = "") -> Iterator[tuple[str, object]]:
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_json(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_json(value, f"{path}[{index}]")


def _scan_json_literals(relative_path: str, payload: object) -> tuple[Literal, ...]:
    return tuple(
        Literal(relative_path, location or ".", node, context=location)
        for location, node in _walk_json(payload)
        if isinstance(node, str) and IS_HEX64.match(node)
    )


def _scan_python_literals(relative_path: str, text: str) -> tuple[Literal, ...]:
    """Find named 64-hex constants, and any other 64-hex literal, with a line."""

    found: list[Literal] = []
    lines = text.splitlines()
    pending_name: str | None = None
    for number, line in enumerate(lines, start=1):
        assignment = re.match(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=\s*\(?\s*$", line)
        matches = HEX64.findall(line)
        if matches:
            for value in matches:
                name = pending_name or _inline_constant_name(line) or "<literal>"
                found.append(Literal(relative_path, f"{name}:{number}", value, context=f"{name} {line}"))
            pending_name = None
        elif assignment is not None:
            pending_name = assignment.group("name")
        elif line.strip() and not line.strip().startswith("#"):
            pending_name = None
    return tuple(found)


def _inline_constant_name(line: str) -> str | None:
    match = re.match(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=", line)
    return match.group("name") if match else None


def _scan_text_literals(relative_path: str, text: str) -> tuple[Literal, ...]:
    found: list[Literal] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for value in HEX64.findall(line):
            found.append(Literal(relative_path, f"line:{number}", value, context=line))
    return tuple(found)


def _pointer_edges(root: Path, relative_path: str, payload: object) -> tuple[PointerEdge, ...]:
    """Discover ``pins <target file>`` edges from sibling key naming.

    Two shapes cover every pin observed in this repository, and both are matched
    structurally rather than by an enumerated field list:

    * ``{"path": "<file>", ...*sha256*: "<hash>"}`` -- every sibling hash pins it;
    * ``{"<name>": "<file>", "<name>_...sha256": "<hash>"}`` -- prefixed siblings.
    """

    edges: list[PointerEdge] = []
    for location, node in _walk_json(payload):
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if not isinstance(value, str) or "/" not in value:
                continue
            if value.startswith("/") or not (root / value).is_file():
                continue
            for sibling, pin in node.items():
                if sibling == key or not isinstance(pin, str) or not IS_HEX64.match(pin):
                    continue
                if "sha256" not in sibling:
                    continue
                if key != "path" and not sibling.startswith(key):
                    continue
                edges.append(
                    PointerEdge(
                        source=relative_path,
                        location=f"{location or '.'}.{sibling}",
                        target=value,
                        value=pin,
                    )
                )
    return tuple(edges)


def _python_pointer_edges(root: Path, relative_path: str, text: str) -> tuple[PointerEdge, ...]:
    """Discover the same pin shape inside module-level Python dict constants.

    A pin does not stop being a pin because it is written in Python.
    ``EXPECTED_T1_CAPACITY_POLICY`` in ``editing_training_gate.py`` is exactly
    the prefixed-sibling shape :func:`_pointer_edges` already understands -- a
    path plus ``<name>_file_sha256`` and ``<name>_sha256`` -- and it went stale
    while this verifier reported agreement, because only JSON was scanned for
    edges.  Every literal-evaluable module-level dict is fed through the same
    rule so that gap cannot reopen for a differently-named constant.
    """

    try:
        module = ast.parse(text)
    except SyntaxError:
        return ()
    edges: list[PointerEdge] = []
    constants: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not names:
            continue
        try:
            payload = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        if isinstance(payload, str):
            constants[names[0]] = payload
            continue
        if not isinstance(payload, dict):
            continue
        for edge in _pointer_edges(root, relative_path, payload):
            edges.append(
                PointerEdge(
                    source=relative_path,
                    location=f"{names[0]}{edge.location}",
                    target=edge.target,
                    value=edge.value,
                )
            )
    edges.extend(_python_constant_pointer_edges(root, relative_path, constants))
    return tuple(edges)


# A constant naming a file drops these suffixes to give the family stem that its
# sibling hash constants share: `CAPACITY_POLICY_RELATIVE_PATH` pins
# `CAPACITY_POLICY_FILE_SHA256` and `CAPACITY_POLICY_SHA256`.
_PATH_CONSTANT_SUFFIXES = ("_RELATIVE_PATH", "_PATH", "_FILENAME", "_FILE")


def _python_constant_pointer_edges(
    root: Path, relative_path: str, constants: dict[str, str]
) -> tuple[PointerEdge, ...]:
    """The same pin shape again, spelled as sibling module-level constants.

    `editing_v2_semantic_t1_decision.py` pins its capacity policy this way and
    went stale exactly as the dict-shaped pin did, so recognising only the dict
    form would have closed one door and left the next one open.  The stem must
    be non-empty and cover at least two underscore-separated tokens, so an
    unrelated `PATH` constant cannot capture every hash in the module.
    """

    edges: list[PointerEdge] = []
    for name, value in constants.items():
        if "/" not in value or value.startswith("/") or not (root / value).is_file():
            continue
        stem = name
        for suffix in _PATH_CONSTANT_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if not stem or stem.count("_") < 1:
            continue
        for sibling, pin in constants.items():
            if sibling == name or not IS_HEX64.match(pin):
                continue
            if "SHA256" not in sibling.upper() or not sibling.startswith(stem):
                continue
            edges.append(
                PointerEdge(
                    source=relative_path,
                    location=sibling,
                    target=value,
                    value=pin,
                )
            )
    return tuple(edges)


def _identity_pins(literals: tuple[Literal, ...]) -> tuple[Literal, ...]:
    """Discover pins on the two computed process identities.

    A process identity is not a file, so it has no physical hash to match against
    and cannot be found by path adjacency.  This repository names every such pin
    with a ``process_identity_sha256`` suffix, in JSON keys and in Python
    constants alike, which is the only convention this function relies on.  The
    pin is then checked against BOTH live identities, so nothing has to guess
    which of the two an artifact meant.
    """

    return tuple(
        literal
        for literal in literals
        if literal.relative_path.endswith(".json")
        and literal.context.lower().endswith("process_identity_sha256")
        or not literal.relative_path.endswith(".json")
        and literal.location.split(":")[0].lower().endswith("process_identity_sha256")
    )


# ---- Value tables ----


def _index_file(index: ValueIndex, artifact: str, raw: bytes) -> None:
    index.add(artifact, "physical_sha256", _sha256_bytes(raw))
    if not artifact.endswith(".json"):
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return
    for field_name, value in _self_hash_fields(payload).items():
        index.add(artifact, field_name, value)


def _live_value_index(root: Path, paths: tuple[Path, ...]) -> ValueIndex:
    index = ValueIndex()
    for path in paths:
        _index_file(index, str(path.relative_to(root)), path.read_bytes())
    index.add("identity:editing_v2_process_identity", "process_identity_sha256",
              str(editing_v2_process_identity()["process_identity_sha256"]))
    index.add("identity:editing_process_v2_identity", "process_identity_sha256",
              str(editing_process_v2_identity()["process_identity_sha256"]))
    return index


def _base_value_index(root: Path, revision: str, paths: tuple[Path, ...]) -> ValueIndex:
    index = ValueIndex()
    for path in paths:
        relative_path = str(path.relative_to(root))
        raw = _git_show(root, revision, relative_path)
        if raw is not None:
            _index_file(index, relative_path, raw)
    return index


# ---- Checks ----


class ContractUnbuildable(RuntimeError):
    """The contract builder could not run, so contract-dependent checks are skipped."""


def _build_contract() -> dict[str, Any]:
    """Build the contract, turning a mid-edit import failure into a clear error.

    The builder imports the Process-V2 resolver.  While another worker is editing
    that module the import can fail, and a traceback would hide which check was
    unable to run.
    """

    try:
        return build_editing_process_v2_contract()
    except (ImportError, AttributeError) as error:
        raise ContractUnbuildable(
            f"build_editing_process_v2_contract() could not import its bound sources: {error}"
        ) from error


def _check_live_contract(root: Path) -> tuple[dict[str, str], list[Finding]]:
    findings: list[Finding] = []
    contract_path = root / V2_CONTRACT_RELATIVE_PATH
    committed = contract_path.read_bytes() if contract_path.is_file() else b""
    rebuilt = serialize_editing_process_v2_contract(_build_contract())
    if committed != rebuilt:
        findings.append(
            Finding(
                "FAIL",
                "contract_drift",
                V2_CONTRACT_RELATIVE_PATH,
                "the committed contract differs from build_editing_process_v2_contract(); "
                "regenerate it with write_editing_process_v2_contract()",
            )
        )
    v1 = editing_v2_process_identity()
    v2 = editing_process_v2_identity()
    if v1["process_identity_sha256"] == v2["process_identity_sha256"]:
        findings.append(
            Finding(
                "FAIL",
                "identity_collision",
                "editing_v2_process_identity/editing_process_v2_identity",
                "the V1 and V2 process identities are equal, so neither can invalidate the other",
            )
        )
    live = {
        "v1_process_identity_sha256": str(v1["process_identity_sha256"]),
        "v2_process_identity_sha256": str(v2["process_identity_sha256"]),
        "v2_contract_sha256": str(_build_contract()["contract_sha256"]),
        "v2_contract_physical_sha256": _sha256_bytes(rebuilt),
        "v2_contract_committed_physical_sha256": _sha256_bytes(committed),
        "v1_contract_physical_sha256": _sha256_bytes((root / V1_CONTRACT_RELATIVE_PATH).read_bytes()),
    }
    return live, findings


def _is_lineage_location(literal: Literal) -> bool:
    haystack = f"{literal.location} {literal.context}".lower()
    return any(token in haystack for token in LINEAGE_PATH_TOKENS)


def _classify_literals(
    literals: tuple[Literal, ...],
    live: ValueIndex,
    base: ValueIndex,
    base_available: bool,
) -> tuple[list[Finding], dict[str, int]]:
    findings: list[Finding] = []
    counts = {"agrees": 0, "stale": 0, "lineage": 0, "unclassified": 0}
    for literal in literals:
        owners = live.owners(literal.value)
        if owners:
            counts["agrees"] += 1
            continue
        if literal.value in {
            SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
            REJECTED_PROCESS_V2_CANDIDATE_IDENTITY_SHA256,
        }:
            if _is_lineage_location(literal):
                counts["lineage"] += 1
                continue
            counts["stale"] += 1
            findings.append(
                Finding(
                    "FAIL",
                    "historical_identity_used_as_a_live_pin",
                    f"{literal.relative_path}:{literal.location}",
                    f"{literal.value} is a recorded historical identity and may appear only in a "
                    "lineage, rejected, or superseded field",
                    artifact=literal.relative_path,
                )
            )
            continue
        if base_available and base.owners(literal.value):
            owner, role = base.owners(literal.value)[0]
            now = live.by_artifact.get(owner, {}).get(role, "<gone>")
            counts["stale"] += 1
            findings.append(
                Finding(
                    "FAIL",
                    "obsoleted_value_still_referenced",
                    f"{literal.relative_path}:{literal.location}",
                    f"{literal.value} was {owner} {role} at the base revision; it is now {now}",
                    artifact=literal.relative_path,
                )
            )
            continue
        counts["unclassified"] += 1
    return findings, counts


def _check_identity_pins(pins: tuple[Literal, ...], live: ValueIndex) -> list[Finding]:
    """Every process-identity pin must equal one of the two live identities."""

    accepted = {
        live.by_artifact[artifact]["process_identity_sha256"]: artifact
        for artifact in IDENTITY_ARTIFACTS
    }
    findings: list[Finding] = []
    for pin in pins:
        if pin.value in accepted or _is_lineage_location(pin):
            continue
        findings.append(
            Finding(
                "FAIL",
                "stale_process_identity_pin",
                f"{pin.relative_path}:{pin.location}",
                f"pins the semantic process identity at {pin.value}, which is neither the live V1 "
                f"identity nor the live V2 identity ({sorted(accepted)})",
                artifact=pin.relative_path,
            )
        )
    return findings


def _target_values(root: Path, target: str, live: ValueIndex) -> dict[str, str]:
    """Live values of a pointer target, hashing files outside the scan on demand."""

    known = live.by_artifact.get(target)
    if known:
        return known
    path = root / target
    if not path.is_file():
        return {}
    return {"physical_sha256": _sha256_bytes(path.read_bytes())}


def _check_pointer_edges(
    root: Path,
    edges: tuple[PointerEdge, ...],
    live: ValueIndex,
    base: ValueIndex,
    base_available: bool,
) -> list[Finding]:
    """Compare every discovered pointer against the live values of its target.

    A pin is only a failure when the verifier can actually derive the value it
    should hold.  A target with no discoverable self-hash may be addressed by a
    semantic derivation this tool does not implement, so a mismatch there is
    reported as unverifiable rather than asserted to be wrong.
    """

    findings: list[Finding] = []
    for edge in edges:
        target_values = _target_values(root, edge.target, live)
        if edge.value in target_values.values():
            continue
        base_values = base.by_artifact.get(edge.target, {}) if base_available else {}
        if edge.value in base_values.values():
            role = next(key for key, value in base_values.items() if value == edge.value)
            findings.append(
                Finding(
                    "FAIL",
                    "stale_pointer",
                    f"{edge.source}:{edge.location}",
                    f"pins {edge.target} {role} at {edge.value}, which moved to "
                    f"{target_values.get(role, '<gone>')}",
                    artifact=edge.source,
                )
            )
            continue
        if not target_values:
            findings.append(
                Finding(
                    "WARN",
                    "pointer_target_missing",
                    f"{edge.source}:{edge.location}",
                    f"pins {edge.target} at {edge.value}, but that file is not present",
                    artifact=edge.source,
                )
            )
            continue
        if set(target_values) == {"physical_sha256"}:
            findings.append(
                Finding(
                    "WARN",
                    "unverifiable_pointer_derivation",
                    f"{edge.source}:{edge.location}",
                    f"pins {edge.target} at {edge.value}; the target carries no self-hash field, "
                    f"so this verifier can only derive its physical hash "
                    f"({target_values['physical_sha256']}) and cannot confirm or refute a "
                    "semantic derivation",
                    artifact=edge.source,
                )
            )
            continue
        findings.append(
            Finding(
                "FAIL",
                "unresolved_pointer",
                f"{edge.source}:{edge.location}",
                f"pins {edge.target} at {edge.value}, which is neither a live value "
                f"({sorted(target_values.values())}) nor a base value of that artifact"
                + ("" if not base_available else "; it was already unresolved at the base"),
                artifact=edge.source,
            )
        )
    return findings


def _check_self_hashes(
    root: Path, revision: str, paths: tuple[Path, ...], base_available: bool
) -> list[Finding]:
    """A config that carried an agreeing self-hash at base must still carry one.

    Nothing here knows a field name: a self-hash is identified by the equation it
    satisfies, so a config that never self-hashed is silently skipped and one
    whose self-hash stopped agreeing is reported.
    """

    if not base_available:
        return []
    findings: list[Finding] = []
    for path in paths:
        relative_path = str(path.relative_to(root))
        if not relative_path.startswith("configs/") or not relative_path.endswith(".json"):
            continue
        raw = _git_show(root, revision, relative_path)
        if raw is None:
            continue
        try:
            before = _self_hash_fields(json.loads(raw))
            after = _self_hash_fields(json.loads(path.read_bytes()))
        except json.JSONDecodeError:
            continue
        for missing in sorted(set(before) - set(after)):
            declared = _self_hash_candidates(json.loads(path.read_bytes())).get(missing)
            findings.append(
                Finding(
                    "FAIL",
                    "self_hash_drifted",
                    f"{relative_path}:{missing}",
                    f"{missing} was the canonical hash of the rest of the object at the base "
                    f"revision and no longer is (it now reads {declared}); re-seal the artifact",
                    artifact=relative_path,
                )
            )
    return findings


def _check_obsoleted_sweep(
    root: Path,
    revision: str,
    changed: tuple[str, ...],
    literals: tuple[Literal, ...],
    doc_literals: tuple[Literal, ...],
    live: ValueIndex,
) -> tuple[list[Finding], list[str]]:
    """Report every reference to a value this edit window obsoleted.

    A value is obsolete when it was the physical hash of a file the window
    changed, or when it appeared in a changed file's base version and not in its
    current version.  A value that is still the live value of some artifact is
    never obsolete, whatever dropped it.
    """

    obsolete: dict[str, str] = {}
    for relative_path in changed:
        raw = _git_show(root, revision, relative_path)
        if raw is None:
            continue
        current_path = root / relative_path
        current = current_path.read_bytes() if current_path.is_file() else b""
        old_physical = _sha256_bytes(raw)
        if old_physical != _sha256_bytes(current):
            obsolete[old_physical] = f"the base physical hash of {relative_path}"
        base_text = raw.decode("utf-8", errors="replace")
        current_text = current.decode("utf-8", errors="replace")
        dropped = set(HEX64.findall(base_text)) - set(HEX64.findall(current_text))
        for value in sorted(dropped):
            obsolete.setdefault(value, f"a 64-hex literal dropped from {relative_path}")
    for value in [value for value in obsolete if live.owners(value)]:
        del obsolete[value]

    findings: list[Finding] = []
    for literal in literals:
        reason = obsolete.get(literal.value)
        if reason is None or _is_lineage_location(literal):
            continue
        findings.append(
            Finding(
                "FAIL",
                "obsoleted_by_this_edit_window",
                f"{literal.relative_path}:{literal.location}",
                f"still references {literal.value}, which is {reason}",
                artifact=literal.relative_path,
            )
        )
    for literal in doc_literals:
        if literal.value in obsolete:
            findings.append(
                Finding(
                    "WARN",
                    "obsoleted_value_recorded_in_documentation",
                    f"{literal.relative_path}:{literal.location}",
                    f"records {literal.value}, which is {obsolete[literal.value]}; "
                    "documentation ledgers may record superseded values on purpose",
                    artifact=literal.relative_path,
                )
            )
    return findings, sorted(obsolete)


def _json_leaves(node: object, path: str = "") -> dict[str, object]:
    leaves: dict[str, object] = {}
    for location, value in _walk_json(node, path):
        if not isinstance(value, (dict, list)):
            leaves[location] = value
    return leaves


def _check_v1_named_configs(
    root: Path, revision: str, changed: tuple[str, ...]
) -> list[Finding]:
    """Every V1-named config change must be hash-pointer-only."""

    findings: list[Finding] = []
    for relative_path in changed:
        if not relative_path.startswith("configs/") or not relative_path.endswith(".json"):
            continue
        if "_v1" not in Path(relative_path).stem:
            continue
        raw = _git_show(root, revision, relative_path)
        if raw is None:
            continue
        current_path = root / relative_path
        if not current_path.is_file():
            findings.append(
                Finding(
                    "FAIL",
                    "v1_config_deleted",
                    relative_path,
                    "a V1-named config was removed",
                    artifact=relative_path,
                )
            )
            continue
        try:
            before = _json_leaves(json.loads(raw))
            after = _json_leaves(json.loads(current_path.read_bytes()))
        except json.JSONDecodeError as error:
            findings.append(
                Finding(
                    "FAIL", "v1_config_unreadable", relative_path, str(error), artifact=relative_path
                )
            )
            continue
        for location in sorted(set(before) ^ set(after)):
            findings.append(
                Finding(
                    "FAIL",
                    "v1_config_shape_changed",
                    f"{relative_path}:{location}",
                    "a V1-named config gained or lost a field; only hash pointers may move",
                    artifact=relative_path,
                )
            )
        for location in sorted(set(before) & set(after)):
            old, new = before[location], after[location]
            if old == new:
                continue
            both_hashes = (
                isinstance(old, str)
                and isinstance(new, str)
                and bool(IS_HEX64.match(old))
                and bool(IS_HEX64.match(new))
            )
            if not both_hashes:
                findings.append(
                    Finding(
                        "FAIL",
                        "v1_config_policy_changed",
                        f"{relative_path}:{location}",
                        f"{old!r} -> {new!r} is not a hash-pointer change; policy, thresholds, "
                        "counts, cell definitions, and authority fields are frozen",
                        artifact=relative_path,
                    )
                )
    return findings


def _check_v1_not_process_v2_authority(root: Path, paths: tuple[Path, ...]) -> list[Finding]:
    findings: list[Finding] = []
    v1_contract = root / V1_CONTRACT_RELATIVE_PATH
    payload = json.loads(v1_contract.read_bytes())
    if payload.get("contract_sha256") != V1_CONTRACT_FROZEN_SELF_SHA256:
        findings.append(
            Finding(
                "FAIL",
                "v1_contract_modified",
                V1_CONTRACT_RELATIVE_PATH,
                f"contract_sha256 is {payload.get('contract_sha256')!r}, not the frozen "
                f"{V1_CONTRACT_FROZEN_SELF_SHA256}",
                artifact=V1_CONTRACT_RELATIVE_PATH,
            )
        )
    for path in paths:
        relative_path = str(path.relative_to(root))
        if not relative_path.startswith("configs/") or "_v1" not in Path(relative_path).stem:
            continue
        try:
            body = json.loads(path.read_bytes())
        except json.JSONDecodeError:
            continue
        for location, value in _walk_json(body):
            lineage = any(token in location.lower() for token in LINEAGE_PATH_TOKENS)
            if value == PROCESS_V2_SEMANTICS and not lineage:
                findings.append(
                    Finding(
                        "FAIL",
                        "v1_artifact_claims_process_v2",
                        f"{relative_path}:{location}",
                        f"a V1-named artifact declares {PROCESS_V2_SEMANTICS!r}",
                        artifact=relative_path,
                    )
                )
            if value == V2_CONTRACT_RELATIVE_PATH and not lineage:
                findings.append(
                    Finding(
                        "FAIL",
                        "v1_artifact_binds_the_v2_contract",
                        f"{relative_path}:{location}",
                        "a V1-named artifact binds the Process-V2 contract as its authority",
                        artifact=relative_path,
                    )
                )
    return findings


def _check_frozen_names(root: Path) -> list[Finding]:
    """The contract's declared names must match the live implementation source.

    Source text is read rather than imported on purpose: the verifier has to work
    while another module is mid-edit, and an import failure there would mask the
    very disagreement this section exists to report.
    """

    findings: list[Finding] = []
    contract = _build_contract()
    authority = contract["atom_delete"]["admission_authority"]

    model_source = (root / RATE_MODEL_RELATIVE_PATH).read_text()
    match = re.search(
        r"^PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS\s*=\s*[\"'](?P<value>[^\"']+)[\"']",
        model_source,
        re.MULTILINE,
    )
    if match is None:
        findings.append(
            Finding(
                "FAIL",
                "model_mode_constant_missing",
                RATE_MODEL_RELATIVE_PATH,
                "PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS is not defined at module scope",
                artifact=RATE_MODEL_RELATIVE_PATH,
            )
        )
    elif match.group("value") != PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS:
        findings.append(
            Finding(
                "FAIL",
                "atom_delete_mode_disagrees",
                RATE_MODEL_RELATIVE_PATH,
                f"the model defines {match.group('value')!r} but the contract binds "
                f"{PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS!r}",
                artifact=RATE_MODEL_RELATIVE_PATH,
            )
        )

    resolver_source = (root / PROCESS_V2_RESOLVER_RELATIVE_PATH).read_text()
    for role in ("resolver", "enumerator", "mask"):
        name = str(authority[role])
        if re.search(rf"^def {re.escape(name)}\(", resolver_source, re.MULTILINE) is None:
            findings.append(
                Finding(
                    "FAIL",
                    "declared_symbol_missing",
                    f"{PROCESS_V2_RESOLVER_RELATIVE_PATH}:{name}",
                    f"the contract binds {role}={name!r}, which the resolver does not define",
                    artifact=PROCESS_V2_RESOLVER_RELATIVE_PATH,
                )
            )
    declared = list(authority["rejection_codes"])
    implemented = re.findall(
        r"^\s{4}[A-Z][A-Z0-9_]*\s*=\s*[\"'](?P<value>[a-z0-9_]+)[\"']",
        _enum_body(resolver_source, "ProcessV2AtomDeleteRejectionCode"),
        re.MULTILINE,
    )
    if implemented != declared:
        findings.append(
            Finding(
                "FAIL",
                "rejection_codes_disagree",
                f"{PROCESS_V2_RESOLVER_RELATIVE_PATH}:ProcessV2AtomDeleteRejectionCode",
                f"implementation {implemented} does not equal the contract's {declared}",
                artifact=PROCESS_V2_RESOLVER_RELATIVE_PATH,
            )
        )
    findings.extend(_check_removed_mode_name(root))
    return findings


def _check_removed_mode_name(root: Path) -> list[Finding]:
    """The removed mode string may survive only as a recorded removed name."""

    findings: list[Finding] = []
    removed = REJECTED_PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
    for path in _tracked_paths(root, SCAN_GLOBS):
        relative_path = str(path.relative_to(root))
        if relative_path.endswith(".json"):
            try:
                payload = json.loads(path.read_bytes())
            except json.JSONDecodeError:
                continue
            occurrences = [
                (location or ".", location)
                for location, node in _walk_json(payload)
                if node == removed
            ]
        else:
            occurrences = [
                (f"line:{number}", line)
                for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
                if removed in line
            ]
        for location, context in occurrences:
            if any(token in context.lower() for token in LINEAGE_PATH_TOKENS):
                continue
            findings.append(
                Finding(
                    "FAIL",
                    "removed_mode_name_survives",
                    f"{relative_path}:{location}",
                    f"{removed!r} was removed and must not be reintroduced or aliased; it may "
                    "appear only as a recorded removed or rejected name",
                    artifact=relative_path,
                )
            )
    return findings


def _enum_body(source: str, class_name: str) -> str:
    match = re.search(rf"^class {re.escape(class_name)}\(.*?\):$", source, re.MULTILINE)
    if match is None:
        return ""
    rest = source[match.end() :]
    end = re.search(r"^\S", rest, re.MULTILINE)
    return rest[: end.start()] if end else rest


# ---- Chain assembly ----


def _chain_report(
    literals: tuple[Literal, ...],
    identity_pins: tuple[Literal, ...],
    live: ValueIndex,
    base: ValueIndex,
    seeds: tuple[str, ...],
    edges: tuple[PointerEdge, ...] = (),
) -> list[dict[str, Any]]:
    """Order the artifacts that pin a chain member, parents before children.

    Membership follows BASE values as well as live ones: an artifact whose pin is
    stale no longer matches any live value, and dropping it from the chain would
    hide exactly the artifacts that still need re-pinning.

    It also follows discovered POINTER EDGES, which name their target
    explicitly.  Value-matching alone loses an artifact whose pin resolves to
    NOTHING -- neither a live nor a base value -- and that is precisely the worst
    case, not a benign one: `editing_training_gate.py` held a value produced and
    superseded inside a single re-pin pass, so it matched nothing, fell out of
    the chain, and its genuine staleness was reported only as a warning.
    """

    pins: dict[str, set[str]] = {}
    for literal in literals:
        owners = live.owners(literal.value) or base.owners(literal.value)
        for artifact, _role in owners:
            if artifact == literal.relative_path:
                continue
            pins.setdefault(literal.relative_path, set()).add(artifact)
    for edge in edges:
        if edge.target != edge.source:
            pins.setdefault(edge.source, set()).add(edge.target)
    for pin in identity_pins:
        pins.setdefault(pin.relative_path, set()).update(IDENTITY_ARTIFACTS)

    # The chain is the connected component of the seeds in the UNDIRECTED pin
    # graph: an artifact that pins a member is downstream of it and needs
    # re-pinning, and an artifact a member pins is an implementation the member
    # content-addresses.  Both are in scope; only one direction would silently
    # exempt the bound implementation sources.
    members = set(seeds)
    changed = True
    while changed:
        changed = False
        for source, targets in pins.items():
            if source in members and not targets <= members:
                members |= targets
                changed = True
            elif source not in members and targets & members:
                members.add(source)
                changed = True

    ordered: list[str] = []
    remaining = set(members)
    while remaining:
        ready = sorted(
            name for name in remaining if not (pins.get(name, set()) & remaining - {name})
        )
        if not ready:  # a cycle: emit deterministically rather than looping
            ready = [min(remaining)]
        for name in ready:
            ordered.append(name)
            remaining.discard(name)

    rows: list[dict[str, Any]] = []
    for name in ordered:
        rows.append(
            {
                "artifact": name,
                "live_values": dict(sorted(live.by_artifact.get(name, {}).items())),
                "pins": sorted(pins.get(name, set())),
            }
        )
    return rows


# ---- Entry point ----


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the Process-V2 identities and verify the complete transitive "
            "re-pin chain. Reads only; writes nothing."
        )
    )
    parser.add_argument(
        "--base-revision",
        default=PROCESS_V2_CORRECTION_BASE_REVISION,
        help="revision the current edits are measured against (default: the correction base)",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT, help="repository root (default: this checkout)"
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON on stdout")
    parser.add_argument(
        "--show-unclassified",
        action="store_true",
        help="list 64-hex literals that belong to no known artifact",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root: Path = args.repo_root.resolve()
    if not (root / V1_CONTRACT_RELATIVE_PATH).is_file():
        print(f"not a COMPOSE checkout: {root}", file=sys.stderr)
        return 2

    paths = _tracked_paths(root, SCAN_GLOBS)
    doc_paths = _tracked_paths(root, DOCUMENTATION_GLOBS)

    contract_buildable = True
    try:
        live_values, findings = _check_live_contract(root)
    except ContractUnbuildable as error:
        contract_buildable = False
        live_values, findings = {}, [
            Finding(
                "FAIL",
                "contract_unbuildable",
                V2_CONTRACT_RELATIVE_PATH,
                f"{error}; every contract-dependent check was SKIPPED, not passed",
                artifact=V2_CONTRACT_RELATIVE_PATH,
            )
        ]
    live = _live_value_index(root, paths)

    literals: list[Literal] = []
    edges: list[PointerEdge] = []
    for path in paths:
        relative_path = str(path.relative_to(root))
        raw = path.read_bytes()
        if relative_path.endswith(".json"):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as error:
                findings.append(
                    Finding("FAIL", "unparseable_config", relative_path, str(error))
                )
                continue
            literals.extend(_scan_json_literals(relative_path, payload))
            edges.extend(_pointer_edges(root, relative_path, payload))
        else:
            text = raw.decode(errors="replace")
            literals.extend(_scan_python_literals(relative_path, text))
            edges.extend(_python_pointer_edges(root, relative_path, text))

    doc_literals: list[Literal] = []
    for path in doc_paths:
        relative_path = str(path.relative_to(root))
        doc_literals.extend(
            _scan_text_literals(relative_path, path.read_text(errors="replace"))
        )

    base_available = True
    base = ValueIndex()
    changed: tuple[str, ...] = ()
    skip_reason = ""
    try:
        changed = _git_changed_paths(root, args.base_revision)
        base = _base_value_index(root, args.base_revision, paths)
    except GitUnavailable as error:
        base_available = False
        skip_reason = str(error)
        findings.append(
            Finding(
                "WARN",
                "base_revision_unavailable",
                args.base_revision,
                f"base-relative checks were SKIPPED, not passed: {error}",
            )
        )

    identity_pins = _identity_pins(tuple(literals))
    classification, counts = _classify_literals(
        tuple(literals), live, base, base_available
    )
    findings.extend(classification)
    findings.extend(_check_identity_pins(identity_pins, live))
    findings.extend(_check_pointer_edges(root, tuple(edges), live, base, base_available))
    obsolete: list[str] = []
    if base_available:
        sweep, obsolete = _check_obsoleted_sweep(
            root, args.base_revision, changed, tuple(literals), tuple(doc_literals), live
        )
        findings.extend(sweep)
        findings.extend(_check_v1_named_configs(root, args.base_revision, changed))
        findings.extend(_check_self_hashes(root, args.base_revision, paths, base_available))
    findings.extend(_check_v1_not_process_v2_authority(root, paths))
    if contract_buildable:
        findings.extend(_check_frozen_names(root))

    chain = _chain_report(
        tuple(literals),
        identity_pins,
        live,
        base,
        seeds=(V1_CONTRACT_RELATIVE_PATH, V2_CONTRACT_RELATIVE_PATH, *IDENTITY_ARTIFACTS),
        edges=tuple(edges),
    )
    members = frozenset(row["artifact"] for row in chain)
    findings = [finding.scoped(members) for finding in findings]
    failures = [finding for finding in findings if finding.severity == "FAIL"]
    report = {
        "base_revision": args.base_revision if base_available else None,
        "base_revision_skipped_reason": skip_reason or None,
        "changed_paths": list(changed),
        "chain": chain,
        "findings": [
            {
                "artifact": finding.artifact,
                "category": finding.category,
                "detail": finding.detail,
                "in_process_v2_chain": finding.artifact in members,
                "location": finding.location,
                "severity": finding.severity,
            }
            for finding in findings
        ],
        "literal_counts": counts,
        "live_values": live_values,
        "obsoleted_values": obsolete,
        "status": "AGREES" if not failures else "DISAGREES",
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _render(report, findings, live, args.show_unclassified, literals)
    return 0 if not failures else 1


def _render(
    report: dict[str, Any],
    findings: list[Finding],
    live: ValueIndex,
    show_unclassified: bool,
    literals: list[Literal],
) -> None:
    print("== live Process-V2 values ==")
    for key, value in sorted(report["live_values"].items()):
        print(f"  {key:42s} {value}")
    print()
    print(f"== transitive chain ({len(report['chain'])} artifacts, parents first) ==")
    for row in report["chain"]:
        roles = ", ".join(f"{role}={value[:12]}" for role, value in row["live_values"].items())
        print(f"  {row['artifact']}")
        if roles:
            print(f"      live: {roles}")
        if row["pins"]:
            print(f"      pins: {', '.join(row['pins'])}")
    print()
    if show_unclassified:
        print("== 64-hex literals belonging to no known artifact ==")
        for literal in literals:
            if not live.owners(literal.value):
                print(f"  {literal.relative_path}:{literal.location} {literal.value}")
        print()
    print(f"== findings ({len(findings)}) ==")
    if not findings:
        print("  none")
    for finding in findings:
        print(f"  {finding.render()}")
    print()
    counts = report["literal_counts"]
    print(
        "literals: {agrees} agree, {stale} stale, {lineage} lineage, "
        "{unclassified} unclassified".format(**counts)
    )
    print(f"status: {report['status']}")


if __name__ == "__main__":
    raise SystemExit(main())
