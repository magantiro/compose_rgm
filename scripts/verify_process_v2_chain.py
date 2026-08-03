#!/usr/bin/env python
"""Sound generic verifier for a declared Process-V2 pointer graph.

WHAT WAS UNSOUND BEFORE
-----------------------
``scripts/verify_process_v2_hash_chain.py`` can report ``AGREES`` while failing to
prove the graph it claims to have checked.  Three mechanisms, each fixed here:

1. **Identity edges fanned out to both nodes.**  It accepted *either* live process
   identity for every artifact and recorded each pin as addressing both, so a V2
   artifact carrying the V1 value passed.  This verifier resolves an identity edge
   **contextually** -- from the declared provider, identity schema, process
   semantics and value together -- to **exactly one** node.  A declaration that
   resolves to no node is a failure, and nothing ever resolves to two.

2. **A pointer existed only if its target did.**  Edges were discovered by
   scanning for strings that named an *existing* file, so a deleted or misspelled
   target simply dropped out of the graph and the missing-target diagnostic was
   unreachable.  This verifier discovers **declared typed pointers** structurally,
   through ``validate_typed_pointer``.  A declared pointer cannot vanish, so a
   missing repository target is a reportable failure.

3. **Warnings coexisted with agreement.**  An unverifiable semantic derivation was
   a warning and the run still printed ``AGREES``.  Here a required edge that
   cannot be decided yields ``INCONCLUSIVE``, which is not success.  Only a
   ``remote_artifact`` pointer may go unchecked, because its stage loader owns it,
   and every such deferral is listed by name rather than silently dropped.

WHAT IT CHECKS
--------------
Starting from declared root artifacts it walks every declared typed pointer:

* ``repository_config`` -- the target must exist.  A ``physical_sha256`` pointer is
  compared to the file's bytes; a ``semantic_sha256`` pointer is compared under the
  algorithm the pointer itself declares (``self_hash_field_v1`` or
  ``whole_canonical_body_v1``), so the verifier never has to guess.  An algorithm
  the target cannot satisfy is a failure, not a warning.
* ``remote_artifact`` -- deferred to its stage loader.  Never looked for on the
  local filesystem and therefore never misreported as a missing local file.
* ``lineage_reference`` -- deliberately historical.  Its value is required NOT to
  equal the target's live value: a "superseded" pin that equals what is live is a
  false claim, not lineage.
* ``external_asset`` -- verified physically when it resolves inside the checkout,
  deferred otherwise.

STATUS
------
``AGREES``       every required edge was decided and every one agreed.
``INCONCLUSIVE`` no disagreement found, but at least one required edge could not be
                 decided.  Not success.
``DISAGREES``    at least one proven disagreement.

Exit status is ``0`` for ``AGREES``, ``1`` for ``INCONCLUSIVE`` or ``DISAGREES``,
and ``2`` on a usage or I/O failure.  It writes nothing.

Usage::

    .venv/bin/python scripts/verify_process_v2_chain.py
    .venv/bin/python scripts/verify_process_v2_chain.py --root configs/other.json
    .venv/bin/python scripts/verify_process_v2_chain.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
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
    validate_typed_pointer,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (  # noqa: E402
    PROCESS_V2_CHAIN_ARTIFACTS,
    SELF_HASH_FIELD_ALGORITHM,
    WHOLE_CANONICAL_BODY_ALGORITHM,
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
"""A required edge the verifier could not decide. Blocks ``AGREES``."""

DEFERRED = "DEFERRED"
"""Delegated to a stage loader by the pointer's own declared kind."""

# The exact field set a declared process-identity edge carries. Discovery is
# structural: an object with exactly these keys is an identity edge.
_IDENTITY_EDGE_FIELDS = frozenset(
    {
        "identity_schema",
        "identity_schema_version",
        "module",
        "process_identity_sha256",
        "process_semantics",
        "provider",
    }
)


# ---- Identity nodes ----


@dataclass(frozen=True)
class IdentityNode:
    """One computed process identity, addressed by its declaration, not its value."""

    provider: str
    identity_schema: str
    process_semantics: str
    value: str

    @property
    def name(self) -> str:
        return f"identity:{self.provider}"

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.provider, self.identity_schema, self.process_semantics)


def live_identity_nodes() -> tuple[IdentityNode, ...]:
    """The identity nodes this checkout currently computes.

    Each node is keyed by ``(provider, identity schema, process semantics)``.  The
    two Process identities differ in all three, so a declaration naming one can
    never select the other, whatever its value happens to be.
    """

    nodes = []
    for provider, identity in (
        ("editing_v2_process_identity", editing_v2_process_identity()),
        ("editing_process_v2_identity", editing_process_v2_identity()),
    ):
        nodes.append(
            IdentityNode(
                provider=provider,
                identity_schema=str(identity["schema"]),
                process_semantics=str(identity["process_semantics"]),
                value=str(identity["process_identity_sha256"]),
            )
        )
    return tuple(nodes)


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

    def add(self, severity: str, category: str, location: str, detail: str) -> None:
        self.findings.append(Finding(severity, category, location, detail))

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
            "status": self.status,
        }


# ---- Reading ----


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
        and isinstance(value, str)
        and len(value) == 64
        and value == canonical_sha256({k: v for k, v in payload.items() if k != key})
    }


# ---- Edge checking ----


def _check_repository_pointer(
    pointer: Mapping[str, Any], *, location: str, repo_root: Path, report: Report
) -> str:
    """Check one repository pointer and return its result token."""

    target_path = repo_root / str(pointer["target"])
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

    if pointer["identity_role"] == IdentityRole.PHYSICAL:
        observed = _sha256_bytes(raw)
        if observed == pointer["sha256"]:
            return "agrees"
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
        return "agrees"
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

    target_path = repo_root / str(pointer["target"])
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
    pointer: Mapping[str, Any], *, location: str, repo_root: Path, report: Report
) -> str:
    kind = pointer["kind"]
    if kind == PointerKind.REPOSITORY_CONFIG:
        return _check_repository_pointer(
            pointer, location=location, repo_root=repo_root, report=report
        )
    if kind == PointerKind.LINEAGE_REFERENCE:
        return _check_lineage_pointer(
            pointer, location=location, repo_root=repo_root, report=report
        )
    if kind == PointerKind.REMOTE_ARTIFACT:
        report.add(
            DEFERRED,
            "remote_artifact_deferred_to_stage_loader",
            location,
            f"pins the volume-relative artifact {pointer['target']} at "
            f"{pointer['sha256']}; its stage loader verifies it, and it is "
            "deliberately not looked for on the local filesystem",
        )
        return "deferred"
    # external_asset
    if (repo_root / str(pointer["target"])).is_file():
        return _check_repository_pointer(
            pointer, location=location, repo_root=repo_root, report=report
        )
    report.add(
        DEFERRED,
        "external_asset_outside_the_checkout",
        location,
        f"pins the external asset {pointer['target']} at {pointer['sha256']}, which "
        "does not resolve inside this checkout",
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
    """Resolve one declared identity edge to exactly one node, then compare."""

    declared = (
        str(edge.get("provider")),
        str(edge.get("identity_schema")),
        str(edge.get("process_semantics")),
    )
    resolved = [node for node in nodes if node.key == declared]
    row: dict[str, Any] = {
        "declared": {
            "identity_schema": declared[1],
            "process_semantics": declared[2],
            "provider": declared[0],
        },
        "location": location,
        "resolved_nodes": [node.name for node in resolved],
        "sha256": str(edge.get("process_identity_sha256")),
    }
    if not resolved:
        report.add(
            FAIL,
            "unclassified_identity_edge",
            location,
            f"declares provider {declared[0]!r}, schema {declared[1]!r} and process "
            f"semantics {declared[2]!r}, which name no identity this checkout "
            f"computes ({[node.key for node in nodes]}). An unclassified identity "
            "edge resolves to no node; it must never fan out to every candidate",
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


# ---- Traversal ----


_Located = list[tuple[str, Mapping[str, Any]]]


def _discover(payload: object) -> tuple[_Located, _Located]:
    """Return the declared typed pointers and identity edges inside one artifact.

    Discovery is structural.  A mapping is a pointer if and only if it validates
    as one, and an identity edge if and only if it carries exactly the identity
    edge field set.  Nothing is inferred from a string that happens to look like a
    path, which is what let a misspelled target disappear from the old graph.
    """

    pointers: list[tuple[str, Mapping[str, Any]]] = []
    identities: list[tuple[str, Mapping[str, Any]]] = []
    malformed: list[tuple[str, str]] = []
    for location, node in _walk(payload):
        if not isinstance(node, Mapping):
            continue
        if frozenset(node) == _IDENTITY_EDGE_FIELDS:
            identities.append((location, node))
            continue
        if frozenset(node) != POINTER_FIELDS:
            continue
        try:
            pointers.append((location, validate_typed_pointer(node, label=location)))
        except ProcessV2SchemaError as error:
            # A node carrying the pointer key set IS a declared pointer. Skipping
            # one that fails validation drops it from the graph exactly as a
            # missing target used to be dropped, so its diagnostic becomes
            # unreachable and a mislabelled role degrades into a generic stale
            # pointer somewhere else. It fails here instead.
            malformed.append((location, str(error)))
    return pointers, identities, malformed


def verify_process_v2_chain(
    *, repo_root: Path, roots: Sequence[str] = PROCESS_V2_CHAIN_ARTIFACTS
) -> dict[str, Any]:
    """Verify the declared pointer graph reachable from ``roots``.

    Returns the report. Its ``status`` is ``AGREES`` only when every required edge
    was decided and agreed; ``INCONCLUSIVE`` when something required could not be
    decided; ``DISAGREES`` on any proven disagreement.
    """

    repo_root = Path(repo_root)
    nodes = live_identity_nodes()
    report = Report()
    seen: set[str] = set()
    queue: list[str] = list(roots)

    while queue:
        relative_path = queue.pop(0)
        if relative_path in seen:
            continue
        seen.add(relative_path)
        report.artifacts.append(relative_path)
        path = repo_root / relative_path
        if not path.is_file():
            report.add(
                FAIL,
                "missing_graph_artifact",
                relative_path,
                "is declared as a member of the graph but is not present",
            )
            continue
        if not relative_path.endswith(".json"):
            continue
        try:
            payload = json.loads(path.read_bytes())
        except json.JSONDecodeError as error:
            report.add(FAIL, "unparseable_artifact", relative_path, str(error))
            continue

        pointers, identities, malformed = _discover(payload)
        for location, detail in malformed:
            report.add(
                FAIL,
                "malformed_declared_pointer",
                location,
                f"carries the declared pointer key set but is not a valid pointer: "
                f"{detail}",
            )
        for location, pointer in pointers:
            where = f"{relative_path}:{location}"
            result = _check_pointer(
                pointer, location=where, repo_root=repo_root, report=report
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
                and pointer["target"] not in seen
                and pointer["target"] not in queue
            ):
                queue.append(str(pointer["target"]))
        for location, edge in identities:
            report.identity_edges.append(
                _check_identity_edge(
                    edge,
                    location=f"{relative_path}:{location}",
                    nodes=nodes,
                    report=report,
                )
            )

    if not report.identity_edges:
        report.add(
            UNVERIFIED,
            "no_identity_edge_declared",
            ",".join(roots),
            "the graph declares no process-identity edge, so nothing proves which "
            "process it describes",
        )
    return report.as_dict()


# ---- Entry point ----


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a declared Process-V2 pointer graph. Resolves every identity edge "
            "to exactly one node, checks every declared typed pointer, and refuses to "
            "report agreement when a required edge could not be decided. Reads only."
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
    parser.add_argument("--json", action="store_true", help="emit the report as JSON on stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root: Path = args.repo_root.resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    roots = tuple(args.roots) if args.roots else PROCESS_V2_CHAIN_ARTIFACTS
    report = verify_process_v2_chain(repo_root=root, roots=roots)
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
    print(f"== identity edges ({len(report['identity_edges'])}) ==")
    for row in report["identity_edges"]:
        resolved = ", ".join(row["resolved_nodes"]) or "<none>"
        print(f"  {row['location']}")
        print(f"      declares {row['declared']['provider']} -> {resolved} [{row['result']}]")
    print()
    print(f"== declared pointers ({len(report['edges'])}) ==")
    for edge in report["edges"]:
        print(
            f"  {edge['result']:22s} {edge['kind']:18s} {edge['identity_role']:22s} "
            f"{edge['target']}"
        )
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
