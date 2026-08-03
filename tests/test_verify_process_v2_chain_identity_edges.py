"""A required chain artifact's identity edge must not be able to vanish.

``scripts/verify_process_v2_chain.py`` claims that every required artifact declares
exactly one Process-V2 identity edge, resolved contextually to exactly one node.
This file attacks that claim one artifact at a time, leaving the other six valid,
and drives the production entry point ``verify_process_v2_chain`` over a mutated
copy of the real graph.  Nothing here reimplements a check.

**The defect these tests were written against.**  Identity edges are discovered by
an exact field-set match, and cardinality was asserted once per *graph*
(``if not report.identity_edges``).  So an artifact whose edge was deleted, or
whose edge gained or lost a single key, stopped being discovered and its absence
was never reported: the other six artifacts' edges satisfied the graph-global
guard.  Measured on the committed graph, **28 of 84** attacks (4 attack kinds x 7
artifacts) produced no identity-edge diagnostic at all.

**Why every test asserts on the finding and not merely on the status.**  Six of the
seven required artifacts are *pinned by another artifact*, so mutating one restales
a child's physical and semantic pointers and the run reports ``DISAGREES`` for a
reason that has nothing to do with identity.  A test asserting only "not
``AGREES``" would therefore pass for those six even with the identity rule deleted.
Only ``P50_RECIPE_POLICY``, the graph leaf nothing pins, exposed the defect as a
clean ``AGREES``.  Every test below names the category it expects and requires the
finding to be located at the offending artifact.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    P50_RECIPE_POLICY,
    PROCESS_IDENTITY_EDGE_FIELDS,
    PROCESS_IDENTITY_PIN_FIELD,
    PROCESS_V2_CHAIN_ARTIFACTS,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    REJECTED_PROCESS_V2_CANDIDATE_IDENTITY_SHA256,
    SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
    editing_v2_process_identity,
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

# The mutation helpers are REUSED from the sibling suite rather than rewritten:
# ``_reseal`` seals through the production rule, so a mutation cannot fail for the
# wrong reason by also breaking the artifact's self-hash.
from test_verify_process_v2_chain import _graph_copy, _reseal  # noqa: E402
from verify_process_v2_chain import (  # noqa: E402
    AGREES,
    DISAGREES,
    FAIL,
    INCONCLUSIVE,
    UNVERIFIED,
    main,
    verify_process_v2_chain,
)

#: The top-level key each chain artifact seals its identity edge under.  A fact
#: about the artifacts, read back below from the committed graph itself.
_EDGE_KEY = "process_identity"

#: Reachable from the seven roots but not roots themselves.  These frozen parents
#: declare no identity edge, which is why the cardinality rule is scoped to the
#: REQUIRED artifacts rather than to everything the walk reaches.
_NON_ROOT_MEMBERS: tuple[str, ...] = (
    "configs/editing_gate_zero_semantic_model_process_v2.json",
    "configs/editing_v2_semantic_process_v2.json",
    "configs/editing_corpus_v2_contract.json",
    "src/compose_v4/data/editing_v2_semantic_capability_cells.py",
)


# ---- Helpers ----


def _report_after(artifact: str, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Mutate one artifact of a throwaway copy of the real graph, then verify it."""

    with _graph_copy() as root:
        _reseal(root, artifact, mutate)
        return verify_process_v2_chain(repo_root=root)


def _findings_at(report: dict[str, Any], *, category: str, artifact: str) -> list[dict[str, Any]]:
    """Findings of one category whose location names ``artifact``.

    Locations are either the artifact path itself (an artifact-level finding) or
    ``"<artifact>:<json path>"`` (an edge-level one).
    """

    return [
        finding
        for finding in report["findings"]
        if finding["category"] == category
        and (
            finding["location"] == artifact or finding["location"].startswith(f"{artifact}:")
        )
    ]


def _edges_of(report: dict[str, Any], artifact: str) -> list[dict[str, Any]]:
    return [
        row for row in report["identity_edges"] if row["location"].startswith(f"{artifact}:")
    ]


def _set_value(new: object) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        payload[_EDGE_KEY][PROCESS_IDENTITY_PIN_FIELD] = new

    return mutate


def _set_field(name: str, new: str) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        payload[_EDGE_KEY][name] = new

    return mutate


# ---- The committed graph, counted rather than set-collapsed ----


def test_the_committed_graph_declares_exactly_one_identity_edge_per_required_artifact() -> None:
    """Without this, a duplicate edge on the committed graph goes unnoticed.

    The sibling suite collapses edge locations into a ``set`` before comparing them
    to the required artifacts, and a set cannot count: an artifact declaring the
    same edge twice still contributes one member. This counts.
    """

    report = verify_process_v2_chain(repo_root=_ROOT)
    per_artifact = Counter(row["location"].split(":", 1)[0] for row in report["identity_edges"])
    assert per_artifact == Counter(PROCESS_V2_CHAIN_ARTIFACTS)
    assert len(report["identity_edges"]) == len(PROCESS_V2_CHAIN_ARTIFACTS)
    assert report["status"] == AGREES, report["findings"]


def test_the_identity_edge_key_and_field_set_are_read_from_the_committed_artifacts() -> None:
    """Pins the fixture's two assumptions against production, not against itself.

    Every mutation below addresses the edge through ``_EDGE_KEY`` and drops or adds
    keys relative to ``PROCESS_IDENTITY_EDGE_FIELDS``. If either drifted from what
    the artifacts actually carry, the mutations would silently stop touching the
    identity edge and every attack test would pass while attacking nothing.
    """

    for artifact in PROCESS_V2_CHAIN_ARTIFACTS:
        payload = json.loads((_ROOT / artifact).read_bytes())
        assert _EDGE_KEY in payload, artifact
        assert set(payload[_EDGE_KEY]) == set(PROCESS_IDENTITY_EDGE_FIELDS), artifact
        assert PROCESS_IDENTITY_PIN_FIELD in payload[_EDGE_KEY], artifact


def test_a_reachable_non_root_artifact_is_not_required_to_declare_an_identity_edge() -> None:
    """The cardinality rule is scoped to the REQUIRED artifacts, and must stay so.

    Four frozen parents are reachable from the roots and legitimately declare no
    process identity. Applying the rule to everything the walk reaches would turn
    the committed graph ``INCONCLUSIVE``; this test fails the moment that scope
    widens.
    """

    report = verify_process_v2_chain(repo_root=_ROOT)
    for member in _NON_ROOT_MEMBERS:
        assert member in report["artifacts"]
        assert member not in set(PROCESS_V2_CHAIN_ARTIFACTS)
        assert _edges_of(report, member) == []
    assert report["status"] == AGREES, report["findings"]


# ---- (1) missing: the edge removed entirely ----


@pytest.mark.parametrize("artifact", PROCESS_V2_CHAIN_ARTIFACTS)
def test_a_deleted_identity_edge_is_reported_against_its_own_artifact(artifact: str) -> None:
    """Deleting one artifact's edge left six valid ones, and the graph still agreed.

    Without this test the graph-global "at least one identity edge declared" guard
    can come back, and a required artifact that proves nothing about which process
    it describes passes because its neighbours do.
    """

    report = _report_after(artifact, lambda payload: payload.pop(_EDGE_KEY))
    assert report["status"] != AGREES
    assert _edges_of(report, artifact) == []
    named = _findings_at(report, category="no_identity_edge_declared", artifact=artifact)
    assert named, report["findings"]
    assert all(finding["severity"] == UNVERIFIED for finding in named)


# ---- (2) malformed: a broken but present declaration ----


@pytest.mark.parametrize("dropped", PROCESS_IDENTITY_EDGE_FIELDS)
@pytest.mark.parametrize("artifact", PROCESS_V2_CHAIN_ARTIFACTS)
def test_an_identity_edge_that_loses_a_declared_field_is_reported_as_absent(
    artifact: str, dropped: str
) -> None:
    """Discovery is an exact field-set match, so a short edge is not malformed, it is gone.

    This is the same "a declared thing must not vanish" class the typed pointers
    were introduced to close, reached through the identity edge instead. Removing
    ANY one of the six declared keys must still be reported against this artifact.
    """

    report = _report_after(artifact, lambda payload: payload[_EDGE_KEY].pop(dropped))
    assert report["status"] != AGREES
    assert _edges_of(report, artifact) == []
    assert _findings_at(report, category="no_identity_edge_declared", artifact=artifact), (
        report["findings"]
    )


@pytest.mark.parametrize(
    "value",
    ["z" * 64, "0" * 63, "0" * 65, "", None, 12345, {}],
    ids=["non_hex", "too_short", "too_long", "empty", "null", "int", "object"],
)
@pytest.mark.parametrize("artifact", PROCESS_V2_CHAIN_ARTIFACTS)
def test_an_unusable_identity_value_fails_against_its_artifact(
    artifact: str, value: object
) -> None:
    """A value that is not a SHA-256 at all must not be quietly compared and dropped.

    Nothing else in the verifier type-checks an identity value -- unlike a typed
    pointer, an identity edge is not run through ``validate_typed_pointer`` -- so if
    this comparison were ever relaxed, a garbage identity would go unreported. The
    non-string cases matter separately: an uncaught ``TypeError`` would abort the
    run, and an aborted run reports neither agreement nor a finding.
    """

    report = _report_after(artifact, _set_value(value))
    assert report["status"] == DISAGREES
    named = _findings_at(report, category="stale_identity_pin", artifact=artifact)
    assert named, report["findings"]
    assert all(finding["severity"] == FAIL for finding in named)


@pytest.mark.parametrize(
    "value",
    [
        str(editing_v2_process_identity()[PROCESS_IDENTITY_PIN_FIELD]),
        SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
        REJECTED_PROCESS_V2_CANDIDATE_IDENTITY_SHA256,
    ],
    ids=["live_v1", "superseded_v1", "rejected_candidate"],
)
@pytest.mark.parametrize("artifact", PROCESS_V2_CHAIN_ARTIFACTS)
def test_a_v1_or_rejected_identity_value_fails_against_its_artifact(
    artifact: str, value: str
) -> None:
    """The three values that must never be readable as a current V2 identity.

    The predecessor accepted the live V1 value in a V2 artifact because it matched
    *either* live identity. The superseded V1 value and the rejected pre-run
    candidate are real hashes that exist in this repository as lineage, so a
    copy-paste from the lineage block into an identity edge is a plausible mistake
    that must not resolve.
    """

    report = _report_after(artifact, _set_value(value))
    assert report["status"] == DISAGREES
    named = _findings_at(report, category="stale_identity_pin", artifact=artifact)
    assert named, report["findings"]
    assert value in named[0]["detail"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "editing_v2_process_identity"),
        ("identity_schema", "compose.editing.semantic_process_identity"),
        ("process_semantics", "semantic_editing_v2_v1"),
    ],
    ids=["provider", "identity_schema", "process_semantics"],
)
@pytest.mark.parametrize("artifact", PROCESS_V2_CHAIN_ARTIFACTS)
def test_a_mislabelled_identity_declaration_resolves_to_no_node(
    artifact: str, field: str, value: str
) -> None:
    """Each of the three keys is load-bearing on its own, and none may fan out.

    The declaration is resolved from provider, identity schema and process
    semantics TOGETHER. Corrupting any one of them alone -- here to the V1
    counterpart, so the edge still names something real -- must resolve to no node
    rather than to the node the other two keys suggest.
    """

    report = _report_after(artifact, _set_field(field, value))
    assert report["status"] == DISAGREES
    named = _findings_at(report, category="unclassified_identity_edge", artifact=artifact)
    assert named, report["findings"]
    rows = _edges_of(report, artifact)
    assert [row["resolved_nodes"] for row in rows] == [[]]
    assert [row["result"] for row in rows] == ["unclassified"]


# ---- (3) extra field: the edge carrying a key beyond the declared set ----


@pytest.mark.parametrize("artifact", PROCESS_V2_CHAIN_ARTIFACTS)
def test_an_identity_edge_that_gains_a_field_is_reported_as_absent(artifact: str) -> None:
    """The most likely surviving hole, and it survived: an added key hid the edge.

    An exact field-set match is a discovery rule, not a validation rule, so an edge
    with one extra key is not reported as malformed -- it stops being an edge. A
    schema addition made in good faith would silently unpin the artifact's process.
    """

    report = _report_after(
        artifact, lambda payload: payload[_EDGE_KEY].update(note="a key beyond the declared set")
    )
    assert report["status"] != AGREES
    assert _edges_of(report, artifact) == []
    assert _findings_at(report, category="no_identity_edge_declared", artifact=artifact), (
        report["findings"]
    )


# ---- (4) two edges: more than one declaration in one artifact ----


@pytest.mark.parametrize("artifact", PROCESS_V2_CHAIN_ARTIFACTS)
def test_two_identity_edges_in_one_artifact_fail_against_that_artifact(artifact: str) -> None:
    """Two declarations leave "which process is this?" unanswerable from the artifact.

    Both copies resolve and both agree, so every per-edge check passes and the graph
    agreed. Cardinality is the only thing that catches it, and it must be counted
    per artifact.
    """

    def duplicate(payload: dict[str, Any]) -> None:
        payload["a_second_process_identity"] = dict(payload[_EDGE_KEY])

    report = _report_after(artifact, duplicate)
    assert report["status"] == DISAGREES
    assert len(_edges_of(report, artifact)) == 2
    named = _findings_at(report, category="multiple_identity_edges_declared", artifact=artifact)
    assert named, report["findings"]
    assert all(finding["severity"] == FAIL for finding in named)


@pytest.mark.parametrize("artifact", PROCESS_V2_CHAIN_ARTIFACTS)
def test_a_second_identity_edge_naming_a_different_process_also_fails(artifact: str) -> None:
    """A second edge that RESOLVES to the other live node is the dangerous shape.

    It is a genuine, self-consistent declaration -- the V1 process, correctly
    labelled and correctly valued -- so no per-edge check can object to it. What is
    wrong is that the artifact now claims two processes at once.
    """

    v1 = editing_v2_process_identity()

    def add_v1_edge(payload: dict[str, Any]) -> None:
        payload["a_second_process_identity"] = {
            "identity_schema": str(v1["schema"]),
            "identity_schema_version": int(v1["schema_version"]),
            "module": payload[_EDGE_KEY]["module"],
            PROCESS_IDENTITY_PIN_FIELD: str(v1[PROCESS_IDENTITY_PIN_FIELD]),
            "process_semantics": str(v1["process_semantics"]),
            "provider": "editing_v2_process_identity",
        }

    report = _report_after(artifact, add_v1_edge)
    assert report["status"] == DISAGREES
    assert {row["result"] for row in _edges_of(report, artifact)} == {"agrees"}
    assert _findings_at(
        report, category="multiple_identity_edges_declared", artifact=artifact
    ), report["findings"]


# ---- The masking that makes a status-only assertion vacuous ----


def test_at_the_graph_leaf_a_missing_identity_edge_is_the_only_finding() -> None:
    """Proves the identity rule alone does the work, with no collateral help.

    Six of the seven required artifacts are pinned by another artifact, so mutating
    them stales a child's hash pointer and the run ``DISAGREES`` for an unrelated
    reason -- a test asserting only "not ``AGREES``" would pass for them even with
    the identity rule removed. ``P50_RECIPE_POLICY`` is the leaf nothing pins: here
    the identity finding is the ONLY finding, so this test fails if the rule is
    weakened, and it also fails if the leaf ever acquires a child.
    """

    report = _report_after(P50_RECIPE_POLICY, lambda payload: payload.pop(_EDGE_KEY))
    assert [finding["category"] for finding in report["findings"]] == [
        "no_identity_edge_declared"
    ]
    assert report["findings"][0]["location"] == P50_RECIPE_POLICY
    assert report["status"] == INCONCLUSIVE


def test_a_missing_identity_edge_is_not_a_successful_exit_code() -> None:
    """``INCONCLUSIVE`` must not exit zero, or a launch gate reads it as agreement."""

    with _graph_copy() as root:
        _reseal(root, P50_RECIPE_POLICY, lambda payload: payload.pop(_EDGE_KEY))
        assert main(["--repo-root", str(root)]) == 1
