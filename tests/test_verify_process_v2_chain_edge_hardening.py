"""A declaration must not be able to vanish, agree by coincidence, or defer forever.

Four independent holes in ``scripts/verify_process_v2_chain.py``'s traversal, each
of which let a run report agreement it had not proved:

1. **Discovery by exact field-set match is not validation.**  A declared typed
   pointer or a declared identity edge that GAINED, LOST or MISSPELLED one key
   stopped being discovered.  It did not become malformed; it became *absent* --
   the same "a declared thing must not vanish" defect typed pointers were
   introduced to close, reached one level up.
2. **Identity edges were matched on three naming fields, not five.**  A digest and
   a role are not a name: the implementing MODULE and the identity SCHEMA VERSION
   were ignored, so a declaration naming a module that no longer computes the
   identity, or a schema revision that no longer exists, still resolved.
3. **A repository target was joined to the checkout and read.**  An absolute
   target, a ``..`` component, or a symlink leaving the tree makes a pin address a
   file the checkout does not govern -- and then hash, agree, and read as checked.
4. **Deferral was indistinguishable from agreement.**  ``remote_artifact`` and
   out-of-checkout ``external_asset`` pins printed a note and the run still said
   ``AGREES``.  "A stage loader owns it" names who checks it, not that anyone did.

Plus the check that no hash comparison can make: a pointer's declared
``target_schema`` against the target's own ``schema``.  Both sides of a hash move
together, so a pin that binds the right bytes under the wrong contract is invisible
to every hash in the graph.

Nothing here hard-codes a contract hash or a contract schema version; every value a
test needs is read from the live artifact it is about.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data.editing_v2_process_v2_schema import (
    POINTER_FIELDS,
    IdentityRole,
    PointerKind,
    authority_false_block,
    self_hashed,
    typed_pointer,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    CAPABILITY_CELLS,
    EDITING_CORPUS_V2_CONTRACT,
    P50_RECIPE_POLICY,
    PROCESS_IDENTITY_EDGE_FIELDS,
    SEMANTIC_PROCESS_V2,
    T1_CAPACITY_POLICY,
    T1_PANEL_POLICY,
    write_process_v2_chain,
)
from compose_v4.rewrite.editing_v2_process_identity import editing_process_v2_identity

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from test_verify_process_v2_chain import _graph_copy, _reseal  # noqa: E402
from verify_process_v2_chain import (  # noqa: E402
    AGREES,
    DEFERRED,
    DISAGREES,
    FAIL,
    INCONCLUSIVE,
    STAGE_RECEIPT_ENTRY_FIELDS,
    STAGE_RECEIPT_SCHEMA,
    STAGE_RECEIPT_SCHEMA_VERSION,
    STAGE_RECEIPT_SELF_HASH_FIELD,
    STAGE_RECEIPT_STATUS,
    UNVERIFIED,
    _NEAR_MISS_KEY_EDITS,
    live_identity_nodes,
    normalized_repository_path,
    verify_process_v2_chain,
)

_EDGE_KEY = "process_identity"
_REMOTE = "/editing_v2/semantic_v4_migration/fixture/SEMANTIC_MIGRATION_COMPLETE.json"

#: The stage whose receipt the fixtures below write.  A resolution entry is
#: attributed to the stage that carries it, so the two are one value here.
_STAGE = "semantic_v4_migration"


def _a_parent_role(
    artifact: str, *, slots: set[str], target_schema_declared: bool | None = None
) -> str:
    """A parent role of ``artifact`` with the shape a test needs, chosen structurally.

    Naming a role would couple these tests to the dependency graph, which the chain
    owner revises. What each test actually needs is a SHAPE: a role carrying both
    pointer slots, or one whose target declares a schema of its own, or one whose
    target is not JSON and therefore declares none.
    """

    parents = json.loads((_ROOT / artifact).read_bytes())["parents"]
    found = sorted(
        role
        for role, edge in parents.items()
        if set(edge) == slots
        and (
            target_schema_declared is None
            or (edge["physical"]["target_schema"] is not None) is target_schema_declared
        )
    )
    assert found, (artifact, slots, target_schema_declared)
    return found[0]


#: Both pointer slots, so a mutation exercises the physical and the semantic path.
_T1_PANEL_ROLE = _a_parent_role(T1_PANEL_POLICY, slots={"physical", "semantic"})
#: The same, on the graph leaf, whose target declares a schema of its own.
_P50_JSON_ROLE = _a_parent_role(
    P50_RECIPE_POLICY, slots={"physical", "semantic"}, target_schema_declared=True
)
#: A physical-only role whose target has no separable schema at all.
_CELLS_UNSCHEMED_ROLE = _a_parent_role(
    CAPABILITY_CELLS, slots={"physical"}, target_schema_declared=False
)


# ---- Helpers ----


def _categories(report: dict[str, Any], severity: str | None = None) -> list[str]:
    return [
        finding["category"]
        for finding in report["findings"]
        if severity is None or finding["severity"] == severity
    ]


def _findings(report: dict[str, Any], category: str) -> list[dict[str, Any]]:
    return [finding for finding in report["findings"] if finding["category"] == category]


def _edges_to(report: dict[str, Any], target: str) -> list[dict[str, Any]]:
    return [edge for edge in report["edges"] if edge["target"] == target]


def _report_after(artifact: str, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    with _graph_copy() as root:
        _reseal(root, artifact, mutate)
        return verify_process_v2_chain(repo_root=root)


def _pointer_of(payload: dict[str, Any], role: str, slot: str) -> dict[str, Any]:
    return payload["parents"][role][slot]


def _resolution(
    kind: str, target: str, sha256: str, *, verified_by: str = _STAGE
) -> dict[str, str]:
    """One resolution entry, attributed by default to the stage that carries it."""

    return {"kind": kind, "target": target, "sha256": sha256, "verified_by": verified_by}


def _stage_receipt(
    root: Path,
    name: str,
    entries: list[dict[str, Any]],
    *,
    schema: str = STAGE_RECEIPT_SCHEMA,
    schema_version: int = STAGE_RECEIPT_SCHEMA_VERSION,
    stage: str = _STAGE,
    status: str = STAGE_RECEIPT_STATUS,
    authority: dict[str, bool] | None = None,
    extra: dict[str, Any] | None = None,
    omit: tuple[str, ...] = (),
    corrupt_self_hash: bool = False,
    outside: Path | None = None,
) -> str:
    """Write one stage receipt, sealed through the production self-hash rule.

    ``outside`` writes it beyond the checkout and returns its ABSOLUTE path, which
    is the shape a containment rule has to refuse.
    """

    body: dict[str, Any] = {
        "resolved": entries,
        "schema": schema,
        "schema_version": schema_version,
        "stage": stage,
        "status": status,
        **(authority if authority is not None else authority_false_block()),
        **(extra or {}),
    }
    for dropped in omit:
        body.pop(dropped, None)
    payload = self_hashed(body, field=STAGE_RECEIPT_SELF_HASH_FIELD)
    if corrupt_self_hash:
        payload[STAGE_RECEIPT_SELF_HASH_FIELD] = "0" * 64
    path = (outside / name) if outside is not None else (root / name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    )
    return str(path) if outside is not None else name


def _inject_into_the_frozen_parent(root: Path, pointer: dict[str, Any], key: str) -> None:
    """Add a declared pointer to a frozen parent, then rebuild the chain over it.

    Injecting into a ROOT would be refused by that root's owning validator, which
    is correct but masks the edge-level behaviour under test. The corpus contract
    is a frozen parent, so rebuilding the chain re-pins it and every other check in
    the run agrees again -- leaving the injected edge as the only thing left to
    decide.
    """

    path = root / EDITING_CORPUS_V2_CONTRACT
    payload = json.loads(path.read_bytes())
    payload[key] = pointer
    path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    )
    write_process_v2_chain(root)


# ---- (1) a near miss of a declared shape must fail, not disappear ----


def test_the_committed_graph_declares_no_near_miss_of_either_shape() -> None:
    report = verify_process_v2_chain(repo_root=_ROOT)
    assert "near_miss_declared_pointer" not in _categories(report)
    assert "near_miss_declared_identity_edge" not in _categories(report)
    assert report["status"] == AGREES, report["findings"]


@pytest.mark.parametrize("dropped", sorted(POINTER_FIELDS))
def test_a_declared_pointer_that_loses_a_field_fails_rather_than_vanishing(
    dropped: str,
) -> None:
    report = _report_after(
        T1_PANEL_POLICY,
        lambda payload: _pointer_of(payload, _T1_PANEL_ROLE, "physical").pop(dropped),
    )
    named = _findings(report, "near_miss_declared_pointer")
    assert named, report["findings"]
    assert named[0]["severity"] == FAIL
    assert named[0]["location"].endswith(f".parents.{_T1_PANEL_ROLE}.physical")
    assert dropped in named[0]["detail"]


@pytest.mark.parametrize("renamed", sorted(POINTER_FIELDS))
def test_a_misspelled_pointer_field_fails_rather_than_vanishing(renamed: str) -> None:
    """One rename is one deletion plus one insertion, the widest near miss allowed."""

    def misspell(payload: dict[str, Any]) -> None:
        pointer = _pointer_of(payload, _T1_PANEL_ROLE, "physical")
        pointer[f"{renamed}_"] = pointer.pop(renamed)

    report = _report_after(T1_PANEL_POLICY, misspell)
    named = _findings(report, "near_miss_declared_pointer")
    assert named, report["findings"]
    assert f"{renamed}_" in named[0]["detail"]


def test_a_declared_pointer_that_gains_a_field_fails_rather_than_vanishing() -> None:
    report = _report_after(
        T1_PANEL_POLICY,
        lambda payload: _pointer_of(payload, _T1_PANEL_ROLE, "physical").update(
            note="beyond the field set"
        ),
    )
    named = _findings(report, "near_miss_declared_pointer")
    assert named, report["findings"]
    assert "note" in named[0]["detail"]


@pytest.mark.parametrize("dropped", sorted(PROCESS_IDENTITY_EDGE_FIELDS))
def test_an_identity_edge_that_loses_a_field_also_fails_as_a_near_miss(dropped: str) -> None:
    """The cardinality rule already reports the ABSENCE; this reports the CAUSE.

    ``no_identity_edge_declared`` is scoped to the required roots, so a near-miss
    identity edge anywhere else would still vanish without a word. Both findings
    are wanted: one says the artifact proves no process, the other says which
    declaration was one key away from doing so.
    """

    report = _report_after(T1_CAPACITY_POLICY, lambda payload: payload[_EDGE_KEY].pop(dropped))
    named = _findings(report, "near_miss_declared_identity_edge")
    assert named, report["findings"]
    assert named[0]["severity"] == FAIL
    assert named[0]["location"] == f"{T1_CAPACITY_POLICY}:.{_EDGE_KEY}"
    assert dropped in named[0]["detail"]
    assert "no_identity_edge_declared" in _categories(report, UNVERIFIED)


def test_an_identity_edge_that_gains_a_field_also_fails_as_a_near_miss() -> None:
    report = _report_after(
        T1_CAPACITY_POLICY, lambda payload: payload[_EDGE_KEY].update(note="beyond the set")
    )
    named = _findings(report, "near_miss_declared_identity_edge")
    assert named, report["findings"]
    assert "note" in named[0]["detail"]


def test_the_frozen_provider_description_is_measurably_outside_the_near_miss_bound() -> None:
    """The bound is not free, so the margin it rests on is asserted, not assumed.

    ``configs/editing_v2_semantic_process_v2.json:.process_identity`` describes the
    identity PROVIDER and deliberately pins no value. It is the closest thing in the
    committed graph to an identity edge that is not one, and it sits at three key
    edits. Widening ``_NEAR_MISS_KEY_EDITS`` to three would make that frozen
    artifact fail; this test fails first, and it fails again if some future artifact
    moves into the gap.
    """

    payload = json.loads((_ROOT / SEMANTIC_PROCESS_V2).read_bytes())
    observed = frozenset(payload[_EDGE_KEY])
    distance = len(observed ^ frozenset(PROCESS_IDENTITY_EDGE_FIELDS))
    assert distance > _NEAR_MISS_KEY_EDITS, (
        f"{SEMANTIC_PROCESS_V2}:.{_EDGE_KEY} is now {distance} key edits from a "
        "declared identity edge, which the near-miss rule would refuse"
    )
    report = verify_process_v2_chain(repo_root=_ROOT)
    assert "near_miss_declared_identity_edge" not in _categories(report)


# ---- (2) identity matching uses all five naming fields ----


def test_the_live_identity_nodes_are_keyed_on_five_naming_fields() -> None:
    nodes = live_identity_nodes()
    assert len(nodes) == 2
    assert all(len(node.key) == 5 for node in nodes)
    assert len({node.key for node in nodes}) == 2
    for node in nodes:
        assert (_ROOT / node.module).is_file(), node.module
        assert node.identity_schema_version == int(node.identity_schema_version)


def test_the_declared_module_is_the_one_the_live_provider_is_defined_in() -> None:
    """A transcribed module path keeps matching after the provider moves."""

    declared = json.loads((_ROOT / P50_RECIPE_POLICY).read_bytes())[_EDGE_KEY]["module"]
    v2 = next(
        node for node in live_identity_nodes() if node.name.endswith("editing_process_v2_identity")
    )
    assert v2.module == declared


def test_an_identity_edge_naming_the_wrong_module_resolves_to_no_node() -> None:
    """Everything else agrees: provider, schema, semantics and the live digest."""

    live = str(editing_process_v2_identity()["process_identity_sha256"])
    other_module = "src/compose_v4/rewrite/process_v2_atom_delete.py"
    assert (_ROOT / other_module).is_file()

    report = _report_after(
        T1_PANEL_POLICY, lambda payload: payload[_EDGE_KEY].update(module=other_module)
    )
    assert report["status"] == DISAGREES
    named = _findings(report, "unclassified_identity_edge")
    assert named, report["findings"]
    row = next(
        entry for entry in report["identity_edges"] if entry["location"].startswith(T1_PANEL_POLICY)
    )
    assert row["resolved_nodes"] == []
    assert row["sha256"] == live, "the digest must still be the live one, or this proves nothing"


def test_an_identity_edge_naming_a_nonexistent_schema_version_resolves_to_no_node() -> None:
    live_version = json.loads((_ROOT / T1_PANEL_POLICY).read_bytes())[_EDGE_KEY][
        "identity_schema_version"
    ]
    report = _report_after(
        T1_PANEL_POLICY,
        lambda payload: payload[_EDGE_KEY].update(identity_schema_version=live_version + 1),
    )
    assert report["status"] == DISAGREES
    assert _findings(report, "unclassified_identity_edge"), report["findings"]


def test_a_schema_version_written_as_a_string_is_not_coerced_into_matching() -> None:
    """``str()`` coercion on a naming field makes the validator weaker than the builder."""

    live_version = json.loads((_ROOT / T1_PANEL_POLICY).read_bytes())[_EDGE_KEY][
        "identity_schema_version"
    ]
    report = _report_after(
        T1_PANEL_POLICY,
        lambda payload: payload[_EDGE_KEY].update(identity_schema_version=str(live_version)),
    )
    assert report["status"] == DISAGREES
    assert _findings(report, "unclassified_identity_edge"), report["findings"]


# ---- (3) a normalized repository target stays inside the checkout ----


@pytest.mark.parametrize(
    "target",
    ["../outside.json", "configs/../../outside.json", "/etc/hosts", "/tmp/outside.json"],
    ids=["parent", "buried_parent", "absolute_system", "absolute_tmp"],
)
def test_a_repository_target_that_escapes_the_checkout_fails(target: str) -> None:
    def escape(payload: dict[str, Any]) -> None:
        _pointer_of(payload, _T1_PANEL_ROLE, "physical")["target"] = target

    report = _report_after(T1_PANEL_POLICY, escape)
    assert report["status"] == DISAGREES
    named = _findings(report, "pointer_target_escapes_the_checkout")
    assert named, report["findings"]
    assert named[0]["severity"] == FAIL
    assert [edge["result"] for edge in _edges_to(report, target)] == ["escapes_checkout"]
    assert target not in report["artifacts"], "an escaping target must not be traversed"


def test_a_symlink_that_leaves_the_checkout_fails_even_when_its_bytes_agree() -> None:
    """The sharp case: containment is decided BEFORE the hash, so agreement is no defence.

    Without normalization this pin reads a file outside the checkout, hashes it,
    finds the declared value, and reports ``agrees``.
    """

    with tempfile.TemporaryDirectory() as outside_raw:
        outside = Path(outside_raw) / "smuggled.json"
        outside.write_bytes(b'{"schema": "smuggled"}\n')
        digest = hashlib.sha256(outside.read_bytes()).hexdigest()
        smuggled = "configs/smuggled.json"

        def repoint(payload: dict[str, Any]) -> None:
            pointer = _pointer_of(payload, _T1_PANEL_ROLE, "physical")
            pointer["target"] = smuggled
            pointer["sha256"] = digest
            pointer["target_schema"] = "smuggled"

        with _graph_copy() as root:
            (root / smuggled).symlink_to(outside)
            assert (root / smuggled).is_file(), "the fixture symlink must resolve"
            assert hashlib.sha256((root / smuggled).read_bytes()).hexdigest() == digest
            _reseal(root, T1_PANEL_POLICY, repoint)
            report = verify_process_v2_chain(repo_root=root)

    assert report["status"] == DISAGREES
    assert _findings(report, "pointer_target_escapes_the_checkout"), report["findings"]
    assert [edge["result"] for edge in _edges_to(report, smuggled)] == ["escapes_checkout"]


def test_a_target_that_resolves_inside_but_is_not_normalized_is_still_refused() -> None:
    """Discriminates normalization from containment; a resolve-based check alone passes this.

    ``configs/../configs/x.json`` names a file the checkout does govern, so every
    escape test above would still pass with the ``..`` rule deleted. What is wrong
    is that a target is an IDENTITY: the chain compares declared targets by string,
    so a second spelling of one file is a pointer that no longer compares equal to
    the edge it was written for.
    """

    spelling = f"configs/../{EDITING_CORPUS_V2_CONTRACT}"
    assert (_ROOT / spelling).is_file(), "the fixture must resolve to a real, governed file"
    assert normalized_repository_path(_ROOT, spelling) is None

    def respell(payload: dict[str, Any]) -> None:
        for slot in ("physical", "semantic"):
            _pointer_of(payload, _P50_JSON_ROLE, slot)["target"] = spelling

    report = _report_after(P50_RECIPE_POLICY, respell)
    assert report["status"] == DISAGREES
    named = _findings(report, "pointer_target_escapes_the_checkout")
    assert len(named) == 2, report["findings"]
    assert {edge["result"] for edge in _edges_to(report, spelling)} == {"escapes_checkout"}


def test_a_symlink_that_stays_inside_the_checkout_is_accepted() -> None:
    """Containment, not a blanket refusal of symlinks."""

    with _graph_copy() as root:
        inside = root / "configs/alias_of_the_corpus_contract.json"
        inside.symlink_to(root / EDITING_CORPUS_V2_CONTRACT)
        assert normalized_repository_path(root, "configs/alias_of_the_corpus_contract.json")
    assert normalized_repository_path(_ROOT, EDITING_CORPUS_V2_CONTRACT) is not None
    assert normalized_repository_path(_ROOT, "../escape") is None
    assert normalized_repository_path(_ROOT, "/etc/hosts") is None
    assert normalized_repository_path(_ROOT, "") is None


# ---- (4) the declared target schema must be the target's own schema ----


def test_the_committed_graph_agrees_on_every_declared_target_schema() -> None:
    report = verify_process_v2_chain(repo_root=_ROOT)
    assert "target_schema_disagrees" not in _categories(report)


def test_a_pointer_binding_the_right_bytes_under_the_wrong_schema_fails() -> None:
    """No hash in the graph can see this: both sides of every hash still agree.

    The mutation is applied to the graph LEAF, which no artifact pins, so nothing
    else in the run can go stale and the absence of every hash-staleness finding is
    a measurement rather than an accident of which artifact was chosen.
    """

    def relabel(payload: dict[str, Any]) -> None:
        for slot in ("physical", "semantic"):
            _pointer_of(payload, _P50_JSON_ROLE, slot)["target_schema"] = "compose.other"

    report = _report_after(P50_RECIPE_POLICY, relabel)
    assert report["status"] == DISAGREES
    named = _findings(report, "target_schema_disagrees")
    assert len(named) == 2, report["findings"]
    assert all(finding["severity"] == FAIL for finding in named)
    assert "stale_physical_pointer" not in _categories(report)
    assert "stale_semantic_pointer" not in _categories(report)
    assert set(_categories(report)) == {
        "root_artifact_refused_by_its_validator",
        "target_schema_disagrees",
    }


def test_a_pointer_declaring_no_schema_for_a_target_that_has_one_fails() -> None:
    """``None`` is a claim too: it says the target carries no schema of its own."""

    live = json.loads((_ROOT / P50_RECIPE_POLICY).read_bytes())
    assert live["parents"][_P50_JSON_ROLE]["physical"]["target_schema"] is not None

    def blank(payload: dict[str, Any]) -> None:
        for slot in ("physical", "semantic"):
            _pointer_of(payload, _P50_JSON_ROLE, slot)["target_schema"] = None

    report = _report_after(P50_RECIPE_POLICY, blank)
    assert report["status"] == DISAGREES
    assert len(_findings(report, "target_schema_disagrees")) == 2, report["findings"]


def test_a_pointer_declaring_a_schema_for_a_target_that_has_none_fails() -> None:
    """A non-JSON target has no separable schema, so any declared one is wrong.

    The capability-cell classifier is a Python source file, pinned physically and
    correctly declaring ``target_schema: null``.
    """

    live = json.loads((_ROOT / CAPABILITY_CELLS).read_bytes())
    assert live["parents"][_CELLS_UNSCHEMED_ROLE]["physical"]["target_schema"] is None

    def invent(payload: dict[str, Any]) -> None:
        _pointer_of(payload, _CELLS_UNSCHEMED_ROLE, "physical")["target_schema"] = (
            "compose.editing_v2.capability_cell_classifier"
        )

    report = _report_after(CAPABILITY_CELLS, invent)
    assert report["status"] == DISAGREES
    named = _findings(report, "target_schema_disagrees")
    assert len(named) == 1, report["findings"]
    assert named[0]["location"].endswith(f".parents.{_CELLS_UNSCHEMED_ROLE}.physical")


def test_a_lineage_pointer_target_schema_is_deliberately_not_compared() -> None:
    """A lineage pin describes the SUPERSEDED artifact and makes no currency claim.

    Comparing it to the live schema would turn a correct lineage record into a
    failure exactly when the live artifact is renamed, which is when lineage is
    load-bearing.
    """

    def relabel(payload: dict[str, Any]) -> None:
        for slot in ("physical", "semantic"):
            payload["superseded_design_lineage"][slot]["target_schema"] = "compose.old.name"

    report = _report_after(T1_PANEL_POLICY, relabel)
    assert "target_schema_disagrees" not in _categories(report)
    assert "lineage_value_is_live" not in _categories(report)


# ---- (5) a deferral is undecided until a versioned stage receipt resolves it ----


def _remote_pointer(sha256: str = "c" * 64) -> dict[str, Any]:
    return typed_pointer(
        kind=PointerKind.REMOTE_ARTIFACT,
        provider="semantic_migration_completion",
        target=_REMOTE,
        target_schema="compose.editing_v2.semantic_migration_complete",
        identity_role=IdentityRole.PHYSICAL,
        sha256=sha256,
    )


def test_an_unresolved_remote_pointer_is_inconclusive_not_agreeing() -> None:
    with _graph_copy() as root:
        _inject_into_the_frozen_parent(root, _remote_pointer(), "a_remote_prerequisite")
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == INCONCLUSIVE
    assert _categories(report, FAIL) == []
    assert _categories(report, UNVERIFIED) == ["remote_artifact_unresolved_by_any_stage_receipt"]
    assert [edge["result"] for edge in _edges_to(report, _REMOTE)] == ["deferred"]


def _resolving(entry: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Verify the injected remote pointer with one receipt carrying ``entry``."""

    with _graph_copy() as root:
        _inject_into_the_frozen_parent(root, _remote_pointer(), "a_remote_prerequisite")
        receipt = _stage_receipt(root, "diagnostics/stage_receipt.json", [entry], **kwargs)
        return verify_process_v2_chain(repo_root=root, stage_receipts=(receipt,))


def _the_resolving_entry(sha256: str = "c" * 64, **kwargs: Any) -> dict[str, Any]:
    return _resolution(PointerKind.REMOTE_ARTIFACT, _REMOTE, sha256, **kwargs)


def test_a_versioned_stage_receipt_resolves_a_remote_pointer() -> None:
    report = _resolving(_the_resolving_entry())
    assert report["status"] == AGREES, report["findings"]
    assert _categories(report, DEFERRED) == ["remote_artifact_resolved_by_stage_receipt"]
    assert [edge["result"] for edge in _edges_to(report, _REMOTE)] == ["resolved_by_stage_receipt"]
    assert [row["result"] for row in report["stage_receipts"]] == ["accepted"]


def test_a_receipt_naming_a_different_digest_resolves_nothing() -> None:
    report = _resolving(_the_resolving_entry(sha256="d" * 64))
    assert report["status"] == INCONCLUSIVE
    assert "remote_artifact_unresolved_by_any_stage_receipt" in _categories(report, UNVERIFIED)


def test_a_receipt_naming_a_different_target_resolves_nothing() -> None:
    """The other half of the resolution key, which had no test at all.

    A receipt resolves a pointer only when kind, TARGET and digest all match. The
    digest half was covered and the target half was not, so a resolution index
    keyed on kind and digest alone -- one receipt resolving every pointer that
    happens to share a hash -- passed the suite.
    """

    report = _resolving(
        _resolution(PointerKind.REMOTE_ARTIFACT, f"{_REMOTE}.other", "c" * 64)
    )
    assert report["status"] == INCONCLUSIVE
    assert "remote_artifact_unresolved_by_any_stage_receipt" in _categories(report, UNVERIFIED)
    assert [row["result"] for row in report["stage_receipts"]] == ["accepted"], (
        "the receipt itself is sound; it simply resolves something else"
    )


def test_a_receipt_naming_a_different_kind_resolves_nothing() -> None:
    """And the third: an external-asset resolution does not resolve a remote pin."""

    report = _resolving(_resolution(PointerKind.EXTERNAL_ASSET, _REMOTE, "c" * 64))
    assert report["status"] == INCONCLUSIVE
    assert "remote_artifact_unresolved_by_any_stage_receipt" in _categories(report, UNVERIFIED)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"corrupt_self_hash": True},
        {"schema_version": STAGE_RECEIPT_SCHEMA_VERSION + 1},
        {"schema": "compose.some.other.receipt"},
        {"authority": {**authority_false_block(), "bounded_p50_authorized": True}},
        {"authority": {**authority_false_block(), "training_authorized": True}},
        {"extra": {"p500_authorized": True}},
        {"extra": {"resolved_by": {"nested_grant_authorized": True}}},
        {"status": "STAGE_RECEIPT_REFUSED_NOTHING_VERIFIED"},
        {"status": ""},
        {"stage": ""},
        {"stage": 7},
        {"omit": ("status",)},
        {"omit": ("stage",)},
        {"omit": ("resolved",)},
        {"omit": ("schema_version",)},
    ],
    ids=[
        "broken_self_hash",
        "wrong_version",
        "wrong_schema",
        "grants_bounded_p50",
        "grants_training",
        "grants_an_authority_outside_the_vocabulary",
        "grants_authority_nested",
        "refused_status",
        "blank_status",
        "blank_stage",
        "non_string_stage",
        "omits_status",
        "omits_stage",
        "omits_resolved",
        "omits_schema_version",
    ],
)
def test_a_receipt_that_is_not_itself_sound_resolves_nothing(kwargs: dict[str, Any]) -> None:
    """A receipt is the only thing that can turn a deferral into agreement.

    So it is held to the standard of what it resolves: a declared schema and
    version, a verified self-hash, the one disposition under which it resolves
    anything, its full required field set, and no granted authority under ANY
    spelling. An unversioned or authority-granting receipt that still resolved
    would be a launch permit written by the thing it permits.

    ``p500_authorized`` is the out-of-vocabulary case that made the difference:
    ``require_authority_false`` is bound to the seven known field names, so the
    one authority spelling this repository has outside them was accepted, and a
    ``status`` saying the receipt had verified nothing was accepted too, because
    the field was required and never read.
    """

    report = _resolving(_the_resolving_entry(), **kwargs)
    assert report["status"] == DISAGREES
    assert "malformed_stage_receipt" in _categories(report, FAIL)
    assert "remote_artifact_unresolved_by_any_stage_receipt" in _categories(report, UNVERIFIED)
    assert [row["result"] for row in report["stage_receipts"]] == ["refused"]


@pytest.mark.parametrize(
    "entry",
    [
        _resolution(PointerKind.REMOTE_ARTIFACT, _REMOTE, "c" * 64, verified_by="nobody at all"),
        _resolution(PointerKind.REMOTE_ARTIFACT, _REMOTE, "c" * 64, verified_by=""),
        _resolution(PointerKind.REMOTE_ARTIFACT, _REMOTE, "C" * 64),
        _resolution(PointerKind.REMOTE_ARTIFACT, _REMOTE, "c" * 63),
        _resolution(PointerKind.REMOTE_ARTIFACT, _REMOTE, "not a digest at all"),
        {**_the_resolving_entry(), "note": "a field beyond the entry set"},
        {k: v for k, v in _the_resolving_entry().items() if k != "target"},
        {"kind": PointerKind.REMOTE_ARTIFACT, "target": _REMOTE},
    ],
    ids=[
        "attributed_to_nobody",
        "attributed_to_nothing",
        "uppercase_digest",
        "short_digest",
        "unparseable_digest",
        "extra_field",
        "missing_target",
        "half_an_entry",
    ],
)
def test_a_receipt_entry_that_is_not_itself_sound_resolves_nothing(
    entry: dict[str, Any],
) -> None:
    """An entry carries exactly the compared fields, and every one is compared.

    ``verified_by`` was the odd one out: required, non-empty, and never compared
    to anything, so ``"nobody at all"`` resolved a deferred edge. There is no
    registry of stage loaders here to decide it against, so it is bound to the
    receipt's own ``stage`` -- one receipt carries one stage's evidence -- which
    makes every declared field of an entry a field that must agree with something.
    """

    report = _resolving(entry)
    assert report["status"] == DISAGREES
    assert "malformed_stage_receipt" in _categories(report, FAIL)
    assert "remote_artifact_unresolved_by_any_stage_receipt" in _categories(report, UNVERIFIED)
    assert [row["result"] for row in report["stage_receipts"]] == ["refused"]


def test_the_entry_field_set_is_exactly_the_fields_that_are_compared() -> None:
    """No field of a resolution entry may be recorded without being decided.

    ``kind``, ``target`` and ``sha256`` are compared to the pointer;
    ``verified_by`` is compared to the receipt's stage. A field added here without
    a comparison would be the same "required, recorded, never read" shape that let
    a refused receipt resolve edges.
    """

    assert set(STAGE_RECEIPT_ENTRY_FIELDS) == {"kind", "target", "sha256", "verified_by"}


def test_a_non_list_resolved_block_resolves_nothing() -> None:
    """``resolved`` is iterated, so a mapping would silently yield its KEYS.

    The diagnostic is asserted, not just the refusal. Without the list check a
    mapping still gets refused -- by the per-entry rule, complaining that its first
    KEY is not an entry -- so a run that reported the wrong defect, and one that
    raised ``TypeError`` on an integer, both looked like a pass.
    """

    for resolved in ({"kind": "remote_artifact"}, "a string", 7, None):
        with _graph_copy() as root:
            _inject_into_the_frozen_parent(root, _remote_pointer(), "a_remote_prerequisite")
            receipt = _stage_receipt(root, "diagnostics/stage_receipt.json", resolved)  # type: ignore[arg-type]
            report = verify_process_v2_chain(repo_root=root, stage_receipts=(receipt,))
        assert report["status"] == DISAGREES, resolved
        named = _findings(report, "malformed_stage_receipt")
        assert named, (resolved, report["findings"])
        assert "must be a list" in named[0]["detail"], (resolved, named[0]["detail"])
        assert [row["result"] for row in report["stage_receipts"]] == ["refused"]


def test_a_receipt_read_from_outside_the_checkout_resolves_nothing() -> None:
    """The receipt PATH is a repository path, held to the containment rule.

    ``repo_root / relative_path`` lets an absolute operand win outright, so a
    receipt named by absolute path was read from anywhere on the filesystem and
    accepted -- an ungoverned file deciding which deferred edges this checkout
    agrees about, which is the strongest thing a receipt can do.
    """

    with tempfile.TemporaryDirectory() as elsewhere:
        with _graph_copy() as root:
            _inject_into_the_frozen_parent(root, _remote_pointer(), "a_remote_prerequisite")
            receipt = _stage_receipt(
                root,
                "smuggled_receipt.json",
                [_the_resolving_entry()],
                outside=Path(elsewhere),
            )
            assert Path(receipt).is_absolute() and Path(receipt).is_file()
            report = verify_process_v2_chain(repo_root=root, stage_receipts=(receipt,))
    assert report["status"] == DISAGREES
    assert "stage_receipt_path_escapes_the_checkout" in _categories(report, FAIL)
    assert "remote_artifact_unresolved_by_any_stage_receipt" in _categories(report, UNVERIFIED)
    assert [row["result"] for row in report["stage_receipts"]] == ["refused"]
    assert [row["resolved"] for row in report["stage_receipts"]] == [0]


def test_an_external_asset_outside_the_checkout_is_undecided_until_a_receipt_resolves_it() -> None:
    asset = "vendor/guacamol/guacamol_subset_500000.smiles"
    pointer = typed_pointer(
        kind=PointerKind.EXTERNAL_ASSET,
        provider="guacamol_release",
        target=asset,
        target_schema=None,
        identity_role=IdentityRole.PHYSICAL,
        sha256="e" * 64,
    )
    with _graph_copy() as root:
        _inject_into_the_frozen_parent(root, pointer, "an_external_source_asset")
        without = verify_process_v2_chain(repo_root=root)
        receipt = _stage_receipt(
            root,
            "diagnostics/asset_receipt.json",
            [
                _resolution(
                    PointerKind.EXTERNAL_ASSET, asset, "e" * 64, verified_by="corpus_ingest"
                )
            ],
            stage="corpus_ingest",
        )
        with_receipt = verify_process_v2_chain(repo_root=root, stage_receipts=(receipt,))
    assert without["status"] == INCONCLUSIVE
    assert "external_asset_unresolved_outside_the_checkout" in _categories(without, UNVERIFIED)
    assert with_receipt["status"] == AGREES, with_receipt["findings"]
    assert [edge["result"] for edge in _edges_to(with_receipt, asset)] == [
        "resolved_by_stage_receipt"
    ]


def test_a_missing_receipt_file_is_a_failure_not_a_silent_skip() -> None:
    with _graph_copy() as root:
        report = verify_process_v2_chain(
            repo_root=root, stage_receipts=("diagnostics/no_such_receipt.json",)
        )
    assert report["status"] == DISAGREES
    assert "unreadable_stage_receipt" in _categories(report, FAIL)
