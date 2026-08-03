"""The generic verifier must not report agreement it did not prove.

Its predecessor could print ``AGREES`` while three mechanisms silently declined to
check anything, so every test here is a mutation that the predecessor passed and
this verifier must fail. They drive the production entry point
``verify_process_v2_chain`` over a mutated copy of the real graph and assert on the
report it returns; none of them reimplements a check.

The three mechanisms, and the tests that close them:

1. an identity pin was accepted if it matched *either* live identity, and was
   recorded as addressing both nodes -- see the identity section;
2. an edge existed only if its target did, so deleting or misspelling a target
   removed the edge instead of failing -- see the pointer-target section;
3. an unverifiable edge was a warning and the run still said ``AGREES`` -- see the
   status section.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data.editing_v2_process_v2_policy_registry import FROZEN_POLICY_SOURCES
from compose_v4.data.editing_v2_process_v2_schema import (
    IdentityRole,
    PointerKind,
    typed_pointer,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    CAPABILITY_CELLS,
    DEVELOPMENT_CELL_ROLES,
    EDITING_CORPUS_V2_CONTRACT,
    P50_RECIPE_POLICY,
    PROCESS_V2_CHAIN_ARTIFACTS,
    SELF_HASH_FIELD,
    SELF_HASH_FIELD_ALGORITHM,
    T1_CAPACITY_POLICY,
    T1_PANEL_POLICY,
    WHOLE_CANONICAL_BODY_ALGORITHM,
    process_v2_chain_self_hash,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_process_v2_identity,
    editing_v2_process_identity,
)

import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from verify_process_v2_chain import (  # noqa: E402
    AGREES,
    DEFERRED,
    DISAGREES,
    FAIL,
    INCONCLUSIVE,
    UNVERIFIED,
    live_identity_nodes,
    main,
    verify_process_v2_chain,
)

#: Everything reachable from the seven roots, so a copied graph is complete.
_GRAPH_MEMBERS: tuple[str, ...] = (
    *PROCESS_V2_CHAIN_ARTIFACTS,
    "configs/editing_gate_zero_semantic_model_process_v2.json",
    "configs/editing_v2_semantic_process_v2.json",
    EDITING_CORPUS_V2_CONTRACT,
    "src/compose_v4/data/editing_v2_semantic_capability_cells.py",
)


# ---- Helpers ----


@contextmanager
def _graph_copy() -> Iterator[Path]:
    """A throwaway root holding the complete declared graph.

    The frozen policy sources are copied alongside it. They are not graph members
    -- no artifact declares a pointer at them -- but the chain validator projects
    from them, so a root without them would make a chain-side assertion fail for
    the wrong reason.
    """

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        sources = [source.relative_path for source in FROZEN_POLICY_SOURCES.values()]
        for relative_path in (*_GRAPH_MEMBERS, *sources):
            target = root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_ROOT / relative_path, target)
        yield root


def _reseal(root: Path, name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Mutate one artifact and reseal its self-hash so it stays self-consistent.

    Resealing matters: a mutation that also broke the self-hash could fail for the
    wrong reason, and then the test would prove nothing about the defect it names.
    ``process_v2_chain_self_hash`` is the production sealing rule; transcribing the
    canonical-JSON encoding here instead would be copying production logic into a
    test.
    """

    path = root / name
    payload = json.loads(path.read_bytes())
    mutate(payload)
    payload.pop(SELF_HASH_FIELD, None)
    payload[SELF_HASH_FIELD] = process_v2_chain_self_hash(payload)
    path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    )


def _categories(report: dict[str, Any], severity: str | None = None) -> list[str]:
    return [
        finding["category"]
        for finding in report["findings"]
        if severity is None or finding["severity"] == severity
    ]


def _edges_to(report: dict[str, Any], target: str) -> list[dict[str, Any]]:
    return [edge for edge in report["edges"] if edge["target"] == target]


# ---- The committed graph agrees ----


def test_the_committed_graph_agrees_with_no_findings() -> None:
    report = verify_process_v2_chain(repo_root=_ROOT)
    assert report["status"] == AGREES, report["findings"]
    assert report["findings"] == []
    assert set(report["artifacts"]) == set(_GRAPH_MEMBERS)
    assert report["edges"], "the graph declares no pointers"
    assert all(
        edge["result"] in {"agrees", "historical"} for edge in report["edges"]
    ), report["edges"]


def test_every_chain_artifact_declares_exactly_one_resolved_identity_edge() -> None:
    report = verify_process_v2_chain(repo_root=_ROOT)
    located = {row["location"].split(":")[0] for row in report["identity_edges"]}
    assert located == set(PROCESS_V2_CHAIN_ARTIFACTS)
    for row in report["identity_edges"]:
        assert row["resolved_nodes"] == ["identity:editing_process_v2_identity"], row
        assert row["result"] == "agrees", row


# ---- (1) identity edges resolve to exactly one node ----


def test_a_resealed_v2_config_carrying_the_live_v1_identity_fails() -> None:
    """The predecessor accepted this: it took either live identity for any artifact."""

    live_v1 = str(editing_v2_process_identity()["process_identity_sha256"])
    live_v2 = str(editing_process_v2_identity()["process_identity_sha256"])
    assert live_v1 != live_v2

    def swap(payload: dict[str, Any]) -> None:
        payload["process_identity"]["process_identity_sha256"] = live_v1

    with _graph_copy() as root:
        _reseal(root, P50_RECIPE_POLICY, swap)
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    assert "stale_identity_pin" in _categories(report, FAIL)
    row = next(
        entry
        for entry in report["identity_edges"]
        if entry["location"].startswith(P50_RECIPE_POLICY)
    )
    assert row["resolved_nodes"] == ["identity:editing_process_v2_identity"]
    assert row["result"] == "disagrees"
    detail = next(
        finding["detail"]
        for finding in report["findings"]
        if finding["category"] == "stale_identity_pin"
    )
    assert "identity:editing_v2_process_identity" in detail


def test_an_unclassified_identity_does_not_fan_out_to_both_nodes() -> None:
    """An identity this checkout does not compute resolves to NO node.

    The predecessor reported every identity pin as addressing both the V1 and the
    V2 node, which made "which process does this artifact describe" unanswerable
    from its own report.
    """

    def unclassify(payload: dict[str, Any]) -> None:
        payload["process_identity"]["provider"] = "some_other_identity_provider"

    with _graph_copy() as root:
        _reseal(root, P50_RECIPE_POLICY, unclassify)
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    assert "unclassified_identity_edge" in _categories(report, FAIL)
    row = next(
        entry
        for entry in report["identity_edges"]
        if entry["location"].startswith(P50_RECIPE_POLICY)
    )
    assert row["resolved_nodes"] == [], "an unclassified identity must resolve to no node"
    assert row["result"] == "unclassified"


def test_a_v1_declaration_carrying_the_v1_value_still_fails_in_a_v2_artifact() -> None:
    """Relabelling the declaration to match the value is not a repair.

    ``editing_v2_process_identity`` is a node this checkout computes, so the
    declaration resolves and its value agrees. What must still fail is that a
    Process-V2 chain artifact now declares the V1 process, which the CHAIN
    validator refuses -- the generic verifier is deliberately process-agnostic and
    reports agreement of the edge itself.
    """

    v1 = editing_v2_process_identity()

    def relabel(payload: dict[str, Any]) -> None:
        payload["process_identity"] = {
            "identity_schema": str(v1["schema"]),
            "identity_schema_version": int(v1["schema_version"]),
            "module": "src/compose_v4/rewrite/editing_v2_process_identity.py",
            "process_identity_sha256": str(v1["process_identity_sha256"]),
            "process_semantics": str(v1["process_semantics"]),
            "provider": "editing_v2_process_identity",
        }

    from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
        ProcessV2ChainError,
        load_process_v2_chain_artifact,
    )

    with _graph_copy() as root:
        _reseal(root, P50_RECIPE_POLICY, relabel)
        report = verify_process_v2_chain(repo_root=root)
        row = next(
            entry
            for entry in report["identity_edges"]
            if entry["location"].startswith(P50_RECIPE_POLICY)
        )
        assert row["resolved_nodes"] == ["identity:editing_v2_process_identity"]
        assert row["result"] == "agrees"
        with pytest.raises(ProcessV2ChainError, match="resolves only"):
            load_process_v2_chain_artifact(P50_RECIPE_POLICY, repo_root=root)


def test_the_live_identity_nodes_are_keyed_uniquely() -> None:
    nodes = live_identity_nodes()
    assert len({node.key for node in nodes}) == len(nodes) == 2
    assert len({node.value for node in nodes}) == 2


# ---- (2) a declared pointer cannot vanish ----


def test_a_missing_parent_path_fails() -> None:
    with _graph_copy() as root:
        (root / DEVELOPMENT_CELL_ROLES).unlink()
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    categories = _categories(report, FAIL)
    assert "missing_repository_target" in categories
    assert "missing_graph_artifact" in categories
    assert _edges_to(report, DEVELOPMENT_CELL_ROLES), (
        "the edge must remain in the graph after its target is deleted"
    )
    assert all(
        edge["result"] == "missing_target" for edge in _edges_to(report, DEVELOPMENT_CELL_ROLES)
    )


def test_a_misspelled_parent_path_fails() -> None:
    """The predecessor's exact blind spot: it discovered an edge only if the
    target existed, so a typo removed the edge rather than failing it."""

    misspelled = "configs/editing_v2_process_v2_development_cell_role.json"

    def typo(payload: dict[str, Any]) -> None:
        for slot in payload["parents"]["development_cell_roles"].values():
            slot["target"] = misspelled

    with _graph_copy() as root:
        _reseal(root, T1_PANEL_POLICY, typo)
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    assert "missing_repository_target" in _categories(report, FAIL)
    assert _edges_to(report, misspelled), "a misspelled target must still be an edge"


def test_a_remote_artifact_pointer_is_deferred_not_reported_missing() -> None:
    """A volume path is verified by its stage loader, never by the local filesystem."""

    remote = "/editing_v2/semantic_v4_migration/fixture/SEMANTIC_MIGRATION_COMPLETE.json"

    def inject(payload: dict[str, Any]) -> None:
        payload["parents"]["a_remote_prerequisite"] = {
            "physical": typed_pointer(
                kind=PointerKind.REMOTE_ARTIFACT,
                provider="semantic_migration_completion",
                target=remote,
                target_schema="compose.editing_v2.semantic_migration_complete",
                identity_role=IdentityRole.PHYSICAL,
                sha256="c" * 64,
            )
        }

    with _graph_copy() as root:
        _reseal(root, P50_RECIPE_POLICY, inject)
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == AGREES, report["findings"]
    assert _categories(report, FAIL) == []
    assert "remote_artifact_deferred_to_stage_loader" in _categories(report, DEFERRED)
    assert [edge["result"] for edge in _edges_to(report, remote)] == ["deferred"]


# ---- (3) semantic hashes are checked under the declared algorithm ----


def test_an_incorrect_canonical_semantic_hash_fails() -> None:
    def corrupt(payload: dict[str, Any]) -> None:
        payload["parents"]["source_requirements"]["semantic"]["sha256"] = "d" * 64

    with _graph_copy() as root:
        _reseal(root, T1_PANEL_POLICY, corrupt)
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    assert "stale_semantic_pointer" in _categories(report, FAIL)
    detail = next(
        finding["detail"]
        for finding in report["findings"]
        if finding["category"] == "stale_semantic_pointer"
    )
    assert WHOLE_CANONICAL_BODY_ALGORITHM in detail


def test_a_wrong_semantic_algorithm_fails_rather_than_warning() -> None:
    """Declaring the whole-body rule for a target that self-hashes is a defect.

    The predecessor emitted a warning here and still reported ``AGREES``.
    """

    def swap_algorithm(payload: dict[str, Any]) -> None:
        pointer = payload["parents"]["development_cell_roles"]["semantic"]
        pointer["hash_algorithm"] = WHOLE_CANONICAL_BODY_ALGORITHM

    with _graph_copy() as root:
        _reseal(root, T1_CAPACITY_POLICY, swap_algorithm)
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    assert "wrong_semantic_algorithm" in _categories(report, FAIL)


def test_a_self_hash_algorithm_on_a_target_without_one_fails() -> None:
    def swap_algorithm(payload: dict[str, Any]) -> None:
        pointer = payload["parents"]["source_requirements"]["semantic"]
        pointer["hash_algorithm"] = SELF_HASH_FIELD_ALGORITHM

    with _graph_copy() as root:
        _reseal(root, T1_PANEL_POLICY, swap_algorithm)
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    assert "semantic_algorithm_unsatisfiable" in _categories(report, FAIL)


def test_a_stale_physical_pointer_fails() -> None:
    def corrupt(payload: dict[str, Any]) -> None:
        payload["parents"]["capability_cell_registry"]["physical"]["sha256"] = "e" * 64

    with _graph_copy() as root:
        _reseal(root, DEVELOPMENT_CELL_ROLES, corrupt)
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    assert "stale_physical_pointer" in _categories(report, FAIL)


def test_an_edited_parent_makes_its_child_pointer_stale() -> None:
    """The whole point of a content-addressed chain, exercised end to end.

    The parent is RESEALED after the edit, so it stays internally self-consistent
    and the only thing left to catch is the child's now-stale pin. Without the
    reseal the parent's self-hash would break first and the child's semantic
    pointer would fail as unsatisfiable, which proves something weaker.
    """

    def edit(payload: dict[str, Any]) -> None:
        payload["family_contexts"]["atom_delete"].append("an_invented_context")

    with _graph_copy() as root:
        _reseal(root, CAPABILITY_CELLS, edit)
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    categories = _categories(report, FAIL)
    assert "stale_physical_pointer" in categories
    assert "stale_semantic_pointer" in categories


# ---- Lineage references carry no currency claim ----


def test_a_lineage_value_equal_to_the_live_value_fails() -> None:
    """"Superseded" that equals what is live is a false claim, not lineage.

    The lie is aimed at a DIFFERENT artifact than the one being resealed. Aiming
    it at the artifact's own self-hash could never be written down: resealing to
    keep the artifact self-consistent moves the very value the lie must equal.
    """

    with _graph_copy() as root:
        live_roles = json.loads((root / DEVELOPMENT_CELL_ROLES).read_bytes())[
            SELF_HASH_FIELD
        ]

        def lie(payload: dict[str, Any]) -> None:
            lineage = payload["superseded_design_lineage"]["semantic"]
            lineage["target"] = DEVELOPMENT_CELL_ROLES
            lineage["sha256"] = live_roles

        _reseal(root, P50_RECIPE_POLICY, lie)
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    assert "lineage_value_is_live" in _categories(report, FAIL)


def test_lineage_pointers_are_historical_in_the_committed_graph() -> None:
    report = verify_process_v2_chain(repo_root=_ROOT)
    lineage = [edge for edge in report["edges"] if edge["kind"] == PointerKind.LINEAGE_REFERENCE]
    assert len(lineage) == 2 * len(PROCESS_V2_CHAIN_ARTIFACTS)
    assert {edge["result"] for edge in lineage} == {"historical"}


# ---- (4) warnings cannot coexist with agreement ----


def test_a_graph_with_no_identity_edge_is_inconclusive_not_agreeing() -> None:
    """Nothing proves which process such a graph describes, so it is not ``AGREES``."""

    report = verify_process_v2_chain(
        repo_root=_ROOT, roots=(EDITING_CORPUS_V2_CONTRACT,)
    )
    assert report["status"] == INCONCLUSIVE
    assert _categories(report, FAIL) == []
    assert "no_identity_edge_declared" in _categories(report, UNVERIFIED)


def test_inconclusive_is_not_a_successful_exit_code() -> None:
    assert main(["--repo-root", str(_ROOT), "--root", EDITING_CORPUS_V2_CONTRACT]) == 1


def test_the_committed_graph_exits_zero() -> None:
    assert main(["--repo-root", str(_ROOT)]) == 0
    assert main(["--repo-root", str(_ROOT), "--json"]) == 0


def test_a_disagreeing_graph_exits_nonzero() -> None:
    with _graph_copy() as root:
        (root / DEVELOPMENT_CELL_ROLES).unlink()
        assert main(["--repo-root", str(root)]) == 1


def test_a_missing_repo_root_is_a_usage_failure() -> None:
    assert main(["--repo-root", str(_ROOT / "no-such-directory")]) == 2


# ---- Structural discovery ----


def test_an_undeclared_path_string_is_not_treated_as_an_edge() -> None:
    """Only DECLARED pointers are edges; a bare path is not a pin.

    The predecessor inferred an edge from any string that named an existing file,
    which is how it came to check pins nobody declared while missing the ones that
    were misspelled.
    """

    def add_prose(payload: dict[str, Any]) -> None:
        payload["scientific_scope"] = (
            payload["scientific_scope"] + "__see_configs/editing_corpus_v2_contract.json"
        )

    with _graph_copy() as root:
        before = len(verify_process_v2_chain(repo_root=root)["edges"])
        _reseal(root, P50_RECIPE_POLICY, add_prose)
        after = verify_process_v2_chain(repo_root=root)
    assert len(after["edges"]) == before
    assert after["status"] == AGREES, after["findings"]


def test_a_pointer_hidden_deeper_in_the_body_is_still_discovered() -> None:
    """Discovery is structural, so a pointer's position does not hide it."""

    def bury(payload: dict[str, Any]) -> None:
        payload["cache"]["a_nested_prerequisite"] = typed_pointer(
            kind=PointerKind.REPOSITORY_CONFIG,
            provider="frozen_repository_artifact",
            target=EDITING_CORPUS_V2_CONTRACT,
            target_schema="compose.editing_corpus_contract",
            identity_role=IdentityRole.PHYSICAL,
            sha256="f" * 64,
        )

    with _graph_copy() as root:
        _reseal(root, P50_RECIPE_POLICY, bury)
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    assert "stale_physical_pointer" in _categories(report, FAIL)
    buried = [
        edge
        for edge in report["edges"]
        if edge["location"].endswith(".cache.a_nested_prerequisite")
    ]
    assert len(buried) == 1


def test_a_file_pinned_as_a_process_identity_fails() -> None:
    def mislabel(payload: dict[str, Any]) -> None:
        payload["parents"]["development_cell_roles"]["physical"]["identity_role"] = (
            IdentityRole.PROCESS_IDENTITY
        )

    with _graph_copy() as root:
        _reseal(root, T1_PANEL_POLICY, mislabel)
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    # Caught at either layer, and the finding must NAME the defect rather than
    # degrade into a generic stale pointer. The shared schema refuses the
    # incoherent role when the serialized pointer is validated, so the verifier
    # reports it as a malformed declared pointer; were the schema to admit it,
    # the verifier's own role check reports it directly. Both are correct, and
    # both say a file is pinned as a process identity.
    categories = _categories(report, FAIL)
    assert {"malformed_declared_pointer", "file_pinned_as_a_process_identity"} & set(
        categories
    ), categories
    named = [
        finding
        for finding in report["findings"]
        if "process" in finding["detail"] and "identity" in finding["detail"]
    ]
    assert named, "the refusal must name the process-identity role"


def test_an_unparseable_artifact_is_a_failure() -> None:
    with _graph_copy() as root:
        (root / T1_CAPACITY_POLICY).write_bytes(b"{not json")
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    assert "unparseable_artifact" in _categories(report, FAIL)


def test_the_physical_hash_check_uses_real_bytes() -> None:
    """A whitespace-only edit moves the physical hash and must be caught."""

    with _graph_copy() as root:
        target = root / "configs/editing_gate_zero_semantic_model_process_v2.json"
        target.write_bytes(target.read_bytes() + b"\n")
        report = verify_process_v2_chain(repo_root=root)
    assert report["status"] == DISAGREES
    assert "stale_physical_pointer" in _categories(report, FAIL)
    live = hashlib.sha256(
        (_ROOT / "configs/editing_gate_zero_semantic_model_process_v2.json").read_bytes()
    ).hexdigest()
    assert any(
        live in finding["detail"]
        for finding in report["findings"]
        if finding["category"] == "stale_physical_pointer"
    )
