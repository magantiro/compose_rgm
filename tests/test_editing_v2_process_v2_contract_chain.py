"""The Process-V2 contract chain must mirror V1 exactly and never mix with it.

Four failures are possible here and all four are silent, so each gets explicit
tests.

1. **Redesign disguised as a re-pin.** A chain that re-pins hashes while quietly
   moving a threshold, a cell definition, or a partition count is not a mirror; it
   is a new policy with an old name. Every mirrored block is therefore read back
   out of the FROZEN V1 config on disk and compared, so an edited projection fails
   here rather than shipping.
2. **A mixed V1/V2 artifact.** The V1 chain and the V2 chain describe different
   processes and their hashes are not interchangeable. Each mixing defect -- a V1
   identity pin, a ``_v1`` parent path, a moved Active8 order, a granted authority
   flag, a stale self-hash, a changed field set -- is asserted to raise with a
   message naming THAT defect.
3. **An implied dependency.** Tuple order is not an edge. Schema version 2 declares
   Gate 0 -> T1 panel -> T1 capacity -> P50 as typed pointers, and the tests assert
   the edges exist, resolve, and were added rather than silently resealed into
   version 1.
4. **A partially published chain read as authoritative.** Publication writes a
   content-addressed generation, validates the whole graph there, and commits a
   marker last. The interruption tests stage without committing and require every
   reader to refuse the result.

Expectations are literals or values read from the V1/parent files themselves, never
recomputations of the builder under test.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_process_v2_policy_registry import (
    FROZEN_POLICY_SOURCES,
    policy_registry_identity,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    GENERATION_COMMITTED_MARKER,
    GENERATION_COMMITTED_SCHEMA,
    IdentityRole,
    PointerKind,
    validate_typed_pointer,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    ACTIVE8_DECISION_RUNTIME,
    AUTHORITY_FIELDS,
    CAPABILITY_CELLS,
    CHAIN_CONTRACT_REVISION,
    CHAIN_SCHEMA_VERSION,
    DEPENDENCY_EDGES_ADDED_IN_SCHEMA_VERSION_2,
    DEVELOPMENT_CELL_ROLES,
    EDITING_CORPUS_V2_CONTRACT,
    GATE_ZERO_STRUCTURAL,
    GENERATION_SELF_HASH_FIELD,
    P50_RECIPE_POLICY,
    PROCESS_V2_CHAIN_ARTIFACTS,
    SELF_HASH_FIELD,
    SELF_HASH_FIELD_ALGORITHM,
    SUPERSEDED_CHAIN_CONTRACT_REVISION,
    SUPERSEDED_CHAIN_CONTRACT_REVISION_V2,
    T1_CAPACITY_POLICY,
    T1_PANEL_POLICY,
    WHOLE_CANONICAL_BODY_ALGORITHM,
    ProcessV2ChainError,
    build_process_v2_chain,
    build_process_v2_chain_artifact,
    commit_process_v2_chain_generation,
    committed_generations,
    load_process_v2_chain_artifact,
    materialize_committed_generation,
    process_v2_chain_self_hash,
    process_v2_dependency_edges,
    process_v2_generation_id,
    process_v2_transitive_dependencies,
    publish_process_v2_chain,
    read_committed_generation,
    serialize_process_v2_chain_artifact,
    stage_process_v2_chain_generation,
    validate_process_v2_chain_artifact,
    write_process_v2_chain,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_process_v2_identity,
    editing_v2_process_identity,
)

_ROOT = Path(__file__).resolve().parents[1]

# The commit that froze the V1 chain.  Nothing under a ``_v1`` name may differ from
# its bytes at this revision.
_FROZEN_V1_BASE_REVISION = "d5cfcaf"

# The revision that sealed contract-chain schema version 1.
_SUPERSEDED_CHAIN_REVISION = "3f3258e"

# The revision that sealed contract-chain schema version 2.  Its bytes were
# inherited unchanged at this branch's base, so `git show` at either address
# yields the same seven artifacts.
_SUPERSEDED_CHAIN_REVISION_V2 = "b39420c"

# Every superseded generation, oldest first, as (schema version, revision that
# sealed it, its named `contract_revision`).
_SUPERSEDED_GENERATION_SOURCES: tuple[tuple[int, str, str], ...] = (
    (1, _SUPERSEDED_CHAIN_REVISION, SUPERSEDED_CHAIN_CONTRACT_REVISION),
    (2, _SUPERSEDED_CHAIN_REVISION_V2, SUPERSEDED_CHAIN_CONTRACT_REVISION_V2),
)

# The four frozen non-chain files the chain binds.  None of them is V1-named.
_EXTERNAL_PARENTS: tuple[str, ...] = (
    "configs/editing_corpus_v2_contract.json",
    "configs/editing_gate_zero_semantic_model_process_v2.json",
    "configs/editing_v2_semantic_process_v2.json",
    "src/compose_v4/data/editing_v2_semantic_capability_cells.py",
)

# The V1 artifact each chain member mirrors.
_V1_COUNTERPART: dict[str, str] = {
    ACTIVE8_DECISION_RUNTIME: "configs/editing_v2_semantic_active8_decision_runtime_v1.json",
    CAPABILITY_CELLS: "configs/editing_v2_semantic_capability_cells_v1.json",
    DEVELOPMENT_CELL_ROLES: "configs/editing_v2_semantic_development_cell_roles_v1.json",
    GATE_ZERO_STRUCTURAL: "configs/editing_v2_semantic_gate_zero_structural_v1.json",
    T1_PANEL_POLICY: "configs/editing_v2_semantic_t1_panel_policy_v1.json",
    T1_CAPACITY_POLICY: "configs/editing_v2_semantic_t1_capacity_policy_v1.json",
    P50_RECIPE_POLICY: "configs/editing_v2_semantic_p50_recipe_policy_v1.json",
}


# ---- Helpers ----


def _load(relative_path: str) -> dict:
    return json.loads((_ROOT / relative_path).read_bytes())


def _physical(relative_path: str) -> str:
    return hashlib.sha256((_ROOT / relative_path).read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@contextmanager
def _isolated_repo() -> Iterator[Path]:
    """A throwaway repo root holding the frozen external parents and policy sources.

    The chain is written there rather than into the working tree so that a
    byte-stability test can never rewrite a committed artifact as a side effect.
    """

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        sources = [source.relative_path for source in FROZEN_POLICY_SOURCES.values()]
        for relative_path in (*_EXTERNAL_PARENTS, *sources):
            target = root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_ROOT / relative_path, target)
        yield root


def _git_show(revision: str, relative_path: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "-C", str(_ROOT), "show", f"{revision}:{relative_path}"],
        capture_output=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def _revision_available(revision: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(_ROOT), "rev-parse", "--verify", f"{revision}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def _v1_config_paths() -> list[str]:
    return sorted(
        str(path.relative_to(_ROOT))
        for path in (_ROOT / "configs").glob("*.json")
        if "_v1" in path.stem
    )


# ---- (a) build -> validate -> load round trip ----


def test_every_artifact_round_trips_build_validate_load() -> None:
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        built = build_process_v2_chain_artifact(name, repo_root=_ROOT)
        validated = validate_process_v2_chain_artifact(built, name=name, repo_root=_ROOT)
        assert validated == built
        loaded = load_process_v2_chain_artifact(name, repo_root=_ROOT)
        assert loaded == built, f"{name} on disk differs from its deterministic rebuild"
        assert (_ROOT / name).read_bytes() == serialize_process_v2_chain_artifact(built)


def test_chain_has_seven_artifacts_in_dependency_order() -> None:
    assert PROCESS_V2_CHAIN_ARTIFACTS == (
        ACTIVE8_DECISION_RUNTIME,
        CAPABILITY_CELLS,
        DEVELOPMENT_CELL_ROLES,
        GATE_ZERO_STRUCTURAL,
        T1_PANEL_POLICY,
        T1_CAPACITY_POLICY,
        P50_RECIPE_POLICY,
    )
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        assert name.startswith("configs/editing_v2_process_v2_")
        assert "_v1" not in Path(name).stem


def test_each_artifact_carries_exactly_one_self_hash_field() -> None:
    """The generic verifier DISCOVERS a self-hash by its equation, so there must be one."""

    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        payload = _load(name)
        satisfying = [
            key
            for key, value in payload.items()
            if key.endswith("_sha256")
            and isinstance(value, str)
            and value == _canonical({k: v for k, v in payload.items() if k != key})
        ]
        assert satisfying == ["contract_sha256"], f"{name} self-hash fields: {satisfying}"
        assert payload["contract_sha256"] == process_v2_chain_self_hash(payload)


# ---- (b) byte stability ----


def test_write_process_v2_chain_is_byte_stable_across_runs() -> None:
    with _isolated_repo() as root:
        first = write_process_v2_chain(root)
        first_bytes = {n: (root / n).read_bytes() for n in PROCESS_V2_CHAIN_ARTIFACTS}
        second = write_process_v2_chain(root)
        second_bytes = {n: (root / n).read_bytes() for n in PROCESS_V2_CHAIN_ARTIFACTS}
        assert first == second
        assert first_bytes == second_bytes
        assert list(first) == list(PROCESS_V2_CHAIN_ARTIFACTS)
        for name in PROCESS_V2_CHAIN_ARTIFACTS:
            assert first_bytes[name] == (_ROOT / name).read_bytes(), (
                f"the committed {name} is not what the builder produces"
            )
            assert first_bytes[name].endswith(b"\n")


def test_build_process_v2_chain_seals_parents_before_children() -> None:
    """A child pins its parent's FINAL bytes, so an unwritten parent must be an error."""

    with _isolated_repo() as root:
        with pytest.raises(ProcessV2ChainError, match="parents-first"):
            build_process_v2_chain_artifact(GATE_ZERO_STRUCTURAL, repo_root=root)
        sealed = build_process_v2_chain(root)
        structural = json.loads(sealed[GATE_ZERO_STRUCTURAL])
        for role, parent in (
            ("decision_runtime", ACTIVE8_DECISION_RUNTIME),
            ("capability_cell_registry", CAPABILITY_CELLS),
            ("development_cell_roles", DEVELOPMENT_CELL_ROLES),
        ):
            assert structural["parents"][role]["semantic"]["sha256"] == (
                json.loads(sealed[parent])[SELF_HASH_FIELD]
            )


def test_building_the_chain_writes_nothing() -> None:
    with _isolated_repo() as root:
        before = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        build_process_v2_chain(root)
        after = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        assert before == after


# ---- (c) declared typed pointers resolve ----


def test_every_edge_is_a_declared_typed_pointer_that_resolves() -> None:
    # ``semantic`` expectations are read out of the target itself, never
    # recomputed with the builder's own helper.
    corpus = "configs/editing_corpus_v2_contract.json"
    classifier = "src/compose_v4/data/editing_v2_semantic_capability_cells.py"
    expected_semantic: dict[str, str] = {
        "configs/editing_gate_zero_semantic_model_process_v2.json": _load(
            "configs/editing_gate_zero_semantic_model_process_v2.json"
        )["contract_sha256"],
        "configs/editing_v2_semantic_process_v2.json": _load(
            "configs/editing_v2_semantic_process_v2.json"
        )["contract_sha256"],
        corpus: _canonical(_load(corpus)),
    }
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        expected_semantic[name] = _load(name)["contract_sha256"]

    seen_targets: set[str] = set()
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        parents = _load(name)["parents"]
        assert parents, f"{name} declares no edge"
        for role, edge in parents.items():
            physical = validate_typed_pointer(edge["physical"], label=f"{name}:{role}")
            assert physical["kind"] == PointerKind.REPOSITORY_CONFIG
            assert physical["identity_role"] == IdentityRole.PHYSICAL
            target = physical["target"]
            seen_targets.add(target)
            assert (_ROOT / target).is_file()
            assert physical["sha256"] == _physical(target), f"{name}:{role} physical"
            if target == classifier:
                # A non-JSON target has no separable semantic body, so it is
                # pinned physically rather than by a "semantic" hash that is
                # secretly the physical one.
                assert set(edge) == {"physical"}, f"{name}:{role}"
                continue
            semantic = validate_typed_pointer(edge["semantic"], label=f"{name}:{role}")
            assert semantic["identity_role"] == IdentityRole.SEMANTIC
            assert semantic["target"] == target
            assert semantic["sha256"] == expected_semantic[target], f"{name}:{role} semantic"
            assert semantic["hash_algorithm"] == (
                WHOLE_CANONICAL_BODY_ALGORITHM
                if target == corpus
                else SELF_HASH_FIELD_ALGORITHM
            )

    assert seen_targets - set(PROCESS_V2_CHAIN_ARTIFACTS) == set(_EXTERNAL_PARENTS)


def test_every_artifact_binds_the_live_process_v2_identity() -> None:
    live_v2 = str(editing_process_v2_identity()["process_identity_sha256"])
    live_v1 = str(editing_v2_process_identity()["process_identity_sha256"])
    assert live_v1 != live_v2
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        identity = _load(name)["process_identity"]
        assert identity["process_identity_sha256"] == live_v2
        assert identity["process_semantics"] == "semantic_editing_v2_v2"
        assert identity["identity_schema"] == "compose.editing.semantic_process_v2_identity"
        assert identity["provider"] == "editing_process_v2_identity"


def test_the_scientific_process_v2_identity_did_not_move() -> None:
    """The chain may not change what it describes.

    ``editing_process_v2_identity()`` hashes ``configs/editing_v2_semantic_process
    _v2.json`` and the bound implementation sources, none of which this module
    touches. Pinning the value here makes an accidental edit to an identity-source
    file fail in the chain's own suite rather than silently at a launch gate.
    """

    assert str(editing_process_v2_identity()["process_identity_sha256"]) == (
        "0c938177a34819e6e828920c1f66e240c6eb251fe7c9ea6cfe6757829dceb2dd"
    )
    assert str(editing_v2_process_identity()["process_identity_sha256"]) == (
        "6c4721f0dd37132aae657e7aa5f1bfc01cef270662f228171c4587eb7dd48491"
    )


# ---- (d) measured evidence is NOT in a prospective contract ----


def test_no_artifact_carries_a_fillable_measured_evidence_slot() -> None:
    """The version-1 ``admitted_source`` slot could never be filled.

    The builder always emitted null hashes, the validator rebuilt the null body
    and required equality, so a correctly resealed body with measured hashes
    failed. Version 2 removes the slot; measured provenance lives in a resolved
    evidence binding that cites the contract instead.
    """

    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        payload = _load(name)
        assert "admitted_source" not in payload, name
        declaration = payload["resolved_evidence_binding"]
        assert declaration["measured_evidence_in_this_contract"] is False
        assert declaration["cited_by_schema"] == (
            "compose.editing_v2.process_v2.resolved_evidence_binding"
        )
        null_hashes = [
            path
            for path, key, value in _walk(payload)
            if key.endswith("sha256") and value is None
        ]
        assert null_hashes == [], (
            f"{name} carries {null_hashes}; a null hash is how a fillable slot is "
            "spelled, and a deterministic artifact cannot have one"
        )


def _walk(node: object, path: str = "") -> Iterator[tuple[str, str, object]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield f"{path}.{key}", key, value
            yield from _walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")


def test_a_filled_measured_slot_cannot_be_reintroduced_by_resealing() -> None:
    """The confirmation the removal rests on, kept as a regression."""

    payload = build_process_v2_chain_artifact(GATE_ZERO_STRUCTURAL, repo_root=_ROOT)
    payload["admitted_source"] = {
        "completion_sha256": "a" * 64,
        "run_identity_sha256": "b" * 64,
    }
    del payload[SELF_HASH_FIELD]
    payload[SELF_HASH_FIELD] = process_v2_chain_self_hash(payload)
    assert payload[SELF_HASH_FIELD] == process_v2_chain_self_hash(payload)
    with pytest.raises(ProcessV2ChainError, match="field set differs"):
        validate_process_v2_chain_artifact(
            payload, name=GATE_ZERO_STRUCTURAL, repo_root=_ROOT
        )


# ---- (e) the dependency graph is explicit ----


def test_the_declared_graph_encodes_gate_zero_then_t1_then_p50() -> None:
    edges = process_v2_dependency_edges()
    assert edges[T1_PANEL_POLICY]["gate_zero_structural"] == GATE_ZERO_STRUCTURAL
    assert edges[T1_PANEL_POLICY]["source_requirements"] == EDITING_CORPUS_V2_CONTRACT
    assert edges[T1_CAPACITY_POLICY]["gate_zero_structural"] == GATE_ZERO_STRUCTURAL
    assert edges[T1_CAPACITY_POLICY]["t1_panel_policy"] == T1_PANEL_POLICY
    assert edges[P50_RECIPE_POLICY]["t1_capacity_policy"] == T1_CAPACITY_POLICY
    assert edges[P50_RECIPE_POLICY]["gate_zero_structural"] == GATE_ZERO_STRUCTURAL


def test_p50_transitively_depends_on_everything_it_consumes() -> None:
    reachable = process_v2_transitive_dependencies(P50_RECIPE_POLICY)
    assert {
        ACTIVE8_DECISION_RUNTIME,
        CAPABILITY_CELLS,
        DEVELOPMENT_CELL_ROLES,
        GATE_ZERO_STRUCTURAL,
        T1_PANEL_POLICY,
        T1_CAPACITY_POLICY,
        EDITING_CORPUS_V2_CONTRACT,
    } <= reachable
    assert P50_RECIPE_POLICY not in reachable, "an artifact must not depend on itself"


def test_no_artifact_points_at_a_later_stage() -> None:
    """An earlier artifact is never made mutable to reference a later result."""

    order = {name: index for index, name in enumerate(PROCESS_V2_CHAIN_ARTIFACTS)}
    for name, edges in process_v2_dependency_edges().items():
        for role, target in edges.items():
            if target in order:
                assert order[target] < order[name], f"{name}:{role} -> {target}"


def test_the_added_edges_are_present_and_were_absent_in_schema_version_one() -> None:
    if not _revision_available(_SUPERSEDED_CHAIN_REVISION):
        pytest.skip(f"{_SUPERSEDED_CHAIN_REVISION} is not reachable from this checkout")
    for name, roles in DEPENDENCY_EDGES_ADDED_IN_SCHEMA_VERSION_2.items():
        superseded = json.loads(_git_show(_SUPERSEDED_CHAIN_REVISION, name) or b"{}")
        assert superseded["schema_version"] == 1
        current = _load(name)
        for role in roles:
            assert role in current["parents"], f"{name}:{role}"
            assert role not in superseded["parents"], (
                f"{name}:{role} is declared as newly added but version 1 already had it"
            )


def test_declaring_a_dependency_edge_required_a_new_schema_version() -> None:
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        payload = _load(name)
        assert payload["schema_version"] == CHAIN_SCHEMA_VERSION == 3
        assert payload["contract_revision"] == CHAIN_CONTRACT_REVISION


def test_a_missing_dependency_edge_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(P50_RECIPE_POLICY, repo_root=_ROOT)
    payload["parents"] = {
        role: edge
        for role, edge in payload["parents"].items()
        if role != "t1_capacity_policy"
    }
    with pytest.raises(ProcessV2ChainError, match="declared dependency edges differ"):
        validate_process_v2_chain_artifact(payload, name=P50_RECIPE_POLICY, repo_root=_ROOT)


# ---- (f) superseded design lineage ----


def test_every_artifact_preserves_every_superseded_generation_as_lineage() -> None:
    """Both generations, recovered from git rather than restated here.

    Version 2 recorded only version 1.  Had version 3 kept that shape it would
    have dropped version 1 in order to record version 2, so the chain could no
    longer say what a hash from two generations ago addressed.
    """

    unreachable = [
        revision
        for _version, revision, _named in _SUPERSEDED_GENERATION_SOURCES
        if not _revision_available(revision)
    ]
    if unreachable:
        pytest.skip(f"{unreachable} not reachable from this checkout")
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        lineage = _load(name)["superseded_design_lineage"]
        assert isinstance(lineage, list)
        assert len(lineage) == len(_SUPERSEDED_GENERATION_SOURCES), name
        for entry, (version, revision, named) in zip(
            lineage, _SUPERSEDED_GENERATION_SOURCES, strict=True
        ):
            raw = _git_show(revision, name)
            assert raw is not None, (name, revision)
            assert entry["schema_version"] == version, name
            assert entry["contract_revision"] == named, name
            assert entry["physical"]["sha256"] == hashlib.sha256(raw).hexdigest(), name
            assert entry["semantic"]["sha256"] == json.loads(raw)[SELF_HASH_FIELD], name
            for slot in ("physical", "semantic"):
                assert entry[slot]["kind"] == PointerKind.LINEAGE_REFERENCE
                assert entry[slot]["target"] == name


def test_the_lineage_is_ordered_oldest_first_and_precedes_the_live_version() -> None:
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        versions = [entry["schema_version"] for entry in _load(name)["superseded_design_lineage"]]
        assert versions == sorted(versions), name
        assert len(set(versions)) == len(versions), name
        assert max(versions) < CHAIN_SCHEMA_VERSION, name


def test_a_lineage_pointer_that_is_not_a_lineage_kind_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=_ROOT)
    payload["superseded_design_lineage"][0]["semantic"]["kind"] = PointerKind.REPOSITORY_CONFIG
    with pytest.raises(ProcessV2ChainError, match="must carry no currency claim"):
        validate_process_v2_chain_artifact(payload, name=CAPABILITY_CELLS, repo_root=_ROOT)


def test_a_lineage_hash_equal_to_the_live_self_hash_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=_ROOT)
    payload["superseded_design_lineage"][-1]["semantic"]["sha256"] = payload[SELF_HASH_FIELD]
    with pytest.raises(ProcessV2ChainError, match="so it is not superseded"):
        validate_process_v2_chain_artifact(payload, name=CAPABILITY_CELLS, repo_root=_ROOT)


def test_no_lineage_physical_hash_check_is_claimed_because_it_is_unfalsifiable() -> None:
    """A negative result, pinned so it is not "fixed" back in.

    A "lineage physical equals the live file hash" guard reads as the obvious
    companion to the semantic one, and it was written and then removed. It cannot
    be exercised: the pointer is part of the body that IS the file, so writing the
    live physical hash into it changes the live physical hash, and a validator
    recomputing the expectation from the payload under test compares the mutation
    against itself. What actually refuses a physical lie is the target check.
    """

    payload = build_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=_ROOT)
    live_physical_before = hashlib.sha256(
        serialize_process_v2_chain_artifact(payload)
    ).hexdigest()
    payload["superseded_design_lineage"][0]["physical"]["sha256"] = live_physical_before
    live_physical_after = hashlib.sha256(
        serialize_process_v2_chain_artifact(payload)
    ).hexdigest()
    assert live_physical_after != live_physical_before, (
        "the fixed point is unreachable, which is why no such guard is claimed"
    )
    # It still fails, on the deterministic rebuild, which is the honest reason.
    with pytest.raises(ProcessV2ChainError, match="differs from the deterministic rebuild"):
        validate_process_v2_chain_artifact(payload, name=CAPABILITY_CELLS, repo_root=_ROOT)


def test_a_duplicated_lineage_generation_is_rejected() -> None:
    """Two generations that hash alike are one generation written twice."""

    payload = build_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=_ROOT)
    lineage = payload["superseded_design_lineage"]
    lineage[1]["physical"]["sha256"] = lineage[0]["physical"]["sha256"]
    with pytest.raises(ProcessV2ChainError, match="already recorded at"):
        validate_process_v2_chain_artifact(payload, name=CAPABILITY_CELLS, repo_root=_ROOT)


def test_an_out_of_order_or_repeated_lineage_version_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=_ROOT)
    payload["superseded_design_lineage"] = list(
        reversed(payload["superseded_design_lineage"])
    )
    with pytest.raises(ProcessV2ChainError, match="does not follow"):
        validate_process_v2_chain_artifact(payload, name=CAPABILITY_CELLS, repo_root=_ROOT)

    payload = build_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=_ROOT)
    payload["superseded_design_lineage"][1]["schema_version"] = 1
    with pytest.raises(ProcessV2ChainError, match="does not follow"):
        validate_process_v2_chain_artifact(payload, name=CAPABILITY_CELLS, repo_root=_ROOT)


def test_a_lineage_entry_naming_the_live_revision_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=_ROOT)
    payload["superseded_design_lineage"][-1]["contract_revision"] = CHAIN_CONTRACT_REVISION
    with pytest.raises(ProcessV2ChainError, match="is the live revision"):
        validate_process_v2_chain_artifact(payload, name=CAPABILITY_CELLS, repo_root=_ROOT)


def test_a_dropped_lineage_generation_is_rejected() -> None:
    """The specific regression: republishing while keeping only the last shape."""

    payload = build_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=_ROOT)
    payload["superseded_design_lineage"] = payload["superseded_design_lineage"][-1:]
    with pytest.raises(ProcessV2ChainError, match="records 1 generations"):
        validate_process_v2_chain_artifact(payload, name=CAPABILITY_CELLS, repo_root=_ROOT)


def test_a_lineage_entry_targeting_another_artifact_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=_ROOT)
    payload["superseded_design_lineage"][0]["semantic"]["target"] = GATE_ZERO_STRUCTURAL
    with pytest.raises(ProcessV2ChainError, match="records its OWN superseded hashes"):
        validate_process_v2_chain_artifact(payload, name=CAPABILITY_CELLS, repo_root=_ROOT)


# ---- (g) a V1 identity pin is rejected ----


def test_v1_identity_pin_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(GATE_ZERO_STRUCTURAL, repo_root=_ROOT)
    payload["process_identity"] = dict(payload["process_identity"])
    payload["process_identity"]["process_identity_sha256"] = str(
        editing_v2_process_identity()["process_identity_sha256"]
    )
    with pytest.raises(ProcessV2ChainError, match="binds the V1 semantic process identity"):
        validate_process_v2_chain_artifact(
            payload, name=GATE_ZERO_STRUCTURAL, repo_root=_ROOT
        )


def test_an_unknown_identity_pin_is_rejected_distinctly() -> None:
    payload = build_process_v2_chain_artifact(T1_PANEL_POLICY, repo_root=_ROOT)
    payload["process_identity"] = dict(payload["process_identity"])
    payload["process_identity"]["process_identity_sha256"] = "0" * 64
    with pytest.raises(ProcessV2ChainError) as raised:
        validate_process_v2_chain_artifact(payload, name=T1_PANEL_POLICY, repo_root=_ROOT)
    message = str(raised.value)
    assert "which is not the live Process-V2 identity" in message
    assert "V1 semantic process identity" not in message


def test_a_relabelled_v1_identity_declaration_is_rejected() -> None:
    """Making the declaration match the V1 value is not a repair."""

    v1 = editing_v2_process_identity()
    payload = build_process_v2_chain_artifact(T1_PANEL_POLICY, repo_root=_ROOT)
    payload["process_identity"] = {
        "identity_schema": str(v1["schema"]),
        "identity_schema_version": int(v1["schema_version"]),
        "module": "src/compose_v4/rewrite/editing_v2_process_identity.py",
        "process_identity_sha256": str(v1["process_identity_sha256"]),
        "process_semantics": str(v1["process_semantics"]),
        "provider": "editing_v2_process_identity",
    }
    with pytest.raises(ProcessV2ChainError, match="resolves only"):
        validate_process_v2_chain_artifact(payload, name=T1_PANEL_POLICY, repo_root=_ROOT)


# ---- (h) a _v1 parent path is rejected ----


def test_v1_parent_path_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(T1_PANEL_POLICY, repo_root=_ROOT)
    v1_path = "configs/editing_v2_semantic_development_cell_roles_v1.json"
    v1_roles = _load(v1_path)
    payload["parents"]["development_cell_roles"] = {
        "physical": {
            **payload["parents"]["development_cell_roles"]["physical"],
            "target": v1_path,
            "sha256": _physical(v1_path),
        },
        "semantic": {
            **payload["parents"]["development_cell_roles"]["semantic"],
            "target": v1_path,
            "sha256": v1_roles["policy_sha256"],
        },
    }
    with pytest.raises(ProcessV2ChainError, match="must never bind a _v1 config"):
        validate_process_v2_chain_artifact(payload, name=T1_PANEL_POLICY, repo_root=_ROOT)


def test_a_stale_parent_pin_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(T1_CAPACITY_POLICY, repo_root=_ROOT)
    payload["parents"] = json.loads(json.dumps(payload["parents"]))
    payload["parents"]["t1_panel_policy"]["semantic"]["sha256"] = "1" * 64
    with pytest.raises(ProcessV2ChainError, match="but the live value is"):
        validate_process_v2_chain_artifact(
            payload, name=T1_CAPACITY_POLICY, repo_root=_ROOT
        )


def test_a_malformed_edge_pointer_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(T1_CAPACITY_POLICY, repo_root=_ROOT)
    payload["parents"]["t1_panel_policy"]["semantic"] = {"path": T1_PANEL_POLICY}
    with pytest.raises(ProcessV2ChainError, match="not a typed pointer"):
        validate_process_v2_chain_artifact(
            payload, name=T1_CAPACITY_POLICY, repo_root=_ROOT
        )


# ---- (i) authority flags ----


def test_every_authority_flag_is_false_in_all_seven_artifacts() -> None:
    assert AUTHORITY_FIELDS == (
        "training_authorized",
        "gate_zero_authorized",
        "t1_authorized",
        "bounded_p50_authorized",
        "long_training_authorized",
        "checkpoint_selection_authorized",
        "final_test_selection_authorized",
    )
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        payload = _load(name)
        for field in AUTHORITY_FIELDS:
            assert payload[field] is False, f"{name}:{field}"
        assert payload["status"].endswith("_NO_DOWNSTREAM_AUTHORITY"), name
        assert "p50_authorized" not in payload, name


def test_a_granted_authority_flag_is_rejected() -> None:
    for field in AUTHORITY_FIELDS:
        payload = build_process_v2_chain_artifact(P50_RECIPE_POLICY, repo_root=_ROOT)
        payload[field] = True
        with pytest.raises(ProcessV2ChainError, match="grants authority"):
            validate_process_v2_chain_artifact(
                payload, name=P50_RECIPE_POLICY, repo_root=_ROOT
            )


def test_a_grant_under_a_name_outside_the_vocabulary_is_rejected() -> None:
    """The P50 recipe body carries `p500_authorized`, which AUTHORITY_FIELDS omits.

    The vocabulary-aware guard can only judge names it knows, so without the
    vocabulary-free one a contract shipping this field granted would validate.
    Its absence from the vocabulary is correct: it belongs to stage decisions and
    permits, where a true value is legitimate. In a prospective contract it is a
    policy field and must be false.
    """

    payload = build_process_v2_chain_artifact(P50_RECIPE_POLICY, repo_root=_ROOT)
    assert payload["p500_authorized"] is False
    assert "p500_authorized" not in AUTHORITY_FIELDS
    payload["p500_authorized"] = True
    with pytest.raises(ProcessV2ChainError, match="grants authority"):
        validate_process_v2_chain_artifact(
            payload, name=P50_RECIPE_POLICY, repo_root=_ROOT
        )


def test_the_retired_authority_spelling_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(P50_RECIPE_POLICY, repo_root=_ROOT)
    payload["p50_authorized"] = False
    with pytest.raises(ProcessV2ChainError, match="field set differs"):
        validate_process_v2_chain_artifact(payload, name=P50_RECIPE_POLICY, repo_root=_ROOT)


def test_gate_zero_structural_grants_no_authority_on_pass() -> None:
    decision = _load(GATE_ZERO_STRUCTURAL)["decision_policy"]
    granted = {key: value for key, value in decision.items() if key.startswith("pass_grants_")}
    assert granted and all(value is False for value in granted.values())


# ---- (j) Active8 order ----


def test_active8_order_matches_the_corpus_contract_constant() -> None:
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        assert tuple(_load(name)["active_families"]) == tuple(ACTIVE8_FAMILIES), name
    assert tuple(_load(T1_CAPACITY_POLICY)["required_families"]) == tuple(ACTIVE8_FAMILIES)


def test_a_reordered_active8_list_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=_ROOT)
    payload["active_families"] = list(reversed(ACTIVE8_FAMILIES))
    with pytest.raises(ProcessV2ChainError, match="differs from the Active8"):
        validate_process_v2_chain_artifact(payload, name=CAPABILITY_CELLS, repo_root=_ROOT)


# ---- (k) the shared policy registry ----


def test_every_artifact_pins_the_shared_policy_registry() -> None:
    live = policy_registry_identity(repo_root=_ROOT)
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        pin = _load(name)["shared_policy_registry"]
        assert pin["registry_identity_sha256"] == live["registry_identity_sha256"], name
        assert pin["schema"] == live["schema"]
        assert pin["schema_version"] == live["schema_version"]
        assert "frozen_sources" not in pin, (
            "the registry body must be pinned, not copied into every artifact"
        )


def test_a_stale_registry_pin_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=_ROOT)
    payload["shared_policy_registry"]["registry_identity_sha256"] = "3" * 64
    with pytest.raises(ProcessV2ChainError, match="but the live registry identity is"):
        validate_process_v2_chain_artifact(payload, name=CAPABILITY_CELLS, repo_root=_ROOT)


def test_an_edited_frozen_policy_source_makes_the_chain_unbuildable() -> None:
    """A frozen contract cannot be edited and laundered through the projection."""

    with _isolated_repo() as root:
        write_process_v2_chain(root)
        target = root / FROZEN_POLICY_SOURCES["capability_cells"].relative_path
        target.write_bytes(target.read_bytes() + b"\n")
        with pytest.raises(ProcessV2ChainError, match="frozen policy source"):
            build_process_v2_chain_artifact(CAPABILITY_CELLS, repo_root=root)


# ---- (l) remaining loud-failure modes ----


def test_a_disagreeing_self_hash_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(DEVELOPMENT_CELL_ROLES, repo_root=_ROOT)
    payload["contract_sha256"] = "2" * 64
    with pytest.raises(ProcessV2ChainError, match="disagrees"):
        validate_process_v2_chain_artifact(
            payload, name=DEVELOPMENT_CELL_ROLES, repo_root=_ROOT
        )


def test_a_changed_field_set_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(T1_PANEL_POLICY, repo_root=_ROOT)
    payload["extra_policy_knob"] = 1
    with pytest.raises(ProcessV2ChainError, match="field set differs"):
        validate_process_v2_chain_artifact(payload, name=T1_PANEL_POLICY, repo_root=_ROOT)

    payload = build_process_v2_chain_artifact(T1_PANEL_POLICY, repo_root=_ROOT)
    del payload["support_time_hex"]
    with pytest.raises(ProcessV2ChainError, match="field set differs"):
        validate_process_v2_chain_artifact(payload, name=T1_PANEL_POLICY, repo_root=_ROOT)


def test_a_silently_moved_policy_value_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(T1_CAPACITY_POLICY, repo_root=_ROOT)
    payload["thresholds"] = dict(payload["thresholds"])
    payload["thresholds"]["minimum_unique_state_teacher_successor_probability"] = 0.5
    with pytest.raises(ProcessV2ChainError, match="mirrored policy values are frozen"):
        validate_process_v2_chain_artifact(
            payload, name=T1_CAPACITY_POLICY, repo_root=_ROOT
        )


def test_a_wrong_schema_version_is_rejected() -> None:
    payload = build_process_v2_chain_artifact(T1_PANEL_POLICY, repo_root=_ROOT)
    payload["schema_version"] = 1
    with pytest.raises(ProcessV2ChainError, match="adding dependency edges required"):
        validate_process_v2_chain_artifact(payload, name=T1_PANEL_POLICY, repo_root=_ROOT)


def test_an_unknown_artifact_name_is_rejected() -> None:
    with pytest.raises(ProcessV2ChainError, match="is not a Process-V2 chain artifact"):
        build_process_v2_chain_artifact("configs/nope.json", repo_root=_ROOT)


def test_a_non_object_artifact_is_rejected() -> None:
    with pytest.raises(ProcessV2ChainError, match="must be a JSON object"):
        validate_process_v2_chain_artifact([], name=T1_PANEL_POLICY, repo_root=_ROOT)


# ---- (m) transactional publication ----


def test_publication_writes_one_content_addressed_generation() -> None:
    with _isolated_repo() as root:
        generations = root / "generations"
        generation = publish_process_v2_chain(root, generations_root=generations)
        assert generation.parent == generations
        sealed = build_process_v2_chain(root)
        assert generation.name == process_v2_generation_id(sealed)
        assert read_committed_generation(generation) == sealed
        assert committed_generations(generations) == [generation]


def test_the_marker_is_published_last_and_a_staged_generation_is_invisible() -> None:
    """An interruption cannot expose a partial generation as authoritative.

    Staging without committing is exactly the state a process killed before the
    marker leaves behind, so no failure has to be injected to reach it.
    """

    with _isolated_repo() as root:
        generations = root / "generations"
        generation = stage_process_v2_chain_generation(root, generations_root=generations)
        assert not (generation / GENERATION_COMMITTED_MARKER).exists()
        for name in PROCESS_V2_CHAIN_ARTIFACTS:
            assert (generation / name).is_file()
        with pytest.raises(ProcessV2ChainError, match="not committed"):
            read_committed_generation(generation)
        assert committed_generations(generations) == []
        with pytest.raises(ProcessV2ChainError, match="not committed"):
            materialize_committed_generation(generation, repo_root=root)
        for name in PROCESS_V2_CHAIN_ARTIFACTS:
            assert not (root / name).exists(), (
                "an uncommitted generation must never reach the canonical paths"
            )

        commit_process_v2_chain_generation(generation)
        assert committed_generations(generations) == [generation]
        materialize_committed_generation(generation, repo_root=root)
        for name in PROCESS_V2_CHAIN_ARTIFACTS:
            assert (root / name).is_file()


def test_an_incomplete_generation_cannot_be_committed() -> None:
    with _isolated_repo() as root:
        generations = root / "generations"
        generation = stage_process_v2_chain_generation(root, generations_root=generations)
        (generation / P50_RECIPE_POLICY).unlink()
        with pytest.raises(ProcessV2ChainError, match="is incomplete"):
            commit_process_v2_chain_generation(generation)


def test_a_tampered_committed_generation_is_refused() -> None:
    with _isolated_repo() as root:
        generations = root / "generations"
        generation = publish_process_v2_chain(root, generations_root=generations)
        target = generation / T1_PANEL_POLICY
        target.write_bytes(target.read_bytes() + b"\n")
        with pytest.raises(ProcessV2ChainError, match="does not match"):
            read_committed_generation(generation)
        assert committed_generations(generations) == []


def test_a_marker_that_grants_authority_is_refused() -> None:
    with _isolated_repo() as root:
        generations = root / "generations"
        generation = publish_process_v2_chain(root, generations_root=generations)
        marker_path = generation / GENERATION_COMMITTED_MARKER
        marker = json.loads(marker_path.read_bytes())
        marker["training_authorized"] = True
        del marker[GENERATION_SELF_HASH_FIELD]
        body = json.dumps(marker, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        marker[GENERATION_SELF_HASH_FIELD] = hashlib.sha256(body.encode()).hexdigest()
        marker_path.write_bytes(json.dumps(marker, indent=2, sort_keys=True).encode())
        with pytest.raises(ProcessV2ChainError, match="grants authority"):
            read_committed_generation(generation)


def test_the_marker_declares_the_frozen_generation_schema_and_no_authority() -> None:
    with _isolated_repo() as root:
        generation = publish_process_v2_chain(root, generations_root=root / "generations")
        marker = json.loads((generation / GENERATION_COMMITTED_MARKER).read_bytes())
        assert marker["schema"] == GENERATION_COMMITTED_SCHEMA
        for field in AUTHORITY_FIELDS:
            assert marker[field] is False, field
        assert set(marker["artifacts"]) == set(PROCESS_V2_CHAIN_ARTIFACTS)


def test_a_failed_publication_leaves_the_canonical_paths_untouched() -> None:
    with _isolated_repo() as root:
        write_process_v2_chain(root)
        before = {n: (root / n).read_bytes() for n in PROCESS_V2_CHAIN_ARTIFACTS}
        target = root / FROZEN_POLICY_SOURCES["t1_capacity_policy"].relative_path
        target.write_bytes(target.read_bytes() + b"\n")
        with pytest.raises(ProcessV2ChainError):
            write_process_v2_chain(root)
        after = {n: (root / n).read_bytes() for n in PROCESS_V2_CHAIN_ARTIFACTS}
        assert before == after


# ---- (n) the frozen V1 chain is untouched ----


def test_no_v1_config_changed_since_the_frozen_base_revision() -> None:
    if not _revision_available(_FROZEN_V1_BASE_REVISION):
        pytest.skip(f"{_FROZEN_V1_BASE_REVISION} is not reachable from this checkout")
    paths = _v1_config_paths()
    assert len(paths) >= 7, "the V1 chain configs must be present to be checked"
    for relative_path in paths:
        frozen = _git_show(_FROZEN_V1_BASE_REVISION, relative_path)
        assert frozen is not None, f"{relative_path} is absent at the frozen base"
        assert (_ROOT / relative_path).read_bytes() == frozen, (
            f"{relative_path} is a frozen V1 artifact and must not be modified"
        )


def test_every_mirrored_v1_counterpart_is_present_and_distinct() -> None:
    for name, counterpart in _V1_COUNTERPART.items():
        assert (_ROOT / counterpart).is_file()
        assert _physical(name) != _physical(counterpart)


# ---- (o) the mirror really is a mirror ----


def test_capability_cell_definitions_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[CAPABILITY_CELLS])
    v2 = _load(CAPABILITY_CELLS)
    assert v2["family_contexts"] == v1["family_contexts"]
    assert v2["cell_identity_policy"] == v1["cell_identity_policy"]
    assert v2["cell_identity_policy"]["namespace"] == "editing_v2_active8_v1"
    assert v2["exact_evidence_strata"] == v1["exact_evidence_strata"]
    assert v2["fail_closed_policy"] == v1["fail_closed_policy"]
    assert v2["scope"] == v1["scope"]
    assert v2["bindings"]["data_lanes"] == v1["bindings"]["data_lanes"]
    assert v2["bindings"]["partition_roles"] == v1["bindings"]["partition_roles"]
    assert (
        v2["bindings"]["action_codec_schema_version"]
        == v1["bindings"]["action_codec_schema_version"]
    )


def test_development_cell_roles_and_counts_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[DEVELOPMENT_CELL_ROLES])
    v2 = _load(DEVELOPMENT_CELL_ROLES)
    for field in (
        "required_cell_ids",
        "conditional_cell_ids",
        "separate_lane_cell_ids",
        "conditional_policy",
        "separate_lane_policy",
        "partition_policy",
        "scope",
    ):
        assert v2[field] == v1[field], field
    assert v2["partition_policy"]["required_cell_count"] == 17
    assert v2["partition_policy"]["conditional_cell_count"] == 3
    assert v2["partition_policy"]["separate_lane_cell_count"] == 2
    assert len(v2["required_cell_ids"]) == 17
    assert len(v2["conditional_cell_ids"]) == 3
    assert len(v2["separate_lane_cell_ids"]) == 2


def test_decision_runtime_model_and_software_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[ACTIVE8_DECISION_RUNTIME])
    v2 = _load(ACTIVE8_DECISION_RUNTIME)
    assert v2["software"] == v1["software"]
    # The ONE intended difference: the V2 operator capability expectation, read
    # from the Gate-0 Process-V2 model/process contract this runtime pins.
    assert set(v2["model"]) - set(v1["model"]) == {"operator_capability_fingerprint"}
    assert {k: v for k, v in v2["model"].items() if k in v1["model"]} == v1["model"]
    assert v2["model"]["operator_capability_fingerprint"] == (
        _load("configs/editing_gate_zero_semantic_model_process_v2.json")["model_identity"][
            "operator_capability_fingerprint"
        ]
    )
    assert v2["model"]["operator_capability_fingerprint"] == "d79ffe8ef65f3fb3"


def test_gate_zero_structural_blocks_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[GATE_ZERO_STRUCTURAL])
    v2 = _load(GATE_ZERO_STRUCTURAL)
    for field in ("required_architecture", "structural_checks", "decision_policy"):
        assert v2[field] == v1[field], field


def test_t1_panel_policy_thresholds_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[T1_PANEL_POLICY])
    v2 = _load(T1_PANEL_POLICY)
    for field in (
        "maximum_entries_by_family",
        "minimum_entries_by_family",
        "objective_unit",
        "panel_kind",
        "support_time_hex",
        "cache_handoff",
        "hazard_included",
        "gate_thresholds_included",
        "optimizer_policy_included",
        "p50_policy_included",
        "repeated_state_panel_included",
        "successor_fiber_cache_compiled",
        "empirical_multiplicity_receipts_included",
    ):
        assert v2[field] == v1[field], field


def test_t1_capacity_policy_blocks_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[T1_CAPACITY_POLICY])
    v2 = _load(T1_CAPACITY_POLICY)
    for field in (
        "thresholds",
        "panel_cardinality",
        "sampling_law",
        "optimization",
        "required_families",
        "empirical_repeated_state_gate",
        "objective_unit",
        "panel_kind",
        "support_time_hex",
        "hazard_included",
    ):
        assert v2[field] == v1[field], field


def test_p50_recipe_policy_blocks_are_identical_to_v1() -> None:
    v1 = _load(_V1_COUNTERPART[P50_RECIPE_POLICY])
    v2 = _load(P50_RECIPE_POLICY)
    for field in (
        "optimization",
        "objective",
        "time_derivation",
        "sampling",
        "cache",
        "thresholds",
        "active_families",
        "required_physical_binding_purposes",
        "scientific_scope",
        "p500_authorized",
    ):
        assert v2[field] == v1[field], field


_ENVELOPE_FIELDS = {
    "admitted_source",
    "contract_revision",
    "parents",
    "resolved_evidence_binding",
    "schema_version",
    "shared_policy_registry",
    "superseded_design_lineage",
    SELF_HASH_FIELD,
}


def test_no_policy_value_moved_across_any_schema_version() -> None:
    """Every version bump moved the envelope only. No mirrored policy value.

    Checked against BOTH superseded generations rather than only the oldest: a
    value that moved at version 2 and moved back at version 3 would agree with
    version 1 and still have been unstable, and a value that moved only at
    version 3 is invisible in a version-1 comparison whose field set already
    differs.
    """

    for _version, revision, _named in _SUPERSEDED_GENERATION_SOURCES:
        if not _revision_available(revision):
            pytest.skip(f"{revision} is not reachable from this checkout")
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        current = _load(name)
        for version, revision, _named in _SUPERSEDED_GENERATION_SOURCES:
            superseded = json.loads(_git_show(revision, name) or b"{}")
            assert superseded["schema_version"] == version, (name, revision)
            for field in sorted(set(current) & set(superseded) - _ENVELOPE_FIELDS):
                assert current[field] == superseded[field], f"{name}.{field}@{revision}"


def test_the_field_set_changed_only_where_a_version_bump_says_it_did() -> None:
    """Version 1 -> 2 removed the dead slot and added three envelope fields.
    Version 2 -> 3 changed no field at all: only their values moved."""

    if not _revision_available(_SUPERSEDED_CHAIN_REVISION):
        pytest.skip(f"{_SUPERSEDED_CHAIN_REVISION} is not reachable from this checkout")
    if not _revision_available(_SUPERSEDED_CHAIN_REVISION_V2):
        pytest.skip(f"{_SUPERSEDED_CHAIN_REVISION_V2} is not reachable from this checkout")
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        current = _load(name)
        version_one = json.loads(_git_show(_SUPERSEDED_CHAIN_REVISION, name) or b"{}")
        version_two = json.loads(_git_show(_SUPERSEDED_CHAIN_REVISION_V2, name) or b"{}")
        assert set(version_one) - set(current) == {"admitted_source"}, name
        assert set(current) - set(version_one) == {
            "contract_revision",
            "resolved_evidence_binding",
            "shared_policy_registry",
            "superseded_design_lineage",
        }, name
        assert set(current) == set(version_two), name


def test_version_three_changed_exactly_the_binding_version_and_the_lineage() -> None:
    """The substantive delta, stated so a reviewer need not diff two generations."""

    if not _revision_available(_SUPERSEDED_CHAIN_REVISION_V2):
        pytest.skip(f"{_SUPERSEDED_CHAIN_REVISION_V2} is not reachable from this checkout")
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        current = _load(name)
        previous = json.loads(_git_show(_SUPERSEDED_CHAIN_REVISION_V2, name) or b"{}")
        moved = {
            field
            for field in set(current) | set(previous)
            if current.get(field) != previous.get(field)
        }
        # `parents` moves for any artifact with a chain-member parent, because a
        # child pins its parent's FINAL bytes and every chain member was
        # regenerated. It is checked below rather than excluded.
        assert moved <= {
            "contract_revision",
            "parents",
            "resolved_evidence_binding",
            "schema_version",
            "superseded_design_lineage",
            SELF_HASH_FIELD,
        }, (name, sorted(moved))
        assert moved >= {
            "contract_revision",
            "resolved_evidence_binding",
            "schema_version",
            "superseded_design_lineage",
            SELF_HASH_FIELD,
        }, (name, sorted(moved))
        assert previous["resolved_evidence_binding"]["cited_by_schema_version"] == 1
        assert current["resolved_evidence_binding"]["cited_by_schema_version"] == 2

        # The GRAPH did not move: same roles, same targets. Only the pinned hash
        # of a regenerated chain-member parent may differ, and an external parent
        # must be pinned identically because none of them was touched.
        assert set(current["parents"]) == set(previous["parents"]), name
        for role, edge in sorted(current["parents"].items()):
            was = previous["parents"][role]
            for slot in sorted(edge):
                assert edge[slot]["target"] == was[slot]["target"], f"{name}:{role}"
                if edge[slot]["target"] in PROCESS_V2_CHAIN_ARTIFACTS:
                    assert edge[slot]["sha256"] != was[slot]["sha256"], f"{name}:{role}"
                else:
                    assert edge[slot] == was[slot], f"{name}:{role}.{slot}"


# ---- The Gate-0 consumption boundary ----


def test_gate_zero_cannot_yet_consume_the_process_v2_structural_contract() -> None:
    """The V2 Gate-0 contract exists and validates; no runner consumes it yet.

    This is a boundary, not a defect, and it is pinned so it cannot be mistaken
    for working wiring. `load_semantic_gate_zero_structural_contract` hard-binds
    `contract_id == "editing_v2_semantic_gate_zero_structural_v1"` and
    `FROZEN_CONTRACT_SHA256`, and enforces an exact V1 field set, so it refuses
    the V2 contract by construction.

    Running Gate 0 under Process V2 therefore needs a V2 runner that consumes
    this contract. Teaching the V1 loader to accept a structurally different V2
    body would entangle the two chains, which is precisely what keeping them
    distinct is for. If someone later adds that runner, this test should be
    replaced by one that exercises it -- not deleted.
    """

    from compose_v4.experiments.editing_v2_semantic_gate_zero import (
        SemanticGateZeroStructuralError,
        load_semantic_gate_zero_structural_contract,
    )

    relative = "configs/editing_v2_process_v2_gate_zero_structural.json"

    # It is a real, self-consistent contract under its own authority.
    loaded = load_process_v2_chain_artifact(relative, repo_root=_ROOT)
    assert len(loaded["contract_sha256"]) == 64

    # And the V1 Gate-0 loader refuses it.
    with pytest.raises(SemanticGateZeroStructuralError):
        load_semantic_gate_zero_structural_contract(_ROOT / relative, repo_root=_ROOT)

    # While the V1 contract still loads unchanged through that same loader.
    assert load_semantic_gate_zero_structural_contract(repo_root=_ROOT) is not None
