"""The shared policy registry must project the frozen sources, byte for byte.

The registry replaced roughly five hundred lines of policy that had been
transcribed by hand into the Process-V2 contract module. Replacing a transcription
is only safe if the replacement is proved equivalent to what it replaced, so the
central test here compares each projection to the frozen artifact itself and to
the version-1 contract bodies sealed at ``3f3258e``. Both references are read out
of files; neither is recomputed with the registry.

The second risk is the opposite one: a registry that silently keeps projecting
after its frozen source changed would launder an edit to a frozen contract into
every artifact built from it. Every mutation below therefore asserts refusal.
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

from compose_v4.data.editing_v2_process_v2_policy_registry import (
    FROZEN_POLICY_SOURCES,
    POLICY_REGISTRY_SCHEMA,
    POLICY_REGISTRY_SCHEMA_VERSION,
    SHARED_POLICY_BLOCKS,
    SHARED_POLICY_BODIES,
    PolicyRegistryError,
    policy_registry_identity,
    project_shared_policy,
    project_shared_policy_body,
)
from compose_v4.data.editing_v2_process_v2_schema import AUTHORITY_FIELDS, canonical_bytes
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    ACTIVE8_DECISION_RUNTIME,
    CAPABILITY_CELLS,
    DEVELOPMENT_CELL_ROLES,
    GATE_ZERO_STRUCTURAL,
    P50_RECIPE_POLICY,
    T1_CAPACITY_POLICY,
    T1_PANEL_POLICY,
)

_ROOT = Path(__file__).resolve().parents[1]

#: The revision whose committed contract bodies are the version-1 reference.
_SUPERSEDED_CHAIN_REVISION = "3f3258e"

#: ``block -> (frozen V1 config, dotted path inside it)``.  Written out by hand on
#: purpose: an expectation derived from ``SHARED_POLICY_BLOCKS`` would move with the
#: code under test, which is the failure mode where a comparison cannot fail.
_EXPECTED_BLOCK_SOURCE: dict[str, tuple[str, tuple[str, ...]]] = {
    "runtime_model": (
        "configs/editing_v2_semantic_active8_decision_runtime_v1.json",
        ("model",),
    ),
    "runtime_software": (
        "configs/editing_v2_semantic_active8_decision_runtime_v1.json",
        ("software",),
    ),
    "cell_scope": ("configs/editing_v2_semantic_capability_cells_v1.json", ("scope",)),
    "cell_identity_policy": (
        "configs/editing_v2_semantic_capability_cells_v1.json",
        ("cell_identity_policy",),
    ),
    "family_contexts": (
        "configs/editing_v2_semantic_capability_cells_v1.json",
        ("family_contexts",),
    ),
    "exact_evidence_strata": (
        "configs/editing_v2_semantic_capability_cells_v1.json",
        ("exact_evidence_strata",),
    ),
    "fail_closed_policy": (
        "configs/editing_v2_semantic_capability_cells_v1.json",
        ("fail_closed_policy",),
    ),
    "data_lanes": (
        "configs/editing_v2_semantic_capability_cells_v1.json",
        ("bindings", "data_lanes"),
    ),
    "partition_roles": (
        "configs/editing_v2_semantic_capability_cells_v1.json",
        ("bindings", "partition_roles"),
    ),
    "action_codec_schema_version": (
        "configs/editing_v2_semantic_capability_cells_v1.json",
        ("bindings", "action_codec_schema_version"),
    ),
    "required_cell_ids": (
        "configs/editing_v2_semantic_development_cell_roles_v1.json",
        ("required_cell_ids",),
    ),
    "conditional_cell_ids": (
        "configs/editing_v2_semantic_development_cell_roles_v1.json",
        ("conditional_cell_ids",),
    ),
    "separate_lane_cell_ids": (
        "configs/editing_v2_semantic_development_cell_roles_v1.json",
        ("separate_lane_cell_ids",),
    ),
    "conditional_policy": (
        "configs/editing_v2_semantic_development_cell_roles_v1.json",
        ("conditional_policy",),
    ),
    "separate_lane_policy": (
        "configs/editing_v2_semantic_development_cell_roles_v1.json",
        ("separate_lane_policy",),
    ),
    "cell_role_partition_policy": (
        "configs/editing_v2_semantic_development_cell_roles_v1.json",
        ("partition_policy",),
    ),
    "cell_role_scope": (
        "configs/editing_v2_semantic_development_cell_roles_v1.json",
        ("scope",),
    ),
    "required_architecture": (
        "configs/editing_v2_semantic_gate_zero_structural_v1.json",
        ("required_architecture",),
    ),
    "structural_checks": (
        "configs/editing_v2_semantic_gate_zero_structural_v1.json",
        ("structural_checks",),
    ),
    "gate_zero_decision_policy": (
        "configs/editing_v2_semantic_gate_zero_structural_v1.json",
        ("decision_policy",),
    ),
}

#: ``body -> (frozen V1 config, superseded V2 contract that carried the same body)``.
_EXPECTED_BODY_SOURCE: dict[str, tuple[str, str]] = {
    "t1_panel_body": (
        "configs/editing_v2_semantic_t1_panel_policy_v1.json",
        T1_PANEL_POLICY,
    ),
    "t1_capacity_body": (
        "configs/editing_v2_semantic_t1_capacity_policy_v1.json",
        T1_CAPACITY_POLICY,
    ),
    "p50_recipe_body": (
        "configs/editing_v2_semantic_p50_recipe_policy_v1.json",
        P50_RECIPE_POLICY,
    ),
}

#: ``block -> the superseded V2 contract whose body carried the same value``.
_BLOCK_IN_SUPERSEDED_CONTRACT: dict[str, tuple[str, tuple[str, ...]]] = {
    "runtime_software": (ACTIVE8_DECISION_RUNTIME, ("software",)),
    "cell_scope": (CAPABILITY_CELLS, ("scope",)),
    "cell_identity_policy": (CAPABILITY_CELLS, ("cell_identity_policy",)),
    "family_contexts": (CAPABILITY_CELLS, ("family_contexts",)),
    "exact_evidence_strata": (CAPABILITY_CELLS, ("exact_evidence_strata",)),
    "fail_closed_policy": (CAPABILITY_CELLS, ("fail_closed_policy",)),
    "data_lanes": (CAPABILITY_CELLS, ("bindings", "data_lanes")),
    "partition_roles": (CAPABILITY_CELLS, ("bindings", "partition_roles")),
    "action_codec_schema_version": (
        CAPABILITY_CELLS,
        ("bindings", "action_codec_schema_version"),
    ),
    "required_cell_ids": (DEVELOPMENT_CELL_ROLES, ("required_cell_ids",)),
    "conditional_cell_ids": (DEVELOPMENT_CELL_ROLES, ("conditional_cell_ids",)),
    "separate_lane_cell_ids": (DEVELOPMENT_CELL_ROLES, ("separate_lane_cell_ids",)),
    "conditional_policy": (DEVELOPMENT_CELL_ROLES, ("conditional_policy",)),
    "separate_lane_policy": (DEVELOPMENT_CELL_ROLES, ("separate_lane_policy",)),
    "cell_role_partition_policy": (DEVELOPMENT_CELL_ROLES, ("partition_policy",)),
    "cell_role_scope": (DEVELOPMENT_CELL_ROLES, ("scope",)),
    "required_architecture": (GATE_ZERO_STRUCTURAL, ("required_architecture",)),
    "structural_checks": (GATE_ZERO_STRUCTURAL, ("structural_checks",)),
    "gate_zero_decision_policy": (GATE_ZERO_STRUCTURAL, ("decision_policy",)),
}


# ---- Helpers ----


def _resolve(payload: object, path: tuple[str, ...]) -> object:
    node = payload
    for key in path:
        assert isinstance(node, dict), path
        node = node[key]
    return node


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


@contextmanager
def _mutable_copy() -> Iterator[Path]:
    """A throwaway root holding copies of every frozen policy source."""

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        for source in FROZEN_POLICY_SOURCES.values():
            target = root / source.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_ROOT / source.relative_path, target)
        yield root


# ---- (a) the projection is byte-equivalent to the frozen source ----


def test_every_block_is_byte_equivalent_to_its_frozen_source() -> None:
    """The proof that replacing the hand copies changed no value.

    The expectation is read out of the frozen V1 contract on disk, at a path this
    test states independently, so it cannot move with the registry.
    """

    assert set(SHARED_POLICY_BLOCKS) == set(_EXPECTED_BLOCK_SOURCE)
    for block, (relative_path, path) in sorted(_EXPECTED_BLOCK_SOURCE.items()):
        frozen = json.loads((_ROOT / relative_path).read_bytes())
        expected = _resolve(frozen, path)
        projected = project_shared_policy(block, repo_root=_ROOT)
        assert canonical_bytes(projected) == canonical_bytes(expected), block


def test_every_body_is_byte_equivalent_to_its_frozen_source() -> None:
    assert set(SHARED_POLICY_BODIES) == set(_EXPECTED_BODY_SOURCE)
    for body, (relative_path, _) in sorted(_EXPECTED_BODY_SOURCE.items()):
        frozen = json.loads((_ROOT / relative_path).read_bytes())
        projected = project_shared_policy_body(body, repo_root=_ROOT)
        assert projected, body
        for name, value in sorted(projected.items()):
            assert canonical_bytes(value) == canonical_bytes(frozen[name]), f"{body}.{name}"


def test_a_projected_body_carries_no_frozen_source_envelope() -> None:
    """A body projection must not sweep the source's own schema or authority.

    The panel, capacity and recipe policies live at the root of their frozen
    artifact alongside its ``schema``, ``status`` and authority envelope.
    Projecting the whole root would carry another artifact's envelope into a new
    one.

    ``p500_authorized`` is deliberately NOT excluded: it is a recipe policy
    statement ("this P50 recipe does not authorize P500"), not a field of the
    artifact's own authority envelope, and it is absent from the frozen
    ``AUTHORITY_FIELDS`` vocabulary for exactly that reason.
    """

    forbidden = {
        "schema",
        "schema_version",
        "status",
        "policy_id",
        "policy_sha256",
        "request_id",
        "cell_role_policy_sha256",
        "panel_policy_sha256",
        *AUTHORITY_FIELDS,
    }
    for body in sorted(SHARED_POLICY_BODIES):
        projected = project_shared_policy_body(body, repo_root=_ROOT)
        assert not (set(projected) & forbidden), body
    assert "p500_authorized" not in AUTHORITY_FIELDS
    assert "p500_authorized" in project_shared_policy_body("p50_recipe_body", repo_root=_ROOT)


# ---- (b) the projection equals what the superseded contracts carried ----


def test_the_projection_reproduces_the_superseded_contract_bodies() -> None:
    """Nothing moved between contract-chain schema version 1 and version 2.

    Version 1 built these values from hand-copied constants. Reading the bodies
    that generation actually sealed is an independent reference for the same
    equality the frozen-source test proves from the other side.
    """

    if not _revision_available(_SUPERSEDED_CHAIN_REVISION):
        pytest.skip(f"{_SUPERSEDED_CHAIN_REVISION} is not reachable from this checkout")
    for block, (contract, path) in sorted(_BLOCK_IN_SUPERSEDED_CONTRACT.items()):
        raw = _git_show(_SUPERSEDED_CHAIN_REVISION, contract)
        assert raw is not None, contract
        expected = _resolve(json.loads(raw), path)
        projected = project_shared_policy(block, repo_root=_ROOT)
        assert canonical_bytes(projected) == canonical_bytes(expected), block
    for body, (_, contract) in sorted(_EXPECTED_BODY_SOURCE.items()):
        raw = _git_show(_SUPERSEDED_CHAIN_REVISION, contract)
        assert raw is not None, contract
        superseded = json.loads(raw)
        for name, value in sorted(project_shared_policy_body(body, repo_root=_ROOT).items()):
            assert canonical_bytes(value) == canonical_bytes(superseded[name]), (
                f"{body}.{name}"
            )


def test_the_runtime_model_block_is_the_v1_model_without_the_v2_delta() -> None:
    """The one intended difference stays an explicit delta, not a registry field."""

    projected = project_shared_policy("runtime_model", repo_root=_ROOT)
    assert "operator_capability_fingerprint" not in projected
    committed = json.loads((_ROOT / ACTIVE8_DECISION_RUNTIME).read_bytes())["model"]
    assert set(committed) - set(projected) == {"operator_capability_fingerprint"}
    assert {k: v for k, v in committed.items() if k in projected} == projected


# ---- (c) a moved frozen source is refused ----


def test_an_edited_frozen_source_is_refused_by_physical_hash() -> None:
    with _mutable_copy() as root:
        target = root / FROZEN_POLICY_SOURCES["capability_cells"].relative_path
        target.write_bytes(target.read_bytes() + b"\n")
        with pytest.raises(PolicyRegistryError, match="physical hash"):
            project_shared_policy("family_contexts", repo_root=root)


def test_a_resealed_frozen_source_is_refused_by_self_hash() -> None:
    """A physical pin alone cannot tell an edit from an intentional reseal.

    A body edited and correctly resealed under a NEW self-hash has a new physical
    hash too, so the physical check fires first. Editing only the declared
    self-hash leaves the physical check to fire on a different value; both must be
    reported, so the message has to name which identity moved.
    """

    with _mutable_copy() as root:
        source = FROZEN_POLICY_SOURCES["development_cell_roles"]
        target = root / source.relative_path
        payload = json.loads(target.read_bytes())
        payload[source.self_hash_field] = "0" * 64
        target.write_bytes(json.dumps(payload, indent=2, sort_keys=True).encode())
        with pytest.raises(PolicyRegistryError) as raised:
            project_shared_policy("required_cell_ids", repo_root=root)
        assert source.relative_path in str(raised.value)


def test_an_absent_frozen_source_is_refused() -> None:
    with _mutable_copy() as root:
        (root / FROZEN_POLICY_SOURCES["t1_panel_policy"].relative_path).unlink()
        with pytest.raises(PolicyRegistryError, match="not readable"):
            project_shared_policy_body("t1_panel_body", repo_root=root)


# ---- (d) the allowlist is an allowlist ----


def test_an_undeclared_block_is_refused() -> None:
    with pytest.raises(PolicyRegistryError, match="not a declared shared policy block"):
        project_shared_policy("thresholds_i_invented", repo_root=_ROOT)
    with pytest.raises(PolicyRegistryError, match="not a declared shared policy body"):
        project_shared_policy_body("some_other_body", repo_root=_ROOT)


def test_every_declared_block_names_a_declared_source() -> None:
    for block in sorted(SHARED_POLICY_BLOCKS):
        assert SHARED_POLICY_BLOCKS[block].source in FROZEN_POLICY_SOURCES, block
    for body in sorted(SHARED_POLICY_BODIES):
        assert SHARED_POLICY_BODIES[body][0] in FROZEN_POLICY_SOURCES, body


def test_every_frozen_source_is_a_prospective_contract_not_a_measured_artifact() -> None:
    """Deriving V2 policy from measured evidence is forbidden, so nothing here is."""

    for key, source in sorted(FROZEN_POLICY_SOURCES.items()):
        assert source.relative_path.startswith("configs/"), key
        assert "diagnostics/" not in source.relative_path, key
        assert "results/" not in source.relative_path, key
        assert (_ROOT / source.relative_path).is_file(), key


def test_a_projection_is_a_deep_copy() -> None:
    """A caller that mutates a projection must not poison the next read."""

    first = project_shared_policy("family_contexts", repo_root=_ROOT)
    first["atom_delete"].append("a_context_that_does_not_exist")
    second = project_shared_policy("family_contexts", repo_root=_ROOT)
    assert "a_context_that_does_not_exist" not in second["atom_delete"]


# ---- (e) identity ----


def test_the_registry_identity_pins_every_frozen_source() -> None:
    identity = policy_registry_identity(repo_root=_ROOT)
    assert identity["schema"] == POLICY_REGISTRY_SCHEMA
    assert identity["schema_version"] == POLICY_REGISTRY_SCHEMA_VERSION
    assert set(identity["frozen_sources"]) == set(FROZEN_POLICY_SOURCES)
    for key, source in sorted(FROZEN_POLICY_SOURCES.items()):
        pinned = identity["frozen_sources"][key]
        raw = (_ROOT / source.relative_path).read_bytes()
        assert pinned["file_sha256"] == hashlib.sha256(raw).hexdigest(), key
        assert pinned["semantic_sha256"] == json.loads(raw)[source.self_hash_field], key
    assert identity["blocks"] == sorted(SHARED_POLICY_BLOCKS)
    assert identity["bodies"] == sorted(SHARED_POLICY_BODIES)
    assert len(identity["registry_identity_sha256"]) == 64


def test_the_registry_identity_refuses_when_a_frozen_source_moved() -> None:
    with _mutable_copy() as root:
        target = root / FROZEN_POLICY_SOURCES["p50_recipe_policy"].relative_path
        target.write_bytes(target.read_bytes() + b" ")
        with pytest.raises(PolicyRegistryError):
            policy_registry_identity(repo_root=root)
