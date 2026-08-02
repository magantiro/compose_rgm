"""Prospective editing cell roles are exact, disjoint, and nonauthorizing."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.experiments.editing_v2_semantic_development_cell_roles import (
    SemanticDevelopmentCellRoleError,
    load_semantic_development_cell_roles,
    validate_semantic_development_cell_roles,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "editing_v2_semantic_development_cell_roles_v1.json"


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_repository_policy_freezes_exact_seventeen_three_two_partition() -> None:
    roles = load_semantic_development_cell_roles()
    assert len(roles.required_cell_ids) == 17
    assert len(roles.conditional_cell_ids) == 3
    assert len(roles.separate_lane_cell_ids) == 2
    assert not roles.required_cell_set & roles.conditional_cell_set
    assert not roles.required_cell_set & roles.separate_lane_cell_set
    assert not roles.conditional_cell_set & roles.separate_lane_cell_set
    assert (
        roles.role_for(
            "editing_v2_active8_v1:cycle_insert:close_to_articulated_polycyclic_ring_system"
        )
        == "conditional_editing"
    )
    assert (
        roles.role_for("editing_v2_active8_v1:atom_insert:root_birth")
        == "separate_lane_null_de_novo"
    )


def test_rehashed_role_promotion_fails_closed() -> None:
    payload = json.loads(POLICY.read_bytes())
    mutated = copy.deepcopy(payload)
    promoted = mutated["conditional_cell_ids"].pop()
    mutated["required_cell_ids"].append(promoted)
    body = dict(mutated)
    body.pop("policy_sha256")
    mutated["policy_sha256"] = _sha(body)
    with pytest.raises(SemanticDevelopmentCellRoleError, match="partition all 22"):
        validate_semantic_development_cell_roles(mutated, repo_root=ROOT)


def test_unknown_cell_cannot_acquire_an_implicit_role() -> None:
    roles = load_semantic_development_cell_roles()
    with pytest.raises(SemanticDevelopmentCellRoleError, match="outside"):
        roles.role_for("editing_v2_active8_v1:cycle_insert:unknown")
