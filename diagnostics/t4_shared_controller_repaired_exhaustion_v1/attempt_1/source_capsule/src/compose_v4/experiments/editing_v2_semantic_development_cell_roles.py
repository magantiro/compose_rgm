"""Prospective semantic-cell roles for bounded source-conditioned editing.

The capability-cell registry is an ontology of representable contexts.  This
policy is a separate experimental obligation: it declares which contexts must
be present and gated for the source-conditioned Gate0/T1/P50 lane, which
editing contexts remain conditional diagnostics, and which null-boundary
contexts belong to a separate timed de-novo lane.  Neither non-required role
may enter balanced editing objectives implicitly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_semantic_capability_cells import (
    REGISTRY_RELATIVE_PATH,
    SemanticCapabilityCellRegistry,
    load_semantic_capability_cell_registry,
)

POLICY_RELATIVE_PATH = "configs/editing_v2_semantic_development_cell_roles_v1.json"
SCHEMA = "compose.editing_v2.semantic_development_cell_roles"
SCHEMA_VERSION = 1
STATUS = "FROZEN_PROSPECTIVE_EDITING_CELL_ROLES_NO_DOWNSTREAM_AUTHORITY"
SCOPE = "source_conditioned_editing_gate0_t1_p50"

REQUIRED_CONTEXTS = {
    "atom_insert": ("one_neighbor_birth",),
    "atom_delete": ("leaf_death", "connected_nonleaf_death"),
    "atom_restate": ("element_identity_change", "valence_state_change"),
    "bond_reorder": ("bond_order_increase", "bond_order_decrease"),
    "bond_reroute": (
        "single_atom_pendant_acyclic_source",
        "multi_atom_pendant_acyclic_source",
        "single_atom_pendant_cyclic_source",
        "multi_atom_pendant_cyclic_source",
    ),
    "cycle_insert": (
        "close_to_monocyclic_ring_system",
        "close_to_nonarticulated_polycyclic_ring_system",
    ),
    "cycle_attach": (
        "open_from_monocyclic_ring_system",
        "open_from_nonarticulated_polycyclic_ring_system",
    ),
    "ring_system_restate": (
        "aromatization",
        "dearomatization",
    ),
}
CONDITIONAL_CONTEXTS = {
    "cycle_insert": ("close_to_articulated_polycyclic_ring_system",),
    "cycle_attach": ("open_from_articulated_polycyclic_ring_system",),
    "ring_system_restate": ("coordinated_ring_bond_state_change",),
}
SEPARATE_LANE_CONTEXTS = {
    "atom_insert": ("root_birth",),
    "atom_delete": ("singleton_to_null_death",),
}

_NO_AUTHORITY = {
    "training_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}


class SemanticDevelopmentCellRoleError(ValueError):
    """The bounded editing cell-role policy is malformed or stale."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticDevelopmentCellRoleError(
            "semantic development cell-role policy is not finite canonical JSON"
        ) from error


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise SemanticDevelopmentCellRoleError(f"cell-role parent is absent: {source}")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _cell_ids(
    registry: SemanticCapabilityCellRegistry,
    contexts: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    return tuple(
        f"{registry.namespace}:{family}:{context}"
        for family in ACTIVE8_FAMILIES
        for context in contexts.get(family, ())
    )


@dataclass(frozen=True, slots=True)
class SemanticDevelopmentCellRoles:
    """Validated stage-specific partition of every registered capability cell."""

    policy_id: str
    policy_sha256: str
    registry_file_sha256: str
    registry_sha256: str
    required_cell_ids: tuple[str, ...]
    conditional_cell_ids: tuple[str, ...]
    separate_lane_cell_ids: tuple[str, ...]
    _required_cell_set: frozenset[str] = field(init=False, repr=False, compare=False)
    _conditional_cell_set: frozenset[str] = field(init=False, repr=False, compare=False)
    _separate_lane_cell_set: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_required_cell_set", frozenset(self.required_cell_ids))
        object.__setattr__(self, "_conditional_cell_set", frozenset(self.conditional_cell_ids))
        object.__setattr__(self, "_separate_lane_cell_set", frozenset(self.separate_lane_cell_ids))

    @property
    def required_cell_set(self) -> frozenset[str]:
        return self._required_cell_set

    @property
    def conditional_cell_set(self) -> frozenset[str]:
        return self._conditional_cell_set

    @property
    def separate_lane_cell_set(self) -> frozenset[str]:
        return self._separate_lane_cell_set

    def role_for(self, cell_id: str) -> str:
        if cell_id in self.required_cell_set:
            return "required_editing"
        if cell_id in self.conditional_cell_set:
            return "conditional_editing"
        if cell_id in self.separate_lane_cell_set:
            return "separate_lane_null_de_novo"
        raise SemanticDevelopmentCellRoleError(
            f"capability cell is outside the frozen development role partition: {cell_id}"
        )


def validate_semantic_development_cell_roles(
    value: object,
    *,
    repo_root: Path | None = None,
    registry: SemanticCapabilityCellRegistry | None = None,
) -> SemanticDevelopmentCellRoles:
    """Validate exact role membership, parent bytes, scope, and no authority."""

    if not isinstance(value, Mapping):
        raise SemanticDevelopmentCellRoleError("cell-role policy must be an object")
    policy = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "policy_id",
        "status",
        *_NO_AUTHORITY,
        "scope",
        "capability_registry",
        "required_cell_ids",
        "conditional_cell_ids",
        "separate_lane_cell_ids",
        "conditional_policy",
        "separate_lane_policy",
        "partition_policy",
        "policy_sha256",
    }
    body = dict(policy)
    supplied_sha = body.pop("policy_sha256", None)
    if (
        set(policy) != expected_fields
        or policy.get("schema") != SCHEMA
        or policy.get("schema_version") != SCHEMA_VERSION
        or policy.get("policy_id") != "editing-v2-source-conditioned-cell-roles-v1"
        or policy.get("status") != STATUS
        or policy.get("scope") != SCOPE
        or supplied_sha != _sha(body)
        or any(policy.get(field) is not expected for field, expected in _NO_AUTHORITY.items())
    ):
        raise SemanticDevelopmentCellRoleError(
            "cell-role policy identity, scope, authority, or self-hash disagrees"
        )

    root = Path(repo_root or _repository_root()).resolve()
    selected_registry = registry or load_semantic_capability_cell_registry(
        root / REGISTRY_RELATIVE_PATH
    )
    registry_binding = policy.get("capability_registry")
    registry_path = (root / REGISTRY_RELATIVE_PATH).resolve()
    if (
        not isinstance(registry_binding, Mapping)
        or set(registry_binding) != {"path", "file_sha256", "registry_sha256"}
        or registry_binding.get("path") != REGISTRY_RELATIVE_PATH
        or registry_binding.get("file_sha256") != _file_sha(registry_path)
        or registry_binding.get("registry_sha256") != selected_registry.registry_sha256
    ):
        raise SemanticDevelopmentCellRoleError(
            "cell-role policy capability-registry binding disagrees"
        )

    required = policy.get("required_cell_ids")
    conditional = policy.get("conditional_cell_ids")
    separate_lane = policy.get("separate_lane_cell_ids")
    expected_required = _cell_ids(selected_registry, REQUIRED_CONTEXTS)
    expected_conditional = _cell_ids(selected_registry, CONDITIONAL_CONTEXTS)
    expected_separate_lane = _cell_ids(selected_registry, SEPARATE_LANE_CONTEXTS)
    registered = tuple(
        f"{selected_registry.namespace}:{family}:{context}"
        for family, contexts in selected_registry.family_contexts
        for context in contexts
    )
    if (
        not isinstance(required, list)
        or not isinstance(conditional, list)
        or not isinstance(separate_lane, list)
        or tuple(required) != expected_required
        or tuple(conditional) != expected_conditional
        or tuple(separate_lane) != expected_separate_lane
        or len(required) != 17
        or len(conditional) != 3
        or len(separate_lane) != 2
        or set(required).intersection(conditional)
        or set(required).intersection(separate_lane)
        or set(conditional).intersection(separate_lane)
        or set(required).union(conditional, separate_lane) != set(registered)
    ):
        raise SemanticDevelopmentCellRoleError(
            "cell-role policy must partition all 22 registered cells as 17 required, "
            "3 conditional editing, and 2 separate-lane cells"
        )
    if any(not any(f":{family}:" in cell for cell in required) for family in ACTIVE8_FAMILIES):
        raise SemanticDevelopmentCellRoleError(
            "every Active8 family must retain at least one required editing cell"
        )

    conditional_policy = policy.get("conditional_policy")
    if conditional_policy != {
        "scientific_role": "conditional_source_conditioned_editing",
        "gate_zero_nonempty_required": False,
        "included_in_unique_state_editing_t1": False,
        "included_in_balanced_editing_p50": False,
        "validate_when_present": True,
        "reported_as_audit_context": True,
    }:
        raise SemanticDevelopmentCellRoleError("conditional editing policy disagrees")
    separate_lane_policy = policy.get("separate_lane_policy")
    if separate_lane_policy != {
        "scientific_role": "separate_timed_de_novo_null_boundary",
        "editing_v2_authority": False,
        "gate_zero_nonempty_required": False,
        "included_in_unique_state_editing_t1": False,
        "included_in_balanced_editing_p50": False,
        "reported_as_audit_context": True,
    }:
        raise SemanticDevelopmentCellRoleError("separate-lane policy disagrees")
    partition_policy = policy.get("partition_policy")
    if partition_policy != {
        "registered_cells_must_be_classified_exactly_once": True,
        "required_cell_count": 17,
        "conditional_cell_count": 3,
        "separate_lane_cell_count": 2,
        "role_assignment_frozen_before_gate_zero_outputs": True,
    }:
        raise SemanticDevelopmentCellRoleError("cell-role partition policy disagrees")

    return SemanticDevelopmentCellRoles(
        policy_id=str(policy["policy_id"]),
        policy_sha256=str(supplied_sha),
        registry_file_sha256=str(registry_binding["file_sha256"]),
        registry_sha256=str(registry_binding["registry_sha256"]),
        required_cell_ids=tuple(required),
        conditional_cell_ids=tuple(conditional),
        separate_lane_cell_ids=tuple(separate_lane),
    )


def load_semantic_development_cell_roles(
    path: str | Path | None = None,
    *,
    repo_root: Path | None = None,
    registry: SemanticCapabilityCellRegistry | None = None,
) -> SemanticDevelopmentCellRoles:
    root = Path(repo_root or _repository_root()).resolve()
    source = Path(path) if path is not None else root / POLICY_RELATIVE_PATH
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticDevelopmentCellRoleError(
            f"cell-role policy is absent or invalid: {source}"
        ) from error
    if raw != _canonical_bytes(payload) + b"\n":
        raise SemanticDevelopmentCellRoleError(
            "cell-role policy must use canonical JSON bytes with one newline"
        )
    return validate_semantic_development_cell_roles(
        payload,
        repo_root=root,
        registry=registry,
    )


__all__ = [
    "CONDITIONAL_CONTEXTS",
    "POLICY_RELATIVE_PATH",
    "REQUIRED_CONTEXTS",
    "SEPARATE_LANE_CONTEXTS",
    "SemanticDevelopmentCellRoleError",
    "SemanticDevelopmentCellRoles",
    "load_semantic_development_cell_roles",
    "validate_semantic_development_cell_roles",
]
