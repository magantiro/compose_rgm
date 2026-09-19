"""Non-Cartesian within-family semantic census for exact Active8 rows.

This module derives labels only from stored persistent-slot states, stored
actions, and immutable row context.  Each axis is counted independently.  It
never forms the Cartesian product of axes and it never interprets sparse or
absent labels as a corpus, model, or training-authority decision.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
    BOND_TRIPLE,
    ELEMENTS,
    FORMAL_CHARGES,
    ORGANIC_RING_ELEMENTS,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.persistent_state_identity import (
    PERSISTENT_STATE_DIGEST_SCHEMA,
    PERSISTENT_STATE_DIGEST_VERSION,
    persistent_slot_state_sha256,
)
from compose_v4.data.active8_trace_inventory import ACTIVE8_FAMILIES
from compose_v4.rewrite.action_codec import canonical_family
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondInsert,
    BondReorder,
    BondReroute,
)
from compose_v4.rewrite.tracelets import RingSystemRestate

SEMANTIC_CENSUS_SCHEMA = "compose.experiments.editing_active8_semantic_census"
# Version 3 renames the report's bounded-P50 decision field to the one frozen
# spelling, `bounded_p50_authorized`. Both spellings named one concept and a
# consumer grepping either silently missed the other, so this is an
# incompatibility rather than an alias: a version-2 report keeps the old name and
# stays readable as the version-2 artifact it is, and is not rewritten in place.
SEMANTIC_CENSUS_SCHEMA_VERSION = 3
SEMANTIC_CENSUS_STATUS = "DIAGNOSTIC_ONLY_NOT_TRAINING_AUTHORITY"
SEMANTIC_CONTRACT_SCHEMA = "compose.experiments.editing_active8_semantic_census_contract"
SEMANTIC_CONTRACT_SCHEMA_VERSION = 2

# This physical-byte pin is filled only after the V2 contract is finalized. It
# is deliberately external to the contract's own canonical self-hash, so a
# mutated contract cannot become valid merely by recomputing its self-hash.
PINNED_SEMANTIC_CENSUS_CONTRACT_FILE_SHA256 = (
    "c0271037673a958d2436d64b4f519e90c6999399cd9c3eb2a9ee9cb3ec709dd5"
)

PINNED_T1_SUCCESSOR_GATE_V8 = {
    "path": "configs/editing_t1_successor_gate_v8.json",
    "file_sha256": "48f532b94a8396e8467763ee922e3f89f1d98110a5586a4b61b5fdd5bfc966f5",
    "contract_sha256": "7e377fb72037112f0982986fb4ea1c2ceb55ac4abc9df410e2c493715530c13e",
}
PINNED_ACTIVE8_PARENT_IDENTITY = {
    "inventory_manifest_file_sha256": (
        "bbf44483c4d14bfd6c4bde7cbd2dc4fc708c46a14acff5eadb0b6f82902832ab"
    ),
    "inventory_sha256": "ce1a1b0a43fdfeac4e5771d03a95d456ec8b2df46f207446efd4f8a17ecadd42",
    "effective_source_corpus_cache_sha256": (
        "4de860baf1906934ef01a608a366daf81789bedf8985d3ca37f9f30494add3c9"
    ),
    "support_contract_sha256": ("f3d09c84657518f5868c424ab93716d5d4b6f49da9d03abc6613c9edcff5d6d5"),
    "unified_packed_manifest_sha256": (
        "ed874884ba1406f8fa794faecd607ba8599b613f87f06475b3e5936e69866758"
    ),
}

_REAL_ELEMENTS = tuple(element for element in ELEMENTS if element not in {"null", "SCAR"})
_BOND_LABELS = {
    BOND_SINGLE: "single",
    BOND_DOUBLE: "double",
    BOND_TRIPLE: "triple",
    BOND_AROMATIC: "aromatic",
}
_ACTION_TYPES = {
    "atom_insert": AtomInsert,
    "atom_delete": AtomDelete,
    "atom_restate": AtomRestate,
    "bond_reorder": BondReorder,
    "bond_reroute": BondReroute,
    "cycle_insert": BondInsert,
    "cycle_attach": BondDelete,
    "ring_system_restate": RingSystemRestate,
}
_COMMON_AXES = (
    "data_lane",
    "trace_step_role",
    "source_atom_count",
    "source_cycle_rank",
)
_FAMILY_AXES = {
    "atom_insert": (
        "insertion_mode",
        "inserted_element",
        "inserted_formal_charge",
        "attachment_bond_class",
        "attachment_element",
    ),
    "atom_delete": (
        "deletion_mode",
        "deleted_element",
        "deleted_formal_charge",
        "deleted_heavy_degree",
        "incident_bond_signature",
    ),
    "atom_restate": (
        "element_transition",
        "formal_charge_transition",
        "implicit_h_delta",
        "implicit_h_direction",
        "site_heavy_degree",
        "site_cycle_context",
        "cyclic_target_support_class",
    ),
    "bond_reorder": (
        "bond_order_transition",
        "endpoint_element_pair",
        "endpoint_degree_pair",
        "source_edge_cycle_context",
    ),
    "bond_reroute": (
        "bond_order_transition",
        "removed_endpoint_element_pair",
        "added_endpoint_element_pair",
        "reused_removed_endpoint_count",
        "cut_component_size_pair",
        "added_endpoint_degree_pair",
        "removed_edge_cycle_context",
    ),
    "cycle_insert": (
        "closure_bond_class",
        "endpoint_element_pair",
        "endpoint_degree_pair",
        "preclosure_shortest_path_edges",
        "closed_shortest_cycle_size",
    ),
    "cycle_attach": (
        "opened_bond_class",
        "endpoint_element_pair",
        "endpoint_degree_pair",
        "remaining_shortest_path_edges",
        "opened_shortest_cycle_size",
    ),
    "ring_system_restate": (
        "changed_bond_count",
        "affected_atom_count",
        "bond_order_transition_multiset",
        "affected_element_set",
        "changed_edge_cycle_context",
        "aromatic_transition_presence",
    ),
}
_ORTHOGONAL_METRICS = (
    "raw_legal_mark_count",
    "canonical_successor_count",
    "teacher_exact_alias_count",
    "teacher_canonical_alias_count",
)


class Active8SemanticCensusError(RuntimeError):
    """The requested semantic census cannot be derived without ambiguity."""


@dataclass(frozen=True)
class OrthogonalCandidateAliasMetrics:
    """Candidate/fiber diagnostics kept separate from semantic labels."""

    raw_legal_mark_count: int
    canonical_successor_count: int
    teacher_exact_alias_count: int
    teacher_canonical_alias_count: int

    def __post_init__(self) -> None:
        for name in _ORTHOGONAL_METRICS:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in _ORTHOGONAL_METRICS}


@dataclass(frozen=True)
class SemanticObservation:
    """One exact nonterminal teacher row projected onto marginal axes."""

    family: str
    axes: Mapping[str, str]
    source_state_sha256: str
    successor_state_sha256: str
    candidate_alias: OrthogonalCandidateAliasMetrics | None = None

    def __post_init__(self) -> None:
        if self.family not in ACTIVE8_FAMILIES:
            raise ValueError(f"family {self.family!r} is not Active8")
        expected = set(_COMMON_AXES) | set(_FAMILY_AXES[self.family])
        actual = set(self.axes)
        if actual != expected:
            raise ValueError(
                f"{self.family} axes disagree with schema: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        if any(not isinstance(value, str) or not value for value in self.axes.values()):
            raise ValueError("semantic axis values must be non-empty strings")
        for name in ("source_state_sha256", "successor_state_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise Active8SemanticCensusError(f"JSON object repeats key {key!r}")
        payload[key] = value
    return payload


def _contract_self_sha256(contract: Mapping[str, object]) -> str:
    body = dict(contract)
    body.pop("contract_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def load_semantic_census_contract(path: Path) -> dict[str, object]:
    """Load and validate the small, versioned diagnostic policy."""

    path = Path(path)
    observed_file_sha256 = file_sha256(path)
    if observed_file_sha256 != PINNED_SEMANTIC_CENSUS_CONTRACT_FILE_SHA256:
        raise Active8SemanticCensusError(
            "semantic census contract physical SHA-256 mismatch: "
            f"observed={observed_file_sha256}, "
            f"expected={PINNED_SEMANTIC_CENSUS_CONTRACT_FILE_SHA256}"
        )
    try:
        contract = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise Active8SemanticCensusError(
            f"cannot read semantic census contract {path}: {exc}"
        ) from exc
    recorded_self_sha256 = contract.get("contract_sha256")
    computed_self_sha256 = _contract_self_sha256(contract)
    if recorded_self_sha256 != computed_self_sha256:
        raise Active8SemanticCensusError(
            "semantic census contract canonical self-hash mismatch: "
            f"recorded={recorded_self_sha256!r}, computed={computed_self_sha256}"
        )
    if (
        contract.get("schema") != SEMANTIC_CONTRACT_SCHEMA
        or contract.get("schema_version") != SEMANTIC_CONTRACT_SCHEMA_VERSION
    ):
        raise Active8SemanticCensusError("semantic census contract schema/version mismatch")
    if contract.get("status") != SEMANTIC_CENSUS_STATUS:
        raise Active8SemanticCensusError("semantic census contract has an unsafe status")
    if contract.get("partition") != "train":
        raise Active8SemanticCensusError("semantic census must remain train-only")
    if contract.get("active_families") != list(ACTIVE8_FAMILIES):
        raise Active8SemanticCensusError("semantic census contract does not preserve Active8 order")
    if contract.get("parent_t1_successor_gate") != PINNED_T1_SUCCESSOR_GATE_V8:
        raise Active8SemanticCensusError(
            "semantic census contract is not bound to the current T1 successor V8 gate"
        )
    if contract.get("parent_active8_identity") != PINNED_ACTIVE8_PARENT_IDENTITY:
        raise Active8SemanticCensusError(
            "semantic census contract is not bound to the exact current Active8 parent"
        )
    threshold = contract.get("sparse_observed_value_max_rows")
    if type(threshold) is not int or threshold < 0:
        raise Active8SemanticCensusError("sparse threshold must be a nonnegative integer")
    axes = contract.get("axes")
    if not isinstance(axes, dict):
        raise Active8SemanticCensusError("semantic census contract axes must be an object")
    expected_axes = set(_COMMON_AXES)
    for family_axes in _FAMILY_AXES.values():
        expected_axes.update(family_axes)
    if set(axes) != expected_axes:
        raise Active8SemanticCensusError("semantic census contract axis registry is incomplete")
    for axis_name, raw in axes.items():
        if not isinstance(raw, dict) or raw.get("value_kind") not in {
            "declared_finite",
            "open_observed",
        }:
            raise Active8SemanticCensusError(f"axis {axis_name!r} has an invalid value kind")
        domain = raw.get("declared_domain")
        if raw["value_kind"] == "declared_finite":
            if (
                not isinstance(domain, list)
                or not domain
                or len(set(domain)) != len(domain)
                or any(not isinstance(value, str) or not value for value in domain)
            ):
                raise Active8SemanticCensusError(
                    f"axis {axis_name!r} requires a unique non-empty declared domain"
                )
        elif domain is not None:
            raise Active8SemanticCensusError(
                f"open-observed axis {axis_name!r} cannot declare an absence domain"
            )
    return contract


def validate_parent_t1_successor_gate(
    contract: Mapping[str, object],
    *,
    path: Path,
) -> dict[str, object]:
    """Verify the external V8 parent file and its copied Active8 identities."""

    path = Path(path)
    observed_file_sha256 = file_sha256(path)
    if observed_file_sha256 != PINNED_T1_SUCCESSOR_GATE_V8["file_sha256"]:
        raise Active8SemanticCensusError(
            "T1 successor V8 parent file SHA-256 mismatch: "
            f"observed={observed_file_sha256}, "
            f"expected={PINNED_T1_SUCCESSOR_GATE_V8['file_sha256']}"
        )
    try:
        parent = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise Active8SemanticCensusError(
            f"cannot read T1 successor V8 parent {path}: {exc}"
        ) from exc
    if parent.get("contract_sha256") != PINNED_T1_SUCCESSOR_GATE_V8["contract_sha256"]:
        raise Active8SemanticCensusError("T1 successor V8 logical contract identity mismatch")
    field_map = {
        "inventory_manifest_file_sha256": "active8_inventory_manifest_file_sha256",
        "inventory_sha256": "active8_inventory_sha256",
        "effective_source_corpus_cache_sha256": ("active8_effective_source_corpus_cache_sha256"),
        "support_contract_sha256": "active8_support_contract_sha256",
        "unified_packed_manifest_sha256": "active8_unified_packed_manifest_sha256",
    }
    copied_parent = contract["parent_active8_identity"]
    assert isinstance(copied_parent, Mapping)
    for contract_field, parent_field in field_map.items():
        expected = copied_parent[contract_field]
        if parent.get(parent_field) != expected:
            raise Active8SemanticCensusError(
                f"T1 successor V8 parent identity mismatch on {parent_field}: "
                f"observed={parent.get(parent_field)!r}, expected={expected!r}"
            )
    return parent


def _element(state: MolecularGraph, slot: int) -> str:
    if not 0 <= slot < state.n_atoms or not bool(is_element(np.asarray(state.atom_types[slot]))):
        raise Active8SemanticCensusError(f"slot {slot} is not a real element")
    return ELEMENTS[int(state.atom_types[slot])]


def _charge(state: MolecularGraph, slot: int) -> int:
    value = int(state.formal_charges[slot])
    if value not in FORMAL_CHARGES:
        raise Active8SemanticCensusError(f"slot {slot} has unsupported formal charge {value}")
    return value


def _bond_label(order: int, *, allow_aromatic: bool = True) -> str:
    order = int(order)
    if order not in _BOND_LABELS or (not allow_aromatic and order == BOND_AROMATIC):
        raise Active8SemanticCensusError(f"unsupported bond class {order}")
    return _BOND_LABELS[order]


def _real_vertices(state: MolecularGraph) -> tuple[int, ...]:
    return tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))


def _neighbors(state: MolecularGraph, vertex: int) -> tuple[int, ...]:
    return tuple(
        int(v)
        for v in np.flatnonzero(state.bonds[vertex] != 0)
        if bool(is_element(np.asarray(state.atom_types[v])))
    )


def _degree(state: MolecularGraph, vertex: int) -> int:
    return len(_neighbors(state, vertex))


def _pair(values: Iterable[object]) -> str:
    materialized = tuple(values)
    if all(type(value) is int for value in materialized):
        ordered = sorted(materialized)
    else:
        ordered = sorted(materialized, key=str)
    return "|".join(str(value) for value in ordered)


def _shortest_path_edges(
    state: MolecularGraph,
    source: int,
    target: int,
    *,
    excluded_edge: frozenset[int] | None = None,
) -> int | None:
    if source == target:
        return 0
    visited = {int(source)}
    frontier = [(int(source), 0)]
    for vertex, distance in frontier:
        for neighbor in _neighbors(state, vertex):
            if excluded_edge is not None and frozenset((vertex, neighbor)) == excluded_edge:
                continue
            if neighbor == target:
                return distance + 1
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, distance + 1))
    return None


def _edge_cycle_context(state: MolecularGraph, a: int, b: int) -> str:
    if int(state.bonds[a, b]) == 0:
        raise Active8SemanticCensusError(f"edge ({a}, {b}) is absent")
    alternative = _shortest_path_edges(
        state,
        a,
        b,
        excluded_edge=frozenset((int(a), int(b))),
    )
    return "nonbridge" if alternative is not None else "bridge"


def _site_cycle_context(state: MolecularGraph, vertex: int) -> str:
    for neighbor in _neighbors(state, vertex):
        if _edge_cycle_context(state, vertex, neighbor) == "nonbridge":
            return "cyclic"
    return "acyclic"


def _component_after_cut(state: MolecularGraph, root: int, a: int, b: int) -> set[int]:
    excluded = frozenset((int(a), int(b)))
    seen = {int(root)}
    frontier = [int(root)]
    for vertex in frontier:
        for neighbor in _neighbors(state, vertex):
            if frozenset((vertex, neighbor)) == excluded or neighbor in seen:
                continue
            seen.add(neighbor)
            frontier.append(neighbor)
    return seen


def _cycle_rank(state: MolecularGraph) -> int:
    vertices = set(_real_vertices(state))
    edge_count = sum(_degree(state, vertex) for vertex in vertices) // 2
    components = 0
    unseen = set(vertices)
    while unseen:
        components += 1
        root = min(unseen)
        reached = {root}
        frontier = [root]
        for vertex in frontier:
            for neighbor in _neighbors(state, vertex):
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        unseen.difference_update(reached)
    return edge_count - len(vertices) + components


def _step_role(step_index: int, path_length: int) -> str:
    if type(step_index) is not int or type(path_length) is not int:
        raise Active8SemanticCensusError("step index and path length must be integers")
    if path_length <= 0 or not 0 <= step_index < path_length:
        raise Active8SemanticCensusError("step index lies outside the nonterminal trace rows")
    if path_length == 1:
        return "only_step"
    if step_index == 0:
        return "first"
    if step_index == path_length - 1:
        return "last"
    return "interior"


def _common_axes(
    source: MolecularGraph,
    *,
    data_lane: str,
    step_index: int,
    path_length: int,
) -> dict[str, str]:
    if not isinstance(data_lane, str) or not data_lane:
        raise Active8SemanticCensusError("data_lane must be non-empty text")
    return {
        "data_lane": data_lane,
        "trace_step_role": _step_role(step_index, path_length),
        "source_atom_count": str(len(_real_vertices(source))),
        "source_cycle_rank": str(_cycle_rank(source)),
    }


def derive_semantic_observation(
    source: MolecularGraph,
    successor: MolecularGraph,
    *,
    executor_rule: str,
    action: object,
    data_lane: str,
    step_index: int,
    path_length: int,
    candidate_alias: OrthogonalCandidateAliasMetrics | None = None,
) -> SemanticObservation:
    """Derive one deterministic label vector from an exact stored row."""

    family = canonical_family(executor_rule)
    if family not in ACTIVE8_FAMILIES:
        raise Active8SemanticCensusError(f"executor rule maps outside Active8: {family!r}")
    expected_type = _ACTION_TYPES[family]
    if type(action) is not expected_type:
        raise Active8SemanticCensusError(
            f"{family} requires {expected_type.__name__}, got {type(action).__name__}"
        )
    axes = _common_axes(
        source,
        data_lane=data_lane,
        step_index=step_index,
        path_length=path_length,
    )

    if family == "atom_insert":
        assert isinstance(action, AtomInsert)
        neighbor_count = len(action.neighbors)
        if neighbor_count not in {0, 1}:
            raise Active8SemanticCensusError(
                "Active8 birth support permits only root or exactly-one-neighbor insertion"
            )
        inserted = int(action.slot)
        inserted_type = int(action.atom_type)
        if not bool(is_element(np.asarray(inserted_type))):
            raise Active8SemanticCensusError("atom-insert action does not insert a real element")
        inserted_element = ELEMENTS[inserted_type]
        if _element(successor, inserted) != inserted_element:
            raise Active8SemanticCensusError(
                "stored atom-insert successor disagrees with its action"
            )
        if _charge(successor, inserted) != int(action.formal_charge) or int(
            successor.implicit_h_counts[inserted]
        ) != int(action.implicit_h_count):
            raise Active8SemanticCensusError("stored inserted charge/H disagrees with its action")
        if neighbor_count:
            neighbor, order = action.neighbors[0]
            attachment_element = _element(source, int(neighbor))
            attachment_bond = _bond_label(int(order), allow_aromatic=False)
            if int(successor.bonds[inserted, int(neighbor)]) != int(order):
                raise Active8SemanticCensusError(
                    "stored atom-insert attachment disagrees with its action"
                )
        else:
            attachment_element = "none"
            attachment_bond = "none"
        axes.update(
            insertion_mode="root" if neighbor_count == 0 else "one_neighbor",
            inserted_element=inserted_element,
            inserted_formal_charge=str(int(action.formal_charge)),
            attachment_bond_class=attachment_bond,
            attachment_element=attachment_element,
        )

    elif family == "atom_delete":
        assert isinstance(action, AtomDelete)
        vertex = int(action.v)
        if bool(is_element(np.asarray(successor.atom_types[vertex]))):
            raise Active8SemanticCensusError(
                "stored atom-delete successor leaves a real element in the deleted slot"
            )
        neighbors = _neighbors(source, vertex)
        degree = len(neighbors)
        if len(_real_vertices(source)) == 1:
            deletion_mode = "last_atom"
        elif degree == 0:
            deletion_mode = "isolated"
        elif degree == 1:
            deletion_mode = "leaf"
        else:
            deletion_mode = "multi_neighbor"
        signature = (
            "+".join(sorted(_bond_label(int(source.bonds[vertex, other])) for other in neighbors))
            or "none"
        )
        axes.update(
            deletion_mode=deletion_mode,
            deleted_element=_element(source, vertex),
            deleted_formal_charge=str(_charge(source, vertex)),
            deleted_heavy_degree=str(degree),
            incident_bond_signature=signature,
        )

    elif family == "atom_restate":
        assert isinstance(action, AtomRestate)
        vertex = int(action.v)
        old_h = int(source.implicit_h_counts[vertex])
        delta_h = int(action.implicit_h_count) - old_h
        restated_type = int(action.atom_type)
        if not bool(is_element(np.asarray(restated_type))):
            raise Active8SemanticCensusError("atom-restate action does not name a real element")
        restated_element = ELEMENTS[restated_type]
        if _element(successor, vertex) != restated_element:
            raise Active8SemanticCensusError(
                "stored atom-restate successor disagrees with its action"
            )
        if _charge(successor, vertex) != int(action.formal_charge) or int(
            successor.implicit_h_counts[vertex]
        ) != int(action.implicit_h_count):
            raise Active8SemanticCensusError(
                "stored atom-restate charge/H successor disagrees with its action"
            )
        direction = "decrease" if delta_h < 0 else "increase" if delta_h > 0 else "unchanged"
        site_cycle_context = _site_cycle_context(source, vertex)
        if site_cycle_context == "acyclic":
            cyclic_target_support_class = "acyclic"
        elif restated_type in ORGANIC_RING_ELEMENTS:
            cyclic_target_support_class = "cyclic_declared_ring_element"
        else:
            cyclic_target_support_class = "cyclic_nonring_element"
        axes.update(
            element_transition=f"{_element(source, vertex)}->{restated_element}",
            formal_charge_transition=f"{_charge(source, vertex)}->{int(action.formal_charge)}",
            implicit_h_delta=str(delta_h),
            implicit_h_direction=direction,
            site_heavy_degree=str(_degree(source, vertex)),
            site_cycle_context=site_cycle_context,
            cyclic_target_support_class=cyclic_target_support_class,
        )

    elif family == "bond_reorder":
        assert isinstance(action, BondReorder)
        a, b = int(action.a), int(action.b)
        old_order = int(source.bonds[a, b])
        if int(successor.bonds[a, b]) != int(action.new_order):
            raise Active8SemanticCensusError(
                "stored bond-reorder successor disagrees with its action"
            )
        axes.update(
            bond_order_transition=(
                f"{_bond_label(old_order, allow_aromatic=False)}"
                f"->{_bond_label(int(action.new_order), allow_aromatic=False)}"
            ),
            endpoint_element_pair=_pair((_element(source, a), _element(source, b))),
            endpoint_degree_pair=_pair((_degree(source, a), _degree(source, b))),
            source_edge_cycle_context=_edge_cycle_context(source, a, b),
        )

    elif family == "bond_reroute":
        assert isinstance(action, BondReroute)
        a, b, u, v = map(int, (action.a, action.b, action.u, action.v))
        old_order = int(source.bonds[a, b])
        if int(successor.bonds[a, b]) != 0 or int(successor.bonds[u, v]) != int(action.new_order):
            raise Active8SemanticCensusError(
                "stored bond-reroute successor disagrees with its action"
            )
        component = _component_after_cut(source, a, a, b)
        other_size = len(_real_vertices(source)) - len(component)
        axes.update(
            bond_order_transition=(
                f"{_bond_label(old_order, allow_aromatic=False)}"
                f"->{_bond_label(int(action.new_order), allow_aromatic=False)}"
            ),
            removed_endpoint_element_pair=_pair((_element(source, a), _element(source, b))),
            added_endpoint_element_pair=_pair((_element(source, u), _element(source, v))),
            reused_removed_endpoint_count=str(len({a, b} & {u, v})),
            cut_component_size_pair=_pair((len(component), other_size)),
            added_endpoint_degree_pair=_pair((_degree(source, u), _degree(source, v))),
            removed_edge_cycle_context=_edge_cycle_context(source, a, b),
        )

    elif family == "cycle_insert":
        assert isinstance(action, BondInsert)
        a, b = int(action.a), int(action.b)
        path_edges = _shortest_path_edges(source, a, b)
        if path_edges is None:
            raise Active8SemanticCensusError("cycle close endpoints have no preclosure path")
        if int(successor.bonds[a, b]) != int(action.order):
            raise Active8SemanticCensusError(
                "stored cycle-close successor disagrees with its action"
            )
        axes.update(
            closure_bond_class=_bond_label(int(action.order), allow_aromatic=False),
            endpoint_element_pair=_pair((_element(source, a), _element(source, b))),
            endpoint_degree_pair=_pair((_degree(source, a), _degree(source, b))),
            preclosure_shortest_path_edges=str(path_edges),
            closed_shortest_cycle_size=str(path_edges + 1),
        )

    elif family == "cycle_attach":
        assert isinstance(action, BondDelete)
        a, b = int(action.a), int(action.b)
        remaining = _shortest_path_edges(
            source,
            a,
            b,
            excluded_edge=frozenset((a, b)),
        )
        if remaining is None:
            raise Active8SemanticCensusError("cycle open action targets a bridge, not a cycle edge")
        if int(successor.bonds[a, b]) != 0:
            raise Active8SemanticCensusError(
                "stored cycle-open successor disagrees with its action"
            )
        axes.update(
            opened_bond_class=_bond_label(int(source.bonds[a, b])),
            endpoint_element_pair=_pair((_element(source, a), _element(source, b))),
            endpoint_degree_pair=_pair((_degree(source, a), _degree(source, b))),
            remaining_shortest_path_edges=str(remaining),
            opened_shortest_cycle_size=str(remaining + 1),
        )

    elif family == "ring_system_restate":
        assert isinstance(action, RingSystemRestate)
        if not action.changes:
            raise Active8SemanticCensusError("ring-system restate has no changed bonds")
        transitions: list[str] = []
        affected: set[int] = set()
        contexts: set[str] = set()
        aromatic_transition = False
        for change in action.changes:
            a, b = int(change.a), int(change.b)
            old_order = int(source.bonds[a, b])
            new_order = int(change.new_order)
            old_label = _bond_label(old_order)
            new_label = _bond_label(new_order)
            if int(successor.bonds[a, b]) != new_order:
                raise Active8SemanticCensusError(
                    "stored ring-system-restatement successor disagrees with its action"
                )
            transitions.append(f"{old_label}->{new_label}")
            affected.update((a, b))
            contexts.add(_edge_cycle_context(source, a, b))
            aromatic_transition |= BOND_AROMATIC in {old_order, new_order}
        context = next(iter(contexts)) if len(contexts) == 1 else "mixed"
        axes.update(
            changed_bond_count=str(len(action.changes)),
            affected_atom_count=str(len(affected)),
            bond_order_transition_multiset="+".join(sorted(transitions)),
            affected_element_set="+".join(sorted({_element(source, v) for v in affected})),
            changed_edge_cycle_context=context,
            aromatic_transition_presence="present" if aromatic_transition else "absent",
        )
    else:  # pragma: no cover - family coverage is guarded above.
        raise AssertionError(f"unhandled Active8 family {family}")

    return SemanticObservation(
        family=family,
        axes=axes,
        source_state_sha256=persistent_slot_state_sha256(source),
        successor_state_sha256=persistent_slot_state_sha256(successor),
        candidate_alias=candidate_alias,
    )


class _ExactStateMembershipCensus:
    """Disk-capable exact distinct-state counter for row marginals."""

    def __init__(self, database: str | Path) -> None:
        database_text = str(database)
        if database_text != ":memory:" and Path(database_text).exists():
            raise Active8SemanticCensusError(
                f"exact-state uniqueness database already exists: {database_text}"
            )
        self._connection = sqlite3.connect(database_text)
        self._connection.executescript(
            """
            PRAGMA synchronous = OFF;
            PRAGMA journal_mode = OFF;
            CREATE TABLE family_state (
                state_role TEXT NOT NULL,
                family TEXT NOT NULL,
                state_digest BLOB NOT NULL,
                PRIMARY KEY (state_role, family, state_digest)
            ) WITHOUT ROWID;
            CREATE TABLE axis_state (
                state_role TEXT NOT NULL,
                family TEXT NOT NULL,
                axis_name TEXT NOT NULL,
                axis_value TEXT NOT NULL,
                state_digest BLOB NOT NULL,
                PRIMARY KEY (state_role, family, axis_name, axis_value, state_digest)
            ) WITHOUT ROWID;
            """
        )

    def add(self, observation: SemanticObservation) -> None:
        identities = (
            ("source", bytes.fromhex(observation.source_state_sha256)),
            ("successor", bytes.fromhex(observation.successor_state_sha256)),
        )
        self._connection.executemany(
            "INSERT OR IGNORE INTO family_state VALUES (?, ?, ?)",
            ((state_role, observation.family, digest) for state_role, digest in identities),
        )
        self._connection.executemany(
            "INSERT OR IGNORE INTO axis_state VALUES (?, ?, ?, ?, ?)",
            (
                (state_role, observation.family, axis, value, digest)
                for state_role, digest in identities
                for axis, value in observation.axes.items()
            ),
        )

    def finish(self) -> None:
        self._connection.commit()

    def family_count(self, *, state_role: str, family: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM family_state WHERE state_role = ? AND family = ?",
            (state_role, family),
        ).fetchone()
        assert row is not None
        return int(row[0])

    def global_count(self, *, state_role: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(DISTINCT state_digest) FROM family_state WHERE state_role = ?",
            (state_role,),
        ).fetchone()
        assert row is not None
        return int(row[0])

    def axis_counts(
        self,
        *,
        state_role: str,
        family: str,
        axis: str,
    ) -> dict[str, int]:
        rows = self._connection.execute(
            """
            SELECT axis_value, COUNT(*)
            FROM axis_state
            WHERE state_role = ? AND family = ? AND axis_name = ?
            GROUP BY axis_value
            ORDER BY axis_value
            """,
            (state_role, family, axis),
        )
        return {str(value): int(count) for value, count in rows}

    def close(self) -> None:
        self._connection.close()


def build_semantic_census(
    observations: Iterable[SemanticObservation],
    *,
    contract: Mapping[str, object],
    provenance: Mapping[str, object],
    uniqueness_database: str | Path = ":memory:",
) -> dict[str, object]:
    """Aggregate independent marginals without constructing semantic joint cells."""

    threshold = int(contract["sparse_observed_value_max_rows"])
    axis_contract = contract["axes"]
    assert isinstance(axis_contract, Mapping)
    family_rows = Counter({family: 0 for family in ACTIVE8_FAMILIES})
    marginal_counts: dict[str, dict[str, Counter[str]]] = {
        family: defaultdict(Counter) for family in ACTIVE8_FAMILIES
    }
    metric_counts: dict[str, dict[str, Counter[str]]] = {
        family: {metric: Counter() for metric in _ORTHOGONAL_METRICS} for family in ACTIVE8_FAMILIES
    }
    candidate_rows = Counter({family: 0 for family in ACTIVE8_FAMILIES})
    exact_states = _ExactStateMembershipCensus(uniqueness_database)

    try:
        for observation in observations:
            family_rows[observation.family] += 1
            exact_states.add(observation)
            for axis, value in observation.axes.items():
                raw_axis = axis_contract[axis]
                assert isinstance(raw_axis, Mapping)
                domain = raw_axis.get("declared_domain")
                if domain is not None and value not in domain:
                    raise Active8SemanticCensusError(
                        f"observed value {value!r} is outside declared axis {axis!r}"
                    )
                marginal_counts[observation.family][axis][value] += 1
            if observation.candidate_alias is not None:
                candidate_rows[observation.family] += 1
                for metric, value in observation.candidate_alias.as_dict().items():
                    metric_counts[observation.family][metric][str(value)] += 1
        exact_states.finish()

        families: dict[str, object] = {}
        for family in ACTIVE8_FAMILIES:
            axes: dict[str, object] = {}
            for axis in (*_COMMON_AXES, *_FAMILY_AXES[family]):
                raw_axis = axis_contract[axis]
                assert isinstance(raw_axis, Mapping)
                observed = marginal_counts[family][axis]
                source_counts = exact_states.axis_counts(
                    state_role="source", family=family, axis=axis
                )
                successor_counts = exact_states.axis_counts(
                    state_role="successor", family=family, axis=axis
                )
                declared_domain = raw_axis.get("declared_domain")
                absent = (
                    [value for value in declared_domain if value not in observed]
                    if isinstance(declared_domain, list)
                    else None
                )
                axes[axis] = {
                    "value_kind": raw_axis["value_kind"],
                    "description": raw_axis["description"],
                    "declared_domain": declared_domain,
                    "observed_row_counts": dict(sorted(observed.items())),
                    "unique_exact_source_state_counts": source_counts,
                    "unique_exact_successor_state_counts": successor_counts,
                    "observed_value_count": len(observed),
                    "sparse_observed_values": [
                        {
                            "value": value,
                            "rows": rows,
                            "unique_exact_source_states": source_counts[value],
                            "unique_exact_successor_states": successor_counts[value],
                        }
                        for value, rows in sorted(observed.items())
                        if rows <= threshold
                    ],
                    "absent_declared_values": absent,
                }
            supplied = candidate_rows[family]
            total = family_rows[family]
            availability = (
                "not_supplied" if supplied == 0 else "complete" if supplied == total else "partial"
            )
            families[family] = {
                "rows": total,
                "unique_exact_source_states": exact_states.family_count(
                    state_role="source", family=family
                ),
                "unique_exact_successor_states": exact_states.family_count(
                    state_role="successor", family=family
                ),
                "unique_scaffolds": None,
                "scaffold_count_claimed": False,
                "axes": axes,
                "orthogonal_candidate_alias_diagnostics": {
                    "availability": availability,
                    "rows_with_metrics": supplied,
                    "rows_without_metrics": total - supplied,
                    "metric_marginals": {
                        metric: dict(sorted(metric_counts[family][metric].items()))
                        for metric in _ORTHOGONAL_METRICS
                    },
                    "joined_to_semantic_axes": False,
                },
            }

        total_rows = sum(family_rows.values())
        return {
            "schema": SEMANTIC_CENSUS_SCHEMA,
            "schema_version": SEMANTIC_CENSUS_SCHEMA_VERSION,
            "status": SEMANTIC_CENSUS_STATUS,
            "scope": {
                "partition": contract["partition"],
                "active_families": list(ACTIVE8_FAMILIES),
                "census_unit": "accepted_nonterminal_teacher_row",
                "aggregation": "independent_within_family_axis_marginals_only",
                "cartesian_joint_cells_constructed": False,
                "exact_state_identity": {
                    "schema": PERSISTENT_STATE_DIGEST_SCHEMA,
                    "schema_version": PERSISTENT_STATE_DIGEST_VERSION,
                },
                "scaffold_identity_available": False,
                "scaffold_count_claimed": False,
            },
            "evidence_ledger": contract["evidence_ledger"],
            "policy": {
                "sparse_observed_value_max_rows": threshold,
                "sparse_or_absent_is_corpus_failure": False,
                "open_observed_axis_absence_is_defined": False,
                "candidate_alias_metrics_joined_to_semantic_axes": False,
                "row_density_interpreted_as_unique_state_coverage": False,
            },
            "provenance": dict(provenance),
            "counts": {
                "rows": total_rows,
                "rows_by_family": {family: family_rows[family] for family in ACTIVE8_FAMILIES},
                "unique_exact_source_states": exact_states.global_count(state_role="source"),
                "unique_exact_successor_states": exact_states.global_count(state_role="successor"),
                "unique_scaffolds": None,
            },
            "families": families,
            "decisions": {
                "corpus_failure": None,
                "model_learnability": None,
                "training_authorized": False,
                # The one frozen spelling. This report is a diagnostic and grants
                # nothing, so the field exists to say so; under the retired name
                # it said so where a consumer checking the frozen vocabulary was
                # not looking.
                "bounded_p50_authorized": False,
            },
            "limitations": list(contract["limitations"]),
        }
    finally:
        exact_states.close()


def semantic_census_bytes(payload: Mapping[str, object]) -> bytes:
    """Stable pretty JSON bytes for a completed diagnostic derivative."""

    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def implementation_identity(*, repo_root: Path | None = None) -> dict[str, object]:
    """Bind every local source that defines exact label semantics."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    sources = (
        "scripts/build_editing_active8_semantic_census_v2.py",
        "src/compose_v4/experiments/editing_active8_semantic_census_v2.py",
        "src/compose_v4/chem/molecular_graph.py",
        "src/compose_v4/chem/persistent_state_identity.py",
        "src/compose_v4/rewrite/action_codec.py",
        "src/compose_v4/rewrite/operators.py",
        "src/compose_v4/rewrite/tracelets.py",
        "src/compose_v4/data/active8_trace_inventory.py",
        "src/compose_v4/data/packed_charge_policy_audit.py",
        "src/compose_v4/data/packed_trace_store.py",
    )
    hashes = {relative: file_sha256(root / relative) for relative in sources}
    return {
        "sources": hashes,
        "implementation_sha256": hashlib.sha256(_canonical_json_bytes(hashes)).hexdigest(),
    }


__all__ = [
    "PINNED_ACTIVE8_PARENT_IDENTITY",
    "PINNED_SEMANTIC_CENSUS_CONTRACT_FILE_SHA256",
    "PINNED_T1_SUCCESSOR_GATE_V8",
    "SEMANTIC_CENSUS_SCHEMA",
    "SEMANTIC_CENSUS_SCHEMA_VERSION",
    "SEMANTIC_CENSUS_STATUS",
    "Active8SemanticCensusError",
    "OrthogonalCandidateAliasMetrics",
    "SemanticObservation",
    "build_semantic_census",
    "derive_semantic_observation",
    "file_sha256",
    "implementation_identity",
    "load_semantic_census_contract",
    "semantic_census_bytes",
    "validate_parent_t1_successor_gate",
]
