"""Train-only audit of atom-restatement limits induced by the current encoder.

This diagnostic is deliberately parameter independent.  It computes a
sufficient certificate for two atom sites to have identical node states after
the six additive message-passing rounds used by Editing V2.  It then crosses
that certificate with the target atom-valence class, executes every legal
generic atom-restatement mark, and measures how much family-conditional
teacher-successor probability any model with the current information path can
assign.

The audit does not change support.  In particular, generic broad-organic
cyclic restatement is the current model behavior.  The alternate policies in
the prospective contract are explicitly hypothetical views used to quantify
tradeoffs.  They are not recommendations, and the declared organic ring-element
set is not treated as authoritative for generic atom restatement or confused
with the narrower currently wired macro ring head.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.graph_primitives import compute_topology_features
from compose_v4.chem.molecular_graph import (
    BOND_CLASS_TO_H_CHANGE,
    IDX_TO_ELEMENT,
    MAX_H_COUNT,
    ORGANIC_RING_ELEMENTS,
    ORGANIC_VOCABULARY,
    AtomVocabulary,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    RewriteSystem,
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.operators import AtomRestate

AUDIT_SCHEMA = "compose.experiments.atom_restate_neural_orbit_audit"
AUDIT_SCHEMA_VERSION = 1
AUDIT_STATUS = "DIAGNOSTIC_ONLY_NOT_TRAINING_AUTHORITY"
CONTRACT_SCHEMA = "compose.experiments.atom_restate_neural_orbit_audit_contract"
CONTRACT_SCHEMA_VERSION = 1
DEFAULT_MESSAGE_PASSING_ROUNDS = 6

# Pinned after the prospective contract was finalized.  Keeping this outside
# the self-hashed JSON means a changed contract cannot validate itself merely
# by recomputing its own logical hash.
PINNED_CONTRACT_FILE_SHA256 = "eb0cd83e6bc92fe9f08a9fde5068ad3962413c0083d1cf216bda0ea8c4dd644d"

CURRENT_POLICY_ID = "current_generic_broad_restate_all_real_sites"
HYPOTHETICAL_DECLARED_RING_ELEMENT_POLICY_ID = (
    "hypothetical_cyclic_targets_restricted_to_declared_organic_ring_elements"
)
HYPOTHETICAL_ACYCLIC_POLICY_ID = "hypothetical_acyclic_sites_only"


class AtomRestateNeuralOrbitAuditError(RuntimeError):
    """The requested audit could not be derived exactly and safely."""


@dataclass(frozen=True)
class RestatePolicy:
    """One diagnostic view over the unchanged current broad restate fiber."""

    policy_id: str
    policy_kind: str
    hypothetical: bool

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("restate policy requires a non-empty policy_id")
        if self.policy_kind not in {
            "current_broad",
            "cyclic_target_declared_organic_ring_element_set",
            "acyclic_sites_only",
        }:
            raise ValueError(f"unknown restate policy kind {self.policy_kind!r}")
        if (self.policy_kind == "current_broad") == self.hypothetical:
            raise ValueError(
                "only current_broad may be non-hypothetical; alternate views must be explicit"
            )

    def retains(
        self,
        *,
        site_is_cyclic: bool,
        target_element: int,
    ) -> bool:
        if self.policy_kind == "current_broad":
            return True
        if self.policy_kind == "cyclic_target_declared_organic_ring_element_set":
            return (not site_is_cyclic) or int(target_element) in ORGANIC_RING_ELEMENTS
        if self.policy_kind == "acyclic_sites_only":
            return not site_is_cyclic
        raise AssertionError("validated policy kind became unreachable")


DEFAULT_POLICIES = (
    RestatePolicy(CURRENT_POLICY_ID, "current_broad", False),
    RestatePolicy(
        HYPOTHETICAL_DECLARED_RING_ELEMENT_POLICY_ID,
        "cyclic_target_declared_organic_ring_element_set",
        True,
    ),
    RestatePolicy(HYPOTHETICAL_ACYCLIC_POLICY_ID, "acyclic_sites_only", True),
)


@dataclass(frozen=True)
class TeacherProgressAddress:
    """Immutable packed trace-progress address for one train-only teacher."""

    packed_shard_content_sha256: str
    packed_shard_name: str
    entry_index: int
    trace_id: str
    layer: str
    partition: str
    progress_index: int

    def __post_init__(self) -> None:
        _require_sha256(
            self.packed_shard_content_sha256,
            field="packed_shard_content_sha256",
        )
        for name in ("packed_shard_name", "trace_id", "layer", "partition"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("entry_index", "progress_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.partition != "train":
            raise ValueError(
                "atom-restatement orbit audits are train-only; "
                f"received partition {self.partition!r}"
            )

    @classmethod
    def from_packed_address(
        cls,
        address: PackedTraceAddress,
        *,
        progress_index: int,
    ) -> TeacherProgressAddress:
        return cls(
            packed_shard_content_sha256=address.packed_shard_content_sha256,
            packed_shard_name=address.packed_shard_name,
            entry_index=int(address.entry_index),
            trace_id=address.trace_id,
            layer=address.layer,
            partition=address.partition,
            progress_index=int(progress_index),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "packed_shard_content_sha256": self.packed_shard_content_sha256,
            "packed_shard_name": self.packed_shard_name,
            "entry_index": self.entry_index,
            "trace_id": self.trace_id,
            "layer": self.layer,
            "partition": self.partition,
            "progress_index": self.progress_index,
        }


@dataclass(frozen=True)
class RestateCandidate:
    """One executable current-fiber atom-restatement mark."""

    coordinate: tuple[int, int]
    action: AtomRestate
    node_orbit: int
    target_class_index: int
    target_class: tuple[int, int]
    site_is_cyclic: bool
    source_element: int
    successor_key: str
    successor_state_sha256: str

    @property
    def neural_class_key(self) -> tuple[int, int]:
        return self.node_orbit, self.target_class_index


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_ids(signatures: Sequence[tuple[Any, ...]]) -> tuple[int, ...]:
    """Intern exact symbolic signatures into deterministic within-state IDs."""

    unique = sorted(set(signatures))
    index = {signature: class_index for class_index, signature in enumerate(unique)}
    return tuple(index[signature] for signature in signatures)


def additive_message_orbit_classes(
    initial_node_labels: Sequence[tuple[int, int, int, int]],
    neural_bonds: np.ndarray,
    real_mask: Sequence[bool],
    *,
    rounds: int = DEFAULT_MESSAGE_PASSING_ROUNDS,
) -> tuple[int, ...]:
    """Return a sufficient equality certificate for additive message passing.

    Each update sees the previous self state, the sum of transformed neighbor
    states, and the sum of edge embeddings.  Consequently it loses the pairing
    between a neighbor label and the edge class incident to that neighbor.  The
    exact symbolic update is therefore::

        (self label,
         multiset(neighbor labels),
         multiset(incident neural edge classes)).

    Time and optional property context are shared by all nodes in one state and
    therefore cannot split a class certified equal here.
    """

    if type(rounds) is not int or rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    labels = tuple(tuple(int(value) for value in label) for label in initial_node_labels)
    real = np.asarray(real_mask, dtype=np.bool_)
    bonds = np.asarray(neural_bonds)
    n_slots = len(labels)
    if real.shape != (n_slots,):
        raise ValueError("real_mask does not align with initial node labels")
    if bonds.shape != (n_slots, n_slots):
        raise ValueError("neural_bonds must be square and align with node labels")
    if not np.array_equal(bonds, bonds.T) or np.any(np.diag(bonds) != 0):
        raise ValueError("neural_bonds must be symmetric with a zero diagonal")
    if np.any(bonds < 0):
        raise ValueError("neural bond classes must be nonnegative")

    real_vertices = tuple(int(vertex) for vertex in np.flatnonzero(real))
    if not real_vertices:
        return tuple(-1 for _ in range(n_slots))
    current_real = _class_ids(tuple(labels[vertex] for vertex in real_vertices))
    current = [-1] * n_slots
    for vertex, class_index in zip(real_vertices, current_real, strict=True):
        current[vertex] = class_index

    for _ in range(rounds):
        signatures = []
        for vertex in real_vertices:
            neighbors = tuple(
                int(neighbor)
                for neighbor in np.flatnonzero(bonds[vertex] != 0)
                if real[int(neighbor)]
            )
            signatures.append(
                (
                    int(current[vertex]),
                    tuple(sorted(int(current[neighbor]) for neighbor in neighbors)),
                    tuple(sorted(int(bonds[vertex, neighbor]) for neighbor in neighbors)),
                )
            )
        next_real = _class_ids(tuple(signatures))
        for vertex, class_index in zip(real_vertices, next_real, strict=True):
            current[vertex] = class_index
    return tuple(int(value) for value in current)


def molecular_additive_message_orbits(
    state: MolecularGraph,
    *,
    rounds: int = DEFAULT_MESSAGE_PASSING_ROUNDS,
) -> tuple[int, ...]:
    """Build the six-round certificate from the exact production input views."""

    atom_topology, _, _ = compute_topology_features(state)
    initial = tuple(
        (
            int(state.atom_types[vertex]),
            int(state.formal_charges[vertex]),
            int(state.implicit_h_counts[vertex]),
            int(atom_topology[vertex]),
        )
        for vertex in range(state.n_atoms)
    )
    return additive_message_orbit_classes(
        initial,
        resonance_invariant_bond_classes(state),
        is_element(state.atom_types),
        rounds=rounds,
    )


def enumerate_current_broad_restate_candidates(
    state: MolecularGraph,
    *,
    vocabulary: AtomVocabulary = ORGANIC_VOCABULARY,
    rounds: int = DEFAULT_MESSAGE_PASSING_ROUNDS,
    system: RewriteSystem | None = None,
) -> tuple[RestateCandidate, ...]:
    """Enumerate and execute the current generic broad-organic restate mask.

    This mirrors the parameter-free mask in ``FactorizedTraceletRateModel``:
    all real sites are eligible when heterogeneous ring scanning is enabled;
    target charge is zero; target hydrogen count is fixed by the exact
    executable bond-valence view and target atom-valence class; exact no-ops are
    removed.  Every retained mark is committed through the production executor.
    """

    rewrite_system = system or de_novo_rewrite_system()
    real = is_element(state.atom_types)
    atom_topology, _, _ = compute_topology_features(state)
    node_orbits = molecular_additive_message_orbits(state, rounds=rounds)
    candidates: list[RestateCandidate] = []
    for vertex in (int(value) for value in np.flatnonzero(real)):
        bond_valence = sum(int(BOND_CLASS_TO_H_CHANGE[int(order)]) for order in state.bonds[vertex])
        for target_class_index, (target_element, target_valence) in enumerate(vocabulary.classes):
            target_h = int(target_valence) - int(bond_valence)
            if not 0 <= target_h <= MAX_H_COUNT:
                continue
            if (
                int(state.atom_types[vertex]) == int(target_element)
                and int(state.implicit_h_counts[vertex]) == target_h
                and int(state.formal_charges[vertex]) == 0
            ):
                continue
            action = AtomRestate(
                v=vertex,
                atom_type=int(target_element),
                formal_charge=0,
                implicit_h_count=target_h,
            )
            try:
                successor = rewrite_system.apply(state, "atom_restate", action)
            except InvalidRewrite as exc:
                raise AtomRestateNeuralOrbitAuditError(
                    "the current model restate mask admitted a mark rejected by the production "
                    f"executor at coordinate {(vertex, target_class_index)}"
                ) from exc
            candidates.append(
                RestateCandidate(
                    coordinate=(vertex, target_class_index),
                    action=action,
                    node_orbit=int(node_orbits[vertex]),
                    target_class_index=int(target_class_index),
                    target_class=(int(target_element), int(target_valence)),
                    site_is_cyclic=bool(atom_topology[vertex] != 0),
                    source_element=int(state.atom_types[vertex]),
                    successor_key=canonical_state_key(successor),
                    successor_state_sha256=persistent_slot_state_sha256(successor),
                )
            )
    return tuple(sorted(candidates, key=lambda item: item.coordinate))


def _semantic_counts(candidates: Sequence[RestateCandidate]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {
        "site_cycle_context": Counter(),
        "source_to_target_element": Counter(),
        "target_atom_valence_class": Counter(),
    }
    for candidate in candidates:
        source_symbol = IDX_TO_ELEMENT[candidate.source_element]
        target_element, target_valence = candidate.target_class
        target_symbol = IDX_TO_ELEMENT[target_element]
        counts["site_cycle_context"]["cyclic" if candidate.site_is_cyclic else "acyclic"] += 1
        counts["source_to_target_element"][f"{source_symbol}->{target_symbol}"] += 1
        counts["target_atom_valence_class"][
            f"{candidate.target_class_index}:{target_symbol}({target_valence})"
        ] += 1
    return {axis: dict(sorted(counter.items())) for axis, counter in sorted(counts.items())}


def _policy_result(
    candidates: Sequence[RestateCandidate],
    *,
    teacher_successor_key: str,
    policy: RestatePolicy,
) -> dict[str, object]:
    retained = tuple(
        candidate
        for candidate in candidates
        if policy.retains(
            site_is_cyclic=candidate.site_is_cyclic,
            target_element=candidate.target_class[0],
        )
    )
    grouped: defaultdict[tuple[int, int], list[RestateCandidate]] = defaultdict(list)
    for candidate in retained:
        grouped[candidate.neural_class_key].append(candidate)

    class_rows = []
    ceiling_terms: list[tuple[int, int]] = []
    class_size_histogram: Counter[str] = Counter()
    successor_cardinality_histogram: Counter[str] = Counter()
    for neural_class_key in sorted(grouped):
        members = tuple(grouped[neural_class_key])
        successor_counts = Counter(member.successor_key for member in members)
        teacher_count = int(successor_counts.get(teacher_successor_key, 0))
        if teacher_count:
            ceiling_terms.append((teacher_count, len(members)))
        class_size_histogram[str(len(members))] += 1
        successor_cardinality_histogram[str(len(successor_counts))] += 1
        class_rows.append(
            {
                "node_orbit": neural_class_key[0],
                "target_class_index": neural_class_key[1],
                "mark_count": len(members),
                "canonical_successor_count": len(successor_counts),
                "teacher_successor_mark_count": teacher_count,
                "teacher_successor_fraction": (
                    teacher_count / len(members) if teacher_count else 0.0
                ),
                "successor_multiplicities": [
                    {"successor_key": key, "mark_count": count}
                    for key, count in sorted(successor_counts.items())
                ],
                "coordinates": [list(member.coordinate) for member in members],
            }
        )

    if ceiling_terms:
        numerator, denominator = max(
            ceiling_terms,
            key=lambda value: (value[0] / value[1], value[0], -value[1]),
        )
        ceiling: dict[str, object] | None = {
            "numerator": numerator,
            "denominator": denominator,
            "value": numerator / denominator,
        }
        teacher_status = "teacher_successor_retained"
    else:
        ceiling = None
        teacher_status = "teacher_successor_outside_policy_view"

    return {
        "policy_id": policy.policy_id,
        "policy_kind": policy.policy_kind,
        "hypothetical": policy.hypothetical,
        "teacher_status": teacher_status,
        "family_conditional_teacher_probability_ceiling": ceiling,
        "retained_mark_count": len(retained),
        "neural_class_count": len(grouped),
        "canonical_successor_count": len({item.successor_key for item in retained}),
        "class_size_histogram": dict(sorted(class_size_histogram.items())),
        "class_successor_cardinality_histogram": dict(
            sorted(successor_cardinality_histogram.items())
        ),
        "semantic_counts": _semantic_counts(retained),
        "classes": class_rows,
    }


def audit_atom_restate_teacher(
    state: MolecularGraph,
    teacher_action: AtomRestate,
    stored_successor: MolecularGraph,
    address: TeacherProgressAddress,
    *,
    policies: Sequence[RestatePolicy] = DEFAULT_POLICIES,
    rounds: int = DEFAULT_MESSAGE_PASSING_ROUNDS,
    vocabulary: AtomVocabulary = ORGANIC_VOCABULARY,
    system: RewriteSystem | None = None,
) -> dict[str, object]:
    """Audit one exact train teacher under current and hypothetical views."""

    if address.partition != "train":
        raise AtomRestateNeuralOrbitAuditError("only train teachers may be audited")
    if not policies or policies[0].policy_id != CURRENT_POLICY_ID:
        raise AtomRestateNeuralOrbitAuditError(
            "the current broad behavior must be the first diagnostic policy"
        )
    policy_ids = tuple(policy.policy_id for policy in policies)
    if len(set(policy_ids)) != len(policy_ids):
        raise AtomRestateNeuralOrbitAuditError("diagnostic policy IDs must be unique")
    rewrite_system = system or de_novo_rewrite_system()
    try:
        executed_teacher = rewrite_system.apply(state, "atom_restate", teacher_action)
    except InvalidRewrite as exc:
        raise AtomRestateNeuralOrbitAuditError(
            "stored atom-restatement teacher is not executable"
        ) from exc
    if persistent_slot_state_sha256(executed_teacher) != persistent_slot_state_sha256(
        stored_successor
    ):
        raise AtomRestateNeuralOrbitAuditError(
            "executed teacher does not reproduce the exact stored successor state"
        )
    candidates = enumerate_current_broad_restate_candidates(
        state,
        vocabulary=vocabulary,
        rounds=rounds,
        system=rewrite_system,
    )
    teacher_matches = tuple(
        candidate for candidate in candidates if candidate.action == teacher_action
    )
    if len(teacher_matches) != 1:
        raise AtomRestateNeuralOrbitAuditError(
            "teacher must occur exactly once in the current broad restate fiber; "
            f"observed {len(teacher_matches)} matches"
        )
    teacher = teacher_matches[0]
    teacher_successor_key = canonical_state_key(stored_successor)
    if teacher.successor_key != teacher_successor_key:
        raise AtomRestateNeuralOrbitAuditError(
            "teacher candidate and stored successor canonical identities disagree"
        )

    return {
        "schema": AUDIT_SCHEMA,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": AUDIT_STATUS,
        "partition_role": "train_only",
        "authorizes_training": False,
        "selects_support_policy": False,
        "address": address.as_dict(),
        "source_state_sha256": persistent_slot_state_sha256(state),
        "stored_successor_state_sha256": persistent_slot_state_sha256(stored_successor),
        "teacher_successor_key": teacher_successor_key,
        "teacher_coordinate": list(teacher.coordinate),
        "teacher_action": {
            "v": int(teacher_action.v),
            "atom_type": int(teacher_action.atom_type),
            "formal_charge": int(teacher_action.formal_charge),
            "implicit_h_count": int(teacher_action.implicit_h_count),
        },
        "message_passing_certificate": {
            "rounds": rounds,
            "initial_node_label": [
                "atom_type",
                "formal_charge",
                "implicit_h_count",
                "atom_topology",
            ],
            "round_update": [
                "self_label",
                "multiset_neighbor_labels",
                "multiset_incident_neural_edge_classes",
            ],
            "neighbor_edge_pairing_preserved": False,
            "candidate_class_addition": "target_atom_valence_class_index",
        },
        "current_broad_mark_count": len(candidates),
        "policy_comparisons": [
            _policy_result(
                candidates,
                teacher_successor_key=teacher_successor_key,
                policy=policy,
            )
            for policy in policies
        ],
    }


def aggregate_teacher_audits(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate row-level evidence without dropping exact problematic addresses."""

    row_list = list(rows)
    if not row_list:
        raise AtomRestateNeuralOrbitAuditError("cannot aggregate zero teacher audits")
    policy_aggregates: dict[str, dict[str, Any]] = {}
    seen_addresses: set[tuple[str, int, int]] = set()
    for row in row_list:
        if (
            row.get("schema") != AUDIT_SCHEMA
            or row.get("schema_version") != AUDIT_SCHEMA_VERSION
            or row.get("partition_role") != "train_only"
            or row.get("authorizes_training") is not False
        ):
            raise AtomRestateNeuralOrbitAuditError(
                "row is not a valid non-authorizing train-only orbit audit"
            )
        address = row.get("address")
        if not isinstance(address, Mapping):
            raise AtomRestateNeuralOrbitAuditError("row lacks an exact address")
        address_key = (
            str(address.get("packed_shard_content_sha256")),
            int(address.get("entry_index", -1)),
            int(address.get("progress_index", -1)),
        )
        if address_key in seen_addresses:
            raise AtomRestateNeuralOrbitAuditError(
                f"duplicate teacher progress address {address_key}"
            )
        seen_addresses.add(address_key)
        comparisons = row.get("policy_comparisons")
        if not isinstance(comparisons, list):
            raise AtomRestateNeuralOrbitAuditError("row policy comparisons are absent")
        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                raise AtomRestateNeuralOrbitAuditError("policy comparison must be an object")
            policy_id = str(comparison.get("policy_id"))
            aggregate = policy_aggregates.setdefault(
                policy_id,
                {
                    "policy_kind": comparison.get("policy_kind"),
                    "hypothetical": comparison.get("hypothetical"),
                    "teacher_count": 0,
                    "teacher_outside_policy_count": 0,
                    "ceiling_below_one_count": 0,
                    "minimum_ceiling": None,
                    "retained_mark_count": 0,
                    "exact_ceiling_below_one_addresses": [],
                },
            )
            if aggregate["policy_kind"] != comparison.get("policy_kind") or aggregate[
                "hypothetical"
            ] != comparison.get("hypothetical"):
                raise AtomRestateNeuralOrbitAuditError(
                    f"policy {policy_id!r} changes identity across rows"
                )
            aggregate["teacher_count"] += 1
            aggregate["retained_mark_count"] += int(comparison["retained_mark_count"])
            ceiling = comparison.get("family_conditional_teacher_probability_ceiling")
            if ceiling is None:
                aggregate["teacher_outside_policy_count"] += 1
                continue
            if not isinstance(ceiling, Mapping):
                raise AtomRestateNeuralOrbitAuditError("ceiling must be an object or null")
            value = float(ceiling["value"])
            previous = aggregate["minimum_ceiling"]
            aggregate["minimum_ceiling"] = value if previous is None else min(previous, value)
            if value < 1.0:
                aggregate["ceiling_below_one_count"] += 1
                aggregate["exact_ceiling_below_one_addresses"].append(dict(address))

    return {
        "schema": f"{AUDIT_SCHEMA}.summary",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": AUDIT_STATUS,
        "partition_role": "train_only",
        "authorizes_training": False,
        "selects_support_policy": False,
        "teacher_count": len(row_list),
        "policy_aggregates": {
            policy_id: policy_aggregates[policy_id] for policy_id in sorted(policy_aggregates)
        },
    }


def load_audit_contract(path: Path) -> dict[str, object]:
    """Load the prospective contract with physical and logical fail-loud pins."""

    path = Path(path)
    if not path.is_file():
        raise AtomRestateNeuralOrbitAuditError(f"audit contract is absent: {path}")
    physical = file_sha256(path)
    if PINNED_CONTRACT_FILE_SHA256 and physical != PINNED_CONTRACT_FILE_SHA256:
        raise AtomRestateNeuralOrbitAuditError(
            "atom-restatement orbit contract physical SHA-256 mismatch: "
            f"expected {PINNED_CONTRACT_FILE_SHA256}, observed {physical}"
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AtomRestateNeuralOrbitAuditError("audit contract is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AtomRestateNeuralOrbitAuditError("audit contract must be a JSON object")
    if (
        payload.get("schema") != CONTRACT_SCHEMA
        or payload.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or payload.get("status") != "PROSPECTIVE_DIAGNOSTIC_ONLY_NOT_TRAINING_AUTHORITY"
        or payload.get("partition_role") != "train_only"
        or payload.get("authorizes_training") is not False
        or payload.get("selects_support_policy") is not False
    ):
        raise AtomRestateNeuralOrbitAuditError(
            "audit contract schema, status, or non-authorization boundary is invalid"
        )
    certificate = payload.get("message_passing_certificate")
    if not isinstance(certificate, dict) or certificate != {
        "rounds": DEFAULT_MESSAGE_PASSING_ROUNDS,
        "initial_node_label": [
            "atom_type",
            "formal_charge",
            "implicit_h_count",
            "atom_topology",
        ],
        "round_update": [
            "self_label",
            "multiset_neighbor_labels",
            "multiset_incident_neural_edge_classes",
        ],
        "neighbor_edge_pairing_preserved": False,
        "candidate_class_addition": "target_atom_valence_class_index",
    }:
        raise AtomRestateNeuralOrbitAuditError(
            "message-passing certificate disagrees with the six-round additive encoder"
        )
    declared_policies = payload.get("policies")
    expected_policies = [
        {
            "policy_id": policy.policy_id,
            "policy_kind": policy.policy_kind,
            "hypothetical": policy.hypothetical,
        }
        for policy in DEFAULT_POLICIES
    ]
    if declared_policies != expected_policies:
        raise AtomRestateNeuralOrbitAuditError(
            "contract policies disagree with the prospective diagnostic views"
        )
    logical = payload.get("contract_sha256")
    _require_sha256(logical, field="contract_sha256")
    unhashed = dict(payload)
    unhashed.pop("contract_sha256")
    observed = hashlib.sha256(_canonical_json_bytes(unhashed)).hexdigest()
    if logical != observed:
        raise AtomRestateNeuralOrbitAuditError(
            f"audit contract self-hash mismatch: expected {logical}, observed {observed}"
        )
    return payload


__all__ = [
    "AUDIT_SCHEMA",
    "AUDIT_SCHEMA_VERSION",
    "AUDIT_STATUS",
    "CURRENT_POLICY_ID",
    "DEFAULT_MESSAGE_PASSING_ROUNDS",
    "DEFAULT_POLICIES",
    "HYPOTHETICAL_ACYCLIC_POLICY_ID",
    "HYPOTHETICAL_DECLARED_RING_ELEMENT_POLICY_ID",
    "AtomRestateNeuralOrbitAuditError",
    "RestatePolicy",
    "TeacherProgressAddress",
    "additive_message_orbit_classes",
    "aggregate_teacher_audits",
    "audit_atom_restate_teacher",
    "enumerate_current_broad_restate_candidates",
    "file_sha256",
    "load_audit_contract",
    "molecular_additive_message_orbits",
]
