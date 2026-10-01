"""Generic legal Active8 successor fibers and target-free WHERE/HOW ranking.\n\nThis module is benchmark-independent. It exposes exact canonical rewrite fibers\nthat can be reused by route-distilled T4 and PMO decoders.\n"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import (
    NULL_IDX,
    ORGANIC_VOCABULARY,
    is_element,
)
from compose_v4.control.route_distilled_program_policy import stage_descriptor
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.factorized_fiber import enumerate_pendant_graft_actions
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    MICRO_BOND_CLASSES,
    AtomDelete,
    AtomInsert,
    BondReorder,
    enumerate_cycle_close_edges,
    enumerate_cycle_open_edges,
    enumerate_semantic_atom_restates,
)
from compose_v4.rewrite.tracelet_fiber import (
    enumerate_ring_system_restate_actions,
)

SCHEMA = "generic_legal_action_policy_v1"
CHECKPOINT_SCHEMA = "generic_legal_action_ranker_v1"
RULES = (
    "atom_delete",
    "atom_insert",
    "atom_restate_semantic",
    "bond_reorder",
    "bond_reroute",
    "cycle_close",
    "cycle_open",
    "ring_system_restate",
)


@dataclass(frozen=True)
class LegalSuccessor:
    rule: str
    action_record: dict
    successor: object
    successor_key: str


def _record_parts(record: dict, graph) -> tuple[set[int], set[int], int | None, int | None]:
    """Return the generic action footprint, references, created and deleted slots."""

    rule, payload = record["executor_rule"], record["payload"]
    created = deleted = None
    if rule == "atom_insert":
        created = int(payload["slot"])
        references = {int(row[0]) for row in payload["neighbors"]}
        footprint = {*references, created}
    elif rule in {"atom_delete", "atom_restate_semantic"}:
        atom = int(payload["v"])
        references = {atom}
        footprint = {atom}
        if rule == "atom_delete":
            deleted = atom
            footprint.update(int(value) for value in np.flatnonzero(graph.bonds[atom]))
    elif rule in {"cycle_close", "cycle_open", "bond_reorder"}:
        references = {int(payload["a"]), int(payload["b"])}
        footprint = set(references)
    elif rule == "bond_reroute":
        references = {int(payload[name]) for name in ("a", "b", "u", "v")}
        footprint = set(references)
    elif rule == "ring_system_restate":
        references = {int(change[name]) for change in payload["changes"] for name in ("a", "b")}
        footprint = set(references)
    else:
        raise ValueError(f"unsupported primitive rule: {rule!r}")
    return footprint, references, created, deleted


@dataclass(frozen=True)
class RankerConfig:
    negatives_per_teacher: int = 32
    variance_floor: float = 0.05
    coefficient_norm: float = 4.0

    def __post_init__(self):
        if self.negatives_per_teacher < 1:
            raise ValueError("negatives_per_teacher must be positive")
        if self.variance_floor <= 0 or self.coefficient_norm <= 0:
            raise ValueError("ranker scales must be positive")


def _candidate_actions(graph, rule: str):
    real = tuple(int(slot) for slot in np.flatnonzero(is_element(graph.atom_types)))
    system = editing_v2_semantic_rewrite_system()
    if rule == "atom_delete":
        yield from (AtomDelete(slot) for slot in real)
        return
    if rule == "atom_insert":
        empty = np.flatnonzero(graph.atom_types == NULL_IDX)
        if not len(empty) or graph.n_real_atoms >= 40:
            return
        slot = int(empty[0])
        for neighbor in real:
            for order in MICRO_BOND_CLASSES:
                for class_index in range(len(ORGANIC_VOCABULARY)):
                    hydrogen = ORGANIC_VOCABULARY.h_count(class_index, order, 0)
                    if hydrogen is not None:
                        yield AtomInsert(
                            slot,
                            ORGANIC_VOCABULARY.element_of(class_index),
                            0,
                            hydrogen,
                            ((neighbor, order),),
                        )
        return
    if rule == "atom_restate_semantic":
        yield from enumerate_semantic_atom_restates(graph)
        return
    if rule == "bond_reorder":
        for left, right in combinations(real, 2):
            current = int(graph.bonds[left, right])
            if current:
                for order in MICRO_BOND_CLASSES:
                    if order != current:
                        yield BondReorder(left, right, order)
        return
    if rule == "bond_reroute":
        yield from enumerate_pendant_graft_actions(graph)
        return
    if rule == "cycle_close":
        yield from enumerate_cycle_close_edges(graph)
        return
    if rule == "cycle_open":
        yield from enumerate_cycle_open_edges(graph)
        return
    if rule == "ring_system_restate":
        yield from enumerate_ring_system_restate_actions(graph, system=system)
        return
    raise ValueError(f"unsupported Active8 rule: {rule}")


def enumerate_rule_successors(graph, rule: str) -> tuple[LegalSuccessor, ...]:
    """Enumerate one generic rule fiber and quotient canonical successors."""

    if rule not in RULES:
        raise ValueError(f"unsupported Active8 rule: {rule}")
    system = editing_v2_semantic_rewrite_system()
    source_key = canonical_state_key(graph)
    successors = {}
    for action in _candidate_actions(graph, rule):
        try:
            successor = system.apply(graph, rule, action)
        except (InvalidRewrite, ValueError):
            continue
        key = canonical_state_key(successor)
        if key == source_key or key in successors:
            continue
        successors[key] = LegalSuccessor(
            rule=rule,
            action_record=encode_action(rule, action),
            successor=successor,
            successor_key=key,
        )
    return tuple(successors[key] for key in sorted(successors))


def enumerate_legal_successors(graph) -> tuple[LegalSuccessor, ...]:
    return tuple(
        candidate for rule in RULES for candidate in enumerate_rule_successors(graph, rule)
    )


def _cycle_slots(graph) -> set[int]:
    network = nx.Graph()
    real = tuple(int(slot) for slot in np.flatnonzero(is_element(graph.atom_types)))
    network.add_nodes_from(real)
    network.add_edges_from(
        (left, right) for left, right in combinations(real, 2) if int(graph.bonds[left, right])
    )
    return {slot for cycle in nx.cycle_basis(network) for slot in cycle}


def action_features(graph, candidate: LegalSuccessor) -> np.ndarray:
    """Address-free global and structural features for one exact legal successor."""

    record = candidate.action_record
    footprint, references, created, deleted = _record_parts(record, graph)
    del footprint
    cycle_slots = _cycle_slots(graph)
    elements = np.zeros(10, dtype=float)
    degree = np.zeros(5, dtype=float)
    bond_orders = np.zeros(3, dtype=float)
    hydrogens, charges, cyclic = [], [], []
    for slot in references:
        atom_type = int(graph.atom_types[slot])
        if 1 <= atom_type <= 10:
            elements[atom_type - 1] += 1
        observed_degree = int(np.count_nonzero(graph.bonds[slot]))
        degree[min(observed_degree, 4)] += 1
        hydrogens.append(int(graph.implicit_h_counts[slot]) / 4)
        charges.append(int(graph.formal_charges[slot]) / 2)
        cyclic.append(float(slot in cycle_slots))
        for order in graph.bonds[slot]:
            if 1 <= int(order) <= 3:
                bond_orders[int(order) - 1] += 1
    denominator = max(1, len(references))
    elements /= denominator
    degree /= denominator
    bond_orders /= max(1, bond_orders.sum())
    rule = np.zeros(len(RULES), dtype=float)
    rule[RULES.index(candidate.rule)] = 1.0
    payload = record["payload"]
    target_elements = np.zeros(10, dtype=float)
    target_class = 0.0
    target_hydrogen = 0.0
    target_order = 0.0
    if candidate.rule == "atom_insert":
        atom_type = int(payload["atom_type"])
        if 1 <= atom_type <= 10:
            target_elements[atom_type - 1] = 1.0
        target_hydrogen = int(payload["implicit_h_count"]) / 4
        target_order = int(payload["neighbors"][0][1]) / 3
    elif candidate.rule == "atom_restate_semantic":
        class_index = int(payload["target_class_index"])
        atom_type = ORGANIC_VOCABULARY.element_of(class_index)
        if 1 <= atom_type <= 10:
            target_elements[atom_type - 1] = 1.0
        target_class = (class_index + 1) / len(ORGANIC_VOCABULARY)
    elif candidate.rule in {"bond_reorder", "cycle_close"}:
        target_order = int(payload.get("new_order", payload.get("order", 0))) / 3
    local = np.concatenate(
        (
            rule,
            elements,
            degree,
            bond_orders,
            target_elements,
            np.asarray(
                [
                    len(references) / 4,
                    np.mean(hydrogens) if hydrogens else 0.0,
                    np.mean(charges) if charges else 0.0,
                    np.mean(cyclic) if cyclic else 0.0,
                    float(created is not None),
                    float(deleted is not None),
                    (candidate.successor.n_real_atoms - graph.n_real_atoms) / 8,
                    target_class,
                    target_hydrogen,
                    target_order,
                ],
                dtype=float,
            ),
        )
    )
    result = np.concatenate(
        (
            molecule_features(canonical_state_key(graph)).astype(float),
            stage_descriptor(graph, (record,)),
            local,
        )
    )
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite legal-action features")
    return result


def fit_diagonal_contrastive_ranker(
    differences: list[tuple[np.ndarray, float]], config: RankerConfig
) -> dict:
    """Fit a deterministic diagonal, source-balanced contrastive scorer."""

    if not differences:
        raise ValueError("legal-action ranker requires contrastive differences")
    matrix = np.stack([row[0] for row in differences])
    weights = np.asarray([row[1] for row in differences], dtype=float)
    weights /= weights.sum()
    mean = np.sum(matrix * weights[:, None], axis=0)
    variance = np.sum((matrix - mean) ** 2 * weights[:, None], axis=0)
    scale = np.maximum(np.sqrt(variance), config.variance_floor)
    coefficients = mean / scale
    norm = float(np.linalg.norm(coefficients))
    if norm:
        coefficients *= config.coefficient_norm / norm
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "feature_mean": [0.0] * matrix.shape[1],
        "feature_scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "training_pairs": len(differences),
    }


def score_features(checkpoint: dict, features: np.ndarray) -> float:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("legal-action checkpoint schema mismatch")
    scale = np.asarray(checkpoint["feature_scale"], dtype=float)
    coefficients = np.asarray(checkpoint["coefficients"], dtype=float)
    value = np.asarray(features, dtype=float)
    if value.shape != scale.shape or value.shape != coefficients.shape:
        raise ValueError("legal-action feature dimension mismatch")
    return float((value / scale) @ coefficients)


def rank_of_teacher(
    candidates: tuple[LegalSuccessor, ...],
    teacher_successor_key: str,
    scores: dict[str, float] | None = None,
) -> int | None:
    return rank_successor_keys(
        tuple(candidate.successor_key for candidate in candidates),
        teacher_successor_key,
        scores,
    )


def rank_successor_keys(
    successor_keys: tuple[str, ...],
    teacher_successor_key: str,
    scores: dict[str, float] | None = None,
) -> int | None:
    """Rank one canonical successor without retaining its molecular graph."""

    ordered = sorted(
        successor_keys,
        key=((lambda key: (-scores[key], key)) if scores is not None else (lambda key: key)),
    )
    for rank, successor_key in enumerate(ordered, start=1):
        if successor_key == teacher_successor_key:
            return rank
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1 / rank


def log_rule_uniform_probability(rule_probability: float, fiber_size: int) -> float:
    if not 0 < rule_probability <= 1 or fiber_size < 1:
        return float("-inf")
    return math.log(rule_probability) - math.log(fiber_size)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "RULES",
    "SCHEMA",
    "LegalSuccessor",
    "RankerConfig",
    "action_features",
    "enumerate_legal_successors",
    "enumerate_rule_successors",
    "fit_diagonal_contrastive_ranker",
    "log_rule_uniform_probability",
    "rank_of_teacher",
    "rank_successor_keys",
    "reciprocal_rank",
    "score_features",
]
