"""Zero-oracle endpoint-recovery metrics for route-proposal policies.

This module evaluates already generated, ranked proposal pools. It does not fit
or invoke a controller and it never calls a task oracle. Exact endpoint identity
and a bounded local transformation equivalence are reported separately.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass

import networkx as nx

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.docking_value import identity
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "route_proposal_quality_input_v1"
REPORT_SCHEMA = "route_proposal_quality_report_v1"
DEFAULT_CUTOFFS = (1, 5, 10, 32, 128)
CONTEXT_RADIUS = 2
COMPLETE = "complete"


@dataclass(frozen=True)
class TeacherEndpoint:
    teacher_id: str
    endpoint_state: dict

    def __post_init__(self):
        if not self.teacher_id or not isinstance(self.endpoint_state, dict):
            raise ValueError("teacher endpoint requires an identity and exact state")


@dataclass(frozen=True)
class ProposalAttempt:
    attempt_id: str
    rank: int
    status: str
    endpoint_state: dict | None

    def __post_init__(self):
        if not self.attempt_id or type(self.rank) is not int or self.rank < 1:
            raise ValueError("proposal attempt requires an identity and positive rank")
        if self.status == COMPLETE and not isinstance(self.endpoint_state, dict):
            raise ValueError("complete proposal requires an exact endpoint state")
        if self.status != COMPLETE and self.endpoint_state is not None:
            raise ValueError("rejected proposal cannot carry an endpoint state")


def _atom_label(graph, slot):
    if not bool(is_element(graph.atom_types[slot])):
        return None
    return (
        int(graph.atom_types[slot]),
        int(graph.formal_charges[slot]),
        int(graph.implicit_h_counts[slot]),
    )


def _change_neighborhood(source, endpoint, *, radius=CONTEXT_RADIUS):
    """Build an address-free colored graph around the net molecular change.

    The graph includes before/after atom and bond labels, two shells of context,
    and typed boundary stubs. A WL hash is only a lookup key; equivalence always
    requires exact colored graph isomorphism afterward.
    """

    if source.n_atoms != endpoint.n_atoms or type(radius) is not int or radius < 1:
        raise ValueError(
            "transformation comparison requires aligned slots and radius >= 1"
        )
    slots = range(source.n_atoms)
    source_atoms = {slot: _atom_label(source, slot) for slot in slots}
    endpoint_atoms = {slot: _atom_label(endpoint, slot) for slot in slots}
    changed = {
        slot
        for slot in slots
        if source_atoms[slot] != endpoint_atoms[slot]
        or any(
            int(source.bonds[slot, other]) != int(endpoint.bonds[slot, other])
            for other in slots
        )
    }
    if not changed:
        raise ValueError("a transformation-equivalence target cannot be a self event")
    union_neighbors = {
        slot: {
            other
            for other in slots
            if other != slot
            and (
                int(source.bonds[slot, other]) > 0
                or int(endpoint.bonds[slot, other]) > 0
            )
        }
        for slot in slots
    }
    distances = {slot: 0 for slot in changed}
    queue = deque(sorted(changed))
    while queue:
        slot = queue.popleft()
        if distances[slot] == radius:
            continue
        for other in sorted(union_neighbors[slot]):
            if other not in distances:
                distances[other] = distances[slot] + 1
                queue.append(other)
    selected = set(distances)
    graph = nx.Graph()
    for slot in sorted(selected):
        boundary = sorted(
            (
                int(source.bonds[slot, other]),
                int(endpoint.bonds[slot, other]),
                source_atoms[other],
                endpoint_atoms[other],
            )
            for other in union_neighbors[slot] - selected
        )
        label = json.dumps(
            {
                "changed": slot in changed,
                "distance": distances[slot],
                "before": source_atoms[slot],
                "after": endpoint_atoms[slot],
                "boundary": boundary,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        graph.add_node(slot, label=label)
    for left in sorted(selected):
        for right in sorted(selected):
            if left >= right:
                continue
            before = int(source.bonds[left, right])
            after = int(endpoint.bonds[left, right])
            if before or after:
                graph.add_edge(left, right, label=f"{before}>{after}")
    graph.graph["hash"] = nx.weisfeiler_lehman_graph_hash(
        graph, node_attr="label", edge_attr="label", iterations=4
    )
    return graph


def transformation_equivalent(source, left, right, *, radius=CONTEXT_RADIUS):
    """Exact colored-isomorphism check after a non-authoritative WL prefilter."""

    first = _change_neighborhood(source, left, radius=radius)
    second = _change_neighborhood(source, right, radius=radius)
    if first.graph["hash"] != second.graph["hash"]:
        return False
    return nx.is_isomorphic(
        first,
        second,
        node_match=lambda a, b: a["label"] == b["label"],
        edge_match=lambda a, b: a["label"] == b["label"],
    )


def _first_rank(matches, proposals):
    return min(
        (proposal.rank for proposal in proposals if matches(proposal)), default=None
    )


def _mean(rows, name):
    return float(sum(row[name] for row in rows) / len(rows)) if rows else 0.0


def _policy_metrics(cases, policy_id, *, cutoffs, radius):
    per_source = []
    total_attempts = total_complete = total_unique = 0
    total_seconds = 0.0
    for case in cases:
        source = decode_state(case["source_state"])
        teachers = [
            TeacherEndpoint(row["teacher_id"], row["endpoint_state"])
            for row in case["teachers"]
        ]
        if not teachers:
            raise ValueError(f"source has no teacher targets: {case['source_id']}")
        pool = next(
            (row for row in case["policy_pools"] if row["policy_id"] == policy_id),
            None,
        )
        if pool is None:
            raise ValueError(f"source is missing policy pool {policy_id!r}")
        attempts = [ProposalAttempt(**row) for row in pool["attempts"]]
        seconds = float(pool["proposal_seconds"])
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("proposal seconds must be finite and nonnegative")
        ranks = [row.rank for row in attempts]
        if len(set(ranks)) != len(ranks) or ranks != sorted(ranks):
            raise ValueError("proposal ranks must be unique and sorted")
        complete = [row for row in attempts if row.status == COMPLETE]
        seen, proposals = set(), []
        for proposal in complete:
            endpoint = decode_state(proposal.endpoint_state)
            key = canonical_state_key(endpoint)
            if key == canonical_state_key(source):
                raise ValueError("complete proposal cannot be a canonical self event")
            if key not in seen:
                seen.add(key)
                proposals.append(proposal)
        teacher_graphs = [decode_state(row.endpoint_state) for row in teachers]
        teacher_exact = list(
            dict.fromkeys(canonical_state_key(graph) for graph in teacher_graphs)
        )
        teacher_equivalence_classes = []
        for graph in teacher_graphs:
            if not any(
                transformation_equivalent(source, graph, representative, radius=radius)
                for representative in teacher_equivalence_classes
            ):
                teacher_equivalence_classes.append(graph)
        exact_ranks = []
        for endpoint in teacher_exact:
            exact_ranks.append(
                _first_rank(
                    lambda proposal, target=endpoint: canonical_state_key(
                        decode_state(proposal.endpoint_state)
                    )
                    == target,
                    proposals,
                )
            )
        equivalent_ranks = []
        for target in teacher_equivalence_classes:
            equivalent_ranks.append(
                _first_rank(
                    lambda proposal, target=target, source_graph=source: transformation_equivalent(
                        source_graph,
                        decode_state(proposal.endpoint_state),
                        target,
                        radius=radius,
                    ),
                    proposals,
                )
            )
        row = {
            "source_id": case["source_id"],
            "teacher_endpoints": len(teacher_exact),
            "teacher_transformation_classes": len(teacher_equivalence_classes),
            "attempts": len(attempts),
            "complete": len(complete),
            "unique_complete": len(proposals),
            "proposal_seconds": seconds,
            "cutoffs": {},
        }
        for cutoff in cutoffs:
            emitted = [proposal for proposal in proposals if proposal.rank <= cutoff]
            exact_hits = sum(
                rank is not None and rank <= cutoff for rank in exact_ranks
            )
            equivalent_hits = sum(
                rank is not None and rank <= cutoff for rank in equivalent_ranks
            )
            matching_exact = sum(
                canonical_state_key(decode_state(proposal.endpoint_state))
                in set(teacher_exact)
                for proposal in emitted
            )
            matching_equivalent = sum(
                any(
                    transformation_equivalent(
                        source,
                        decode_state(proposal.endpoint_state),
                        target,
                        radius=radius,
                    )
                    for target in teacher_equivalence_classes
                )
                for proposal in emitted
            )
            row["cutoffs"][str(cutoff)] = {
                "exact_recall": exact_hits / len(teacher_exact),
                "transformation_recall": equivalent_hits
                / len(teacher_equivalence_classes),
                "exact_any": float(exact_hits > 0),
                "transformation_any": float(equivalent_hits > 0),
                "exact_precision_emitted": matching_exact / max(1, len(emitted)),
                "transformation_precision_emitted": matching_equivalent
                / max(1, len(emitted)),
                "exact_precision_fixed_k": matching_exact / cutoff,
                "transformation_precision_fixed_k": matching_equivalent / cutoff,
                "unique_yield_fixed_k": len(emitted) / cutoff,
            }
        row["exact_mrr"] = sum(
            0 if rank is None else 1 / rank for rank in exact_ranks
        ) / len(exact_ranks)
        row["transformation_mrr"] = sum(
            0 if rank is None else 1 / rank for rank in equivalent_ranks
        ) / len(equivalent_ranks)
        per_source.append(row)
        total_attempts += len(attempts)
        total_complete += len(complete)
        total_unique += len(proposals)
        total_seconds += seconds
    aggregate = {
        "sources": len(per_source),
        "source_balanced_exact_mrr": _mean(per_source, "exact_mrr"),
        "source_balanced_transformation_mrr": _mean(per_source, "transformation_mrr"),
        "execution_precision": total_complete / max(1, total_attempts),
        "unique_endpoint_yield": total_unique / max(1, total_attempts),
        "attempts": total_attempts,
        "complete": total_complete,
        "unique_complete": total_unique,
        "proposal_seconds": total_seconds,
        "attempts_per_second": total_attempts / max(total_seconds, 1e-12),
        "unique_endpoints_per_second": total_unique / max(total_seconds, 1e-12),
        "cutoffs": {},
    }
    for cutoff in cutoffs:
        name = str(cutoff)
        aggregate["cutoffs"][name] = {
            metric: float(
                sum(row["cutoffs"][name][metric] for row in per_source)
                / len(per_source)
            )
            for metric in per_source[0]["cutoffs"][name]
        }
    return {"aggregate": aggregate, "per_source": per_source}


def evaluate(payload, *, cutoffs=DEFAULT_CUTOFFS, radius=CONTEXT_RADIUS):
    """Evaluate frozen proposal artifacts under source-balanced zero-oracle metrics."""

    if payload.get("schema_version") != SCHEMA or payload.get("oracle_calls") != 0:
        raise ValueError(
            "route-proposal benchmark requires its zero-oracle input schema"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("route-proposal benchmark requires nonempty source cases")
    source_ids = [row.get("source_id") for row in cases]
    if any(not source for source in source_ids) or len(set(source_ids)) != len(
        source_ids
    ):
        raise ValueError("source cases require unique nonempty identities")
    split = payload.get("split")
    if not isinstance(split, dict) or split.get("evaluation_role") not in {
        "calibration",
        "test",
    }:
        raise ValueError("a frozen calibration/test source split is required")
    partitions = {
        role: tuple(split.get(f"{role}_sources", ()))
        for role in ("train", "calibration", "test")
    }
    flattened = [source for values in partitions.values() for source in values]
    if (
        any(not isinstance(source, str) or not source for source in flattened)
        or len(flattened) != len(set(flattened))
        or set(source_ids) != set(partitions[split["evaluation_role"]])
    ):
        raise ValueError("source partitions must be disjoint and match evaluated cases")
    cutoffs = tuple(cutoffs)
    if (
        not cutoffs
        or any(type(value) is not int or value < 1 for value in cutoffs)
        or tuple(sorted(set(cutoffs))) != cutoffs
    ):
        raise ValueError("rank cutoffs must be sorted unique positive integers")
    for case in cases:
        names = [row.get("policy_id") for row in case.get("policy_pools", [])]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(
                "policy pools require unique nonempty identities per source"
            )
    policy_sets = [
        {row["policy_id"] for row in case.get("policy_pools", [])} for case in cases
    ]
    if (
        not policy_sets
        or not policy_sets[0]
        or any(row != policy_sets[0] for row in policy_sets)
    ):
        raise ValueError("every source must contain the same nonempty policy set")
    policies = {
        policy_id: _policy_metrics(cases, policy_id, cutoffs=cutoffs, radius=radius)
        for policy_id in sorted(policy_sets[0])
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "evidence": "offline zero-oracle proposal recovery; not task optimization",
        "split": split,
        "context_radius": radius,
        "cutoffs": list(cutoffs),
        "policies": policies,
        "oracle_calls": 0,
        "equivalence_contract": {
            "exact_endpoint": "canonical supported molecular graph identity",
            "transformation": (
                "exact colored graph isomorphism over before/after atom and bond labels, "
                "radius-2 union-graph context and typed boundary stubs"
            ),
            "wl_hash_is_authoritative": False,
            "stereochemistry_supported": False,
        },
        "limitations": [
            "Teacher recovery is answer-known offline evidence, not autonomous score improvement.",
            "Transformation equivalence is local to the declared radius and supported graph fields.",
            "A high module-label rank alone is not an endpoint-recovery result.",
        ],
    }
    return {**report, "report_sha256": identity(report)}
