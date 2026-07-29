"""A2.1: measure bounded chemistries and select an exact-control benchmark by a preregistered rule.

"Enumerable" is an EMPIRICAL property of the real executor, not something to assert from a small atom cap.
So this module measures candidates and applies the frozen selection rule from
``configs/experiment_registry.yaml -> exact_sizing``; it never chooses a graph because a controller looked
better on it.

What governs size (measured, not assumed)
----------------------------------------
The reachable set is bounded by the seed state's ATOM-SLOT COUNT, not by the declared chemistry.
``atom_insert`` fills a NULL slot; it cannot extend the state array. Measured from cyclopropane under
carbon-only chemistry: 14 states reachable, atom counts ``{0:1, 1:1, 2:3, 3:9}`` -- ``atom_insert`` fired 18
times without ever producing a 4-atom molecule, because every insertion merely refilled a slot a prior
``atom_delete`` had emptied. Consequently the slot count is the size dial:

    slots 4 ->    51 states,    374 edges
    slots 5 ->   197 states,  2,140 edges
    slots 6 ->   967 states, 14,432 edges

roughly 5x per added slot. Both endpoints of a shared component close to the SAME GRAPH, not merely the same
counts: ``CCCCCC`` and ``C1CCCCC1`` share the fingerprint ``84121ff86cbc1ba8`` from different BFS profiles.

Reuse boundary
--------------
Transitions come from PRODUCTION enumerators and the PRODUCTION canonical key:
``rewrite.fiber.enumerate_action_fiber`` for every non-null state (it routes each candidate through the
production runtime and drops identity transitions), and ``factorized_fiber._factorized_candidates`` for the
null state's root insertions, filtered to the declared element vocabulary. The reachability search,
statistics, fingerprint, non-degeneracy tests and selection rule are written here from scratch. No prior
exact-control script is imported.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    MolecularGraph,
    smiles_to_molecular_graph,
)
from compose_v4.rewrite.fiber import ActionFiberSpec, AtomState, enumerate_action_fiber
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system

# The zero-atom state, whose canonical key the production canonicalizer special-cases.
NULL_KEY = "<NULL>"

# ---- the empty-state boundary (resolved by audit, not by solver convenience) --------------------------
#
# Audited against the real production enumerator and executor:
#
#   * ``canonical_state_key`` has an explicit ``<NULL>`` branch, so the empty state is anticipated;
#   * ``is_rdkit_valid(empty)`` returns True and ``molecular_graph_to_smiles(empty)`` returns "" -- an empty
#     SMILES parses to a zero-atom Mol, so validity does NOT block it;
#   * deleting the final atom IS legally exposed: the production factorized enumerator on 1-atom "C" offers
#     4 candidates, one of which empties the molecule;
#   * from the empty state the production enumerator offers 4 ``atom_insert`` root insertions, all of which
#     execute, giving {C, N, O, F}.
#
# So PRODUCTION treats the empty graph as a REVERSIBLE SOURCE (delete in, root-insert out), not a cemetery.
# That is coherent and is not a defect -- and it means a de-novo generative process may legitimately start
# from the empty state.
#
# RESOLUTION (owner decision): the empty graph is INCLUDED as a distinguished reversible source, with its
# root insertion restricted to the declared element vocabulary -- the same restriction already applied to
# every other action in the slice. E6 is a carbon-only six-slot restriction of production, so from the empty
# state the restriction admits
#
#     null -> C          and omits          null -> {N, O, F}
#
# giving the closed slice
#
#     S_{C,6} = {null} u { G : 1 <= |V(G)| <= 6, G connected, valence-valid, carbon-only }
#
# in which null <-> C is a genuine two-way production-derived transition. This is a VOCABULARY-RESTRICTED
# PRODUCTION KERNEL, not a solver-only exception.
#
# Excluding null was rejected because it would delete a transition production genuinely exposes (C -> null)
# and impose an artificial reflecting boundary at one atom. The upper six-slot bound is forced by the
# representation; the lower bound is not -- production intentionally permits both death to null and birth
# from it. Admitting N/O/F was rejected because it would convert the benchmark to a mixed-element chemistry,
# requiring resizing and yielding less interpretable exact results, with nothing gained for the theorem.
#
# Terminology, which the manuscript must keep honest: the empty graph is NOT a molecule. Every NON-NULL
# committed state is a valid connected molecule, with a distinguished null source enabling trans-dimensional
# generation from zero atoms. Chemical descriptors and topology summaries EXCLUDE null; probability
# calculations INCLUDE it.
EMPTY_STATE_POLICY = "included_as_distinguished_source"
EMPTY_STATE_PRODUCTION_SEMANTICS = "reversible_source"

# Executor rule names, and the structural role each plays in the non-degeneracy conditions.
ATOM_BIRTH_RULES = frozenset({"atom_insert"})
ATOM_DEATH_RULES = frozenset({"atom_delete"})
# RingCore-V1: ring topology changes are compositional cycle_close/cycle_open, whose EXECUTOR names are
# bond_insert/bond_delete. The dense head scores them as cycle_insert/cycle_attach.
CYCLE_RULES = frozenset({"bond_insert", "bond_delete"})


class SizingInfeasible(RuntimeError):
    """No candidate satisfied the preregistered window and non-degeneracy conditions."""


@dataclass(frozen=True)
class Candidate:
    """One bounded chemistry to measure. ``candidate_id`` breaks selection ties lexicographically."""

    candidate_id: str
    seed_smiles: str
    elements: tuple[str, ...]
    max_hydrogens: int
    horizon: int | None = None  # None = expand to closure

    def spec(self) -> ActionFiberSpec:
        return ActionFiberSpec(
            atom_states=tuple(
                AtomState(ELEMENT_TO_IDX[element], 0, hydrogens)
                for element in self.elements
                for hydrogens in range(self.max_hydrogens + 1)
            )
        )


@dataclass
class ReachableGraph:
    """The K-step reachable set under the real executor, with canonical merging."""

    states: dict[str, MolecularGraph]
    edges: set[tuple[str, str]]
    layer_sizes: list[int]
    rule_counts: Counter
    out_degree: Counter
    depth: int
    stop_reason: str
    seconds: float
    # successor key -> number of DISTINCT one-step parents, for the multi-path condition
    parent_counts: Counter = field(default_factory=Counter)

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def _successor_elements(key: str) -> set[str]:
    """Element symbols present in a canonical key, for the declared-vocabulary restriction."""
    if key == NULL_KEY:
        return set()
    from rdkit import Chem  # noqa: PLC0415

    molecule = Chem.MolFromSmiles(key, sanitize=False)
    if molecule is None:
        return {"?"}
    return {atom.GetSymbol() for atom in molecule.GetAtoms()}


def root_insertion_transitions(
    empty_state: MolecularGraph, candidate: Candidate, system
) -> list[Any]:
    """Production root insertions from the empty state, restricted to the declared element vocabulary.

    The E6 fiber language cannot supply these: its atom insertions attach to an existing atom, so it returns
    nothing from the empty state. The PRODUCTION factorized enumerator does offer them (4 candidates ->
    {C, N, O, F}), so they are taken from production and then filtered by the same element restriction that
    governs every other action in the slice. Filtering on the SUCCESSOR rather than on the action payload
    keeps this robust to the payload representation.
    """
    from compose_v4.rewrite.factorized_fiber import _factorized_candidates  # noqa: PLC0415
    from compose_v4.rewrite.fiber import MarkedTransition  # noqa: PLC0415

    allowed = set(candidate.elements)
    transitions = []
    for rule_name, action in _factorized_candidates(empty_state, allow_bond_reroute=False):
        try:
            successor = system.apply(empty_state, rule_name, action)
        except Exception:
            continue
        key = canonical_state_key(successor)
        if key == NULL_KEY:
            continue
        if _successor_elements(key) <= allowed:
            transitions.append(MarkedTransition(rule_name, action, successor, key))
    return transitions


def graph_fingerprint(graph: ReachableGraph) -> str:
    """Deterministic identity of the exact graph: sorted canonical states plus sorted directed edges.

    A2.2 and every downstream exact-control result cite this, so a graph that changes for any reason -- a
    support-rule correction, an operator change -- gets a visibly different identity instead of silently
    replacing an earlier one.
    """
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    for key in sorted(graph.states):
        digest.update(b"S")
        digest.update(key.encode())
    for source, target in sorted(graph.edges):
        digest.update(b"E")
        digest.update(source.encode())
        digest.update(b">")
        digest.update(target.encode())
    return digest.hexdigest()[:16]


def build_reachable_graph(
    candidate: Candidate,
    *,
    state_cap: int,
    edge_cap: int,
    deadline_seconds: float | None = None,
) -> ReachableGraph:
    """Breadth-first expansion to ``candidate.horizon`` (or closure), merging identical molecules.

    Stops on closure, the horizon, a cap, or a deadline, and reports WHICH -- a truncated search must never
    be mistaken for a closed one.
    """
    spec = candidate.spec()
    system = de_novo_rewrite_system()
    start = smiles_to_molecular_graph(candidate.seed_smiles)
    start_key = canonical_state_key(start)

    states: dict[str, MolecularGraph] = {start_key: start}
    edges: set[tuple[str, str]] = set()
    rule_counts: Counter = Counter()
    out_degree: Counter = Counter()
    parents: dict[str, set[str]] = {}
    layer_sizes = [1]
    frontier = [start_key]
    started = time.monotonic()
    stop_reason = "closed"
    depth = 0

    while frontier:
        if candidate.horizon is not None and depth >= candidate.horizon:
            stop_reason = "horizon"
            break
        depth += 1
        next_frontier: list[str] = []
        for key in frontier:
            if deadline_seconds is not None and time.monotonic() - started > deadline_seconds:
                stop_reason = "deadline"
                break
            if len(states) >= state_cap:
                stop_reason = "state_cap"
                break
            if len(edges) >= edge_cap:
                stop_reason = "edge_cap"
                break
            if key == NULL_KEY:
                # Distinguished source: production root insertions, restricted to the declared
                # vocabulary. The E6 fiber attaches to an existing atom and yields nothing here.
                transitions = root_insertion_transitions(states[key], candidate, system)
            else:
                transitions = enumerate_action_fiber(states[key], spec=spec, system=system)
            out_degree[key] = len({transition.successor_key for transition in transitions})
            for transition in transitions:
                rule_counts[transition.rule_name] += 1
                edges.add((key, transition.successor_key))
                parents.setdefault(transition.successor_key, set()).add(key)
                if transition.successor_key not in states:
                    states[transition.successor_key] = transition.successor
                    next_frontier.append(transition.successor_key)
        if stop_reason != "closed":
            break
        layer_sizes.append(len(next_frontier))
        frontier = next_frontier

    return ReachableGraph(
        states=states,
        edges=edges,
        layer_sizes=layer_sizes,
        rule_counts=rule_counts,
        out_degree=out_degree,
        depth=depth,
        stop_reason=stop_reason,
        seconds=round(time.monotonic() - started, 3),
        parent_counts=Counter({key: len(value) for key, value in parents.items()}),
    )


def cycle_rank(state: MolecularGraph, key: str | None = None) -> int:
    """First Betti number: ``bonds - atoms + components``. Zero for the null state.

    Computed directly from the graph, NOT from a ring-perception count. RDKit's ``CalcNumRings`` is the
    SYMMETRIZED SSSR count, which over-counts symmetry-equivalent rings in bridged systems: for
    bicyclo[2.2.2]octane (8 atoms, 9 bonds) the cycle rank is 9 - 8 + 1 = 2 but ``CalcNumRings`` returns 3.
    An earlier version of this function used that count and therefore mis-reported the cycle-rank
    distribution for every bridged state. The Betti number needs no ring perception and is exact.
    """
    resolved = key if key is not None else canonical_state_key(state)
    if resolved == NULL_KEY:
        return 0
    from rdkit import Chem  # noqa: PLC0415

    molecule = Chem.MolFromSmiles(resolved, sanitize=False)
    if molecule is None:
        return 0
    components = len(Chem.GetMolFrags(molecule))
    return int(molecule.GetNumBonds() - molecule.GetNumAtoms() + components)


def graph_statistics(graph: ReachableGraph) -> dict[str, Any]:
    """Everything the sizing report must contain for a candidate to be auditable."""
    atom_counts = Counter(
        state.n_real_atoms for key, state in graph.states.items() if key != NULL_KEY
    )
    ranks = Counter(
        cycle_rank(state, key) for key, state in graph.states.items() if key != NULL_KEY
    )
    degrees = sorted(graph.out_degree.values())
    multi_path = sum(1 for count in graph.parent_counts.values() if count >= 2)
    return {
        "graph_fingerprint": graph_fingerprint(graph),
        "empty_state_policy": EMPTY_STATE_POLICY,
        "empty_state_production_semantics": EMPTY_STATE_PRODUCTION_SEMANTICS,
        "n_states": graph.n_states,
        "n_states_excluding_null": sum(1 for key in graph.states if key != NULL_KEY),
        "n_edges": graph.n_edges,
        "depth": graph.depth,
        "layer_sizes": list(graph.layer_sizes),
        "stop_reason": graph.stop_reason,
        "seconds": graph.seconds,
        "null_state_reachable": NULL_KEY in graph.states,
        "atom_count_distribution": dict(sorted(atom_counts.items())),
        "cycle_rank_distribution": dict(sorted(ranks.items())),
        "rule_counts": dict(sorted(graph.rule_counts.items())),
        "branching": {
            "min": degrees[0] if degrees else 0,
            "median": degrees[len(degrees) // 2] if degrees else 0,
            "max": degrees[-1] if degrees else 0,
            "mean": round(sum(degrees) / len(degrees), 3) if degrees else 0.0,
        },
        "states_with_multiple_distinct_parents": multi_path,
    }


def non_degeneracy(statistics: dict[str, Any]) -> dict[str, bool]:
    """The six preregistered conditions, each reported separately.

    ``multi_path_terminal_event`` is load-bearing: with a single route to each target, several controllers
    become indistinguishable and the exact panel would measure nothing.
    """
    rules = statistics["rule_counts"]
    return {
        "atom_birth": any(rules.get(name, 0) > 0 for name in ATOM_BIRTH_RULES),
        "atom_death": any(rules.get(name, 0) > 0 for name in ATOM_DEATH_RULES),
        "cycle_change": any(rules.get(name, 0) > 0 for name in CYCLE_RULES),
        "two_distinct_atom_counts": len(statistics["atom_count_distribution"]) >= 2,
        "two_distinct_cycle_ranks": len(statistics["cycle_rank_distribution"]) >= 2,
        "multi_path_terminal_event": statistics["states_with_multiple_distinct_parents"] > 0,
    }


def evaluate_candidate(
    candidate: Candidate,
    *,
    n_min: int,
    n_max: int,
    e_max: int,
    deadline_seconds: float | None = None,
) -> dict[str, Any]:
    """Measure one candidate and record whether it qualifies, with an explicit reason either way."""
    graph = build_reachable_graph(
        candidate,
        state_cap=n_max + 1,  # +1 so exceeding the window is observable rather than truncated at it
        edge_cap=e_max + 1,
        deadline_seconds=deadline_seconds,
    )
    statistics = graph_statistics(graph)
    conditions = non_degeneracy(statistics)

    reasons: list[str] = []
    if statistics["stop_reason"] not in {"closed", "horizon"}:
        reasons.append(f"search did not complete: {statistics['stop_reason']}")
    if statistics["n_states"] < n_min:
        reasons.append(f"n_states {statistics['n_states']} < N_min {n_min}")
    if statistics["n_states"] > n_max:
        reasons.append(f"n_states {statistics['n_states']} > N_max {n_max}")
    if statistics["n_edges"] > e_max:
        reasons.append(f"n_edges {statistics['n_edges']} > E_max {e_max}")
    unmet = sorted(name for name, held in conditions.items() if not held)
    if unmet:
        reasons.append(f"non-degeneracy conditions unmet: {unmet}")

    return {
        "candidate_id": candidate.candidate_id,
        "seed_smiles": candidate.seed_smiles,
        "elements": list(candidate.elements),
        "max_hydrogens": candidate.max_hydrogens,
        "horizon": candidate.horizon,
        "atom_slots": len(smiles_to_molecular_graph(candidate.seed_smiles).atom_types),
        "statistics": statistics,
        "non_degeneracy": conditions,
        "qualifies": not reasons,
        "rejection_reasons": reasons,
    }


def select_candidate(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered rule: SMALLEST qualifying candidate.

    Ties break by fewer states, then fewer edges, then lexicographic candidate id -- deterministic, and
    deliberately independent of any controller result.
    """
    qualifying = [report for report in reports if report["qualifies"]]
    if not qualifying:
        raise SizingInfeasible(
            "no candidate satisfied the window and non-degeneracy conditions; per the frozen policy the "
            "exact panel is reported INFEASIBLE (or split into two slices) rather than approximated"
        )
    return min(
        qualifying,
        key=lambda report: (
            report["statistics"]["n_states"],
            report["statistics"]["n_edges"],
            report["candidate_id"],
        ),
    )


def default_candidate_ladder() -> tuple[Candidate, ...]:
    """The preregistered shrink order: heavy-atom cap, then horizon, then vocabulary, then source.

    Ordered smallest-first so the report reads as a monotone ladder and the selection is transparent.
    """
    ladder: list[Candidate] = []
    for slots in (4, 5, 6, 7):
        ladder.append(
            Candidate(
                candidate_id=f"carbon_{slots}_slots",
                seed_smiles="C" * slots,
                elements=("C",),
                max_hydrogens=3,
            )
        )
    ladder.append(
        Candidate(
            candidate_id="carbon_6_slots_ring_seed",
            seed_smiles="C1CCCCC1",
            elements=("C",),
            max_hydrogens=3,
        )
    )
    ladder.append(
        Candidate(
            candidate_id="carbon_nitrogen_5_slots",
            seed_smiles="CCCCC",
            elements=("C", "N"),
            max_hydrogens=3,
        )
    )
    return tuple(ladder)
