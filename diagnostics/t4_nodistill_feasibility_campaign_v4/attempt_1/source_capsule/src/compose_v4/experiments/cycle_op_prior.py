"""Compositional ring-op (cycle_close / cycle_open) supervision records (native Generator Matching).

P3 of the ring-support redesign. Each real broad-organic molecule is decomposed by its ring (non-bridge)
bonds: removing a ring bond (bond_delete = cycle_open) yields a valid connected precursor, and adding it back
(bond_insert = cycle_close) reconstructs the molecule exactly. Both directions are emitted per ring bond, so
cycle_open/cycle_close receive massive, real, round-trip-verified supervision -- unlike the whole-ring grow
macro, whose inverse was mask-illegal. Topology-independent: single rings, fused, spiro, bridged, macrocycles,
hetero/S/P rings, charged contexts all decompose the same way.

Layering mirrors corrupted_source_prior: this module wraps executor-verified traces into the ``PathRecord``s
the GM trainer consumes. Eligibility is graph bridge/non-bridge status (§6), never one SSSR basis.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import (
    MolecularGraphError,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import BondDelete, BondInsert
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace


def _net_charge(state) -> int:
    return int(state.formal_charges[is_element(state.atom_types)].sum())


def _ring_bonds(state) -> list[tuple[int, int, int]]:
    """Non-bridge (cycle) bonds (a<b, order): removing one preserves connectedness and reduces cycle rank."""
    real = [i for i in range(len(state.atom_types)) if is_element(state.atom_types)[i]]
    graph = nx.Graph()
    graph.add_nodes_from(real)
    for a in real:
        for b in real:
            if a < b and int(state.bonds[a, b]) != 0:
                graph.add_edge(a, b)
    bridges = {frozenset(edge) for edge in nx.bridges(graph)}
    return [
        (a, b, int(state.bonds[a, b]))
        for a, b in graph.edges()
        if frozenset((a, b)) not in bridges and a < b
    ]


def build_cycle_op_records(
    smiles,
    *,
    n_slots: int,
    system=None,
    seed: int = 0,
    max_bonds_per_molecule: int = 4,
    checkpoint_interval: int | None = None,
    both_directions: bool = True,
):
    """Return ``(records, n_attempted)``. Per molecule, up to ``max_bonds_per_molecule`` ring bonds (sampled
    without replacement so multi-ring molecules do not dominate) each yield a cycle_open trace (real ->
    precursor) and a cycle_close trace (precursor -> real), both executor-verified: precursor valid +
    connected + <= n_slots + charge-preserving, and an exact close-after-open round trip to the original."""
    system = system or de_novo_rewrite_system()
    rng = np.random.default_rng(seed)
    records: list[PathRecord] = []
    attempted = 0
    for text in smiles:
        try:
            graph = smiles_to_molecular_graph(text)
        except MolecularGraphError:
            continue
        if graph is None:
            continue
        try:
            target = pad_molecular_graph(graph, n_slots)
        except ValueError:
            continue
        bonds = _ring_bonds(target)
        if not bonds:
            continue
        if len(bonds) > max_bonds_per_molecule:
            idx = rng.choice(len(bonds), size=max_bonds_per_molecule, replace=False)
            bonds = [bonds[i] for i in idx]
        q0 = _net_charge(target)
        target_key = canonical_state_key(target)
        for a, b, order in bonds:
            attempted += 1
            try:
                precursor = system.apply(target, "bond_delete", BondDelete(a, b))
            except Exception:  # noqa: BLE001 -- executor rejects (valence etc.); skip this bond
                continue
            if not (is_valid_state(precursor) and is_connected_or_null(precursor)):
                continue
            if int(is_element(precursor.atom_types).sum()) > n_slots:
                continue
            if _net_charge(precursor) != q0:  # charge-preserving
                continue
            try:  # exact close-after-open round trip
                recovered = system.apply(precursor, "bond_insert", BondInsert(a, b, order))
            except Exception:  # noqa: BLE001
                continue
            if canonical_state_key(recovered) != target_key:
                continue
            cycle_open = RewriteTrace(
                source=target,
                target=precursor,
                steps=(RewriteStep("bond_delete", BondDelete(a, b)),),
                metadata={"prior": "cycle_open", "n_steps": 1},
            )
            cycle_close = RewriteTrace(
                source=precursor,
                target=target,
                steps=(RewriteStep("bond_insert", BondInsert(a, b, order)),),
                metadata={"prior": "cycle_close", "n_steps": 1},
            )
            traces = (cycle_open, cycle_close) if both_directions else (cycle_close,)
            for trace in traces:
                records.append(
                    PathRecord(
                        canonical_state_key(trace.target),
                        TraceProgressCTMC(trace, checkpoint_interval=checkpoint_interval),
                    )
                )
    return tuple(records), attempted
