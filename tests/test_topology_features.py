from __future__ import annotations

from compose_v4.chem.graph_primitives import compute_topology_features
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph


def _topology(smiles: str, n_slots: int):
    state = pad_molecular_graph(smiles_to_molecular_graph(smiles), n_slots)
    return compute_topology_features(state)


def test_cycle_size_if_bonded_and_ring_system_growth_are_explicit() -> None:
    atom_topology, closure_topology, ring_system_topology = _topology("CCCCCC", 8)
    assert atom_topology[:6].tolist() == [0] * 6
    assert int(closure_topology[0, 5]) == 4  # six-membered ring
    assert int(ring_system_topology[0, 5]) == 1  # first ring in system

    atom_topology, closure_topology, ring_system_topology = _topology(
        "C1CCCCC1CC",
        8,
    )
    assert all(int(value) == 4 for value in atom_topology[:6])
    assert int(closure_topology[0, 7]) > 0
    assert int(ring_system_topology[0, 7]) == 2  # closure grows bicyclic system
