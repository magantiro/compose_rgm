from __future__ import annotations

import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_SINGLE,
    smiles_to_molecular_graph,
)


def test_resonance_invariant_view_maps_benzene_to_class_four() -> None:
    state = smiles_to_molecular_graph("c1ccccc1")
    perceived = resonance_invariant_bond_classes(state)

    assert int(np.triu(perceived == BOND_AROMATIC, k=1).sum()) == 6
    assert not np.array_equal(perceived, state.bonds)
    assert set(state.bonds[state.bonds != 0]) == {1, 2}


def test_resonance_invariant_view_preserves_saturated_ring_bonds() -> None:
    state = smiles_to_molecular_graph("C1CCCCC1")
    perceived = resonance_invariant_bond_classes(state)

    assert np.array_equal(perceived, state.bonds)
    assert set(perceived[perceived != 0]) == {BOND_SINGLE}
