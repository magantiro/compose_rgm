"""Topology is shared by reports and controllers, not separately reimplemented."""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.chem.topology import cycle_rank
from compose_v4.control import region_replacement
from compose_v4.experiments import enumerable_ringcore


def test_legacy_import_paths_are_the_same_public_function():
    assert region_replacement.cycle_rank is cycle_rank
    assert enumerable_ringcore.cycle_rank is cycle_rank


@pytest.mark.parametrize(
    "smiles,expected",
    [("C", 0), ("CCO", 0), ("C1CCCCC1", 1), ("C1CC2CCC1CC2", 2)],
)
def test_shared_descriptor_preserves_connected_controller_cycle_bound(smiles, expected):
    graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)
    slots = np.flatnonzero(is_element(graph.atom_types))
    # Independent graph-theory oracle on the supplied connected graph.
    edges = np.count_nonzero(graph.bonds[np.ix_(slots, slots)]) // 2
    assert cycle_rank(graph) == int(edges - len(slots) + 1) == expected


def test_null_and_explicit_key_keep_the_existing_report_interface():
    assert cycle_rank(None, "<NULL>") == 0
    graph = smiles_to_molecular_graph("C1CCCCC1")
    assert cycle_rank(graph, "C1CCCCC1") == cycle_rank(graph) == 1
