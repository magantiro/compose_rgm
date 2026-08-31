"""Locality is graph geometry over the executed trajectory, not fingerprints."""

import numpy as np
import pytest

from compose_v4.control.graph_geometry import (
    atom_environment, kinetic_difficulty, structural_displacement, topology,
    trajectory_excursion,
)
from compose_v4.control.region_rewrite import Lineage


class _S:
    def __init__(self, types, bonds, h=None):
        n = len(types)
        self.atom_types = np.array(types)
        self.formal_charges = np.zeros(n, dtype=int)
        self.implicit_h_counts = np.array(h if h else [0] * n)
        self.bonds = np.array(bonds)


def _chain(n, pad=0):
    t = [2] * n + [0] * pad
    b = np.zeros((n + pad, n + pad), dtype=int)
    for i in range(n - 1):
        b[i][i + 1] = b[i + 1][i] = 1
    return _S(t, b)


def _ring(n, pad=0):
    s = _chain(n, pad)
    s.bonds[0][n - 1] = s.bonds[n - 1][0] = 1
    return s


def test_topology_counts_rings_from_the_adjacency():
    assert topology(_chain(6))["cycle_rank"] == 0
    r = topology(_ring(6))
    assert r["cycle_rank"] == 1 and r["n_ring_systems"] == 1 and r["n_ring_atoms"] == 6


def test_environment_sees_neighbours_not_just_the_atom():
    s = _chain(3)
    assert atom_environment(s, 0, radius=0) == atom_environment(s, 2, radius=0)
    # terminal atoms match; the middle atom has two neighbours
    assert atom_environment(s, 1) != atom_environment(s, 0)


def test_identity_trajectory_has_zero_displacement():
    s = _chain(5)
    lin = Lineage.initial(range(5))
    d = structural_displacement(s, s, lin, lin)
    assert d["changed_fraction"] == 0.0
    assert d["n_changed_regions"] == 0
    assert not d["topology_changed"]


def test_ring_closure_registers_as_topology_change():
    a, b = _chain(6), _ring(6)
    lin = Lineage.initial(range(6))
    d = structural_displacement(a, b, lin, lin)
    assert d["topology_changed"]
    assert d["d_cycle_rank"] == 1
    assert d["changed_fraction"] > 0


def test_coherent_versus_scattered_changes_are_distinguished():
    """Same number of changed atoms, different move: this is the whole point."""
    base = _chain(10)
    lin = Lineage.initial(range(10))

    coherent = _chain(10)
    for i in (4, 5, 6):                       # one contiguous block
        coherent.atom_types[i] = 3
    scattered = _chain(10)
    for i in (1, 5, 9):                       # three unrelated sites
        scattered.atom_types[i] = 3

    dc = structural_displacement(base, coherent, lin, lin)
    ds = structural_displacement(base, scattered, lin, lin)
    assert dc["n_changed_regions"] < ds["n_changed_regions"]
    assert dc["largest_changed_region"] > ds["largest_changed_region"]
    assert dc["coherence"] > ds["coherence"]


def test_excursion_sees_a_detour_that_the_endpoint_hides():
    """Restructure heavily, then return: endpoint-only analysis cannot see it."""
    s0 = _chain(6)
    mid = _chain(6)
    for i in range(6):
        mid.atom_types[i] = 3                 # everything changed at t=1
    lin = Lineage.initial(range(6))
    exc = trajectory_excursion([s0, mid, s0], [lin, lin, lin])
    assert exc["endpoint_changed_fraction"] == 0.0
    assert exc["excursion_changed_fraction"] == 1.0


def test_insertions_and_deletions_are_counted_separately():
    s0 = _chain(4, pad=1)
    lin0 = Lineage.initial(range(4))
    s1 = _chain(5)
    lin1 = lin0.observe("atom_insert", type("A", (), {"slot": 4, "atom_type": 2,
                                                      "neighbors": ((3, 1),)})())
    d = structural_displacement(s0, s1, lin0, lin1)
    assert d["n_inserted"] == 1 and d["n_deleted"] == 0
    assert d["d_heavy"] == 1


def test_kinetic_difficulty_is_negative_log_path_mass():
    assert kinetic_difficulty([np.log(0.1), np.log(0.01)]) == pytest.approx(
        -(np.log(0.1) + np.log(0.01)))
