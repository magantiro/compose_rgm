"""Cyclic graft-for-editing: relocate a pendant tree fragment across a single-bond bridge.

B's dense graft mask is gated to ``all_single_tree`` (de-novo carbon skeletons); on a real cyclic
molecule it produced no graft at all. The editing model turns on ``compute_cyclic_graft`` to enumerate
the well-defined subset -- move a PENDANT tree fragment across a single-bond bridge while the ring core
stays fixed -- reusing B's existing ``bond_reroute`` head (architecture-preserving; no new parameters).

These tests pin the invariants: the flag is off by default (B behavior preserved), pendant trees
relocate, ring-bearing fragments do not, every mask graft executes through the real ``BondReroute``
executor, and the successor quotient partitions grafts exactly by resulting molecule.
"""

from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.model.factorized_tracelet_rate_model import prepare_factorized_mark_batch
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import BondReroute

_SLOTS = 40


def _batch(smiles, *, compute_cyclic_graft):
    states = tuple(pad_molecular_graph(smiles_to_molecular_graph(s), _SLOTS) for s in smiles)
    n = len(states)
    batch = prepare_factorized_mark_batch(
        states,
        (0.5,) * n,
        (None,) * n,
        (None,) * n,
        (0.0,) * n,
        compute_cyclic_graft=compute_cyclic_graft,
    )
    return states, batch


def test_cyclic_graft_off_by_default_and_tree_branch_unaffected() -> None:
    # A cyclic molecule with a pendant chain gets NO graft when the flag is off (B behavior); a pure
    # tree (all single bonds) is handled by the untouched tree branch, identical with the flag on/off.
    _, off = _batch(["c1ccccc1CCC", "CCCC"], compute_cyclic_graft=False)
    _, on = _batch(["c1ccccc1CCC", "CCCC"], compute_cyclic_graft=True)
    assert int(off.graft_mask[0].sum()) == 0  # cyclic, flag off -> no graft
    assert int(on.graft_mask[0].sum()) > 0  # cyclic, flag on -> pendant grafts
    assert int(off.graft_mask[1].sum()) == int(on.graft_mask[1].sum()) > 0  # tree branch unaffected


def test_cyclic_graft_relocates_pendant_and_executes() -> None:
    # Every graft the cyclic branch emits must be a legal BondReroute the real executor accepts, with the
    # (moved, target, removed_neighbor) representation the sampler/teacher reconstruct.
    states, on = _batch(["c1ccccc1CCC"], compute_cyclic_graft=True)
    system = de_novo_rewrite_system()
    state = states[0]
    pairs = np.argwhere(on.graft_mask[0].numpy())
    assert len(pairs) > 0
    for moved, target in pairs:
        removed_neighbor = int(on.graft_remove_neighbors[0, int(moved), int(target)])
        action = BondReroute(a=int(moved), b=removed_neighbor, u=int(moved), v=int(target))
        assert system.apply(state, "bond_reroute", action) is not None


def test_cyclic_graft_excludes_ring_bearing_fragments() -> None:
    # Every bridge here separates two rings (benzene | pyridine), so no movable *tree* fragment exists;
    # relocating a ring system is the deliberately deferred case, so the mask stays empty.
    _, on = _batch(["O=C(Cc1ccccc1)Nc1ccncc1"], compute_cyclic_graft=True)
    assert int(on.graft_mask[0].sum()) == 0


def test_cyclic_graft_quotient_partitions_by_successor() -> None:
    # Generator Matching supervises the molecular successor, so grafts sharing a successor must share a
    # group id and grafts with distinct successors must not -- an exact partition by canonical molecule.
    states, on = _batch(["c1ccccc1CCC"], compute_cyclic_graft=True)
    system = de_novo_rewrite_system()
    state = states[0]
    groups = on.graft_successor_groups[0].numpy()
    key_by_pair: dict[tuple[int, int], object] = {}
    for moved, target in np.argwhere(on.graft_mask[0].numpy()):
        removed_neighbor = int(on.graft_remove_neighbors[0, int(moved), int(target)])
        action = BondReroute(a=int(moved), b=removed_neighbor, u=int(moved), v=int(target))
        key_by_pair[(int(moved), int(target))] = canonical_state_key(
            system.apply(state, "bond_reroute", action)
        )
    for (m1, t1), k1 in key_by_pair.items():
        for (m2, t2), k2 in key_by_pair.items():
            same_group = int(groups[m1, t1]) == int(groups[m2, t2])
            assert same_group == (k1 == k2)
