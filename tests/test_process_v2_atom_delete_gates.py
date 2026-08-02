"""Gates of the Process-V2 deletion fiber that no other test can fail on.

Mutation testing of the committed suite found three production mutations that
every focused test survived:

* bypassing the executor validity check,
* bypassing the declared-support check, and
* zeroing the V1 dense delete mask.

The first two survived because no fixture reaches those rejection codes: on real
drug-like chemistry, deleting an atom preserves every surviving neighbour's
class valence, so the executor and support gates never bind. That makes them
defence in depth, but a gate no test can fail on is indistinguishable from a
gate that is not there. Each is given a reachable witness below.

The third survived for a structural reason: the committed comparisons build both
sides from `_graph_application_masks`, so any change to the V1 branch moves the
expectation with the observation. The absolute frozen fixture here is
independent of that function.
"""

from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.model.factorized_tracelet_rate_model import prepare_factorized_mark_batch
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import AtomDelete, apply_atom_delete, is_valid_atom_delete
from compose_v4.rewrite.process_v2_atom_delete import (
    ProcessV2AtomDeleteRejectionCode,
    process_v2_connected_nonleaf_atom_delete_mask,
    resolve_process_v2_connected_nonleaf_atom_delete,
)
from compose_v4.rewrite.trace_shard_v3 import MAX_ACTIVE_ATOMS


def _state(smiles: str, n_slots: int):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), n_slots)


def test_executor_rejection_is_the_only_gate_stopping_a_hypervalent_ring_delete() -> None:
    """`C1CC[SH4]CC1` passes every other condition and only the executor stops it.

    The successor is connected, canonicalizable, and within declared support, so
    if the executor check were removed the deletion would be admitted and the
    model would learn a rate for a transition the production executor refuses.
    """

    state = _state("C1CC[SH4]CC1", 12)
    assert is_valid_state(state)

    for slot in (2, 4):
        action = AtomDelete(slot)
        successor = apply_atom_delete(state, action)
        # Every non-executor condition passes, which is what makes this a witness.
        assert is_connected_or_null(successor)
        assert canonical_state_key(successor) == "CCCC[SH5]"
        # The executor refuses it, and so does the resolver, for that reason.
        assert not is_valid_atom_delete(state, action)
        resolution = resolve_process_v2_connected_nonleaf_atom_delete(state, action)
        assert not resolution.admitted
        assert resolution.rejection_code is ProcessV2AtomDeleteRejectionCode.EXECUTOR_REJECTED

    assert not process_v2_connected_nonleaf_atom_delete_mask(state)[[2, 4]].any()


def test_declared_support_gate_rejects_a_successor_above_the_atom_bound() -> None:
    """A 42-membered carbocycle deletes to 41 active atoms, above the bound.

    The executor accepts the deletion and the successor is connected, so the
    declared-support condition is the only thing that rejects it.
    """

    oversized = _state("C1" + "C" * 41 + "1", MAX_ACTIVE_ATOMS + 4)
    assert int(is_element(oversized.atom_types).sum()) == MAX_ACTIVE_ATOMS + 2
    action = AtomDelete(3)
    assert is_valid_atom_delete(oversized, action)
    assert is_connected_or_null(apply_atom_delete(oversized, action))

    resolution = resolve_process_v2_connected_nonleaf_atom_delete(oversized, action)
    assert not resolution.admitted
    assert (
        resolution.rejection_code
        is ProcessV2AtomDeleteRejectionCode.SUCCESSOR_OUTSIDE_SUPPORT
    )
    assert not process_v2_connected_nonleaf_atom_delete_mask(oversized).any()

    # One atom smaller deletes to exactly the bound and is admitted, so the test
    # pins the boundary rather than merely "large rings are rejected".
    at_bound = _state("C1" + "C" * 40 + "1", MAX_ACTIVE_ATOMS + 4)
    assert int(is_element(at_bound.atom_types).sum()) == MAX_ACTIVE_ATOMS + 1
    assert resolve_process_v2_connected_nonleaf_atom_delete(at_bound, AtomDelete(3)).admitted


# Frozen expectation for the legacy dense atom-delete mask, recorded as literal
# slot indices. Deliberately NOT derived from `_graph_application_masks`: a
# comparison against that function cannot fail when the function itself changes.
_FROZEN_LEGACY_DELETE_SLOTS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("C1CCCCC1", ()),
    ("c1ccccc1", ()),
    ("CC1CCCCC1", (0,)),
    ("Cc1ccccc1", (0,)),
    ("CCO", (0, 2)),
    ("C", (0,)),
    ("CC", (0, 1)),
    ("O=C1NC(O)C2CCCCC12", (0, 4)),
    ("C[N+](C)(C)CC(=O)[O-]", (0, 2, 3, 6, 7)),
    ("OC1CCC(N)CC1", (0, 5)),
    ("N#CC1CCCCC1", (0,)),
    ("ClCC1CCCCC1", (0,)),
)


def test_legacy_delete_mask_matches_a_frozen_literal_expectation() -> None:
    """The V1 dense mask must equal a hand-recorded fixture, not a recomputation.

    This is the absolute form of the handoff's "unchanged on a frozen fixture"
    requirement. It also re-states the structural fact the whole disjoint-union
    design rests on: every legacy-admitted slot has real-atom degree at most one.
    """

    smiles = tuple(entry[0] for entry in _FROZEN_LEGACY_DELETE_SLOTS)
    states = tuple(_state(entry, 40) for entry in smiles)
    count = len(states)
    batch = prepare_factorized_mark_batch(
        states,
        tuple(0.5 for _ in range(count)),
        tuple(None for _ in range(count)),
        tuple(None for _ in range(count)),
        tuple(0.0 for _ in range(count)),
    )
    assert batch.atom_delete_admission_mask is None
    for index, (source, expected) in enumerate(_FROZEN_LEGACY_DELETE_SLOTS):
        observed = tuple(int(v) for v in np.flatnonzero(batch.atom_delete_mask[index].numpy()))
        assert observed == expected, source
        for slot in observed:
            degree = int((batch.bonds[index][slot].numpy() != 0).sum())
            assert degree <= 1, (source, slot, degree)
