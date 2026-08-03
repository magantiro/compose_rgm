"""Gates of the Process-V2 deletion fiber that no other test can fail on.

Mutation testing of the round-one suite found three production mutations that
every focused test survived: bypassing the executor validity check, bypassing
the declared-support check, and zeroing the legacy dense delete mask. The first
two survived because no fixture reached those rejection codes; on real drug-like
chemistry, deleting an atom preserves every surviving neighbour's class valence.
That makes them defence in depth, but a gate no test can fail on is
indistinguishable from a gate that is not there. Each keeps a reachable witness
below.

The third survived for a structural reason: the round-one comparisons built both
sides from ``_graph_application_masks``, so any change to the legacy branch moved
the expectation with the observation. Both frozen fixtures here are absolute slot
literals, independent of that function, and the Process-V2 fixture is the one
that fails if the corrected uniform gating is weakened back towards a union with
the legacy rule.
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
    process_v2_atom_delete_mask,
    resolve_process_v2_atom_delete,
)
from compose_v4.rewrite.trace_shard_v3 import MAX_ACTIVE_ATOMS


def _state(smiles: str, n_slots: int):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), n_slots)


def test_executor_rejection_is_the_only_gate_stopping_a_hypervalent_ring_delete() -> None:
    """`C1CC[SH4]CC1` passes every other condition and only the executor stops it.

    The successor is connected, charge-preserving, canonicalizable and within
    declared support, so if the executor check were removed the deletion would be
    admitted and the model would learn a rate for a transition the production
    executor refuses.
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
        resolution = resolve_process_v2_atom_delete(state, action)
        assert not resolution.admitted
        assert resolution.rejection_code is ProcessV2AtomDeleteRejectionCode.EXECUTOR_REJECTED

    assert not process_v2_atom_delete_mask(state)[[2, 4]].any()


def test_declared_support_gate_rejects_a_successor_above_the_atom_bound() -> None:
    """A 42-membered carbocycle deletes to 41 active atoms, above the bound.

    The executor accepts the deletion, the successor is connected and the charge
    policy is preserved, so the declared-support condition is the only thing that
    rejects it.
    """

    oversized = _state("C1" + "C" * 41 + "1", MAX_ACTIVE_ATOMS + 4)
    assert int(is_element(oversized.atom_types).sum()) == MAX_ACTIVE_ATOMS + 2
    action = AtomDelete(3)
    assert is_valid_atom_delete(oversized, action)
    assert is_connected_or_null(apply_atom_delete(oversized, action))

    resolution = resolve_process_v2_atom_delete(oversized, action)
    assert not resolution.admitted
    assert (
        resolution.rejection_code
        is ProcessV2AtomDeleteRejectionCode.SUCCESSOR_OUTSIDE_SUPPORT
    )
    assert not process_v2_atom_delete_mask(oversized).any()

    # One atom smaller deletes to exactly the bound and is admitted, so the test
    # pins the boundary rather than merely "large rings are rejected".
    at_bound = _state("C1" + "C" * 40 + "1", MAX_ACTIVE_ATOMS + 4)
    assert int(is_element(at_bound.atom_types).sum()) == MAX_ACTIVE_ATOMS + 1
    assert resolve_process_v2_atom_delete(at_bound, AtomDelete(3)).admitted


# Frozen expectations recorded as literal slot indices, deliberately NOT derived
# from `_graph_application_masks` or from the resolver: a comparison against the
# function under test cannot fail when that function changes.
#
# Column 2 is the legacy dense mask, which this round must leave byte-identical.
# Column 3 is the corrected effective Process-V2 mask.  The two charged rows are
# the round-one defect: the legacy rule admits five slots of the zwitterion and
# two of the carboxylate, and Process V2 must now admit one and one.
_FROZEN_DELETE_SLOTS: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = (
    ("C1CCCCC1", (), (0, 1, 2, 3, 4, 5)),
    ("c1ccccc1", (), ()),
    ("CC1CCCCC1", (0,), (0, 2, 3, 4, 5, 6)),
    ("Cc1ccccc1", (0,), (0,)),
    ("CCO", (0, 2), (0, 2)),
    ("C", (0,), (0,)),
    ("CC", (0, 1), (0, 1)),
    ("O=C1NC(O)C2CCCCC12", (0, 4), (0, 2, 4, 5, 6, 7, 8, 9, 10)),
    ("C[N+](C)(C)CC(=O)[O-]", (0, 2, 3, 6, 7), (6,)),
    ("[O-]C(=O)C1CCCCC1", (0, 2), (2, 4, 5, 6, 7, 8)),
    ("[NH4+]", (0,), ()),
    ("OC1CCC(N)CC1", (0, 5), (0, 2, 3, 5, 6, 7)),
    ("N#CC1CCCCC1", (0,), (0, 3, 4, 5, 6, 7)),
    ("ClCC1CCCCC1", (0,), (0, 3, 4, 5, 6, 7)),
)


def _legacy_batch(smiles: tuple[str, ...]):
    states = tuple(_state(entry, 40) for entry in smiles)
    count = len(states)
    return prepare_factorized_mark_batch(
        states,
        tuple(0.5 for _ in range(count)),
        tuple(None for _ in range(count)),
        tuple(None for _ in range(count)),
        tuple(0.0 for _ in range(count)),
    )


def test_legacy_delete_mask_matches_a_frozen_literal_expectation() -> None:
    """The legacy dense mask must equal a hand-recorded fixture, not a recomputation.

    This is the absolute form of the handoff's "unchanged on a frozen fixture"
    requirement, and it is the only thing standing between this round and a
    silent change to legacy behaviour.
    """

    smiles = tuple(entry[0] for entry in _FROZEN_DELETE_SLOTS)
    batch = _legacy_batch(smiles)
    assert batch.atom_delete_admission_mask is None
    for index, (source, expected, _) in enumerate(_FROZEN_DELETE_SLOTS):
        observed = tuple(int(v) for v in np.flatnonzero(batch.atom_delete_mask[index].numpy()))
        assert observed == expected, source
        # The legacy rule admits only real-atom degree at most one, which is why
        # it never reached a ring atom and why Process V2 is an expansion there.
        for slot in observed:
            degree = int((batch.bonds[index][slot].numpy() != 0).sum())
            assert degree <= 1, (source, slot, degree)


def test_process_v2_mask_matches_a_frozen_literal_expectation() -> None:
    """The corrected effective mask, as absolute literals.

    Rows 9 to 11 are the corrected charged behaviour and are what fails if the
    uniform gating is weakened back to a union with the legacy rule: the union
    would readmit slots 0, 2, 3 and 7 of the zwitterion, slot 0 of the
    carboxylate and slot 0 of the ammonium cation.
    """

    lost = 0
    gained = 0
    for source, legacy, expected in _FROZEN_DELETE_SLOTS:
        state = _state(source, 40)
        observed = tuple(int(v) for v in np.flatnonzero(process_v2_atom_delete_mask(state)))
        assert observed == expected, source
        lost += len(set(legacy) - set(expected))
        gained += len(set(expected) - set(legacy))
    # Process V2 is neither a superset nor a subset of the legacy rule.
    assert lost == 6, lost
    assert gained == 37, gained


def test_a_malformed_action_payload_is_refused_rather_than_coerced() -> None:
    """The authority must decide the action it was given, not a coerced one.

    ``int(action.v)`` used to run before the type check, so ``AtomDelete(1.9)``
    and ``AtomDelete(True)`` were truncated to slot 1 and ADMITTED. A caller
    that then executed their own object would hit the executor with a payload
    the authority never actually approved.

    ``np.int64`` must keep working: slots arrive that way from
    ``np.flatnonzero`` throughout this codebase, so rejecting every non-``int``
    would be a different defect.
    """

    state = _state("C1CCCCC1", 12)

    assert resolve_process_v2_atom_delete(state, AtomDelete(1)).admitted
    assert resolve_process_v2_atom_delete(state, AtomDelete(np.int64(1))).admitted

    class _AtomDeleteSubclass(AtomDelete):
        pass

    for action in (
        AtomDelete(1.9),
        AtomDelete(1.0),
        AtomDelete(True),
        _AtomDeleteSubclass(1),
    ):
        resolution = resolve_process_v2_atom_delete(state, action)
        assert not resolution.admitted, action
        assert resolution.rejection_code is ProcessV2AtomDeleteRejectionCode.INVALID_SLOT
        # -1 records that no slot was ever resolved, so the report cannot be
        # mistaken for a decision about slot 1.
        assert resolution.slot == -1, action
        assert resolution.successor is None


def test_the_retracted_round_one_mode_string_does_not_survive() -> None:
    """The old mode name falsely implies inherited candidates are unfiltered.

    It must not survive in either module this worker owns, including in a
    comment, a docstring or an alias.
    """

    from pathlib import Path

    from compose_v4.model.factorized_tracelet_rate_model import (
        PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    )

    assert (
        PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
        == "process_v2_uniform_gated_atom_delete_v2"
    )
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "src/compose_v4/rewrite/process_v2_atom_delete.py",
        "src/compose_v4/model/factorized_tracelet_rate_model.py",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "process_v2_connected_nonleaf_atom_delete_v1" not in text, relative
        assert "process_v2_connected_nonleaf_atom_delete_mask" not in text, relative
        assert "resolve_process_v2_connected_nonleaf_atom_delete" not in text, relative
        assert "enumerate_process_v2_connected_nonleaf_atom_deletes" not in text, relative
