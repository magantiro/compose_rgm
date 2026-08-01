"""Non-authorizing semantic cycle-close prototype tests."""

from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.aromatic_cycle_close_semantics import (
    PROTOTYPE_STATUS,
    AromaticCycleCloseRejectionCode,
    resolve_component_factored_cycle_close,
)
from compose_v4.experiments.cycle_open_kekule_invariance import (
    build_alternate_kekule_pair,
)
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.operators import (
    BondInsert,
    CycleOpenEdge,
    is_valid_bond_insert,
)


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)


def _raw_cycle_close_candidates(state) -> tuple[BondInsert, ...]:
    real = tuple(int(index) for index in np.flatnonzero(is_element(state.atom_types)))
    return tuple(
        action
        for offset, left in enumerate(real)
        for right in real[offset + 1 :]
        for order in (1, 2, 3)
        if is_valid_bond_insert(
            state,
            action := BondInsert(left, right, order),
        )
    )


def test_prototype_identifies_the_observed_ambiguous_cycle_close() -> None:
    source = _state("Cc1cccc(Cl)c1")
    resolution = resolve_component_factored_cycle_close(
        source,
        BondInsert(0, 4, 1),
    )
    assert resolution.prototype_status == PROTOTYPE_STATUS
    assert resolution.admitted is False
    assert (
        resolution.rejection_code
        == AromaticCycleCloseRejectionCode.AMBIGUOUS_CANONICAL_PRODUCT
    )
    assert resolution.enumerated_affected_assignment_count == 2
    assert len(resolution.canonical_product_keys) == 2


def test_unambiguous_aromatic_cycle_close_is_retained() -> None:
    source = _state("Cc1cccc(Cl)c1")
    resolution = resolve_component_factored_cycle_close(
        source,
        BondInsert(0, 3, 1),
    )
    assert resolution.admitted is True
    assert resolution.rejection_code is None
    assert resolution.successor is not None
    assert resolution.executed_product_count == 2
    assert len(resolution.canonical_product_keys) == 1


def test_nonaromatic_cycle_close_matches_raw_execution() -> None:
    source = _state("CCCCCC")
    action = BondInsert(0, 5, 1)
    resolution = resolve_component_factored_cycle_close(source, action)
    assert resolution.admitted is True
    assert resolution.successor is not None
    expected = editing_v2_rewrite_system().apply(source, "bond_insert", action)
    assert canonical_state_key(resolution.successor) == canonical_state_key(expected)
    assert resolution.aromatic_component_count == 0
    assert resolution.enumerated_affected_assignment_count == 1


def test_alternate_kekule_sources_have_identical_semantic_close_fiber() -> None:
    pair = build_alternate_kekule_pair(_state("Cc1cccc(Cl)c1"))
    candidates = tuple(
        sorted(
            set(_raw_cycle_close_candidates(pair.original))
            | set(_raw_cycle_close_candidates(pair.alternate)),
            key=lambda action: (action.a, action.b, action.order),
        )
    )

    def resolved(source) -> dict[BondInsert, str | None]:
        result: dict[BondInsert, str | None] = {}
        for action in candidates:
            resolution = resolve_component_factored_cycle_close(source, action)
            result[action] = (
                resolution.canonical_product_keys[0] if resolution.admitted else None
            )
        return result

    original = resolved(pair.original)
    alternate = resolved(pair.alternate)
    assert original == alternate
    assert original[BondInsert(0, 4, 1)] is None
    assert any(value is not None for value in original.values())


def test_admitted_close_has_semantic_open_inverse_at_molecular_level() -> None:
    source = _state("Cc1cccc(Cl)c1")
    action = BondInsert(0, 3, 1)
    resolution = resolve_component_factored_cycle_close(source, action)
    assert resolution.successor is not None
    restored = editing_v2_rewrite_system().apply(
        resolution.successor,
        "cycle_open",
        CycleOpenEdge(action.a, action.b),
    )
    assert canonical_state_key(restored) == canonical_state_key(source)
