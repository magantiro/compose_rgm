"""The composite footprint must restore support without bypassing the context."""

import pytest

from compose_v4.control.region_rewrite import (
    RewriteContext,
    admissible_indices,
    created_slot,
    touched_slots,
)
from compose_v4.rewrite.tracelets import BondOrderChange, RingSystemRestate


def test_nested_endpoints_and_no_birth():
    action = RingSystemRestate((BondOrderChange(4, 8, 2), BondOrderChange(8, 18, 1)))
    assert touched_slots(action) == {4, 8, 18}
    assert created_slot(action) is None


@pytest.mark.parametrize(
    "edges,reason",
    [
        (((3, 4), (4, 5)), None),
        (((1, 3), (3, 4)), None),
        (((0, 3),), "frozen_touched"),
        (((1, 2),), "context_only"),
        (((1, 2), (3, 4)), "context_only"),
        ((), "not_in_locus"),
    ],
)
def test_composite_admission_respects_each_frozen_bond(edges, reason):
    context = RewriteContext(
        frozenset({0, 1, 2}),
        frozenset({3, 4, 5}),
        ((1, 3, 1), (2, 5, 1)),
        "segment",
        1,
    )
    action = RingSystemRestate(tuple(BondOrderChange(a, b, 2) for a, b in edges))
    indices, rejections = admissible_indices(("ring_system_restate",), (action,), context)
    assert indices == ([0] if reason is None else [])
    assert sum(rejections.values()) == (0 if reason is None else 1)
    if reason:
        assert rejections[reason] == 1


def test_restate_and_generic_channels_execute_the_admitted_action():
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.option_continuation import OptionContinuationKernel, OptionState
    from compose_v4.control.region_rewrite import Lineage, context_preserved
    from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system

    graph = pad_molecular_graph(smiles_to_molecular_graph("CC1CCCCC1"), 48)
    action = RingSystemRestate((BondOrderChange(1, 2, 2), BondOrderChange(4, 5, 2)))
    context = RewriteContext(frozenset({0}), frozenset(range(1, 7)), ((0, 1, 1),), "pendant", 1)
    system = editing_v2_semantic_rewrite_system()
    kernel = OptionContinuationKernel(
        lambda _: (("ring_system_restate",), (action,), (1.0,)),
        system,
        max_executor_applications=1,
    )
    expected = canonical_state_key(system.apply(graph, "ring_system_restate", action))
    for option in ("restate", "generic"):
        node = OptionState(
            graph, graph, context, Lineage.initial(range(7)), option, 0, 1, "fixture"
        )
        row = kernel.row(node)
        assert len(row.successors) == 1
        product = row.successors[0].graph
        assert canonical_state_key(product) == expected
        assert context_preserved(graph, product, context.frozen, context.terminal_context_slots)
