from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.rewrite.commuting_schedule import schedule_priority_events_earliest
from compose_v4.rewrite.trace import execute_trace
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target


@pytest.mark.parametrize(
    "smiles",
    (
        "c1ccncc1CCO",
        "C1CCCCC1NCCO",
        "C1CCC2CCCCC2C1O",
        "C1CC2CCC1C2CCN",
        "C1CCC2(CC1)CCCC2CO",
    ),
)
def test_exact_scheduler_moves_ring_transactions_earlier_without_changing_target(
    smiles: str,
) -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 20)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(812),
        n_slots=target.n_atoms,
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    scheduled, report = schedule_priority_events_earliest(trace)
    endpoint, states = execute_trace(
        scheduled.source,
        scheduled.steps,
        return_states=True,
    )

    assert np.array_equal(endpoint.atom_types, target.atom_types)
    assert np.array_equal(endpoint.formal_charges, target.formal_charges)
    assert np.array_equal(endpoint.implicit_h_counts, target.implicit_h_counts)
    assert np.array_equal(endpoint.bonds, target.bonds)
    assert all(is_valid_state(state) for state in states)
    assert all(is_connected_or_null(state) for state in states)
    assert len(report.priority_positions_after) == len(
        report.priority_positions_before
    )
    assert all(
        after <= before
        for before, after in zip(
            report.priority_positions_before,
            report.priority_positions_after,
        )
    )
    assert scheduled.metadata["event_schedule"] == "exact_earliest_priority_v1"


def test_resize_events_remain_before_early_ring_schedule() -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("c1ccncc1CCO"), 20)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms - 3,)).sample(
        np.random.default_rng(997),
        n_slots=target.n_atoms,
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        flexible_size=True,
    )
    scheduled, _ = schedule_priority_events_earliest(trace)
    resize_positions = [
        index
        for index, step in enumerate(scheduled.steps)
        if step.rule_name in {"atom_insert", "atom_delete"}
    ]
    ring_positions = [
        index
        for index, step in enumerate(scheduled.steps)
        if step.rule_name == "ring_system_grow"
    ]

    assert resize_positions
    assert ring_positions
    assert max(resize_positions) < min(ring_positions)
