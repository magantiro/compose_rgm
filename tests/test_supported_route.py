import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.supported_route import (
    BridgeConfig,
    exact_key,
    lower_supported,
    transported_target,
)
from compose_v4.control.supported_route import (
    permute_persistent_slots as production_permute,
)
from compose_v4.experiments.quotient_invariance import permute_persistent_slots
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import AtomInsert, BondReorder, CycleCloseEdge, CycleOpenEdge


def test_open_edit_close_uses_only_supplied_positive_marks():
    system = editing_v2_semantic_rewrite_system()
    source = pad_molecular_graph(smiles_to_molecular_graph("C1CCCCC1"), 48)
    target = system.apply(source, "bond_reorder", BondReorder(1, 2, 2))
    marks = [
        ("cycle_open", CycleOpenEdge(0, 5)),
        ("bond_reorder", BondReorder(1, 2, 2)),
        ("cycle_close", CycleCloseEdge(0, 5, 1)),
    ]
    states = [source]
    for family, action in marks:
        states.append(system.apply(states[-1], family, action))
    assert exact_key(states[-1]) == exact_key(target)
    offered = {exact_key(s): ([f], [a], [1.0]) for s, (f, a) in zip(states, marks)}
    result = lower_supported(
        source, target, lambda s: offered.get(exact_key(s), ([], [], [])), system
    )
    assert result["status"] == "supported"
    assert len(result["steps"]) == 3
    assert all(step["selected_mark_probability"] == 1 for step in result["steps"])
    absent = lower_supported(source, target, lambda s: ([], [], []), system)
    assert absent["status"] == "bridge_unresolved"
    limited = lower_supported(
        source,
        target,
        lambda s: offered.get(exact_key(s), ([], [], [])),
        system,
        BridgeConfig(1, 8),
    )
    assert limited["status"] == "bridge_unresolved"


def test_zero_probability_does_not_certify_support():
    system = editing_v2_semantic_rewrite_system()
    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)
    action = BondReorder(0, 1, 2)
    target = system.apply(source, "bond_reorder", action)
    result = lower_supported(source, target, lambda s: (["bond_reorder"], [action], [0.0]), system)
    assert result["status"] == "bridge_unresolved"


def test_birth_correspondence_keeps_sparse_exact_state():
    system = editing_v2_semantic_rewrite_system()
    dense = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)
    source = permute_persistent_slots(dense, list(reversed(range(48))))
    assert exact_key(source) == exact_key(production_permute(dense, list(reversed(range(48)))))
    action = AtomInsert(3, int(source.atom_types[46]), 0, 3, ((46, 1),))
    following = system.apply(source, "atom_insert", action)
    target, order = transported_target(source, following, source, list(range(48)))
    assert order[0] == 3 and order[3] == 0
    assert np.array_equal(source.bonds, permute_persistent_slots(source, order).bonds)
    expected = system.apply(
        source, "atom_insert", AtomInsert(0, action.atom_type, 0, 3, ((46, 1),))
    )
    assert exact_key(target) == exact_key(expected)
