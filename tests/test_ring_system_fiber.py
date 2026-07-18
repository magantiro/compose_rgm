from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.ring_system_fiber import (
    enumerate_executable_ring_grow_candidates,
    enumerate_ring_system_deletes,
    enumerate_ring_system_grows,
    enumerate_ring_system_template_placements,
    enumerate_structured_ring_system_deletes,
    matching_ring_system_template_indices,
    ring_system_placement_has_local_atom_support,
    ring_system_placement_local_atom_type_mask,
    ring_system_grow_electronic_key,
    ring_system_electronic_template_support_mask,
    RingSystemPlacement,
    ring_system_template_local_support_mask,
    ring_system_template_support_mask,
    structured_ring_system_electronic_aliases,
    structured_ring_system_template_aliases,
    structured_ring_system_templates,
)
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import execute_trace
from compose_v4.rewrite.tracelets import (
    BondOrderChange,
    RingBond,
    inverse_ring_system_grow,
    is_valid_ring_system_delete,
    is_valid_ring_system_grow,
)
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import (
    build_typed_ring_catalog,
    build_typed_ring_catalog_from_paths,
)


@pytest.mark.parametrize(
    "smiles",
    (
        "C1CCCCC1",
        "c1ccncc1",
        "C1CCC2CCCCC2C1",
        "C1CC2CCC1C2",
        "C1CCC2(CC1)CCCC2",
    ),
)
def test_exact_ring_system_fiber_contains_teacher_and_inverse(smiles: str) -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 20)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(91),
        n_slots=20,
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((trace,))
    _, states = execute_trace(trace.source, trace.steps, return_states=True)
    progress = next(
        index
        for index, step in enumerate(trace.steps)
        if step.rule_name == "ring_system_grow"
    )
    teacher = trace.steps[progress].action
    inverse = inverse_ring_system_grow(states[progress], teacher)
    grows = enumerate_ring_system_grows(states[progress], catalog)
    deletes = enumerate_ring_system_deletes(states[progress + 1], catalog)

    assert teacher in grows
    assert inverse in deletes
    assert all(is_valid_ring_system_grow(states[progress], action) for action in grows)
    assert all(
        is_valid_ring_system_delete(states[progress + 1], action)
        for action in deletes
    )


def test_structured_delete_uninstalls_unseen_heteroatom_labels() -> None:
    training_target = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 12)
    heldout_target = pad_molecular_graph(smiles_to_molecular_graph("c1ccncc1"), 12)
    source = DegreeBoundedCarbonTreePrior(sizes=(6,)).sample(
        np.random.default_rng(211),
        n_slots=12,
    )
    training_trace = compile_carbon_tree_to_target(
        source,
        training_target,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((training_trace,))
    deletes = enumerate_structured_ring_system_deletes(heldout_target, catalog)

    assert deletes
    assert all(is_valid_ring_system_delete(heldout_target, action) for action in deletes)


def test_tree_dp_template_support_equals_explicit_match_existence() -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("c1ccncc1"), 12)
    source = DegreeBoundedCarbonTreePrior(sizes=(6,)).sample(
        np.random.default_rng(613),
        n_slots=12,
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((trace,))
    templates = structured_ring_system_templates(catalog)
    progress = next(
        index
        for index, step in enumerate(trace.steps)
        if step.rule_name == "ring_system_grow"
    )
    state = execute_trace(
        trace.source,
        trace.steps[:progress],
    )
    support = ring_system_template_support_mask(state, templates)

    assert support.tolist() == [
        bool(enumerate_ring_system_template_placements(state, template))
        for template in templates
    ]


def test_capacity_decorated_support_equals_explicit_local_support() -> None:
    target = pad_molecular_graph(
        smiles_to_molecular_graph("c1ccc2ncccc2c1"),
        20,
    )
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(617),
        n_slots=20,
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((trace,))
    templates = structured_ring_system_templates(catalog)
    aliases = structured_ring_system_template_aliases(catalog)
    progress = next(
        index
        for index, step in enumerate(trace.steps)
        if step.rule_name == "ring_system_grow"
    )
    state = execute_trace(trace.source, trace.steps[:progress])
    support = ring_system_template_local_support_mask(state, templates, aliases)
    explicit = tuple(
        any(
            ring_system_placement_has_local_atom_support(state, placement)
            for alias in template_aliases
            for placement in enumerate_ring_system_template_placements(state, alias)
        )
        for template_aliases in aliases
    )

    assert support.tolist() == list(explicit)


def test_local_ring_label_support_rejects_overloaded_placement() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CC(C)(C)C"), 8)
    real = tuple(int(slot) for slot in np.flatnonzero(state.atom_types != 0))
    center = next(
        slot for slot in real if int(np.count_nonzero(state.bonds[slot])) == 4
    )
    leaves = tuple(
        slot for slot in real if int(state.bonds[center, slot]) == 1
    )[:2]
    members = tuple(sorted((center, *leaves)))
    scaffold = tuple(
        RingBond(min(center, leaf), max(center, leaf), 1) for leaf in leaves
    )
    closure = (RingBond(min(leaves), max(leaves), 1),)
    possible = RingSystemPlacement(
        system_atoms=members,
        scaffold_bonds=scaffold,
        bond_reorders=(),
        bond_insertions=closure,
        aromatic_edges=(),
        topology_class="monocyclic",
    )
    overloaded = RingSystemPlacement(
        system_atoms=members,
        scaffold_bonds=scaffold,
        bond_reorders=(
            BondOrderChange(min(center, leaves[0]), max(center, leaves[0]), 2),
        ),
        bond_insertions=closure,
        aromatic_edges=(),
        topology_class="monocyclic",
    )

    possible_mask = ring_system_placement_local_atom_type_mask(state, possible)
    overloaded_mask = ring_system_placement_local_atom_type_mask(state, overloaded)
    center_index = members.index(center)

    assert possible_mask.any(axis=1).all()
    assert ring_system_placement_has_local_atom_support(state, possible)
    assert not overloaded_mask[center_index].any()
    assert not ring_system_placement_has_local_atom_support(state, overloaded)


def test_exact_electronic_candidates_contain_teacher_without_label_mixing() -> None:
    paths = []
    for seed, smiles in enumerate(("c1cc[nH]c1", "c1ccoc1"), start=701):
        target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 12)
        source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
            np.random.default_rng(seed),
            n_slots=12,
        )
        trace = compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            align_source=True,
        )
        paths.append(TraceProgressCTMC(trace))
    catalog = build_typed_ring_catalog_from_paths(tuple(paths))
    templates = structured_ring_system_templates(catalog)
    electronic_groups = structured_ring_system_electronic_aliases(catalog)
    nitrogen = int(ELEMENT_TO_IDX["N"])
    oxygen = int(ELEMENT_TO_IDX["O"])

    for path in paths:
        progress = next(
            index
            for index, step in enumerate(path.trace.steps)
            if step.rule_name == "ring_system_grow"
        )
        teacher = path.trace.steps[progress].action
        template_indices = matching_ring_system_template_indices(teacher, templates)
        electronic_support = ring_system_electronic_template_support_mask(
            path.state_at(progress),
            templates,
            electronic_groups,
        )
        explicit_support = tuple(
            bool(
                enumerate_executable_ring_grow_candidates(
                    path.state_at(progress),
                    aliases,
                )
            )
            for aliases in electronic_groups
        )
        candidates = tuple(
            candidate
            for index in template_indices
            for candidate in enumerate_executable_ring_grow_candidates(
                path.state_at(progress),
                electronic_groups[index],
            )
        )

        assert ring_system_grow_electronic_key(teacher) in {
            ring_system_grow_electronic_key(candidate.action) for candidate in candidates
        }
        assert electronic_support.tolist() == list(explicit_support)
        assert all(
            is_valid_ring_system_grow(path.state_at(progress), candidate.action)
            for candidate in candidates
        )
        assert all(
            not ({nitrogen, oxygen} <= set(candidate.atom_types))
            for candidate in candidates
        )
    structured_ring_system_electronic_aliases,
