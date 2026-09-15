from compose_v4.control.structural_subgoal_components import (
    CORE_COMPONENT_FAMILIES,
    GRANULAR_COMPONENT_FAMILIES,
    ExactComponentVocabulary,
    atom_attribute_tokens,
    atom_attributes,
    attachment_edge_tokens,
    attachment_pattern,
    bond_attribute_tokens,
    bond_attributes,
    component_families,
    dependency_motif,
    dependency_tokens,
    exact_equivalent,
    source_graphlets_up_to_3,
    source_region_motif,
    target_graphlets_up_to_3,
    target_radius_fragments,
    target_topology,
    target_topology_graphlets_up_to_3,
    whole_patch_component,
)
from compose_v4.control.structural_subgoal_policy import StructuralDeltaTemplate


def _template(*, swapped: bool = False, output_bond: int = 1):
    carbon = (0, 0, 2, 1)
    nitrogen = (1, 0, 1, 2)
    oxygen = (2, 0, 1, 1)
    if not swapped:
        return StructuralDeltaTemplate(
            input_atoms=(carbon, nitrogen),
            input_bonds=((0, 1), (1, 0)),
            target_atoms=(carbon, nitrogen),
            output_atoms=(oxygen,),
            target_bonds=((0, 1, 0), (1, 0, output_bond), (0, output_bond, 0)),
        )
    return StructuralDeltaTemplate(
        input_atoms=(nitrogen, carbon),
        input_bonds=((0, 1), (1, 0)),
        target_atoms=(nitrogen, carbon),
        output_atoms=(oxygen,),
        target_bonds=((0, 1, output_bond), (1, 0, 0), (output_bond, 0, 0)),
    )


def test_component_equivalence_is_invariant_to_input_role_order():
    left, right = _template(), _template(swapped=True)
    for constructor in (
        whole_patch_component,
        source_region_motif,
        target_topology,
        atom_attributes,
        bond_attributes,
        attachment_pattern,
    ):
        assert exact_equivalent(constructor(left), constructor(right))


def test_component_equivalence_requires_exact_attributes_after_hash_bucket():
    single = _template(output_bond=1)
    double = _template(output_bond=2)
    assert not exact_equivalent(
        whole_patch_component(single), whole_patch_component(double)
    )
    assert exact_equivalent(target_topology(single), target_topology(double))
    assert not exact_equivalent(bond_attributes(single), bond_attributes(double))
    vocabulary = ExactComponentVocabulary(
        (whole_patch_component(single), whole_patch_component(single))
    )
    assert vocabulary.size == 1
    assert not vocabulary.contains(whole_patch_component(double))


def test_fragment_families_are_rooted_bounded_and_address_free():
    left, right = _template(), _template(swapped=True)
    radius_one = target_radius_fragments(left, 1)
    radius_two = target_radius_fragments(left, 2)
    graphlets = target_graphlets_up_to_3(left)
    assert len(radius_one) == len(radius_two) == 3
    assert all(len(row.node_labels) <= 3 for row in radius_one)
    assert all(len(row.node_labels) <= 3 for row in radius_two)
    assert len(graphlets) == 6
    right_vocab = ExactComponentVocabulary(target_graphlets_up_to_3(right))
    assert all(right_vocab.contains(row) for row in graphlets)


def test_granular_components_transfer_under_role_permutation():
    left, right = _template(), _template(swapped=True)
    constructors = (
        atom_attribute_tokens,
        bond_attribute_tokens,
        attachment_edge_tokens,
        source_graphlets_up_to_3,
        target_topology_graphlets_up_to_3,
    )
    for constructor in constructors:
        left_components = constructor(left)
        right_vocabulary = ExactComponentVocabulary(constructor(right))
        assert left_components
        assert all(right_vocabulary.contains(row) for row in left_components)


def test_dependency_motif_preserves_direction_and_rule_labels_without_indices():
    actions = (
        {"executor_rule": "atom_insert"},
        {"executor_rule": "cycle_close"},
    )
    component = {"primitive_indices": [0, 1]}
    first = dependency_motif(
        actions,
        component,
        created_dependency_edges=({"producer": 0, "consumer": 1},),
        cycle_dependency_edges=(),
    )
    second = dependency_motif(
        actions[::-1],
        component,
        created_dependency_edges=({"producer": 1, "consumer": 0},),
        cycle_dependency_edges=(),
    )
    assert exact_equivalent(first, second)
    no_edge = dependency_motif(
        actions,
        component,
        created_dependency_edges=(),
        cycle_dependency_edges=(),
    )
    assert not exact_equivalent(first, no_edge)
    first_tokens = dependency_tokens(first)
    second_vocabulary = ExactComponentVocabulary(dependency_tokens(second))
    assert all(second_vocabulary.contains(row) for row in first_tokens)


def test_component_family_registry_exposes_strict_and_granular_views():
    dependency = dependency_motif(
        ({"executor_rule": "atom_insert"},),
        {"primitive_indices": [0]},
        created_dependency_edges=(),
        cycle_dependency_edges=(),
    )
    families = component_families(_template(), dependency)
    expected = {
        "whole_patch",
        *CORE_COMPONENT_FAMILIES,
        *GRANULAR_COMPONENT_FAMILIES,
        "target_radius_1_fragments",
        "target_radius_2_fragments",
        "target_graphlets_up_to_3",
    }
    assert set(families) == expected
    assert all(families[name] for name in expected)
