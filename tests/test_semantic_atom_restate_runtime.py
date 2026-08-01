"""Production-boundary tests for semantic Editing-V2 atom restatement."""

from __future__ import annotations

import pytest

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.experiments.cycle_open_kekule_invariance import (
    build_alternate_kekule_pair,
)
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    de_novo_rewrite_system,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    AtomRestate,
    SemanticAtomRestate,
    apply_atom_restate,
    enumerate_semantic_atom_restates,
)
from compose_v4.rewrite.semantic_atom_restate import (
    SemanticAtomRestateRejectionCode,
    prepare_semantic_atom_restate_context,
    resolve_semantic_atom_restate,
)


def _class_index(symbol: str, valence: int) -> int:
    from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX

    return ORGANIC_VOCABULARY.class_index(
        ELEMENT_TO_IDX[symbol],
        bond_order_sum=valence,
        implicit_h_count=0,
    )


def _aromatic_pair():
    return build_alternate_kekule_pair(
        pad_molecular_graph(smiles_to_molecular_graph("Cc1cccc(Cl)c1"), 16)
    )


def test_ambiguous_sulfur_restate_is_rejected_from_both_kekule_phases() -> None:
    pair = _aromatic_pair()
    sulfur_six = _class_index("S", 6)
    assert sulfur_six is not None
    for source in (pair.original, pair.alternate):
        resolution = resolve_semantic_atom_restate(
            source,
            vertex=5,
            target_class_index=sulfur_six,
        )
        assert resolution.admitted is False
        assert (
            resolution.rejection_code
            is SemanticAtomRestateRejectionCode.AMBIGUOUS_CANONICAL_PRODUCT
        )
        assert resolution.component_assignment_count == 2
        assert resolution.target_compatible_alias_count == 2
        assert len(resolution.canonical_product_keys) == 2


def test_common_aromatic_nitrogen_restate_is_phase_invariant() -> None:
    pair = _aromatic_pair()
    nitrogen_three = _class_index("N", 3)
    assert nitrogen_three is not None
    resolutions = tuple(
        resolve_semantic_atom_restate(
            source,
            vertex=2,
            target_class_index=nitrogen_three,
        )
        for source in (pair.original, pair.alternate)
    )
    assert all(resolution.admitted for resolution in resolutions)
    assert all(
        resolution.target_compatible_alias_count == 2 for resolution in resolutions
    )
    assert (
        resolutions[0].canonical_product_keys == resolutions[1].canonical_product_keys
    )
    assert resolutions[0].successor is not None
    assert resolutions[1].successor is not None
    assert canonical_state_key(resolutions[0].successor) == "Cc1cc(Cl)ccn1"
    assert canonical_state_key(resolutions[1].successor) == "Cc1cc(Cl)ccn1"


def test_nonaromatic_semantic_restate_matches_legacy_molecular_product() -> None:
    source = smiles_to_molecular_graph("CCC")
    nitrogen_three = _class_index("N", 3)
    assert nitrogen_three is not None
    resolution = resolve_semantic_atom_restate(
        source,
        vertex=1,
        target_class_index=nitrogen_three,
    )
    assert resolution.admitted
    assert resolution.semantic_aromatic_site is False
    assert resolution.successor is not None
    legacy = apply_atom_restate(
        source,
        AtomRestate(
            v=1,
            atom_type=ORGANIC_VOCABULARY.element_of(nitrogen_three),
            formal_charge=0,
            implicit_h_count=1,
        ),
    )
    assert canonical_state_key(resolution.successor) == canonical_state_key(legacy)


def test_semantic_enumerator_has_the_same_productive_law_across_phases() -> None:
    pair = _aromatic_pair()
    runtime = editing_v2_semantic_rewrite_system()

    def molecular_fiber(source):
        return {
            action: canonical_state_key(
                runtime.apply(source, "atom_restate_semantic", action)
            )
            for action in enumerate_semantic_atom_restates(source)
        }

    assert molecular_fiber(pair.original) == molecular_fiber(pair.alternate)


def test_complete_editing_runtime_excludes_raw_atom_restate() -> None:
    source = smiles_to_molecular_graph("CCC")
    runtime = editing_v2_semantic_rewrite_system()
    semantic = SemanticAtomRestate(1, _class_index("N", 3))
    successor = runtime.apply(source, "atom_restate_semantic", semantic)
    assert canonical_state_key(successor) == "CNC"
    with pytest.raises(InvalidRewrite, match="unknown rewrite rule: atom_restate"):
        runtime.apply(source, "atom_restate", AtomRestate(1, 3, 0, 1))
    with pytest.raises(InvalidRewrite, match="unknown rewrite rule"):
        de_novo_rewrite_system().apply(
            source,
            "atom_restate_semantic",
            semantic,
        )


def test_precomputed_context_is_bound_to_exact_molecular_source() -> None:
    pair = _aromatic_pair()
    context = prepare_semantic_atom_restate_context(pair.original)
    with pytest.raises(ValueError, match="another source"):
        resolve_semantic_atom_restate(
            pair.alternate,
            vertex=2,
            target_class_index=_class_index("N", 3),
            context=context,
        )


def test_charge_policy_is_enforced_for_edited_and_unedited_charged_centers() -> None:
    source = pad_molecular_graph(smiles_to_molecular_graph("[nH+]1ccccc1"), 16)
    carbon_four = _class_index("C", 4)
    nitrogen_three = _class_index("N", 3)
    assert carbon_four is not None
    assert nitrogen_three is not None

    charged_target = resolve_semantic_atom_restate(
        source,
        vertex=0,
        target_class_index=carbon_four,
    )
    assert charged_target.admitted is False
    assert (
        charged_target.rejection_code
        is SemanticAtomRestateRejectionCode.UNSUPPORTED_FORMAL_CHARGE
    )

    neutral_target = resolve_semantic_atom_restate(
        source,
        vertex=1,
        target_class_index=nitrogen_three,
    )
    assert neutral_target.admitted
    assert neutral_target.component_assignment_count == 2
    assert neutral_target.target_compatible_alias_count == 2
    assert neutral_target.charge_policy_preserving_product_count == 1
    assert neutral_target.successor is not None
    assert canonical_state_key(neutral_target.successor) == "c1cc[nH+]nc1"
    assert charge_policy_preserved(source, neutral_target.successor)
