"""Production factorized marked law -> canonical molecular successor integration.

These tests use real model masks, the real executor, and real molecular
canonicalization.  The independent dictionary implementation is imported only
here as an oracle; reported results use the segmented production path.
"""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pytest
import torch

import compose_v4.model.factorized_tracelet_rate_model as rate_model_module
from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import empty_molecular_graph, pad_molecular_graph
from compose_v4.data.charge_policy import (
    CHARGE_POLICY_VERSION,
    charge_policy_preserved,
)
from compose_v4.experiments.production_successor_kernel import (
    ProductionSuccessorKernelError,
    _TABLE_FAMILIES,
    _coordinate_action,
    canonical_successor_result,
    enumerate_factorized_marked_law,
)
from compose_v4.experiments.score_free_successor_support import (
    enumerate_factorized_legal_support_many,
)
from compose_v4.experiments.reference_successor_kernel import (
    compare_against_reference,
    reference_successor_batch,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.ring_system_fiber import (
    enumerate_clean_ring_system_deletes,
)
from compose_v4.rewrite.tracelet_fiber import (
    enumerate_ring_system_restate_actions,
)
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SLOTS = 12
_SYSTEM = de_novo_rewrite_system()
_PRIMITIVE_TABLE_FAMILIES = _TABLE_FAMILIES[:8]
_ACTIVE_EIGHT_FAMILIES = {
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_insert",
    "cycle_attach",
    "ring_system_restate",
}
_CHARGED_FIXTURES = (
    "C[N+](C)(C)C",
    "C1CC[NH2+]CC1",
    "CC[NH2+]CC",
    "CC(=O)[O-]",
    "C[N+](C)(C)Cc1ccccc1",
)


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), _SLOTS)


@pytest.fixture(scope="module")
def ring_catalog():
    def trace(smiles: str):
        target = _state(smiles)
        source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
            np.random.default_rng(1), n_slots=_SLOTS
        )
        return compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            align_source=True,
        )

    return build_typed_ring_catalog(tuple(trace(smiles) for smiles in ("c1ccccc1", "C1CCNCC1")))


@pytest.fixture(scope="module")
def model(ring_catalog):
    torch.manual_seed(7)
    return FactorizedTraceletRateModel(
        ring_catalog,
        hidden_dim=12,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()


def _prepared_batch(model, state, *, compute_ring_system_delete=None):
    capabilities = model.operator_capabilities
    return prepare_factorized_mark_batch(
        (state,),
        (0.37,),
        (None,),
        (None,),
        (0.0,),
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=capabilities.compute_ring_grow_support,
        compute_ring_restates=capabilities.compute_ring_restates,
        compute_cyclic_graft=capabilities.compute_cyclic_graft,
        compute_ring_opening=capabilities.compute_ring_opening,
        compute_ring_system_delete=(
            capabilities.compute_ring_system_delete
            if compute_ring_system_delete is None
            else compute_ring_system_delete
        ),
    )


def _coordinates(mask):
    return {tuple(int(value) for value in row) for row in torch.nonzero(mask[0], as_tuple=False)}


def test_score_free_support_matches_scored_legal_coordinates(model, monkeypatch):
    states = (_state("CCO"), _state("c1ccccc1"))
    scored = tuple(enumerate_factorized_marked_law(model, state, 0.37) for state in states)

    def encoder_must_not_run(*_args, **_kwargs):
        raise AssertionError("score-free support called the neural encoder")

    monkeypatch.setattr(model, "_encode_batch", encoder_must_not_run)
    selected_family = frozenset({"atom_delete", "cycle_attach"})
    support = enumerate_factorized_legal_support_many(
        model,
        states,
        (0.37, 0.37),
        included_families=(selected_family, selected_family),
    )
    for row, law in zip(support, scored, strict=True):
        assert row.raw_mark_count == len(law.marks)
        expected = {
            (mark.family_name, mark.table_name, mark.coordinate)
            for mark in law.marks
            if mark.family_name in selected_family
        }
        observed = {(mark.family_name, mark.table_name, mark.coordinate) for mark in row.marks}
        assert observed == expected


def _raw_and_filtered_tables(model, state):
    batch = _prepared_batch(model, state)
    with torch.no_grad():
        node, global_state, pair = model._encode_batch(batch)
        with patch.object(
            rate_model_module,
            "_apply_charge_policy_to_action_masks",
            side_effect=lambda prepared, masks: dict(masks),
        ):
            raw_masks, _, _ = model._action_tables(
                batch,
                node,
                global_state,
                pair,
                require_exact_ring_support=False,
            )
        filtered_masks, _, _ = model._action_tables(
            batch,
            node,
            global_state,
            pair,
            require_exact_ring_support=False,
        )
    return batch, raw_masks, filtered_masks


def _oracle_preserving_actions(state, rule_name, actions):
    retained = []
    for action in actions:
        try:
            successor = _SYSTEM.apply(state, rule_name, action)
        except InvalidRewrite:
            continue
        if charge_policy_preserved(state, successor):
            retained.append(action)
    return tuple(retained)


@pytest.mark.parametrize("smiles", ("CCO", "c1ccccc1", "C1CCNCC1"))
def test_complete_marked_law_is_normalized_and_executable(model, smiles):
    state = _state(smiles)
    law = enumerate_factorized_marked_law(model, state, 0.37)

    assert law.marks
    assert law.total_probability == pytest.approx(1.0, abs=2e-5)
    assert law.total_hazard > 0.0
    for mark in law.marks:
        successor = _SYSTEM.apply(
            state,
            mark.executor_rule_name,
            mark.action,
        )
        assert canonical_state_key(successor)


def test_charged_marked_law_is_exhaustively_charge_preserving(model):
    represented_families = set()
    for smiles in _CHARGED_FIXTURES:
        state = _state(smiles)
        law = enumerate_factorized_marked_law(model, state, 0.37)

        assert law.marks
        assert law.total_probability == pytest.approx(1.0, abs=2e-5)
        for mark in law.marks:
            successor = _SYSTEM.apply(
                state,
                mark.executor_rule_name,
                mark.action,
            )
            assert charge_policy_preserved(state, successor), (
                smiles,
                mark.family_name,
                mark.action,
            )
            represented_families.add(mark.family_name)

    assert represented_families == _ACTIVE_EIGHT_FAMILIES


def test_fast_primitive_charge_masks_equal_exhaustive_executor_oracle(model):
    retained_families = set()
    excluded_families = set()

    for smiles in _CHARGED_FIXTURES:
        state = _state(smiles)
        batch, raw_masks, filtered_masks = _raw_and_filtered_tables(model, state)

        for family_name, table_name in _PRIMITIVE_TABLE_FAMILIES:
            raw_coordinates = _coordinates(raw_masks[table_name])
            expected_coordinates = set()
            for coordinate in raw_coordinates:
                rule_name, action = _coordinate_action(
                    model,
                    state,
                    batch,
                    family_name=family_name,
                    table_name=table_name,
                    coordinate=coordinate,
                )
                successor = _SYSTEM.apply(state, rule_name, action)
                if charge_policy_preserved(state, successor):
                    expected_coordinates.add(coordinate)

            observed_coordinates = _coordinates(filtered_masks[table_name])
            assert observed_coordinates == expected_coordinates, (
                smiles,
                family_name,
                table_name,
                sorted(observed_coordinates ^ expected_coordinates),
            )
            if expected_coordinates:
                retained_families.add(family_name)
            if raw_coordinates - expected_coordinates:
                excluded_families.add(family_name)

    primitive_families = {family_name for family_name, _ in _PRIMITIVE_TABLE_FAMILIES}
    assert retained_families == primitive_families
    assert excluded_families == primitive_families


def test_ring_macro_candidates_equal_exact_charge_policy_oracle(
    model,
    ring_catalog,
):
    removed = {
        "ring_system_delete": False,
        "ring_system_restate": False,
    }
    retained = {
        "ring_system_delete": False,
        "ring_system_restate": False,
    }
    for smiles in (
        "C1CC[NH2+]CC1",
        "C[N+](C)(C)Cc1ccccc1",
    ):
        state = _state(smiles)
        batch = _prepared_batch(
            model,
            state,
            compute_ring_system_delete=True,
        )

        raw_delete = enumerate_clean_ring_system_deletes(
            state,
            ring_catalog,
        )
        expected_delete = _oracle_preserving_actions(
            state,
            "ring_system_delete",
            raw_delete,
        )
        assert batch.ring_delete_actions is not None
        assert batch.ring_delete_actions[0] == expected_delete
        removed["ring_system_delete"] |= bool(set(raw_delete) - set(expected_delete))
        retained["ring_system_delete"] |= bool(expected_delete)

        raw_restate = enumerate_ring_system_restate_actions(
            state,
            system=_SYSTEM,
        )
        expected_restate = _oracle_preserving_actions(
            state,
            "ring_system_restate",
            raw_restate,
        )
        assert batch.ring_restate_actions[0] == expected_restate
        removed["ring_system_restate"] |= bool(set(raw_restate) - set(expected_restate))
        retained["ring_system_restate"] |= bool(expected_restate)

    assert removed == {
        "ring_system_delete": True,
        "ring_system_restate": True,
    }
    assert retained == {
        "ring_system_delete": True,
        "ring_system_restate": True,
    }


@pytest.mark.parametrize("smiles", ("CCO", "c1ccccc1", "C1CCNCC1"))
def test_charge_policy_intersection_preserves_uncharged_support(
    model,
    ring_catalog,
    smiles,
):
    state = _state(smiles)
    batch, raw_masks, filtered_masks = _raw_and_filtered_tables(model, state)
    for _, table_name in _PRIMITIVE_TABLE_FAMILIES:
        assert torch.equal(
            raw_masks[table_name],
            filtered_masks[table_name],
        ), table_name

    macro_batch = _prepared_batch(
        model,
        state,
        compute_ring_system_delete=True,
    )
    assert macro_batch.ring_delete_actions is not None
    assert macro_batch.ring_delete_actions[0] == (
        enumerate_clean_ring_system_deletes(state, ring_catalog)
    )
    assert macro_batch.ring_restate_actions[0] == (
        enumerate_ring_system_restate_actions(state, system=_SYSTEM)
    )


def test_null_root_marked_law_contains_only_executable_valence_classes(model):
    state = empty_molecular_graph(_SLOTS)
    law = enumerate_factorized_marked_law(model, state, 0.37)

    assert law.marks
    assert {mark.family_name for mark in law.marks} == {"atom_insert"}
    assert law.total_probability == pytest.approx(1.0, abs=2e-5)
    for mark in law.marks:
        successor = _SYSTEM.apply(
            state,
            mark.executor_rule_name,
            mark.action,
        )
        assert canonical_state_key(successor)


def test_segmented_production_path_matches_dictionary_oracle(model):
    state = _state("c1ccccc1")
    result = canonical_successor_result(model, state, 0.41)
    reference = reference_successor_batch(
        state,
        [
            (
                mark.executor_rule_name,
                mark.action,
                mark.probability,
            )
            for mark in result.marked_law.marks
        ],
        system=_SYSTEM,
        identity=result.batch.identity,
    )

    produced_probabilities = {
        successor.key: successor.probability for successor in result.batch.successors
    }
    reference_probabilities = {
        successor.key: successor.probability for successor in reference.successors
    }
    assert (
        compare_against_reference(
            produced_probabilities,
            reference_probabilities,
            tolerance=2e-7,
        )
        == []
    )
    assert {successor.key: successor.alias_count for successor in result.batch.successors} == {
        successor.key: successor.alias_count for successor in reference.successors
    }
    assert result.batch.virtual_mass == pytest.approx(
        reference.virtual_mass,
        abs=2e-7,
    )


def test_alias_mass_is_summed_before_productive_conditioning(model):
    state = _state("c1ccccc1")
    result = canonical_successor_result(model, state, 0.29)
    aliased = [successor for successor in result.batch.successors if successor.alias_count > 1]
    assert aliased, "symmetric benzene should expose at least one aliased successor"

    raw_mass_by_successor: dict[str, float] = defaultdict(float)
    raw_count_by_successor: dict[str, int] = defaultdict(int)
    for mark in result.marked_law.marks:
        successor = _SYSTEM.apply(
            state,
            mark.executor_rule_name,
            mark.action,
        )
        key = canonical_state_key(successor)
        if key == result.batch.source_key:
            continue
        raw_mass_by_successor[key] += mark.probability
        raw_count_by_successor[key] += 1

    for successor in aliased:
        assert successor.alias_count == raw_count_by_successor[successor.key]
        assert successor.probability == pytest.approx(
            raw_mass_by_successor[successor.key] / result.diagnostics.raw_productive_mass,
            abs=2e-7,
        )


def test_compositional_ring_close_and_open_are_both_in_the_marked_law(model):
    acyclic = enumerate_factorized_marked_law(model, _state("CCCCCC"), 0.5)
    cyclic = enumerate_factorized_marked_law(model, _state("C1CCCCC1"), 0.5)

    assert any(mark.family_name == "cycle_insert" for mark in acyclic.marks)
    assert any(mark.family_name == "cycle_attach" for mark in cyclic.marks)
    assert not any(mark.family_name == "ring_system_grow" for mark in cyclic.marks)


def test_delete_disabled_direct_kernel_identity_and_support_are_explicit(
    ring_catalog,
):
    model = FactorizedTraceletRateModel(
        ring_catalog,
        hidden_dim=8,
        message_passing_steps=1,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()
    result = canonical_successor_result(model, _state("C1CCCCC1"), 0.5)

    flags = dict(result.batch.identity.support_signature.capability_flags)
    assert flags["enable_ring_system_delete"] is False
    assert result.batch.identity.support_signature.charge_policy == CHARGE_POLICY_VERSION
    assert result.batch.identity.support_signature.ringcore_configuration.endswith(
        ":ring_system_delete_disabled"
    )
    assert not any(mark.family_name == "ring_system_delete" for mark in result.marked_law.marks)
    assert any(mark.family_name == "cycle_attach" for mark in result.marked_law.marks)


def test_legacy_ring_macro_fails_closed(ring_catalog):
    model = FactorizedTraceletRateModel(
        ring_catalog,
        hidden_dim=8,
        message_passing_steps=1,
        enable_ring_grow_macro=True,
        enable_cycle_ops=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()
    with pytest.raises(ProductionSuccessorKernelError, match="legacy ring_system_grow"):
        enumerate_factorized_marked_law(model, _state("CCO"), 0.5)
