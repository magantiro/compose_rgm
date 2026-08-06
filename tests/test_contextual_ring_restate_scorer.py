"""Score-only contextual capacity for canonical ring restatements."""

from __future__ import annotations

import numpy as np
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.editing_gate_zero_runtime import (
    build_production_ringcore_catalog,
)
from compose_v4.experiments.quotient_invariance import permute_persistent_slots
from compose_v4.experiments.successor_micro_overfit import (
    configure_micro_overfit_parameters,
)
from compose_v4.model.contextual_ring_restate_rate_model import (
    CONTEXTUAL_RING_RESTATE_SCORER_MODE,
    ContextualRingRestateFactorizedTraceletRateModel,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    prepare_factorized_mark_batch,
)
from compose_v4.model.relational_reroute_rate_model import (
    RelationalRerouteFactorizedTraceletRateModel,
)


def _model(model_type):
    return model_type(
        build_production_ringcore_catalog(max_atoms=40),
        hidden_dim=12,
        message_passing_steps=1,
        mark_dim=8,
        enable_ring_restates=True,
        ring_restate_scorer_mode=SEMANTIC_RING_RESTATE_SCORER_MODE,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()


def _batch(model, smiles: str):
    state = pad_molecular_graph(smiles_to_molecular_graph(smiles), 40)
    capabilities = model.operator_capabilities
    return state, prepare_factorized_mark_batch(
        (state,),
        (0.37,),
        (None,),
        (None,),
        (0.0,),
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=capabilities.compute_ring_grow_support,
        compute_ring_restates=capabilities.compute_ring_restates,
        compute_cyclic_graft=capabilities.compute_cyclic_graft,
        compute_ring_opening=capabilities.compute_ring_opening,
        compute_ring_system_delete=capabilities.compute_ring_system_delete,
        editing_process_semantics=capabilities.editing_process_semantics,
        atom_restate_action_semantics=capabilities.atom_restate_action_semantics,
        ring_restate_scorer_mode=capabilities.ring_restate_scorer_mode,
        cycle_close_action_semantics=capabilities.cycle_close_action_semantics,
        cycle_open_action_semantics=capabilities.cycle_open_action_semantics,
    )


def test_contextual_revision_is_one_zero_initialized_score_residual() -> None:
    torch.manual_seed(41)
    predecessor = _model(RelationalRerouteFactorizedTraceletRateModel)
    torch.manual_seed(41)
    revised = _model(ContextualRingRestateFactorizedTraceletRateModel)

    predecessor_state = predecessor.state_dict()
    revised_state = revised.state_dict()
    assert all(
        name in revised_state and torch.equal(value, revised_state[name])
        for name, value in predecessor_state.items()
    )
    assert tuple(sorted(set(revised_state) - set(predecessor_state))) == (
        "ring_restate_context_head.weight",
    )
    assert not bool(revised.ring_restate_context_head.weight.count_nonzero())
    assert (
        revised.ring_restate_context_scorer_mode
        == CONTEXTUAL_RING_RESTATE_SCORER_MODE
    )

    _state, batch = _batch(
        predecessor,
        "C(CN1CCC(Cc2ccccc2)CC1)#[PH]C1=NC=NC1",
    )
    predecessor_node, predecessor_global, predecessor_pair = (
        predecessor._encode_batch(batch)
    )
    revised_node, revised_global, revised_pair = revised._encode_batch(batch)
    predecessor_masks, predecessor_logits, predecessor_log_z = (
        predecessor._action_tables(
            batch,
            predecessor_node,
            predecessor_global,
            predecessor_pair,
        )
    )
    revised_masks, revised_logits, revised_log_z = revised._action_tables(
        batch,
        revised_node,
        revised_global,
        revised_pair,
    )
    assert predecessor_masks.keys() == revised_masks.keys()
    assert predecessor_logits.keys() == revised_logits.keys()
    assert all(
        torch.equal(predecessor_masks[name], revised_masks[name])
        for name in predecessor_masks
    )
    assert all(
        torch.equal(predecessor_logits[name], revised_logits[name])
        for name in predecessor_logits
    )
    assert torch.equal(predecessor_log_z, revised_log_z)

    selected = configure_micro_overfit_parameters(
        revised,
        ("ring_system_restate",),
        scope="heads_only",
    )
    assert "ring_restate_context_head.weight" in selected


def test_context_separates_successors_with_equal_bond_descriptors() -> None:
    torch.manual_seed(43)
    model = _model(ContextualRingRestateFactorizedTraceletRateModel)
    _state, batch = _batch(
        model,
        "C(CN1CCC(Cc2ccccc2)CC1)#[PH]C1=NC=NC1",
    )
    descriptors = batch.ring_restate_successor_group_descriptors
    assert descriptors is not None
    row = tuple(variants[0] for variants in descriptors[0])
    equal_descriptor_groups = tuple(
        (left, right)
        for left in range(len(row))
        for right in range(left + 1, len(row))
        if row[left] == row[right]
    )
    assert len(equal_descriptor_groups) == 1

    node, global_state, pair = model._encode_batch(batch)
    del global_state
    features = model._ring_restate_context_features(batch, node, pair)
    action_by_group = {
        int(group_index): action_index
        for action_index, group_index in enumerate(
            batch.ring_restate_successor_group_ids[0]
        )
    }
    left, right = equal_descriptor_groups[0]
    distance = torch.linalg.vector_norm(
        features[0, action_by_group[left]]
        - features[0, action_by_group[right]]
    )
    assert float(distance.detach()) > 1e-4


def test_raw_aliases_share_one_canonical_group_context() -> None:
    torch.manual_seed(47)
    model = _model(ContextualRingRestateFactorizedTraceletRateModel)
    _state, batch = _batch(model, "c1ccccc1-c1ccccc1")
    assert batch.ring_restate_successor_group_ids == ((0, 0),)
    node, _global_state, pair = model._encode_batch(batch)
    features = model._ring_restate_context_features(batch, node, pair)
    assert torch.allclose(features[0, 0], features[0, 1], atol=1e-7, rtol=1e-7)


def test_context_feature_is_persistent_slot_relabel_equivariant() -> None:
    torch.manual_seed(53)
    model = _model(ContextualRingRestateFactorizedTraceletRateModel)
    state, batch = _batch(model, "Cc1ccccc1CCc1cccc(Cl)c1")
    permutation = np.random.default_rng(7).permutation(state.n_atoms)
    relabeled = permute_persistent_slots(state, tuple(int(v) for v in permutation))
    capabilities = model.operator_capabilities
    relabeled_batch = prepare_factorized_mark_batch(
        (relabeled,),
        (0.37,),
        (None,),
        (None,),
        (0.0,),
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=capabilities.compute_ring_grow_support,
        compute_ring_restates=capabilities.compute_ring_restates,
        compute_cyclic_graft=capabilities.compute_cyclic_graft,
        compute_ring_opening=capabilities.compute_ring_opening,
        compute_ring_system_delete=capabilities.compute_ring_system_delete,
        editing_process_semantics=capabilities.editing_process_semantics,
        atom_restate_action_semantics=capabilities.atom_restate_action_semantics,
        ring_restate_scorer_mode=capabilities.ring_restate_scorer_mode,
        cycle_close_action_semantics=capabilities.cycle_close_action_semantics,
        cycle_open_action_semantics=capabilities.cycle_open_action_semantics,
    )
    node, _global_state, pair = model._encode_batch(batch)
    relabeled_node, _relabeled_global, relabeled_pair = model._encode_batch(
        relabeled_batch
    )
    original = model._ring_restate_context_features(batch, node, pair)
    observed = model._ring_restate_context_features(
        relabeled_batch,
        relabeled_node,
        relabeled_pair,
    )
    assert any(
        torch.allclose(original, candidate, atol=2e-6, rtol=2e-6)
        for candidate in (observed, observed.flip(dims=(1,)))
    )
