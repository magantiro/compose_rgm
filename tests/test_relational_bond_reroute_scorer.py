from __future__ import annotations

import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.successor_micro_overfit import (
    configure_micro_overfit_parameters,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_TO_INDEX,
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.model.relational_reroute_rate_model import (
    RelationalRerouteFactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import BondReroute
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SOURCE = "COc1cc(CNc2cc3c(cn2)[nH]c2ccccc23)cc(OC)c1"
_TEACHER_SUCCESSOR = "COc1cccc(CNc2cc3c(cn2)[nH]c2ccccc23)c1OC"
_TIED_COMPETITOR = "COc1ccc(OC)c(CNc2cc3c(cn2)[nH]c2ccccc23)c1"


def _models_and_batch():
    state = pad_molecular_graph(smiles_to_molecular_graph(_SOURCE), 40)
    batch = prepare_factorized_mark_batch(
        (state,),
        (0.5,),
        (None,),
        (None,),
        (0.0,),
        compute_cyclic_graft=True,
    )
    catalog = build_typed_ring_catalog(())
    torch.manual_seed(17)
    base = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=2,
        enable_cyclic_graft=True,
    )
    torch.manual_seed(17)
    relational = RelationalRerouteFactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=2,
        enable_cyclic_graft=True,
    )
    return state, batch, base, relational


def _group_by_key(state, batch) -> dict[str, int]:
    runtime = de_novo_rewrite_system()
    result: dict[str, int] = {}
    for moved, target in torch.nonzero(batch.graft_mask[0], as_tuple=False):
        moved_index = int(moved)
        target_index = int(target)
        removed = int(batch.graft_remove_neighbors[0, moved_index, target_index])
        successor = runtime.apply(
            state,
            "bond_reroute",
            BondReroute(
                a=moved_index,
                b=removed,
                u=moved_index,
                v=target_index,
            ),
        )
        result[canonical_state_key(successor)] = int(
            batch.graft_successor_groups[0, moved_index, target_index]
        )
    return result


def _group_score(logits, mask, groups, group: int) -> torch.Tensor:
    selected = mask & (groups == group)
    assert int(selected.sum()) == 2
    return torch.logsumexp(logits[selected], dim=0)


def test_zero_initialized_relational_residual_preserves_every_existing_logit() -> None:
    _state, batch, base, relational = _models_and_batch()
    base_state = base.state_dict()
    relational_state = relational.state_dict()
    assert set(relational_state) - set(base_state) == {"graft_relation_head.weight"}
    assert all(torch.equal(base_state[name], relational_state[name]) for name in base_state)
    assert torch.count_nonzero(relational.graft_relation_head.weight) == 0

    base_encoded = base._encode_batch(batch)
    relational_encoded = relational._encode_batch(batch)
    assert all(
        torch.equal(base_value, relational_value)
        for base_value, relational_value in zip(base_encoded, relational_encoded, strict=True)
    )
    _base_masks, base_logits, base_family = base._action_tables(
        batch, *base_encoded
    )
    _relational_masks, relational_logits, relational_family = (
        relational._action_tables(batch, *relational_encoded)
    )
    assert torch.equal(base_family, relational_family)
    assert set(base_logits) == set(relational_logits)
    assert all(
        torch.equal(base_logits[family], relational_logits[family])
        for family in base_logits
    )


def test_removed_target_relation_breaks_the_measured_symmetric_graft_tie() -> None:
    state, batch, _base, model = _models_and_batch()
    groups_by_key = _group_by_key(state, batch)
    teacher_group = groups_by_key[_TEACHER_SUCCESSOR]
    competitor_group = groups_by_key[_TIED_COMPETITOR]
    assert teacher_group != competitor_group

    node, global_state, pair = model._encode_batch(batch)
    masks, logits, action_log_z = model._action_tables(
        batch, node, global_state, pair
    )
    graft_mask = masks["bond_reroute"][0]
    group_ids = batch.graft_successor_groups[0]
    teacher_before = _group_score(
        logits["bond_reroute"][0], graft_mask, group_ids, teacher_group
    )
    competitor_before = _group_score(
        logits["bond_reroute"][0], graft_mask, group_ids, competitor_group
    )
    assert torch.equal(teacher_before, competitor_before)

    relation = model._graft_relation_features(batch, pair)[0]
    teacher_relation = relation[graft_mask & (group_ids == teacher_group)].mean(dim=0)
    competitor_relation = relation[
        graft_mask & (group_ids == competitor_group)
    ].mean(dim=0)
    difference = teacher_relation - competitor_relation
    assert float(difference.norm().detach()) > 0.0
    with torch.no_grad():
        model.graft_relation_head.weight.copy_(difference.unsqueeze(0))

    masks, logits, action_log_z = model._action_tables(
        batch, node, global_state, pair
    )
    teacher_after = _group_score(
        logits["bond_reroute"][0], masks["bond_reroute"][0], group_ids, teacher_group
    )
    competitor_after = _group_score(
        logits["bond_reroute"][0],
        masks["bond_reroute"][0],
        group_ids,
        competitor_group,
    )
    assert float((teacher_after - competitor_after).detach()) > 0.0
    expected_reroute_log_z = torch.logsumexp(
        logits["bond_reroute"][0][masks["bond_reroute"][0]],
        dim=0,
    )
    assert torch.allclose(
        action_log_z[0, MARK_RULE_TO_INDEX["bond_reroute"]],
        expected_reroute_log_z,
    )


def test_relational_parameters_are_owned_by_the_bond_reroute_head_scope() -> None:
    _state, _batch, _base, model = _models_and_batch()
    selected = configure_micro_overfit_parameters(
        model,
        ("bond_reroute",),
        scope="heads_only",
    )
    assert "graft_relation_head.weight" in selected
