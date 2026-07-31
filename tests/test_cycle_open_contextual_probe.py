"""Prospective cycle-opening scorer capacity probe.

The exact-bond contextual mode is diagnostic only.  It must pass a separate
alternate-Kekule canonical-successor-law invariance gate before it can become a
production architecture.
"""

from __future__ import annotations

import torch

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records
from compose_v4.experiments.successor_micro_overfit import (
    configure_micro_overfit_parameters,
    examples_from_traces,
    prepare_successor_panel,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    CYCLE_OPEN_SCORER_MODES,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_HIDDEN_DIM = 8
_PROBE_MODE = "exact_bond_contextual_probe"


def _model(*, mode: str | None = None) -> FactorizedTraceletRateModel:
    kwargs = {}
    if mode is not None:
        kwargs["cycle_open_scorer_mode"] = mode
    return FactorizedTraceletRateModel(
        build_typed_ring_catalog(()),
        hidden_dim=_HIDDEN_DIM,
        message_passing_steps=1,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
        **kwargs,
    )


def _probe_head():
    torch.manual_seed(19)
    return _model(mode=_PROBE_MODE).cycle_open_head


def _zero_probe_head(head) -> None:
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()


def test_legacy_cycle_open_state_dict_remains_strictly_compatible() -> None:
    assert CYCLE_OPEN_SCORER_MODES[0] == "pair_linear"
    assert "probe" in _PROBE_MODE

    torch.manual_seed(7)
    implicit_legacy = _model()
    torch.manual_seed(7)
    explicit_legacy = _model(mode="pair_linear")

    implicit_state = implicit_legacy.state_dict()
    explicit_state = explicit_legacy.state_dict()
    assert tuple(implicit_state) == tuple(explicit_state)
    assert all(torch.equal(implicit_state[name], explicit_state[name]) for name in implicit_state)
    explicit_legacy.load_state_dict(implicit_state, strict=True)
    assert {name for name in implicit_state if name.startswith("cycle_open_head.")} == {
        "cycle_open_head.weight",
        "cycle_open_head.bias",
    }


def test_probe_exact_bond_order_changes_score_for_same_neural_pair() -> None:
    head = _probe_head()
    _zero_probe_head(head)
    with torch.no_grad():
        head.exact_deleted_bond_order_embedding.weight[1, 0] = 1.0
        head.exact_deleted_bond_order_embedding.weight[2, 0] = -1.0
        # Input layout is pair, global, exact order, pair-by-global.
        head.scorer[0].weight[0, 2 * _HIDDEN_DIM] = 1.0
        head.scorer[2].weight[0, 0] = 1.0

    pair = torch.zeros((1, 1, 1, _HIDDEN_DIM))
    global_state = torch.zeros((1, _HIDDEN_DIM))
    single = head(pair, global_state, torch.ones((1, 1, 1), dtype=torch.long))
    double = head(
        pair,
        global_state,
        torch.full((1, 1, 1), 2, dtype=torch.long),
    )
    assert float(single.detach()) > float(double.detach())


def test_probe_pair_by_global_context_can_reverse_candidate_ranking() -> None:
    head = _probe_head()
    _zero_probe_head(head)
    with torch.no_grad():
        # Read only the explicit pair-by-global interaction.  A shared additive
        # global bias could not reverse this within-state ranking.
        head.scorer[0].weight[0, 3 * _HIDDEN_DIM] = 1.0
        head.scorer[2].weight[0, 0] = 1.0

    pair = torch.zeros((1, 1, 2, _HIDDEN_DIM))
    pair[0, 0, 0, 0] = 1.0
    pair[0, 0, 1, 0] = -1.0
    exact_orders = torch.ones((1, 1, 2), dtype=torch.long)
    positive_context = torch.zeros((1, _HIDDEN_DIM))
    positive_context[0, 0] = 1.0
    negative_context = -positive_context

    positive_scores = head(pair, positive_context, exact_orders)
    negative_scores = head(pair, negative_context, exact_orders)
    assert int(positive_scores.argmax(dim=-1)) == 0
    assert int(negative_scores.argmax(dim=-1)) == 1


def test_probe_parameters_receive_gradient_through_cycle_open_model_path() -> None:
    records, attempted = build_cycle_op_records(
        ("C1CCCCC1", "c1ccccc1", "C1CC2CCC1C2"),
        n_slots=12,
        seed=4,
        max_bonds_per_molecule=2,
    )
    assert attempted > 0
    (example,) = examples_from_traces(
        (record.path.trace for record in records),
        families=("cycle_attach",),
        maximum_per_family=1,
        unique_source_molecules=True,
    )
    model = _model(mode=_PROBE_MODE).train()
    panel = prepare_successor_panel(model, (example,))

    prediction = model.forward_mark_batch(panel.batch)
    loss = -prediction.selected_mark_log_probability.mean()
    loss.backward()

    head = model.cycle_open_head
    exact_order_gradient = head.exact_deleted_bond_order_embedding.weight.grad
    scorer_gradients = tuple(parameter.grad for parameter in head.scorer.parameters())
    assert exact_order_gradient is not None
    assert torch.isfinite(exact_order_gradient).all()
    assert float(exact_order_gradient.abs().sum()) > 0.0
    assert all(gradient is not None for gradient in scorer_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in scorer_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in scorer_gradients) > 0.0


def test_probe_parameters_remain_inside_cycle_open_capacity_head_scope() -> None:
    model = _model(mode=_PROBE_MODE)
    selected = configure_micro_overfit_parameters(
        model,
        ("cycle_attach",),
        scope="heads_only",
    )
    probe_parameters = {
        name for name, _parameter in model.named_parameters() if name.startswith("cycle_open_head.")
    }
    assert probe_parameters
    assert probe_parameters <= set(selected)


def test_probe_mode_rejects_missing_cycle_support() -> None:
    try:
        FactorizedTraceletRateModel(
            build_typed_ring_catalog(()),
            enable_cycle_ops=False,
            cycle_open_scorer_mode=_PROBE_MODE,
        )
    except ValueError as error:
        assert "require enable_cycle_ops=True" in str(error)
    else:
        raise AssertionError("probe scorer was accepted without cycle-op support")
