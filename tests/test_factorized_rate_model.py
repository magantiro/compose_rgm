from __future__ import annotations

import numpy as np
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.gm.loss import multi_successor_rate_bregman_loss
from compose_v4.model.rate_model import FactorizedRateModel
from compose_v4.rewrite.factorized_fiber import enumerate_factorized_cnof_fiber


def _padded(smiles: str, n_slots: int = 8):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), n_slots)


def test_hierarchical_rates_sum_exactly_to_total_hazard() -> None:
    torch.manual_seed(21)
    state = _padded("CCO")
    fiber = enumerate_factorized_cnof_fiber(state)
    model = FactorizedRateModel(hidden_dim=24, message_passing_steps=2)
    prediction = model.predict_factorized_fiber(state, 0.4, fiber=fiber)
    assert torch.all(prediction.marked_rates > 0)
    successor_total = torch.stack(
        list(prediction.successor_rate_dict().values())
    ).sum()
    assert torch.allclose(prediction.total_hazard, successor_total, atol=1e-6)


def test_factorized_successor_rates_are_slot_permutation_invariant() -> None:
    torch.manual_seed(22)
    state = _padded("C1COC1")
    permutation = np.asarray([5, 2, 7, 0, 4, 1, 6, 3])
    permuted = type(state)(
        atom_types=state.atom_types[permutation],
        formal_charges=state.formal_charges[permutation],
        implicit_h_counts=state.implicit_h_counts[permutation],
        bonds=state.bonds[np.ix_(permutation, permutation)],
    )
    model = FactorizedRateModel(
        hidden_dim=24,
        message_passing_steps=2,
        use_rewrite_context=True,
    ).eval()
    original = model.predict_factorized_fiber(state, 0.3).successor_rate_dict()
    changed = model.predict_factorized_fiber(permuted, 0.3).successor_rate_dict()
    assert original.keys() == changed.keys()
    for key in original:
        assert torch.allclose(original[key], changed[key], atol=2e-6, rtol=2e-6)


def test_topology_context_preserves_slot_permutation_invariance() -> None:
    torch.manual_seed(220)
    state = _padded("C1CCC2CCCCC2C1", 10)
    permutation = np.asarray([7, 2, 5, 0, 9, 1, 6, 3, 4, 8])
    permuted = type(state)(
        atom_types=state.atom_types[permutation],
        formal_charges=state.formal_charges[permutation],
        implicit_h_counts=state.implicit_h_counts[permutation],
        bonds=state.bonds[np.ix_(permutation, permutation)],
    )
    model = FactorizedRateModel(
        hidden_dim=24,
        message_passing_steps=2,
        use_topology_context=True,
    ).eval()
    original = model.predict_factorized_fiber(state, 0.3).successor_rate_dict()
    changed = model.predict_factorized_fiber(permuted, 0.3).successor_rate_dict()
    assert original.keys() == changed.keys()
    for key in original:
        assert torch.allclose(original[key], changed[key], atol=3e-6, rtol=3e-6)


def test_factorized_model_overfits_a_complete_successor_vector() -> None:
    torch.manual_seed(23)
    state = _padded("CO", 5)
    fiber = enumerate_factorized_cnof_fiber(state)
    keys = sorted({item.successor_key for item in fiber.transitions})[:3]
    teacher = torch.tensor([1.5, 0.7, 0.3])
    model = FactorizedRateModel(hidden_dim=24, message_passing_steps=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    def objective():
        prediction = model.predict_factorized_fiber(state, 0.45, fiber=fiber)
        selected = torch.stack([prediction.successor_rate(key) for key in keys])
        return multi_successor_rate_bregman_loss(
            prediction.total_hazard,
            selected,
            teacher,
        )

    initial = float(objective().detach())
    for _ in range(100):
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        optimizer.step()
    final = float(objective().detach())
    assert final < initial - 1.0
