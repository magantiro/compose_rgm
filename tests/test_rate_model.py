from __future__ import annotations

import numpy as np
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.gm.loss import rate_bregman_loss
from compose_v4.model.rate_model import WholeGraphRateModel
from compose_v4.rewrite.compiler import compile_null_to_target
from compose_v4.rewrite.fiber import enumerate_action_fiber
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace import execute_trace


def _padded(smiles: str, n_slots: int = 5):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), n_slots)


def test_model_rates_are_positive_and_complete_successor_aggregated() -> None:
    torch.manual_seed(4)
    model = WholeGraphRateModel(hidden_dim=24, message_passing_steps=2)
    state = _padded("C")
    transitions = enumerate_action_fiber(state)
    prediction = model.predict_fiber(state, 0.4, transitions=transitions)
    assert prediction.marked_rates.shape == (len(transitions),)
    assert torch.all(prediction.marked_rates > 0)
    assert torch.allclose(
        prediction.total_hazard,
        torch.stack(list(prediction.successor_rate_dict().values())).sum(),
    )


def test_successor_rates_are_invariant_to_slot_permutation() -> None:
    torch.manual_seed(9)
    model = WholeGraphRateModel(hidden_dim=24, message_passing_steps=2).eval()
    state = _padded("CCO", 6)
    permutation = np.asarray([3, 0, 5, 2, 1, 4])
    permuted = type(state)(
        atom_types=state.atom_types[permutation],
        formal_charges=state.formal_charges[permutation],
        implicit_h_counts=state.implicit_h_counts[permutation],
        bonds=state.bonds[np.ix_(permutation, permutation)],
    )
    original = model.predict_fiber(state, 0.31).successor_rate_dict()
    changed = model.predict_fiber(permuted, 0.31).successor_rate_dict()
    assert original.keys() == changed.keys()
    for key in original:
        assert torch.allclose(original[key], changed[key], atol=2e-6, rtol=2e-6)


def test_one_teacher_example_can_reduce_generator_matching_loss() -> None:
    torch.manual_seed(13)
    target = _padded("CO", 4)
    trace = compile_null_to_target(target, rng=np.random.default_rng(3))
    _, states = execute_trace(trace.source, trace.steps, return_states=True)
    state = states[0]
    teacher_key = canonical_state_key(states[1])
    transitions = enumerate_action_fiber(state)
    model = WholeGraphRateModel(hidden_dim=24, message_passing_steps=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    def loss_value():
        prediction = model.predict_fiber(state, 0.2, transitions=transitions)
        return rate_bregman_loss(
            prediction.total_hazard,
            prediction.successor_rate(teacher_key),
            torch.tensor(float(len(trace.steps))),
        )

    initial = float(loss_value().detach())
    for _ in range(80):
        optimizer.zero_grad()
        loss = loss_value()
        loss.backward()
        optimizer.step()
    final = float(loss_value().detach())
    assert final < initial - 1.0
