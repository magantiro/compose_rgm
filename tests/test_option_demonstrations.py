"""Focused checks for executable labels and target-free proposal fitting."""

import copy

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.option_demonstrations import (
    DemonstrationFitConfig,
    append_ignored_context,
    batch_option_scores,
    descriptor_menu,
    fit_demonstration_actor,
    recognize_trace,
    source_weights,
)
from compose_v4.control.option_features import structural_option_features
from compose_v4.control.option_policy import (
    AdvantageWeightedOptionActor,
    conservative_option_distribution,
)
from compose_v4.control.option_selector import balanced_option_prior
from compose_v4.control.ring_program import RingSpec, construction_branches, real_slots
from compose_v4.experiments.whole_ring_plan import RingRequest, compile_ring


@pytest.mark.parametrize("topology,size", [("pendant", 6), ("fused", 5)])
def test_compound_label_uses_existing_ring_contract(topology, size):
    graph = pad_molecular_graph(smiles_to_molecular_graph("C1CCCCC1"), 48)
    spec = RingSpec(topology, size, (size, 0, 0), "nonaromatic")
    anchor, pattern = construction_branches(graph, real_slots(graph), spec)[0]
    _, receipt = compile_ring(graph, RingRequest(spec, anchor, ("C",) * spec.growth, pattern))
    labels = recognize_trace(receipt["states"], receipt["actions"])
    assert len(labels) == 1
    assert labels[0].compound and labels[0].option == spec.option
    assert labels[0].stop == spec.horizon
    # A malformed saved intermediate must never become a training example.
    corrupt = copy.deepcopy(receipt["states"])
    corrupt[1] = corrupt[0]
    with pytest.raises(ValueError, match="executor replay"):
        recognize_trace(corrupt, receipt["actions"])
    # Incomplete construction remains an ordinary grow, not a completed ring.
    partial = recognize_trace(receipt["states"][:2], receipt["actions"][:1])
    assert partial[0].option == "grow" and not partial[0].compound


def test_factored_batch_and_runtime_context_are_equivalent():
    torch.manual_seed(2)
    actor = AdvantageWeightedOptionActor(5, 3, 8)
    states, options = torch.randn(4, 5), torch.randn(7, 3)
    actual = batch_option_scores(actor, states, options)
    expected = torch.stack([actor(state, options) for state in states])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    extended = append_ignored_context(actor, 9)
    with_context = torch.cat((states, torch.randn(4, 9) * 100), dim=1)
    torch.testing.assert_close(batch_option_scores(extended, with_context, options), expected)
    actual.sum().backward()
    assert all(parameter.grad is not None for parameter in actor.parameters())


def test_fit_is_source_balanced_deterministic_and_preserves_generic():
    torch.set_num_threads(1)
    names = ("generic", "cyclize", "grow")
    base = balanced_option_prior(names)
    x, y, sources = [[-1.0], [1.0]], [1, 2], ["a", "b"]
    config = DemonstrationFitConfig(hidden=16, updates=100, batch_size=16, seed=5)
    actor, _ = fit_demonstration_actor(x, y, sources, names, base, config=config)
    second, _ = fit_demonstration_actor(x, y, sources, names, base, config=config)
    option = torch.from_numpy(np.stack([structural_option_features(name) for name in names]))
    for features, selected in zip(x, y, strict=True):
        scores = actor(torch.tensor(features), option).detach().numpy()
        q = conservative_option_distribution(names, base, scores)
        assert q.proposal[selected] > base[selected]
        assert np.all(np.asarray(q.proposal) >= 0.1 * base - 1e-12)
    for left, right in zip(actor.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert source_weights(["a", "a", "b"]) == pytest.approx([0.25, 0.25, 0.5])
    with pytest.raises(ValueError, match="invalid demonstration"):
        fit_demonstration_actor(x, [1.5, 2.0], sources, names, base, config=config)


def test_menu_is_data_independent_and_retains_broad_channels():
    names = descriptor_menu()
    assert len(names) == len(set(names))
    assert {"generic", "local", "grow", "shrink", "rebuild", "decorate"} <= set(names)
    assert RingSpec("fused", 6, (3, 2, 1), "aromatic").option in names
