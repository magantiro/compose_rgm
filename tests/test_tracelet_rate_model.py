from __future__ import annotations

import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import empty_molecular_graph, pad_molecular_graph
from compose_v4.model.tracelet_rate_model import (
    TRACELET_RULE_NAMES,
    TraceletRateModel,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.tracelet_compiler import compile_null_to_target_tracelets
from compose_v4.rewrite.tracelet_fiber import (
    TraceletFiber,
    enumerate_tracelet_cnof_fiber,
)
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog


def test_tracelet_rate_model_normalizes_hazard_and_enabled_families() -> None:
    torch.manual_seed(7)
    state = empty_molecular_graph(12)
    fiber = enumerate_tracelet_cnof_fiber(state)
    model = TraceletRateModel(hidden_dim=16, message_passing_steps=1)
    prediction = model.predict_tracelet_fiber(state, 0.25, fiber=fiber)

    assert prediction.marked_rates.shape == (len(fiber.transitions),)
    assert torch.isfinite(prediction.marked_rates).all()
    assert bool((prediction.marked_rates > 0).all())
    assert torch.allclose(prediction.marked_rates.sum(), prediction.total_hazard)
    assert set(fiber.enabled_families) == {"atom_insert", "cycle_insert"}
    assert len(TRACELET_RULE_NAMES) == 10


def test_transport_rate_model_scores_atomic_bridge_reroutes() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CC(C)C"), 8)
    fiber = enumerate_tracelet_cnof_fiber(state, allow_bond_reroute=True)
    assert fiber.by_family["bond_reroute"]
    model = TraceletRateModel(
        hidden_dim=16,
        message_passing_steps=1,
        enable_bond_reroute=True,
    )
    prediction = model.predict_tracelet_fiber(state, 0.5, fiber=fiber)
    reroute_indices = [
        index
        for index, transition in enumerate(prediction.transitions)
        if transition.rule_name == "bond_reroute"
    ]
    assert reroute_indices
    assert bool((prediction.marked_rates[reroute_indices] > 0).all())


def test_batched_ring_action_encoding_matches_scalar_reference() -> None:
    targets = tuple(
        pad_molecular_graph(smiles_to_molecular_graph(text), 16)
        for text in (
            "c1ccc2ccccc2c1",
            "c1ccccc1-c2ccccc2",
        )
    )
    traces = tuple(
        compile_null_to_target_tracelets(target, typed_ring_payloads=True)
        for target in targets
    )
    catalog = build_typed_ring_catalog(
        traces,
        max_cycle_templates=16,
        max_attach_templates=16,
        max_ear_templates=16,
    )
    path = TraceProgressCTMC(traces[0])
    state = path.states[1]
    fiber = enumerate_tracelet_cnof_fiber(state, ring_catalog=catalog)
    model = TraceletRateModel(
        hidden_dim=16,
        message_passing_steps=1,
        ring_catalog=catalog,
    )
    node_states, global_state, time_state, bonds = model._encode_graph(state, 0.4)
    topology = model._topology_features(state)
    batched = torch.stack(
        model._encode_action_rows(
            fiber.transitions,
            state,
            node_states,
            global_state,
            time_state,
            bonds,
            topology,
        )
    )
    scalar = torch.stack(
        [
            model._encode_action(
                transition,
                state,
                node_states,
                global_state,
                time_state,
                bonds,
                topology,
            )
            for transition in fiber.transitions
        ]
    )
    assert torch.allclose(batched, scalar, atol=1e-6)


def test_tracelet_rate_model_scores_each_compiled_macro_teacher() -> None:
    target = pad_molecular_graph(
        smiles_to_molecular_graph("c1ccc2ccccc2c1"),
        16,
    )
    runtime = de_novo_rewrite_system()
    trace = compile_null_to_target_tracelets(target, system=runtime)
    model = TraceletRateModel(hidden_dim=16, message_passing_steps=1)
    state = trace.source

    for step in trace.steps:
        successor = runtime.apply(state, step.rule_name, step.action)
        fiber = enumerate_tracelet_cnof_fiber(state)
        prediction = model.predict_tracelet_fiber(state, 0.5, fiber=fiber)
        assert float(
            prediction.successor_rate(canonical_state_key(successor)).detach()
        ) > 0.0
        state = successor


def test_quotient_energy_collapses_resonance_aliases_before_normalization() -> None:
    torch.manual_seed(11)
    target = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 12)
    state = TraceProgressCTMC(compile_null_to_target_tracelets(target)).states[-2]
    fiber = enumerate_tracelet_cnof_fiber(state)
    restates = fiber.by_family["ring_system_restate"]
    successor_groups: dict[str, list] = {}
    for transition in restates:
        successor_groups.setdefault(transition.successor_key, []).append(transition)
    assert sorted(len(group) for group in successor_groups.values()) == [2]

    model = TraceletRateModel(
        hidden_dim=16,
        message_passing_steps=1,
        rate_factorization="quotient_energy",
    )
    prediction = model.predict_tracelet_fiber(state, 0.8, fiber=fiber)
    original_rates = prediction.successor_rate_dict()

    duplicated = tuple(
        transition
        for family in TRACELET_RULE_NAMES
        for transition in fiber.by_family.get(family, ())
    )
    alias = restates[0]
    by_family = {
        family: tuple(fiber.by_family.get(family, ()))
        for family in TRACELET_RULE_NAMES
    }
    by_family["ring_system_restate"] = (*restates, alias)
    duplicated_fiber = TraceletFiber(by_family=by_family)
    assert len(duplicated_fiber.transitions) == len(duplicated) + 1
    duplicated_prediction = model.predict_tracelet_fiber(
        state,
        0.8,
        fiber=duplicated_fiber,
    )
    duplicated_rates = duplicated_prediction.successor_rate_dict()

    assert original_rates.keys() == duplicated_rates.keys()
    for key in original_rates:
        assert torch.allclose(original_rates[key], duplicated_rates[key], atol=1e-6)
