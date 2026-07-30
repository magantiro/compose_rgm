from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.ring_macro_enumerability import (
    RingMacroEnumerabilityError,
    RingMacroEnumerationLimits,
    enumerate_ring_macro_action_law,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.tracelets import is_valid_ring_system_grow
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import (
    build_typed_ring_catalog_from_paths,
)


@pytest.fixture(scope="module")
def ring_macro_case():
    target = pad_molecular_graph(
        smiles_to_molecular_graph("c1ccccc1"),
        12,
    )
    source = DegreeBoundedCarbonTreePrior(
        sizes=(target.n_real_atoms,),
    ).sample(
        np.random.default_rng(409),
        n_slots=12,
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    path = TraceProgressCTMC(trace)
    catalog = build_typed_ring_catalog_from_paths((path,))
    progress = next(
        index for index, step in enumerate(trace.steps) if step.rule_name == "ring_system_grow"
    )
    return catalog, path.state_at(progress)


def _macro_only_model(catalog, *, electronic_mode: str):
    torch.manual_seed(7301)
    return FactorizedTraceletRateModel(
        catalog,
        hidden_dim=8,
        message_passing_steps=1,
        ring_electronic_mode=electronic_mode,
        enable_cycle_ops=False,
        enable_ring_grow_macro=True,
    ).eval()


def test_semantic_probe_enumerates_normalized_executable_complete_actions(
    ring_macro_case,
) -> None:
    catalog, state = ring_macro_case
    model = _macro_only_model(
        catalog,
        electronic_mode="factorized_contextual",
    )

    result = enumerate_ring_macro_action_law(model, state)

    assert result.normalization_applicable
    assert result.legal_template_count > 0
    assert result.raw_placement_count >= result.supported_placement_count > 0
    assert result.prefix_node_count > result.complete_encoding_count > 0
    assert result.distinct_complete_action_count == len(result.actions) > 0
    assert (
        result.complete_encoding_count
        >= result.distinct_complete_action_count
        >= result.resonance_invariant_action_count
    )
    assert result.encoding_total_probability == pytest.approx(1.0, abs=2e-5)
    assert result.action_total_probability == pytest.approx(1.0, abs=2e-5)
    raw_encoding_total = sum(
        math.exp(address.conditional_log_probability)
        for item in result.actions
        for address in item.encoding_addresses
    )
    aggregated_action_total = sum(item.conditional_probability for item in result.actions)
    assert raw_encoding_total == pytest.approx(
        result.encoding_total_probability,
        abs=2e-5,
    )
    assert aggregated_action_total == pytest.approx(
        result.action_total_probability,
        abs=2e-5,
    )
    assert raw_encoding_total == pytest.approx(aggregated_action_total, abs=2e-5)
    assert result.max_teacher_scorer_log_error <= 3e-5
    assert all(item.conditional_probability > 0.0 for item in result.actions)
    assert all(item.primitive_lowering_length > 0 for item in result.actions)
    assert all(is_valid_ring_system_grow(state, item.action) for item in result.actions)
    assert sum(item.encoding_count for item in result.actions) == (result.complete_encoding_count)


def test_catalog_probe_aggregates_deliberately_duplicated_encodings(
    ring_macro_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, state = ring_macro_case
    model = _macro_only_model(catalog, electronic_mode="catalog_exact")
    original = model._ring_exact_candidate_log_probabilities

    def duplicated_candidate_table(
        source_state,
        template_index,
        node,
        global_state,
        pair,
    ):
        candidates, log_probabilities = original(
            source_state,
            template_index,
            node,
            global_state,
            pair,
        )
        duplicated_candidates = tuple(candidate for candidate in candidates for _ in range(2))
        duplicated_log_probabilities = torch.repeat_interleave(
            log_probabilities - math.log(2.0),
            repeats=2,
        )
        return duplicated_candidates, duplicated_log_probabilities

    monkeypatch.setattr(
        model,
        "_ring_exact_candidate_log_probabilities",
        duplicated_candidate_table,
    )
    result = enumerate_ring_macro_action_law(model, state)

    assert result.normalization_applicable
    assert result.electronic_mode == "catalog_exact"
    assert result.complete_encoding_count > result.distinct_complete_action_count
    assert result.duplicate_encoding_count == (
        result.complete_encoding_count - result.distinct_complete_action_count
    )
    assert any(item.encoding_count >= 2 for item in result.actions)
    raw_encoding_total = sum(
        math.exp(address.conditional_log_probability)
        for item in result.actions
        for address in item.encoding_addresses
    )
    assert raw_encoding_total == pytest.approx(
        result.encoding_total_probability,
        abs=2e-5,
    )
    assert result.action_total_probability == pytest.approx(1.0, abs=2e-5)
    assert result.max_teacher_scorer_log_error <= 3e-5


def test_probe_refuses_non_macro_only_profile(ring_macro_case) -> None:
    catalog, state = ring_macro_case
    primitive_model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=8,
        message_passing_steps=1,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
    ).eval()

    with pytest.raises(
        RingMacroEnumerabilityError,
        match="macro-only profile",
    ):
        enumerate_ring_macro_action_law(primitive_model, state)


def test_probe_fails_closed_when_semantic_complete_law_is_unavailable(
    ring_macro_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, state = ring_macro_case
    model = _macro_only_model(
        catalog,
        electronic_mode="factorized_local",
    )
    original = model._ring_supported_placement_tables

    def empty_supported_placement_table(*args, **kwargs):
        placement_logits, semantic_tables, placement_mask = original(
            *args,
            **kwargs,
        )
        return (
            placement_logits,
            semantic_tables,
            torch.zeros_like(placement_mask),
        )

    monkeypatch.setattr(
        model,
        "_ring_supported_placement_tables",
        empty_supported_placement_table,
    )
    with pytest.raises(
        RingMacroEnumerabilityError,
        match="exact law unavailable",
    ):
        enumerate_ring_macro_action_law(model, state)


def test_probe_stops_at_explicit_enumeration_bound(ring_macro_case) -> None:
    catalog, state = ring_macro_case
    model = _macro_only_model(
        catalog,
        electronic_mode="factorized_local",
    )
    limits = RingMacroEnumerationLimits(max_prefix_nodes=1)

    with pytest.raises(
        RingMacroEnumerabilityError,
        match="max_prefix_nodes",
    ):
        enumerate_ring_macro_action_law(
            model,
            state,
            limits=limits,
        )


@pytest.mark.parametrize(
    "invalid_time",
    (-1e-6, 1.000001, float("-inf"), float("inf"), float("nan")),
)
def test_probe_rejects_time_outside_closed_unit_interval(
    ring_macro_case,
    invalid_time: float,
) -> None:
    catalog, state = ring_macro_case
    model = _macro_only_model(
        catalog,
        electronic_mode="factorized_local",
    )

    with pytest.raises(
        ValueError,
        match=r"time must be finite and lie in \[0, 1\]",
    ):
        enumerate_ring_macro_action_law(
            model,
            state,
            time=invalid_time,
        )
