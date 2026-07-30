"""Differentiable teacher-successor aggregation uses the production fiber."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import fields

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.factorized_successor_training import (
    CanonicalSuccessorAliasGroup,
    CompiledStateSuccessorMap,
    CompiledSuccessorMark,
    SuccessorTrainingError,
    compile_state_productive_support,
    compile_state_successor_map,
    compile_teacher_successor_fiber,
    factorized_successor_bregman_loss,
    factorized_successor_identity_loss,
    forward_teacher_successor_batch,
    teacher_successor_fiber_from_compiled_state,
    teacher_successor_fiber_from_exact_digest,
)
from compose_v4.experiments.production_successor_kernel import (
    canonical_successor_result,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SLOTS = 12
_SYSTEM = de_novo_rewrite_system()


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), _SLOTS)


def _model(catalog, *, seed: int) -> FactorizedTraceletRateModel:
    torch.manual_seed(seed)
    return FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=2,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    )


@pytest.fixture
def model():
    target = _state("c1ccccc1")
    source = DegreeBoundedCarbonTreePrior(
        sizes=(target.n_real_atoms,)
    ).sample(np.random.default_rng(3), n_slots=_SLOTS)
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((trace,))
    return _model(catalog, seed=17)


def test_differentiable_fiber_probability_matches_production_mark_sum(model):
    source = _state("c1ccccc1")
    result = canonical_successor_result(model.eval(), source, 0.41)
    by_key: dict[str, list] = defaultdict(list)
    state_by_key = {}
    for mark in result.marked_law.marks:
        successor = _SYSTEM.apply(
            source,
            mark.executor_rule_name,
            mark.action,
        )
        key = canonical_state_key(successor)
        if key == result.batch.source_key:
            continue
        by_key[key].append(mark)
        state_by_key[key] = successor
    target_key, target_marks = max(
        by_key.items(),
        key=lambda item: len(item[1]),
    )
    assert len(target_marks) > 1
    target = state_by_key[target_key]
    expected_probability = sum(mark.probability for mark in target_marks)

    fiber = compile_teacher_successor_fiber(
        model,
        source,
        target,
        time=0.41,
    )
    assert len(fiber.aliases) == len(target_marks)
    batch = prepare_factorized_mark_batch(
        (source,),
        (0.41,),
        (None,),
        (None,),
        (1.0,),
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=False,
        compute_ring_restates=True,
        compute_cyclic_graft=True,
        compute_ring_opening=True,
    )
    prediction = forward_teacher_successor_batch(model, batch, (fiber,))
    observed_probability = float(
        prediction.selected_successor_log_probability.detach().exp()
    )
    assert observed_probability == pytest.approx(expected_probability, abs=2e-6)
    assert float(prediction.productive_log_probability.detach().exp()) == pytest.approx(
        result.diagnostics.raw_productive_mass,
        abs=2e-6,
    )
    assert float(
        prediction.selected_productive_successor_log_probability.detach().exp()
    ) == pytest.approx(
        result.batch.probability_of(target_key),
        abs=2e-6,
    )
    assert float(prediction.productive_hazard.detach()) == pytest.approx(
        result.marked_law.total_hazard
        * result.diagnostics.raw_productive_mass,
        abs=2e-6,
    )


def test_successor_bregman_path_backpropagates(model):
    source = _state("C1CCCCC1")
    result = canonical_successor_result(model.eval(), source, 0.37)
    target_successor = result.batch.successors[0]
    fiber = compile_teacher_successor_fiber(
        model,
        source,
        target_successor.state,
        time=0.37,
    )
    batch = prepare_factorized_mark_batch(
        (source,),
        (0.37,),
        (None,),
        (None,),
        (1.0,),
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=False,
        compute_ring_restates=True,
        compute_cyclic_graft=True,
        compute_ring_opening=True,
    )

    model.train()
    model.zero_grad(set_to_none=True)
    prediction = forward_teacher_successor_batch(model, batch, (fiber,))
    loss = factorized_successor_bregman_loss(prediction, batch)
    identity_loss = factorized_successor_identity_loss(prediction, batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(identity_loss)
    loss.backward()
    assert model.family_head[0].weight.grad is not None
    assert torch.isfinite(model.family_head[0].weight.grad).all()
    assert float(model.family_head[0].weight.grad.abs().sum()) > 0.0


def test_terminal_row_requires_and_accepts_productive_state_support(model):
    source = _state("C1CCCCC1")
    support = compile_state_productive_support(model, source, time=0.23)
    batch = prepare_factorized_mark_batch(
        (source,),
        (0.23,),
        (None,),
        (None,),
        (0.0,),
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=False,
        compute_ring_restates=True,
        compute_cyclic_graft=True,
        compute_ring_opening=True,
    )

    prediction = forward_teacher_successor_batch(
        model,
        batch,
        (None,),
        state_supports=(support,),
    )
    result = canonical_successor_result(model.eval(), source, 0.23)
    assert float(prediction.productive_log_probability.detach().exp()) == pytest.approx(
        result.diagnostics.raw_productive_mass,
        abs=2e-6,
    )
    assert torch.isfinite(factorized_successor_bregman_loss(prediction, batch))


def test_hot_path_checks_exact_slots_without_quotienting_atom_relabeling(model):
    source = _state("CO")
    support = compile_state_productive_support(model, source, time=0.23)
    permutation = np.asarray([1, 0, *range(2, _SLOTS)])
    permuted = type(source)(
        atom_types=source.atom_types[permutation],
        formal_charges=source.formal_charges[permutation],
        implicit_h_counts=source.implicit_h_counts[permutation],
        bonds=source.bonds[np.ix_(permutation, permutation)],
    )
    assert canonical_state_key(permuted) == canonical_state_key(source)
    batch = prepare_factorized_mark_batch(
        (permuted,),
        (0.23,),
        (None,),
        (None,),
        (0.0,),
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=False,
        compute_ring_restates=True,
        compute_cyclic_graft=True,
        compute_ring_opening=True,
    )

    with pytest.raises(SuccessorTrainingError, match="exact terminal batch state"):
        forward_teacher_successor_batch(
            model,
            batch,
            (None,),
            state_supports=(support,),
        )


def test_compiled_support_is_invariant_to_model_weights_and_time(model):
    """The cache is a support derivative, never a checkpoint-dependent score."""

    source = _state("c1ccccc1")
    result = canonical_successor_result(model.eval(), source, 0.17)
    target = max(
        result.batch.successors,
        key=lambda successor: successor.alias_count,
    ).state

    independently_initialized = _model(model.ring_catalog, seed=991).eval()
    early_map = compile_state_successor_map(
        model,
        source,
        time=0.03,
    )
    late_map = compile_state_successor_map(
        independently_initialized,
        source,
        time=0.97,
    )
    assert early_map == late_map
    early = teacher_successor_fiber_from_compiled_state(early_map, target)
    late = teacher_successor_fiber_from_compiled_state(late_map, target)
    assert early == late
    assert early_map.state_support == late_map.state_support

    forbidden_fragments = ("time", "prob", "logit", "hazard", "rate", "weight")
    static_field_names = tuple(
        field.name
        for record_type in (
            CompiledStateSuccessorMap,
            CanonicalSuccessorAliasGroup,
            CompiledSuccessorMark,
        )
        for field in fields(record_type)
    )
    assert not any(
        fragment in field_name
        for field_name in static_field_names
        for fragment in forbidden_fragments
    )
    assert all(
        len(mark.action_sha256) == 64
        for group in early_map.successor_groups
        for mark in group.marks
    )


def test_state_compiler_derivatives_preserve_public_fiber_behavior(model):
    source = _state("C1CCCCC1")
    result = canonical_successor_result(model.eval(), source, 0.31)
    target = max(
        result.batch.successors,
        key=lambda successor: successor.alias_count,
    ).state

    compiled = compile_state_successor_map(model, source, time=0.31)
    derived = teacher_successor_fiber_from_compiled_state(compiled, target)
    strict = teacher_successor_fiber_from_exact_digest(
        compiled,
        derived.target_state_sha256,
    )

    assert strict == derived
    assert compile_teacher_successor_fiber(
        model,
        source,
        target,
        time=0.31,
    ) == derived
    assert compile_state_productive_support(
        model,
        source,
        time=0.31,
    ) == compiled.state_support

    with pytest.raises(
        SuccessorTrainingError,
        match="exact teacher successor is absent",
    ):
        teacher_successor_fiber_from_exact_digest(compiled, "f" * 64)
