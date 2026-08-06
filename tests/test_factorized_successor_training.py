"""Differentiable teacher-successor aggregation uses the production fiber."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
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
    compile_teacher_successor_fibers_support_only,
    factorized_successor_bregman_loss,
    factorized_successor_identity_loss,
    forward_teacher_successor_batch,
    rewrite_action_codec_sha256,
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
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(3), n_slots=_SLOTS
    )
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
    observed_probability = float(prediction.selected_successor_log_probability.detach().exp())
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
        result.marked_law.total_hazard * result.diagnostics.raw_productive_mass,
        abs=2e-6,
    )


def test_support_only_teacher_compiler_matches_exhaustive_partition(model, monkeypatch):
    source = _state("c1ccccc1")
    result = canonical_successor_result(model.eval(), source, 0.41)
    marks_by_key: dict[str, list] = defaultdict(list)
    states_by_key = {}
    for mark in result.marked_law.marks:
        successor = _SYSTEM.apply(source, mark.executor_rule_name, mark.action)
        key = canonical_state_key(successor)
        if key == result.marked_law.source_key:
            continue
        marks_by_key[key].append((mark, successor))
        states_by_key[key] = successor
    target_key, target_marks = max(marks_by_key.items(), key=lambda item: len(item[1]))
    target = states_by_key[target_key]
    target_digest = persistent_slot_state_sha256(target)
    teacher_mark = next(
        mark
        for mark, successor in target_marks
        if persistent_slot_state_sha256(successor) == target_digest
    )
    exhaustive = compile_state_successor_map(model, source, time=0.41)
    exhaustive_fiber = teacher_successor_fiber_from_exact_digest(exhaustive, target_digest)

    def encoder_must_not_run(*_args, **_kwargs):
        raise AssertionError("support-only compilation called the neural encoder")

    monkeypatch.setattr(model, "_encode_batch", encoder_must_not_run)
    (compiled,) = compile_teacher_successor_fibers_support_only(
        model,
        (source,),
        (target,),
        teacher_action_sha256s=(
            rewrite_action_codec_sha256(
                teacher_mark.executor_rule_name,
                teacher_mark.action,
            ),
        ),
        teacher_families=(teacher_mark.family_name,),
        times=(0.41,),
    )
    assert compiled.teacher_fiber == exhaustive_fiber
    assert compiled.raw_mark_count == len(result.marked_law.marks)
    assert compiled.decoded_mark_count < compiled.raw_mark_count


def test_alias_aggregation_changes_loss_and_gradients(model):
    """The trainer optimizes aggregate successor mass, not one chosen alias."""

    source = _state("c1ccccc1")
    result = canonical_successor_result(model.eval(), source, 0.41)
    target_successor = max(
        result.batch.successors,
        key=lambda successor: successor.alias_count,
    )
    fiber = compile_teacher_successor_fiber(
        model,
        source,
        target_successor.state,
        time=0.41,
    )
    assert len(fiber.aliases) > 1
    incomplete_single_alias = replace(fiber, aliases=(fiber.aliases[0],))
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

    def loss_and_gradients(selected_fiber):
        model.zero_grad(set_to_none=True)
        prediction = forward_teacher_successor_batch(
            model,
            batch,
            (selected_fiber,),
        )
        loss = factorized_successor_identity_loss(prediction, batch)
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return float(loss.detach()), gradients

    aggregate_loss, aggregate_gradients = loss_and_gradients(fiber)
    single_loss, single_gradients = loss_and_gradients(incomplete_single_alias)
    assert aggregate_loss < single_loss
    assert aggregate_gradients.keys() == single_gradients.keys()
    assert any(
        not torch.allclose(
            aggregate_gradients[name],
            single_gradients[name],
            atol=1e-9,
            rtol=1e-7,
        )
        for name in aggregate_gradients
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
    assert float(prediction.selected_successor_log_probability) == 0.0
    with pytest.raises(
        SuccessorTrainingError,
        match="requires at least one molecular jump",
    ):
        factorized_successor_identity_loss(prediction, batch)


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
    assert (
        compile_teacher_successor_fiber(
            model,
            source,
            target,
            time=0.31,
        )
        == derived
    )
    assert (
        compile_state_productive_support(
            model,
            source,
            time=0.31,
        )
        == compiled.state_support
    )

    with pytest.raises(
        SuccessorTrainingError,
        match="exact teacher successor is absent",
    ):
        teacher_successor_fiber_from_exact_digest(compiled, "f" * 64)
