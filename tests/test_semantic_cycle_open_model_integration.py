"""Editing-V2 semantic cycle-open integration through model and kernel."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.cycle_open_kekule_invariance import (
    build_alternate_kekule_pair,
)
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkDataset,
    FactorizedMarkExample,
    _concatenate_factorized_mark_batches,
)
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.factorized_successor_training import (
    compile_state_successor_map,
)
from compose_v4.experiments.production_successor_kernel import (
    canonical_successor_result,
)
from compose_v4.experiments.reference_successor_kernel import (
    compare_against_reference,
    reference_successor_batch,
)
from compose_v4.experiments.quotient_invariance import permute_persistent_slots
from compose_v4.model.factorized_tracelet_rate_model import (
    LEGACY_RING_RESTATE_SCORER_MODE,
    SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
    LEGACY_CYCLE_CLOSE_ACTION_SEMANTICS,
    LEGACY_CYCLE_OPEN_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    BondDelete,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
    enumerate_cycle_close_edges,
    enumerate_cycle_open_edges,
    enumerate_semantic_atom_restates,
)
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SLOTS = 16


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), _SLOTS)


@pytest.fixture(scope="module")
def ring_catalog():
    target = _state("c1ccccc1")
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(1), n_slots=_SLOTS
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    return build_typed_ring_catalog((trace,))


@pytest.fixture(scope="module")
def semantic_model(ring_catalog):
    torch.manual_seed(11)
    return FactorizedTraceletRateModel(
        ring_catalog,
        hidden_dim=12,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_heteroatom_scan=True,
        enable_cycle_ops=True,
        editing_process_semantics=SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
        atom_restate_action_semantics=SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
        ring_restate_scorer_mode=SEMANTIC_RING_RESTATE_SCORER_MODE,
        cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
        cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()


def _batch(model, state, *, action=None, rule_name=None):
    capabilities = model.operator_capabilities
    return prepare_factorized_mark_batch(
        (state,),
        (0.37,),
        (action,),
        (rule_name,),
        (1.0 if action is not None else 0.0,),
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


def test_semantic_atom_restate_teacher_reaches_the_dataset_collator_path(
    semantic_model,
    monkeypatch,
) -> None:
    """Action V4 executor names must survive the actual sampled training path."""

    source = _state("CO")
    action = enumerate_semantic_atom_restates(source)[0]
    runtime = editing_v2_semantic_rewrite_system()
    target = runtime.apply(source, "atom_restate_semantic", action)
    trace = RewriteTrace(
        source=source,
        target=target,
        steps=(RewriteStep("atom_restate_semantic", action),),
        metadata={},
    )
    record = PathRecord(
        target_key=canonical_state_key(target),
        path=TraceProgressCTMC(trace, system=runtime),
    )
    import compose_v4.experiments.factorized_mark_conditional as conditional

    monkeypatch.setattr(
        conditional,
        "_sample_tracelet_progress",
        lambda *_args, **_kwargs: (0, 1.0),
    )
    dataset = FactorizedMarkDataset(
        (record,),
        start_index=0,
        length=1,
        seed=0,
        late_time_fraction=0.0,
        operational_horizon=1.0,
        progress_stratification_fraction=0.0,
        ring_catalog=semantic_model.ring_catalog,
    )
    example = dataset[0]
    assert example.teacher_rule_name == "atom_restate_semantic"
    capabilities = semantic_model.operator_capabilities
    batch = FactorizedMarkCollator(
        True,
        semantic_model.ring_catalog,
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
    )([example])
    prediction = semantic_model.forward_mark_batch(batch)
    assert bool(torch.isfinite(prediction.selected_mark_log_probability).all())


def test_semantic_mode_is_explicit_and_rejects_representation_dependent_scorer(
    ring_catalog,
) -> None:
    legacy = FactorizedTraceletRateModel(
        ring_catalog,
        hidden_dim=8,
        message_passing_steps=1,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
    )
    assert legacy.cycle_open_action_semantics == LEGACY_CYCLE_OPEN_ACTION_SEMANTICS
    assert legacy.cycle_close_action_semantics == LEGACY_CYCLE_CLOSE_ACTION_SEMANTICS
    with pytest.raises(ValueError, match="require enable_cycle_ops"):
        FactorizedTraceletRateModel(
            ring_catalog,
            hidden_dim=8,
            message_passing_steps=1,
            cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
            cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
        )
    with pytest.raises(ValueError, match="stored Kekule order"):
        FactorizedTraceletRateModel(
            ring_catalog,
            hidden_dim=8,
            message_passing_steps=1,
            enable_cycle_ops=True,
            enable_ring_grow_macro=False,
            cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
            cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
            cycle_open_scorer_mode="exact_bond_contextual_probe",
        )


def test_semantic_batch_mask_equals_executor_fiber(semantic_model) -> None:
    state = _state("Cc1cccc(Cl)c1")
    batch = _batch(semantic_model, state)
    expected_close = {
        (int(action.a), int(action.b), int(action.order) - 1)
        for action in enumerate_cycle_close_edges(state)
    }
    observed_close = {
        tuple(int(value) for value in coordinate)
        for coordinate in torch.nonzero(
            batch.cycle_close_admission_mask[0], as_tuple=False
        )
    }
    assert observed_close == expected_close
    expected_restates = {
        (int(action.v), int(action.target_class_index))
        for action in enumerate_semantic_atom_restates(state)
    }
    observed_restates = {
        tuple(int(value) for value in coordinate)
        for coordinate in torch.nonzero(
            batch.atom_restate_admission_mask[0],
            as_tuple=False,
        )
    }
    assert observed_restates == expected_restates
    expected = {
        (int(action.a), int(action.b)) for action in enumerate_cycle_open_edges(state)
    }
    observed = {
        tuple(int(value) for value in coordinate)
        for coordinate in torch.nonzero(
            batch.cycle_open_admission_mask[0], as_tuple=False
        )
    }
    assert observed == expected
    assert batch.cycle_open_action_semantics == SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS


def test_batch_and_model_fail_closed_across_action_semantics(semantic_model) -> None:
    state = _state("c1ccccc1")
    action = enumerate_cycle_open_edges(state)[0]
    with pytest.raises(ValueError, match="raw BondDelete"):
        prepare_factorized_mark_batch(
            (state,),
            (0.2,),
            (BondDelete(action.a, action.b),),
            ("bond_delete",),
            (1.0,),
            ring_catalog=semantic_model.ring_catalog,
            cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
            cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
        )
    legacy_batch = prepare_factorized_mark_batch(
        (state,),
        (0.2,),
        (None,),
        (None,),
        (0.0,),
        ring_catalog=semantic_model.ring_catalog,
    )
    with pytest.raises(
        ValueError, match="batch/model editing process semantics disagree"
    ):
        semantic_model.forward_mark_batch(legacy_batch)


def test_semantic_teacher_is_legal_and_has_finite_probability(semantic_model) -> None:
    state = _state("Cc1cccc(Cl)c1")
    action = enumerate_cycle_open_edges(state)[0]
    prediction = semantic_model.forward_mark_batch(
        _batch(semantic_model, state, action=action, rule_name="cycle_open")
    )
    assert torch.isfinite(prediction.selected_mark_log_probability).all()


def test_semantic_ring_restate_teacher_is_scored_at_successor_group_level(
    ring_catalog,
) -> None:
    torch.manual_seed(17)
    model = FactorizedTraceletRateModel(
        ring_catalog,
        hidden_dim=12,
        message_passing_steps=1,
        enable_ring_restates=True,
        ring_restate_scorer_mode=SEMANTIC_RING_RESTATE_SCORER_MODE,
    ).eval()
    source = _state("C1CCCCC1")
    candidates = _batch(model, source)
    assert len(candidates.ring_restate_actions[0]) == 2
    assert candidates.ring_restate_successor_group_ids == ((0, 0),)
    assert candidates.ring_restate_successor_group_multiplicities == ((2,),)
    teacher = candidates.ring_restate_actions[0][0]
    prediction = model.forward_mark_batch(
        _batch(
            model,
            source,
            action=teacher,
            rule_name="ring_system_restate",
        )
    )
    assert torch.allclose(
        prediction.selected_mark_log_probability,
        prediction.family_log_probabilities[:, 9],
        atol=1e-6,
        rtol=1e-6,
    )
    model.zero_grad(set_to_none=True)
    two_ring_source = _state("Cc1ccccc1CCc1cccc(Cl)c1")
    two_ring_candidates = _batch(model, two_ring_source)
    assert len(two_ring_candidates.ring_restate_actions[0]) == 2
    two_ring_prediction = model.forward_mark_batch(
        _batch(
            model,
            two_ring_source,
            action=two_ring_candidates.ring_restate_actions[0][0],
            rule_name="ring_system_restate",
        )
    )
    (-two_ring_prediction.selected_mark_log_probability.mean()).backward()
    transition_gradient = model.restate_transition_embedding.weight.grad
    assert transition_gradient is not None
    assert torch.isfinite(transition_gradient).all()
    assert float(transition_gradient.abs().sum()) > 0.0
    head_gradients = tuple(
        parameter.grad for parameter in model.ring_restate_head.parameters()
    )
    assert all(gradient is not None for gradient in head_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in head_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in head_gradients) > 0.0


def test_legacy_ring_restate_scorer_preserves_the_historical_state_dict(
    ring_catalog,
) -> None:
    torch.manual_seed(23)
    implicit_legacy = FactorizedTraceletRateModel(
        ring_catalog,
        hidden_dim=12,
        message_passing_steps=1,
        enable_ring_restates=True,
    )
    torch.manual_seed(23)
    explicit_legacy = FactorizedTraceletRateModel(
        ring_catalog,
        hidden_dim=12,
        message_passing_steps=1,
        enable_ring_restates=True,
        ring_restate_scorer_mode=LEGACY_RING_RESTATE_SCORER_MODE,
    )
    implicit_state = implicit_legacy.state_dict()
    explicit_state = explicit_legacy.state_dict()
    assert tuple(implicit_state) == tuple(explicit_state)
    assert all(
        torch.equal(implicit_state[name], explicit_state[name])
        for name in implicit_state
    )
    assert "restate_order_embedding.weight" in implicit_state
    assert "restate_transition_embedding.weight" not in implicit_state


def test_collator_and_worker_concatenation_preserve_cycle_process_identity(
    semantic_model,
) -> None:
    capabilities = semantic_model.operator_capabilities
    collator = FactorizedMarkCollator(
        True,
        semantic_model.ring_catalog,
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
    examples = [
        FactorizedMarkExample(
            state=_state(smiles),
            time=0.2 + 0.1 * index,
            teacher_action=None,
            teacher_rule_name=None,
            teacher_rate=0.0,
            importance_weight=1.0,
        )
        for index, smiles in enumerate(("c1ccccc1", "Cc1cccc(Cl)c1"))
    ]
    direct = collator(examples)
    merged = _concatenate_factorized_mark_batches(
        (collator(examples[:1]), collator(examples[1:]))
    )
    assert merged.cycle_close_action_semantics == direct.cycle_close_action_semantics
    assert merged.cycle_open_action_semantics == direct.cycle_open_action_semantics
    assert torch.equal(
        merged.cycle_close_admission_mask,
        direct.cycle_close_admission_mask,
    )
    assert torch.equal(
        merged.cycle_open_admission_mask,
        direct.cycle_open_admission_mask,
    )

    legacy = prepare_factorized_mark_batch(
        (examples[0].state,),
        (examples[0].time,),
        (None,),
        (None,),
        (0.0,),
        ring_catalog=semantic_model.ring_catalog,
    )
    with pytest.raises(ValueError, match="mix semantic action process identities"):
        _concatenate_factorized_mark_batches((merged.subbatch(0, 1), legacy))


def test_production_kernel_emits_only_semantic_cycle_open_marks(semantic_model) -> None:
    state = _state("Cc1cccc(Cl)c1")
    result = canonical_successor_result(semantic_model, state, 0.37)
    cycle_open_marks = tuple(
        mark for mark in result.marked_law.marks if mark.family_name == "cycle_attach"
    )
    assert cycle_open_marks
    assert all(mark.executor_rule_name == "cycle_open" for mark in cycle_open_marks)
    assert all(type(mark.action) is CycleOpenEdge for mark in cycle_open_marks)
    flags = dict(result.batch.identity.support_signature.capability_flags)
    assert flags["semantic_cycle_open"] is True
    assert flags["semantic_cycle_close"] is True
    assert (
        result.batch.identity.support_signature.ringcore_configuration
        == "editing_v2_semantic_actions_v1:ring_system_delete_disabled"
    )

    cycle_close_marks = tuple(
        mark for mark in result.marked_law.marks if mark.family_name == "cycle_insert"
    )
    assert cycle_close_marks
    assert all(mark.executor_rule_name == "cycle_close" for mark in cycle_close_marks)
    assert all(type(mark.action) is CycleCloseEdge for mark in cycle_close_marks)

    atom_restate_marks = tuple(
        mark for mark in result.marked_law.marks if mark.family_name == "atom_restate"
    )
    assert atom_restate_marks
    assert all(
        mark.executor_rule_name == "atom_restate_semantic"
        for mark in atom_restate_marks
    )
    assert all(type(mark.action) is SemanticAtomRestate for mark in atom_restate_marks)

    reference = reference_successor_batch(
        state,
        (
            (mark.executor_rule_name, mark.action, mark.probability)
            for mark in result.marked_law.marks
        ),
        system=editing_v2_semantic_rewrite_system(),
    )
    produced_probabilities = {
        successor.key: successor.probability for successor in result.batch.successors
    }
    reference_probabilities = {
        successor.key: successor.probability for successor in reference.successors
    }
    assert not compare_against_reference(
        produced_probabilities,
        reference_probabilities,
        tolerance=2e-7,
    )


def test_successor_training_compiler_uses_the_semantic_process(semantic_model) -> None:
    compiled = compile_state_successor_map(
        semantic_model,
        _state("Cc1cccc(Cl)c1"),
    )
    assert compiled.successor_groups


def test_prepared_batch_is_bound_to_the_exact_persistent_slot_state(
    semantic_model,
) -> None:
    source = _state("c1ccccc1")
    relabeled = permute_persistent_slots(
        source,
        (7, 2, 9, 0, 4, 1, 6, 3, 5, 8, 15, 10, 14, 11, 13, 12),
    )
    prepared = _batch(semantic_model, relabeled)
    with pytest.raises(
        RuntimeError,
        match="exact persistent-slot state does not match",
    ):
        canonical_successor_result(
            semantic_model,
            source,
            0.37,
            prepared_batch=prepared,
        )


def test_alternate_kekule_sources_have_same_semantic_molecular_kernel(
    semantic_model,
) -> None:
    pair = build_alternate_kekule_pair(_state("Cc1cccc(Cl)c1"))
    original = canonical_successor_result(semantic_model, pair.original, 0.37).batch
    alternate = canonical_successor_result(semantic_model, pair.alternate, 0.37).batch
    original_probabilities = {
        successor.key: successor.probability for successor in original.successors
    }
    alternate_probabilities = {
        successor.key: successor.probability for successor in alternate.successors
    }
    assert not compare_against_reference(
        original_probabilities,
        alternate_probabilities,
        tolerance=2e-7,
    )
