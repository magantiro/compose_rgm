from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
from compose_v4.chem.graph_primitives import compute_topology_features
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.experiments.factorized_mark_conditional import (
    _concatenate_factorized_mark_batches,
    FactorizedMarkCollator,
    FactorizedMarkDataset,
    FactorizedMarkExample,
    cosine_warmup_learning_rate,
    factorized_adamw_parameter_groups,
    factorized_mark_metrics,
    sample_factorized_mark_batch,
    train_factorized_mark_model,
)
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.tracelet_conditional import build_tracelet_path_records
from compose_v4.experiments.training_support_compiler import (
    attach_ring_teacher_semantic_certificates,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    MARK_RULE_TO_INDEX,
    SparseBinaryRows,
    _graph_application_masks,
    factorized_mark_bregman_loss,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import BondReroute
from compose_v4.rewrite.typed_ring_catalog import (
    build_typed_ring_catalog,
    build_typed_ring_catalog_from_paths,
)
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.ring_system_fiber import (
    matching_ring_system_template_indices,
    ring_system_grow_electronic_key,
    ring_system_placement,
    ring_system_placement_key,
    semantic_ring_categories_for_action,
    semantic_ring_next_category_mask,
    semantic_ring_prefix_is_completable,
    structured_ring_trace_supported,
)
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.tracelets import is_valid_ring_system_grow


def _catalog_and_records(smiles: tuple[str, ...]):
    records = build_tracelet_path_records(
        smiles,
        n_slots=12,
        typed_ring_payloads=True,
    )
    catalog = build_typed_ring_catalog(
        (record.path.trace for record in records),
        max_cycle_templates=32,
        max_attach_templates=32,
        max_ear_templates=32,
    )
    return catalog, records


def _ring_support_case():
    target = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 12)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
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
        index
        for index, step in enumerate(trace.steps)
        if step.rule_name == "ring_system_grow"
    )
    return catalog, path.state_at(progress), trace.steps[progress].action


def test_ring_template_support_avoids_witness_search_after_semantic_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, state, action = _ring_support_case()
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=8,
        message_passing_steps=1,
    )
    template_index = matching_ring_system_template_indices(
        action,
        model.ring_system_templates,
    )[0]
    assert model.ring_system_electronic_witness_aliases is not None

    monkeypatch.setattr(
        model,
        "_ring_witness_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("semantic proof should avoid executor witness search")
        ),
    )
    monkeypatch.setattr(
        "compose_v4.model.factorized_tracelet_rate_model."
        "semantic_ring_prefix_is_completable",
        lambda *_args, **_kwargs: True,
    )

    support = model._ring_grow_support(state)
    assert support[template_index]


def test_ring_template_support_falls_back_when_catalog_has_no_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, state, action = _ring_support_case()
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=8,
        message_passing_steps=1,
    )
    template_index = matching_ring_system_template_indices(
        action,
        model.ring_system_templates,
    )[0]
    semantic_calls = 0

    monkeypatch.setattr(
        model,
        "_ring_witness_candidates",
        lambda current_state, current_index: (),
    )

    def accept_semantic_fallback(*_args, **_kwargs):
        nonlocal semantic_calls
        semantic_calls += 1
        return True

    monkeypatch.setattr(
        "compose_v4.model.factorized_tracelet_rate_model."
        "semantic_ring_prefix_is_completable",
        accept_semantic_fallback,
    )

    support = model._ring_grow_support(state)
    assert support[template_index]
    assert semantic_calls > 0


def test_legacy_ring_catalog_retains_complete_semantic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, state, action = _ring_support_case()
    legacy_catalog = replace(
        catalog,
        ring_system_electronic_alias_version=0,
        ring_system_electronic_aliases=(),
        ring_system_electronic_alias_counts=(),
    )
    model = FactorizedTraceletRateModel(
        legacy_catalog,
        hidden_dim=8,
        message_passing_steps=1,
    )
    template_index = matching_ring_system_template_indices(
        action,
        model.ring_system_templates,
    )[0]
    semantic_calls = 0

    def accept_semantic_fallback(*_args, **_kwargs):
        nonlocal semantic_calls
        semantic_calls += 1
        return True

    monkeypatch.setattr(
        "compose_v4.model.factorized_tracelet_rate_model."
        "semantic_ring_prefix_is_completable",
        accept_semantic_fallback,
    )

    support = model._ring_grow_support(state)
    assert model.ring_system_electronic_witness_aliases is None
    assert support[template_index]
    assert semantic_calls > 0


def test_ring_enablement_certificate_matches_complete_support() -> None:
    catalog, state, _ = _ring_support_case()
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=8,
        message_passing_steps=1,
    )

    certificate = model._ring_grow_enablement_certificate(state)
    complete = model._ring_grow_support(state)

    assert sum(certificate) <= 1
    assert any(certificate) == any(complete)
    assert all(not selected or complete[index] for index, selected in enumerate(certificate))


def test_objective_aware_ring_certificate_preserves_nonring_loss_and_gradients() -> None:
    paths = []
    for seed, smiles in enumerate(
        ("c1ccccc1", "C1CCCCC1", "c1ncnnc1", "C1CCC2CCCCC2C1"),
        start=100,
    ):
        target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)
        source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
            np.random.default_rng(seed),
            n_slots=16,
        )
        paths.append(
            TraceProgressCTMC(
                compile_carbon_tree_to_target(
                    source,
                    target,
                    use_bond_reroute=True,
                    align_source=True,
                )
            )
        )
    catalog = build_typed_ring_catalog_from_paths(tuple(paths))
    oracle = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
    )
    selected = None
    for path in paths:
        for progress, step in enumerate(path.trace.steps):
            if step.rule_name == "ring_system_grow":
                continue
            state = path.state_at(progress)
            complete = oracle._ring_grow_support(state)
            if sum(complete) > 1:
                selected = (path, progress, step, state, complete)
                break
        if selected is not None:
            break
    assert selected is not None
    path, progress, step, state, complete = selected
    certificate = oracle._ring_grow_enablement_certificate(state)
    assert sum(certificate) == 1 < sum(complete)

    base = prepare_factorized_mark_batch(
        (state,),
        (0.5,),
        (step.action,),
        (step.rule_name,),
        (path.operational_jump_rate(progress),),
    )
    complete_batch = replace(
        base,
        ring_grow_support_mask=torch.tensor((complete,), dtype=torch.bool),
        ring_grow_support_is_exact=torch.tensor((True,)),
        ring_grow_enablement_is_exact=torch.tensor((True,)),
    )
    certificate_batch = replace(
        base,
        ring_grow_support_mask=torch.tensor((certificate,), dtype=torch.bool),
        ring_grow_support_is_exact=torch.tensor((False,)),
        ring_grow_enablement_is_exact=torch.tensor((True,)),
    )
    complete_model = copy.deepcopy(oracle)
    certificate_model = copy.deepcopy(oracle)

    complete_prediction = complete_model.forward_mark_batch(complete_batch)
    certificate_prediction = certificate_model.forward_mark_batch(certificate_batch)
    complete_loss = factorized_mark_bregman_loss(complete_prediction, complete_batch)
    certificate_loss = factorized_mark_bregman_loss(
        certificate_prediction,
        certificate_batch,
    )
    complete_loss.backward()
    certificate_loss.backward()

    assert torch.equal(
        complete_prediction.enabled_families,
        certificate_prediction.enabled_families,
    )
    assert torch.allclose(
        complete_prediction.family_log_probabilities,
        certificate_prediction.family_log_probabilities,
    )
    assert torch.allclose(
        complete_prediction.selected_mark_log_probability,
        certificate_prediction.selected_mark_log_probability,
    )
    assert torch.allclose(complete_loss, certificate_loss)
    for (complete_name, complete_parameter), (certificate_name, certificate_parameter) in zip(
        complete_model.named_parameters(),
        certificate_model.named_parameters(),
    ):
        assert complete_name == certificate_name
        if complete_parameter.grad is None or certificate_parameter.grad is None:
            assert complete_parameter.grad is certificate_parameter.grad
        else:
            assert torch.allclose(complete_parameter.grad, certificate_parameter.grad)


def test_factorized_dataset_uses_full_support_only_for_ring_teacher_rows() -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("c1ncnnc1"), 12)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(977),
        n_slots=12,
    )
    path = TraceProgressCTMC(
        compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            align_source=True,
        )
    )
    catalog = build_typed_ring_catalog_from_paths((path,))
    dataset = FactorizedMarkDataset(
        (PathRecord("ring-target", path),),
        start_index=0,
        length=128,
        seed=29,
        late_time_fraction=0.5,
        operational_horizon=2.0,
        progress_stratification_fraction=0.5,
        ring_catalog=catalog,
    )
    observed_ring = False
    observed_other = False
    for index in range(len(dataset)):
        example = dataset[index]
        assert example.ring_grow_enablement_is_exact
        if example.teacher_rule_name == "ring_system_grow":
            observed_ring = True
            assert example.ring_grow_support_is_exact
        else:
            observed_other = True
            assert not example.ring_grow_support_is_exact
            assert example.ring_grow_support_mask is not None
            assert sum(example.ring_grow_support_mask) <= 1
        if observed_ring and observed_other:
            break
    assert observed_ring and observed_other


def test_factorized_mark_forward_and_backward_without_successor_fiber() -> None:
    catalog, records = _catalog_and_records(("CCO", "c1ccccc1"))
    examples = []
    for record in records:
        progress = 0
        step = record.path.trace.steps[progress]
        examples.append(
            (
                record.path.states[progress],
                0.4,
                step.action,
                step.rule_name,
                record.path.operational_jump_rate(progress),
            )
        )
    batch = prepare_factorized_mark_batch(
        tuple(row[0] for row in examples),
        tuple(row[1] for row in examples),
        tuple(row[2] for row in examples),
        tuple(row[3] for row in examples),
        tuple(row[4] for row in examples),
    )
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=32,
        message_passing_steps=1,
    )
    prediction = model.forward_mark_batch(batch)
    loss = factorized_mark_bregman_loss(prediction, batch)
    loss.backward()

    assert prediction.total_hazard.shape == (2,)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_factorized_batch_concatenation_retains_graft_successor_groups() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CCCCCCCC"), 12)
    source = prepare_factorized_mark_batch(
        (state,),
        (0.5,),
        (None,),
        (None,),
        (0.0,),
    )
    merged = _concatenate_factorized_mark_batches((source, source))
    assert merged.batch_size == 2
    assert torch.equal(merged.graft_successor_groups[0], source.graft_successor_groups[0])
    assert torch.equal(merged.graft_successor_groups[1], source.graft_successor_groups[0])


def test_factorized_adamw_excludes_lookup_and_scale_parameters_from_decay() -> None:
    catalog, _ = _catalog_and_records(("CCO", "c1ccccc1"))
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
    )
    groups = factorized_adamw_parameter_groups(model, weight_decay=0.1)
    decayed_ids = {id(parameter) for parameter in groups[0]["params"]}
    no_decay_ids = {id(parameter) for parameter in groups[1]["params"]}

    assert id(model.atom_embedding.weight) in no_decay_ids
    assert id(model.cycle_key.weight) in no_decay_ids
    assert id(model.update_norms[0].weight) in no_decay_ids
    assert id(model.total_hazard_head[-1].bias) in no_decay_ids
    assert id(model.total_hazard_head[-1].weight) in decayed_ids
    assert decayed_ids.isdisjoint(no_decay_ids)
    assert decayed_ids | no_decay_ids == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }

    lookup_before = model.cycle_key.weight.detach().clone()
    dense_before = model.total_hazard_head[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW(groups, lr=0.01)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()

    assert torch.equal(model.cycle_key.weight, lookup_before)
    assert not torch.equal(model.total_hazard_head[-1].weight, dense_before)


def test_cosine_warmup_learning_rate_has_exact_boundaries() -> None:
    common = {
        "base_learning_rate": 1e-3,
        "total_steps": 100,
        "warmup_steps": 10,
        "minimum_fraction": 0.05,
    }
    first = cosine_warmup_learning_rate(completed_step=1, **common)
    peak = cosine_warmup_learning_rate(completed_step=10, **common)
    middle = cosine_warmup_learning_rate(completed_step=55, **common)
    final = cosine_warmup_learning_rate(completed_step=100, **common)

    assert first == 1e-4
    assert peak == 1e-3
    assert final == 5e-5
    assert peak > middle > final


def test_factorized_dataset_is_index_deterministic_and_resume_stable() -> None:
    _, records = _catalog_and_records(("CCO", "CCN", "c1ccccc1"))
    common = {
        "records": records,
        "seed": 19,
        "late_time_fraction": 0.5,
        "operational_horizon": 2.0,
        "progress_stratification_fraction": 0.5,
    }
    complete = FactorizedMarkDataset(
        start_index=0,
        length=16,
        **common,
    )
    resumed = FactorizedMarkDataset(
        start_index=8,
        length=8,
        **common,
    )

    for local_index in range(8):
        expected = complete[8 + local_index]
        actual = resumed[local_index]
        assert expected.time == actual.time
        assert expected.teacher_rule_name == actual.teacher_rule_name
        assert expected.teacher_action == actual.teacher_action
        assert expected.teacher_rate == actual.teacher_rate
        assert expected.state == actual.state


def test_streamed_factorized_metrics_match_full_batch() -> None:
    catalog, records = _catalog_and_records(("CCO", "CCN", "c1ccccc1"))
    batch = sample_factorized_mark_batch(
        records,
        batch_size=6,
        seed=73,
        late_time_fraction=0.5,
        operational_horizon=2.0,
        progress_stratification_fraction=0.5,
        ring_catalog=catalog,
    )
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=8,
        message_passing_steps=1,
    )
    model.eval()

    full = factorized_mark_metrics(model, batch, use_bf16=False)
    streamed = factorized_mark_metrics(
        model,
        batch,
        use_bf16=False,
        microbatch_size=2,
    )

    assert streamed == pytest.approx(full, rel=1e-6, abs=1e-7)


def test_factorized_training_resume_preserves_validation_and_patience_trajectory() -> None:
    catalog, records = _catalog_and_records(("CCO", "CCN"))
    validation = sample_factorized_mark_batch(
        records,
        batch_size=2,
        seed=71,
        late_time_fraction=0.5,
        operational_horizon=2.0,
        ring_catalog=catalog,
    )
    torch.manual_seed(99)
    template = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=8,
        message_passing_steps=1,
    )
    initial_state = copy.deepcopy(template.state_dict())
    common = {
        "train_records": records,
        "validation_batch": validation,
        "steps": 4,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "seed": 17,
        "workers": 0,
        "late_time_fraction": 0.5,
        "operational_horizon": 2.0,
        "progress_stratification_fraction": 0.5,
        "use_aromatic_bond_view": True,
        "use_bf16": False,
        "ring_catalog": catalog,
        "evaluation_interval": 2,
        "warmup_steps": 2,
        "minimum_learning_rate_fraction": 0.5,
        "early_stopping_patience": 3,
        "early_stopping_min_relative_delta": 0.001,
    }

    uninterrupted = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=8,
        message_passing_steps=1,
    )
    uninterrupted.load_state_dict(initial_state)
    torch.manual_seed(1234)
    full_history, full_best = train_factorized_mark_model(
        uninterrupted,
        **common,
    )
    full_rng_state = torch.get_rng_state().clone()

    captured: dict[str, object] = {}

    class ExpectedInterruption(RuntimeError):
        pass

    def interrupt_after_two_steps(state: dict[str, object]) -> None:
        captured.update(state)
        raise ExpectedInterruption

    interrupted = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=8,
        message_passing_steps=1,
    )
    interrupted.load_state_dict(initial_state)
    torch.manual_seed(1234)
    with pytest.raises(ExpectedInterruption):
        train_factorized_mark_model(
            interrupted,
            checkpoint_interval=2,
            checkpoint_callback=interrupt_after_two_steps,
            **common,
        )

    resumed = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=8,
        message_passing_steps=1,
    )
    resumed_history, resumed_best = train_factorized_mark_model(
        resumed,
        resume_state=captured,
        **common,
    )

    assert resumed_history == full_history
    assert resumed_best == full_best
    assert torch.equal(torch.get_rng_state(), full_rng_state)
    for name, value in uninterrupted.state_dict().items():
        assert torch.equal(resumed.state_dict()[name], value)

    terminal_recovery = copy.deepcopy(captured)
    terminal_recovery["evaluations_without_improvement"] = common[
        "early_stopping_patience"
    ]
    terminal = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=8,
        message_passing_steps=1,
    )
    terminal_history, terminal_best = train_factorized_mark_model(
        terminal,
        resume_state=terminal_recovery,
        **common,
    )

    assert terminal_history == terminal_recovery["history"]
    assert terminal_best == terminal_recovery["best_metrics"]
    for name, value in terminal_recovery["best_state_dict"].items():
        assert torch.equal(terminal.state_dict()[name], value)


def test_support_worker_cache_cap_and_clear_preserve_exact_support() -> None:
    catalog, _ = _catalog_and_records(("CCO", "c1ccccc1", "C1CCCCC1"))
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=1,
        message_passing_steps=1,
        mark_dim=1,
        ring_candidate_cache_limit=1,
    )
    state = pad_molecular_graph(smiles_to_molecular_graph("CCCCCC"), 12)

    expected = model._ring_grow_support(state)
    assert len(model._ring_grow_support_cache) <= 1
    model.clear_ring_candidate_caches()

    assert not model._ring_grow_support_cache
    assert not model._ring_grow_enablement_certificate_cache
    assert model._ring_grow_support(state) == expected


def test_complete_ring_fiber_cannot_overgrow_an_existing_ring_system() -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("C1CCCCC1"), 12)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(37),
        n_slots=12,
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((trace,))
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
    ).eval()
    batch = prepare_factorized_mark_batch(
        (target,),
        (0.5,),
        (None,),
        (None,),
        (0.0,),
    )
    with torch.no_grad():
        node, global_state, pair = model._encode_batch(batch)
        masks, _, _ = model._action_tables(batch, node, global_state, pair)
    assert not bool(masks["ring_system_grow"][0].any())
    assert bool(masks["ring_system_delete"][0].any())


def test_production_factorization_has_no_ring_ear_family() -> None:
    assert "ring_ear_insert" not in MARK_RULE_TO_INDEX
    assert "ring_system_grow" in MARK_RULE_TO_INDEX
    assert "ring_system_delete" in MARK_RULE_TO_INDEX


def test_sampling_rule_ablation_removes_full_ring_grow_family() -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("C1CCCCC1"), 8)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(41),
        n_slots=8,
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((trace,))
    path = TraceProgressCTMC(trace)
    progress = next(
        index for index, step in enumerate(trace.steps) if step.rule_name == "ring_system_grow"
    )
    state = path.states[progress]
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
    ).eval()
    with torch.no_grad():
        model.family_head[-1].weight.zero_()
        model.family_head[-1].bias.fill_(-100.0)
        model.family_head[-1].bias[MARK_RULE_TO_INDEX["ring_system_grow"]] = 100.0

    baseline = model.sample_rewrite_mark(state, 0.5, np.random.default_rng(7))
    model.disabled_sampling_rule_names = frozenset({"ring_system_grow"})
    ablated = model.sample_rewrite_mark(state, 0.5, np.random.default_rng(7))

    assert baseline.rule_name == "ring_system_grow"
    assert ablated.rule_name != "ring_system_grow"


def test_factorized_graft_and_full_ring_teachers_are_legal_and_trainable() -> None:
    target = pad_molecular_graph(
        smiles_to_molecular_graph("C1CCC2CCCCC2C1"),
        16,
    )
    source = DegreeBoundedCarbonTreePrior(
        sizes=(target.n_real_atoms,),
    ).sample(np.random.default_rng(323), n_slots=16)
    proposal = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((proposal,))
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
        ring_catalog=catalog,
    )
    path = TraceProgressCTMC(trace)
    progress = [
        index
        for index, step in enumerate(trace.steps)
        if step.rule_name in {"bond_reroute", "ring_system_grow"}
    ]
    assert {trace.steps[index].rule_name for index in progress} == {
        "bond_reroute",
        "ring_system_grow",
    }
    batch = prepare_factorized_mark_batch(
        tuple(path.states[index] for index in progress),
        tuple(0.5 for _ in progress),
        tuple(trace.steps[index].action for index in progress),
        tuple(trace.steps[index].rule_name for index in progress),
        tuple(path.operational_jump_rate(index) for index in progress),
    )
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=32,
        message_passing_steps=1,
    )
    prediction = model.forward_mark_batch(batch)
    loss = factorized_mark_bregman_loss(prediction, batch)
    loss.backward()

    assert torch.isfinite(prediction.selected_mark_log_probability).all()
    assert torch.isfinite(loss)
    assert model.graft_head[-1].weight.grad is not None
    assert torch.isfinite(model.graft_head[-1].weight.grad).all()
    assert model.ring_system_grow_head[-1].weight.grad is not None
    assert torch.isfinite(model.ring_system_grow_head[-1].weight.grad).all()
    assert model.ring_system_template_key.weight.grad is not None
    assert torch.isfinite(model.ring_system_template_key.weight.grad).all()


def test_dense_graft_support_excludes_canonical_self_successors() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CCCCCCCC"), 12)
    batch = prepare_factorized_mark_batch(
        (state,),
        (0.5,),
        (None,),
        (None,),
        (0.0,),
    )
    runtime = de_novo_rewrite_system()
    current_key = canonical_state_key(state)
    enabled = torch.nonzero(batch.graft_mask[0], as_tuple=False)

    # A labeled eight-carbon chain has 42 root-free coordinates before
    # quotienting; 12 are molecular self-successors and must disappear.
    real = tuple(int(value) for value in np.flatnonzero(is_element(state.atom_types)))
    raw_coordinates = sum(
        int(moved != target and int(state.bonds[moved, target]) == 0)
        for moved in real
        for target in real
    )
    assert raw_coordinates == 42
    assert len(enabled) == 30
    assert int((batch.graft_successor_groups[0] >= 0).sum()) == len(enabled)

    successor_keys_by_group: dict[int, set[str]] = {}
    for moved_tensor, target_tensor in enabled:
        moved, target = int(moved_tensor), int(target_tensor)
        removed = int(batch.graft_remove_neighbors[0, moved, target])
        successor = runtime.apply(
            state,
            "bond_reroute",
            BondReroute(a=moved, b=removed, u=moved, v=target),
        )
        successor_key = canonical_state_key(successor)
        assert successor_key != current_key
        group = int(batch.graft_successor_groups[0, moved, target])
        successor_keys_by_group.setdefault(group, set()).add(successor_key)
    assert all(len(keys) == 1 for keys in successor_keys_by_group.values())
    grouped_keys = [next(iter(keys)) for keys in successor_keys_by_group.values()]
    assert len(grouped_keys) == len(set(grouped_keys))


@pytest.mark.parametrize("n_atoms", (6, 12, 24, 40))
def test_incremental_graft_keys_match_exhaustive_canonicalization(
    n_atoms: int,
) -> None:
    states = [
        pad_molecular_graph(smiles_to_molecular_graph("C" * n_atoms), 40),
        DegreeBoundedCarbonTreePrior(sizes=(n_atoms,)).sample(
            np.random.default_rng(8100 + n_atoms),
            n_slots=40,
        ),
    ]
    if n_atoms == 40:
        # A reflection-symmetric, genuinely colored tree exercises the
        # incremental/orbit path without relying on an all-carbon alphabet.
        states.append(
            pad_molecular_graph(
                smiles_to_molecular_graph("N" + "C" * 38 + "N"),
                40,
            )
        )
    for state in states:
        atom_topology = compute_topology_features(state)[0]
        exhaustive = _graph_application_masks(
            state,
            atom_topology,
            optimize_graft_canonicalization=False,
        )
        optimized = _graph_application_masks(
            state,
            atom_topology,
            optimize_graft_canonicalization=True,
        )
        assert all(
            np.array_equal(reference, candidate)
            for reference, candidate in zip(exhaustive, optimized)
        )


def test_chemistry_features_are_computed_once_per_exact_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CCCCCCCC"), 12)
    calls = 0
    original = compute_topology_features

    def counted(current_state):
        nonlocal calls
        calls += 1
        return original(current_state)

    monkeypatch.setattr(
        "compose_v4.model.factorized_tracelet_rate_model.compute_topology_features",
        counted,
    )
    cache = OrderedDict()
    for _ in range(2):
        batch = prepare_factorized_mark_batch(
            (state, state, state),
            (0.2, 0.5, 0.8),
            (None, None, None),
            (None, None, None),
            (0.0, 0.0, 0.0),
            chemistry_feature_cache=cache,
        )
        assert batch.batch_size == 3
    assert calls == 1


def test_collator_does_not_recompute_precomputed_ring_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, state, _ = _ring_support_case()
    width = len(FactorizedTraceletRateModel(catalog).ring_system_templates)
    example = FactorizedMarkExample(
        state=state,
        time=0.5,
        teacher_action=None,
        teacher_rule_name=None,
        teacher_rate=0.0,
        importance_weight=1.0,
        ring_grow_support_indices=(0,),
        ring_grow_support_width=width,
        ring_grow_enablement_is_exact=True,
    )

    def reject_duplicate_support(*_args, **_kwargs):
        raise AssertionError("collator recomputed a precomputed ring-support row")

    monkeypatch.setattr(
        "compose_v4.model.factorized_tracelet_rate_model."
        "ring_system_template_local_support_mask",
        reject_duplicate_support,
    )
    batch = FactorizedMarkCollator(True, catalog)([example])
    assert batch.ring_grow_support_mask is None
    assert batch.ring_grow_support_sparse is not None
    assert torch.equal(
        batch.ring_grow_support_sparse.to_dense(),
        torch.tensor([[True] + [False] * (width - 1)]),
    )


def test_sparse_ring_support_is_loss_equivalent_to_dense_support() -> None:
    catalog, state, action = _ring_support_case()
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
    )
    support = model._ring_grow_support(state)
    dense_batch = prepare_factorized_mark_batch(
        (state,),
        (0.5,),
        (action,),
        ("ring_system_grow",),
        (1.0,),
    )
    dense_batch = replace(
        dense_batch,
        ring_grow_support_mask=torch.tensor((support,), dtype=torch.bool),
        ring_grow_support_is_exact=torch.tensor((True,)),
        ring_grow_enablement_is_exact=torch.tensor((True,)),
    )
    sparse_batch = replace(
        dense_batch,
        ring_grow_support_mask=None,
        ring_grow_support_sparse=SparseBinaryRows.from_dense(
            dense_batch.ring_grow_support_mask
        ),
    )

    dense_prediction = model.forward_mark_batch(dense_batch)
    sparse_prediction = model.forward_mark_batch(sparse_batch)
    assert torch.equal(
        dense_prediction.enabled_families,
        sparse_prediction.enabled_families,
    )
    assert torch.allclose(
        dense_prediction.selected_mark_log_probability,
        sparse_prediction.selected_mark_log_probability,
    )
    assert torch.allclose(
        factorized_mark_bregman_loss(dense_prediction, dense_batch),
        factorized_mark_bregman_loss(sparse_prediction, sparse_batch),
    )


def test_graft_teacher_score_aggregates_mark_aliases_by_successor() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CCCCCCCC"), 12)
    support = prepare_factorized_mark_batch(
        (state,),
        (0.5,),
        (None,),
        (None,),
        (0.0,),
    )
    groups = support.graft_successor_groups[0]
    group_values, counts = torch.unique(groups[groups >= 0], return_counts=True)
    group = int(group_values[int(torch.argmax(counts))])
    aliases = torch.nonzero(
        support.graft_mask[0] & (groups == group),
        as_tuple=False,
    )
    assert len(aliases) > 1
    moved, target = (int(value) for value in aliases[0])
    removed = int(support.graft_remove_neighbors[0, moved, target])
    teacher = BondReroute(a=moved, b=removed, u=moved, v=target)
    batch = prepare_factorized_mark_batch(
        (state,),
        (0.5,),
        (teacher,),
        ("bond_reroute",),
        (1.0,),
    )
    catalog, _ = _catalog_and_records(("CCCCCCCC",))
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
    )
    node, global_state, pair = model._encode_batch(batch)
    masks, logits, _ = model._action_tables(batch, node, global_state, pair)
    score, legal = model._teacher_action_score(
        0,
        teacher,
        "bond_reroute",
        batch,
        node,
        global_state,
        pair,
        masks,
        logits,
    )
    group_mask = masks["bond_reroute"][0] & (
        batch.graft_successor_groups[0] == group
    )
    expected = torch.logsumexp(
        logits["bond_reroute"][0][group_mask],
        dim=0,
    )
    assert bool(legal)
    assert int(group_mask.sum()) == len(aliases)
    assert torch.allclose(score, expected)
    assert not torch.allclose(score, logits["bond_reroute"][0, moved, target])


def test_forced_graft_sampling_never_emits_a_molecular_self_transition() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CCCCCCCC"), 12)
    catalog, _ = _catalog_and_records(("CCCCCCCC",))
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
    ).eval()
    with torch.no_grad():
        model.family_head[-1].weight.zero_()
        model.family_head[-1].bias.fill_(-100.0)
        model.family_head[-1].bias[MARK_RULE_TO_INDEX["bond_reroute"]] = 100.0
    runtime = de_novo_rewrite_system()
    current_key = canonical_state_key(state)
    for seed in range(32):
        sampled = model.sample_rewrite_mark(state, 0.5, np.random.default_rng(seed))
        assert sampled.rule_name == "bond_reroute"
        successor = runtime.apply(state, sampled.rule_name, sampled.action)
        assert canonical_state_key(successor) != current_key


def test_structured_ring_decoder_generalizes_atom_labels_beyond_catalog() -> None:
    training_target = pad_molecular_graph(
        smiles_to_molecular_graph("c1ccccc1"),
        12,
    )
    heldout_target = pad_molecular_graph(
        smiles_to_molecular_graph("c1ccncc1"),
        12,
    )
    source = DegreeBoundedCarbonTreePrior(sizes=(6,)).sample(
        np.random.default_rng(509),
        n_slots=12,
    )
    training_trace = compile_carbon_tree_to_target(
        source,
        training_target,
        use_bond_reroute=True,
        align_source=True,
    )
    heldout_trace = compile_carbon_tree_to_target(
        source,
        heldout_target,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((training_trace,))
    path = TraceProgressCTMC(heldout_trace)
    progress = next(
        index
        for index, step in enumerate(heldout_trace.steps)
        if step.rule_name == "ring_system_grow"
    )
    step = heldout_trace.steps[progress]
    batch = prepare_factorized_mark_batch(
        (path.states[progress],),
        (0.5,),
        (step.action,),
        (step.rule_name,),
        (path.operational_jump_rate(progress),),
    )
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=24,
        message_passing_steps=1,
    )
    prediction = model.forward_mark_batch(batch)
    loss = factorized_mark_bregman_loss(prediction, batch)
    loss.backward()

    assert not catalog.supports_trace(heldout_trace)
    assert structured_ring_trace_supported(heldout_trace, catalog)
    assert torch.isfinite(prediction.selected_mark_log_probability).all()
    assert torch.isfinite(loss)
    assert model.ring_system_atom_head[-1].weight.grad is not None


def test_structured_ring_decoder_resolves_unseen_kekule_alias() -> None:
    training_target = pad_molecular_graph(
        smiles_to_molecular_graph("c1ccccc1"),
        12,
    )
    heldout_target = pad_molecular_graph(
        smiles_to_molecular_graph("O=c1cccco1"),
        12,
    )
    training_source = DegreeBoundedCarbonTreePrior(sizes=(6,)).sample(
        np.random.default_rng(811),
        n_slots=12,
    )
    heldout_source = DegreeBoundedCarbonTreePrior(sizes=(7,)).sample(
        np.random.default_rng(821),
        n_slots=12,
    )
    training_path = TraceProgressCTMC(
        compile_carbon_tree_to_target(
            training_source,
            training_target,
            use_bond_reroute=True,
            align_source=True,
        )
    )
    heldout_trace = compile_carbon_tree_to_target(
        heldout_source,
        heldout_target,
        use_bond_reroute=True,
        align_source=True,
    )
    heldout_path = TraceProgressCTMC(heldout_trace)
    catalog = build_typed_ring_catalog_from_paths((training_path,))
    progress = next(
        index
        for index, step in enumerate(heldout_trace.steps)
        if step.rule_name == "ring_system_grow"
    )
    step = heldout_trace.steps[progress]
    batch = prepare_factorized_mark_batch(
        (heldout_path.state_at(progress),),
        (0.5,),
        (step.action,),
        (step.rule_name,),
        (heldout_path.operational_jump_rate(progress),),
        ring_catalog=catalog,
    )
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
    )
    prediction = model.forward_mark_batch(batch)
    placement_groups = model._ring_template_placement_groups(
        heldout_path.state_at(progress),
        0,
    )

    assert bool(batch.ring_grow_support_mask[0, 0])
    assert len(placement_groups) == 1
    assert len(placement_groups[0]) == 1
    assert torch.isfinite(prediction.selected_mark_log_probability).all()


def test_semantic_ring_teacher_and_sampler_share_one_normalized_support() -> None:
    target = pad_molecular_graph(
        smiles_to_molecular_graph("c1cc[nH]c1"),
        12,
    )
    source = DegreeBoundedCarbonTreePrior(
        sizes=(target.n_real_atoms,),
    ).sample(np.random.default_rng(1701), n_slots=12)
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    path = TraceProgressCTMC(trace)
    catalog = build_typed_ring_catalog_from_paths((path,))
    progress = next(
        index
        for index, step in enumerate(trace.steps)
        if step.rule_name == "ring_system_grow"
    )
    step = trace.steps[progress]
    state = path.state_at(progress)
    batch = prepare_factorized_mark_batch(
        (state,),
        (0.5,),
        (step.action,),
        (step.rule_name,),
        (path.operational_jump_rate(progress),),
        ring_catalog=catalog,
    )
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=24,
        message_passing_steps=1,
    )
    prediction = model.forward_mark_batch(batch)
    loss = factorized_mark_bregman_loss(prediction, batch)
    loss.backward()

    template_index = matching_ring_system_template_indices(
        step.action,
        model.ring_system_templates,
    )[0]
    placement_groups = model._ring_template_placement_groups(
        state,
        template_index,
    )
    assert placement_groups
    assert all(len(group) == 1 for group in placement_groups)
    with torch.no_grad():
        device_batch = batch.to(model.device)
        node, global_state, pair = model._encode_batch(device_batch)
        _, semantic_tables, placement_mask = model._ring_supported_placement_tables(
            state,
            placement_groups,
            node[0],
            global_state[0],
            pair[0],
        )
        matching_index = next(
            index
            for index, (decoder, _) in enumerate(semantic_tables)
            if ring_system_placement_key(decoder.placement)
            == ring_system_placement_key(ring_system_placement(step.action))
        )
        assert bool(placement_mask[matching_index])
        decoder, category_logits = semantic_tables[matching_index]
        teacher_categories = semantic_ring_categories_for_action(
            decoder,
            step.action,
        )
        teacher_score, teacher_legal = (
            model._ring_semantic_sequence_log_probability(
                decoder,
                category_logits,
                teacher_categories,
            )
        )

        sequence_scores = []

        def visit(prefix: tuple[int, ...]) -> None:
            if len(prefix) == decoder.span:
                score, legal = model._ring_semantic_sequence_log_probability(
                    decoder,
                    category_logits,
                    prefix,
                )
                assert bool(legal)
                sequence_scores.append(score)
                return
            mask = semantic_ring_next_category_mask(decoder, prefix)
            for category, enabled in enumerate(mask):
                if enabled:
                    visit((*prefix, category))

        visit(())
        model.family_head[-1].weight.zero_()
        model.family_head[-1].bias.fill_(-100.0)
        model.family_head[-1].bias[MARK_RULE_TO_INDEX["ring_system_grow"]] = 100.0
        sampled = model.sample_rewrite_mark(
            state,
            0.5,
            np.random.default_rng(1702),
        )

    assert bool(teacher_legal)
    assert torch.isfinite(teacher_score)
    assert sequence_scores
    assert torch.exp(torch.stack(sequence_scores)).sum().item() == pytest.approx(1.0)
    assert sampled.rule_name == "ring_system_grow"
    assert is_valid_ring_system_grow(state, sampled.action)
    sampled_decoder = next(
        decoder
        for decoder, _ in semantic_tables
        if ring_system_placement_key(decoder.placement)
        == ring_system_placement_key(ring_system_placement(sampled.action))
    )
    sampled_categories = semantic_ring_categories_for_action(
        sampled_decoder,
        sampled.action,
    )
    assert semantic_ring_prefix_is_completable(
        sampled_decoder,
        sampled_categories,
    )
    assert model.ring_system_role_head[-1].weight.grad is not None
    assert torch.isfinite(model.ring_system_role_head[-1].weight.grad).all()


def test_precomputed_ring_teacher_certificate_is_score_exact() -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("c1cc[nH]c1"), 12)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(1711),
        n_slots=12,
    )
    path = TraceProgressCTMC(
        compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            align_source=True,
        )
    )
    catalog = build_typed_ring_catalog_from_paths((path,))
    progress = next(
        index
        for index, step in enumerate(path.trace.steps)
        if step.rule_name == "ring_system_grow"
    )
    step = path.trace.steps[progress]
    state = path.state_at(progress)
    batch = prepare_factorized_mark_batch(
        (state,),
        (0.5,),
        (step.action,),
        (step.rule_name,),
        (path.operational_jump_rate(progress),),
        ring_catalog=catalog,
    )
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=24,
        message_passing_steps=1,
    )
    reference = model.forward_mark_batch(batch).selected_mark_log_probability
    certified_batch = attach_ring_teacher_semantic_certificates(
        batch,
        ring_catalog=catalog,
        ring_electronic_mode="factorized_local",
        workers=0,
    )
    certificate = certified_batch.ring_teacher_semantic_certificates[0]
    certified = model.forward_mark_batch(
        certified_batch
    ).selected_mark_log_probability

    assert certificate is not None
    assert certificate.action_is_valid
    assert certificate.templates
    assert torch.equal(reference, certified)


def test_fused_heteroaromatic_teacher_keeps_executable_resonance_alias() -> None:
    target = pad_molecular_graph(
        smiles_to_molecular_graph("c1ccc2ncccc2c1"),
        20,
    )
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(701),
        n_slots=20,
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
    step = trace.steps[progress]
    batch = prepare_factorized_mark_batch(
        (path.state_at(progress),),
        (0.5,),
        (step.action,),
        (step.rule_name,),
        (path.operational_jump_rate(progress),),
    )
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
    )

    prediction = model.forward_mark_batch(batch)
    loss = factorized_mark_bregman_loss(prediction, batch)

    assert torch.isfinite(prediction.selected_mark_log_probability).all()
    assert torch.isfinite(loss)


def test_exact_ring_electronic_decoder_shares_normalized_teacher_sampler_table() -> None:
    target = pad_molecular_graph(
        smiles_to_molecular_graph("c1ccc2ncccc2c1"),
        20,
    )
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(761),
        n_slots=20,
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
    state = path.state_at(progress)
    step = trace.steps[progress]
    batch = prepare_factorized_mark_batch(
        (state,),
        (0.5,),
        (step.action,),
        (step.rule_name,),
        (path.operational_jump_rate(progress),),
        ring_catalog=catalog,
    )
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=24,
        message_passing_steps=1,
        ring_electronic_mode="catalog_exact",
    )
    prediction = model.forward_mark_batch(batch)
    loss = factorized_mark_bregman_loss(prediction, batch)
    loss.backward()

    template_index = matching_ring_system_template_indices(
        step.action,
        model.ring_system_templates,
    )[0]
    with torch.no_grad():
        node, global_state, pair = model._encode_batch(batch.to(model.device))
        candidates, log_probabilities = model._ring_exact_candidate_log_probabilities(
            state,
            template_index,
            node[0],
            global_state[0],
            pair[0],
        )
        first_cache_result = model._ring_paired_candidates(state, template_index)
        second_cache_result = model._ring_paired_candidates(state, template_index)
        model.family_head[-1].weight.zero_()
        model.family_head[-1].bias.fill_(-100.0)
        model.family_head[-1].bias[MARK_RULE_TO_INDEX["ring_system_grow"]] = 100.0
        sampled = model.sample_rewrite_mark(state, 0.5, np.random.default_rng(762))

    assert torch.isfinite(prediction.selected_mark_log_probability).all()
    assert torch.isfinite(loss)
    assert candidates
    assert torch.exp(log_probabilities).sum().item() == pytest.approx(1.0)
    assert first_cache_result is second_cache_result
    assert ring_system_grow_electronic_key(step.action) in {
        ring_system_grow_electronic_key(candidate.action) for candidate in candidates
    }
    assert sampled.rule_name == "ring_system_grow"
    assert ring_system_grow_electronic_key(sampled.action) in {
        ring_system_grow_electronic_key(candidate.action) for candidate in candidates
    }
    assert model.ring_system_grow_head[-1].weight.grad is not None
    assert model.ring_system_atom_head[-1].weight.grad is not None


def test_aromatic_nitrogen_masks_cover_ring_nitrogen_teachers() -> None:
    for seed, smiles in enumerate(
        (
            "c1ncnnc1",
            "c1ccc2[nH]ccc2c1",
            "O=c1cc[nH]c(=O)[nH]1",
            "O=c1ccc2ccccc2o1",
        ),
        start=811,
    ):
        target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 20)
        source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
            np.random.default_rng(seed),
            n_slots=20,
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
        step = trace.steps[progress]
        batch = prepare_factorized_mark_batch(
            (path.state_at(progress),),
            (0.5,),
            (step.action,),
            (step.rule_name,),
            (path.operational_jump_rate(progress),),
        )
        model = FactorizedTraceletRateModel(
            catalog,
            hidden_dim=16,
            message_passing_steps=1,
        )

        prediction = model.forward_mark_batch(batch)
        loss = factorized_mark_bregman_loss(prediction, batch)

        assert torch.isfinite(prediction.selected_mark_log_probability).all()
        assert torch.isfinite(loss)


def test_precomputed_ring_application_conditions_match_dynamic_forward() -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("c1ncnnc1"), 12)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(907),
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
    step = trace.steps[progress]
    arguments = (
        (path.state_at(progress),),
        (0.5,),
        (step.action,),
        (step.rule_name,),
        (path.operational_jump_rate(progress),),
    )
    dynamic_batch = prepare_factorized_mark_batch(*arguments)
    precomputed_batch = prepare_factorized_mark_batch(
        *arguments,
        ring_catalog=catalog,
    )
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
    )

    dynamic = model.forward_mark_batch(dynamic_batch)
    precomputed = model.forward_mark_batch(precomputed_batch)

    assert precomputed_batch.ring_grow_support_mask is not None
    assert precomputed_batch.ring_delete_actions is not None
    assert torch.equal(dynamic.enabled_families, precomputed.enabled_families)
    assert torch.allclose(
        dynamic.selected_mark_log_probability,
        precomputed.selected_mark_log_probability,
    )
