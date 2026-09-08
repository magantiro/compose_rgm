"""CPU qualification of a source-conditioned law, not generated-lipid quality."""

import io
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import NULL_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import empty_molecular_graph, pad_molecular_graph
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkExample,
    _concatenate_factorized_mark_batches,
)
from compose_v4.experiments.tracelet_conditional import build_tracelet_path_records
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import InvalidRewrite
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondReorder,
    BondReroute,
)
from compose_v4.rewrite.scaffold_construction import ScaffoldContext
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog


def graph(smiles):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 8)


@pytest.fixture(params=("hierarchical", "superposed"))
def model(request):
    records = build_tracelet_path_records(("CCO", "C1CC1", "N1CC1"), n_slots=8)
    ring_trace = compile_carbon_tree_to_target(
        graph("CCC"), graph("C1CC1"), use_bond_reroute=True, align_source=True
    )
    catalog = build_typed_ring_catalog((*tuple(r.path.trace for r in records), ring_trace))
    torch.manual_seed(911)
    return FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=2,
        mark_dim=8,
        enable_ring_restates=True,
        enable_heteroatom_scan=True,
        rate_factorization=request.param,
        scaffold_conditioning=True,
    ).eval()


def batch_for(model, state, context, action=None, family=None):
    return prepare_factorized_mark_batch(
        (state,),
        (0.4,),
        (action,),
        (family,),
        (1.0 if action is not None else 0.0,),
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        compute_ring_restates=True,
        scaffold_contexts=None if context is None else (context,),
    )


def test_prepared_support_identity_binds_vocabulary_contents_not_object_address(model):
    from compose_v4.chem.molecular_graph import AtomVocabulary
    from compose_v4.model.scaffold_prepared import scaffold_support_key

    original = model.atom_vocabulary
    before = scaffold_support_key(model)
    model.atom_vocabulary = AtomVocabulary(original.classes)
    assert model.atom_vocabulary is not original
    assert scaffold_support_key(model) == before
    model.atom_vocabulary = AtomVocabulary(tuple(reversed(original.classes)))
    assert scaffold_support_key(model) != before


def test_ring_restate_teacher_undirected_reencoding_preserves_score_and_gradient(model):
    from compose_v4.rewrite.tracelets import BondOrderChange, RingSystemRestate

    state = graph("C1CCCCC1")
    context = ScaffoldContext.from_source(empty_molecular_graph(8))
    candidates = batch_for(model, state, context).ring_restate_actions[0]
    assert candidates
    canonical = candidates[0]
    alias = RingSystemRestate(tuple(BondOrderChange(c.b, c.a, c.new_order)
                                   for c in reversed(canonical.changes)))
    assert alias != canonical
    outputs, gradients = [], []
    for action in (canonical, alias):
        model.zero_grad(set_to_none=True)
        batch = batch_for(model, state, context, action, "ring_system_restate")
        from compose_v4.experiments.factorized_mark_conditional import DEFAULT_MARK_TRAINING_OBJECTIVE
        DEFAULT_MARK_TRAINING_OBJECTIVE.validate_batch(batch)
        output = model.forward_mark_batch(batch)
        output.selected_mark_log_probability.sum().backward()
        outputs.append(output.selected_mark_log_probability.detach())
        gradients.append({name: None if p.grad is None else p.grad.clone()
                          for name, p in model.named_parameters()})
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for name, left in gradients[0].items():
        right = gradients[1][name]
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def tables(model, batch):
    return model._action_tables(batch, *model._encode_batch(batch))


@pytest.mark.parametrize(
    "source,current,ports",
    [
        ("C", "CCC", (0,)),
        ("NC", "NCC", (1,)),
        ("NC", "NCC", (0, 1)),
        ("CC", "CC(C)C", (1,)),
        ("C1CC1", "C1CC1C", (2,)),
    ],
)
def test_primitive_masks_equal_independent_executor_filter(model, source, current, ports):
    state = graph(current)
    context = ScaffoldContext.from_source(graph(source), ports)
    empty = ScaffoldContext.from_source(empty_molecular_graph(8))
    base, _, _ = tables(model, batch_for(model, state, empty))
    conditional, _, _ = tables(model, batch_for(model, state, context))
    runtime = context.rewrite_system()
    prepared = batch_for(model, state, empty)
    free_slot = int(np.flatnonzero(state.atom_types == NULL_IDX)[0])

    for name in (
        "grow_connected",
        "atom_delete",
        "atom_restate",
        "bond_reorder",
        "bond_reroute",
        "cycle_attach",
    ):
        expected = torch.zeros_like(base[name])
        for coordinate in torch.nonzero(base[name][0], as_tuple=False).tolist():
            if name == "grow_connected":
                anchor, order, cls = coordinate
                element, valence = model.atom_vocabulary.classes[cls]
                action = AtomInsert(
                    free_slot, element, 0, valence - order - 1, ((anchor, order + 1),)
                )
                family = "atom_insert"
            elif name == "atom_delete":
                action, family = AtomDelete(coordinate[0]), name
            elif name == "atom_restate":
                site, cls = coordinate
                element, valence = model.atom_vocabulary.classes[cls]
                action = AtomRestate(site, element, 0, valence - int(state.bonds[site].sum()))
                family = name
            elif name == "bond_reorder":
                a, b, order = coordinate
                action, family = BondReorder(a, b, order + 1), name
            elif name == "bond_reroute":
                moved, target = coordinate
                old = int(prepared.graft_remove_neighbors[0, moved, target])
                action, family = BondReroute(moved, old, moved, target), name
            else:
                anchor, template_index = coordinate
                template = model.attach_templates[template_index]
                free = tuple(int(v) for v in np.flatnonzero(state.atom_types == NULL_IDX))
                action = template.instantiate(anchor, free[: template.span])
                family = name
            try:
                runtime.apply(state, family, action)
            except InvalidRewrite:
                continue
            expected[(0, *coordinate)] = True
        assert torch.equal(conditional[name], expected), name


def test_context_is_neural_input_with_gradient_and_protected_teachers_have_zero_probability(model):
    state = graph("NCC")
    contexts = [ScaffoldContext.from_source(graph("NC"), ports) for ports in ((1,), (0, 1))]
    batches = [batch_for(model, state, c) for c in contexts]
    nodes = [model._encode_batch(b)[0] for b in batches]
    assert not torch.allclose(*nodes)
    nodes[0].square().sum().backward()
    assert torch.isfinite(model.scaffold_encoder.weight.grad).all()
    assert model.scaffold_encoder.weight.grad.abs().sum() > 0
    deleted = model.forward_mark_batch(
        batch_for(model, state, contexts[0], AtomDelete(0), "atom_delete")
    )
    assert torch.isneginf(deleted.selected_mark_log_probability).all()


def test_collation_streaming_and_movement_preserve_explicit_contexts(model):
    state = graph("NCC")
    contexts = [ScaffoldContext.from_source(graph("NC"), p) for p in ((1,), (0, 1))]
    collator = FactorizedMarkCollator.from_capabilities(
        model.operator_capabilities, use_aromatic_bond_view=True, ring_catalog=model.ring_catalog
    )
    examples = [
        FactorizedMarkExample(state, 0.4, None, None, 0.0, 1.0, scaffold_context=c)
        for c in contexts
    ]
    batch = collator(examples)
    joined = _concatenate_factorized_mark_batches((batch.subbatch(0, 1), batch.subbatch(1, 2)))
    assert joined.scaffold_contexts == tuple(contexts)
    assert torch.equal(joined.scaffold_node_flags, batch.scaffold_node_flags)
    assert joined.to(torch.device("cpu")).scaffold_contexts == joined.scaffold_contexts
    original = model.forward_mark_batch(batch)
    replay = model.forward_mark_batch(joined)
    torch.testing.assert_close(original.total_hazard, replay.total_hazard, rtol=0, atol=0)
    torch.testing.assert_close(
        original.family_log_probabilities, replay.family_log_probabilities, rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="explicit empty"):
        collator([examples[0], replace(examples[1], scaffold_context=None)])


def test_unconditioned_defaults_and_empty_context_are_unchanged(model):
    kwargs = {
        "hidden_dim": 16,
        "message_passing_steps": 2,
        "mark_dim": 8,
        "enable_ring_restates": True,
        "enable_heteroatom_scan": True,
        "rate_factorization": model.rate_factorization,
    }
    torch.manual_seed(911)
    base = FactorizedTraceletRateModel(model.ring_catalog, **kwargs).eval()
    assert "scaffold_encoder.weight" not in base.state_dict()
    for name, parameter in base.state_dict().items():
        torch.testing.assert_close(parameter, model.state_dict()[name], rtol=0, atol=0)
    for state in (graph("CCC"), empty_molecular_graph(8)):
        empty = ScaffoldContext.from_source(empty_molecular_graph(8))
        original = base.forward_mark_batch(batch_for(base, state, None))
        conditioned = model.forward_mark_batch(batch_for(model, state, empty))
        torch.testing.assert_close(original.total_hazard, conditioned.total_hazard, rtol=0, atol=0)
        torch.testing.assert_close(
            original.family_log_probabilities, conditioned.family_log_probabilities, rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="cannot silently ignore"):
        base.forward_mark_batch(
            batch_for(base, graph("C"), ScaffoldContext.from_source(graph("C"), (0,)))
        )


def test_sampler_uses_same_hazard_and_supported_law_and_keeps_core(model):
    state = graph("NCC")
    contexts = [ScaffoldContext.from_source(graph("NC"), p) for p in ((1,), (0, 1))]
    rng = np.random.default_rng(913)
    for context in contexts * 2:  # Interleave same-state/different-condition cache hits.
        reference = model.forward_mark_batch(batch_for(model, state, context))
        for _ in range(12):
            mark = model.sample_rewrite_mark_conditioned(
                state, 0.4, rng, property_values=None, scaffold_context=context
            )
            assert mark.total_hazard == pytest.approx(reference.total_hazard.item(), rel=1e-6)
            successor = context.rewrite_system().apply(state, mark.rule_name, mark.action)
            assert context.accepts(successor)
            scored = model.forward_mark_batch(
                batch_for(model, state, context, mark.action, mark.rule_name)
            )
            assert torch.isfinite(scored.selected_mark_log_probability).all()
    assert len(model._sampling_state_cache) == 2


def test_ring_language_is_conditioned_and_has_normalized_probability(model):
    # The existing model enumerates ring closure on carbon carriers.  Its
    # semantic decoder can then choose heteroatoms at unprotected positions.
    # A typed NCC carrier is tested at decoder level in test_scaffold_ring_decoder,
    # not falsely advertised as support of this unchanged model enumerator.
    state = graph("CCC")
    context = ScaffoldContext.from_source(graph("CC"), (0, 1))
    batch = batch_for(model, state, context)
    node, global_state, pair = model._encode_batch(batch)
    from compose_v4.rewrite.ring_system_fiber import (
        instantiate_semantic_ring_system_grow,
        semantic_ring_next_category_mask,
    )

    supported = model._ring_grow_support(state, context)
    assert any(supported)
    blocked = ScaffoldContext.from_source(graph("CC"), (1,))
    assert not any(model._ring_grow_support(state, blocked))
    typed_state = graph("NCC")
    typed_core = ScaffoldContext.from_source(graph("NC"), (0, 1))
    assert not model._ring_template_placement_groups(typed_state, 0)
    assert not any(model._ring_grow_support(typed_state, typed_core))
    for index, active in enumerate(supported):
        if not active:
            continue
        groups = model._ring_template_placement_groups(state, index)
        _, semantic_tables, mask = model._ring_supported_placement_tables(
            state, groups, node[0], global_state[0], pair[0], scaffold_context=context
        )
        for (decoder, logits), enabled in zip(semantic_tables, mask):
            if not enabled:
                continue
            probabilities = []
            for labels in product(range(logits.shape[1]), repeat=decoder.span):
                if any(
                    not semantic_ring_next_category_mask(decoder, labels[:i])[label]
                    for i, label in enumerate(labels)
                ):
                    continue
                score, legal = model._ring_semantic_sequence_log_probability(
                    decoder, logits, labels
                )
                assert legal
                probabilities.append(score.exp())
                action = instantiate_semantic_ring_system_grow(state, decoder, labels)
                context.rewrite_system().apply(state, "ring_system_grow", action)
            torch.testing.assert_close(
                torch.stack(probabilities).sum(), torch.tensor(1.0), rtol=1e-5, atol=1e-6
            )


def test_stale_flags_or_inference_only_overrides_fail_closed(model):
    state = graph("NCC")
    context = ScaffoldContext.from_source(graph("NC"), (1,))
    batch = batch_for(model, state, context)
    with pytest.raises(ValueError, match="flags disagree"):
        model.forward_mark_batch(
            replace(batch, scaffold_node_flags=torch.zeros_like(batch.scaffold_node_flags))
        )
    model.disabled_sampling_rule_names = ("atom_insert",)
    with pytest.raises(ValueError, match="inference-only"):
        model.sample_rewrite_mark_conditioned(
            state, 0.4, np.random.default_rng(1), property_values=None, scaffold_context=context
        )


def prepared_ring_example(model):
    state = graph("CCC")
    context = ScaffoldContext.from_source(graph("CC"), (0, 1))
    supported = model._ring_grow_support(state, context)
    index = next(i for i, enabled in enumerate(supported) if enabled)
    action = model._ring_witness_candidates(state, index, context)[0].action
    return batch_for(model, state, context, action, "ring_system_grow")


def test_prepared_ring_forward_and_gradients_need_no_chemistry_discovery(model, monkeypatch):
    import compose_v4.model.factorized_tracelet_rate_model as rate
    from compose_v4.model.scaffold_prepared import prepare_scaffold_mark_batch

    batch = prepared_ring_example(model)
    model.zero_grad(set_to_none=True)
    reference = model.forward_mark_batch(batch)
    (-reference.selected_mark_log_probability.sum() + reference.total_hazard.sum()).backward()
    gradients = {
        name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()
    }
    # Compilation is chemical preprocessing, not a neural encoding pass.
    with monkeypatch.context() as preparation_guard:
        preparation_guard.setattr(
            model,
            "_encode_batch",
            lambda _: (_ for _ in ()).throw(
                AssertionError("neural encoding during support preparation")
            ),
        )
        prepared = prepare_scaffold_mark_batch(model, batch)
    buffer = io.BytesIO()
    torch.save(prepared, buffer)
    buffer.seek(0)
    decoded = torch.load(buffer, weights_only=False).to(torch.device("cpu"))
    model.clear_ring_candidate_caches()

    def forbidden(*args, **kwargs):
        raise AssertionError("chemistry discovery during prepared forward")

    for name in ("_ring_grow_support", "_ring_semantic_decoder", "_ring_template_placement_groups"):
        monkeypatch.setattr(model, name, forbidden)
    for name in (
        "is_valid_ring_system_grow",
        "semantic_ring_prefix_is_completable",
        "semantic_ring_next_category_mask",
        "semantic_ring_categories_for_action",
        "matching_ring_system_template_indices",
    ):
        monkeypatch.setattr(rate, name, forbidden)
    monkeypatch.setattr(ScaffoldContext, "rewrite_system", forbidden)
    model.zero_grad(set_to_none=True)
    replay = model.forward_mark_batch(decoded)
    for name in ("total_hazard", "family_log_probabilities", "selected_mark_log_probability"):
        torch.testing.assert_close(getattr(replay, name), getattr(reference, name), rtol=0, atol=0)
    (-replay.selected_mark_log_probability.sum() + replay.total_hazard.sum()).backward()
    for name, parameter in model.named_parameters():
        if gradients[name] is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)


def test_prepared_scaffold_bindings_cannot_be_reused_for_other_teachers_or_contexts(model):
    from compose_v4.model.scaffold_prepared import prepare_scaffold_mark_batch

    batch = prepared_ring_example(model)
    prepared = prepare_scaffold_mark_batch(model, batch)
    for changed, message in (
        (replace(prepared, teacher_actions=(AtomDelete(2),)), "teacher binding"),
        (
            replace(
                prepared,
                scaffold_prepared_rows=(
                    replace(
                        prepared.scaffold_prepared_rows[0], model_support_key="different_catalog"
                    ),
                ),
            ),
            "catalog",
        ),
        (
            replace(
                prepared,
                scaffold_prepared_rows=(
                    replace(prepared.scaffold_prepared_rows[0], ring_support=()),
                ),
            ),
            "width",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            model.forward_mark_batch(changed)
    blocked = ScaffoldContext.from_source(graph("CC"), (1,))
    with pytest.raises(ValueError, match="state/context binding"):
        model.forward_mark_batch(
            replace(
                prepared,
                scaffold_contexts=(blocked,),
                scaffold_node_flags=torch.from_numpy(blocked.node_flags()[None]),
            )
        )
    with pytest.raises(ValueError, match="already prepared"):
        prepare_scaffold_mark_batch(model, prepared)
    with pytest.raises(ValueError, match="CPU-only"):
        prepare_scaffold_mark_batch(model, batch.to(torch.device("meta")))


def test_preparation_does_not_rescue_an_unsupported_ring_teacher(model):
    from compose_v4.model.scaffold_prepared import prepare_scaffold_mark_batch

    batch = prepared_ring_example(model)
    blocked = ScaffoldContext.from_source(graph("CC"), (1,))
    batch = replace(
        batch,
        scaffold_contexts=(blocked,),
        scaffold_node_flags=torch.from_numpy(blocked.node_flags()[None]),
    )
    prepared = prepare_scaffold_mark_batch(model, batch)
    assert not prepared.scaffold_prepared_rows[0].teacher_certificate.action_is_valid
    assert torch.isneginf(model.forward_mark_batch(batch).selected_mark_log_probability).all()
    assert torch.isneginf(model.forward_mark_batch(prepared).selected_mark_log_probability).all()


def test_prepared_rows_stay_aligned_after_streaming(model):
    from compose_v4.model.scaffold_prepared import prepare_scaffold_mark_batch

    ring = prepared_ring_example(model)
    primitive = batch_for(
        model,
        graph("NCC"),
        ScaffoldContext.from_source(graph("NC"), (1,)),
        AtomDelete(2),
        "atom_delete",
    )
    full = _concatenate_factorized_mark_batches((ring, primitive))
    prepared = prepare_scaffold_mark_batch(model, full)
    rebuilt = _concatenate_factorized_mark_batches(
        (prepared.subbatch(0, 1), prepared.subbatch(1, 2))
    )
    assert rebuilt.scaffold_prepared_rows == prepared.scaffold_prepared_rows
    reference, replay = model.forward_mark_batch(full), model.forward_mark_batch(rebuilt)
    torch.testing.assert_close(
        reference.selected_mark_log_probability,
        replay.selected_mark_log_probability,
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="prepared and unprepared"):
        _concatenate_factorized_mark_batches((prepared.subbatch(0, 1), primitive))


def test_slot_permutation_preserves_conditioned_teacher_rate(model):
    state = graph("NCC")
    context = ScaffoldContext.from_source(graph("NC"), (1,))
    old_to_new = np.random.default_rng(923).permutation(state.n_atoms)
    inverse = np.argsort(old_to_new)
    permuted_state = type(state)(
        state.atom_types[inverse],
        state.formal_charges[inverse],
        state.implicit_h_counts[inverse],
        state.bonds[np.ix_(inverse, inverse)],
    )
    permuted_context = context.permuted(old_to_new)
    assert permuted_context.accepts(permuted_state)
    assert context.state_cache_key(state) != permuted_context.state_cache_key(permuted_state)
    original = model.forward_mark_batch(
        batch_for(model, state, context, AtomDelete(2), "atom_delete")
    )
    moved = model.forward_mark_batch(
        batch_for(
            model, permuted_state, permuted_context, AtomDelete(int(old_to_new[2])), "atom_delete"
        )
    )
    torch.testing.assert_close(original.total_hazard, moved.total_hazard, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        original.selected_mark_log_probability,
        moved.selected_mark_log_probability,
        rtol=1e-5,
        atol=1e-6,
    )


def test_catalog_exact_ring_choices_are_normalized_and_protected(model):
    model = FactorizedTraceletRateModel(
        model.ring_catalog,
        hidden_dim=16,
        message_passing_steps=2,
        mark_dim=8,
        enable_ring_restates=True,
        enable_heteroatom_scan=True,
        rate_factorization=model.rate_factorization,
        ring_electronic_mode="catalog_exact",
        scaffold_conditioning=True,
    ).eval()
    state = graph("CCC")
    context = ScaffoldContext.from_source(graph("CC"), (0, 1))
    node, global_state, pair = model._encode_batch(batch_for(model, state, context))
    supported = model._ring_grow_support(state, context)
    assert any(supported)
    for index, enabled in enumerate(supported):
        if not enabled:
            continue
        candidates, log_probabilities = model._ring_exact_candidate_log_probabilities(
            state, index, node[0], global_state[0], pair[0], context
        )
        assert candidates
        torch.testing.assert_close(
            log_probabilities.exp().sum(), torch.tensor(1.0), rtol=1e-5, atol=1e-6
        )
        for candidate in candidates:
            assert context.accepts(
                context.rewrite_system().apply(state, "ring_system_grow", candidate.action)
            )
