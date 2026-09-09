"""Hydrogen-reserve candidate-law regressions, with no fitted model."""

import hashlib
import io
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from test_scaffold_conditioned_model import batch_for, graph, model  # noqa: F401
from test_scaffold_ring_decoder import language, placement

from compose_v4.model.scaffold_prepared import prepare_scaffold_mark_batch
from compose_v4.rewrite.compiler import TraceCompilationError
from compose_v4.rewrite.operators import AtomDelete, AtomInsert
from compose_v4.rewrite.ring_system_fiber import build_semantic_ring_system_decoder
from compose_v4.rewrite.scaffold_construction import (
    ScaffoldContext,
    compile_scaffold_to_target,
    compile_scaffold_to_target_tracelets,
)


@pytest.mark.parametrize(
    "reserve", [((0, 0),), ((0, 4),), ((3, 1),), ((0, True),), ((0, 1), (0, 2)), [[0, 1]]]
)
def test_bad_reserves_rejected(reserve):
    with pytest.raises(ValueError):
        ScaffoldContext.from_source(graph("N"), (0,), minimum_h_counts=reserve)


def test_legacy_identity_and_permuted_serialized_reserve():
    base = ScaffoldContext.from_source(graph("N"), (0,))
    fields = asdict(base)
    fields.pop("minimum_h_counts")
    legacy = hashlib.sha256(
        json.dumps({"format": "preserved_scaffold_v1", **fields}, sort_keys=True).encode()
    ).hexdigest()
    assert base.identity == legacy
    context = replace(base, minimum_h_counts=((0, 1),))
    assert base.identity != context.identity
    p = np.arange(8)[::-1]
    assert context.permuted(p).minimum_h_counts == ((7, 1),)
    assert context.permuted(p).permuted(p) == context
    buffer = io.BytesIO()
    torch.save(context, buffer)
    buffer.seek(0)
    assert torch.load(buffer, weights_only=False) == context


@pytest.mark.parametrize(
    "compiler", [compile_scaffold_to_target, compile_scaffold_to_target_tracelets]
)
def test_compilers_keep_unused_capacity_and_refuse_over_substituted_target(compiler):
    source = graph("NC=O")
    good = compiler(
        source,
        graph("N(C=O)C"),
        {0: 0, 1: 1, 2: 2},
        attachment_slots=(0,),
        minimum_h_counts=((0, 1),),
    )
    assert good.context.accepts(good.trace.source)
    assert good.context.accepts(good.trace.target)
    assert good.trace.source.implicit_h_counts[0] == 2
    assert good.trace.target.implicit_h_counts[0] == 1
    with pytest.raises(TraceCompilationError, match="supplied chemistry"):
        compiler(
            source,
            graph("N(C=O)(C)C"),
            {0: 0, 1: 1, 2: 2},
            attachment_slots=(0,),
            minimum_h_counts=((0, 1),),
        )


def test_bound_prepared_live_gradients_sampler_and_forbidden_teacher(model):  # noqa: F811
    state = graph("N(C=O)C")
    base = ScaffoldContext.from_source(graph("NC=O"), (0,))
    bound = replace(base, minimum_h_counts=((0, 1),))
    carbon = int(graph("C").atom_types[0])
    forbidden = AtomInsert(4, carbon, 0, 3, ((0, 1),))
    assert torch.isfinite(
        model.forward_mark_batch(
            batch_for(model, state, base, forbidden, "atom_insert")
        ).selected_mark_log_probability
    ).all()
    assert torch.isneginf(
        model.forward_mark_batch(
            batch_for(model, state, bound, forbidden, "atom_insert")
        ).selected_mark_log_probability
    ).all()
    batch = batch_for(model, state, bound, AtomDelete(3), "atom_delete")
    prepared = prepare_scaffold_mark_batch(model, batch)
    buffer = io.BytesIO()
    torch.save(prepared, buffer)
    buffer.seek(0)
    replay = torch.load(buffer, weights_only=False).to(torch.device("cpu")).subbatch(0, 1)
    values, grads = [], []
    for item in (batch, replay):
        model.zero_grad(set_to_none=True)
        out = model.forward_mark_batch(item)
        (-out.selected_mark_log_probability.sum() + out.total_hazard.sum()).backward()
        values.append(out)
        grads.append(
            {n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}
        )
    for name in (
        "total_hazard",
        "family_log_probabilities",
        "selected_mark_log_probability",
        "enabled_families",
    ):
        torch.testing.assert_close(
            getattr(values[0], name), getattr(values[1], name), rtol=0, atol=0
        )
    for name, left in grads[0].items():
        if left is None:
            assert grads[1][name] is None
        else:
            torch.testing.assert_close(left, grads[1][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="state/context binding"):
        model.forward_mark_batch(replace(prepared, scaffold_contexts=(base,)))
    with pytest.raises(ValueError, match="hydrogen reserve"):
        model.forward_mark_batch(replace(batch, scaffold_min_h_counts=None))
    with pytest.raises(ValueError, match="hydrogen reserve"):
        model.forward_mark_batch(
            replace(batch, scaffold_min_h_counts=torch.zeros_like(batch.scaffold_min_h_counts))
        )
    rng = np.random.default_rng(929)
    with torch.no_grad():
        for context in (base, bound, base, bound):
            expected = model.forward_mark_batch(batch_for(model, state, context))
            for _ in range(16):
                mark = model.sample_rewrite_mark_conditioned(
                    state, 0.4, rng, property_values=None, scaffold_context=context
                )
                after = context.rewrite_system().apply(state, mark.rule_name, mark.action)
                assert context.accepts(after)
                assert mark.total_hazard == pytest.approx(expected.total_hazard.item(), rel=1e-6)
                assert torch.isfinite(
                    model.forward_mark_batch(
                        batch_for(model, state, context, mark.action, mark.rule_name)
                    ).selected_mark_log_probability
                ).all()


def test_ring_decoder_prunes_reserve_before_prefix_normalization():
    state = graph("NCC")
    base = ScaffoldContext.from_source(graph("N"), (0,))
    one = replace(base, minimum_h_counts=((0, 1),))
    two = replace(base, minimum_h_counts=((0, 2),))
    unrestricted, allowed, forbidden = (
        build_semantic_ring_system_decoder(state, placement(state), scaffold_context=c)
        for c in (base, one, two)
    )
    assert language(allowed) == language(unrestricted)
    assert language(allowed)
    assert not language(forbidden)


def test_ring_restates_filter_reserved_hydrogen_and_reject_forbidden_teacher(model):  # noqa: F811
    state = graph("C1CCCCC1")
    base = ScaffoldContext.from_source(graph("C"), (0,))
    bound = replace(base, minimum_h_counts=((0, 2),))
    before = batch_for(model, state, base).ring_restate_actions[0]
    after = batch_for(model, state, bound).ring_restate_actions[0]
    expected, rejected = [], []
    for action in before:
        successor = base.rewrite_system().apply(state, "ring_system_restate", action)
        (expected if successor.implicit_h_counts[0] >= 2 else rejected).append(action)
    assert tuple(expected) == after and rejected
    with pytest.raises(RuntimeError, match="outside exact dynamic candidates"):
        model.forward_mark_batch(batch_for(model, state, bound, rejected[0], "ring_system_restate"))


def test_reserve_slot_permutation_preserves_rate_and_candidate_probability(model):  # noqa: F811
    state = graph("N(C=O)C")
    context = ScaffoldContext.from_source(graph("NC=O"), (0,), minimum_h_counts=((0, 1),))
    p = np.random.default_rng(999).permutation(8)
    index = np.argsort(p)
    moved = type(state)(
        state.atom_types[index],
        state.formal_charges[index],
        state.implicit_h_counts[index],
        state.bonds[np.ix_(index, index)],
    )
    before = model.forward_mark_batch(
        batch_for(model, state, context, AtomDelete(3), "atom_delete")
    )
    after = model.forward_mark_batch(
        batch_for(model, moved, context.permuted(p), AtomDelete(int(p[3])), "atom_delete")
    )
    for name in ("total_hazard", "selected_mark_log_probability"):
        torch.testing.assert_close(
            getattr(before, name), getattr(after, name), rtol=1e-5, atol=1e-6
        )
