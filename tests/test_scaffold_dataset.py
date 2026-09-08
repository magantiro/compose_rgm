"""Source conditioning preserves the existing index-deterministic CTMC clock."""

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import empty_molecular_graph, pad_molecular_graph
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.factorized_mark_conditional import FactorizedMarkDataset
from compose_v4.model.factorized_tracelet_rate_model import molecular_state_cache_key
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.scaffold_construction import compile_scaffold_to_target_tracelets


def records():
    source = pad_molecular_graph(smiles_to_molecular_graph("N"), 8)
    target = pad_molecular_graph(smiles_to_molecular_graph("NCC"), 8)
    empty = compile_scaffold_to_target_tracelets(empty_molecular_graph(8), target, {})
    completion = compile_scaffold_to_target_tracelets(source, target, {0: 0}, attachment_slots=(0,))
    return tuple(
        PathRecord("NCC", TraceProgressCTMC(item.trace, system=item.context.rewrite_system()))
        for item in (empty, completion)
    ), (empty.context, completion.context)


def dataset(items, **kwargs):
    defaults = {
        "start_index": 0,
        "length": 32,
        "seed": 911,
        "late_time_fraction": 0.5,
        "operational_horizon": 16.0,
        "progress_stratification_fraction": 0.5,
    }
    return FactorizedMarkDataset(items, **{**defaults, **kwargs})


def assert_same_draw(left, right):
    assert molecular_state_cache_key(left.state) == molecular_state_cache_key(right.state)
    for name in (
        "time",
        "teacher_action",
        "teacher_rule_name",
        "teacher_rate",
        "importance_weight",
        "record_index",
        "progress_index",
    ):
        assert getattr(left, name) == getattr(right, name)


def test_context_transport_does_not_change_clock_teacher_weights_or_resume_stream():
    items, contexts = records()
    original = dataset(items)
    conditioned = dataset(items, record_scaffold_contexts=contexts)
    resumed = dataset(items, record_scaffold_contexts=contexts, start_index=16, length=16)
    draws = [conditioned[index] for index in range(32)]
    assert {row.record_index for row in draws} == {0, 1}
    assert any(row.teacher_rate == 0 for row in draws)
    assert any(row.teacher_rate > 0 for row in draws)
    for index in reversed(range(32)):
        row = conditioned[index]
        assert_same_draw(original[index], row)
        assert_same_draw(draws[index], row)
        assert row.scaffold_context == contexts[row.record_index]
        assert row.scaffold_context.accepts(row.state)
        assert row.ring_teacher_semantic_certificate is None
        if index >= 16:
            assert_same_draw(row, resumed[index - 16])
            assert row.scaffold_context == resumed[index - 16].scaffold_context
    assert conditioned._ring_support_model is None


def test_unaligned_conditions_and_legacy_support_shortcuts_are_rejected():
    items, contexts = records()
    with pytest.raises(ValueError, match="align"):
        dataset(items, record_scaffold_contexts=contexts[:1])
    with pytest.raises(ValueError, match="align"):
        dataset(items, record_scaffold_contexts=(contexts[1], contexts[0]))
    with pytest.raises(ValueError, match="state-only"):
        dataset(items, record_scaffold_contexts=contexts, ring_catalog=object())
