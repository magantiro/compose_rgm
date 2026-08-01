"""Whole-trace migration gates from historical actions to semantic Active8."""

from __future__ import annotations

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.operators import AtomInsert, BondInsert, CycleCloseEdge
from compose_v4.rewrite.semantic_trace_migration import (
    SemanticTraceMigrationRejectionCode,
    migrate_and_encode_semantic_trace,
    migrate_legacy_trace_to_editing_v2,
)
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard_v3 import decode_semantic_trace_record


def _one_step_trace(source, rule_name: str, action) -> RewriteTrace:
    target = de_novo_rewrite_system().apply(source, rule_name, action)
    return RewriteTrace(
        source=source,
        target=target,
        steps=(RewriteStep(rule_name, action),),
        metadata={"legacy": True},
    )


def test_raw_bond_insert_migrates_to_semantic_cycle_close_and_exact_v3() -> None:
    source = pad_molecular_graph(smiles_to_molecular_graph("CCCCCC"), 16)
    trace = _one_step_trace(source, "bond_insert", BondInsert(0, 5, 1))
    result = migrate_legacy_trace_to_editing_v2(trace)
    assert result.admitted
    assert result.rejection is None
    assert result.trace is not None
    assert result.trace.steps == (
        RewriteStep("cycle_close", CycleCloseEdge(0, 5, 1)),
    )

    record, rejection = migrate_and_encode_semantic_trace(
        trace,
        trace_id="legacy-close",
        data_lane="operator_aware_real_endpoint",
        split="development",
        source_address={"shard": "fixture", "row": 0},
        lineage={"source_schema_version": 2},
    )
    assert rejection is None
    assert record is not None
    decoded = decode_semantic_trace_record(record)
    assert decoded.steps == result.trace.steps


def test_multi_neighbor_birth_rejects_the_complete_trace() -> None:
    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 4)
    action = AtomInsert(
        slot=2,
        atom_type=2,
        formal_charge=0,
        implicit_h_count=2,
        neighbors=((0, 1), (1, 1)),
    )
    trace = _one_step_trace(source, "atom_insert", action)
    result = migrate_legacy_trace_to_editing_v2(trace)
    assert not result.admitted
    assert result.trace is None
    assert result.rejection is not None
    assert (
        result.rejection.code
        == SemanticTraceMigrationRejectionCode.SEMANTIC_ACTION_REJECTED
    )
    assert result.rejection.step_index == 0


def test_disabled_middle_rule_rejects_without_partial_output() -> None:
    source = pad_molecular_graph(smiles_to_molecular_graph("CCCCCC"), 16)
    first = RewriteStep("bond_insert", BondInsert(0, 5, 1))
    intermediate = de_novo_rewrite_system().apply(
        source,
        first.rule_name,
        first.action,
    )
    # The historical raw deletion is executable but its public macro name is
    # outside Active8. This deliberately tests atomic whole-trace rejection.
    from compose_v4.rewrite.tracelets import CycleDelete

    second = RewriteStep("cycle_delete", CycleDelete((0, 1, 2, 3, 4, 5)))
    target = de_novo_rewrite_system().apply(
        intermediate,
        second.rule_name,
        second.action,
    )
    trace = RewriteTrace(source, target, (first, second), {})
    result = migrate_legacy_trace_to_editing_v2(trace)
    assert not result.admitted
    assert result.trace is None
    assert result.rejection is not None
    assert (
        result.rejection.code
        == SemanticTraceMigrationRejectionCode.DISABLED_OR_UNKNOWN_RULE
    )
    assert result.rejection.step_index == 1
