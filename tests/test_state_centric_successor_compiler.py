"""State-centric successor compilation is exact, bounded, and fail-closed."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.packed_trace_store import (
    PackedTraceAddress,
    PackedTraceProgress,
)
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheRecord,
)
from compose_v4.experiments import successor_fiber_cache_builder as builder
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.factorized_successor_training import (
    compile_state_productive_support,
    compile_teacher_successor_fiber,
)
from compose_v4.experiments.production_successor_kernel import (
    canonical_successor_result,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.operators import AtomInsert, BondInsert
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SLOTS = 8
_PACKED_SHA256 = "7" * 64
_SYSTEM = de_novo_rewrite_system()


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), _SLOTS)


def _address(
    *,
    entry_index: int,
    trace_id: str,
    source,
    target,
    path_length: int,
) -> PackedTraceAddress:
    return PackedTraceAddress(
        packed_shard_content_sha256=_PACKED_SHA256,
        packed_shard_name="state-centric-fixture.jsonl.gz",
        entry_index=entry_index,
        trace_id=trace_id,
        layer="corruption",
        partition="validation",
        source_key=canonical_state_key(source),
        target_key=canonical_state_key(target),
        path_length=path_length,
    )


def _record(
    *,
    entry_index: int,
    source,
    successor,
    step: RewriteStep,
) -> PathRecord:
    trace = RewriteTrace(
        source=source,
        target=successor,
        steps=(step,),
        metadata={},
    )
    path = TraceProgressCTMC(trace, system=_SYSTEM)
    address = _address(
        entry_index=entry_index,
        trace_id=f"state-centric-{entry_index}",
        source=source,
        target=successor,
        path_length=1,
    )
    return PathRecord(
        target_key=address.target_key,
        path=path,
        corpus_address=address,
    )


@pytest.fixture(scope="module")
def compiler_fixture():
    ring_target = _state("c1ccccc1")
    ring_source = DegreeBoundedCarbonTreePrior(
        sizes=(ring_target.n_real_atoms,)
    ).sample(np.random.default_rng(8), n_slots=_SLOTS)
    catalog_trace = compile_carbon_tree_to_target(
        ring_source,
        ring_target,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((catalog_trace,))
    torch.manual_seed(23)
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()

    source = _state("C")
    result = canonical_successor_result(model, source, 0.2)
    representatives = {}
    for mark in result.marked_law.marks:
        successor = _SYSTEM.apply(
            source,
            mark.executor_rule_name,
            mark.action,
        )
        target_key = canonical_state_key(successor)
        if target_key == result.batch.source_key or successor.n_real_atoms == 0:
            continue
        representatives.setdefault(
            target_key,
            (
                successor,
                RewriteStep(mark.executor_rule_name, mark.action),
            ),
        )
    assert len(representatives) >= 2
    selected = [
        representatives[target_key]
        for target_key in sorted(representatives)[:2]
    ]
    records = tuple(
        _record(
            entry_index=index,
            source=source,
            successor=successor,
            step=step,
        )
        for index, (successor, step) in enumerate(selected)
    )
    return model, records


def _record_by_record_reference(
    model: FactorizedTraceletRateModel,
    records: tuple[PathRecord, ...],
    *,
    time: float,
) -> tuple[SuccessorFiberCacheRecord, ...]:
    """The pre-optimization per-progress projection contract."""

    rows = []
    for record in records:
        assert record.corpus_address is not None
        for progress_index in range(record.path.path_length + 1):
            source = record.path.state_at(progress_index)
            address = SuccessorFiberCacheAddress.from_packed_trace(
                record.corpus_address,
                progress_index=progress_index,
            )
            if progress_index == record.path.path_length:
                support = compile_state_productive_support(
                    model,
                    source,
                    time=time,
                )
                fiber = None
            else:
                fiber = compile_teacher_successor_fiber(
                    model,
                    source,
                    record.path.state_at(progress_index + 1),
                    time=time,
                )
                support = fiber.state_support
            rows.append(
                SuccessorFiberCacheRecord(
                    address=address,
                    state_support=support,
                    teacher_fiber=fiber,
                )
            )
    return tuple(rows)


def test_shard_output_matches_record_by_record_compilation(
    compiler_fixture,
) -> None:
    model, records = compiler_fixture
    expected = _record_by_record_reference(model, records, time=0.2)

    observed = builder.compile_successor_fiber_shard(
        model,
        records,
        expected_packed_entry_count=2,
        time=0.2,
    )

    assert observed == expected


def test_shard_compiles_each_unique_exact_state_once_across_targets(
    compiler_fixture,
    monkeypatch,
) -> None:
    model, records = compiler_fixture
    calls = Counter()
    compile_state = builder.compile_state_successor_map

    def counted_compile_state(model, source, **kwargs):
        calls[persistent_slot_state_sha256(source)] += 1
        return compile_state(model, source, **kwargs)

    monkeypatch.setattr(
        builder,
        "compile_state_successor_map",
        counted_compile_state,
    )
    rows = builder.compile_successor_fiber_shard(
        model,
        records,
        expected_packed_entry_count=2,
        time=0.2,
    )

    expected_digests = {
        persistent_slot_state_sha256(
            record.path.state_at(progress_index)
        )
        for record in records
        for progress_index in range(record.path.path_length + 1)
    }
    assert calls == Counter({digest: 1 for digest in expected_digests})
    assert rows[0].source_state_sha256 == rows[2].source_state_sha256
    assert rows[0].target_state_sha256 != rows[2].target_state_sha256
    assert rows[0].state_support == rows[2].state_support


def test_builder_rejects_address_source_mismatch(compiler_fixture) -> None:
    model, records = compiler_fixture
    record = records[0]
    assert record.corpus_address is not None
    mismatched = replace(
        record,
        corpus_address=replace(
            record.corpus_address,
            source_key="not-the-trace-source",
        ),
    )

    with pytest.raises(
        builder.SuccessorFiberCacheBuildError,
        match="address source key disagrees",
    ):
        builder.compile_successor_fiber_trace(model, mismatched, time=0.2)


def test_builder_rejects_packed_trace_source_mismatch(
    compiler_fixture,
) -> None:
    model, records = compiler_fixture
    record = records[0]
    actual_source = record.path.state_at(0)
    actual_target = record.path.state_at(1)
    mismatched_trace = RewriteTrace(
        source=_state("N"),
        target=actual_target,
        steps=record.path.trace.steps,
        metadata={},
    )
    mismatched_path = PackedTraceProgress(
        mismatched_trace,
        (actual_source, actual_target),
    )
    mismatched_record = replace(record, path=mismatched_path)

    with pytest.raises(
        builder.SuccessorFiberCacheBuildError,
        match="trace source and packed progress source disagree",
    ):
        builder.compile_successor_fiber_trace(
            model,
            mismatched_record,
            time=0.2,
        )


def test_builder_rejects_tampered_step_when_target_remains_reachable(
    compiler_fixture,
) -> None:
    model, records = compiler_fixture
    record, alternative = records
    source = record.path.state_at(0)
    reachable_target = record.path.state_at(1)
    wrong_step = alternative.path.trace.steps[0]
    wrong_target = _SYSTEM.apply(
        source,
        wrong_step.rule_name,
        wrong_step.action,
    )
    assert (
        persistent_slot_state_sha256(wrong_target)
        != persistent_slot_state_sha256(reachable_target)
    )
    tampered_trace = RewriteTrace(
        source=source,
        target=reachable_target,
        steps=(wrong_step,),
        metadata={},
    )
    tampered_path = PackedTraceProgress(
        tampered_trace,
        (source, reachable_target),
    )
    tampered_record = replace(record, path=tampered_path)

    with pytest.raises(
        builder.SuccessorFiberCacheBuildError,
        match="teacher step does not execute to the stored exact next state",
    ):
        builder.compile_successor_fiber_trace(
            model,
            tampered_record,
            time=0.2,
        )


def test_builder_rejects_unsupported_operand_alias_of_supported_successor(
    compiler_fixture,
) -> None:
    model, _records = compiler_fixture
    source = _state("CCC")
    supported_action = BondInsert(a=0, b=2, order=1)
    unsupported_alias = BondInsert(a=2, b=0, order=1)
    target = _SYSTEM.apply(source, "bond_insert", supported_action)
    assert persistent_slot_state_sha256(
        _SYSTEM.apply(source, "bond_insert", unsupported_alias)
    ) == persistent_slot_state_sha256(target)

    trace = RewriteTrace(
        source=source,
        target=target,
        steps=(RewriteStep("bond_insert", unsupported_alias),),
        metadata={},
    )
    path = TraceProgressCTMC(trace, system=_SYSTEM)
    address = _address(
        entry_index=0,
        trace_id="unsupported-operand-alias",
        source=source,
        target=target,
        path_length=1,
    )
    record = PathRecord(
        target_key=address.target_key,
        path=path,
        corpus_address=address,
    )

    with pytest.raises(
        builder.SuccessorFiberCacheBuildError,
        match="teacher action is absent from the exact target marked support",
    ):
        builder.compile_successor_fiber_trace(model, record, time=0.2)


def test_builder_rejects_exact_target_absent_from_support(
    compiler_fixture,
) -> None:
    model, _records = compiler_fixture
    source = _state("CC")
    unsupported_insert = AtomInsert(
        slot=2,
        atom_type=ELEMENT_TO_IDX["C"],
        formal_charge=0,
        implicit_h_count=2,
        neighbors=((0, 1), (1, 1)),
    )
    absent_target = _SYSTEM.apply(
        source,
        "atom_insert",
        unsupported_insert,
    )
    trace = RewriteTrace(
        source=source,
        target=absent_target,
        steps=(RewriteStep("atom_insert", unsupported_insert),),
        metadata={},
    )
    path = TraceProgressCTMC(trace, system=_SYSTEM)
    address = _address(
        entry_index=0,
        trace_id="absent-target",
        source=source,
        target=absent_target,
        path_length=1,
    )
    absent_record = PathRecord(
        target_key=address.target_key,
        path=path,
        corpus_address=address,
    )

    with pytest.raises(
        builder.SuccessorFiberCacheBuildError,
        match="exact trace target is absent",
    ):
        builder.compile_successor_fiber_trace(
            model,
            absent_record,
            time=0.2,
        )
