"""Successor batches join the existing deterministic stream without resampling."""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass, replace

import pytest
import torch
from torch.utils.data import DataLoader

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheRecord,
)
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.editing_training_sentinel import (
    FAMILY_PARAMETER_PREFIXES,
    EditingTrainingSentinelError,
    P50GradientCollapseSentinel,
    P50LiveExposureObserver,
    assert_successor_exposure_matches,
    audit_successor_exposure_plan,
)
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkDataset,
    train_factorized_mark_model,
)
from compose_v4.experiments.factorized_successor_data import (
    FactorizedSuccessorBatch,
    FactorizedSuccessorCollator,
    FactorizedSuccessorDataError,
    FactorizedSuccessorDataset,
    assert_successor_wrapper_is_rng_neutral,
    factorized_successor_loader,
)
from compose_v4.experiments.factorized_successor_objective import (
    CanonicalSuccessorTrainingObjective,
    factorized_successor_metrics,
)
from compose_v4.experiments.factorized_successor_training import (
    SuccessorTrainingError,
)
from compose_v4.experiments.successor_fiber_cache_builder import (
    SuccessorFiberCacheBuildError,
    compile_successor_fiber_shard,
    compile_successor_fiber_trace,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_TO_INDEX,
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_PACKED_SHA256 = "1" * 64


@dataclass
class _ExactCacheStub:
    records: dict[tuple[str, int, int], SuccessorFiberCacheRecord]
    calls: int = 0

    def require(self, address, *, progress_index, source_state):
        self.calls += 1
        key = (
            address.packed_shard_content_sha256,
            address.entry_index,
            progress_index,
        )
        record = self.records[key]
        assert record.address == SuccessorFiberCacheAddress.from_packed_trace(
            address,
            progress_index=progress_index,
        )
        if (
            record.source_state_sha256
            != persistent_slot_state_sha256(source_state)
        ):
            raise RuntimeError("sampled exact state mismatch")
        return record


def _identity_collate(examples):
    return examples


@pytest.fixture(scope="module")
def fixture_bundle():
    source = pad_molecular_graph(smiles_to_molecular_graph("C"), 6)
    target = pad_molecular_graph(smiles_to_molecular_graph("CC"), 6)
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
        flexible_size=True,
    )
    path = TraceProgressCTMC(trace)
    address = PackedTraceAddress(
        packed_shard_content_sha256=_PACKED_SHA256,
        packed_shard_name="tiny.jsonl.gz",
        entry_index=0,
        trace_id="tiny-0",
        layer="corruption",
        partition="train",
        source_key=canonical_state_key(path.state_at(0)),
        target_key=canonical_state_key(path.state_at(path.path_length)),
        path_length=path.path_length,
    )
    record = PathRecord(
        target_key=address.target_key,
        path=path,
        corpus_address=address,
    )
    catalog = build_typed_ring_catalog((trace,))
    torch.manual_seed(7)
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
    cache_records = compile_successor_fiber_trace(
        model,
        record,
        time=0.2,
    )
    return record, catalog, model, cache_records


def _mark_dataset(record: PathRecord, *, start_index: int = 0, length: int = 32):
    return FactorizedMarkDataset(
        (record,),
        start_index=start_index,
        length=length,
        seed=31,
        late_time_fraction=0.5,
        operational_horizon=2.0,
        progress_stratification_fraction=0.5,
    )


def _successor_dataset(record, cache_records, **kwargs):
    mark_dataset = _mark_dataset(record, **kwargs)
    cache = _ExactCacheStub(
        {
            (
                row.address.packed_shard_content_sha256,
                row.address.entry_index,
                row.address.progress_index,
            ): row
            for row in cache_records
        }
    )
    return mark_dataset, cache, FactorizedSuccessorDataset(
        mark_dataset,
        cache,
    )


def _mark_collator(catalog, model) -> FactorizedMarkCollator:
    capabilities = model.operator_capabilities
    return FactorizedMarkCollator(
        True,
        catalog,
        compute_ring_grow_support=capabilities.compute_ring_grow_support,
        compute_ring_restates=capabilities.compute_ring_restates,
        compute_cyclic_graft=capabilities.compute_cyclic_graft,
        compute_ring_opening=capabilities.compute_ring_opening,
    )


def _semantic_cells(cache_records):
    return {
        (
            row.address.packed_shard_content_sha256,
            row.address.entry_index,
            row.address.progress_index,
        ): (
            None
            if row.address.is_terminal
            else f"progress-{row.address.progress_index}"
        )
        for row in cache_records
    }


def _validation_batch(record, catalog, model, cache_records, *, length=16):
    mark_dataset = _mark_dataset(record, length=length)
    cache = _ExactCacheStub(
        {
            (
                row.address.packed_shard_content_sha256,
                row.address.entry_index,
                row.address.progress_index,
            ): row
            for row in cache_records
        }
    )
    dataset = FactorizedSuccessorDataset(
        mark_dataset,
        cache,
        semantic_cell_ids=_semantic_cells(cache_records),
        require_semantic_cell_ids=True,
    )
    return FactorizedSuccessorCollator(_mark_collator(catalog, model))(
        [dataset[index] for index in range(length)]
    )


def test_bounded_builder_compiles_complete_exact_chain(fixture_bundle) -> None:
    record, _catalog, model, cache_records = fixture_bundle
    assert len(cache_records) == record.path.path_length + 1
    assert cache_records[-1].address.is_terminal
    assert cache_records[-1].teacher_fiber is None
    for progress, row in enumerate(cache_records):
        assert row.address.progress_index == progress
        assert row.source_state_sha256 == persistent_slot_state_sha256(
            record.path.state_at(progress)
        )
        if progress < record.path.path_length:
            assert row.teacher_fiber is not None
            assert row.target_state_sha256 == persistent_slot_state_sha256(
                record.path.state_at(progress + 1)
            )

    # The bounded correctness builder must yield identical support at a
    # different time; it never stores the model probabilities it traversed.
    assert compile_successor_fiber_trace(
        model,
        record,
        time=0.8,
    ) == cache_records


def test_wrapper_preserves_draw_and_resume_stream(fixture_bundle) -> None:
    record, _catalog, _model, cache_records = fixture_bundle
    mark_dataset, cache, successor_dataset = _successor_dataset(
        record,
        cache_records,
        length=32,
    )
    assert_successor_wrapper_is_rng_neutral(
        mark_dataset,
        successor_dataset,
        indices=tuple(range(32)),
    )
    assert cache.calls == 32

    resumed_mark, _resumed_cache, resumed_successor = _successor_dataset(
        record,
        cache_records,
        start_index=16,
        length=16,
    )
    for local_index in range(16):
        complete = successor_dataset[16 + local_index].mark_example
        resumed = resumed_successor[local_index].mark_example
        assert complete == resumed_mark[local_index]
        assert complete == resumed


def test_wrapper_requires_addresses_cells_and_exact_next_state(
    fixture_bundle,
) -> None:
    record, _catalog, _model, cache_records = fixture_bundle
    with pytest.raises(FactorizedSuccessorDataError, match="without immutable"):
        FactorizedSuccessorDataset(
            _mark_dataset(replace(record, corpus_address=None)),
            _ExactCacheStub({}),
        )

    mark_dataset, cache, _ = _successor_dataset(
        record,
        cache_records,
        length=8,
    )
    with pytest.raises(
        FactorizedSuccessorDataError,
        match="semantic-cell sidecar",
    ):
        FactorizedSuccessorDataset(
            mark_dataset,
            cache,
            semantic_cell_ids={},
            require_semantic_cell_ids=True,
        )[0]

    jump_progress = next(
        row.address.progress_index
        for row in cache_records
        if row.teacher_fiber is not None
    )
    original = cache_records[jump_progress]
    broken_fiber = replace(
        original.teacher_fiber,
        target_state_sha256="f" * 64,
    )
    broken_record = replace(
        original,
        teacher_fiber=broken_fiber,
    )
    broken_rows = tuple(
        broken_record if row.address.progress_index == jump_progress else row
        for row in cache_records
    )
    mark_dataset = _mark_dataset(record, length=256)
    broken_cache = _ExactCacheStub(
        {
            (
                row.address.packed_shard_content_sha256,
                row.address.entry_index,
                row.address.progress_index,
            ): row
            for row in broken_rows
        }
    )
    dataset = FactorizedSuccessorDataset(mark_dataset, broken_cache)
    matching_index = next(
        index
        for index in range(len(mark_dataset))
        if mark_dataset[index].progress_index == jump_progress
    )
    with pytest.raises(
        FactorizedSuccessorDataError,
        match="exact next path state",
    ):
        dataset[matching_index]


def test_semantic_sidecar_distinguishes_missing_from_explicit_terminal_null(
    fixture_bundle,
) -> None:
    record, _catalog, _model, cache_records = fixture_bundle
    mark_dataset = _mark_dataset(record, length=256)
    cache = _ExactCacheStub(
        {
            (
                row.address.packed_shard_content_sha256,
                row.address.entry_index,
                row.address.progress_index,
            ): row
            for row in cache_records
        }
    )
    cells = _semantic_cells(cache_records)
    terminal_progress = record.path.path_length
    terminal_index = next(
        index
        for index in range(len(mark_dataset))
        if mark_dataset[index].progress_index == terminal_progress
    )
    explicit = FactorizedSuccessorDataset(
        mark_dataset,
        cache,
        semantic_cell_ids=cells,
        require_semantic_cell_ids=True,
    )[terminal_index]
    assert explicit.semantic_cell_id is None

    terminal_key = next(
        key for key in cells if key[2] == terminal_progress
    )
    missing_terminal = dict(cells)
    missing_terminal.pop(terminal_key)
    with pytest.raises(
        FactorizedSuccessorDataError,
        match="absent from the frozen semantic-cell sidecar",
    ):
        FactorizedSuccessorDataset(
            mark_dataset,
            cache,
            semantic_cell_ids=missing_terminal,
            require_semantic_cell_ids=True,
        )[terminal_index]


def test_semantic_sidecar_enforces_terminal_and_nonterminal_value_shapes(
    fixture_bundle,
) -> None:
    record, _catalog, _model, cache_records = fixture_bundle
    mark_dataset = _mark_dataset(record, length=256)
    cache = _ExactCacheStub(
        {
            (
                row.address.packed_shard_content_sha256,
                row.address.entry_index,
                row.address.progress_index,
            ): row
            for row in cache_records
        }
    )
    cells = _semantic_cells(cache_records)
    terminal_progress = record.path.path_length
    terminal_index = next(
        index
        for index in range(len(mark_dataset))
        if mark_dataset[index].progress_index == terminal_progress
    )
    nonterminal_index = next(
        index
        for index in range(len(mark_dataset))
        if mark_dataset[index].progress_index < terminal_progress
    )
    terminal_key = next(
        key for key in cells if key[2] == terminal_progress
    )
    nonterminal_progress = mark_dataset[nonterminal_index].progress_index
    nonterminal_key = next(
        key for key in cells if key[2] == nonterminal_progress
    )

    wrong_terminal = dict(cells)
    wrong_terminal[terminal_key] = "not-null"
    with pytest.raises(
        FactorizedSuccessorDataError,
        match="terminal sampled row must carry an explicit null",
    ):
        FactorizedSuccessorDataset(
            mark_dataset,
            cache,
            semantic_cell_ids=wrong_terminal,
            require_semantic_cell_ids=True,
        )[terminal_index]

    wrong_nonterminal = dict(cells)
    wrong_nonterminal[nonterminal_key] = None
    with pytest.raises(
        FactorizedSuccessorDataError,
        match="nonterminal sampled row must carry a nonempty",
    ):
        FactorizedSuccessorDataset(
            mark_dataset,
            cache,
            semantic_cell_ids=wrong_nonterminal,
            require_semantic_cell_ids=True,
        )[nonterminal_index]


def test_collator_batch_alignment_subbatch_move_and_pin(
    fixture_bundle,
    monkeypatch,
) -> None:
    record, catalog, model, cache_records = fixture_bundle
    _mark, _cache, successor_dataset = _successor_dataset(
        record,
        cache_records,
        length=32,
    )
    examples = [successor_dataset[index] for index in range(8)]
    collator = FactorizedSuccessorCollator(
        _mark_collator(catalog, model)
    )
    batch = collator(examples)
    assert isinstance(batch, FactorizedSuccessorBatch)
    assert batch.batch_size == 8
    assert batch.subbatch(2, 6).cache_addresses == batch.cache_addresses[2:6]
    moved = batch.to(torch.device("cpu"))
    assert moved.cache_addresses is batch.cache_addresses
    assert torch.equal(moved.mark_batch.atom_types, batch.mark_batch.atom_types)

    monkeypatch.setattr(
        FactorizedMarkBatch,
        "pin_memory",
        lambda self: self,
    )
    pinned = batch.pin_memory()
    assert pinned.mark_batch is batch.mark_batch
    assert pinned.fibers is batch.fibers


def test_dataset_is_pickle_and_multiworker_safe(fixture_bundle) -> None:
    record, _catalog, _model, cache_records = fixture_bundle
    _mark, _cache, successor_dataset = _successor_dataset(
        record,
        cache_records,
        length=8,
    )
    restored = pickle.loads(pickle.dumps(successor_dataset))
    restored_example = restored[0].mark_example
    original_example = successor_dataset[0].mark_example
    assert persistent_slot_state_sha256(
        restored_example.state
    ) == persistent_slot_state_sha256(original_example.state)
    assert restored_example.time == original_example.time
    assert restored_example.teacher_action == original_example.teacher_action
    assert (
        restored_example.teacher_rule_name
        == original_example.teacher_rule_name
    )
    assert restored_example.record_index == original_example.record_index
    assert restored_example.progress_index == original_example.progress_index

    loader = DataLoader(
        successor_dataset,
        batch_size=4,
        num_workers=2,
        collate_fn=_identity_collate,
    )
    rows = next(iter(loader))
    assert len(rows) == 4
    assert all(row.cache_record.address.entry_index == 0 for row in rows)


def test_successor_objective_metrics_microbatch_and_gradients(
    fixture_bundle,
) -> None:
    record, catalog, model, cache_records = fixture_bundle
    batch = _validation_batch(record, catalog, model, cache_records)
    full = factorized_successor_metrics(
        model,
        batch,
        use_bf16=False,
    )
    streamed = factorized_successor_metrics(
        model,
        batch,
        use_bf16=False,
        microbatch_size=3,
    )
    for metric in (
        "production_weighted_canonical_successor_nll",
        "balanced_semantic_cell_canonical_successor_nll",
        "successor_generator_bregman_loss",
        "hazard_bregman_loss",
        "mean_teacher_productive_successor_probability",
        "teacher_family_nll_atom_insert",
        "within_family_successor_nll_atom_insert",
    ):
        assert streamed[metric] == pytest.approx(full[metric], abs=2e-6)
    assert full["teacher_family_nll_atom_insert"] >= 0.0
    assert full["within_family_successor_nll_atom_insert"] >= 0.0

    trainable = copy.deepcopy(model).train()
    objective = CanonicalSuccessorTrainingObjective(
        mode="productive_identity_plus_hazard",
        hazard_weight=0.1,
    )
    trainable.zero_grad(set_to_none=True)
    loss = objective.loss(trainable, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in trainable.parameters()
    )

    missing_cells = replace(
        batch,
        semantic_cell_ids=(None,) * batch.batch_size,
    )
    with pytest.raises(SuccessorTrainingError, match="semantic cell"):
        factorized_successor_metrics(
            model,
            missing_cells,
            use_bf16=False,
        )


def test_productive_identity_objective_has_no_hazard_gradient(
    fixture_bundle,
) -> None:
    record, catalog, template, cache_records = fixture_bundle
    batch = _validation_batch(record, catalog, template, cache_records)
    model = copy.deepcopy(template).train()
    objective = CanonicalSuccessorTrainingObjective(
        mode="productive_identity",
        hazard_weight=0.0,
    )

    model.zero_grad(set_to_none=True)
    loss = objective.loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()

    hazard_parameters = tuple(model.total_hazard_head.parameters())
    assert hazard_parameters
    assert all(parameter.grad is None for parameter in hazard_parameters)
    assert any(
        parameter.grad is not None
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.family_head.parameters()
    )


def test_exposure_preflight_and_p50_gradient_sentinel(fixture_bundle) -> None:
    record, catalog, template, cache_records = fixture_bundle
    cache = _ExactCacheStub(
        {
            (
                row.address.packed_shard_content_sha256,
                row.address.entry_index,
                row.address.progress_index,
            ): row
            for row in cache_records
        }
    )
    mark_dataset = _mark_dataset(record, length=16)
    loader = factorized_successor_loader(
        mark_dataset,
        cache,
        _mark_collator(catalog, template),
        batch_size=8,
        workers=0,
        pin_memory=False,
        seed=31,
        semantic_cell_ids=_semantic_cells(cache_records),
        require_semantic_cell_ids=True,
    )
    exposure = audit_successor_exposure_plan(
        loader,
        expected_steps=2,
        required_families=("atom_insert",),
        minimum_teacher_examples={"atom_insert": 1},
    )
    assert exposure.teacher_examples_by_family["atom_insert"] > 0
    assert exposure.gradient_opportunities_by_family["atom_insert"] == 2
    assert exposure.nonterminal_examples_by_step[0] > 0
    assert len(exposure.ordered_address_stream_sha256) == 64
    assert len(exposure.ordered_training_stream_sha256) == 64
    assert exposure.examples_by_layer == {"corruption": 16}
    assert (
        exposure.terminal_example_count + exposure.nonterminal_example_count
        == exposure.total_examples
    )
    assert exposure.identity_importance_weight_by_family["atom_insert"] > 0.0
    assert (
        exposure.generator_teacher_coefficient_by_family["atom_insert"] > 0.0
    )

    matching_loader = factorized_successor_loader(
        _mark_dataset(record, length=16),
        cache,
        _mark_collator(catalog, template),
        batch_size=8,
        workers=0,
        pin_memory=False,
        seed=31,
        semantic_cell_ids=_semantic_cells(cache_records),
        require_semantic_cell_ids=True,
    )
    matching_exposure = audit_successor_exposure_plan(
        matching_loader,
        expected_steps=2,
        required_families=("atom_insert",),
        minimum_teacher_examples={"atom_insert": 1},
    )
    assert_successor_exposure_matches(exposure, matching_exposure)

    mutated_loader = factorized_successor_loader(
        _mark_dataset(record, length=16),
        cache,
        _mark_collator(catalog, template),
        batch_size=8,
        workers=0,
        pin_memory=False,
        seed=31,
        semantic_cell_ids=_semantic_cells(cache_records),
        require_semantic_cell_ids=True,
    )
    mutated_batches = list(mutated_loader)
    first_batch = mutated_batches[0]
    mutated_address = replace(
        first_batch.cache_addresses[0],
        packed_shard_name="mutated.jsonl.gz",
    )
    mutated_batches[0] = replace(
        first_batch,
        cache_addresses=(
            mutated_address,
            *first_batch.cache_addresses[1:],
        ),
    )
    mutated_exposure = audit_successor_exposure_plan(
        mutated_batches,
        expected_steps=2,
        required_families=("atom_insert",),
        minimum_teacher_examples={"atom_insert": 1},
    )
    with pytest.raises(
        EditingTrainingSentinelError,
        match="address stream differs",
    ):
        assert_successor_exposure_matches(exposure, mutated_exposure)

    starved_loader = factorized_successor_loader(
        _mark_dataset(record, length=16),
        cache,
        _mark_collator(catalog, template),
        batch_size=8,
        workers=0,
        pin_memory=False,
        seed=31,
        semantic_cell_ids=_semantic_cells(cache_records),
        require_semantic_cell_ids=True,
    )
    with pytest.raises(
        EditingTrainingSentinelError,
        match="starves required families",
    ):
        audit_successor_exposure_plan(
            starved_loader,
            expected_steps=2,
            required_families=("atom_insert", "atom_delete"),
            minimum_teacher_examples={
                "atom_insert": 1,
                "atom_delete": 1,
            },
        )
    gradient_starved_loader = factorized_successor_loader(
        _mark_dataset(record, length=16),
        cache,
        _mark_collator(catalog, template),
        batch_size=8,
        workers=0,
        pin_memory=False,
        seed=31,
        semantic_cell_ids=_semantic_cells(cache_records),
        require_semantic_cell_ids=True,
    )
    with pytest.raises(
        EditingTrainingSentinelError,
        match="gradient_opportunities",
    ):
        audit_successor_exposure_plan(
            gradient_starved_loader,
            expected_steps=2,
            required_families=("atom_insert",),
            minimum_gradient_opportunities={"atom_insert": 3},
        )

    validation = _validation_batch(
        record,
        catalog,
        template,
        cache_records,
    )
    objective = CanonicalSuccessorTrainingObjective(
        mode="productive_identity",
        hazard_weight=0.0,
    )
    baseline = objective.metrics(
        template,
        validation,
        use_bf16=False,
        microbatch_size=None,
    )
    sentinel = P50GradientCollapseSentinel(
        required_families=("atom_insert",),
        minimum_gradient_updates={"atom_insert": 1},
        maximum_family_nll_regression={"atom_insert": 0.0},
        baseline_validation_metrics=baseline,
    )
    model = copy.deepcopy(template).train()
    for step in range(1, 51):
        model.zero_grad(set_to_none=True)
        loss = objective.loss(model, validation)
        loss.backward()
        sentinel(model, validation.mark_batch, step, loss.detach())
        if step in {1, 50}:
            sentinel.observe_validation(
                completed_step=step,
                metrics=baseline,
            )
    report = sentinel.finalize()
    assert report["completed_optimizer_steps"] == 50
    assert report["gradient_updates_by_family"]["atom_insert"] == 50
    assert (
        report["family_gate_gradient_updates_by_family"]["atom_insert"] == 50
    )
    assert (
        report["action_route_gradient_updates_by_family"]["atom_insert"] == 50
    )
    assert report["final_validation_metrics"] == pytest.approx(baseline)
    assert report["final_family_nll_regressions"]["atom_insert"] == 0.0


def test_p50_live_observer_binds_actual_batches_to_preflight(fixture_bundle) -> None:
    record, catalog, template, cache_records = fixture_bundle
    cache = _ExactCacheStub(
        {
            (
                row.address.packed_shard_content_sha256,
                row.address.entry_index,
                row.address.progress_index,
            ): row
            for row in cache_records
        }
    )
    loader = factorized_successor_loader(
        _mark_dataset(record, length=16),
        cache,
        _mark_collator(catalog, template),
        batch_size=8,
        workers=0,
        pin_memory=False,
        seed=31,
        semantic_cell_ids=_semantic_cells(cache_records),
        require_semantic_cell_ids=True,
    )
    base_batches = tuple(loader)
    frozen_batches = base_batches * 25
    planned = audit_successor_exposure_plan(
        frozen_batches,
        expected_steps=50,
        required_families=("atom_insert",),
        minimum_gradient_opportunities={"atom_insert": 50},
    )
    observer = P50LiveExposureObserver(
        planned=planned,
        required_families=("atom_insert",),
        minimum_gradient_opportunities={"atom_insert": 50},
    )
    for completed_step, batch in enumerate(frozen_batches, start=1):
        observer(batch, completed_step)
    assert observer.finalize() == planned

    first = frozen_batches[0]
    mutated_first = replace(
        first,
        cache_addresses=(
            replace(
                first.cache_addresses[0],
                packed_shard_name="live-stream-mutated.jsonl.gz",
            ),
            *first.cache_addresses[1:],
        ),
    )
    mismatched = P50LiveExposureObserver(
        planned=planned,
        required_families=("atom_insert",),
        minimum_gradient_opportunities={"atom_insert": 50},
    )
    for completed_step, batch in enumerate(
        (mutated_first, *frozen_batches[1:]),
        start=1,
    ):
        mismatched(batch, completed_step)
    with pytest.raises(
        EditingTrainingSentinelError,
        match="address stream differs",
    ):
        mismatched.finalize()


@pytest.mark.parametrize("dead_route", ("family_gate", "action_route"))
def test_p50_cannot_hide_one_dead_family_gradient_route(
    fixture_bundle,
    dead_route,
) -> None:
    record, catalog, template, cache_records = fixture_bundle
    validation = _validation_batch(
        record,
        catalog,
        template,
        cache_records,
    )
    objective = CanonicalSuccessorTrainingObjective(
        mode="productive_identity",
        hazard_weight=0.0,
    )
    baseline = objective.metrics(
        template,
        validation,
        use_bf16=False,
        microbatch_size=None,
    )
    model = copy.deepcopy(template).train()
    model.zero_grad(set_to_none=True)
    loss = objective.loss(model, validation)
    loss.backward()

    family = "atom_insert"
    family_index = MARK_RULE_TO_INDEX[family]
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if dead_route == "family_gate" and name in {
            "family_head.2.weight",
            "family_head.2.bias",
        }:
            parameter.grad[family_index].zero_()
        if dead_route == "action_route" and name.startswith(
            FAMILY_PARAMETER_PREFIXES[family]
        ):
            parameter.grad.zero_()

    sentinel = P50GradientCollapseSentinel(
        required_families=(family,),
        minimum_gradient_updates={family: 1},
        maximum_family_nll_regression={family: 0.0},
        baseline_validation_metrics=baseline,
    )
    for step in range(1, 51):
        sentinel(model, validation.mark_batch, step, loss.detach())
    sentinel.observe_validation(
        completed_step=50,
        metrics=baseline,
    )
    expected_live_route = (
        sentinel.action_route_gradient_updates
        if dead_route == "family_gate"
        else sentinel.family_gate_gradient_updates
    )
    expected_dead_route = (
        sentinel.family_gate_gradient_updates
        if dead_route == "family_gate"
        else sentinel.action_route_gradient_updates
    )
    assert expected_live_route[family] == 50
    assert expected_dead_route[family] == 0
    with pytest.raises(
        EditingTrainingSentinelError,
        match="gradient shortfalls",
    ):
        sentinel.finalize()


def test_shared_trainer_executes_successor_dry_and_update_paths(
    fixture_bundle,
) -> None:
    record, catalog, template, cache_records = fixture_bundle
    steps = 2
    batch_size = 8
    path_length = record.path.path_length
    exposure_stream = _mark_dataset(
        record,
        length=steps * batch_size,
    )
    assert all(
        any(
            exposure_stream[offset].progress_index < path_length
            for offset in range(
                step * batch_size,
                (step + 1) * batch_size,
            )
        )
        for step in range(steps)
    )
    cache = _ExactCacheStub(
        {
            (
                row.address.packed_shard_content_sha256,
                row.address.entry_index,
                row.address.progress_index,
            ): row
            for row in cache_records
        }
    )
    cells = _semantic_cells(cache_records)

    def loader_factory(start_step: int):
        mark_dataset = FactorizedMarkDataset(
            (record,),
            start_index=start_step * batch_size,
            length=(steps - start_step) * batch_size,
            seed=31,
            late_time_fraction=0.5,
            operational_horizon=2.0,
            progress_stratification_fraction=0.5,
        )
        return factorized_successor_loader(
            mark_dataset,
            cache,
            _mark_collator(catalog, template),
            batch_size=batch_size,
            workers=0,
            pin_memory=False,
            seed=31,
            semantic_cell_ids=cells,
            require_semantic_cell_ids=True,
        )

    validation = _validation_batch(
        record,
        catalog,
        template,
        cache_records,
    )
    objective = CanonicalSuccessorTrainingObjective(
        mode="productive_identity_plus_hazard",
        hazard_weight=0.1,
    )
    common = {
        "train_records": (record,),
        "validation_batch": validation,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "seed": 31,
        "workers": 0,
        "late_time_fraction": 0.5,
        "operational_horizon": 2.0,
        "progress_stratification_fraction": 0.5,
        "use_aromatic_bond_view": True,
        "use_bf16": False,
        "ring_catalog": catalog,
        "evaluation_interval": 1,
        "training_objective": objective,
        "training_loader_factory": loader_factory,
    }

    dry_model = copy.deepcopy(template)
    before = copy.deepcopy(dry_model.state_dict())
    dry_result = train_factorized_mark_model(
        dry_model,
        dry_launch=True,
        **common,
    )
    assert dry_result["training_objective"] == objective.name
    assert dry_result["selection_metric"] == objective.selection_metric
    assert dry_result["train_loss_finite"]
    assert dry_result["validation_selection_finite"]
    for name, value in before.items():
        assert torch.equal(dry_model.state_dict()[name], value)

    recovery_states = []
    training_batch_steps = []
    gradient_steps = []
    validation_steps = []

    def gradient_audit(_model, _batch, completed_step, loss):
        assert torch.isfinite(loss)
        assert any(
            parameter.grad is not None
            for parameter in _model.parameters()
        )
        gradient_steps.append(completed_step)

    def training_batch_audit(batch, completed_step):
        assert isinstance(batch, FactorizedSuccessorBatch)
        training_batch_steps.append(completed_step)

    def validation_audit(completed_step, metrics):
        assert objective.selection_metric in metrics
        validation_steps.append(completed_step)

    trained = copy.deepcopy(template)
    history, best = train_factorized_mark_model(
        trained,
        checkpoint_interval=1,
        checkpoint_callback=recovery_states.append,
        training_batch_audit_callback=training_batch_audit,
        gradient_audit_callback=gradient_audit,
        validation_audit_callback=validation_audit,
        **common,
    )
    assert len(history) == steps
    assert objective.selection_metric in best
    assert training_batch_steps == [1, 2]
    assert gradient_steps == [1, 2]
    assert validation_steps == [1, 2]
    assert recovery_states
    assert recovery_states[-1]["training_objective_name"] == objective.name
    assert (
        recovery_states[-1]["training_selection_metric"]
        == objective.selection_metric
    )
    resumed = copy.deepcopy(template)
    resumed_history, resumed_best = train_factorized_mark_model(
        resumed,
        resume_state=recovery_states[0],
        **common,
    )
    assert resumed_history == history
    assert resumed_best == best
    for name, value in trained.state_dict().items():
        assert torch.equal(resumed.state_dict()[name], value)

    incompatible = CanonicalSuccessorTrainingObjective(
        mode="productive_identity",
        hazard_weight=0.0,
    )
    with pytest.raises(ValueError, match="another training objective"):
        train_factorized_mark_model(
            copy.deepcopy(template),
            resume_state=recovery_states[0],
            **{
                **common,
                "training_objective": incompatible,
            },
        )

    blocked_checkpoints = []

    def reject_validation(_completed_step, _metrics):
        raise EditingTrainingSentinelError("sentinel rejected validation")

    with pytest.raises(
        EditingTrainingSentinelError,
        match="sentinel rejected validation",
    ):
        train_factorized_mark_model(
            copy.deepcopy(template),
            checkpoint_interval=1,
            checkpoint_callback=blocked_checkpoints.append,
            validation_audit_callback=reject_validation,
            **common,
        )
    assert blocked_checkpoints == []

    def reject_live_batch(_batch, _completed_step):
        raise EditingTrainingSentinelError("live exposure differs")

    with pytest.raises(
        EditingTrainingSentinelError,
        match="live exposure differs",
    ):
        train_factorized_mark_model(
            copy.deepcopy(template),
            checkpoint_interval=1,
            checkpoint_callback=blocked_checkpoints.append,
            training_batch_audit_callback=reject_live_batch,
            **common,
        )
    assert blocked_checkpoints == []


def test_builder_refuses_unaddressed_trace(fixture_bundle) -> None:
    record, _catalog, model, _cache_records = fixture_bundle
    with pytest.raises(
        SuccessorFiberCacheBuildError,
        match="immutable packed address",
    ):
        compile_successor_fiber_trace(
            model,
            replace(record, corpus_address=None),
        )
    assert compile_successor_fiber_shard(
        model,
        (record,),
        expected_packed_entry_count=1,
    )
    with pytest.raises(
        SuccessorFiberCacheBuildError,
        match="complete declared packed census",
    ):
        compile_successor_fiber_shard(
            model,
            (record,),
            expected_packed_entry_count=2,
        )
