"""Step-zero editing checks fail before an optimizer can hide bad provenance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.data.successor_fiber_cache import SuccessorFiberCacheAddress
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records
from compose_v4.experiments.editing_training_gate import (
    EditingTrainingGateError,
    load_editing_training_gate,
    validate_editing_training_gate,
)
from compose_v4.experiments.editing_step_zero_gate import (
    ATOM_VOCABULARY_HEAD_LAYOUT,
    EditingStepZeroGateError,
    assert_initialization_parity,
    audit_successor_support_before_optimization,
    build_compatible_initialization_plan,
    build_scratch_initialization_plan,
)
from compose_v4.experiments.factorized_successor_data import (
    FactorizedSuccessorBatch,
)
from compose_v4.experiments.successor_micro_overfit import (
    examples_from_traces,
    prepare_successor_panel,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog
from scripts.train_tracelet_cnof_gate import (
    _semantic_partial_checkpoint_initialization,
)

_SOURCE_CLASSES = ((2, 4), (3, 3))
_DESTINATION_CLASSES = (*_SOURCE_CLASSES, (4, 2))
_ROOT = Path(__file__).resolve().parents[1]


def test_step_zero_requirements_are_declared_in_the_launch_contract() -> None:
    contract = load_editing_training_gate(_ROOT / "configs" / "editing_training_v2_gate.json")
    support_gate = next(gate for gate in contract["gates"] if gate["id"] == "S0_support_and_labels")
    assert {
        "initialization_transfer_plan_complete",
        "initialized_state_matches_transfer_plan_exactly",
        "all_required_families_have_teachers_and_production_candidates",
        "all_legal_marks_execute",
        "canonical_successor_pushforward_finite_and_normalized",
    } <= set(support_gate["requirements"])


def test_launch_contract_rejects_removal_of_a_step_zero_requirement() -> None:
    contract = load_editing_training_gate(_ROOT / "configs" / "editing_training_v2_gate.json")
    support_gate = next(gate for gate in contract["gates"] if gate["id"] == "S0_support_and_labels")
    support_gate["requirements"].remove("initialization_transfer_plan_complete")

    with pytest.raises(EditingTrainingGateError, match="S0 is missing"):
        validate_editing_training_gate(contract)


def _source_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    return {
        "body.weight": torch.randn(4, 4),
        "restate_head.weight": torch.randn(2, 4),
        "restate_head.bias": torch.randn(2),
        "grow_root_head.weight": torch.randn(2, 4),
        "grow_root_head.bias": torch.randn(2),
        "grow_option.weight": torch.randn(6, 3),
    }


def _fresh_target_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(12)
    return {
        "body.weight": torch.randn(4, 4),
        "restate_head.weight": torch.randn(3, 4),
        "restate_head.bias": torch.randn(3),
        "grow_root_head.weight": torch.randn(3, 4),
        "grow_root_head.bias": torch.randn(3),
        "grow_option.weight": torch.randn(9, 3),
        "new_head.weight": torch.randn(2, 4),
    }


def _initialized_fixture():
    source = _source_state()
    fresh = _fresh_target_state()
    initialized, *_ = _semantic_partial_checkpoint_initialization(
        fresh,
        {"state_dict": source},
        source_classes=_SOURCE_CLASSES,
        dest_classes=_DESTINATION_CLASSES,
    )
    plan = build_compatible_initialization_plan(
        source,
        fresh,
        source_classes=_SOURCE_CLASSES,
        destination_classes=_DESTINATION_CLASSES,
        explicitly_fresh_tensors=("new_head.weight",),
    )
    return source, fresh, initialized, plan


def test_warm_start_parity_accounts_for_exact_mapped_and_fresh_state() -> None:
    source, fresh, initialized, plan = _initialized_fixture()
    report = assert_initialization_parity(
        plan,
        source_state=source,
        fresh_target_state=fresh,
        initialized_state=initialized,
    )

    assert report.regime == "compatible_warm_start"
    assert report.exact_tensor_names == ("body.weight",)
    assert report.fresh_tensor_names == ("new_head.weight",)
    assert report.copied_row_count == 14
    assert report.fresh_row_count == 7
    assert len(report.transfer_plan_sha256) == 64


def test_missing_or_malformed_inherited_mapping_fails_closed() -> None:
    source = _source_state()
    fresh = _fresh_target_state()
    missing_body = dict(source)
    del missing_body["body.weight"]
    with pytest.raises(
        EditingStepZeroGateError,
        match="body.weight.*absent from the source",
    ):
        build_compatible_initialization_plan(
            missing_body,
            fresh,
            source_classes=_SOURCE_CLASSES,
            destination_classes=_DESTINATION_CLASSES,
            explicitly_fresh_tensors=("new_head.weight",),
        )

    missing_layout = dict(ATOM_VOCABULARY_HEAD_LAYOUT)
    del missing_layout["restate_head.weight"]
    with pytest.raises(
        EditingStepZeroGateError,
        match="restate_head.weight.*without a semantic mapping",
    ):
        build_compatible_initialization_plan(
            source,
            fresh,
            source_classes=_SOURCE_CLASSES,
            destination_classes=_DESTINATION_CLASSES,
            explicitly_fresh_tensors=("new_head.weight",),
            semantic_layouts=missing_layout,
        )

    _source, _fresh, _initialized, complete = _initialized_fixture()
    with pytest.raises(ValueError, match="every destination row"):
        replace(
            complete,
            row_assignments=complete.row_assignments[:-1],
        )


def test_initialization_parity_detects_a_mutated_inherited_row() -> None:
    source, fresh, initialized, plan = _initialized_fixture()
    corrupted = {name: tensor.detach().clone() for name, tensor in initialized.items()}
    corrupted["restate_head.weight"][0, 0] += 1.0

    with pytest.raises(
        EditingStepZeroGateError,
        match="disagree with the exact transfer plan",
    ):
        assert_initialization_parity(
            plan,
            source_state=source,
            fresh_target_state=fresh,
            initialized_state=corrupted,
        )


def test_same_shape_semantic_heads_are_mapped_by_label_not_position() -> None:
    source_classes = ((2, 4), (3, 3))
    destination_classes = tuple(reversed(source_classes))
    source = {"restate_head.bias": torch.tensor([2.0, 3.0])}
    fresh = {"restate_head.bias": torch.tensor([-1.0, -1.0])}
    plan = build_compatible_initialization_plan(
        source,
        fresh,
        source_classes=source_classes,
        destination_classes=destination_classes,
    )
    initialized = {"restate_head.bias": torch.tensor([3.0, 2.0])}

    report = assert_initialization_parity(
        plan,
        source_state=source,
        fresh_target_state=fresh,
        initialized_state=initialized,
    )

    assert report.exact_tensor_count == 0
    assert report.copied_row_count == 2


def test_scratch_baseline_makes_no_inheritance_claim() -> None:
    fresh = _fresh_target_state()
    plan = build_scratch_initialization_plan(fresh)
    report = assert_initialization_parity(
        plan,
        fresh_target_state=fresh,
        initialized_state={name: tensor.detach().clone() for name, tensor in fresh.items()},
    )
    assert report.regime == "scratch"
    assert report.exact_tensor_count == 0
    assert report.fresh_tensor_count == len(fresh)


def _cycle_open_successor_batch():
    records, attempted = build_cycle_op_records(
        ("C1CCCCC1",),
        n_slots=10,
        seed=9,
        max_bonds_per_molecule=1,
    )
    assert attempted > 0
    examples = examples_from_traces(
        (record.path.trace for record in records),
        families=("cycle_attach",),
        maximum_per_family=1,
        unique_source_molecules=True,
    )
    assert len(examples) == 1
    torch.manual_seed(13)
    model = FactorizedTraceletRateModel(
        build_typed_ring_catalog(()),
        hidden_dim=12,
        message_passing_steps=1,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()
    panel = prepare_successor_panel(model, examples)
    example = examples[0]
    address = SuccessorFiberCacheAddress(
        packed_shard_content_sha256="1" * 64,
        packed_shard_name="step-zero.jsonl.gz",
        entry_index=0,
        layer="cycle_operations",
        partition="development",
        trace_id="cycle-open-0",
        trace_source_key=example.source_key,
        trace_target_key=example.target_key,
        progress_index=0,
        path_length=1,
    )
    batch = FactorizedSuccessorBatch(
        mark_batch=panel.batch,
        fibers=panel.fibers,
        state_supports=tuple(fiber.state_support for fiber in panel.fibers),
        cache_addresses=(address,),
        semantic_cell_ids=("cycle_attach_ring_opening",),
    )
    return model, batch


def test_successor_support_gate_replays_exact_teacher_and_candidates() -> None:
    model, batch = _cycle_open_successor_batch()
    report = audit_successor_support_before_optimization(
        model,
        (batch,),
        required_families=("cycle_attach",),
        required_semantic_cells=("cycle_attach_ring_opening",),
    )

    assert report.batch_count == 1
    assert report.nonterminal_row_count == 1
    assert report.teacher_examples_by_family == {"cycle_attach": 1}
    assert report.candidate_marks_by_family["cycle_attach"] > 0
    assert report.productive_successors_by_family["cycle_attach"] > 0
    assert report.examples_by_semantic_cell == {
        "cycle_attach_ring_opening": 1
    }


def test_malformed_or_missing_teacher_support_fails_closed() -> None:
    model, batch = _cycle_open_successor_batch()
    fiber = batch.fibers[0]
    assert fiber is not None
    wrong_target = replace(fiber, target_state_sha256="f" * 64)
    malformed = replace(batch, fibers=(wrong_target,))
    with pytest.raises(
        EditingStepZeroGateError,
        match="does not execute to its stored exact successor",
    ):
        audit_successor_support_before_optimization(
            model,
            (malformed,),
            required_families=("cycle_attach",),
            required_semantic_cells=("cycle_attach_ring_opening",),
        )

    with pytest.raises(
        EditingStepZeroGateError,
        match="no teacher examples",
    ):
        audit_successor_support_before_optimization(
            model,
            (batch,),
            required_families=("cycle_insert",),
            required_semantic_cells=("cycle_attach_ring_opening",),
        )

    malformed_rates = replace(
        batch,
        mark_batch=replace(
            batch.mark_batch,
            teacher_rates=torch.full_like(
                batch.mark_batch.teacher_rates,
                float("nan"),
            ),
        ),
    )
    with pytest.raises(
        EditingStepZeroGateError,
        match="nonfinite or negative teacher rates",
    ):
        audit_successor_support_before_optimization(
            model,
            (malformed_rates,),
            required_families=("cycle_attach",),
            required_semantic_cells=("cycle_attach_ring_opening",),
        )

    with pytest.raises(
        EditingStepZeroGateError,
        match="required semantic cells have no nonterminal examples",
    ):
        audit_successor_support_before_optimization(
            model,
            (batch,),
            required_families=("cycle_attach",),
            required_semantic_cells=("missing_cell",),
        )
