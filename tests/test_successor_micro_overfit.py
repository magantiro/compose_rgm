"""Bounded successor micro-overfit uses exact executable molecular targets."""

from __future__ import annotations

import torch

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records
from compose_v4.experiments.successor_micro_overfit import (
    examples_from_traces,
    prepare_successor_panel,
    successor_panel_metrics,
    train_successor_micro_panel,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog


def _model() -> FactorizedTraceletRateModel:
    torch.manual_seed(31)
    return FactorizedTraceletRateModel(
        build_typed_ring_catalog(()),
        hidden_dim=12,
        message_passing_steps=1,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    )


def _cycle_examples():
    records, attempted = build_cycle_op_records(
        ("C1CCCCC1", "c1ccccc1", "C1CC2CCC1C2"),
        n_slots=12,
        seed=4,
        max_bonds_per_molecule=2,
    )
    assert attempted > 0
    return examples_from_traces(
        (record.path.trace for record in records),
        families=("cycle_insert", "cycle_attach"),
        maximum_per_family=2,
        unique_source_molecules=True,
    )


def test_panel_reports_successor_family_and_within_family_metrics():
    model = _model()
    examples = _cycle_examples()
    assert {row.family_name for row in examples} == {
        "cycle_insert",
        "cycle_attach",
    }
    panel = prepare_successor_panel(model, examples)
    metrics = successor_panel_metrics(model, panel)

    assert metrics["n_examples"] == len(examples)
    assert metrics["canonical_successor_nll"] >= 0.0
    assert 0.0 <= metrics["teacher_successor_top1_recall"] <= 1.0
    assert 0.0 < metrics["teacher_family_probability"] <= 1.0
    assert (
        0.0
        < metrics["within_teacher_family_successor_probability"]
        <= 1.0
    )
    assert all(row["alias_multiplicity"] >= 1 for row in metrics["per_example"])


def test_one_example_all_parameter_micro_overfit_improves_successor_nll():
    model = _model()
    example = next(
        row for row in _cycle_examples() if row.family_name == "cycle_attach"
    )
    panel = prepare_successor_panel(model, (example,))
    report = train_successor_micro_panel(
        model,
        panel,
        steps=24,
        learning_rate=2e-2,
        scope="all",
        seed=9,
        report_points=(4, 12),
    )

    assert not report["required_components_without_gradient"]
    assert (
        report["final"]["canonical_successor_nll"]
        < report["initial"]["canonical_successor_nll"]
    )
    assert (
        report["final"]["teacher_successor_probability"]
        > report["initial"]["teacher_successor_probability"]
    )
