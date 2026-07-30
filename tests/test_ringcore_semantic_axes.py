from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import empty_molecular_graph, pad_molecular_graph
from compose_v4.experiments.ringcore_semantic_axes import (
    BoundedSemanticCellCensus,
    SPLIT_UNIT_LABEL,
    SemanticAxisLabelError,
    SemanticTraceContext,
    context_from_path_record,
    graph_cycle_rank,
    label_validation_semantic_axes,
    semantic_axis_labeler_contract,
    semantic_axis_labeler_contract_sha256,
    validate_labeler_against_leaderboard_config,
)
from compose_v4.experiments.ringcore_successor_leaderboard import (
    SEMANTIC_CELL_AXES,
    load_json_object,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "ringcore_v1_successor_leaderboard_v1.json"


def _state(smiles: str, *, slots: int = 12):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), slots)


def _context(
    source: str,
    target: str,
    rules: tuple[str, ...],
    *,
    layer: str = "corruption",
    progress_index: int = 0,
) -> SemanticTraceContext:
    terminal = progress_index == len(rules)
    return SemanticTraceContext(
        layer=layer,
        partition="validation",
        path_length=len(rules),
        progress_index=progress_index,
        trace_source=_state(source),
        trace_target=_state(target),
        trace_rule_names=rules,
        teacher_rule_name=None if terminal else rules[progress_index],
    )


def test_terminal_row_has_null_axes_cell_and_audit_deltas() -> None:
    assignment = label_validation_semantic_axes(
        _context("C", "CC", ("atom_insert",), progress_index=1)
    )
    assert assignment.terminal
    assert assignment.axis_values is None
    assert assignment.semantic_cell_id is None
    assert assignment.atom_count_delta is None
    assert assignment.graph_cycle_rank_delta is None
    assert assignment.diagnostic_values is None
    assert assignment.to_panel_fields() == {
        "semantic_axis_values": None,
        "semantic_cell_id": None,
    }


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (1, "precise_local_1_2"),
        (2, "precise_local_1_2"),
        (3, "ordinary_lead_optimization_3_6"),
        (6, "ordinary_lead_optimization_3_6"),
        (7, "longer_compositional_gt_6"),
    ],
)
def test_path_scale_uses_frozen_corpus_boundaries(
    length: int,
    expected: str,
) -> None:
    rules = ("atom_insert",) * length
    assignment = label_validation_semantic_axes(
        _context("C", "CC", rules)
    )
    assert assignment.axis_values["path_scale"] == expected


@pytest.mark.parametrize(
    ("layer", "expected"),
    [
        ("corruption", "synthetic_general_corruption"),
        ("general_corruption", "synthetic_general_corruption"),
        ("cycle_ops", "synthetic_cycle_operations"),
        ("cycle_operations", "synthetic_cycle_operations"),
        ("mmp_analogue", "real_mmp_analogue_endpoints"),
    ],
)
def test_evidence_origin_normalizes_only_current_layer_aliases(
    layer: str,
    expected: str,
) -> None:
    assignment = label_validation_semantic_axes(
        _context("C", "CC", ("atom_insert",), layer=layer)
    )
    assert assignment.axis_values["evidence_origin"] == expected


def test_endpoint_delta_does_not_mislabel_first_mmp_deletion() -> None:
    # This is a compiled replacement-and-growth path.  Its first jump deletes
    # an atom, but the trace-level evidence is net growth.
    assignment = label_validation_semantic_axes(
        _context(
            "CC",
            "CCC",
            ("atom_delete", "atom_insert", "atom_insert"),
            layer="mmp_analogue",
        )
    )
    assert assignment.atom_count_delta == 1
    assert assignment.axis_values["cardinality_topology_delta"] == (
        "cardinality_grow;cycle_rank_same"
    )
    assert assignment.axis_values["capability_regime"] == (
        "cardinality_primitive"
    )


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("atom_insert", "cardinality_primitive"),
        ("atom_delete", "cardinality_primitive"),
        ("atom_restate", "atom_state"),
        ("bond_reorder", "bond_state"),
        ("bond_reroute", "attachment_reroute"),
        ("bond_insert", "cycle_topology_rewrite"),
        ("bond_delete", "cycle_topology_rewrite"),
        ("ring_system_restate", "ring_state"),
        ("ring_system_delete", "ring_topology_macro"),
    ],
)
def test_capability_regime_is_the_row_local_canonical_teacher_family(
    rule: str,
    expected: str,
) -> None:
    assignment = label_validation_semantic_axes(
        _context("CC", "CC", (rule,))
    )
    assert assignment.axis_values["capability_regime"] == expected


def test_cycle_rank_is_exact_e_minus_v_plus_components() -> None:
    assert graph_cycle_rank(empty_molecular_graph(8)) == 0
    assert graph_cycle_rank(_state("CCC")) == 0
    assert graph_cycle_rank(_state("C1CC1")) == 1
    assert graph_cycle_rank(_state("C1CCC2CCCCC2C1")) == 2


@pytest.mark.parametrize(
    ("source", "target", "rule", "expected_delta", "expected_label"),
    [
        (
            "CCC",
            "C1CC1",
            "bond_insert",
            1,
            "cardinality_same;cycle_rank_increase",
        ),
        (
            "C1CC1",
            "CCC",
            "bond_delete",
            -1,
            "cardinality_same;cycle_rank_decrease",
        ),
    ],
)
def test_topology_direction_uses_graph_cycle_rank(
    source: str,
    target: str,
    rule: str,
    expected_delta: int,
    expected_label: str,
) -> None:
    assignment = label_validation_semantic_axes(
        _context(source, target, (rule,), layer="cycle_ops")
    )
    assert assignment.graph_cycle_rank_delta == expected_delta
    assert assignment.axis_values["cardinality_topology_delta"] == expected_label
    assert (
        assignment.axis_values["capability_regime"]
        == "cycle_topology_rewrite"
    )


def test_selection_chemistry_is_coarse_while_diagnostics_remain_exact() -> None:
    aromatic = label_validation_semantic_axes(
        _context(
            "c1ccccc1",
            "C1CCCCC1",
            ("ring_system_restate",),
        )
    )
    assert aromatic.axis_values["chemistry_charge_stratum"] == (
        "cnof_only_any_aromaticity_any_charge"
    )
    assert aromatic.diagnostic_values == {
        "exact_element_subset": "C",
        "aromaticity_transition": "aromatic_to_nonaromatic",
        "charge_transition": "uncharged_to_uncharged",
    }

    cold = label_validation_semantic_axes(
        _context("CS", "CCl", ("atom_restate",))
    )
    assert cold.axis_values["chemistry_charge_stratum"] == (
        "expanded_organic_any_aromaticity_any_charge"
    )
    assert cold.diagnostic_values == {
        "exact_element_subset": "C_S_Cl",
        "aromaticity_transition": "nonaromatic_to_nonaromatic",
        "charge_transition": "uncharged_to_uncharged",
    }

    charged = label_validation_semantic_axes(
        _context("[NH4+]", "C[NH3+]", ("atom_insert",))
    )
    assert charged.axis_values["chemistry_charge_stratum"] == (
        "cnof_only_any_aromaticity_any_charge"
    )
    assert (
        charged.diagnostic_values["charge_transition"]
        == "net_positive_to_net_positive"
    )


def test_split_cell_is_categorical_and_provenance_is_not_topology_claim() -> None:
    assignment = label_validation_semantic_axes(
        _context("C", "CC", ("atom_insert",))
    )
    assert assignment.axis_values["split_unit"] == SPLIT_UNIT_LABEL
    assert SPLIT_UNIT_LABEL == "held_scaffold"
    contract = semantic_axis_labeler_contract()
    split = contract["split_unit"]
    assert split["scaffold_key_algorithm"] == "murcko+carbonized-wl3"
    assert split["scaffold_key_version"] == 2
    assert "no topology-" in split["claim_scope"]


def test_assignment_is_deterministic_complete_and_config_bound() -> None:
    left = label_validation_semantic_axes(
        _context("CCO", "CCN", ("atom_restate",))
    )
    right = label_validation_semantic_axes(
        _context("CCO", "CCN", ("atom_restate",))
    )
    assert left == right
    assert tuple(left.axis_values) == SEMANTIC_CELL_AXES
    config = load_json_object(CONFIG_PATH)
    assert (
        validate_labeler_against_leaderboard_config(config)
        == semantic_axis_labeler_contract_sha256()
    )

    broken = copy.deepcopy(config)
    broken["panels"]["semantic_cells"]["required_axes"] = [
        *SEMANTIC_CELL_AXES[:-1],
        "invented_axis",
    ]
    with pytest.raises(SemanticAxisLabelError, match="axes disagree"):
        validate_labeler_against_leaderboard_config(broken)


def test_path_record_adapter_binds_address_and_teacher_progress() -> None:
    source = _state("C")
    target = _state("CC")
    trace = SimpleNamespace(
        source=source,
        target=target,
        steps=(SimpleNamespace(rule_name="atom_insert"),),
    )
    address = SimpleNamespace(layer="corruption", partition="validation")
    record = SimpleNamespace(
        path=SimpleNamespace(trace=trace, path_length=1),
        corpus_address=address,
    )
    jump = context_from_path_record(record, progress_index=0)
    assert jump.layer == "corruption"
    assert jump.teacher_rule_name == "atom_insert"
    assert not jump.terminal
    terminal = context_from_path_record(record, progress_index=1)
    assert terminal.teacher_rule_name is None
    assert terminal.terminal
    with pytest.raises(SemanticAxisLabelError, match="outside"):
        context_from_path_record(record, progress_index=2)


def test_labeler_fails_closed_on_unknown_or_misaligned_metadata() -> None:
    with pytest.raises(SemanticAxisLabelError, match="unknown"):
        label_validation_semantic_axes(
            _context("C", "CC", ("atom_insert",), layer="old_mmp")
        )
    bad_teacher = _context("C", "CC", ("atom_insert",))
    bad_teacher = SemanticTraceContext(
        **{
            **bad_teacher.__dict__,
            "teacher_rule_name": "atom_delete",
        }
    )
    with pytest.raises(SemanticAxisLabelError, match="teacher rule"):
        label_validation_semantic_axes(bad_teacher)
    bad_partition = _context("C", "CC", ("atom_insert",))
    bad_partition = SemanticTraceContext(
        **{
            **bad_partition.__dict__,
            "partition": "test",
        }
    )
    with pytest.raises(SemanticAxisLabelError, match="validation"):
        label_validation_semantic_axes(bad_partition)


def test_streaming_census_is_bounded_and_retains_exact_deltas() -> None:
    grow = label_validation_semantic_axes(
        _context("C", "CC", ("atom_insert",))
    )
    close = label_validation_semantic_axes(
        _context("CCC", "C1CC1", ("bond_insert",), layer="cycle_ops")
    )
    terminal = label_validation_semantic_axes(
        _context("C", "CC", ("atom_insert",), progress_index=1)
    )
    census = BoundedSemanticCellCensus(maximum_nonempty_cells=2)
    census.add(grow, importance_weight=2.0)
    census.add(close, importance_weight=3.0)
    census.add(terminal, importance_weight=4.0)
    payload = census.payload()
    assert payload["counts"] == {
        "rows": 3,
        "terminal_rows": 1,
        "nonterminal_rows": 2,
        "nonempty_cells": 2,
    }
    assert payload["importance_weight_sums"] == {
        "all": 9.0,
        "terminal": 4.0,
        "nonterminal": 5.0,
    }
    assert payload["exact_endpoint_delta_row_counts"] == {
        "atoms:+0;cycle_rank:+1": 1,
        "atoms:+1;cycle_rank:+0": 1,
    }
    assert payload["diagnostic_marginal_row_counts"][
        "aromaticity_transition"
    ] == {"nonaromatic_to_nonaromatic": 2}
    assert len(payload["census_sha256"]) == 64

    bounded = BoundedSemanticCellCensus(maximum_nonempty_cells=1)
    bounded.add(grow)
    with pytest.raises(SemanticAxisLabelError, match="exceeded"):
        bounded.add(close)
