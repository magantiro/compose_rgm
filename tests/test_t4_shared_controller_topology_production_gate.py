import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.generic_complete_program_composer_v2 import (
    propose_generic_topology_macro_programs,
)
from compose_v4.experiments.t4_integrated_route_fiber import (
    EXPERTS,
    GENERIC_TOPOLOGY_MACRO_EXPERT,
    PROTONATION_AWARE_EXPERT,
    validate_expert_vocabulary,
)
from compose_v4.experiments.t4_shared_controller_scored_runtime import (
    generic_topology_macro_records,
    proposal_expert_vocabulary,
)
from compose_v4.experiments.t4_shared_controller_topology_production_gate import (
    _recommend_next_cell,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def _contract(experts):
    return {
        "controller": {
            "proposal_experts": list(experts),
            "proposal": {name: {} for name in experts},
        },
        "trajectory_distillation": {"enabled": False},
    }


def test_topology_expert_is_append_only_and_legacy_default_is_unchanged():
    assert proposal_expert_vocabulary(_contract(EXPERTS)) == EXPERTS
    assert proposal_expert_vocabulary(
        _contract((*EXPERTS, GENERIC_TOPOLOGY_MACRO_EXPERT))
    ) == (*EXPERTS, GENERIC_TOPOLOGY_MACRO_EXPERT)
    assert validate_expert_vocabulary((*EXPERTS, GENERIC_TOPOLOGY_MACRO_EXPERT)) == (
        *EXPERTS,
        GENERIC_TOPOLOGY_MACRO_EXPERT,
    )
    assert validate_expert_vocabulary(
        (*EXPERTS, PROTONATION_AWARE_EXPERT, GENERIC_TOPOLOGY_MACRO_EXPERT)
    ) == (*EXPERTS, PROTONATION_AWARE_EXPERT, GENERIC_TOPOLOGY_MACRO_EXPERT)
    with pytest.raises(ValueError, match="unsupported proposal expert vocabulary"):
        validate_expert_vocabulary((GENERIC_TOPOLOGY_MACRO_EXPERT, *EXPERTS))


def test_modern_shared_controller_app_dispatches_and_then_binds_runtime_context():
    source = (ROOT / "modal_apps/t4_shared_controller_completion_v1_app.py").read_text()
    start = source.index('elif expert_name == "generic_topology_macro_v2":')
    end = source.index("    else:", start)
    branch = source[start:end]
    assert "generic_topology_macro_records(" in branch
    assert 'row["parent_score"] = request["parent_score"]' in branch
    assert 'row["delta"] = cell["delta"]' in branch
    assert branch.index("generic_topology_macro_records(") < branch.index(
        'row["parent_score"]'
    )


def test_topology_generation_boundary_has_no_cell_route_or_objective_input(monkeypatch):
    producer_parameters = set(
        inspect.signature(propose_generic_topology_macro_programs).parameters
    )
    adapter_parameters = set(
        inspect.signature(generic_topology_macro_records).parameters
    )
    forbidden = {
        "cell",
        "cell_key",
        "target",
        "delta",
        "support",
        "parent_score",
        "route_expert",
        "teacher_endpoint",
        "objective",
    }
    assert not producer_parameters.intersection(forbidden)
    assert not adapter_parameters.intersection(forbidden)

    telemetry = {
        "schema_version": "generic_topology_macro_composer_telemetry_v2",
        "seed": 7,
        "attempts_per_plan": 1,
        "maximum_primitives": 32,
        "maximum_blocks": 8,
        "unique_committed_endpoints": 0,
        "trajectory_distilled_templates_loaded": 0,
        "route_weights_loaded": 0,
        "runtime_task_cell_or_target_input": False,
        "runtime_teacher_endpoint_input": False,
        "runtime_objective_input": False,
    }
    monkeypatch.setattr(
        "compose_v4.experiments.t4_shared_controller_scored_runtime."
        "propose_generic_topology_macro_programs",
        lambda source, **kwargs: SimpleNamespace(proposals=(), telemetry=telemetry),
    )
    records, observed = generic_topology_macro_records(
        parent="CCOC(=O)NCC",
        proposal_seed_value=7,
        settings={
            "attempts_per_plan": 1,
            "maximum_primitives": 32,
            "maximum_blocks": 8,
        },
    )
    assert records == []
    assert observed["raw_exact_unique"] == 0
    assert observed["endpoint_constraints_applied_during_generation"] is False
    assert observed["intermediate_endpoints_queried"] == 0


def test_topology_adapter_rejects_program_outside_frozen_large_support(monkeypatch):
    source = pad_molecular_graph(smiles_to_molecular_graph("CCOC(=O)NCC"), 48)
    malformed = SimpleNamespace(
        endpoint=source,
        actions=(),
        requested_scale="large",
        realized_scale="large",
        families=("construct_substituted_ring",),
        actual_changes={
            "changed_site_count": 0,
            "surviving_new_atoms": 0,
            "deleted_original_atoms": 0,
        },
        metadata={
            "macro_plan": {"name": "malformed"},
            "observed_macro_fields": {
                "delta_heavy_atoms": 0,
                "delta_cycle_rank": 0,
                "retained_fraction": 1.0,
                "created_attachment_interface_count": 0,
            },
        },
        program={},
        program_graph={},
    )
    monkeypatch.setattr(
        "compose_v4.experiments.t4_shared_controller_scored_runtime."
        "propose_generic_topology_macro_programs",
        lambda source, **kwargs: SimpleNamespace(proposals=(malformed,), telemetry={}),
    )
    with pytest.raises(RuntimeError, match="primitive support"):
        generic_topology_macro_records(
            parent="CCOC(=O)NCC",
            proposal_seed_value=7,
            settings={"attempts_per_plan": 1},
        )


def test_gate_contract_binds_six_losses_two_wins_and_zero_costs():
    contract, identity = load_contract(ROOT)
    assert len(identity) == 64
    assert [row["evidence_role"] for row in contract["cells"]].count(
        "terminal_loss"
    ) == 6
    assert [row["evidence_role"] for row in contract["cells"]].count(
        "contrasting_win"
    ) == 2
    assert contract["costs"] == {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "gpu_seconds": 0,
    }


def test_next_cell_recommendation_uses_only_predeclared_structural_rank():
    def row(cell_key, role, count, plans, cycles, span):
        return {
            "cell_key": cell_key,
            "evidence_role": role,
            "funnel": {"archive_union_unique": count},
            "selection_ready_support": {
                "macro_plan_diversity": plans,
                "cycle_gain_diversity": cycles,
                "heavy_atom_delta_span": span,
            },
        }

    recommendation = _recommend_next_cell(
        [
            row("loss_b", "terminal_loss", 4, 2, 1, 3),
            row("loss_a", "terminal_loss", 4, 3, 1, 2),
            row("win", "contrasting_win", 99, 6, 2, 20),
        ]
    )
    assert recommendation["cell_key"] == "loss_a"
    assert recommendation["authority"] == (
        "recommendation_only_no_scored_calls_authorized"
    )
