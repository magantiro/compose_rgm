from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from compose_v4.experiments.comparator_registry import (
    ComparatorRegistryError,
    load_comparator_registry,
    validate_comparator_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "comparator_registry_v1.json"
EXPERIMENT_REGISTRY_PATH = ROOT / "configs" / "experiment_registry.yaml"


def _claim(registry: dict, claim_id: str) -> dict:
    return next(row for row in registry["claim_comparisons"] if row["claim_id"] == claim_id)


def test_comparator_registry_is_claim_complete_and_well_formed() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    assert registry["schema_version"] == 2
    assert registry["registry_id"] == "compose-editing-and-control-comparators-v2"
    assert registry["supersedes_registry_id"] == "compose-editing-and-control-comparators-v1"
    assert registry["status"] == "PREIMPLEMENTATION"
    assert "does not claim" in registry["availability_policy"]
    assert len(registry["external_small_molecule_baselines"]) == 6


def test_registry_rejects_superseded_schema_version() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    broken = copy.deepcopy(registry)
    broken["schema_version"] = 1
    broken["registry_id"] = "compose-editing-and-control-comparators-v1"
    with pytest.raises(ComparatorRegistryError, match="schema version"):
        validate_comparator_registry(broken)


def test_registry_rejects_extra_claim_or_wrong_experiment_scope() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    extra_claim = copy.deepcopy(registry)
    extra_claim["claim_comparisons"].append(
        {"claim_id": "unregistered_claim", "experiments": ["E99"]}
    )
    with pytest.raises(ComparatorRegistryError, match="comparison claims"):
        validate_comparator_registry(extra_claim)

    wrong_scope = copy.deepcopy(registry)
    transport = _claim(wrong_scope, "learned_state_dependent_transport")
    transport["experiments"] = ["E2", "E7"]
    with pytest.raises(ComparatorRegistryError, match="experiment scope"):
        validate_comparator_registry(wrong_scope)


def test_registry_rejects_missing_transport_law() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    broken = copy.deepcopy(registry)
    transport = _claim(broken, "learned_state_dependent_transport")
    transport["required"] = [
        row
        for row in transport["required"]
        if row["id"] != "state_independent_empirical_family_prior"
    ]
    with pytest.raises(ComparatorRegistryError, match="E2 transport"):
        validate_comparator_registry(broken)


def test_registry_rejects_missing_experiment_scoped_operator_arm() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    broken = copy.deepcopy(registry)
    operators = _claim(broken, "operator_enabled_size_and_topology_adaptation")
    no_reroute = next(row for row in operators["required"] if row["id"] == "no_bond_reroute")
    no_reroute["experiments"].remove("E4")
    with pytest.raises(ComparatorRegistryError, match="E4 operator"):
        validate_comparator_registry(broken)


def test_registry_rejects_missing_same_base_pareto_arm() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    broken = copy.deepcopy(registry)
    pareto = _claim(broken, "pareto_and_dynamic_lead_optimization")
    pareto["required_same_base"].remove("mog_dfm_style_rank_hypercone")
    with pytest.raises(ComparatorRegistryError, match="same-base Pareto"):
        validate_comparator_registry(broken)


def test_registry_rejects_missing_conditional_pareto_arm() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    broken = copy.deepcopy(registry)
    pareto = _claim(broken, "pareto_and_dynamic_lead_optimization")
    pareto["conditional_same_base"].pop()
    with pytest.raises(ComparatorRegistryError, match="conditional same-base"):
        validate_comparator_registry(broken)


def test_registry_rejects_missing_dynamic_comparator() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    broken = copy.deepcopy(registry)
    pareto = _claim(broken, "pareto_and_dynamic_lead_optimization")
    pareto["dynamic_arms"].remove("warm_start_multiobjective_search")
    with pytest.raises(ComparatorRegistryError, match="dynamic comparator"):
        validate_comparator_registry(broken)


def test_registry_rejects_missing_mechanistic_switch_arm() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    broken = copy.deepcopy(registry)
    pareto = _claim(broken, "pareto_and_dynamic_lead_optimization")
    pareto["mechanistic_switch_arms"].remove("continue_from_switch_state")
    with pytest.raises(ComparatorRegistryError, match="mechanistic switch"):
        validate_comparator_registry(broken)


def test_registry_rejects_catalog_comparator_availability_overclaim() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    broken = copy.deepcopy(registry)
    operators = _claim(broken, "operator_enabled_size_and_topology_adaptation")
    operators["conditional"][0]["status"] = "RUNNABLE"
    with pytest.raises(ComparatorRegistryError, match="PROSPECTIVE_UNAVAILABLE"):
        validate_comparator_registry(broken)


def test_registry_rejects_missing_finite_horizon_arm() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    broken = copy.deepcopy(registry)
    finite_horizon = _claim(broken, "finite_horizon_control")
    finite_horizon["required"] = [
        row for row in finite_horizon["required"] if row["id"] != "exact_doob"
    ]
    with pytest.raises(ComparatorRegistryError, match="finite-horizon control"):
        validate_comparator_registry(broken)


def test_experiment_registry_mirrors_comparator_authority() -> None:
    comparator_registry = load_comparator_registry(REGISTRY_PATH)
    pareto = _claim(comparator_registry, "pareto_and_dynamic_lead_optimization")
    experiment_registry = yaml.safe_load(EXPERIMENT_REGISTRY_PATH.read_text())
    experiments = experiment_registry["experiments"]
    declared_controllers = {row["id"]: row for row in experiment_registry["controllers"]}
    assert set(experiments["E7"]["controllers"]) <= set(declared_controllers)
    assert declared_controllers["uniform_canonical_successor"] == {
        "id": "uniform_canonical_successor",
        "comparison_type": "law_only_negative_control",
        "oracle_uses": "none",
        "requires_checkpoint": False,
    }

    assert set(experiments["E2"]["arms"]) == {
        "learned_unguided_prior",
        "uniform_canonical_successor",
        "state_independent_empirical_family_prior",
    }
    assert set(experiments["E3"]["baselines"]) == {
        "full_operator_basis",
        "no_atom_insert",
        "no_atom_delete",
        "fixed_cardinality",
        "no_bond_reroute",
    }
    assert set(experiments["E4"]["ablations"]) == {
        "full_operator_basis",
        "no_cycle_operations",
        "no_bond_reroute",
    }

    catalog = experiments["E4"]["conditional_comparators"]
    assert [row["id"] for row in catalog] == ["finite_catalog_topology_editor"]
    assert catalog[0]["status"] == "PROSPECTIVE_UNAVAILABLE"

    assert set(experiments["E7"]["controllers"]) == {
        "unguided_prior",
        "uniform_canonical_successor",
        "endpoint_reranking",
        "greedy_one_step",
        "local_boltzmann",
        "static_scalarization",
        "mog_dfm_style_rank_hypercone",
        "smc_feynman_kac",
        "learned_doob_value",
    }
    assert experiments["E7"]["static_search_comparators"] == ["nsga2_over_compose_successors"]
    assert set(experiments["E7"]["controllers"]) | set(
        experiments["E7"]["static_search_comparators"]
    ) == set(pareto["required_same_base"])
    conditional = experiments["E7"]["conditional_same_base"]
    assert {row["id"] for row in conditional} == {
        row["id"] for row in pareto["conditional_same_base"]
    }
    assert all(
        row["status"] == "CONDITIONAL_UNIMPLEMENTED" and row["run_gate"] for row in conditional
    )
    expected_dynamic = {
        "restart_replan",
        "myopic_retargeting",
        "warm_start_multiobjective_search",
        "reusable_transport_dynamic_control",
    }
    assert set(experiments["E7"]["dynamic_comparators"]) == expected_dynamic
    assert set(experiments["E7"]["dynamic_comparators"]) == set(pareto["dynamic_arms"])
    assert set(
        experiments["E7"]["flagship_studies"]["dynamic_preference_switching"]["arms"]
    ) == set(pareto["mechanistic_switch_arms"])
    assert set(experiments["E7"]["external_baselines"]) == {
        row["id"] for row in comparator_registry["external_small_molecule_baselines"]
    }


def test_registry_rejects_external_baseline_set_drift() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    broken = copy.deepcopy(registry)
    broken["external_small_molecule_baselines"].pop()
    with pytest.raises(ComparatorRegistryError, match="external baseline set"):
        validate_comparator_registry(broken)


def test_registry_rejects_external_baseline_without_primary_source() -> None:
    registry = load_comparator_registry(REGISTRY_PATH)
    broken = copy.deepcopy(registry)
    broken["external_small_molecule_baselines"][0]["source"] = "not-a-url"
    with pytest.raises(ComparatorRegistryError, match="primary HTTPS source"):
        validate_comparator_registry(broken)
