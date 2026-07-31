"""Validation helpers for claim-scoped COMPOSE comparator plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


EXPECTED_SCHEMA = "compose.comparator_registry"
EXPECTED_SCHEMA_VERSION = 2
EXPECTED_REGISTRY_ID = "compose-editing-and-control-comparators-v2"
SUPERSEDED_REGISTRY_ID = "compose-editing-and-control-comparators-v1"
REQUIRED_CLAIMS = frozenset(
    {
        "learned_state_dependent_transport",
        "operator_enabled_size_and_topology_adaptation",
        "finite_horizon_control",
        "pareto_and_dynamic_lead_optimization",
    }
)
REQUIRED_EXPERIMENTS_BY_CLAIM = {
    "learned_state_dependent_transport": frozenset({"E2"}),
    "operator_enabled_size_and_topology_adaptation": frozenset({"E3", "E4"}),
    "finite_horizon_control": frozenset({"E5", "E6"}),
    "pareto_and_dynamic_lead_optimization": frozenset({"E7"}),
}
REQUIRED_TRANSPORT_ARMS = frozenset(
    {
        "learned_unguided_prior",
        "uniform_canonical_successor",
        "state_independent_empirical_family_prior",
    }
)
REQUIRED_OPERATOR_ARMS_BY_EXPERIMENT = {
    "E3": frozenset(
        {
            "full_operator_basis",
            "no_atom_insert",
            "no_atom_delete",
            "fixed_cardinality",
            "no_bond_reroute",
        }
    ),
    "E4": frozenset(
        {
            "full_operator_basis",
            "no_cycle_operations",
            "no_bond_reroute",
        }
    ),
}
REQUIRED_CATALOG_COMPARATOR = "finite_catalog_topology_editor"
REQUIRED_FINITE_HORIZON_ARMS = frozenset(
    {
        "unguided_prior",
        "endpoint_reranking",
        "greedy_one_step",
        "local_boltzmann",
        "smc_feynman_kac",
        "learned_doob_value",
        "exact_doob",
    }
)
REQUIRED_PARETO_SAME_BASE = frozenset(
    {
        "unguided_prior",
        "uniform_canonical_successor",
        "endpoint_reranking",
        "greedy_one_step",
        "local_boltzmann",
        "static_scalarization",
        "mog_dfm_style_rank_hypercone",
        "smc_feynman_kac",
        "learned_doob_value",
        "nsga2_over_compose_successors",
    }
)
REQUIRED_CONDITIONAL_PARETO_ARMS = frozenset(
    {
        "moead_over_compose_successors",
        "areuredi_style_annealed_lbmh",
    }
)
REQUIRED_DYNAMIC_ARMS = frozenset(
    {
        "restart_replan",
        "myopic_retargeting",
        "warm_start_multiobjective_search",
        "reusable_transport_dynamic_control",
    }
)
REQUIRED_MECHANISTIC_SWITCH_ARMS = frozenset(
    {
        "continue_from_switch_state",
        "restart_from_original_source",
        "restart_from_switch_state_with_fresh_sampler",
        "static_compromise_objective",
    }
)
REQUIRED_EXTERNAL_BASELINES = frozenset(
    {
        "InVirtuoGen",
        "GenMol",
        "GraphGA",
        "MARS",
        "RetMol",
        "HN_GFN",
    }
)


class ComparatorRegistryError(ValueError):
    """The comparator registry cannot support a fair experiment."""


def load_comparator_registry(path: str | Path) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text())
    validate_comparator_registry(registry)
    return registry


def _unique_ids(
    rows: list[dict[str, Any]],
    label: str,
    *,
    key: str = "id",
) -> set[str]:
    identifiers = [str(row.get(key)) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ComparatorRegistryError(f"{label} contains duplicate identifiers")
    return set(identifiers)


def validate_comparator_registry(registry: dict[str, Any]) -> None:
    if registry.get("schema") != EXPECTED_SCHEMA:
        raise ComparatorRegistryError("unexpected comparator-registry schema")
    if registry.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ComparatorRegistryError("unexpected comparator-registry schema version")
    if registry.get("registry_id") != EXPECTED_REGISTRY_ID:
        raise ComparatorRegistryError("unexpected comparator-registry identity")
    if registry.get("supersedes_registry_id") != SUPERSEDED_REGISTRY_ID:
        raise ComparatorRegistryError("comparator registry must identify its superseded schema")
    if not registry.get("path_compatibility_note"):
        raise ComparatorRegistryError("comparator registry must explain its legacy-stable filename")
    if registry.get("status") != "PREIMPLEMENTATION":
        raise ComparatorRegistryError(
            "comparator registry must remain PREIMPLEMENTATION until arm availability is verified"
        )
    if not registry.get("availability_policy"):
        raise ComparatorRegistryError(
            "comparator registry must state that declared arms are not availability claims"
        )

    claims = registry.get("claim_comparisons") or []
    claim_ids = _unique_ids(claims, "claim_comparisons", key="claim_id")
    if claim_ids != REQUIRED_CLAIMS:
        raise ComparatorRegistryError(
            "comparison claims disagree with the frozen authority: "
            f"expected {sorted(REQUIRED_CLAIMS)}, observed {sorted(claim_ids)}"
        )
    by_claim = {str(row["claim_id"]): row for row in claims}
    for claim_id, expected_experiments in REQUIRED_EXPERIMENTS_BY_CLAIM.items():
        observed_experiments = set(by_claim[claim_id].get("experiments") or ())
        if observed_experiments != expected_experiments:
            raise ComparatorRegistryError(
                f"{claim_id} experiment scope disagrees with the frozen authority: "
                f"expected {sorted(expected_experiments)}, "
                f"observed {sorted(observed_experiments)}"
            )

    transport = by_claim["learned_state_dependent_transport"]
    transport_ids = _unique_ids(
        transport.get("required") or [],
        "learned_state_dependent_transport.required",
    )
    if transport_ids != REQUIRED_TRANSPORT_ARMS:
        raise ComparatorRegistryError(
            "E2 transport arms disagree with the frozen authority: "
            f"expected {sorted(REQUIRED_TRANSPORT_ARMS)}, observed {sorted(transport_ids)}"
        )

    operators = by_claim["operator_enabled_size_and_topology_adaptation"]
    operator_rows = operators.get("required") or []
    _unique_ids(
        operator_rows,
        "operator_enabled_size_and_topology_adaptation.required",
    )
    for experiment_id, required_ids in REQUIRED_OPERATOR_ARMS_BY_EXPERIMENT.items():
        scoped_ids = {
            str(row.get("id"))
            for row in operator_rows
            if experiment_id in set(row.get("experiments") or ())
        }
        if scoped_ids != required_ids:
            raise ComparatorRegistryError(
                f"{experiment_id} operator arms disagree with the frozen authority: "
                f"expected {sorted(required_ids)}, observed {sorted(scoped_ids)}"
            )

    conditional_operator_rows = operators.get("conditional") or []
    conditional_operator_ids = _unique_ids(
        conditional_operator_rows,
        "operator_enabled_size_and_topology_adaptation.conditional",
    )
    if conditional_operator_ids != {REQUIRED_CATALOG_COMPARATOR}:
        raise ComparatorRegistryError(
            "E4 must declare only the gated finite-catalog topology comparator"
        )
    catalog = conditional_operator_rows[0]
    if catalog.get("status") != "PROSPECTIVE_UNAVAILABLE":
        raise ComparatorRegistryError(
            "finite-catalog topology comparator must remain PROSPECTIVE_UNAVAILABLE "
            "until its run gate passes"
        )
    if set(catalog.get("experiments") or ()) != {"E4"} or not catalog.get("run_gate"):
        raise ComparatorRegistryError(
            "finite-catalog topology comparator must be gated and scoped only to E4"
        )

    finite_horizon = by_claim["finite_horizon_control"]
    finite_horizon_ids = _unique_ids(
        finite_horizon.get("required") or [],
        "finite_horizon_control.required",
    )
    if finite_horizon_ids != REQUIRED_FINITE_HORIZON_ARMS:
        raise ComparatorRegistryError(
            "finite-horizon control arms disagree with the frozen authority: "
            f"expected {sorted(REQUIRED_FINITE_HORIZON_ARMS)}, "
            f"observed {sorted(finite_horizon_ids)}"
        )

    pareto = by_claim["pareto_and_dynamic_lead_optimization"]
    same_base = set(pareto.get("required_same_base") or ())
    if same_base != REQUIRED_PARETO_SAME_BASE:
        raise ComparatorRegistryError(
            "same-base Pareto arms disagree with the frozen authority: "
            f"expected {sorted(REQUIRED_PARETO_SAME_BASE)}, observed {sorted(same_base)}"
        )
    conditional_rows = pareto.get("conditional_same_base") or []
    conditional_ids = _unique_ids(
        conditional_rows,
        "pareto_and_dynamic_lead_optimization.conditional_same_base",
    )
    if conditional_ids != REQUIRED_CONDITIONAL_PARETO_ARMS:
        raise ComparatorRegistryError(
            "conditional same-base Pareto arms disagree with the frozen authority"
        )
    for row in conditional_rows:
        if row.get("status") != "CONDITIONAL_UNIMPLEMENTED" or not row.get("run_gate"):
            raise ComparatorRegistryError(
                f"conditional same-base arm {row.get('id')} lacks its frozen run gate"
            )
    dynamic = set(pareto.get("dynamic_arms") or ())
    if dynamic != REQUIRED_DYNAMIC_ARMS:
        raise ComparatorRegistryError(
            "dynamic comparator arms disagree with the frozen authority: "
            f"expected {sorted(REQUIRED_DYNAMIC_ARMS)}, observed {sorted(dynamic)}"
        )
    mechanistic = set(pareto.get("mechanistic_switch_arms") or ())
    if mechanistic != REQUIRED_MECHANISTIC_SWITCH_ARMS:
        raise ComparatorRegistryError(
            "mechanistic switch arms disagree with the frozen authority: "
            f"expected {sorted(REQUIRED_MECHANISTIC_SWITCH_ARMS)}, "
            f"observed {sorted(mechanistic)}"
        )

    external = registry.get("external_small_molecule_baselines") or []
    external_ids = _unique_ids(external, "external_small_molecule_baselines")
    if external_ids != REQUIRED_EXTERNAL_BASELINES:
        raise ComparatorRegistryError(
            "external baseline set disagrees with the frozen authority: "
            f"expected {sorted(REQUIRED_EXTERNAL_BASELINES)}, observed {sorted(external_ids)}"
        )
    for row in external:
        source = str(row.get("source") or "")
        parsed = urlparse(source)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ComparatorRegistryError(
                f"external baseline {row.get('id')} lacks a primary HTTPS source"
            )
        if row.get("comparison_type") != "external_baseline":
            raise ComparatorRegistryError(
                f"external baseline {row.get('id')} has the wrong comparison type"
            )
        if not row.get("priority"):
            raise ComparatorRegistryError(
                f"external baseline {row.get('id')} lacks an execution priority"
            )

    adapter_fields = set((registry.get("external_adapter_gate") or {}).get("required_fields") or ())
    required_adapter_fields = {
        "code_repository_and_commit",
        "objective_implementation_hashes",
        "constraint_implementation_hashes",
        "oracle_budget_accounting",
        "selection_rule",
    }
    missing_adapter_fields = required_adapter_fields - adapter_fields
    if missing_adapter_fields:
        raise ComparatorRegistryError(
            f"external adapter gate omits: {sorted(missing_adapter_fields)}"
        )
