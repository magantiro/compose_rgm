"""Validation helpers for claim-scoped COMPOSE comparator plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


EXPECTED_SCHEMA = "compose.comparator_registry"
EXPECTED_SCHEMA_VERSION = 1
REQUIRED_CLAIMS = frozenset(
    {
        "learned_state_dependent_transport",
        "operator_enabled_size_and_topology_adaptation",
        "finite_horizon_control",
        "pareto_and_dynamic_lead_optimization",
    }
)
REQUIRED_PARETO_SAME_BASE = frozenset(
    {
        "unguided_prior",
        "endpoint_reranking",
        "greedy_one_step",
        "local_boltzmann",
        "static_scalarization",
        "smc_feynman_kac",
        "learned_doob_value",
        "nsga2_over_compose_successors",
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

    claims = registry.get("claim_comparisons") or []
    claim_ids = _unique_ids(claims, "claim_comparisons", key="claim_id")
    missing_claims = REQUIRED_CLAIMS - claim_ids
    if missing_claims:
        raise ComparatorRegistryError(
            f"required comparison claims are absent: {sorted(missing_claims)}"
        )
    by_claim = {str(row["claim_id"]): row for row in claims}
    pareto = by_claim["pareto_and_dynamic_lead_optimization"]
    same_base = set(pareto.get("required_same_base") or ())
    missing_same_base = REQUIRED_PARETO_SAME_BASE - same_base
    if missing_same_base:
        raise ComparatorRegistryError(
            f"required same-base Pareto arms are absent: {sorted(missing_same_base)}"
        )

    external = registry.get("external_small_molecule_baselines") or []
    _unique_ids(external, "external_small_molecule_baselines")
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

    adapter_fields = set(
        (registry.get("external_adapter_gate") or {}).get("required_fields") or ()
    )
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
            "external adapter gate omits: "
            f"{sorted(missing_adapter_fields)}"
        )
