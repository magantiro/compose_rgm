"""Deterministic evidence-lane routing for Editing-V2 candidate traces.

The router classifies a trace from hash-bound source provenance and compiled
trace structure. It never upgrades a structurally inferred molecular
relationship into an observed pair relationship. Routing is upstream of split
assignment, whole-trace Active8 admission, and training authorization.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES

ROUTING_POLICY_SCHEMA = "compose.editing_v2_candidate_routing_policy"
ROUTING_POLICY_SCHEMA_VERSION = 1
ROUTING_POLICY_ID = "compose-editing-v2-five-lane-routing-v1"

INFERRED_REAL_ENDPOINT_PAIR = "inferred_real_endpoint_pair"
EXECUTOR_GENERATED_WALK_FROM_REAL_ENDPOINT = "executor_generated_walk_from_real_endpoint"
SOURCE_KINDS = (
    INFERRED_REAL_ENDPOINT_PAIR,
    EXECUTOR_GENERATED_WALK_FROM_REAL_ENDPOINT,
)
_POLICY_FIELDS = {
    "schema",
    "schema_version",
    "policy_id",
    "source_kinds",
    "active_families",
    "attachment_or_topology_families",
    "rules_in_priority_order",
}

INFERRED_PROFILE = "inferred_relation_compiled_path"
SYNTHETIC_PROFILE = "executor_generated_walk"

LOCAL_ANALOGUE_LANE = "observed_local_analogue"
OPERATOR_AWARE_LANE = "operator_aware_real_endpoint"
LINKER_TOPOLOGY_LANE = "linker_positional_topology_analogue"
MULTISTEP_LANE = "real_endpoint_multistep_path"
SYNTHETIC_WALK_LANE = "reversible_synthetic_walk"

ATTACHMENT_OR_TOPOLOGY_FAMILIES = frozenset(
    {
        "bond_reroute",
        "cycle_insert",
        "cycle_attach",
        "ring_system_restate",
    }
)


class EditingV2CandidateRoutingError(ValueError):
    """A candidate cannot be routed without inventing evidence semantics."""


@dataclass(frozen=True)
class CandidateLaneRoute:
    """One deterministic, non-authorizing lane-resolution decision."""

    data_lane: str
    evidence_profile_id: str
    route_reason: str
    routing_policy_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EditingV2CandidateRoutingError(
            "routing policy is not finite deterministic JSON"
        ) from error


def expected_routing_policy() -> dict[str, Any]:
    """Return the exact supported policy for validation and test construction."""

    return {
        "schema": ROUTING_POLICY_SCHEMA,
        "schema_version": ROUTING_POLICY_SCHEMA_VERSION,
        "policy_id": ROUTING_POLICY_ID,
        "source_kinds": list(SOURCE_KINDS),
        "active_families": list(ACTIVE8_FAMILIES),
        "attachment_or_topology_families": sorted(ATTACHMENT_OR_TOPOLOGY_FAMILIES),
        "rules_in_priority_order": [
            {
                "id": "executor_generated_walk",
                "source_kind": EXECUTOR_GENERATED_WALK_FROM_REAL_ENDPOINT,
                "lane": SYNTHETIC_WALK_LANE,
                "evidence_profile_id": SYNTHETIC_PROFILE,
            },
            {
                "id": "direct_attachment_or_topology",
                "source_kind": INFERRED_REAL_ENDPOINT_PAIR,
                "compiler_path_prefix": "direct_",
                "path_length": 1,
                "family_intersection": sorted(ATTACHMENT_OR_TOPOLOGY_FAMILIES),
                "lane": LINKER_TOPOLOGY_LANE,
                "evidence_profile_id": INFERRED_PROFILE,
            },
            {
                "id": "direct_non_topology_active8",
                "source_kind": INFERRED_REAL_ENDPOINT_PAIR,
                "compiler_path_prefix": "direct_",
                "path_length": 1,
                "lane": OPERATOR_AWARE_LANE,
                "evidence_profile_id": INFERRED_PROFILE,
            },
            {
                "id": "short_local_compiled_path",
                "source_kind": INFERRED_REAL_ENDPOINT_PAIR,
                "maximum_path_length": 2,
                "lane": LOCAL_ANALOGUE_LANE,
                "evidence_profile_id": INFERRED_PROFILE,
            },
            {
                "id": "real_endpoint_multistep_path",
                "source_kind": INFERRED_REAL_ENDPOINT_PAIR,
                "minimum_path_length": 3,
                "lane": MULTISTEP_LANE,
                "evidence_profile_id": INFERRED_PROFILE,
            },
        ],
    }


def validate_routing_policy(policy: Any) -> dict[str, Any]:
    """Validate and normalize one versioned routing-policy artifact."""

    if not isinstance(policy, dict) or set(policy) != _POLICY_FIELDS:
        raise EditingV2CandidateRoutingError(
            "routing policy fields disagree with the supported schema"
        )
    normalized = json.loads(_canonical_json_bytes(policy))
    if normalized != expected_routing_policy():
        raise EditingV2CandidateRoutingError(
            "routing policy disagrees with the frozen five-lane authority"
        )
    return normalized


def load_routing_policy(path: str | Path) -> dict[str, Any]:
    """Load one explicit routing policy; no implicit default is permitted."""

    try:
        payload = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise EditingV2CandidateRoutingError(f"cannot load routing policy {path}") from error
    return validate_routing_policy(payload)


def routing_policy_sha256(policy: dict[str, Any]) -> str:
    """Return the semantic identity of a validated routing policy."""

    return hashlib.sha256(_canonical_json_bytes(validate_routing_policy(policy))).hexdigest()


def _normalized_families(operator_families: Sequence[str]) -> tuple[str, ...]:
    if isinstance(operator_families, (str, bytes)) or not operator_families:
        raise EditingV2CandidateRoutingError("operator_families must be a nonempty sequence")
    families = tuple(str(family) for family in operator_families)
    if (
        any(family not in ACTIVE8_FAMILIES for family in families)
        or len(families) != len(set(families))
        or families != tuple(sorted(families))
    ):
        raise EditingV2CandidateRoutingError(
            "operator_families must be unique, sorted, and within Active8"
        )
    return families


def route_editing_v2_candidate(
    *,
    source_kind: str,
    compiler_path_class: str | None,
    path_length: int,
    operator_families: Sequence[str],
    policy: dict[str, Any],
) -> CandidateLaneRoute:
    """Route one candidate without using a caller-supplied lane label."""

    validate_routing_policy(policy)
    if source_kind not in SOURCE_KINDS:
        raise EditingV2CandidateRoutingError(f"unsupported hash-bound source kind: {source_kind!r}")
    if type(path_length) is not int or path_length < 1:
        raise EditingV2CandidateRoutingError("path_length must be a positive integer")
    families = _normalized_families(operator_families)
    policy_sha256 = routing_policy_sha256(policy)

    if source_kind == EXECUTOR_GENERATED_WALK_FROM_REAL_ENDPOINT:
        return CandidateLaneRoute(
            data_lane=SYNTHETIC_WALK_LANE,
            evidence_profile_id=SYNTHETIC_PROFILE,
            route_reason="executor_generated_walk",
            routing_policy_sha256=policy_sha256,
        )

    if not isinstance(compiler_path_class, str) or not compiler_path_class:
        raise EditingV2CandidateRoutingError("real-endpoint candidates require compiler_path_class")
    direct = compiler_path_class.startswith("direct_")
    if direct and path_length != 1:
        raise EditingV2CandidateRoutingError(
            "a direct compiler path must contain exactly one action"
        )
    if direct and ATTACHMENT_OR_TOPOLOGY_FAMILIES.intersection(families):
        route = CandidateLaneRoute(
            data_lane=LINKER_TOPOLOGY_LANE,
            evidence_profile_id=INFERRED_PROFILE,
            route_reason="direct_attachment_or_topology",
            routing_policy_sha256=policy_sha256,
        )
    elif direct:
        route = CandidateLaneRoute(
            data_lane=OPERATOR_AWARE_LANE,
            evidence_profile_id=INFERRED_PROFILE,
            route_reason="direct_non_topology_active8",
            routing_policy_sha256=policy_sha256,
        )
    elif path_length <= 2:
        route = CandidateLaneRoute(
            data_lane=LOCAL_ANALOGUE_LANE,
            evidence_profile_id=INFERRED_PROFILE,
            route_reason="short_local_compiled_path",
            routing_policy_sha256=policy_sha256,
        )
    else:
        route = CandidateLaneRoute(
            data_lane=MULTISTEP_LANE,
            evidence_profile_id=INFERRED_PROFILE,
            route_reason="real_endpoint_multistep_path",
            routing_policy_sha256=policy_sha256,
        )
    return route


def route_receipt_payload(route: CandidateLaneRoute) -> dict[str, str]:
    """Return deterministic fields suitable for a candidate-ledger receipt."""

    return asdict(route)
