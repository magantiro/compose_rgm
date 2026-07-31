from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from compose_v4.data.editing_corpus_contract import (
    load_editing_corpus_contract,
)
from compose_v4.data.editing_v2_candidate_router import (
    EXECUTOR_GENERATED_WALK_FROM_REAL_ENDPOINT,
    INFERRED_REAL_ENDPOINT_PAIR,
    EditingV2CandidateRoutingError,
    expected_routing_policy,
    load_routing_policy,
    route_editing_v2_candidate,
    routing_policy_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "editing_corpus_v2_contract.json"
ROUTING_POLICY_PATH = ROOT / "configs" / "editing_v2_candidate_routing_policy_v1.json"


@pytest.fixture()
def policy() -> dict:
    return load_routing_policy(ROUTING_POLICY_PATH)


def test_inferred_relationship_is_admissible_without_becoming_observed() -> None:
    contract = load_editing_corpus_contract(CONTRACT_PATH)
    lanes = {lane["id"]: lane for lane in contract["data_lanes"]}
    for lane_id in (
        "observed_local_analogue",
        "operator_aware_real_endpoint",
    ):
        assert "inferred_relation_compiled_path" in lanes[lane_id]["admissible_evidence_profiles"]
    inferred = next(
        profile
        for profile in contract["evidence_component_contract"]["profiles"]
        if profile["id"] == "inferred_relation_compiled_path"
    )
    assert inferred["components"]["pair_relationship"] == "structurally_inferred_relationship"


@pytest.mark.parametrize(
    ("compiler_path_class", "path_length", "families", "lane", "reason"),
    [
        (
            "direct_atom_restate",
            1,
            ("atom_restate",),
            "operator_aware_real_endpoint",
            "direct_non_topology_active8",
        ),
        (
            "direct_bond_reroute",
            1,
            ("bond_reroute",),
            "linker_positional_topology_analogue",
            "direct_attachment_or_topology",
        ),
        (
            "delete_insert_fallback",
            2,
            ("atom_delete", "atom_insert"),
            "observed_local_analogue",
            "short_local_compiled_path",
        ),
        (
            "delete_insert_fallback",
            3,
            ("atom_delete", "atom_insert"),
            "real_endpoint_multistep_path",
            "real_endpoint_multistep_path",
        ),
    ],
)
def test_real_endpoint_routes_are_deterministic(
    compiler_path_class: str,
    path_length: int,
    families: tuple[str, ...],
    lane: str,
    reason: str,
    policy: dict,
) -> None:
    route = route_editing_v2_candidate(
        source_kind=INFERRED_REAL_ENDPOINT_PAIR,
        compiler_path_class=compiler_path_class,
        path_length=path_length,
        operator_families=families,
        policy=policy,
    )
    assert route.data_lane == lane
    assert route.evidence_profile_id == "inferred_relation_compiled_path"
    assert route.route_reason == reason
    assert route.routing_policy_sha256 == routing_policy_sha256(policy)


def test_synthetic_walk_never_inherits_real_pair_evidence(policy: dict) -> None:
    route = route_editing_v2_candidate(
        source_kind=EXECUTOR_GENERATED_WALK_FROM_REAL_ENDPOINT,
        compiler_path_class=None,
        path_length=4,
        operator_families=("atom_delete", "cycle_attach"),
        policy=policy,
    )
    assert route.data_lane == "reversible_synthetic_walk"
    assert route.evidence_profile_id == "executor_generated_walk"


def test_router_rejects_off_contract_or_ambiguous_inputs(policy: dict) -> None:
    with pytest.raises(EditingV2CandidateRoutingError, match="source kind"):
        route_editing_v2_candidate(
            source_kind="caller_selected_lane",
            compiler_path_class="direct_atom_restate",
            path_length=1,
            operator_families=("atom_restate",),
            policy=policy,
        )
    with pytest.raises(EditingV2CandidateRoutingError, match="unique, sorted"):
        route_editing_v2_candidate(
            source_kind=INFERRED_REAL_ENDPOINT_PAIR,
            compiler_path_class="delete_insert_fallback",
            path_length=2,
            operator_families=("atom_insert", "atom_delete"),
            policy=policy,
        )
    with pytest.raises(EditingV2CandidateRoutingError, match="exactly one"):
        route_editing_v2_candidate(
            source_kind=INFERRED_REAL_ENDPOINT_PAIR,
            compiler_path_class="direct_bond_reroute",
            path_length=2,
            operator_families=("bond_reroute",),
            policy=policy,
        )


def test_routing_policy_is_complete_and_content_addressed(policy: dict) -> None:
    assert len(policy["rules_in_priority_order"]) == 5
    assert policy == expected_routing_policy()
    assert len(routing_policy_sha256(policy)) == 64
    drifted = copy.deepcopy(policy)
    drifted["rules_in_priority_order"][-1]["minimum_path_length"] = 4
    payload = json.dumps(
        drifted,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    import hashlib

    assert hashlib.sha256(payload).hexdigest() != routing_policy_sha256(policy)
