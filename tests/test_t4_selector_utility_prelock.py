from __future__ import annotations

from copy import deepcopy

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_selector_utility_prelock import build_payloads

CELLS = ["5ht1b_0", "braf_1", "jak2_1", "parp1_0", "fa7_0"]
SUPPORTED = {"jak2_1", "parp1_0"}


def _contract() -> dict:
    return {
        "cells": CELLS,
        "arm": {
            "arm_id": "baseline_learned_complete_macro_selector",
            "source_policy_id": "balanced_joint_autoregressive",
            "ordering": "complete_macro_selector_reranked_learned_pool",
        },
        "selection_semantics": {
            "predeclared_maximum_supported_cells": sorted(SUPPORTED),
            "maximum_new_requests": 2,
        },
        "interpretation_limit": "development only",
    }


def _candidate(cell: str, rank: int, *, eligible: bool) -> dict:
    candidate_id = identity({"cell": cell, "rank": rank})
    source_case_id = identity({"source": cell})
    canonical = f"C{rank}N" if cell in SUPPORTED else f"C{rank}O"
    return {
        "candidate_uid": identity({"candidate": candidate_id, "cell": cell}),
        "candidate_id": candidate_id,
        "generator_revision": "baseline",
        "arm_id": "baseline_learned",
        "policy_id": "balanced_joint_autoregressive",
        "fold": 0,
        "cell": cell,
        "target": cell.split("_")[0],
        "source_case_id": source_case_id,
        "rank": rank,
        "source_state": {"source": cell},
        "source_state_sha256": identity({"source": cell}),
        "source_canonical_smiles": "CC",
        "endpoint_state": {"endpoint": candidate_id},
        "endpoint_state_sha256": identity({"endpoint": candidate_id}),
        "actions": [{"rule": "fixture"}],
        "goal": {"goal": candidate_id},
        "bindings": [],
        "construction_dependencies": [],
        "patch_ids": [candidate_id],
        "realization": {"status": "realized"},
        "proposal_score": float(-rank),
        "proposal_score_role": "fixture proposal",
        "exact_action_replay": True,
        "canonical_smiles": canonical,
        "active_atoms": 2,
        "heavy_atoms": 2,
        "non_self": True,
        "morgan_radius2_2048_similarity_to_source": 0.7,
        "qed": 0.7,
        "sa": 3.0,
        "eligible": eligible,
        "eligibility_exclusion_reasons": [] if eligible else ["fixture_ineligible"],
    }


def _fixture() -> tuple[dict, dict, dict]:
    candidates = {
        cell: [
            _candidate(cell, rank, eligible=cell in SUPPORTED and rank <= 3)
            for rank in range(1, 129)
        ]
        for cell in CELLS
    }
    selector = {
        "baseline_candidate_lock_payload_sha256": "baseline-payload",
        "folds": [
            {
                "fold": 0,
                "status": "fit",
                "cases": [
                    {
                        "source_case_id": rows[0]["source_case_id"],
                        "candidate_count": 128,
                        "ranked_candidates": [
                            {
                                "candidate_identity": row["candidate_id"],
                                "complete_macro_signature": [row["candidate_id"]],
                                "original_rank": row["rank"],
                                "selector_rank": row["rank"],
                                "selector_score": float(129 - row["rank"]),
                            }
                            for row in rows
                        ],
                    }
                    for rows in candidates.values()
                ],
            }
        ],
    }
    eligibility = {
        "candidate_locks": {"baseline": {"payload_sha256": "baseline-payload"}},
        "candidates": [row for rows in candidates.values() for row in rows],
    }
    requests = []
    for cell in sorted(SUPPORTED):
        row = candidates[cell][0]
        request_body = {
            "schema_version": "t4_docking_request_identity_v1",
            "target": row["target"],
            "canonical_smiles": row["canonical_smiles"],
            "docking_seed": 0,
            "evaluator": identity({"evaluator": row["target"]}),
        }
        requests.append(
            {
                "request_id": identity(request_body),
                **request_body,
                "evaluator_payload": {"target": row["target"], "score": "qvina"},
                "memberships": [
                    {
                        "membership_id": identity({"membership": cell}),
                        "arm_id": "baseline_learned",
                        "cell": cell,
                        "candidate_id": row["candidate_id"],
                        "rank": 1,
                    }
                ],
            }
        )
    prior_request = {"requests": requests}
    return selector, eligibility, prior_request


def _build(selector: dict, eligibility: dict, prior_request: dict):
    return build_payloads(
        contract=_contract(),
        selector=selector,
        eligibility=eligibility,
        prior_request=prior_request,
        provenance={
            "prelock": {"contract_sha256": "contract"},
            "immutable_inputs": {"fixture": True},
            "implementation": {"sha256": "implementation"},
        },
    )


def test_selector_prelock_uses_prior_raw_counterparts_and_selects_incremental_requests():
    selector, eligibility, prior_request = _fixture()
    candidate, request, abstention = _build(selector, eligibility, prior_request)

    assert len(candidate["memberships"]) == 2
    assert len(request["requests"]) == 2
    assert request["call_ceiling_if_later_authorized"] == 2
    assert request["automatic_retries"] == 0
    assert request["replacement_or_backfill"] is False
    assert {row["cell"] for row in candidate["memberships"]} == SUPPORTED
    assert {row["rank"] for row in candidate["memberships"]} == {2}
    assert all(row["raw_counterpart"]["rank"] == 1 for row in candidate["memberships"])
    statuses = {row["cell"]: row["status"] for row in abstention["cell_outcomes"]}
    assert {statuses[cell] for cell in set(CELLS) - SUPPORTED} == {
        "abstain_no_eligible_baseline_learned_candidate"
    }
    assert {statuses[cell] for cell in SUPPORTED} == {"selector_candidate_locked"}
    assert request["new_docking_calls"] == 0
    assert request["new_oracle_calls"] == 0


def test_selector_prelock_deduplicates_physical_request_identity_then_advances():
    selector, eligibility, prior_request = _fixture()
    parp_rows = [row for row in eligibility["candidates"] if row["cell"] == "parp1_0"]
    parp_rows[1]["canonical_smiles"] = parp_rows[0]["canonical_smiles"]

    candidate, _, abstention = _build(selector, eligibility, prior_request)

    selected = {row["cell"]: row for row in candidate["memberships"]}
    assert selected["parp1_0"]["rank"] == 3
    parp_status = next(row for row in abstention["cell_outcomes"] if row["cell"] == "parp1_0")
    assert parp_status["skipped_prior_request_identity"] == 1


def test_selector_prelock_abstains_when_raw_counterpart_is_missing():
    selector, eligibility, prior_request = _fixture()
    prior_request["requests"] = [
        row for row in prior_request["requests"] if row["target"] != "jak2"
    ]

    candidate, request, abstention = _build(selector, eligibility, prior_request)

    assert len(candidate["memberships"]) == len(request["requests"]) == 1
    jak2 = next(row for row in abstention["cell_outcomes"] if row["cell"] == "jak2_1")
    assert jak2["status"] == "abstain_missing_raw_learned_counterpart"


def test_selector_prelock_fails_on_incomplete_candidate_join():
    selector, eligibility, prior_request = _fixture()
    broken = deepcopy(selector)
    broken["folds"][0]["cases"][0]["ranked_candidates"][0]["candidate_identity"] = "missing"

    with pytest.raises(ValueError, match="join is not total"):
        _build(broken, eligibility, prior_request)


def test_selector_prelock_never_emits_prospective_scores():
    selector, eligibility, prior_request = _fixture()
    payloads = _build(selector, eligibility, prior_request)

    encoded = repr(payloads)
    assert "docking_score'" not in encoded
    assert "task_score'" not in encoded
    assert "UNAUTHORIZED_NOT_LAUNCHED" in encoded
