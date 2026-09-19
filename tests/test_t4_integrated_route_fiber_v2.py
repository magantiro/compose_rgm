from compose_v4.experiments.t4_integrated_route_fiber_v2 import (
    EXPERTS,
    durable_phase_action,
    expert_census,
    merge_expert_pools,
    proposal_collection_action,
)


def _row(smiles, lane, score=-8.0):
    return {
        "smiles": smiles,
        "proposal_lane": lane,
        "parent": "CC",
        "parent_score": score,
        "families": [lane],
        "program_families": [lane],
    }


def test_v2_union_keeps_retained_core_provenance():
    pools = {name: [] for name in EXPERTS}
    pools["shallow"] = [_row("CCO", "shallow")]
    pools["retained_core_prune"] = [_row("CCO", "retained_core_prune", -9.0)]
    merged = merge_expert_pools(pools)
    assert len(merged) == 1
    assert merged[0]["proposal_experts"] == ["retained_core_prune", "shallow"]
    census = expert_census(merged)
    assert census["retained_core_prune"] == 1
    assert census["multi_expert"] == 1


def test_durable_phase_starts_only_before_lock_and_recovers_complete_receipts():
    assert durable_phase_action(lock_exists=False, receipt_statuses=[]) == "start"
    assert (
        durable_phase_action(
            lock_exists=True, receipt_statuses=["complete", "complete"]
        )
        == "recover"
    )


def test_durable_phase_fails_closed_without_resubmitting_locked_query():
    assert (
        durable_phase_action(
            lock_exists=True, receipt_statuses=["complete", "reserved"]
        )
        == "fail_closed"
    )


def test_proposal_collector_never_waits_past_frozen_deadline():
    assert (
        proposal_collection_action(
            receipt_statuses=["complete", "failed", "complete"],
            now=9.0,
            deadline=10.0,
        )
        == "collect"
    )
    assert (
        proposal_collection_action(
            receipt_statuses=["complete", "running", "missing"],
            now=9.0,
            deadline=10.0,
        )
        == "wait"
    )
    assert (
        proposal_collection_action(
            receipt_statuses=["complete", "running", "missing"],
            now=10.0,
            deadline=10.0,
        )
        == "collect_with_abstentions"
    )
    assert (
        durable_phase_action(lock_exists=True, receipt_statuses=["missing"])
        == "fail_closed"
    )
