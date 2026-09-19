from compose_v4.experiments.t4_integrated_route_fiber_v2 import (
    EXPERTS,
    durable_phase_action,
    expert_census,
    merge_expert_pools,
    migrate_v1_checkpoint,
    proposal_collection_action,
    query_collection_action,
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
        durable_phase_action(lock_exists=True, receipt_statuses=["complete", "complete"])
        == "recover"
    )


def test_durable_phase_fails_closed_without_resubmitting_locked_query():
    assert (
        durable_phase_action(lock_exists=True, receipt_statuses=["complete", "reserved"])
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


def test_query_collector_resumes_complete_and_never_resubmits_unresolved():
    assert (
        query_collection_action(lock_exists=False, receipt_statuses=[], now=0.0, deadline=None)
        == "start"
    )
    assert (
        query_collection_action(
            lock_exists=True,
            receipt_statuses=["complete", "complete"],
            now=2.0,
            deadline=10.0,
        )
        == "recover"
    )
    assert (
        query_collection_action(
            lock_exists=True,
            receipt_statuses=["complete", "reserved", "missing"],
            now=9.0,
            deadline=10.0,
        )
        == "wait"
    )
    assert (
        query_collection_action(
            lock_exists=True,
            receipt_statuses=["complete", "reserved", "missing"],
            now=10.0,
            deadline=10.0,
        )
        == "recover_with_unresolved"
    )
    assert durable_phase_action(lock_exists=True, receipt_statuses=["missing"]) == "fail_closed"


def test_v1_checkpoint_migration_preserves_calls_state_and_provenance():
    legacy = {
        "schema_version": "t4_integrated_route_fiber_checkpoint_v1",
        "status": "running",
        "cell": "example_0",
        "contract_payload_sha256": "old-contract",
        "charged_calls": 9,
        "budget_remaining": 40,
        "archive": {"CC": -7.0, "CCC": -8.0},
        "features": [[1.0], [2.0]],
        "improvements": [0.1, 1.0],
        "history": [{"round": 1, "improved": True}],
        "rounds": [
            {
                "round": 1,
                "charged_calls": 9,
                "charged_this_round": 8,
                "candidate_lock_payload_sha256": "round-lock-payload",
            }
        ],
        "rng_state": {"bit_generator": "PCG64", "state": {"state": 1, "inc": 3}},
    }
    migrated = migrate_v1_checkpoint(
        legacy,
        cell="example_0",
        original_seed="CC",
        new_contract_payload_sha256="new-contract",
        charged_call_ceiling=49,
        legacy_run_id="old-run",
        legacy_checkpoint_sha256="checkpoint-file",
        legacy_checkpoint_payload_sha256="checkpoint-payload",
        legacy_round_lock_sha256="round-lock-file",
        legacy_round_lock_payload_sha256="round-lock-payload",
    )
    assert migrated["charged_calls"] == 9
    assert migrated["budget_remaining"] == 40
    assert migrated["rounds_completed"] == 1
    assert migrated["archive"] == {"CC": -7.0, "CCC": -8.0}
    assert migrated["root_result"]["score"] == -7.0
    assert migrated["root_result"]["recovered_from"] == ("sealed_v1_archive_source_entry")
    assert migrated["migration_provenance"] == {
        "legacy_run_id": "old-run",
        "legacy_contract_payload_sha256": "old-contract",
        "legacy_checkpoint_sha256": "checkpoint-file",
        "legacy_checkpoint_payload_sha256": "checkpoint-payload",
        "legacy_round_lock_sha256": "round-lock-file",
        "legacy_round_lock_payload_sha256": "round-lock-payload",
        "legacy_charged_calls": 9,
    }


def test_v1_checkpoint_migration_rejects_unbound_round_lock():
    legacy = {
        "schema_version": "t4_integrated_route_fiber_checkpoint_v1",
        "status": "running",
        "cell": "example_0",
        "contract_payload_sha256": "old-contract",
        "charged_calls": 9,
        "budget_remaining": 40,
        "archive": {"CC": -7.0},
        "features": [],
        "improvements": [],
        "history": [{"round": 1, "improved": False}],
        "rounds": [
            {
                "round": 1,
                "charged_calls": 9,
                "charged_this_round": 8,
                "candidate_lock_payload_sha256": "different-lock",
            }
        ],
        "rng_state": {"state": 1},
    }
    try:
        migrate_v1_checkpoint(
            legacy,
            cell="example_0",
            original_seed="CC",
            new_contract_payload_sha256="new-contract",
            charged_call_ceiling=49,
            legacy_run_id="old-run",
            legacy_checkpoint_sha256="checkpoint-file",
            legacy_checkpoint_payload_sha256="checkpoint-payload",
            legacy_round_lock_sha256="round-lock-file",
            legacy_round_lock_payload_sha256="round-lock-payload",
        )
    except ValueError as error:
        assert "final round lock" in str(error)
    else:
        raise AssertionError("migration accepted an unbound final round lock")
