import copy

import pytest

from compose_v4.experiments.t4_shared_controller_cell_runtime import (
    QUERY_RECEIPT_SCHEMA,
    make_query_lock,
    make_query_reservation,
    settle_query_lock,
)
from compose_v4.experiments.t4_shared_controller_checkpoint import (
    initialize_after_settled_root,
)
from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
)
from compose_v4.experiments.t4_shared_controller_scored_runtime import (
    AUTHORIZATION_SCHEMA,
    RETAINED_CORE_CONFIG,
    attach_endpoint_fingerprints,
    make_launch_task,
    make_proposal_manifest,
    preview_selected_parents,
    retained_core_route_records,
    validate_authorization_receipt,
    validate_launch_task,
)

CELL = {
    "cell_key": "parp1_0_d04",
    "source_smiles": "CCOC(=O)NCC",
    "delta": 0.4,
    "controller_seed": 2026091900,
}
CONTROLLER = {
    "parents": 4,
    "parent_explore": 0.3,
    "batch": 8,
    "exploration": 2,
    "expert_floor_rounds": 2,
    "route_scale_floor_rounds": 2,
    "value_penalty": 1.0,
    "docking_seed": 20260919,
    "proposal": {
        "shallow": {"draws": 480, "horizon": 3},
        "anchored_replacement": {"draws": 512, "horizon": 3},
        "route_complete_region": {"pool_size": 192},
    },
}
CONTRACT = {
    "scored_calls_requested": 441,
    "charged_calls_per_cell": 49,
    "total_charged_call_ceiling": 441,
    "cells": [{**CELL, "cell_key": f"cell_{index}"} for index in range(9)],
    "controller": CONTROLLER,
}
CONTRACT_HASH = "a" * 64
CONTRACT_FILE_HASH = "b" * 64
AUTHORIZATION_FILE_HASH = "c" * 64
CAPSULE_HASH = "d" * 64


def _authorization():
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "contract_payload_sha256": CONTRACT_HASH,
        "authorized_scored_calls": 441,
        "user_statement": "exact payload-bound authorization fixture",
    }


def _checkpoint():
    lock = make_query_lock(
        cell_key=CELL["cell_key"],
        round_index=0,
        charged_before=0,
        selected_rows=[{"smiles": CELL["source_smiles"]}],
        query_deadline=1.0,
    )
    query = lock["queries"][0]
    receipt = {
        **make_query_reservation(lock, query["query_id"]),
        "schema_version": QUERY_RECEIPT_SCHEMA,
        "status": "complete",
        "answer": {"score": -8.0, "failure": None},
    }
    settled = settle_query_lock(lock, {query["query_id"]: receipt}, now=1.0)
    return initialize_after_settled_root(
        cell_key=CELL["cell_key"],
        source_smiles=CELL["source_smiles"],
        delta=CELL["delta"],
        budget_ceiling=49,
        controller_config=CONTROLLER,
        controller_seed=CELL["controller_seed"],
        root_query_lock=lock,
        settled_root=settled,
    )


def test_launch_requires_separate_exact_payload_authorization():
    authorization = _authorization()
    assert (
        validate_authorization_receipt(
            authorization,
            contract_payload_sha256=CONTRACT_HASH,
            total_charged_call_ceiling=441,
        )
        == authorization
    )
    task = make_launch_task(
        contract=CONTRACT,
        contract_payload_sha256=CONTRACT_HASH,
        contract_file_sha256=CONTRACT_FILE_HASH,
        authorization_receipt=authorization,
        authorization_receipt_sha256=AUTHORIZATION_FILE_HASH,
        code_revision="1" * 40,
        source_capsule_payload_sha256=CAPSULE_HASH,
    )
    assert task["total_charged_call_ceiling"] == 441
    assert len(task["cell_keys"]) == 9
    assert (
        validate_launch_task(
            task,
            contract=CONTRACT,
            contract_payload_sha256=CONTRACT_HASH,
            contract_file_sha256=CONTRACT_FILE_HASH,
            authorization_receipt=authorization,
            authorization_receipt_sha256=AUTHORIZATION_FILE_HASH,
            source_capsule_payload_sha256=CAPSULE_HASH,
        )
        == task
    )

    wrong = copy.deepcopy(authorization)
    wrong["contract_payload_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="another scored contract"):
        validate_authorization_receipt(
            wrong,
            contract_payload_sha256=CONTRACT_HASH,
            total_charged_call_ceiling=441,
        )
    blocked = {**CONTRACT, "scored_calls_requested": 0}
    with pytest.raises(RuntimeError, match="no nonzero authorized payload"):
        make_launch_task(
            contract=blocked,
            contract_payload_sha256=CONTRACT_HASH,
            contract_file_sha256=CONTRACT_FILE_HASH,
            authorization_receipt=authorization,
            authorization_receipt_sha256=AUTHORIZATION_FILE_HASH,
            code_revision="1" * 40,
            source_capsule_payload_sha256=CAPSULE_HASH,
        )


def test_parent_preview_and_proposal_manifest_do_not_advance_checkpoint():
    checkpoint = _checkpoint()
    contract = {**CONTRACT, "cells": [CELL]}
    first = preview_selected_parents(checkpoint, cell=CELL, contract=contract)
    second = preview_selected_parents(checkpoint, cell=CELL, contract=contract)
    assert first == second == [CELL["source_smiles"]]
    assert checkpoint["rounds_completed"] == 0
    particle = {
        CELL["source_smiles"]: {
            "payload": {"job_count": 28},
            "payload_sha256": "e" * 64,
        }
    }
    manifest = make_proposal_manifest(
        checkpoint=checkpoint,
        cell=CELL,
        contract=contract,
        round_index=1,
        deadline=10.0,
        particle_manifests=particle,
    )
    assert manifest["parents"] == first
    assert len(manifest["requests"]) == 3
    assert manifest["proposal_experts"] == [
        "shallow",
        "anchored_replacement",
        "route_complete_region",
    ]
    assert manifest["retained_core"] == RETAINED_CORE_CONFIG
    assert manifest["base_checkpoint_payload_sha256"] == payload_identity(checkpoint)


def test_retained_core_is_tagged_inside_route_expert_and_fingerprinted():
    records, telemetry = retained_core_route_records(
        parent=CELL["source_smiles"],
        parent_score=-8.0,
        original_seed=CELL["source_smiles"],
        delta=0.4,
        support="compose_valid",
    )
    assert telemetry["program_family"] == "retained_core_prune"
    assert telemetry["proposal_expert"] == "route_complete_region"
    for row in records:
        assert row["proposal_lane"] == "route_complete_region"
        assert row["proposal_experts"] == ["route_complete_region"]
        assert row["program_families"] == ["retained_core_prune"]
        assert row["realized_primitives"] <= 32
    prepared = attach_endpoint_fingerprints(
        records,
        original_seed=CELL["source_smiles"],
        delta=0.4,
        support="compose_valid",
    )
    assert len(prepared) == len(records)
    assert all(isinstance(row["fingerprint"], list) for row in prepared)
