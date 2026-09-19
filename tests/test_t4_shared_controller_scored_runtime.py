import copy
import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.experiments.t4_integrated_route_fiber import (
    EXPERTS,
    PROTONATION_AWARE_EXPERT,
)
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
    RETAINED_CORE_REFINE_CONFIG,
    admit_runtime_proposal_pools,
    attach_endpoint_fingerprints,
    attach_generic_scale_band,
    make_launch_task,
    make_proposal_manifest,
    preview_selected_parents,
    protonation_aware_records,
    retained_core_prune_refine_records,
    retained_core_route_records,
    validate_authorization_receipt,
    validate_launch_task,
)

ROOT = Path(__file__).resolve().parents[1]

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


def test_generic_scale_band_uses_exact_programs_and_structural_extent():
    exact = attach_generic_scale_band(
        {"realized_primitives": 14, "created": 1, "deleted": 0, "regions": 1}
    )
    assert exact["proposal_scale_band"] == "large"
    assert exact["proposal_scale_basis"] == "exact_protected_primitive_count"
    structural = attach_generic_scale_band({"created": 5, "deleted": 2, "regions": 2})
    assert structural["proposal_scale_band"] == "medium"
    assert structural["proposal_scale_extent"] == 7
    assert structural["proposal_scale_basis"] == "realized_structural_role_extent"


def _authorization():
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "contract_payload_sha256": CONTRACT_HASH,
        "authorized_scored_calls": 441,
        "user_statement": "exact payload-bound authorization fixture",
    }


def _checkpoint(controller=CONTROLLER):
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
        controller_config=controller,
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


def test_fourth_expert_is_appended_without_changing_existing_seed_streams():
    checkpoint = _checkpoint()
    legacy_contract = {**CONTRACT, "cells": [CELL], "support": "compose_valid"}
    legacy = make_proposal_manifest(
        checkpoint=checkpoint,
        cell=CELL,
        contract=legacy_contract,
        round_index=1,
        deadline=10.0,
        particle_manifests={
            CELL["source_smiles"]: {
                "payload": {"job_count": 28},
                "payload_sha256": "e" * 64,
            }
        },
    )
    fourth_controller = copy.deepcopy(CONTROLLER)
    fourth_controller["proposal_experts"] = [*EXPERTS, PROTONATION_AWARE_EXPERT]
    fourth_controller["proposal"][PROTONATION_AWARE_EXPERT] = {"shallow_draws": 1}
    extended = make_proposal_manifest(
        checkpoint=_checkpoint(fourth_controller),
        cell=CELL,
        contract={
            **legacy_contract,
            "controller": fourth_controller,
            "trajectory_distillation": {"enabled": True},
        },
        round_index=1,
        deadline=10.0,
        particle_manifests={
            CELL["source_smiles"]: {
                "payload": {"job_count": 28},
                "payload_sha256": "e" * 64,
            }
        },
    )

    assert extended["proposal_experts"] == [*EXPERTS, PROTONATION_AWARE_EXPERT]
    keys = ("parent", "parent_index", "expert", "proposal_seed")
    assert [{key: row[key] for key in keys} for row in extended["requests"][:3]] == [
        {key: row[key] for key in keys} for row in legacy["requests"]
    ]
    assert extended["requests"][3]["expert"] == PROTONATION_AWARE_EXPERT


def _stale_binding(*, include: str) -> dict:
    def hashes(label: str, count: int, extra: tuple[str, ...] = ()) -> list[str]:
        values = set(extra)
        index = 0
        while len(values) < count:
            values.add(hashlib.sha256(f"{label}-{index}".encode()).hexdigest())
            index += 1
        return sorted(values)

    return {
        "excluded_canonical_smiles_sha256": {
            "braf_0": hashes("braf-0", 20, (include,)),
            "braf_1": hashes("braf-1", 23),
        }
    }


def test_runtime_admission_keeps_unseen_braf_route_after_stale_filter():
    source = "CCN(CC)CCNC(=O)c1cnn2c(-c3cccc(NC(=O)Nc4ccc(Cl)c(C(F)(F)F)c4)c3)ccnc12"
    unseen = "CCCNC(=O)c1cnn2c(-c3cccc(NC(=O)NC)c3)ccnc12"
    stale = "CCN(CC)CCNC(=O)c1cnn2c(-c3cccc(NC)c3)ccnc12"
    contract = {
        "support": "compose_valid",
        "controller": CONTROLLER,
        "stale_braf_v4_reconciliation": _stale_binding(
            include=hashlib.sha256(stale.encode()).hexdigest()
        ),
    }
    base = {
        "parent": source,
        "parent_score": 0.0,
        "families": ["route_complete_region"],
        "program_families": ["route_complete_region"],
        "realized_primitives": 16,
        "realized_primitive_band": "large",
    }
    pools, ledger = admit_runtime_proposal_pools(
        {
            "shallow": [],
            "anchored_replacement": [],
            "route_complete_region": [
                {**base, "smiles": stale},
                {**base, "smiles": unseen},
            ],
        },
        cell={
            "target": "braf",
            "source_cell": "braf_0",
            "source_smiles": source,
            "delta": 0.6,
        },
        contract=contract,
    )

    assert [row["smiles"] for row in pools["route_complete_region"]] == [unseen]
    assert ledger["fiber_admitted_before_stale_by_expert"]["route_complete_region"] == 2
    assert ledger["stale_query_exclusions"]["excluded_total"] == 1
    assert ledger["selection_ready_by_expert"]["route_complete_region"] == 1


def test_protonation_adapter_and_runtime_admission_are_generic(monkeypatch):
    source = "C1=CC2=NC=C(CCCN3CC[NH+](CCc4ccccc4)CC3)[C@H]2C=C1n1cnnc1"
    endpoint = "C1=CC2=NC=C(CCCN3CCN(CCc4ccccc4)CC3)C2C=C1n1cnnc1"

    monkeypatch.setattr(
        "compose_v4.experiments.t4_shared_controller_scored_runtime."
        "propose_protonation_aware_candidates",
        lambda _source, _route_expert, config: (
            [
                {
                    "smiles": endpoint,
                    "actions": [{"executor_rule": "atom_protonation_restate"}],
                    "primitive_edits": 1,
                    "program_families": [
                        "atom_protonation_restate",
                        "charge_only",
                    ],
                    "program_kind": "charge_only",
                    "structural_lane": "charge_only",
                }
            ],
            {"exact_execution_precision_numerator": 1},
        ),
    )
    settings = {
        "shallow_draws": 1,
        "retained_maximum_fragment_atoms": 16,
        "retained_maximum_stages": 2,
        "retained_maximum_prefixes": 16,
        "route_pool_size": 1,
        "route_realization_limit": 1,
        "route_beam_width": 1,
        "route_expansion_width": 1,
        "route_max_bindings_per_template": 1,
        "route_maximum_expansions": 1,
        "route_candidate_timeout_seconds": 1.0,
    }
    records, telemetry = protonation_aware_records(
        parent=source,
        parent_score=0.0,
        original_seed=source,
        delta=0.6,
        support="compose_valid",
        proposal_seed_value=7,
        route_expert=object(),
        settings=settings,
    )
    controller = copy.deepcopy(CONTROLLER)
    controller["proposal_experts"] = [*EXPERTS, PROTONATION_AWARE_EXPERT]
    controller["proposal"][PROTONATION_AWARE_EXPERT] = settings
    pools, ledger = admit_runtime_proposal_pools(
        {
            "shallow": [],
            "anchored_replacement": [],
            "route_complete_region": [],
            PROTONATION_AWARE_EXPERT: records,
        },
        cell={
            "target": "5ht1b",
            "source_cell": "5ht1b_2",
            "source_smiles": source,
            "delta": 0.6,
        },
        contract={
            "support": "compose_valid",
            "controller": controller,
            "trajectory_distillation": {"enabled": True},
        },
    )

    assert telemetry["fiber_admitted_unique"] == 1
    assert [row["smiles"] for row in pools[PROTONATION_AWARE_EXPERT]] == [endpoint]
    assert ledger["selection_ready_by_expert"][PROTONATION_AWARE_EXPERT] == 1


def test_nodistill_launch_and_manifest_disable_only_route_template_particles():
    checkpoint = _checkpoint()
    contract = {
        **CONTRACT,
        "campaign_cell_count": 3,
        "scored_calls_requested": 147,
        "total_charged_call_ceiling": 147,
        "cells": [{**CELL, "cell_key": f"cell_{index}"} for index in range(3)],
        "trajectory_distillation": {"enabled": False},
    }
    authorization = {
        **_authorization(),
        "authorized_scored_calls": 147,
    }
    task = make_launch_task(
        contract=contract,
        contract_payload_sha256=CONTRACT_HASH,
        contract_file_sha256=CONTRACT_FILE_HASH,
        authorization_receipt=authorization,
        authorization_receipt_sha256=AUTHORIZATION_FILE_HASH,
        code_revision="1" * 40,
        source_capsule_payload_sha256=CAPSULE_HASH,
    )
    assert task["cell_keys"] == ["cell_0", "cell_1", "cell_2"]
    manifest = make_proposal_manifest(
        checkpoint=checkpoint,
        cell=CELL,
        contract={**contract, "cells": [CELL]},
        round_index=1,
        deadline=10.0,
        particle_manifests={},
    )
    assert manifest["trajectory_distillation_enabled"] is False
    assert manifest["particle_parent_manifest_sha256"] == {}
    assert manifest["proposal_experts"] == [
        "shallow",
        "anchored_replacement",
        "route_complete_region",
    ]
    with pytest.raises(ValueError, match="may not schedule"):
        make_proposal_manifest(
            checkpoint=checkpoint,
            cell=CELL,
            contract={**contract, "cells": [CELL]},
            round_index=1,
            deadline=10.0,
            particle_manifests={CELL["source_smiles"]: {"payload_sha256": "e" * 64}},
        )


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


def test_retained_core_refinement_creates_exact_novel_deployment_candidates():
    contract = json.loads(
        (
            ROOT
            / "diagnostics/t4_shared_controller_completion_v1/scored_contract_v4.json"
        ).read_text()
    )["payload"]
    cell = next(row for row in contract["cells"] if row["cell_key"] == "braf_0_d06")
    denied = contract["stale_braf_v4_reconciliation"][
        "excluded_canonical_smiles_sha256"
    ]["braf_0"]

    records, telemetry = retained_core_prune_refine_records(
        parent=cell["source_smiles"],
        parent_score=-8.9,
        original_seed=cell["source_smiles"],
        delta=cell["delta"],
        support=contract["support"],
        excluded_canonical_smiles_sha256=denied,
        refinement_rules=("bond_reorder",),
    )

    assert records
    assert telemetry["eligible_novel_unique"] == len(records)
    assert telemetry["prior_scores_or_receipts_read"] is False
    assert telemetry["configuration"]["refinement_rules"] == ["bond_reorder"]
    assert set(telemetry["configuration"]["refinement_rules"]) <= set(
        RETAINED_CORE_REFINE_CONFIG["refinement_rules"]
    )
    assert all(
        hashlib.sha256(row["smiles"].encode()).hexdigest() not in denied
        for row in records
    )
    assert all(row["realized_primitives"] <= 32 for row in records)
    assert all(row["intermediate_endpoint_queried"] is False for row in records)
    assert all(
        row["protected_program_sha256"]
        == payload_identity(row["protected_program_actions"])
        for row in records
    )
