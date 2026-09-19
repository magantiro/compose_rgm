import copy
import json

import pytest

from compose_v4.experiments.t4_integrated_route_fiber import (
    EXPERTS,
    PROTONATION_AWARE_EXPERT,
)
from compose_v4.experiments.t4_locked_round_recovery import recover_locked_round
from compose_v4.experiments.t4_shared_controller_cell_runtime import (
    QUERY_RECEIPT_SCHEMA,
    make_query_lock,
    make_query_reservation,
    settle_query_lock,
)
from compose_v4.experiments.t4_shared_controller_checkpoint import (
    CANDIDATE_EXHAUSTION,
    RUNNING,
    admit_candidate_union,
    apply_settled_observations,
    initialize_after_settled_root,
    restore_checkpoint,
    select_parent_and_batch,
)
from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
)

CELL = "parp1_0_d04"
SOURCE = "CN(C)C"
DELTA = 0.4
BUDGET = 49
CONFIG = {
    "parents": 4,
    "parent_explore": 0.25,
    "batch": 8,
    "exploration": 2,
    "expert_floor_rounds": 2,
    "route_scale_floor_rounds": 2,
    "value_penalty": 1.0,
}


def _settled_lock(lock, scores):
    receipts = {}
    for query, score in zip(lock["queries"], scores, strict=True):
        reservation = make_query_reservation(lock, query["query_id"])
        receipts[query["query_id"]] = {
            **reservation,
            "schema_version": QUERY_RECEIPT_SCHEMA,
            "status": "complete",
            "answer": {"score": score, "failure": None},
        }
    return settle_query_lock(lock, receipts, now=lock["query_deadline"])


def _initial_checkpoint():
    root_lock = make_query_lock(
        cell_key=CELL,
        round_index=0,
        charged_before=0,
        selected_rows=[{"smiles": SOURCE, "proposal_experts": []}],
        query_deadline=1.0,
    )
    return initialize_after_settled_root(
        cell_key=CELL,
        source_smiles=SOURCE,
        delta=DELTA,
        budget_ceiling=BUDGET,
        controller_config=CONFIG,
        controller_seed=2026091900,
        root_query_lock=root_lock,
        settled_root=_settled_lock(root_lock, [-8.0]),
    )


def _proposal(smiles, expert, parent, parent_score, marker):
    band = ("small", "medium", "large")[marker % 3]
    return {
        "smiles": smiles,
        "proposal_lane": expert,
        "proposal_experts": [expert],
        "parent": parent,
        "parent_score": parent_score,
        "similarity": 0.7,
        "delta": DELTA,
        "qed": 0.7,
        "sa": 3.0,
        "regions": 1 + marker % 2,
        "created": marker % 3,
        "deleted": marker % 2,
        "families": [expert],
        "program_families": [expert],
        "fingerprint": [marker, marker + 100],
        "realized_primitive_band": band,
        "route_proposal_rank": marker + 1,
    }


def test_explicit_fourth_expert_uses_exact_seen_and_cross_expert_dedup_path():
    checkpoint = _initial_checkpoint()
    state = restore_checkpoint(
        checkpoint,
        cell_key=CELL,
        source_smiles=SOURCE,
        delta=DELTA,
        budget_ceiling=BUDGET,
        controller_config=CONFIG,
    ).state
    experts = (*EXPERTS, PROTONATION_AWARE_EXPERT)
    pools = {expert: [] for expert in experts}
    pools["shallow"] = [_proposal("N", "shallow", SOURCE, -8.0, 1)]
    pools[PROTONATION_AWARE_EXPERT] = [
        _proposal("N", PROTONATION_AWARE_EXPERT, SOURCE, -8.0, 2),
        _proposal(SOURCE, PROTONATION_AWARE_EXPERT, SOURCE, -8.0, 3),
    ]

    admitted = admit_candidate_union(
        pools,
        state=state,
        parents=[SOURCE],
        experts=experts,
    )

    assert [row["smiles"] for row in admitted] == ["N"]
    assert admitted[0]["proposal_experts"] == [
        PROTONATION_AWARE_EXPERT,
        "shallow",
    ]
    assert len(admitted[0]["features"]) == 20


def _pools(checkpoint, generation):
    pools = {
        "shallow": [],
        "anchored_replacement": [],
        "route_complete_region": [],
    }
    experts = tuple(pools)
    marker = generation * 1_000
    for parent_index, (parent, score) in enumerate(checkpoint["archive"].items()):
        for expert_index, expert in enumerate(experts):
            for item in range(3):
                value = marker + parent_index * 100 + expert_index * 10 + item
                pools[expert].append(
                    _proposal(
                        f"{expert[0]}{generation}{parent_index}{item}",
                        expert,
                        parent,
                        score,
                        value,
                    )
                )
    return pools


def _plan(checkpoint, generation):
    return select_parent_and_batch(
        checkpoint,
        _pools(checkpoint, generation),
        cell_key=CELL,
        source_smiles=SOURCE,
        delta=DELTA,
        budget_ceiling=BUDGET,
        controller_config=CONFIG,
        query_deadline=float(generation + 10),
    )


def _apply(checkpoint, plan, base_score):
    scores = [base_score - 0.1 * index for index in range(len(plan["selected_rows"]))]
    settlement = _settled_lock(plan["query_lock"], scores)
    return apply_settled_observations(
        checkpoint,
        plan,
        settlement,
        cell_key=CELL,
        source_smiles=SOURCE,
        delta=DELTA,
        budget_ceiling=BUDGET,
        controller_config=CONFIG,
    )


def test_initial_checkpoint_round_trips_all_scientific_state():
    checkpoint = _initial_checkpoint()
    restored = restore_checkpoint(
        json.loads(json.dumps(checkpoint)),
        cell_key=CELL,
        source_smiles=SOURCE,
        delta=DELTA,
        budget_ceiling=BUDGET,
        controller_config=CONFIG,
    )

    assert restored.checkpoint == checkpoint
    assert restored.checkpoint["status"] == RUNNING
    assert restored.checkpoint["charged_count"] == 1
    assert restored.checkpoint["archive"] == {SOURCE: -8.0}
    assert restored.checkpoint["controller_config_identity"] == payload_identity(CONFIG)
    assert restored.state.budget == 48
    assert restored.value.weights is None


def test_uninterrupted_and_resumed_execution_are_byte_identical_after_two_rounds():
    initial = _initial_checkpoint()
    round_one_plan = _plan(initial, 1)
    after_one = _apply(initial, round_one_plan, -8.1)
    assert after_one["rounds_completed"] == 1
    assert len(after_one["value_features"]) == 8

    uninterrupted_plan = _plan(after_one, 2)
    uninterrupted = _apply(after_one, uninterrupted_plan, -8.5)

    resumed_input = json.loads(json.dumps(after_one, sort_keys=True))
    restored = restore_checkpoint(
        resumed_input,
        cell_key=CELL,
        source_smiles=SOURCE,
        delta=DELTA,
        budget_ceiling=BUDGET,
        controller_config=CONFIG,
    ).checkpoint
    resumed_plan = _plan(restored, 2)
    resumed = _apply(restored, resumed_plan, -8.5)

    assert resumed_plan == uninterrupted_plan
    assert resumed == uninterrupted
    assert payload_identity(resumed) == payload_identity(uninterrupted)
    assert resumed["rounds_completed"] == 2
    assert resumed["charged_count"] == 17
    assert (
        resumed["last_settled_query_lock_identity"]
        == resumed["rounds"][-1]["query_lock_payload_sha256"]
    )


def test_locked_round_recovery_uses_the_existing_plan_and_charges_missing_once():
    checkpoint = _initial_checkpoint()
    plan = _plan(checkpoint, 1)
    lock = plan["query_lock"]
    dispatch = {
        "schema_version": "t4_shared_controller_query_dispatch_v1",
        "query_lock_payload_sha256": payload_identity(lock),
        "query_ids": [row["query_id"] for row in lock["queries"]],
        "automatic_retries": 0,
    }
    reservations = {
        row["query_id"]: make_query_reservation(lock, row["query_id"])
        for row in lock["queries"]
    }
    receipts = {}
    for index, row in enumerate(lock["queries"][:-1]):
        query_id = row["query_id"]
        receipts[query_id] = {
            **reservations[query_id],
            "schema_version": QUERY_RECEIPT_SCHEMA,
            "status": "complete",
            "answer": {"score": -8.1 - 0.1 * index, "failure": None},
        }
    expected = {
        "checkpoint_payload_sha256": payload_identity(checkpoint),
        "round_plan_payload_sha256": payload_identity(plan),
        "query_lock_payload_sha256": payload_identity(lock),
        "query_dispatch_payload_sha256": payload_identity(dispatch),
        "query_ids": [row["query_id"] for row in lock["queries"]],
        "reservation_payload_sha256": {
            query_id: payload_identity(payload)
            for query_id, payload in reservations.items()
        },
        "receipt_payload_sha256": {
            query_id: (
                payload_identity(receipts[query_id]) if query_id in receipts else None
            )
            for query_id in reservations
        },
        "settlement_action": "recover_with_unresolved",
        "charged_after": 9,
        "round_after": 1,
    }
    recovered = recover_locked_round(
        checkpoint=checkpoint,
        round_plan=plan,
        query_lock=lock,
        dispatch_intent=dispatch,
        reservations=reservations,
        receipts=receipts,
        expected=expected,
        cell={"cell_key": CELL, "source_smiles": SOURCE, "delta": DELTA},
        controller_config=CONFIG,
        budget_ceiling=BUDGET,
        now=lock["query_deadline"] + 1,
    )
    assert recovered["settlement"]["action"] == "recover_with_unresolved"
    assert recovered["checkpoint"]["charged_count"] == 9
    assert recovered["checkpoint"]["rounds_completed"] == 1
    assert (
        recovered["checkpoint"]["rounds"][-1]["observations"][-1]["failure"]
        == "unresolved_locked_query_missing"
    )

    tampered = copy.deepcopy(receipts)
    first = next(iter(tampered))
    tampered[first]["answer"]["score"] = -99.0
    with pytest.raises(ValueError, match="query receipt .* identity drift"):
        recover_locked_round(
            checkpoint=checkpoint,
            round_plan=plan,
            query_lock=lock,
            dispatch_intent=dispatch,
            reservations=reservations,
            receipts=tampered,
            expected=expected,
            cell={"cell_key": CELL, "source_smiles": SOURCE, "delta": DELTA},
            controller_config=CONFIG,
            budget_ceiling=BUDGET,
            now=lock["query_deadline"] + 1,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"cell_key": "braf_0_d04"}, "cell mismatch"),
        ({"source_smiles": "N"}, "source mismatch"),
        ({"delta": 0.6}, "delta mismatch"),
        ({"budget_ceiling": 48}, "budget ceiling mismatch"),
        (
            {"controller_config": {**CONFIG, "exploration": 1}},
            "controller config drift",
        ),
    ],
)
def test_restore_fails_closed_on_binding_or_config_drift(updates, message):
    arguments = {
        "cell_key": CELL,
        "source_smiles": SOURCE,
        "delta": DELTA,
        "budget_ceiling": BUDGET,
        "controller_config": CONFIG,
    }
    arguments.update(updates)
    with pytest.raises(ValueError, match=message):
        restore_checkpoint(_initial_checkpoint(), **arguments)


def test_apply_rejects_unresolved_duplicate_foreign_and_skipped_work():
    checkpoint = _initial_checkpoint()
    plan = _plan(checkpoint, 1)
    waiting = settle_query_lock(plan["query_lock"], {}, now=0.0)
    with pytest.raises(RuntimeError, match="unresolved"):
        apply_settled_observations(
            checkpoint,
            plan,
            waiting,
            cell_key=CELL,
            source_smiles=SOURCE,
            delta=DELTA,
            budget_ceiling=BUDGET,
            controller_config=CONFIG,
        )

    settlement = _settled_lock(plan["query_lock"], [-8.1] * 8)
    foreign = copy.deepcopy(settlement)
    foreign["observations"][0]["query_id"] = "foreign"
    with pytest.raises(ValueError, match="foreign or reordered"):
        apply_settled_observations(
            checkpoint,
            plan,
            foreign,
            cell_key=CELL,
            source_smiles=SOURCE,
            delta=DELTA,
            budget_ceiling=BUDGET,
            controller_config=CONFIG,
        )

    duplicated = copy.deepcopy(settlement)
    duplicated["observations"][1]["query_id"] = duplicated["observations"][0][
        "query_id"
    ]
    with pytest.raises(ValueError, match="duplicate"):
        apply_settled_observations(
            checkpoint,
            plan,
            duplicated,
            cell_key=CELL,
            source_smiles=SOURCE,
            delta=DELTA,
            budget_ceiling=BUDGET,
            controller_config=CONFIG,
        )

    skipped = copy.deepcopy(plan)
    skipped["round"] = 2
    skipped["query_lock"]["round"] = 2
    skipped["query_lock_payload_sha256"] = payload_identity(skipped["query_lock"])
    with pytest.raises(ValueError, match="skipped or rewound"):
        apply_settled_observations(
            checkpoint,
            skipped,
            settlement,
            cell_key=CELL,
            source_smiles=SOURCE,
            delta=DELTA,
            budget_ceiling=BUDGET,
            controller_config=CONFIG,
        )

    advanced = apply_settled_observations(
        checkpoint,
        plan,
        settlement,
        cell_key=CELL,
        source_smiles=SOURCE,
        delta=DELTA,
        budget_ceiling=BUDGET,
        controller_config=CONFIG,
    )
    with pytest.raises((RuntimeError, ValueError), match="already applied|derived"):
        apply_settled_observations(
            advanced,
            plan,
            settlement,
            cell_key=CELL,
            source_smiles=SOURCE,
            delta=DELTA,
            budget_ceiling=BUDGET,
            controller_config=CONFIG,
        )


def test_budget_and_round_ledger_tampering_fail_closed():
    checkpoint = _apply(_initial_checkpoint(), _plan(_initial_checkpoint(), 1), -8.1)

    budget_drift = copy.deepcopy(checkpoint)
    budget_drift["charged_count"] += 1
    with pytest.raises(ValueError, match="budget accounting"):
        restore_checkpoint(
            budget_drift,
            cell_key=CELL,
            source_smiles=SOURCE,
            delta=DELTA,
            budget_ceiling=BUDGET,
            controller_config=CONFIG,
        )

    rewound = copy.deepcopy(checkpoint)
    rewound["rounds"][0]["round"] = 2
    with pytest.raises(ValueError, match="skipped or rewound"):
        restore_checkpoint(
            rewound,
            cell_key=CELL,
            source_smiles=SOURCE,
            delta=DELTA,
            budget_ceiling=BUDGET,
            controller_config=CONFIG,
        )


def test_empty_supplied_support_serializes_candidate_exhaustion():
    checkpoint = _initial_checkpoint()
    decision = select_parent_and_batch(
        checkpoint,
        {},
        cell_key=CELL,
        source_smiles=SOURCE,
        delta=DELTA,
        budget_ceiling=BUDGET,
        controller_config=CONFIG,
        query_deadline=10.0,
    )

    terminal = decision["checkpoint"]
    assert decision["status"] == CANDIDATE_EXHAUSTION
    assert terminal["status"] == CANDIDATE_EXHAUSTION
    assert terminal["terminal_reason"] == "no_unseen_candidates"
    assert terminal["charged_count"] == 1
    with pytest.raises(RuntimeError, match="terminal checkpoint"):
        select_parent_and_batch(
            terminal,
            {},
            cell_key=CELL,
            source_smiles=SOURCE,
            delta=DELTA,
            budget_ceiling=BUDGET,
            controller_config=CONFIG,
            query_deadline=11.0,
        )


def test_proposals_with_foreign_parents_or_scores_fail_closed():
    checkpoint = _initial_checkpoint()
    foreign = {"shallow": [_proposal("C", "shallow", "foreign", -8.0, 1)]}
    with pytest.raises(ValueError, match="foreign parent"):
        _ = select_parent_and_batch(
            checkpoint,
            foreign,
            cell_key=CELL,
            source_smiles=SOURCE,
            delta=DELTA,
            budget_ceiling=BUDGET,
            controller_config=CONFIG,
            query_deadline=10.0,
        )

    wrong_score = {"shallow": [_proposal("C", "shallow", SOURCE, -7.9, 1)]}
    with pytest.raises(ValueError, match="parent score"):
        _ = select_parent_and_batch(
            checkpoint,
            wrong_score,
            cell_key=CELL,
            source_smiles=SOURCE,
            delta=DELTA,
            budget_ceiling=BUDGET,
            controller_config=CONFIG,
            query_deadline=10.0,
        )
