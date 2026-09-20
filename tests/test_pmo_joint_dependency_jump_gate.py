import json
import random
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.control.pmo_action_roles import action_role_supervision
from compose_v4.control.pmo_joint_dependency_jump import (
    _generic_role_sequence,
    _role_identity,
    enumerate_role_successors,
    fit_joint_checkpoint,
)
from compose_v4.experiments.pmo_complete_route_dynamic_gate import (
    _member_tasks,
    load_envelope,
)
from compose_v4.experiments.pmo_joint_dependency_jump_gate import (
    reduce_arm_receipts,
    validate_contract,
)
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CORPUS = (
    ROOT / "diagnostics/pmo_dependency_region_program_v2/attempt_1/"
    "training_dependency_region_corpus.json.gz"
)
CONTRACT = ROOT / "configs/pmo_joint_dependency_jump_gate_v2.json"


def _celecoxib_route():
    corpus = load_envelope(CORPUS)
    route = next(
        row
        for row in corpus["routes"]
        if "celecoxib_rediscovery" in _member_tasks(row) and len(row["actions"]) <= 32
    )
    return corpus, route


def test_contract_is_zero_oracle_and_self_hashed():
    contract = json.loads(CONTRACT.read_text())
    validate_contract(contract)
    assert contract["oracle_calls_authorized"] == 0
    assert contract["modal_launch_authorized"] is False


def test_shared_checkpoint_is_address_free_and_fold_checkpoint_is_clean():
    corpus, _ = _celecoxib_route()
    shared = fit_joint_checkpoint(corpus, held_fold=None)["payload"]
    clean = fit_joint_checkpoint(corpus, held_fold=1)["payload"]
    serialized = json.dumps(shared, sort_keys=True)
    assert shared["fit_scope"] == "shared_all_routes"
    assert clean["fit_scope"] == "held_fold_clean"
    assert clean["held_fold"] == 1
    assert '"actions"' not in serialized
    assert '"source_state"' not in serialized
    assert '"task_family"' not in serialized


def test_joint_role_fiber_contains_previously_missing_cycle_close():
    _, route = _celecoxib_route()
    roles = _generic_role_sequence(route)
    created = {}
    next_ordinal = 0
    for step, (state, teacher, desired) in enumerate(
        zip(route["states"][:-1], route["actions"], roles, strict=True)
    ):
        graph = decode_state(state)
        if step == 28:
            candidates = enumerate_role_successors(graph, desired, created=created, step=step)
            matching = []
            for candidate in candidates:
                child_created = dict(created)
                observed, _ = action_role_supervision(
                    graph,
                    candidate.action_record,
                    child_created,
                    step,
                    next_ordinal,
                )
                if _role_identity(observed) == _role_identity(desired):
                    matching.append(candidate)
            assert matching
            assert any(candidate.action_record == teacher for candidate in matching)
            return
        _, next_ordinal = action_role_supervision(graph, teacher, created, step, next_ordinal)
    raise AssertionError("expected Celecoxib cycle-close support probe was not reached")


def _reduction_fixture(endpoint_count=40):
    source = {"fixture": "source"}
    plans = [
        {"plan_id": "plan_a", "mass": 0.7},
        {"plan_id": "plan_b", "mass": 0.3},
    ]
    checkpoint = {
        "fit_scope": "shared_all_routes",
        "held_fold": None,
        "plan_latents": plans,
    }
    rows = []
    for plan_index, plan in enumerate(plans):
        candidates = []
        for index in range(plan_index, endpoint_count, len(plans)):
            candidates.append(
                {
                    "plan_id": plan["plan_id"],
                    "endpoint_key": f"endpoint_{index:03d}",
                    "primitive_count": 12 if index % 2 else 6,
                    "exact_replay": True,
                }
            )
        rows.append(
            {
                "plan_id": plan["plan_id"],
                "plan_mass": plan["mass"],
                "bound_candidates": candidates,
            }
        )
    return source, checkpoint, rows


def test_reduction_is_shuffle_stable_and_locks_exact_budget():
    source, checkpoint, rows = _reduction_fixture()
    first = reduce_arm_receipts(
        source, checkpoint, rows, arm="shared_all_routes", candidate_budget=32
    )
    shuffled = list(rows)
    random.Random(20260919).shuffle(shuffled)
    second = reduce_arm_receipts(
        source, checkpoint, shuffled, arm="shared_all_routes", candidate_budget=32
    )
    assert len(first["candidates"]) == 32
    assert identity(first) == identity(second)


def test_reduction_reports_shortfall_and_rejects_incomplete_census():
    source, checkpoint, rows = _reduction_fixture(endpoint_count=2)
    reduced = reduce_arm_receipts(
        source, checkpoint, rows, arm="shared_all_routes", candidate_budget=32
    )
    assert len(reduced["candidates"]) == 2
    with pytest.raises(ValueError, match="receipt set incomplete"):
        reduce_arm_receipts(
            source,
            checkpoint,
            rows[:1],
            arm="shared_all_routes",
            candidate_budget=32,
        )
