"""Guards for the all-23-task PMO diagnostic rung contract and runtime.

Each test here corresponds to a defect that has already cost this project a budget or a
measurement: a contract budget nothing read, a near-miss oracle name resolving silently
to a different oracle, an asset-backed oracle returning a swallowed constant, and an AUC
denominator that did not match the run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_diagnostic_rung_v1 import (
    CONTRACT,
    READOUT_CHECKPOINTS,
    SCHEMA,
    SPIN_UP_CALLS,
    SUITE,
    RungBudgetError,
    asset_backed_plan,
    best_at_checkpoints,
    execute_task,
    load_contract,
    rounds_for_budget,
    runtime_protocol,
    validate_budget,
    verify_suite_against_registry,
)
from compose_v4.experiments.pmo_oracle_assets import (
    ASSET_BACKED_PMO_TASKS,
    POSITIVE_CONTROL_STATUS,
    POSITIVE_CONTROLS,
)
from compose_v4.experiments.pmo_population_v1 import INIT_COUNT, QUERIES_PER_ROUND

_ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict:
    return load_contract(_ROOT)


def _authorized() -> dict:
    contract = dict(_contract())
    contract["scored_launch_authorized"] = True
    return contract


def _synthetic(smiles: str) -> float:
    import hashlib

    return int.from_bytes(hashlib.sha256(smiles.encode()).digest()[:4], "big") / 0xFFFFFFFF


# ---- suite identity --------------------------------------------------------


def test_suite_is_the_verified_twenty_three():
    report = verify_suite_against_registry(_ROOT)
    assert report["n_tasks"] == 23
    assert len(SUITE) == 23
    assert len(set(SUITE)) == 23
    # The verification artifact must not have been produced by constructing an Oracle:
    # Oracle.__init__ fuzzy-matches at threshold 0.8, so construction is not evidence of
    # exact membership, and the registry really does carry near-miss entries.
    assert report["oracle_constructed"] is False
    hazard = report["fuzzy_fallback_hazard"]
    assert hazard["threshold"] == 0.8
    for near_miss in hazard["near_miss_entries_present"]:
        assert near_miss not in SUITE


def test_no_near_miss_name_is_in_the_suite():
    # sitagliptin_mpo_prev / zaleplon_mpo_prev sit one token from two real tasks.
    for task in SUITE:
        assert not task.endswith("_prev")


# ---- budget: the defect that charged 4x its declared ceiling ---------------


def test_budget_is_required_and_has_no_default():
    import inspect

    signature = inspect.signature(execute_task)
    parameter = signature.parameters["charged_calls_per_task"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_validate_budget_rejects_a_budget_that_cannot_populate_its_readouts():
    largest = max(READOUT_CHECKPOINTS)
    with pytest.raises(RungBudgetError):
        validate_budget(largest - 1)
    validate_budget(largest)


def test_validate_budget_rejects_a_budget_that_buys_no_adaptive_round():
    with pytest.raises(RungBudgetError):
        validate_budget(SPIN_UP_CALLS)
    with pytest.raises(RungBudgetError):
        validate_budget(INIT_COUNT - 1)


def test_spin_up_matches_the_campaign_loop_structure():
    # run_program_campaign charges every initialization candidate BEFORE the round loop
    # and round 0 is a bootstrap round, so adaptive selection starts only after both.
    assert SPIN_UP_CALLS == INIT_COUNT + QUERIES_PER_ROUND


def test_rounds_for_budget_makes_the_budget_binding_not_the_round_count():
    for budget in (128, 250, 512, 1000):
        rounds = rounds_for_budget(budget)
        capacity = INIT_COUNT + rounds * QUERIES_PER_ROUND
        assert capacity >= budget, (budget, rounds, capacity)
        # and one round fewer would NOT be enough, so the rung buys no idle rounds
        assert INIT_COUNT + (rounds - 1) * QUERIES_PER_ROUND < budget


def test_runtime_budget_must_match_the_authorizing_contract(tmp_path):
    contract = _authorized()
    declared = contract["budget"]["charged_calls_per_task"]
    with pytest.raises(ValueError, match="does not match the authorizing contract"):
        execute_task(
            contract, _ROOT, tmp_path / "a", "qed",
            evaluate=_synthetic, charged_calls_per_task=declared + 16,
        )


def test_unauthorized_contract_cannot_score(tmp_path):
    contract = _contract()
    assert contract["scored_launch_authorized"] is False
    with pytest.raises(ValueError, match="not authorized"):
        execute_task(
            contract, _ROOT, tmp_path / "b", "qed", evaluate=_synthetic,
            charged_calls_per_task=contract["budget"]["charged_calls_per_task"],
        )


def test_off_suite_task_is_refused(tmp_path):
    contract = _authorized()
    with pytest.raises(ValueError, match="outside the verified"):
        execute_task(
            contract, _ROOT, tmp_path / "c", "sitagliptin_mpo_prev", evaluate=_synthetic,
            charged_calls_per_task=contract["budget"]["charged_calls_per_task"],
        )


# ---- the sealed contract ---------------------------------------------------


def test_contract_is_sealed_and_fail_closed():
    envelope = json.loads((_ROOT / CONTRACT).read_text())
    assert set(envelope) == {"payload", "payload_sha256"}
    payload = envelope["payload"]
    assert envelope["payload_sha256"] == identity(payload)
    assert payload["schema_version"] == SCHEMA
    assert payload["scored_launch_authorized"] is False
    assert payload["modal_launch_authorized"] is False
    assert payload["oracle_calls_authorized"] == 0
    assert payload["status"].startswith("PENDING")


def test_contract_ceiling_is_budget_times_task_count():
    payload = _contract()
    budget = payload["budget"]["charged_calls_per_task"]
    assert payload["budget"]["charged_calls_ceiling"] == budget * 23
    assert len(payload["tasks"]) == 23


def test_contract_budget_fields_agree_with_the_runtime():
    payload = _contract()
    declared = validate_budget(payload["budget"]["charged_calls_per_task"])
    for key in ("rounds", "adaptive_rounds", "spin_up_calls", "adaptive_calls"):
        assert payload["budget"][key] == declared[key], key


def test_contract_declares_auc_as_within_rung_only():
    payload = _contract()
    auc = payload["readouts"]["auc"]
    assert auc["budget"] == payload["budget"]["charged_calls_per_task"]
    assert "within-rung" in auc["comparability"]


# ---- asset-backed oracles: the silent-zero class ---------------------------


def test_every_asset_backed_suite_task_has_a_pinned_positive_control():
    for task in ASSET_BACKED_PMO_TASKS:
        if task not in SUITE:
            continue
        assert task in POSITIVE_CONTROLS, task
        references = POSITIVE_CONTROLS[task]
        # A constant-returning oracle reproduces the 0.0 rows and nothing else, so the
        # panel must carry graded nonzero values, not merely "an active is positive".
        nonzero = [row for row in references if row.expected > 0.0]
        graded = [row for row in references if 0.0 < row.expected < 0.9]
        assert len(nonzero) >= 3, task
        assert graded, f"{task} panel has no intermediate value"
        assert POSITIVE_CONTROL_STATUS[task] == "MEASURED", task


def test_asset_plan_reports_missing_assets_rather_than_assuming_them():
    payload = _contract()
    plan = asset_backed_plan(_ROOT, payload["oracle"]["asset_root"])
    for task, row in plan["tasks"].items():
        assert row["positive_control_required_before_first_charged_call"] is True
        if row["asset_present_in_capsule"]:
            assert row["asset_sha256"]
        else:
            assert row["asset_sha256"] is None
            assert task in plan["blocked_pending_asset_pin"]
    assert set(plan["runnable_tasks"]) | set(plan["blocked_pending_asset_pin"]) == set(SUITE)


def test_runtime_protocol_binds_the_asset_digest_for_asset_backed_tasks():
    payload = _contract()
    for task in SUITE:
        protocol = runtime_protocol(payload, task)
        assert protocol["prescreen"] is False
        assert protocol["direction"] == "maximize"
        assert protocol["calls_include_initialization"] is True
        pinned = payload["oracle"]["assets"].get(task, {}).get("asset_sha256")
        if pinned:
            assert protocol["asset_sha256"] == {f"{task}_current.pkl": pinned}
        else:
            assert protocol["asset_sha256"] == {}


def test_asset_digest_changes_the_oracle_protocol_identity():
    # Two different pickles must not produce reconcilable receipts.
    payload = json.loads(json.dumps(_contract()))
    task = next(
        (t for t in SUITE if payload["oracle"]["assets"].get(t, {}).get("asset_sha256")),
        None,
    )
    assert task, "expected at least one pinned asset-backed task"
    before = identity(runtime_protocol(payload, task))
    payload["oracle"]["assets"][task]["asset_sha256"] = "0" * 64
    assert identity(runtime_protocol(payload, task)) != before


# ---- no prescreen, no reference leakage ------------------------------------


def test_initialization_carries_no_task_information():
    payload = _contract()
    initialization = json.loads(
        (_ROOT / payload["initialization"]["path"]).read_text()
    )
    for row in initialization["candidates"]:
        assert "score" not in row
        assert "task" not in row
    assert payload["reference_molecule_policy"]["prescreen"] is False
    assert payload["reference_molecule_policy"]["reference_molecules_in_initialization"] is False


# ---- readouts --------------------------------------------------------------


def test_best_at_checkpoints_uses_charged_call_index():
    rows = [
        {"status": "complete", "score": 0.1},
        *[{"status": "complete", "score": 0.0} for _ in range(40)],
        {"status": "complete", "score": 0.9},
    ]
    out = best_at_checkpoints(rows)
    assert out["best_at_32"] == 0.1
    assert out["best_at_64"] == 0.9
    # A checkpoint beyond the available rows reports what is known, never a crash.
    assert out["best_at_128"] == 0.9


def test_best_at_checkpoints_is_empty_without_completed_rows():
    out = best_at_checkpoints([])
    for checkpoint in READOUT_CHECKPOINTS:
        assert out[f"best_at_{checkpoint}"] is None
