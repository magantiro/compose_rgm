from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from rdkit import Chem

from compose_v4.control.dynamic_program_synthesis import DynamicProgramOptimizer
from compose_v4.control.dynamic_program_synthesis_v21 import DynamicV21ProgramOptimizer
from compose_v4.control.edit_program import extract_program
from compose_v4.control.program_task import ProgramTask, initialization_lock
from compose_v4.experiments.continuation_profile import publish_json
from compose_v4.experiments.pmo_dynamic_v21 import (
    ARMS,
    OFFLINE_COMPARATORS,
    TASK_SEEDS,
    TASKS,
    configuration,
    execute_task,
    load_contract,
    serialized_configuration,
)
from compose_v4.rewrite.trace_shard import encode_state
from tests.test_edit_program import ring_and_carbonyl
from tools.pmo_dynamic_v21 import build_structural_preflight, status

ROOT = Path(__file__).resolve().parents[1]


def test_contract_fixes_four_tasks_empty_routes_and_offline_comparators():
    contract = load_contract(ROOT)

    assert tuple(contract["dynamic_v21_development"]["tasks"]) == TASKS
    assert tuple(contract["dynamic_v21_development"]["arms"]) == ARMS
    assert set(TASK_SEEDS.values()) == {20260914}
    assert contract["dynamic_v21_development"]["charged_queries_per_task"] == 1000
    assert contract["dynamic_v21_development"]["total_charged_query_ceiling"] == 8000
    assert contract["dynamic_v21_development"]["max_concurrent_workers"] == 4
    assert (
        contract["dynamic_v21_development"]["submission_order"]
        == "task_major_paired_arms"
    )
    assert contract["library"]["programs"] == 0
    assert contract["dynamic_v21_development"]["initial_route_archive_by_arm"] == {
        arm: [] for arm in ARMS
    }
    assert contract["information_regime"]["initial_task_specific_complete_routes"] == 0
    assert contract["postrun_only_comparator"]["path"] == OFFLINE_COMPARATORS
    assert OFFLINE_COMPARATORS not in contract["inputs"]
    assert not any("result.json" in path for path in contract["inputs"])
    assert contract["controller"] == json.loads(json.dumps(asdict(configuration())))
    assert contract["controller"]["score_direction"] == "maximize"


def test_direct_worker_launch_configuration_matches_loaded_json_contract():
    contract = load_contract(ROOT)
    raw = asdict(configuration())
    direct_worker_launch = {"configuration": serialized_configuration()}

    assert raw != contract["controller"]
    assert isinstance(raw["channel_probabilities"], tuple)
    assert isinstance(contract["controller"]["channel_probabilities"], list)
    assert direct_worker_launch["configuration"] == contract["controller"]


def test_pmo_validity_has_no_t4_similarity_qed_or_sa_gate():
    task = ProgramTask("gsk3b", "test-protocol", "pmo")
    valid = task.endpoint_evaluator()({"smiles": "CCO"})
    invalid = task.endpoint_evaluator()({"smiles": "not-a-molecule"})

    assert valid == {"smiles": "CCO", "oracle_eligible": True}
    assert invalid == {"smiles": "not-a-molecule", "oracle_eligible": False}
    assert not {"similarity", "qed", "sa"} & set(valid)


@pytest.mark.parametrize("arm", ARMS)
def test_synthetic_campaign_uses_matched_core_and_charges_all_initialization(
    tmp_path, monkeypatch, arm
):
    source, stages = ring_and_carbonyl()
    program, _ = extract_program(source, stages)
    initialized = initialization_lock(
        [{"state": encode_state(source), "source_id": "fixture"}],
        count=1,
        seed=7,
        source_sha256="a" * 64,
    )
    publish_json(tmp_path / "init.json", initialized)
    publish_json(tmp_path / "empty.json", [])
    contract = {
        "dynamic_v21_development": {
            "tasks": list(TASKS),
        },
        "initialization": {
            "path": "init.json",
            "sha256": __import__(
                "compose_v4.experiments.continuation_profile", fromlist=["sha256_file"]
            ).sha256_file(tmp_path / "init.json"),
            "source_sha256": "a" * 64,
        },
        "oracle": {
            "environment": {},
            "source_files": {},
            "gsk3b_asset": {"sha256": "b" * 64},
        },
    }
    small = replace(
        configuration(71),
        attempts_per_batch=24,
        candidates_per_batch=2,
        wall_seconds=20,
    )
    # Keep this fixture bounded without changing the production configuration.
    import compose_v4.experiments.pmo_dynamic_v21 as experiment

    monkeypatch.setattr(experiment, "configuration", lambda seed=71: small)
    monkeypatch.setattr(experiment, "QUERY_BUDGET", 5)
    monkeypatch.setattr(experiment, "INITIALIZATION_COUNT", 1)
    monkeypatch.setattr(experiment, "MAX_ROUNDS", 2)
    monkeypatch.setattr(experiment, "QUERIES_PER_ROUND", 2)
    result = execute_task(
        contract,
        tmp_path,
        tmp_path / "run",
        arm,
        "celecoxib_rediscovery",
        evaluate=lambda smiles: Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() / 40,
    )

    assert result["charged_oracle_calls"] <= 5
    assert result["initialization_calls"] == 1
    assert (
        result["score_curve"][-1]["charged_queries"] == result["charged_oracle_calls"]
    )
    assert result["arm"] == arm
    if arm == "dynamic_v21":
        assert result["campaign"]["snapshot"]["dynamic_v21"]
    else:
        assert "dynamic_v21" not in result["campaign"]["snapshot"]
    assert result["campaign"]["benchmark_claim"] is False
    assert result["auc_top10_official_10k"] is not None
    assert program.program_id  # Fixture itself is a real exact program.


def test_structural_preflight_is_zero_oracle_and_task_specific():
    row = build_structural_preflight(ROOT)

    assert row["passed"] is True
    assert row["new_oracle_calls"] == 0
    assert [entry["task"] for entry in row["tasks"]] == list(TASKS)
    assert row["initial_route_archive_rows"] == 0
    assert row["initial_route_archive_rows_by_arm"] == {arm: 0 for arm in ARMS}
    assert row["task_informed_curriculum_rows"] == 0
    assert row["runtime_comparison_inputs"] == 0
    assert row["t4_similarity_qed_sa_gates"] == 0
    assert set(row["representative_legal_execution"]) == {
        "shallow_program_channel",
        "structured_program_channel",
    }


def test_task_outside_lock_is_rejected_before_query(tmp_path):
    with pytest.raises(ValueError, match="outside the four-task"):
        execute_task(
            {"dynamic_v21_development": {"tasks": list(TASKS)}},
            tmp_path,
            tmp_path / "run",
            "dynamic_v0",
            "qed",
            evaluate=lambda _: (_ for _ in ()).throw(AssertionError("oracle called")),
        )


def test_task_outside_arm_lock_is_rejected_before_query(tmp_path):
    with pytest.raises(ValueError, match="arm is outside"):
        execute_task(
            {"dynamic_v21_development": {"tasks": list(TASKS)}},
            tmp_path,
            tmp_path / "run",
            "task_tuned_arm",
            "gsk3b",
            evaluate=lambda _: (_ for _ in ()).throw(AssertionError("oracle called")),
        )


def test_campaign_namespace_restores_generic_optimizer():
    from compose_v4.control import program_campaign
    from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer
    from compose_v4.experiments.pmo_dynamic_v21 import _campaign_namespace

    assert program_campaign.ProgramOptimizer is ProgramOptimizer
    with _campaign_namespace("dynamic_v0"):
        assert program_campaign.ProgramOptimizer is DynamicProgramOptimizer
    with _campaign_namespace("dynamic_v21"):
        assert program_campaign.ProgramOptimizer is DynamicV21ProgramOptimizer
    assert program_campaign.ProgramOptimizer is ProgramOptimizer


def test_task_boundary_records_unexpected_failure_without_retry(tmp_path, monkeypatch):
    import compose_v4.experiments.pmo_dynamic_v21 as experiment

    contract_path = tmp_path / experiment.CONTRACT
    preflight_path = tmp_path / experiment.PREFLIGHT
    contract_path.parent.mkdir(parents=True)
    preflight_path.parent.mkdir(parents=True)
    contract_path.write_text("contract\n")
    preflight_path.write_text("preflight\n")
    launch_body = {
        "contract_sha256": experiment.sha256_file(contract_path),
        "preflight_sha256": experiment.sha256_file(preflight_path),
        "tasks": list(TASKS),
        "arms": list(ARMS),
        "charged_query_ceiling": 8000,
        "automatic_retries": 0,
        "configuration": asdict(configuration()),
        "configuration_id": experiment.identity(asdict(configuration())),
    }
    launch = {**launch_body, "run_id": experiment.identity(launch_body)}

    monkeypatch.setattr(
        experiment, "load_contract", lambda _: {"controller": asdict(configuration())}
    )
    monkeypatch.setattr(experiment, "verify_runtime_environment", lambda *_: {})

    @contextmanager
    def harmless_oracle(*_):
        yield lambda _smiles: 0.0

    def fail_unexpectedly(*_, **__):
        raise ZeroDivisionError("unexpected task failure")

    monkeypatch.setattr(experiment, "native_oracle", harmless_oracle)
    monkeypatch.setattr(experiment, "execute_task", fail_unexpectedly)
    result = experiment.run_scored_task(
        tmp_path, launch, "dynamic_v21", "celecoxib_rediscovery"
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "ZeroDivisionError"
    assert result["automatic_retry"] is False
    failure_path = (
        tmp_path
        / "diagnostics/pmo_dynamic_v21/runs"
        / launch["run_id"]
        / "dynamic_v21"
        / "celecoxib_rediscovery/failure.json"
    )
    assert experiment.unseal(failure_path)["error"] == "unexpected task failure"


def test_relaunch_v2_preserves_failed_launch_and_passes_canonical_configuration(
    tmp_path, monkeypatch
):
    import tools.pmo_dynamic_v21 as runner

    v1_body = {
        "schema_version": "pmo_dynamic_v21_matched_launch_v1",
        "configuration": serialized_configuration(),
        "configuration_id": runner.identity(serialized_configuration()),
    }
    failed_launch = {**v1_body, "run_id": runner.identity(v1_body)}
    runner.seal(tmp_path / runner.LAUNCH, failed_launch)
    original_sha256 = runner.sha256_file(tmp_path / runner.LAUNCH)
    contract_path, preflight_path = (
        tmp_path / runner.CONTRACT,
        tmp_path / runner.PREFLIGHT,
    )
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text("contract\n")
    preflight_path.write_text("preflight\n")
    runner.seal(
        tmp_path / runner.PREQUERY_FAILURE,
        {
            "status": "failed_before_unit_start",
            "failed_launch_sha256": original_sha256,
            "oracle_calls": 0,
        },
    )
    contract = {"controller": serialized_configuration()}
    preflight = {"schema_version": "pmo_dynamic_v21_matched_preflight_v1"}
    monkeypatch.setattr(
        runner, "validate_launch_ready", lambda *_args, **_kwargs: (contract, preflight)
    )
    monkeypatch.setattr(
        runner, "_execute_launch", lambda _root, launch_row, **_: launch_row
    )
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *_args, **_kwargs: "d" * 40
    )

    launch_v2 = runner.relaunch_v2(tmp_path, workers=4)

    assert launch_v2["schema_version"] == "pmo_dynamic_v21_matched_launch_v2"
    assert launch_v2["configuration"] == contract["controller"]
    assert runner.sha256_file(tmp_path / runner.LAUNCH) == original_sha256
    assert runner.unseal(tmp_path / runner.LAUNCH_V2) == launch_v2
    assert launch_v2["run_id"] != failed_launch["run_id"]


def test_status_is_read_only_and_reports_live_arm_accounting(tmp_path):
    import compose_v4.experiments.pmo_dynamic_v21 as experiment

    launch = {
        "schema_version": "pmo_dynamic_v21_matched_launch_v1",
        "run_id": "fixture-run",
        "contract_sha256": "a" * 64,
        "preflight_sha256": "b" * 64,
        "code_revision": "c" * 40,
        "tasks": list(TASKS),
        "arms": list(ARMS),
        "task_seeds": TASK_SEEDS,
        "configuration": asdict(configuration()),
        "configuration_id": experiment.identity(asdict(configuration())),
    }
    experiment.seal(tmp_path / experiment.LAUNCH, launch)
    folder = (
        tmp_path
        / "diagnostics/pmo_dynamic_v21/runs/fixture-run"
        / "dynamic_v21/celecoxib_rediscovery"
    )
    publish_json(folder / "started.json", {"arm": "dynamic_v21"})
    query = folder / "oracle/query_000000"
    publish_json(query / "started.json", {"index": 0})
    publish_json(query / "result.json", {"status": "complete", "score": 0.5})
    publish_json(
        folder / "campaign/round_0000/pending.json",
        {
            "batch": {
                "attempts": [
                    {
                        "planner_channel": "structured_program_channel",
                        "status": "duplicate",
                    },
                    {
                        "planner_channel": "shallow_program_channel",
                        "status": "execution_rejected",
                    },
                ],
                "candidates": [
                    {"provenance": {"planner_channel": "structured_program_channel"}}
                ],
                "allocation": {
                    "selected_by_channel": {"structured_program_channel": 1}
                },
            }
        },
    )
    files = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    before = {str(path): experiment.sha256_file(path) for path in files}

    first = status(tmp_path)
    second = status(tmp_path)
    after = {str(path): experiment.sha256_file(path) for path in files}

    assert first == second
    assert before == after
    unit = first["arms"]["dynamic_v21"]["celecoxib_rediscovery"]
    assert unit["actual_charged_queries"] == 1
    assert unit["current"]["best_score"] == 0.5
    assert unit["current"]["top10_score"] == 0.5
    assert unit["current"]["auc_top10_development_1000"] == 0.49975
    assert unit["proposal"]["attempt_status_counts"] == {
        "duplicate": 1,
        "execution_rejected": 1,
    }
    assert unit["proposal"]["latest_v21_allocation"]["selected_by_channel"] == {
        "structured_program_channel": 1
    }
