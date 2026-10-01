"""Feedback-loop fixtures use synthetic scores, never a docking objective."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.reference_guidance import GuidanceConfig
from compose_v4.control.reference_programs import FrozenProgramReference
from compose_v4.experiments import t4_local_campaign as campaign
from compose_v4.model.reference_checkpoint import load_frozen_reference
from tests import test_t4_route_proposals as route_fixtures

BOUNDS, LEAD = route_fixtures.BOUNDS, route_fixtures.LEAD
expert = route_fixtures.expert
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "experiments/reference/model.json").read_text())


def settings(**changes):
    return replace(
        campaign.T4RunConfig(
            LEAD,
            0.4,
            13,
            budget=3,
            batch=2,
            expert_floor_rounds=0,
            route_scale_floor_rounds=0,
            lanes=("route_complete_region",),
            route=BOUNDS,
        ),
        **changes,
    )


def arguments(tmp_path, expert, **changes):
    calls = []

    def synthetic_score(smiles):
        calls.append(smiles)
        # Deliberately not docking or an interpretable molecular property.
        return -float(len(smiles)) / 10

    return {
        "output": tmp_path / "run",
        "config": settings(**changes),
        "evaluate": synthetic_score,
        "route_expert": expert,
        "evaluator_identity": {
            "kind": "synthetic_unit_test",
            "definition": "negative_string_length/10",
        },
        "implementation_identity": {"fixture": "t4-local-loop-v1"},
    }, calls


def read(path):
    return json.loads(path.read_text())["payload"]


def test_real_proposals_synthetic_scores_and_exact_completed_resume(tmp_path, expert):
    kwargs, calls = arguments(tmp_path, expert)
    result = campaign.run_t4(**kwargs)
    assert result["status"] == "budget_exhausted"
    assert result["charged_calls"] == len(calls) == 3
    assert len(set(calls)) == len(calls)
    assert result["rounds"] == 1
    lock = read(kwargs["output"] / "round_000001/lock.json")
    assert lock["reference_selection"][0]["outcome"] == "off"
    assert len(lock["selected"]) == 2
    assert all(row["realized_actions"] for row in lock["candidates"])
    assert campaign.run_t4(**kwargs, resume=True) == result
    assert len(calls) == 3
    with pytest.raises(FileExistsError, match="explicit resume"):
        campaign.run_t4(**kwargs)
    with pytest.raises(ValueError, match="changed on resume"):
        campaign.run_t4(**{**kwargs, "config": settings(budget=4)}, resume=True)
    assert len(calls) == 3


def test_interrupt_after_receipt_resumes_without_reproposal_or_duplicate_query(
    tmp_path, expert, monkeypatch
):
    kwargs, calls = arguments(tmp_path, expert)
    original = campaign.ProgramQueryLedger.query
    interrupted = False

    def interrupt(self, *args, **options):
        nonlocal interrupted
        result = original(self, *args, **options)
        if options["role"] == "candidate" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("fixture interruption after durable result")
        return result

    monkeypatch.setattr(campaign.ProgramQueryLedger, "query", interrupt)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_t4(**kwargs)
    assert len(calls) == 2
    assert not (kwargs["output"] / "result.json").exists()
    monkeypatch.setattr(campaign.ProgramQueryLedger, "query", original)

    def no_reproposal(*args, **options):
        raise AssertionError("locked round must not be proposed again")

    monkeypatch.setattr(campaign, "generate_panel", no_reproposal)
    resumed = campaign.run_t4(**kwargs, resume=True)
    assert resumed["charged_calls"] == len(calls) == 3
    assert len(set(calls)) == 3


def test_replay_uses_score_blind_parents_and_never_fits_value_model(tmp_path, expert, monkeypatch):
    kwargs, calls = arguments(tmp_path, expert, budget=9, max_rounds=5)
    monkeypatch.setattr(
        campaign.ProgramValue,
        "fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no fitted T4 value model")),
    )
    result = campaign.run_t4(**kwargs)
    lock = read(kwargs["output"] / "round_000001/lock.json")
    assert lock["parent_probabilities"] == {LEAD: 1.0}
    count = len(calls)
    assert campaign.run_t4(**kwargs, resume=True) == result
    assert len(calls) == count == result["charged_calls"]


def test_parent_law_ignores_scores_and_tempers_exhaustion():
    archive = {"CC": -10.0, "CCC": -2.0}
    parents, initial = campaign.score_blind_parent_probabilities(archive, {}, 0.2)
    assert parents == ["CC", "CCC"]
    assert initial.tolist() == pytest.approx([0.5, 0.5])
    _, changed_scores = campaign.score_blind_parent_probabilities(
        {"CC": 100.0, "CCC": -100.0}, {}, 0.2
    )
    assert changed_scores.tolist() == pytest.approx(initial.tolist())
    _, exhausted = campaign.score_blind_parent_probabilities(archive, {"CC": 3}, 0.2)
    assert exhausted[0] < exhausted[1]
    assert exhausted[0] > 0


def test_interrupt_after_initialization_receipt_has_exact_uninterrupted_result(
    tmp_path, expert, monkeypatch
):
    kwargs, calls = arguments(tmp_path, expert)
    original = campaign.ProgramQueryLedger.query

    def interrupt(self, *args, **options):
        original(self, *args, **options)
        raise KeyboardInterrupt("fixture interruption after root receipt")

    monkeypatch.setattr(campaign.ProgramQueryLedger, "query", interrupt)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_t4(**kwargs)
    assert len(calls) == 1
    monkeypatch.setattr(campaign.ProgramQueryLedger, "query", original)
    resumed = campaign.run_t4(**kwargs, resume=True)
    uninterrupted, fresh_calls = arguments(tmp_path / "other", expert)
    assert campaign.run_t4(**uninterrupted) == resumed
    assert len(calls) == len(fresh_calls) == 3


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_uncertain_or_failed_query_never_retries(tmp_path, expert, failure):
    kwargs, _ = arguments(tmp_path, expert)
    calls = []

    def fail(smiles):
        calls.append(smiles)
        raise failure("synthetic failure")

    kwargs["evaluate"] = fail
    with pytest.raises(failure, match="synthetic failure"):
        campaign.run_t4(**kwargs)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="no automatic retry|explicit recovery"):
        campaign.run_t4(**kwargs, resume=True)
    assert len(calls) == 1


def test_empty_panel_is_not_a_completed_budget(tmp_path, expert, monkeypatch):
    kwargs, calls = arguments(tmp_path, expert)
    monkeypatch.setattr(campaign, "generate_panel", lambda *args: ([], {"fixture": "empty"}))
    result = campaign.run_t4(**kwargs)
    assert result["status"] == "candidate_exhausted"
    assert result["charged_calls"] == len(calls) == 1
    assert result["budget"] == 3
    assert campaign.run_t4(**kwargs, resume=True) == result
    assert len(calls) == 1


def test_missing_programs_are_retained_as_construction_abstentions(tmp_path, expert, monkeypatch):
    kwargs, calls = arguments(tmp_path, expert)
    original = campaign.generate_panel

    def incomplete(*args):
        rows, telemetry = original(*args)
        assert rows
        for row in rows:
            row.pop("realized_actions")
        return rows, telemetry

    monkeypatch.setattr(campaign, "generate_panel", incomplete)
    result = campaign.run_t4(**kwargs)
    assert result["status"] == "candidate_exhausted"
    assert len(calls) == 1
    lock = read(kwargs["output"] / "round_000001/lock.json")
    assert lock["candidates"] == lock["selected"] == []
    assert lock["construction_abstentions"]
    assert all(
        row["reason"] == "missing_primitive_program" for row in lock["construction_abstentions"]
    )


def test_construction_gate_replays_and_checks_measured_parent(expert):
    state = campaign.SearchState(archive={LEAD: -7.0})
    fiber = campaign.Fiber(LEAD, 0.4)
    rows, _ = campaign.generate_panel(
        [LEAD], state, fiber, np.random.default_rng(3), settings(), expert
    )
    assert rows
    accepted, rejected = campaign.executable_panel(rows, state)
    assert accepted == rows and not rejected
    empty = {**rows[0], "realized_actions": []}
    assert campaign.executable_panel([empty], state)[1][0]["reason"] == "zero_primitive_program"
    wrong = deepcopy(rows[0])
    wrong["parent_score"] -= 1
    with pytest.raises(ValueError, match="measured archive"):
        campaign.executable_panel([wrong], state)
    wrong = deepcopy(rows[0])
    wrong["realized_endpoint_key"] = "CC"
    with pytest.raises(ValueError, match="realized endpoint differs"):
        campaign.executable_panel([wrong], state)


def test_resume_binds_evaluator_assets_and_implementation(tmp_path, expert):
    kwargs, calls = arguments(tmp_path, expert, budget=1)
    campaign.run_t4(**kwargs)
    for field in ("evaluator_identity", "asset_identity", "implementation_identity"):
        with pytest.raises(ValueError, match="changed on resume"):
            campaign.run_t4(**{**kwargs, field: {"changed": True}}, resume=True)
    assert len(calls) == 1


def test_existing_unowned_files_and_missing_run_are_refused(tmp_path, expert):
    kwargs, calls = arguments(tmp_path, expert, budget=1)
    with pytest.raises(FileNotFoundError, match="no run manifest"):
        campaign.run_t4(**kwargs, resume=True)
    output = kwargs["output"]
    (output / "user-data.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="not empty"):
        campaign.run_t4(**kwargs)
    assert (output / "user-data.txt").read_text() == "keep"
    assert not calls


@pytest.mark.parametrize("bad_score", [None, True, "-8.0", float("nan"), float("inf")])
def test_invalid_evaluation_remains_charged_and_blocks_resume(tmp_path, expert, bad_score):
    kwargs, _ = arguments(tmp_path, expert, budget=1)
    kwargs["evaluate"] = lambda smiles: bad_score
    with pytest.raises(ValueError, match="finite real score"):
        campaign.run_t4(**kwargs)
    receipt = json.loads((kwargs["output"] / "oracle/query_000000/result.json").read_text())
    assert receipt["status"] == "failed"
    with pytest.raises(RuntimeError, match="failed query remains charged"):
        campaign.run_t4(**kwargs, resume=True)


def test_concurrent_writer_and_corrupt_record_are_refused(tmp_path, expert):
    kwargs, calls = arguments(tmp_path, expert, budget=1)
    with (
        campaign._exclusive_writer(kwargs["output"]),
        pytest.raises(RuntimeError, match="another writer"),
    ):
        campaign.run_t4(**kwargs)
    campaign.run_t4(**kwargs)
    path = kwargs["output"] / "manifest.json"
    body = json.loads(path.read_text())
    body["payload"]["configuration"]["budget"] = 6
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="invalid immutable"):
        campaign.run_t4(**kwargs, resume=True)
    assert len(calls) == 1


@pytest.mark.external_artifact
def test_loop_uses_frozen_reference_before_lock_and_refuses_changed_strength(tmp_path, expert):
    path = Path(
        os.environ.get(
            "COMPOSE_REFERENCE_CHECKPOINT", ROOT / "local_assets/fragments/r_theta_nll.pt"
        )
    )
    if not path.is_file():
        pytest.skip("fetch the NLL reference checkpoint through Git LFS")
    reference = FrozenProgramReference(
        load_frozen_reference(
            path,
            expected_sha256=REFERENCE["checkpoint"]["sha256"],
            expected_catalog_fingerprint=REFERENCE["catalog"]["fingerprint"],
            catalog_path=ROOT / "local_assets/fragments/catalog.json",
            expected_catalog_sha256=REFERENCE["catalog"]["sha256"],
        )
    )
    kwargs, calls = arguments(tmp_path, expert, guidance=GuidanceConfig("active", 0.25))
    kwargs["reference"] = reference
    weights_before = {k: v.clone() for k, v in reference.reference.model.state_dict().items()}
    result = campaign.run_t4(**kwargs)
    lock = read(kwargs["output"] / "round_000001/lock.json")
    assert lock["reference_selection"][0]["probabilities_changed"]
    assert result["charged_calls"] == len(calls) == 3
    assert all(
        (value == weights_before[key]).all()
        for key, value in reference.reference.model.state_dict().items()
    )
    changed = replace(kwargs["config"], guidance=GuidanceConfig("active", 0.5))
    with pytest.raises(ValueError, match="changed on resume"):
        campaign.run_t4(**{**kwargs, "config": changed}, resume=True)
    assert campaign.run_t4(**kwargs, resume=True) == result
    assert len(calls) == 3


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"budget": True}, "budget"),
        ({"seed": -1}, "seed"),
        ({"horizon": 4}, "horizon"),
        ({"parent_exploration": float("nan")}, "parent_exploration"),
        ({"lanes": ()}, "lanes"),
        ({"lead": "C.C"}, "connected"),
        ({"delta": 0.2}, "delta"),
        ({"lanes": ("made_up",)}, "lanes"),
    ],
)
def test_invalid_run_config_fails_at_boundary(changes, message):
    with pytest.raises(ValueError, match=message):
        settings(**changes)


def test_starting_lead_can_fail_endpoint_gate_without_becoming_best(tmp_path, monkeypatch):
    config = settings(lead="CC", lanes=("shallow",), budget=2)
    assert campaign.Fiber(config.lead, config.delta).check(config.lead) is None
    monkeypatch.setattr(campaign, "generate_panel", lambda *args: ([], {}))
    calls = []

    def score(smiles):
        calls.append(smiles)
        return -1.0

    result = campaign.run_t4(
        output=tmp_path / "ineligible-source",
        config=config,
        evaluate=score,
        evaluator_identity={"kind": "synthetic_unit_test"},
        implementation_identity={"fixture": "ineligible-source"},
    )
    assert calls == ["CC"]
    assert result["charged_calls"] == 1
    assert result["feasible_archive_count"] == 0
    assert result["best_smiles"] is None
    assert result["best_score"] is None


def test_all_supplied_leads_are_valid_starting_states():
    seeds = json.loads(
        (Path(__file__).resolve().parents[1] / "docs/GENMOL_T4_SEEDS.json").read_text()
    )
    assert len(seeds) == 15
    for row in seeds:
        campaign.T4RunConfig(row["smiles"], 0.6, row["idx"], lanes=("shallow",))
