"""The extension guard must resume its own failed attempt without becoming a retry.

An extension archives the prior result and removes ``result.json`` BEFORE the run
starts.  So an extension that dies partway leaves a folder with no result at all, and a
guard keyed on ``result.json`` alone would refuse to resume the very state it created --
while a guard that simply dropped the check would license a genuine retry that
double-charges the oracle.  These tests pin the boundary between the two.
"""

from __future__ import annotations

import json

import pytest

from compose_v4.experiments.pmo_population_v1 import resolve_extension_prior


def _result(path, charged):
    path.write_text(json.dumps({"charged_calls": charged, "task": "celecoxib_rediscovery"}))


def test_completed_run_with_a_result_extends(tmp_path):
    _result(tmp_path / "result.json", 250)
    prior_path, budget = resolve_extension_prior(tmp_path)
    assert prior_path.name == "result.json"
    assert budget == 250


def test_failed_extension_attempt_resumes_from_its_own_archive(tmp_path):
    # Exactly the state the first 1k attempt left: result archived, result.json gone,
    # started.json rewritten by the attempt, failure.json recording why it died.
    _result(tmp_path / "result_at_250_calls.json", 250)
    (tmp_path / "started.json").write_text("{}")
    (tmp_path / "failure.json").write_text(json.dumps({"error_type": "TypeError"}))
    prior_path, budget = resolve_extension_prior(tmp_path)
    assert prior_path.name == "result_at_250_calls.json"
    assert budget == 250


def test_an_in_flight_extension_is_refused(tmp_path):
    # started.json with no failure.json means a container may still be running, and a
    # second container against the same ledger could double-charge.
    _result(tmp_path / "result_at_250_calls.json", 250)
    (tmp_path / "started.json").write_text("{}")
    with pytest.raises(RuntimeError, match="still in flight"):
        resolve_extension_prior(tmp_path)


def test_a_task_that_never_completed_is_still_a_forbidden_retry(tmp_path):
    (tmp_path / "started.json").write_text("{}")
    (tmp_path / "failure.json").write_text(json.dumps({"error_type": "TypeError"}))
    with pytest.raises(RuntimeError, match="requires a COMPLETED prior run"):
        resolve_extension_prior(tmp_path)


def test_an_empty_folder_is_a_forbidden_retry(tmp_path):
    with pytest.raises(RuntimeError, match="requires a COMPLETED prior run"):
        resolve_extension_prior(tmp_path)


def test_an_archive_whose_name_disagrees_with_its_content_is_refused(tmp_path):
    # The filename and the charged-call count inside are two independent records of
    # one number; if they disagree the prior spend is ambiguous, so refuse.
    _result(tmp_path / "result_at_250_calls.json", 400)
    (tmp_path / "failure.json").write_text("{}")
    (tmp_path / "started.json").write_text("{}")
    with pytest.raises(RuntimeError, match="ambiguous prior run"):
        resolve_extension_prior(tmp_path)


def test_the_latest_archive_wins_across_successive_extensions(tmp_path):
    # A 250 -> 1000 -> 2500 ladder must resume from 1000, not from 250.  Sorting is
    # numeric, so a lexical sort putting "250" after "1000" would fail this.
    _result(tmp_path / "result_at_250_calls.json", 250)
    _result(tmp_path / "result_at_1000_calls.json", 1000)
    (tmp_path / "failure.json").write_text("{}")
    (tmp_path / "started.json").write_text("{}")
    prior_path, budget = resolve_extension_prior(tmp_path)
    assert (prior_path.name, budget) == ("result_at_1000_calls.json", 1000)
