"""The voided PMO gsk3b reading must be unreportable, and its run-mates must survive.

The failure this guards is specific: a complete, well-formed `result.json` whose
`best_score` is a runtime defect rather than a measurement.  Nothing in the artifact
distinguishes it from a real zero, so the exclusion has to live somewhere a summary
is forced to consult.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.experiments.pmo_run_status import (
    INVALID_ORACLE_RUNTIME_RELATIVE_PATH,
    VALID,
    PmoReadingExcluded,
    admissible_results,
    assert_admissible,
    entry_for,
    is_excluded,
    load_ledger,
    status_of,
)

ROOT = Path(__file__).resolve().parents[1]
VOIDED_RUN = "79fec4984821e97de6decefd213e45a47b3d2b901a1a9c021ee0dfd3ed1561f4"


def test_the_gsk3b_reading_is_voided_and_cannot_be_reported():
    assert status_of(VOIDED_RUN, "gsk3b") == INVALID_ORACLE_RUNTIME_RELATIVE_PATH
    assert is_excluded(VOIDED_RUN, "gsk3b") is True
    with pytest.raises(PmoReadingExcluded) as failure:
        assert_admissible(VOIDED_RUN, "gsk3b")
    assert "INVALID_ORACLE_RUNTIME_RELATIVE_PATH" in str(failure.value)


def test_the_two_rdkit_only_tasks_in_the_same_run_stay_valid():
    """Voiding by RUN would have discarded two sound measurements."""
    for task, best in (("perindopril_mpo", 0.4864578310337349),
                       ("celecoxib_rediscovery", 0.1958041958041958)):
        assert status_of(VOIDED_RUN, task) == VALID
        assert is_excluded(VOIDED_RUN, task) is False
        assert_admissible(VOIDED_RUN, task)
        assert entry_for(VOIDED_RUN, task)["recorded_reading"]["best_score"] == best


def test_the_voided_entry_preserves_the_artifact_and_says_what_the_zero_means():
    entry = entry_for(VOIDED_RUN, "gsk3b")
    assert entry["artifacts_preserved"] is True
    assert entry["the_zero_carries_no_information_about_gsk3b_chemistry"] is True
    assert entry["recorded_reading"]["charged_oracle_calls"] == 250
    # The counterfactual is what makes this a diagnosis rather than a suspicion.
    counterfactual = entry["counterfactual"]
    assert counterfactual["production_pattern_five_known_actives"] == [0.0] * 5
    assert counterfactual["cwd_held_at_assets_during_call"] == [1.0] * 5
    for relative in entry["evidence"]:
        assert (ROOT / relative).exists(), f"missing evidence file {relative}"


def test_a_summary_filter_drops_only_the_voided_task():
    results = {
        "gsk3b": {"best_score": 0.0},
        "perindopril_mpo": {"best_score": 0.4864578310337349},
        "celecoxib_rediscovery": {"best_score": 0.1958041958041958},
    }
    kept = admissible_results(results, VOIDED_RUN)
    assert set(kept) == {"perindopril_mpo", "celecoxib_rediscovery"}


def test_an_unlisted_reading_is_admissible():
    """The ledger records exceptions, not permissions; a fresh run is not blocked."""
    assert status_of("0" * 64, "gsk3b") == VALID
    assert_admissible("0" * 64, "gsk3b")


def test_the_ledger_is_well_formed():
    ledger = load_ledger()
    assert ledger["schema_version"] == "pmo_run_status_ledger_v1"
    assert ledger["default_status_for_unlisted"] == VALID
    required = {"run_id", "task", "status", "exclude_from_performance_summaries", "reason"}
    for entry in ledger["entries"]:
        assert required <= set(entry), f"incomplete entry: {entry.get('task')}"
        assert len(entry["run_id"]) == 64
        # An exclusion without a reason is a superstition.
        assert entry["reason"].strip()
    raw = json.loads((ROOT / "diagnostics/pmo_run_status_ledger.json").read_text())
    assert raw == ledger
