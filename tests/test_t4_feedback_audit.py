"""Audit iteration begins after the archive's actual event, not a fixed round."""

from test_t4_warm_continuation import fixture_source

from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_warm_continuation import initial_archive
from tools.t4_warm_audit import report


def test_feedback_audit_does_not_require_a_fictitious_completed_round_two(tmp_path):
    source, _, _ = fixture_source(tmp_path)
    warm = initial_archive(unseal(source / "candidate_lock.json"), unseal(source / "docking.json"))
    warm["round"] = 2  # Synthetic partial-event history, not scientific evidence.
    output = tmp_path / "feedback"
    seal(output / "warm_start.json", warm)
    seal(output / "round_3/parents/00.json", {"bundles": [], "cumulative_executor_calls": 123})
    result = report(output)
    assert len(result["rounds"]) == 1 and result["rounds"][0]["round"] == 3
    assert result["rounds"][0]["status"] == "partial_preparation_not_an_oracle_pool"
    assert result["rounds"][0]["checkpointed_executor_calls"] == 123
