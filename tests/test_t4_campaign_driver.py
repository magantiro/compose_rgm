from pathlib import Path

import pytest

from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_campaign_driver import (
    campaign_driver_action,
    campaign_finished,
)
from compose_v4.experiments.t4_integrated_route_fiber_v2 import durable_phase_action

ROOT = Path(__file__).resolve().parents[1]
FROZEN_BRAF_V4_APP = ROOT / "modal_apps/t4_integrated_route_fiber_braf_v2_app.py"
FROZEN_BRAF_V4_APP_SHA256 = "6c092e0300278157e61851ecffdee609b6dd85ddb54c2401c9f3eae548934087"
SHARED_APPS = (
    FROZEN_BRAF_V4_APP,
    ROOT / "modal_apps/t4_shared_retained_fiber_parp1_v2_app.py",
    ROOT / "modal_apps/t4_shared_retained_fiber_jak2_v2_app.py",
    ROOT / "modal_apps/t4_shared_retained_fiber_fa7_d04_app.py",
    ROOT / "modal_apps/t4_shared_retained_fiber_fa7_d06_app.py",
    ROOT / "modal_apps/t4_shared_retained_fiber_5ht1b_d04_v1_app.py",
    ROOT / "modal_apps/t4_shared_retained_fiber_5ht1b_d06_v1_app.py",
    ROOT / "modal_apps/t4_shared_retained_fiber_parp1_p0_rescue_v1_app.py",
)


@pytest.mark.parametrize("status", ["running", "proposals_running", "queries_running"])
def test_every_active_phase_requires_continuation(status):
    assert campaign_finished([status]) is False
    assert campaign_driver_action([status], continuation_state="none") == ("reserve_and_spawn")


def test_mixed_terminal_and_active_cells_are_not_finished():
    statuses = ["complete_budget", "proposals_running", "candidate_exhaustion"]
    assert campaign_finished(statuses) is False
    assert campaign_driver_action(statuses, continuation_state="none") == ("reserve_and_spawn")


def test_only_explicit_terminal_statuses_finish_the_campaign():
    statuses = ["complete_budget", "candidate_exhaustion", "failed"]
    assert campaign_finished(statuses) is True
    assert campaign_driver_action(statuses, continuation_state="running") == "finish"


@pytest.mark.parametrize("state", ["reserved", "running"])
def test_preempted_or_live_continuation_is_not_spawned_twice(state):
    assert campaign_driver_action(["proposals_running"], continuation_state=state) == "wait"


def test_resume_spawns_only_after_prior_continuation_is_known_terminal():
    assert campaign_driver_action(["running"], continuation_state="terminal") == "reserve_and_spawn"


@pytest.mark.parametrize("statuses", [[], ["mystery"], ["running", "mystery"]])
def test_ambiguous_cell_status_fails_closed(statuses):
    with pytest.raises(ValueError, match="no cell statuses|unknown cell statuses"):
        campaign_finished(statuses)


def test_unknown_continuation_state_fails_closed():
    with pytest.raises(ValueError, match="unknown continuation state"):
        campaign_driver_action(["running"], continuation_state="lost")
    with pytest.raises(ValueError, match="unknown continuation state"):
        campaign_driver_action(["complete_budget"], continuation_state="lost")


def test_locked_scored_calls_remain_no_retry_during_driver_resume():
    assert (
        durable_phase_action(
            lock_exists=True,
            receipt_statuses=["complete", "reserved", "missing"],
        )
        == "fail_closed"
    )


def test_live_braf_bug_is_quarantined_from_other_shared_driver_sources():
    unsafe = {
        path
        for path in SHARED_APPS
        if 'all(row["status"] != "running" for row in records)' in path.read_text()
    }
    assert unsafe == {FROZEN_BRAF_V4_APP}
    assert sha256_file(FROZEN_BRAF_V4_APP) == FROZEN_BRAF_V4_APP_SHA256
