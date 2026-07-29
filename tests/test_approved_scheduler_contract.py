"""The owner-approved 16,000-step scheduler is the ACTIVE scientific contract.

This is an approved contract update, not tolerated drift: the 3,000-step schedule was the last approved
configuration before the A100 throughput benchmark, and 16,000 was approved afterwards as the maximum
OBSERVATION horizon. It does not preselect the final checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import ring_core_identity as ri  # noqa: E402

APPROVED = {
    "optimizer": "AdamW",
    "peak_learning_rate": 3e-4,
    "weight_decay": 1e-05,
    "warmup_steps": 500,
    "schedule_steps": 16000,
    "minimum_learning_rate_fraction": 0.05,
    "schedule": "cosine_with_linear_warmup",
}
APPROVED_HASH = "dafd4b5092414394"
SUPERSEDED_HASH = "0b832985c65de1cc"


def test_approved_configuration_reproduces_the_new_hash():
    assert ri.scheduler_config_hash_from_args(**APPROVED) == APPROVED_HASH
    assert ri.SCHEDULER_CONFIG_HASH == APPROVED_HASH
    assert ri.recompute_scheduler_config_hash() == APPROVED_HASH


def test_production_scheduler_matches_the_approved_package():
    assert ri.PRODUCTION_SCHEDULER == APPROVED


def test_the_old_3000_step_configuration_no_longer_satisfies_the_contract():
    """The superseded schedule must now FAIL the active contract, or the update means nothing."""
    old = {**APPROVED, "schedule_steps": 3000}
    old_hash = ri.scheduler_config_hash_from_args(**old)
    assert old_hash == SUPERSEDED_HASH
    assert old_hash != ri.SCHEDULER_CONFIG_HASH


@pytest.mark.parametrize(
    "override",
    [
        {"schedule_steps": 12000},
        {"schedule_steps": 8000},
        {"warmup_steps": 1000},
        {"peak_learning_rate": 1e-4},
        {"weight_decay": 1e-4},
        {"minimum_learning_rate_fraction": 0.1},
        {"optimizer": "Adam"},
        {"schedule": "linear"},
    ],
)
def test_arbitrary_scheduler_changes_still_fail_loudly(override):
    """Approving 16k must not weaken the guard against every OTHER unapproved configuration."""
    drifted = {**APPROVED, **override}
    assert ri.scheduler_config_hash_from_args(**drifted) != ri.SCHEDULER_CONFIG_HASH


def test_lineage_preserves_the_superseded_decision():
    lineage = {entry["scheduler_hash"]: entry for entry in ri.SCHEDULER_LINEAGE}
    assert lineage[SUPERSEDED_HASH]["status"] == "SUPERSEDED"
    assert lineage[SUPERSEDED_HASH]["schedule_steps"] == 3000
    assert lineage[APPROVED_HASH]["status"] == "ACTIVE"
    assert lineage[APPROVED_HASH]["schedule_steps"] == 16000
    assert sum(1 for e in ri.SCHEDULER_LINEAGE if e["status"] == "ACTIVE") == 1


def test_maximum_horizon_is_not_a_checkpoint_preselection():
    """Documented intent: 16,000 bounds observation; selection remains the frozen rule."""
    import re

    source = (REPO / "scripts/ring_core_identity.py").read_text()
    # the comment wraps across lines, so strip comment markers and collapse whitespace before matching
    flat = re.sub(r"\s+", " ", source.replace("#", " "))
    assert "MAXIMUM OBSERVATION HORIZON" in flat
    assert "does not preselect the final checkpoint" in flat
