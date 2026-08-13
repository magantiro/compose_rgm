"""Tests for the shards-to-contrasts analysis.

This script is what turns a smoke run into the numbers a reader sees, so its
failure modes are the expensive ones: reporting statistics from an instrument
that did not pass, or dropping a contrast so quietly that nobody notices it was
never computed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/analyse_pareto_control.py"

ARMS = ("unguided", "greedy_pref", "verified_pref",
        "gen_rank@greedy", "gen_rank@verified")
KERNEL = {"unguided": 26, "greedy_pref": 26, "verified_pref": 300,
          "gen_rank@greedy": 27, "gen_rank@verified": 310}
PREFS = (0.1, 0.3, 0.5, 0.7, 0.9)


def write_shards(directory: Path, n: int = 6, *, identical_arms: bool = False,
                 unequal_endpoints: bool = False, with_fan: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        payload = {"index": i, "source": f"S{i}",
                   "pair": "potency_vs_developability", "arms": {}}
        for arm in ARMS:
            label = "shared" if identical_arms and "pref" in arm else arm
            count = 5
            if unequal_endpoints and arm == "gen_rank@greedy":
                count = 9
            payload["arms"][arm] = {
                "endpoints": [f"{arm}_{i}_{p}" for p in range(count)],
                "endpoint_z": [[0.5 + 2.0 * w, 2.5 - 2.0 * w] for w in PREFS],
                "action_sequences": [[f"{label}{i}{p}{s}" for s in range(6)]
                                     for p in range(5)],
                "normalized_hypervolume": 0.3 + 0.01 * i,
                "preference_coverage": 0.8,
                "nondominated_set_size": 4,
                "endpoint_diversity": 0.6,
                "feasible": [True] * 5,
                "cost": {"native_oracle_calls": KERNEL[arm] * 600,
                         "raw_oracle_calls": KERNEL[arm] * 900,
                         "kernel_calls": KERNEL[arm]},
            }
        if with_fan:
            payload["prefix_branching"] = {
                "branch_point": f"x3_{i}",
                "prefix_states": [f"x0_{i}", f"x1_{i}", f"x2_{i}", f"x3_{i}"],
                "branches": {str(w): {"states": [f"x3_{i}", f"b{i}_{k}"],
                                      "endpoint": f"b{i}_{k}",
                                      "endpoint_z": [0.5 + 2.0 * w, 2.5 - 2.0 * w],
                                      "complete": True}
                             for k, w in enumerate(PREFS)},
            }
        (directory / f"{i:03d}.json").write_text(json.dumps(payload))


def run(shards: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--shards", str(shards), "--out", str(out),
         "--manifest", str(shards / "nonexistent.json")],
        capture_output=True, text=True, cwd=REPO)


def test_a_clean_run_reports_every_contrast(tmp_path):
    shards, out = tmp_path / "shards", tmp_path / "analysis.json"
    write_shards(shards)
    assert run(shards, out).returncode == 0
    data = json.loads(out.read_text())
    assert data["status"] == "SMOKE_HELD_IN"
    assert data["gate"]["global_checks_passed"] is True
    assert data["gate"]["barred_contrasts"] == []
    for name in ("P1_unguided_floor", "P2_future_awareness",
                 "P3_closed_vs_open_loop", "P4_closed_vs_open_loop_verified"):
        assert name in data["contrasts"]
    assert "P5_preference_responsiveness" in data["within_arm_contrasts"]
    assert "P6_same_prefix_branching" in data["within_arm_contrasts"]
    assert not data["skipped_contrasts"]


def test_a_failing_gate_produces_no_statistics_at_all(tmp_path):
    """The point of running the gate first. Numbers with a warning above them
    are read as numbers; every recorded defect in this project was found while
    reading a result that had already been written down and believed."""
    shards, out = tmp_path / "shards", tmp_path / "analysis.json"
    write_shards(shards, identical_arms=True)
    result = run(shards, out)
    assert result.returncode == 1
    data = json.loads(out.read_text())
    assert data["status"] == "INVALID_INSTRUMENT"
    assert "contrasts" not in data
    assert "arms" not in data


def test_unequal_endpoint_counts_BAR_THE_CONTRAST_not_the_whole_run(tmp_path):
    """An arm contributing more endpoints wins on hypervolume by arithmetic, so
    the affected contrast must be withheld.

    But it must NOT suppress every other number. An earlier version wrote
    INVALID_INSTRUMENT and no statistics at all, which made the D6 tolerance the
    only thing standing between the reader and a full report -- exactly the
    pressure that gets tolerances loosened. D6 is per-contrast; D1-D5 are global.
    """
    shards, out = tmp_path / "shards", tmp_path / "analysis.json"
    write_shards(shards, unequal_endpoints=True)
    assert run(shards, out).returncode == 0
    data = json.loads(out.read_text())
    assert data["status"] == "SMOKE_HELD_IN"

    barred = data["gate"]["barred_contrasts"]
    assert "P3_closed_vs_open_loop" in barred
    p3 = data["contrasts"]["P3_closed_vs_open_loop"]
    assert p3["status"] == "INVALID_CONTRAST"
    assert "WITHHELD" in p3["normalized_hypervolume"]

    # everything not implicated still stands
    assert "WITHHELD" not in data["contrasts"]["P2_future_awareness"][
        "normalized_hypervolume"]
    assert data["preference_responsiveness"]
    assert data["efficiency"]["N_90_trajectories"]


def test_a_GLOBAL_check_failure_still_suppresses_every_number(tmp_path):
    """The per-contrast change must not weaken the global checks: if arms are
    indistinguishable, nothing is trustworthy and nothing is written."""
    shards, out = tmp_path / "shards", tmp_path / "analysis.json"
    write_shards(shards, identical_arms=True)
    assert run(shards, out).returncode == 1
    data = json.loads(out.read_text())
    assert data["status"] == "INVALID_INSTRUMENT"
    assert "contrasts" not in data
    assert "D2_arms_are_distinct" in data["failed_checks"]


def test_p5_measures_preference_responsiveness_and_could_be_zero(tmp_path):
    shards, out = tmp_path / "shards", tmp_path / "analysis.json"
    write_shards(shards)
    run(shards, out)
    p5 = json.loads(out.read_text())["within_arm_contrasts"]["P5_preference_responsiveness"]
    # objective 0 runs 0.5+2w, so w=0.9 minus w=0.1 is exactly 1.6.
    assert p5["objective_0_gap"]["mean"] == pytest.approx(1.6, abs=1e-9)
    assert p5["varies"] == "objective"
    assert "preference-blind controller gives 0" in p5["falsifying_range"]


def test_p6_asserts_the_branches_really_shared_the_branch_point(tmp_path):
    shards, out = tmp_path / "shards", tmp_path / "analysis.json"
    write_shards(shards)
    run(shards, out)
    p6 = json.loads(out.read_text())["within_arm_contrasts"]["P6_same_prefix_branching"]
    assert p6["all_branches_started_at_the_branch_point"] is True
    assert p6["mean_distinct_endpoints"] == pytest.approx(5.0)


def test_a_missing_fan_is_recorded_not_silently_dropped(tmp_path):
    shards, out = tmp_path / "shards", tmp_path / "analysis.json"
    write_shards(shards, with_fan=False)
    run(shards, out)
    data = json.loads(out.read_text())
    assert "P6_same_prefix_branching" not in data["within_arm_contrasts"]
    # P5 still computes -- it needs only the per-preference endpoints.
    assert "P5_preference_responsiveness" in data["within_arm_contrasts"]


def test_set_level_hypervolume_is_NOT_marked_sign_guaranteed(tmp_path):
    """This test previously asserted the opposite, and in doing so it ENFORCED a
    defect: the registry was keyed on the contrast alone, so HV_verified -
    HV_greedy was marked sign-guaranteed and its comparison suppressed.

    Pointwise policy improvement guarantees the scalarized value FOR EACH
    preference. It does not guarantee that the five endpoints enclose more
    dominated area AS A SET -- a verified action can improve one preference while
    moving endpoints closer together and reducing complementary coverage. Smoke
    source 000 returned verified 1.0060 against greedy 1.0074, the falsifying
    value the old registry claimed could not exist.
    """
    shards, out = tmp_path / "shards", tmp_path / "analysis.json"
    write_shards(shards)
    run(shards, out)
    hv = json.loads(out.read_text())["contrasts"]["P2_future_awareness"][
        "normalized_hypervolume"]
    assert "sign_is_guaranteed" not in hv
    assert "pvalue_NOT_REPORTED" not in hv


def test_the_two_questions_are_reported_separately(tmp_path):
    """Per-preference and set-level must both appear, with opposite sign status,
    so neither can stand in for the other."""
    shards, out = tmp_path / "shards", tmp_path / "analysis.json"
    write_shards(shards)
    run(shards, out)
    q = json.loads(out.read_text())["two_separated_questions"]
    assert q["per_preference_scalarized_value"]["sign_is_guaranteed"] is True
    assert "pvalue_NOT_REPORTED" in q["per_preference_scalarized_value"]
    assert q["set_level_hypervolume"]["sign_is_guaranteed"] is False
    assert "falsifying_range" in q["set_level_hypervolume"]


def test_p99_exceedance_is_descriptive_and_nothing_is_clipped(tmp_path):
    shards, out = tmp_path / "shards", tmp_path / "analysis.json"
    write_shards(shards)
    run(shards, out)
    e = json.loads(out.read_text())["p99_exceedance"]
    assert "descriptive" in e["statistic"]
    assert "no_clipped_variant" in e
    assert e["fraction_of_endpoints_beyond_z_star"] is not None


def test_withdrawn_statistics_travel_with_the_report(tmp_path):
    """So a reader of the output alone cannot reintroduce one."""
    shards, out = tmp_path / "shards", tmp_path / "analysis.json"
    write_shards(shards)
    run(shards, out)
    withdrawn = json.loads(out.read_text())["withdrawn_statistics"]
    assert "sacrifice_to_win" in withdrawn
    assert "best_candidate_reaches_pooled_p99" in withdrawn
