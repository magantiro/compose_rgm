"""Tests for the executable instrument gate.

A gate that has never been shown to FAIL is not a gate. Each test below feeds it
a synthetic version of one of the five defects already recorded in this project
and asserts that it refuses the run -- because the failure mode being guarded
against is precisely a check that quietly passes everything.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pareto_instrument_gate import (  # noqa: E402
    ArmSemantics,
    Contrast,
    GateReport,
    Statistic,
    check_d1_selection_scoring_distinct,
    check_d1b_falsifying_range_declared,
    check_d2_arms_are_distinct,
    check_d3_no_false_null,
    check_d4_manifest_hashes,
    check_d5_parity,
    check_d6_hv_budget_matched,
    run_gate,
)


def _report() -> GateReport:
    return GateReport()


# ---------------------------------------------------------------------------
# D1 -- select and score with the same function
# ---------------------------------------------------------------------------

def test_d1_rejects_a_statistic_selected_and_scored_by_the_same_function():
    """The sacrifice-to-win defect: action chosen as argmax V_G, then scored by
    V_G. 50 higher, 5 tied, 0 lower out of 55 -- zero losses was definitional."""
    r = _report()
    check_d1_selection_scoring_distinct([Statistic(
        name="sacrifice_to_win", hypothesis="the sacrifice paid off",
        falsifying_range="[0, 1]",
        selection_function="argmax_V_G", scoring_function="argmax_V_G")], r)
    assert not r.passed


def test_d1_accepts_an_endpoint_comparison_against_a_separate_arm():
    r = _report()
    check_d1_selection_scoring_distinct([Statistic(
        name="endpoint_delta", hypothesis="verified lands better",
        falsifying_range="magnitude in [0, inf)",
        selection_function="argmin_V_G",
        scoring_function="endpoint_of_a_separately_run_greedy_arm")], r)
    assert r.passed


def test_d1b_rejects_a_statistic_with_no_declared_falsifying_range():
    r = _report()
    check_d1b_falsifying_range_declared([Statistic(
        name="mystery", hypothesis="something", falsifying_range="   ",
        selection_function="a", scoring_function="b")], r)
    assert not r.passed


# ---------------------------------------------------------------------------
# D2 -- a "verified" arm that is secretly greedy
# ---------------------------------------------------------------------------

def test_d2_rejects_two_arms_whose_committed_actions_are_always_identical():
    """The exact defect that voided two runs: `verified_land` was a plain greedy
    continuation while the loop committed greedy's action, so headroom was 0 by
    construction."""
    identical = {"s1": ("a", "b"), "s2": ("c", "d")}
    r = _report()
    check_d2_arms_are_distinct(
        {"verified_pref": identical, "greedy_pref": dict(identical)},
        [("verified_pref", "greedy_pref")], r)
    assert not r.passed


def test_d2_accepts_arms_that_differ_on_at_least_one_source():
    r = _report()
    check_d2_arms_are_distinct(
        {"verified_pref": {"s1": ("a", "b"), "s2": ("x", "y")},
         "greedy_pref": {"s1": ("a", "b"), "s2": ("c", "d")}},
        [("verified_pref", "greedy_pref")], r)
    assert r.passed


def test_d2_rejects_when_the_two_arms_share_no_sources():
    """Silence is not evidence of distinctness."""
    r = _report()
    check_d2_arms_are_distinct(
        {"verified_pref": {"s1": ("a",)}, "greedy_pref": {"s2": ("b",)}},
        [("verified_pref", "greedy_pref")], r)
    assert not r.passed


# ---------------------------------------------------------------------------
# D3 -- a p-value against a null a theorem makes false
# ---------------------------------------------------------------------------

def test_d3_rejects_a_pvalue_on_a_guaranteed_sign_statistic():
    """Policy improvement guarantees verified >= greedy, so a sign test against
    0.5 tests something already known before any data existed."""
    r = _report()
    check_d3_no_false_null([Statistic(
        name="verified_beats_greedy", hypothesis="verified is better",
        falsifying_range="none -- the sign is guaranteed",
        selection_function="argmin_V_G", scoring_function="endpoint",
        sign_is_guaranteed=True, reports_pvalue=True)], r)
    assert not r.passed


def test_d3_allows_magnitude_only_for_a_guaranteed_sign_statistic():
    r = _report()
    check_d3_no_false_null([Statistic(
        name="verified_minus_greedy_magnitude", hypothesis="the gap is large",
        falsifying_range="magnitude in [0, inf)",
        selection_function="argmin_V_G", scoring_function="endpoint",
        sign_is_guaranteed=True, reports_pvalue=False)], r)
    assert r.passed


# ---------------------------------------------------------------------------
# D4 -- a fabricated SHA-256
# ---------------------------------------------------------------------------

def test_d4_rejects_a_hash_that_does_not_match_the_file(tmp_path):
    target = tmp_path / "artifact.json"
    target.write_text("real contents\n")
    manifest = tmp_path / "handoff.json"
    manifest.write_text(json.dumps({"frozen_inputs": {"artifact.json": "0" * 64}}))
    r = _report()
    check_d4_manifest_hashes(manifest, tmp_path, r)
    assert not r.passed


def test_d4_accepts_a_hash_recomputed_from_disk(tmp_path):
    target = tmp_path / "artifact.json"
    target.write_text("real contents\n")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = tmp_path / "handoff.json"
    manifest.write_text(json.dumps({"frozen_inputs": {"artifact.json": digest}}))
    r = _report()
    check_d4_manifest_hashes(manifest, tmp_path, r)
    assert r.passed


def test_d4_rejects_a_manifest_naming_a_file_that_does_not_exist(tmp_path):
    manifest = tmp_path / "handoff.json"
    manifest.write_text(json.dumps({"frozen_inputs": {"ghost.json": "a" * 64}}))
    r = _report()
    check_d4_manifest_hashes(manifest, tmp_path, r)
    assert not r.passed


# ---------------------------------------------------------------------------
# D5 -- the parity confound
# ---------------------------------------------------------------------------

SEMANTICS = {
    "greedy_B":   ArmSemantics("greedy", "x_3", "H-tau", "B"),
    "verified_B": ArmSemantics("verified", "x_3", "H-tau", "B"),
    "continue_A": ArmSemantics("greedy", "x_3", "H-tau", "A"),
}


def test_d5_rejects_the_exact_confound_the_main_lane_found():
    """`verified_retarget` vs `continue_A` varies controller AND objective. The
    confound inflated the main lane's Q1 by roughly 11%."""
    r = _report()
    check_d5_parity([Contrast("Q1", "verified_B", "continue_A", "objective")],
                    SEMANTICS, r)
    assert not r.passed


def test_d5_accepts_the_matched_form_of_the_same_question():
    r = _report()
    check_d5_parity([Contrast("Q1_matched", "greedy_B", "continue_A", "objective")],
                    SEMANTICS, r)
    assert r.passed


def test_d5_lets_a_context_only_contrast_stand_but_records_what_it_varies():
    r = _report()
    check_d5_parity([Contrast("floor", "verified_B", "continue_A", "objective",
                              status="CONTEXT_ONLY")], SEMANTICS, r)
    assert r.passed
    detail = r.checks[0]["detail"]["contrasts"]["floor"]
    assert detail["varies"] == ["controller", "objective"]
    assert detail["isolates_one_mechanism"] is False


def test_d5_rejects_a_contrast_naming_an_undeclared_arm():
    r = _report()
    check_d5_parity([Contrast("bad", "ghost", "continue_A", "objective")],
                    SEMANTICS, r)
    assert not r.passed


# ---------------------------------------------------------------------------
# D6 -- hypervolume at unmatched budget
# ---------------------------------------------------------------------------

#: A generate-then-rank contrast: handing gen_rank the control arm's budget IS
#: the point, so compute parity is claimed here and is enforced.
CONTRAST = [Contrast("P3", "greedy_pref", "gen_rank", "controller",
                     budget_parity_claimed=True)]
#: A lookahead-vs-myopic contrast: the compute asymmetry is the mechanism under
#: test, so it is reported rather than failed.
INTRINSIC = [Contrast("P2", "verified_pref", "greedy_pref", "controller")]


def test_d6_rejects_unequal_endpoint_counts():
    """An arm contributing more endpoints wins on hypervolume by arithmetic."""
    r = _report()
    check_d6_hv_budget_matched(
        CONTRAST,
        {"greedy_pref": {"native_oracle_calls": 100, "kernel_calls": 26},
         "gen_rank": {"native_oracle_calls": 100, "kernel_calls": 26}},
        {"greedy_pref": 5, "gen_rank": 50}, r)
    assert not r.passed
    assert r.checks[0]["detail"]["contrasts"]["P3"]["flag"] == "UNEQUAL_ENDPOINTS"


def test_d6_rejects_a_kernel_budget_blowout_even_at_matched_oracle_calls():
    """The ~600x budget-axis gap: matching native oracle calls can hand one arm
    a vastly larger kernel budget, which is the inflation channel wearing a
    benchmark convention's clothes."""
    r = _report()
    check_d6_hv_budget_matched(
        CONTRAST,
        {"greedy_pref": {"native_oracle_calls": 15600, "kernel_calls": 26},
         "gen_rank": {"native_oracle_calls": 15600, "kernel_calls": 93600}},
        {"greedy_pref": 5, "gen_rank": 5}, r)
    assert not r.passed
    assert r.checks[0]["detail"]["contrasts"]["P3"]["flag"] == "BUDGET_ASYMMETRIC_KERNEL"


def test_d6_rejects_a_native_oracle_call_gap_beyond_tolerance():
    r = _report()
    check_d6_hv_budget_matched(
        CONTRAST,
        {"greedy_pref": {"native_oracle_calls": 100, "kernel_calls": 26},
         "gen_rank": {"native_oracle_calls": 400, "kernel_calls": 26}},
        {"greedy_pref": 5, "gen_rank": 5}, r)
    assert not r.passed
    assert r.checks[0]["detail"]["contrasts"]["P3"]["flag"] == "BUDGET_ASYMMETRIC_NATIVE"


def test_d6_accepts_a_matched_contrast():
    r = _report()
    check_d6_hv_budget_matched(
        CONTRAST,
        {"greedy_pref": {"native_oracle_calls": 100, "kernel_calls": 26},
         "gen_rank": {"native_oracle_calls": 105, "kernel_calls": 28}},
        {"greedy_pref": 5, "gen_rank": 5}, r)
    assert r.passed


# ---------------------------------------------------------------------------
# The lane's own declarations
# ---------------------------------------------------------------------------

def test_this_lanes_declared_statistics_and_contrasts_pass_the_gate():
    """The design-stage self-test. It failed twice while this lane was being
    written -- once on a G4 statistic with no falsifying range, once on a parity
    table that claimed P1 varied only the controller -- and both were fixed
    before any run."""
    assert run_gate().passed


def test_d6_reports_rather_than_fails_an_intrinsic_compute_asymmetry():
    """P2 compares a lookahead controller to a myopic one. Throttling the
    lookahead to greedy's compute would delete the mechanism under test, so the
    edit budget is held, endpoint counts are still forced equal, and the compute
    ratio is REPORTED. Failing here would make the contrast unrunnable."""
    r = _report()
    check_d6_hv_budget_matched(
        INTRINSIC,
        {"verified_pref": {"native_oracle_calls": 180000, "kernel_calls": 300},
         "greedy_pref": {"native_oracle_calls": 15600, "kernel_calls": 26}},
        {"verified_pref": 5, "greedy_pref": 5}, r)
    assert r.passed
    entry = r.checks[0]["detail"]["contrasts"]["P2"]
    assert entry["flag"] == "COMPUTE_ASYMMETRIC_BY_DESIGN_REPORT_THE_RATIO"
    assert entry["kernel_call_ratio"] > 10


def test_d6_still_forces_equal_endpoint_counts_even_when_compute_is_intrinsic():
    """The inflation channel is closed everywhere: an arm may cost more, but it
    may never contribute more points to the hypervolume."""
    r = _report()
    check_d6_hv_budget_matched(
        INTRINSIC,
        {"verified_pref": {"native_oracle_calls": 180000, "kernel_calls": 300},
         "greedy_pref": {"native_oracle_calls": 15600, "kernel_calls": 26}},
        {"verified_pref": 40, "greedy_pref": 5}, r)
    assert not r.passed
    assert r.checks[0]["detail"]["contrasts"]["P2"]["flag"] == "UNEQUAL_ENDPOINTS"
