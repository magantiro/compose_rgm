"""Tests for the generated STATUS page.

STATUS.md is the first thing a main-session agent reads, per the binding handoff
evaluation order. A wrong heading there misdirects the whole review, so the two
failure modes worth pinning are: calling a failing pair "the adopted pair", and
hand-typed numbers drifting from the artifact.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/pareto_render_status.py"


def census_fixture(*, adopted: str | None) -> dict:
    def pair(name: str, a: str, b: str, verdict: str) -> dict:
        return {
            "pair": name, "objective_a": a, "objective_b": b,
            "spearman_rho": -0.25,
            "I_A": {"tradeoff_move_fraction": 0.50,
                    "state_offers_a_up_b_down": 0.9,
                    "state_offers_a_down_b_up": 0.8,
                    "max_binding_share": 0.6,
                    "mean_distinct_selections": 2.8,
                    "unanimous_fraction": 0.05},
            "gate": {
                "verdict": verdict,
                "failed": [] if verdict == "PASS" else ["G4_no_saturation"],
                "gates": {
                    "G1_alignment": {"pass": True},
                    "G2_local_tradeoff": {"pass": True},
                    "G3_no_domination": {"pass": True},
                    "G4_no_saturation": {"pass": verdict == "PASS",
                                         "reach_fractions": {a: 0.2, b: 0.3}},
                    "G5_front_richness": {"pass": True},
                }},
        }

    verdicts = ("PASS", "FAIL", "FAIL") if adopted else ("FAIL", "FAIL", "FAIL")
    return {
        "adopted_pair": adopted,
        "predeclared_gate_thresholds": {
            "G1_max_rho": 0.7, "G2_min_tradeoff_fraction": 0.2,
            "G2_min_each_direction": 0.05, "G2_min_state_availability": 0.25,
            "G3_max_binding_share": 0.9, "G4_max_reach_fraction": 0.85,
            "G5_min_distinct_selections": 2.0, "G5_max_unanimous_fraction": 0.5},
        "instruments": {
            "I_A": {"n_states": 120, "mean_fiber_width": 586.0},
            "I_B": {"n_pairs": 81500, "n_states": 5592}},
        "C5_reachability": {
            "n_sources": 20,
            "P": {"reach_fraction": 0.2, "median_end": 1.1, "median_movement": 3.3},
            "D": {"reach_fraction": 0.1, "median_end": 0.4, "median_movement": 0.77},
            "S": {"inert": True, "median_abs_movement": 0.0,
                  "reach_fraction_assigned": 1.0},
            "withdrawn_statistic": "best candidate reaches pooled p99"},
        "pairs": [
            pair("potency_vs_developability", "P", "D", verdicts[0]),
            pair("potency_vs_source_similarity", "P", "S", verdicts[1]),
            pair("developability_vs_source_similarity", "D", "S", verdicts[2]),
        ],
    }


def render(tmp_path: Path, *, adopted: str | None) -> str:
    census = tmp_path / "census.json"
    census.write_text(json.dumps(census_fixture(adopted=adopted)))
    out = tmp_path / "STATUS.md"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--census", str(census), "--out", str(out)],
        capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr
    return out.read_text()


def test_a_passing_census_names_the_adopted_pair(tmp_path):
    text = render(tmp_path, adopted="potency_vs_developability")
    assert "**Adopted pair: `potency_vs_developability`**" in text
    assert "The adopted pair `potency_vs_developability`, gate by gate" in text
    assert "NO PAIR PASSED" not in text


def test_an_all_fail_census_never_calls_a_failing_pair_adopted(tmp_path):
    """The misreading this guards against: a reader skimming a heading that says
    'the adopted pair' above a table of failing gates."""
    text = render(tmp_path, adopted=None)
    assert "NO PAIR PASSED" in text
    assert "The adopted pair `" not in text
    assert "all three pairs failed" in text


def test_the_page_states_the_adoption_rule_is_positional(tmp_path):
    """Adoption is the FIRST pair clearing the gate, not the best-scoring one.
    A reader who assumes best-of would misread a rejected pair with better
    numbers as an oversight."""
    text = render(tmp_path, adopted="potency_vs_developability")
    assert "positional" in text
    assert "not the best-scoring" in text


def test_the_page_carries_the_instrument_sizes_and_the_inertness_finding(tmp_path):
    text = render(tmp_path, adopted="potency_vs_developability")
    assert "120 decision states" in text
    assert "586" in text
    assert "81,500" in text
    assert "inert: True" in text


def test_the_withdrawn_statistic_appears_so_it_cannot_be_reintroduced(tmp_path):
    text = render(tmp_path, adopted="potency_vs_developability")
    assert "best candidate reaches pooled p99" in text


def test_held_out_status_is_stated_explicitly(tmp_path):
    """The handoff acceptance gate requires held-out-open status to be explicit,
    not inferred from absence."""
    text = render(tmp_path, adopted="potency_vs_developability")
    assert "Held-out opened:" in text
    assert "NO" in text
