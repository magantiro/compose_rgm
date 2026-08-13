"""The protocol config and the code must not drift apart.

Written after a hand-transcribed constant in this lane turned out to be wrong:
the developability clip ceiling was typed as 1.32669 where the computed value is
1.3267132048600137. Nobody would have noticed, and a gate threshold reads it.

A protocol document that disagrees with the code it describes is worse than no
document, because it is trusted. These tests make the disagreement fail a test
run instead of surviving into a handoff.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

CONFIG = json.loads((REPO / "configs/pareto_control_protocol_v1.json").read_text())


def _census_module():
    import importlib
    return importlib.import_module("pareto_tradeoff_census")


def test_objective_constants_match_the_census_code():
    census = _census_module()
    assert CONFIG["objectives"]["P"]["centre"] == census.CENTRE_P
    assert CONFIG["objectives"]["P"]["scale_iqr"] == census.S_P
    assert CONFIG["objectives"]["D"]["qed_floor"] == census.QED_FLOOR
    assert CONFIG["objectives"]["D"]["qed_scale_iqr"] == census.S_QED
    assert CONFIG["objectives"]["D"]["logp_box"] == list(census.LOGP_BOX)
    assert CONFIG["objectives"]["D"]["logp_scale_iqr"] == census.S_LOGP
    assert CONFIG["objectives"]["D"]["margin_clip"] == census.MARGIN_CLIP
    assert CONFIG["objectives"]["D"]["softmin_tau"] == census.SOFTMIN_TAU


def test_the_developability_clip_ceiling_is_computed_not_typed():
    """clip - tau*log 2, the value of the soft-min when both margins are clipped.

    This is the exact assertion that would have caught the 1.32669 typo.
    """
    clip = CONFIG["objectives"]["D"]["margin_clip"]
    tau = CONFIG["objectives"]["D"]["softmin_tau"]
    exact = float(-tau * np.log(2 * np.exp(-clip / tau)))
    assert CONFIG["objectives"]["D"]["analytic_ceiling"] == pytest.approx(exact, abs=1e-12)
    assert _census_module().Z_D_CEILING == pytest.approx(exact, abs=1e-12)


def test_gate_thresholds_match_the_census_code():
    census = _census_module()
    gate = CONFIG["headroom_gate"]
    assert gate["G1_max_spearman_rho"] == census.G1_MAX_RHO
    assert gate["G2_min_tradeoff_move_fraction"] == census.G2_MIN_TRADEOFF_FRACTION
    assert gate["G2_min_each_direction"] == census.G2_MIN_EACH_DIRECTION
    assert gate["G2_min_state_availability"] == census.G2_MIN_STATE_AVAILABILITY
    assert gate["G3_max_binding_share"] == census.G3_MAX_BINDING_SHARE
    assert gate["G4_max_reach_fraction"] == census.G4_MAX_REACH_FRACTION
    assert gate["G5_min_distinct_selections"] == census.G5_MIN_DISTINCT_SELECTIONS
    assert gate["G5_max_unanimous_fraction"] == census.G5_MAX_UNANIMOUS_FRACTION


def test_preference_grid_and_budget_match_everywhere():
    census = _census_module()
    from compose_v4.experiments import pareto_control

    assert CONFIG["preference_grid"] == list(census.PREFERENCE_GRID)
    assert CONFIG["budget_edits"] == census.BUDGET
    assert CONFIG["scalarization"]["rho"] == pareto_control.CHEBYSHEV_RHO


def test_pair_order_matches_the_census_code_and_is_not_reordered():
    """The adoption rule is positional, so the ORDER is load-bearing, not just
    the membership."""
    census = _census_module()
    assert CONFIG["predeclared_pair_order"] == [p[0] for p in census.PAIR_ORDER]


def test_config_contrasts_match_the_instrument_gates_declarations():
    import pareto_instrument_gate as gate

    declared = {c.name.split("_")[0]: c for c in gate.LANE_CONTRASTS}
    for row in CONFIG["contrasts"]:
        contrast = declared[row["id"]]
        assert contrast.arm == row["arm"], row["id"]
        assert contrast.base == row["base"], row["id"]
        assert contrast.status == row["status"], row["id"]
        assert contrast.budget_parity_axis == row.get("budget_parity_axis"), row["id"]


def test_every_withdrawn_statistic_is_registered_in_the_gate():
    """A statistic withdrawn in the document but not in the gate can be
    reintroduced by the next person who reads only the code."""
    import pareto_instrument_gate as gate

    assert set(CONFIG["withdrawn_statistics"]) == set(gate.WITHDRAWN_STATISTICS)


def test_p1_is_context_only_in_both_the_config_and_the_gate():
    """The confound the gate caught must stay caught."""
    import pareto_instrument_gate as gate

    row = next(c for c in CONFIG["contrasts"] if c["id"] == "P1")
    assert row["status"] == "CONTEXT_ONLY"
    assert sorted(row["varies"]) == ["controller", "objective"]
    contrast = next(c for c in gate.LANE_CONTRASTS if c.name.startswith("P1"))
    assert contrast.status == "CONTEXT_ONLY"


def test_the_config_declares_no_held_out_access():
    assert CONFIG["held_out_opened"] is False
    assert "FROZEN" in CONFIG["r_theta"]
