"""Stage B: the 2x2, the two estimands, and the support-tight handling.

The load-bearing tests here are the ones that show the PRIMARY and SECONDARY
pathwise estimands COME APART. The conditional fraction is intuitive but its
denominator is controller-dependent, so an arm that rarely delivers an
acceptable endpoint can post a dramatic-looking fraction on two trajectories.
`test_conditional_fraction_can_be_dramatic_on_a_tiny_denominator` builds
exactly that case and checks the report exposes it.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from compose_v4.experiments.pathwise_arm_names import (
    STAGE_B_ALL,
    STAGE_B_CORRIDOR_ARMS,
    STAGE_B_DESCRIPTIVE_ARM,
    STAGE_B_SPEC,
)
from compose_v4.experiments.pathwise_arms import (
    ArmContext,
    build_stage_b_arm,
    greedy_path,
)
from compose_v4.experiments.pathwise_reversible_families import (
    clogp_corridor,
    state_is_feasible,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/analyse_pathwise_stage_b.py"
LOW, HIGH = clogp_corridor()

INSIDE = "c1ccc(-c2ccccc2)cc1"                 # cLogP 3.354
INSIDE_2 = "Cc1ccc(Cl)cc1"                     # cLogP 2.648
OUTSIDE = "CCCCCCCCc1ccc(-c2ccccc2)cc1"        # cLogP 6.257


def ctx_for(graph: dict, utility: dict, **kw) -> ArmContext:
    base = {
        "successors": lambda k: list(graph.get(k, [])),
        "utility": lambda k: (utility[k], utility[k]),
        "motif_smarts": "",
        "feasible": lambda k: state_is_feasible("B_physchem_corridor", k),
        "rng": random.Random(0),
        "horizon": 2,
    }
    base.update(kw)
    return ArmContext(**base)


# ------------------------------------------------------------------- the 2x2


def test_the_four_causal_arms_are_a_factorial_of_two_booleans():
    """Each arm differs from its partners in exactly one factor."""
    assert len(STAGE_B_CORRIDOR_ARMS) == 4
    for name in STAGE_B_CORRIDOR_ARMS:
        mask, endpoint_only, controller = STAGE_B_SPEC[name]
        assert mask is True, "all four causal arms enforce the constraint somewhere"
        assert controller in ("greedy", "verified")
    # where-enforced varies within controller, controller varies within where
    assert STAGE_B_SPEC["endpoint_greedy"][1] is True
    assert STAGE_B_SPEC["pathwise_greedy"][1] is False
    assert STAGE_B_SPEC["endpoint_greedy"][2] == STAGE_B_SPEC["pathwise_greedy"][2]
    assert STAGE_B_SPEC["endpoint_verified"][2] == STAGE_B_SPEC["pathwise_verified"][2]


def test_descriptive_arm_is_unconstrained_and_not_a_causal_arm():
    assert STAGE_B_DESCRIPTIVE_ARM not in STAGE_B_CORRIDOR_ARMS
    assert STAGE_B_SPEC[STAGE_B_DESCRIPTIVE_ARM][0] is False
    assert set(STAGE_B_ALL) == set(STAGE_B_CORRIDOR_ARMS) | {STAGE_B_DESCRIPTIVE_ARM}


def test_no_fifth_causal_arm():
    assert len(STAGE_B_ALL) == 5 and len(STAGE_B_CORRIDOR_ARMS) == 4


# ------------------------------------------------- endpoint vs pathwise enforcement


def test_endpoint_only_permits_a_forbidden_intermediate_but_lands_in_C():
    """The whole point: enforcement at t=H alone lets the path leave C."""
    graph = {INSIDE: [(OUTSIDE, 1.0)], OUTSIDE: [(INSIDE_2, 1.0), (OUTSIDE, 0.5)]}
    utility = {INSIDE: 0.0, OUTSIDE: 5.0, INSIDE_2: 1.0}
    ctx = ctx_for(graph, utility)
    result = greedy_path(ctx, INSIDE, mask=True, endpoint_only=True)
    assert result["trajectory"] == [INSIDE, OUTSIDE, INSIDE_2]
    assert state_is_feasible("B_physchem_corridor", result["trajectory"][-1])
    assert not state_is_feasible("B_physchem_corridor", result["trajectory"][1])


def test_pathwise_enforcement_refuses_the_same_intermediate():
    graph = {INSIDE: [(OUTSIDE, 1.0)], OUTSIDE: [(INSIDE_2, 1.0)]}
    utility = {INSIDE: 0.0, OUTSIDE: 5.0, INSIDE_2: 1.0}
    ctx = ctx_for(graph, utility)
    result = greedy_path(ctx, INSIDE, mask=True, endpoint_only=False)
    assert result["trajectory"] == [INSIDE]        # no legal move at all
    assert result["dead_end_step"] == 0


def test_endpoint_terminal_infeasible_is_recorded_not_hidden():
    """If the last step offers nothing inside C the arm cannot deliver, and
    that trajectory must leave the conditioning set rather than be counted."""
    graph = {INSIDE: [(OUTSIDE, 1.0)], OUTSIDE: [(OUTSIDE, 1.0)]}
    utility = {INSIDE: 0.0, OUTSIDE: 5.0}
    ctx = ctx_for(graph, utility)
    result = greedy_path(ctx, INSIDE, mask=True, endpoint_only=True)
    assert result["endpoint_terminal_infeasible"] is True


def test_unconstrained_descriptive_arm_ignores_the_corridor_entirely():
    graph = {INSIDE: [(OUTSIDE, 1.0)], OUTSIDE: [(OUTSIDE, 1.0)]}
    utility = {INSIDE: 0.0, OUTSIDE: 5.0}
    ctx = ctx_for(graph, utility)
    result = build_stage_b_arm(STAGE_B_DESCRIPTIVE_ARM, ctx, INSIDE)
    assert result["trajectory"][-1] == OUTSIDE


def test_pathwise_arms_never_commit_a_violation():
    """BUG DETECTOR, definitional -- barred from the results table."""
    graph = {INSIDE: [(OUTSIDE, 0.5), (INSIDE_2, 0.5)],
             INSIDE_2: [(OUTSIDE, 1.0)], OUTSIDE: [(INSIDE, 1.0)]}
    utility = {INSIDE: 0.0, INSIDE_2: 1.0, OUTSIDE: 9.0}
    for name in ("pathwise_greedy", "pathwise_verified"):
        result = build_stage_b_arm(name, ctx_for(graph, utility), INSIDE)
        for key in result["trajectory"]:
            assert state_is_feasible("B_physchem_corridor", key), (name, key)


# ------------------------------------------------------------ shard fixtures


def make_arm(trajectory: list[str], u_p: float, *, dead=None,
             infeasible=False, overrides=0, disagreements=0) -> dict:
    flags = [state_is_feasible("B_physchem_corridor", k) for k in trajectory]
    violated = [i for i, ok in enumerate(flags[1:], start=1) if not ok]
    return {
        "landing": trajectory[-1], "trajectory": trajectory,
        "clogp": [0.0] * len(trajectory), "edits": len(trajectory) - 1,
        "dead_end_step": dead, "completed": dead is None,
        "endpoint_terminal_infeasible": infeasible,
        "feasible_actions_by_step": [
            {"step": i, "candidates": 5, "feasible": 3}
            for i in range(len(trajectory))],
        "terminal_failure_attribution": None if dead is None else "empty_after_mask",
        "overrides": overrides, "top1_disagreements": disagreements,
        "U_P": u_p, "U_P_at_source": 0.0, "potency_gain": u_p,
        "endpoint_in_C": bool(flags[-1]),
        "any_intermediate_violation": bool(violated),
        "intermediate_violation_count": len(violated),
        "first_violation_index": violated[0] if violated else None,
        "excursions": [], "marginal_kernel_calls": 10,
    }


def make_shard(index: int, *, endpoint_greedy_arm, retention=0.5, **arms) -> dict:
    body = {"endpoint_greedy": endpoint_greedy_arm}
    body.update(arms)
    body.setdefault("pathwise_greedy", make_arm([INSIDE, INSIDE_2], 0.5))
    body.setdefault("endpoint_verified", make_arm([INSIDE, OUTSIDE, INSIDE_2], 1.2))
    body.setdefault("pathwise_verified", make_arm([INSIDE, INSIDE_2], 0.7))
    body.setdefault(STAGE_B_DESCRIPTIVE_ARM, make_arm([INSIDE, OUTSIDE], 2.0))
    census = [{"step": i, "candidates": 10, "kept": max(1, int(retention * 10)),
               "retention": retention, "empty_after_mask": False}
              for i in range(4)]
    return {
        "schema": "compose.pathwise.stage_b_source", "status": "SMOKE_HELD_IN",
        "mask_leak": None, "index": index, "source": INSIDE,
        "start_key": INSIDE, "start_clogp": 3.354, "corridor": [LOW, HIGH],
        "horizon": 6, "goal": "P", "arm_run_order": list(STAGE_B_ALL),
        "support_census": census, "median_retention": retention,
        "support_tight": retention < 0.10, "support_tight_threshold": 0.10,
        "mask_empty_states": 0, "mask_empty_fraction_this_source": 0.0,
        "arms": body, "distinct_landings": 2, "kernel_calls": 120,
        "seconds": 1800.0,
    }


def run_analyser(tmp_path: Path, shards: list[dict]) -> dict:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        (shard_dir / f"{shard['index']:03d}.json").write_text(json.dumps(shard))
    out = tmp_path / "b.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--shards", str(shard_dir), "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO), check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


# ------------------------------------------------------------- the estimands


def test_primary_rate_uses_every_eligible_source_as_denominator(tmp_path):
    hidden = make_arm([INSIDE, OUTSIDE, INSIDE_2], 1.0)      # lands in C, path invalid
    clean = make_arm([INSIDE, INSIDE_2, INSIDE_2], 1.0)      # lands in C, path valid
    shards = [make_shard(i, endpoint_greedy_arm=hidden) for i in range(6)]
    shards += [make_shard(i, endpoint_greedy_arm=clean) for i in range(6, 12)]
    report = run_analyser(tmp_path, shards)
    p = report["primary_analysis_ITT"]["pathwise_estimands"]["endpoint_greedy"][
        "PRIMARY_hidden_path_rate"]
    assert p["denominator_all_eligible_sources"] == 12
    assert p["numerator"] == 6
    assert p["estimate"] == 0.5
    assert p["denominator_is_controller_independent"] is True


def test_conditional_fraction_can_be_dramatic_on_a_tiny_denominator(tmp_path):
    """The reason the unconditional rate is primary.

    Ten of twelve sources fail to deliver an acceptable endpoint at all. Of the
    two that do, both took a forbidden path. The conditional fraction reads
    1.0 -- which sounds decisive -- while the honest unconditional rate is
    2/12. The report must expose both, with the denominator attached.
    """
    hidden = make_arm([INSIDE, OUTSIDE, INSIDE_2], 1.0)
    undelivered = make_arm([INSIDE, OUTSIDE], 1.0, dead=1, infeasible=True)
    shards = [make_shard(i, endpoint_greedy_arm=hidden) for i in range(2)]
    shards += [make_shard(i, endpoint_greedy_arm=undelivered) for i in range(2, 12)]
    report = run_analyser(tmp_path, shards)
    body = report["primary_analysis_ITT"]["pathwise_estimands"]["endpoint_greedy"]

    secondary = body["SECONDARY_hidden_path_fraction"]
    assert secondary["estimate"] == 1.0
    assert secondary["denominator_accepted_endpoints"] == 2
    assert secondary["DENOMINATOR_IS_CONTROLLER_DEPENDENT"] is True

    primary = body["PRIMARY_hidden_path_rate"]
    assert primary["denominator_all_eligible_sources"] == 12
    assert primary["estimate"] == pytest.approx(2 / 12, abs=1e-4)
    assert primary["estimate"] < secondary["estimate"]


def test_both_estimands_can_be_zero(tmp_path):
    clean = make_arm([INSIDE, INSIDE_2, INSIDE_2], 1.0)
    report = run_analyser(tmp_path, [make_shard(i, endpoint_greedy_arm=clean)
                                     for i in range(12)])
    body = report["primary_analysis_ITT"]["pathwise_estimands"]["endpoint_greedy"]
    assert body["PRIMARY_hidden_path_rate"]["estimate"] == 0.0
    assert body["SECONDARY_hidden_path_fraction"]["estimate"] == 0.0


def test_greedy_and_verified_estimands_are_reported_separately(tmp_path):
    """Pooling would mix two different denominators."""
    report = run_analyser(tmp_path, [
        make_shard(i, endpoint_greedy_arm=make_arm([INSIDE, OUTSIDE, INSIDE_2], 1.0))
        for i in range(12)])
    estimands = report["primary_analysis_ITT"]["pathwise_estimands"]
    assert set(estimands) == {"endpoint_greedy", "endpoint_verified"}
    for body in estimands.values():
        assert "PRIMARY_hidden_path_rate" in body


# --------------------------------------------------------------- terminal cost


def test_terminal_cost_is_paired_under_controller_parity(tmp_path):
    report = run_analyser(tmp_path, [
        make_shard(i, endpoint_greedy_arm=make_arm([INSIDE, OUTSIDE, INSIDE_2], 1.0))
        for i in range(12)])
    cost = report["primary_analysis_ITT"]["terminal_cost"]
    assert set(cost) == {"delta_G_greedy_parity", "delta_V_verified_parity"}
    # pathwise_greedy U_P 0.5 vs endpoint_greedy 1.0 -> the mask costs potency
    assert cost["delta_G_greedy_parity"]["summary"]["mean"] == pytest.approx(-0.5)
    assert cost["delta_G_greedy_parity"]["sign_is_free"] is True


def test_a_terminal_cost_is_not_reported_as_a_failure(tmp_path):
    report = run_analyser(tmp_path, [
        make_shard(i, endpoint_greedy_arm=make_arm([INSIDE, OUTSIDE, INSIDE_2], 1.0))
        for i in range(12)])
    interp = report["primary_analysis_ITT"]["terminal_cost_interpretation"]
    assert interp["no_expectation_that_pathwise_beats_endpoint_only"] is True
    assert interp["none_of_these_is_a_failure"] is True
    assert len(interp["declared_informative_in_advance"]) == 3


def test_future_aware_carries_the_guaranteed_sign_warning(tmp_path):
    report = run_analyser(tmp_path, [
        make_shard(i, endpoint_greedy_arm=make_arm([INSIDE, OUTSIDE, INSIDE_2], 1.0))
        for i in range(12)])
    fa = report["primary_analysis_ITT"]["future_aware"]
    assert "policy improvement" in fa["GUARANTEED_SIGN_WARNING"]
    assert "No sign test is computed" in fa["GUARANTEED_SIGN_WARNING"]
    # ceiling reporting when greedy never fails
    assert fa["binary_headroom"]["verdict"] == "CEILING_NOT_A_NULL"
    assert "p_value" not in json.dumps(fa)


# ------------------------------------------------------------- support-tight


def test_support_tight_sources_stay_in_the_primary_ITT(tmp_path):
    shards = [make_shard(i, endpoint_greedy_arm=make_arm(
        [INSIDE, OUTSIDE, INSIDE_2], 1.0), retention=0.02) for i in range(3)]
    shards += [make_shard(i, endpoint_greedy_arm=make_arm(
        [INSIDE, OUTSIDE, INSIDE_2], 1.0), retention=0.6) for i in range(3, 12)]
    report = run_analyser(tmp_path, shards)
    assert report["support_tight"]["sources_tight"] == 3
    assert report["support_tight"]["fraction_tight"] == pytest.approx(0.25)
    assert report["primary_analysis_ITT"]["sources"] == 12       # ITT keeps all
    assert report["sensitivity_excluding_support_tight"]["sources"] == 9


def test_support_tight_threshold_is_frozen_at_the_viability_number(tmp_path):
    report = run_analyser(tmp_path, [
        make_shard(i, endpoint_greedy_arm=make_arm([INSIDE, OUTSIDE, INSIDE_2], 1.0))
        for i in range(12)])
    assert report["support_tight"]["threshold"] == 0.10
    assert report["support_tight"]["predeclared"] is True
    assert "DESCRIPTIVE" in report["support_tight"]["measured_on"]


def test_mask_empty_is_reported_per_source_not_only_pooled(tmp_path):
    report = run_analyser(tmp_path, [
        make_shard(i, endpoint_greedy_arm=make_arm([INSIDE, OUTSIDE, INSIDE_2], 1.0))
        for i in range(12)])
    per_source = report["mask_empty_by_source"]
    assert len(per_source) == 12
    assert all("fraction" in v for v in per_source.values())


def test_terminal_failure_attribution_is_reported(tmp_path):
    failed = make_arm([INSIDE, INSIDE_2], 0.3, dead=1)
    report = run_analyser(tmp_path, [
        make_shard(i, endpoint_greedy_arm=make_arm([INSIDE, OUTSIDE, INSIDE_2], 1.0),
                   pathwise_greedy=failed) for i in range(12)])
    attribution = report["primary_analysis_ITT"]["terminal_failure_attribution"]
    assert attribution["pathwise_greedy"] == {"empty_after_mask": 12}


def test_mask_leak_voids_the_run(tmp_path):
    shard = make_shard(0, endpoint_greedy_arm=make_arm([INSIDE, OUTSIDE, INSIDE_2], 1.0))
    shard["mask_leak"] = {"pathwise_greedy": 2}
    report = run_analyser(tmp_path, [shard])
    assert report["status"] == "INVALID_INSTRUMENT"
    assert report["mask_integrity"]["verdict"] == "FAIL"
    assert report["mask_integrity"]["kind"] == "BUG_DETECTOR_NOT_A_FINDING"


# ----------------------------------------------------------------- the panel


@pytest.mark.skipif(not (REPO / "diagnostics/pathwise_stage_b_panel.json").exists(),
                    reason="stage B panel not built")
def test_stage_b_panel_is_disjoint_and_starts_inside_the_corridor():
    panel = json.loads((REPO / "diagnostics/pathwise_stage_b_panel.json").read_text())
    keys = {r["source"] for r in panel["sources"]}
    assert len(keys) == len(panel["sources"]) == 24
    for name in ("pathwise_constraints_smoke_panel", "pathwise_a2_panel",
                 "retarget_calibration_cohort"):
        other = REPO / f"diagnostics/{name}.json"
        if other.exists():
            assert not (keys & {r["source"]
                                for r in json.loads(other.read_text())["sources"]})
    for row in panel["sources"]:
        assert state_is_feasible("B_physchem_corridor", row["source"])


def test_stage_b_eligibility_is_excursion_blind():
    """Selecting on excursion propensity would make the primary estimand true
    by construction. Pin the signature so it cannot be reintroduced quietly."""
    import inspect

    sys.path.insert(0, str(REPO / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_sel", REPO / "scripts/pathwise_select_stage_b_panel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params = set(inspect.signature(module.eligibility).parameters)
    assert params == {"smiles", "clogp", "heavy", "potency_margin", "low", "high"}
