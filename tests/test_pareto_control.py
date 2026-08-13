"""Local tests for target-free Pareto / preference control.

Every test runs on a TOY state graph with hand-computable optima: no RDKit, no
rewrite kernel, no frozen R_theta checkpoint. That is deliberate. Two of the
five instrument defects already found in this project were control-logic bugs --
"a verified arm that was secretly greedy" and "an action selected by argmax V_G
then scored by V_G" -- and both would have been caught in seconds by a toy graph
whose right answer is known in advance. A test suite that needs the real kernel
is a test suite that does not get run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compose_v4.experiments.pareto_control import (  # noqa: E402
    CostLedger,
    branch_from_common_prefix,
    ReferenceFrontError,
    GUARANTEED_SIGN_REGISTRY,
    GuaranteedSign,
    is_sign_guaranteed,
    preference_ordering,
    INTERNAL_METHOD_UNIVERSE,
    MethodUniverseError,
    budget_to_ninety,
    check_method_universe,
    pooled_attainable_hypervolume,
    summarize_b90,
    preference_region_coverage,
    union_reference_front,
    MeteredProcess,
    Scalarization,
    WeightedSum,
    endpoint_diversity,
    generate_then_rank,
    greedy_preference_run,
    hypervolume,
    normalized_hypervolume,
    pareto_front_indices,
    preference_coverage,
    source_paired_bootstrap,
    trajectories_for_budget,
    trajectories_for_kernel_budget,
    unguided_run,
    verified_preference_run,
)

UTOPIA = np.array([10.0, 10.0])
REFERENCE = np.array([0.0, 0.0])


# ---------------------------------------------------------------------------
# Toy process: a two-step trap where the locally best action destroys the future
# ---------------------------------------------------------------------------

TRAP_EDGES = {
    "root": [("A", 0.5), ("B", 0.5)],
    #  A looks better right now ...
    "A": [("A1", 0.5), ("A2", 0.5)],
    #  ... but every continuation from A is poor, while B's are excellent.
    "B": [("B1", 0.9), ("B2", 0.1)],
}
TRAP_Z = {
    "root": [5.0, 5.0],
    "A": [8.0, 8.0], "A1": [8.0, 1.0], "A2": [1.0, 8.0],
    "B": [6.0, 6.0], "B1": [9.5, 9.5], "B2": [2.0, 2.0],
}


class ToyProcess:
    def __init__(self, edges: dict[str, list[tuple[str, float]]]) -> None:
        self.edges = edges

    def successors(self, key: str) -> list[tuple[str, float]]:
        return list(self.edges.get(key, []))


class ToyObjectives:
    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    def z(self, key: str) -> np.ndarray:
        return np.array(self.table[key], dtype=float)


def trap() -> MeteredProcess:
    return MeteredProcess(ToyProcess(TRAP_EDGES), ToyObjectives(TRAP_Z))


SCALARIZE = Scalarization(UTOPIA)


# ---------------------------------------------------------------------------
# Scalarization
# ---------------------------------------------------------------------------

def test_chebyshev_matches_the_written_formula():
    z = np.array([[8.0, 2.0]])
    got = SCALARIZE(z, 0.5)[0]
    expected = max(0.5 * 2.0, 0.5 * 8.0) + 1e-3 * (2.0 + 8.0)
    assert got == pytest.approx(expected)


def test_chebyshev_selects_a_point_no_weighted_sum_can_ever_select():
    """The reason PROTOCOL forbids scalarizing by weighted sum alone.

    (4, 4) is nondominated but sits strictly inside the convex hull of (10, 0)
    and (0, 10), so max(10w, 10(1-w)) >= 5 > 4 for EVERY w: no weighted sum ever
    picks it. Chebyshev does. An experiment run only on weighted sums would
    report a smaller achievable front than actually exists.
    """
    z = np.array([[10.0, 0.0], [4.0, 4.0], [0.0, 10.0]])
    keys = ["hi_a", "middle", "hi_b"]

    ws = WeightedSum()
    picked_by_sum = {keys[int(np.argmin(ws(z, w)))] for w in np.linspace(0, 1, 201)}
    assert "middle" not in picked_by_sum

    picked_by_cheb = {keys[int(np.argmin(SCALARIZE(z, w)))]
                      for w in np.linspace(0, 1, 201)}
    assert "middle" in picked_by_cheb


def test_binding_index_identifies_which_objective_is_binding():
    z = np.array([[9.0, 2.0]])
    assert int(SCALARIZE.binding_index(z, 0.5)[0]) == 1
    # Weighting objective 0 hard enough makes its (small) gap binding instead.
    assert int(SCALARIZE.binding_index(z, 0.99)[0]) == 0


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def test_greedy_takes_the_locally_best_action_and_walks_into_the_trap():
    run = greedy_preference_run(trap(), "root", 2, 0.5, SCALARIZE)
    assert run.states[1] == "A", "greedy must prefer A, which scores better now"
    assert run.endpoint in {"A1", "A2"}
    assert run.complete


def test_verified_sacrifices_the_locally_best_action_and_lands_better():
    run = verified_preference_run(trap(), "root", 2, 0.5, SCALARIZE, seed=0)
    assert run.states[1] == "B", "remaining-budget lookahead must reject A"
    assert run.endpoint == "B1"
    assert run.overrode_greedy >= 1


def test_verified_and_greedy_are_genuinely_different_arms():
    """Instrument gate D2, as a unit test.

    A 'verified' arm whose committed action sequence equals greedy's on every
    source is greedy wearing a different label. That exact defect voided two
    runs in the main lane and made a headroom of 0 definitional.
    """
    greedy = greedy_preference_run(trap(), "root", 2, 0.5, SCALARIZE)
    verified = verified_preference_run(trap(), "root", 2, 0.5, SCALARIZE, seed=0)
    assert greedy.actions != verified.actions


def test_verified_is_never_worse_than_greedy_which_is_a_theorem_not_a_result():
    """Policy improvement: greedy's action is always in the shortlist, and the
    strict-improvement rule never commits a worse landing. So `verified <=
    greedy` in scalarized loss holds BY CONSTRUCTION.

    This test asserts the guarantee precisely so that no downstream report may
    present it as evidence. A sign test against a null of 0.5 here would be
    testing something already known to be true -- defect 3 of the five.
    """
    for weight in (0.1, 0.3, 0.5, 0.7, 0.9):
        greedy = greedy_preference_run(trap(), "root", 2, weight, SCALARIZE)
        verified = verified_preference_run(trap(), "root", 2, weight, SCALARIZE, seed=0)
        gs = float(SCALARIZE(greedy.endpoint_z, weight)[0])
        vs = float(SCALARIZE(verified.endpoint_z, weight)[0])
        assert vs <= gs + 1e-9


def test_greedy_shortlist_always_contains_greedys_own_action():
    """Without this the strict-improvement rule cannot be a superset of greedy
    and the guarantee above would silently fail."""
    run = verified_preference_run(trap(), "root", 2, 0.5, SCALARIZE, seed=0,
                                  top_immediate=1, top_reference=0, n_random=0)
    assert run.decisions[0]["candidates"] >= 1


def test_unguided_is_reproducible_and_preference_blind():
    a = unguided_run(trap(), "root", 2, seed=7)
    b = unguided_run(trap(), "root", 2, seed=7)
    assert a.states == b.states
    assert np.isnan(a.weight), "the unguided arm must not carry a preference"


def test_unguided_follows_the_reference_law_not_the_objective():
    """B1 has probability 0.9 under the toy reference law and B2 has 0.1, so a
    reference-law sampler must visit B1 far more often. If the arm were secretly
    scoring by objective it would always take B1 (or always A)."""
    counts = {"B1": 0, "B2": 0}
    for seed in range(400):
        run = unguided_run(MeteredProcess(ToyProcess({"B": TRAP_EDGES["B"]}),
                                          ToyObjectives(TRAP_Z)), "B", 1, seed=seed)
        counts[run.endpoint] += 1
    assert 0.80 < counts["B1"] / 400 < 0.98


def test_generate_then_rank_returns_one_endpoint_per_preference():
    metered = trap()
    preferences = (0.1, 0.3, 0.5, 0.7, 0.9)
    out = generate_then_rank(metered, "root", 2, preferences, SCALARIZE,
                             n_trajectories=8, seed=3)
    assert len(out) == len(preferences)
    assert [t.weight for t in out] == list(preferences)


def test_generate_then_rank_budget_parity_is_computed_not_guessed():
    """P3/P4 must vary CONTROLLER only. If gen_rank's trajectory count were a
    free parameter the contrast would vary controller AND budget -- the parity
    defect the main lane's audit caught, which inflated its effect ~11%."""
    assert trajectories_for_budget(target_native_calls=70, budget=6,
                                   mean_fiber_width=500.0) == 10
    assert trajectories_for_budget(target_native_calls=1, budget=6,
                                   mean_fiber_width=500.0) == 1


def test_the_two_budget_axes_disagree_by_orders_of_magnitude():
    """Why P3/P4 must be reported as a bracket rather than a single number.

    One kernel call yields ~600 candidates on the real process, so matching
    generate-then-rank on native oracle calls hands it vastly more search than
    matching it on kernel calls. Neither is "the fair one", and quoting either
    alone is a reporting choice that decides the winner.
    """
    native_matched = trajectories_for_budget(
        target_native_calls=26 * 600, budget=6, mean_fiber_width=600.0)
    kernel_matched = trajectories_for_kernel_budget(
        target_kernel_calls=26, budget=6)
    assert native_matched / kernel_matched > 100


# ---------------------------------------------------------------------------
# Cost ledger
# ---------------------------------------------------------------------------

def test_native_and_raw_oracle_calls_differ_and_both_are_counted():
    """The two conventions must never be substituted for one another. Raw counts
    every scoring invocation; native counts distinct molecules. Their difference
    IS the caching effect, and it is a reported quantity."""
    metered = trap()
    verified_preference_run(metered, "root", 2, 0.5, SCALARIZE, seed=0)
    led = metered.ledger
    assert led.raw_oracle_calls > led.native_oracle_calls
    # Six of the seven toy states are scored. `root` is not: an arm scores
    # CANDIDATES, never the state it is already standing on, so the start state
    # never enters the oracle count.
    assert led.native_oracle_calls == 6
    assert "root" not in metered._z
    assert led.kernel_calls == 3  # root, A, B enumerated once each


def test_kernel_calls_count_distinct_states_only():
    metered = trap()
    metered.successors("root")
    metered.successors("root")
    assert metered.ledger.kernel_calls == 1


def test_cost_ledgers_add():
    total = CostLedger(1, 2, 3) + CostLedger(10, 20, 30)
    assert total.as_dict() == {"native_oracle_calls": 11, "raw_oracle_calls": 22,
                               "kernel_calls": 33}


def test_verified_costs_more_kernel_calls_than_greedy():
    """Not a defect -- it is why P2 must be run at matched oracle budget and why
    the kernel-call ratio is reported alongside."""
    g, v = trap(), trap()
    greedy_preference_run(g, "root", 2, 0.5, SCALARIZE)
    verified_preference_run(v, "root", 2, 0.5, SCALARIZE, seed=0)
    assert v.ledger.kernel_calls >= g.ledger.kernel_calls


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_pareto_front_drops_dominated_points():
    z = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0]])
    assert sorted(pareto_front_indices(z).tolist()) == [1, 2]


def test_hypervolume_of_a_single_point_is_the_rectangle():
    assert hypervolume(np.array([[3.0, 4.0]]), REFERENCE) == pytest.approx(12.0)


def test_hypervolume_of_two_points_is_the_staircase():
    # (4,1) contributes 4*1; (1,4) adds 1*(4-1) on top.
    z = np.array([[4.0, 1.0], [1.0, 4.0]])
    assert hypervolume(z, REFERENCE) == pytest.approx(4.0 + 3.0)


def test_hypervolume_ignores_points_below_the_reference():
    z = np.array([[3.0, 4.0], [-1.0, 100.0]])
    assert hypervolume(z, REFERENCE) == pytest.approx(12.0)


def test_normalized_hypervolume_is_one_at_the_utopia_corner():
    z = np.array([[10.0, 10.0]])
    assert normalized_hypervolume(z, REFERENCE, UTOPIA) == pytest.approx(1.0)


def test_normalized_hypervolume_may_EXCEED_one_and_is_not_clipped():
    """The box is a NORMALIZER, not a cap.

    `z*` is the held-in p99, not a maximum, and controlled trajectories do pass
    it -- the census's G4 potency reach of 0.700 says so. Clipping at `z*` would
    suppress real achievement and make ceiling effects invisible.

    An earlier docstring claimed HV lands in [0, 1]. That claim was false, and
    the first completed smoke source returned 1.0074. The METRIC is unchanged;
    only the false description was.
    """
    beyond = np.array([[12.0, 11.0]])
    assert normalized_hypervolume(beyond, REFERENCE, UTOPIA) > 1.0


def test_ratios_to_hv_star_are_invariant_to_the_box_constant():
    """Why the false bound did not corrupt B_90 or N_90: HV_star carries the
    same normalizer, so the ratio cancels it."""
    fronts = {"a": np.array([[12.0, 2.0]]), "b": np.array([[2.0, 11.0]])}
    z = np.array([[12.0, 2.0]])
    wide = np.array([20.0, 20.0])
    r1 = (normalized_hypervolume(z, REFERENCE, UTOPIA)
          / pooled_attainable_hypervolume(fronts, REFERENCE, UTOPIA))
    r2 = (normalized_hypervolume(z, REFERENCE, wide)
          / pooled_attainable_hypervolume(fronts, REFERENCE, wide))
    assert r1 == pytest.approx(r2)


def test_hypervolume_cannot_be_inflated_by_adding_dominated_points():
    """The inflation channel, closed. Adding molecules that are dominated does
    not raise HV -- which is why HV is computed over committed endpoints only,
    one per preference, so arms cannot differ in point count either."""
    base = np.array([[4.0, 4.0]])
    padded = np.vstack([base, np.array([[1.0, 1.0], [2.0, 3.0], [3.0, 2.0]])])
    assert hypervolume(padded, REFERENCE) == pytest.approx(hypervolume(base, REFERENCE))


def test_preference_coverage_is_minimal_when_all_endpoints_collapse():
    z = np.repeat(np.array([[5.0, 5.0]]), 5, axis=0)
    keys = [f"same{i}" for i in range(5)]
    cov = preference_coverage(z, (0.1, 0.3, 0.5, 0.7, 0.9), SCALARIZE, keys)
    assert cov == pytest.approx(0.0)


def test_preference_coverage_rewards_distinct_preference_matched_endpoints():
    """`z[i]` is the endpoint produced by `preferences[i]`. A higher weight on
    objective 0 penalizes objective 0's gap harder and so pulls toward higher
    objective-0 value, which is why the endpoints run in INCREASING objective 0
    as the weight rises."""
    z = np.array([[1.0, 9.5], [4.0, 8.0], [6.0, 6.0], [8.0, 4.0], [9.5, 1.0]])
    keys = ["e0", "e1", "e2", "e3", "e4"]
    cov = preference_coverage(z, (0.1, 0.3, 0.5, 0.7, 0.9), SCALARIZE, keys)
    assert cov > 0.5


def test_endpoint_diversity_is_zero_for_identical_endpoints():
    assert endpoint_diversity(np.ones((4, 4))) == pytest.approx(0.0)


def test_endpoint_diversity_rises_as_endpoints_separate():
    sim = np.array([[1.0, 0.2], [0.2, 1.0]])
    assert endpoint_diversity(sim) == pytest.approx(0.8)


def test_bootstrap_resamples_sources_not_branches():
    """The source is the independent unit. Passing 5 sources must give n=5 even
    though each source carries 5 preference branches; resampling branches would
    understate the interval fivefold."""
    out = source_paired_bootstrap([0.1, 0.2, 0.3, -0.1, 0.05])
    assert out["n"] == 5
    assert out["ci95_low"] <= out["mean"] <= out["ci95_high"]
    assert out["wins"] == 4 and out["losses"] == 1


def test_bootstrap_reports_losses_so_a_guaranteed_sign_is_visible():
    """If an arm's advantage is guaranteed, `losses` is 0 and the reader can see
    that the statistic had no falsifying range. The function never hides it."""
    out = source_paired_bootstrap([0.1, 0.2, 0.3])
    assert out["losses"] == 0


# ---------------------------------------------------------------------------
# Preference responsiveness -- contrast P5 in miniature
# ---------------------------------------------------------------------------

FAN_EDGES = {"root": [(f"e{i}", 0.2) for i in range(5)]}
FAN_Z = {"root": [5.0, 5.0], "e0": [9.5, 1.0], "e1": [8.0, 4.0],
         "e2": [6.0, 6.0], "e3": [4.0, 8.0], "e4": [1.0, 9.5]}


def test_different_preferences_choose_different_endpoints():
    """The claim, in miniature: one state, five preferences, distinct futures.
    If this collapsed to one endpoint the Pareto claim would be empty -- which
    is exactly what census gate G5 measures on real fibers."""
    picks = set()
    for weight in (0.1, 0.3, 0.5, 0.7, 0.9):
        metered = MeteredProcess(ToyProcess(FAN_EDGES), ToyObjectives(FAN_Z))
        picks.add(greedy_preference_run(metered, "root", 1, weight, SCALARIZE).endpoint)
    assert len(picks) >= 3


def test_preference_ordering_is_monotone_in_the_weight():
    """Raising the weight on objective 0 must never move the choice toward lower
    objective-0 value. A controller that ignored the preference would fail this,
    and so would a sign error in the scalarization."""
    values = []
    for weight in (0.1, 0.3, 0.5, 0.7, 0.9):
        metered = MeteredProcess(ToyProcess(FAN_EDGES), ToyObjectives(FAN_Z))
        run = greedy_preference_run(metered, "root", 1, weight, SCALARIZE)
        values.append(FAN_Z[run.endpoint][0])
    assert values == sorted(values)


def test_dead_end_states_are_reported_incomplete_not_silently_truncated():
    metered = MeteredProcess(ToyProcess({"root": [("leaf", 1.0)]}),
                             ToyObjectives({"root": [5.0, 5.0], "leaf": [6.0, 6.0]}))
    run = greedy_preference_run(metered, "root", 4, 0.5, SCALARIZE)
    assert run.endpoint == "leaf"
    assert run.complete is False


def test_argmin_ties_break_on_key_not_enumeration_order():
    """Determinism: a rerun must commit the same trajectory, so ties cannot
    depend on the order the kernel happened to emit successors in."""
    edges = {"root": [("zzz", 0.5), ("aaa", 0.5)]}
    table = {"root": [5.0, 5.0], "zzz": [7.0, 7.0], "aaa": [7.0, 7.0]}
    forward = greedy_preference_run(
        MeteredProcess(ToyProcess(edges), ToyObjectives(table)), "root", 1, 0.5, SCALARIZE)
    reversed_edges = {"root": list(reversed(edges["root"]))}
    backward = greedy_preference_run(
        MeteredProcess(ToyProcess(reversed_edges), ToyObjectives(table)), "root", 1,
        0.5, SCALARIZE)
    assert forward.endpoint == backward.endpoint == "aaa"


# ---------------------------------------------------------------------------
# Contrast P6 -- one realized history, many preference-dependent futures
# ---------------------------------------------------------------------------

PREFIX_EDGES = {
    "root": [("p1", 1.0)], "p1": [("p2", 1.0)], "p2": [("x3", 1.0)],
    "x3": [(f"e{i}", 0.2) for i in range(5)],
}
PREFIX_Z = {"root": [5.0, 5.0], "p1": [5.0, 5.0], "p2": [5.0, 5.0],
            "x3": [5.0, 5.0], "e0": [1.0, 9.5], "e1": [4.0, 8.0],
            "e2": [6.0, 6.0], "e3": [8.0, 4.0], "e4": [9.5, 1.0]}


def test_all_branches_share_the_identical_prefix():
    """The figure's whole statement is ONE history, many futures. If the
    branches diverged before the branch point it would be five trajectories,
    which is a different and much weaker claim."""
    metered = MeteredProcess(ToyProcess(PREFIX_EDGES), ToyObjectives(PREFIX_Z))
    out = branch_from_common_prefix(metered, "root", 3, 1,
                                    (0.1, 0.3, 0.5, 0.7, 0.9), SCALARIZE)
    assert out["branch_point"] == "x3"
    assert out["prefix"].states == ["root", "p1", "p2", "x3"]
    for run in out["branches"].values():
        assert run.states[0] == "x3"


def test_branches_from_the_identical_state_still_diverge_by_preference():
    metered = MeteredProcess(ToyProcess(PREFIX_EDGES), ToyObjectives(PREFIX_Z))
    out = branch_from_common_prefix(metered, "root", 3, 1,
                                    (0.1, 0.3, 0.5, 0.7, 0.9), SCALARIZE)
    endpoints = {run.endpoint for run in out["branches"].values()}
    assert len(endpoints) >= 3


def test_prefix_is_committed_before_any_branch_preference_is_used():
    """The prefix must be reachable under its own weight alone. If a branch
    preference leaked into prefix construction the prefix would differ between
    branches -- which this asserts it does not."""
    a = branch_from_common_prefix(
        MeteredProcess(ToyProcess(PREFIX_EDGES), ToyObjectives(PREFIX_Z)),
        "root", 3, 1, (0.1,), SCALARIZE)
    b = branch_from_common_prefix(
        MeteredProcess(ToyProcess(PREFIX_EDGES), ToyObjectives(PREFIX_Z)),
        "root", 3, 1, (0.9,), SCALARIZE)
    assert a["prefix"].states == b["prefix"].states


# ---------------------------------------------------------------------------
# HV_ref must not come from COMPOSE, and efficiency is the reported axis
#
# Frozen 2026-08-13, before any hypervolume number existed.
# ---------------------------------------------------------------------------

def test_hv_star_refuses_to_be_derived_from_a_single_arm():
    """The structural-favorability hazard, as an executable check.

    If the denominator of "reached 90% of the attainable front" were COMPOSE's
    own best front, the statement would be partly about COMPOSE's own ceiling
    and every efficiency curve would inherit the bias -- a metric that cannot
    fully disappoint. It raises rather than warns, because a warning would
    eventually be ignored.
    """
    only_compose = {"verified_pref": np.array([[9.0, 1.0], [1.0, 9.0]])}
    with pytest.raises(ReferenceFrontError):
        union_reference_front(only_compose)
    with pytest.raises(ReferenceFrontError):
        pooled_attainable_hypervolume(only_compose, REFERENCE, UTOPIA)


def test_hv_star_is_the_pooled_union_not_the_best_single_method():
    """The union front must dominate every contributor, so no arm can move the
    goalposts toward itself by being the only one measured."""
    fronts = {
        "greedy_pref": np.array([[9.0, 1.0], [5.0, 5.0]]),
        "hn_gfn": np.array([[1.0, 9.0], [4.0, 6.0]]),
    }
    union = union_reference_front(fronts)
    hv_union = hypervolume(union, REFERENCE)
    for name, z in fronts.items():
        assert hypervolume(z, REFERENCE) <= hv_union + 1e-9, name
    # (9,1) and (1,9) are both on the union front; neither method alone has both.
    assert any(np.allclose(row, [9.0, 1.0]) for row in union)
    assert any(np.allclose(row, [1.0, 9.0]) for row in union)


def test_hv_star_grows_when_an_external_method_extends_the_front():
    """An external method reaching further must RAISE the bar, not lower it."""
    base = {"greedy_pref": np.array([[5.0, 5.0]]),
            "unguided": np.array([[4.0, 4.0]])}
    extended = dict(base, hn_gfn=np.array([[9.0, 1.0], [1.0, 9.0]]))
    assert (pooled_attainable_hypervolume(extended, REFERENCE, UTOPIA)
            > pooled_attainable_hypervolume(base, REFERENCE, UTOPIA))


def test_the_frozen_nadir_is_a_parameter_not_read_off_the_results():
    """The reference corner comes from the held-in scales. Passing a different
    nadir must change HV, which is what makes it a frozen input rather than
    something the results can quietly set."""
    z = np.array([[5.0, 5.0]])
    assert hypervolume(z, np.array([0.0, 0.0])) == pytest.approx(25.0)
    assert hypervolume(z, np.array([4.0, 4.0])) == pytest.approx(1.0)


def test_b90_censors_rather_than_substituting_b_max():
    """THE substitution that must never happen.

    Imputing B_max makes a method that never got there look like one that got
    there at the last moment -- a failure rendered as a success. It is easier to
    commit here than anywhere else because the substitution looks like tidiness.
    """
    out = budget_to_ninety([0.1, 0.2, 0.3], [1, 2, 3], hv_star=1.0)
    assert out["censored"] is True
    assert out["value"] is None
    assert out["reported_as"] == "B_90 > B_max"
    assert out["value"] != out["b_max"]


def test_b90_is_the_first_budget_reaching_ninety_percent_of_pooled_attainable_hv():
    out = budget_to_ninety([0.1, 0.5, 0.91, 0.95], [10, 20, 30, 40], hv_star=1.0)
    assert out["censored"] is False and out["value"] == 30.0


def test_b90_uses_best_so_far_so_a_dip_cannot_delay_it():
    out = budget_to_ninety([0.95, 0.10, 0.10], [5, 10, 15], hv_star=1.0)
    assert out["censored"] is False and out["value"] == 5.0


def test_b90_reports_how_close_a_censored_method_got():
    """A censored method that reached 89% and one that reached 5% are different
    facts, and the summary must not flatten them into 'did not reach'."""
    out = budget_to_ninety([0.1, 0.89], [1, 2], hv_star=1.0)
    assert out["censored"] is True
    assert out["best_fraction_of_hv_star"] == pytest.approx(0.89)


def test_censoring_survives_aggregation_across_sources():
    """The same substitution one level up: a median that quietly pooled censored
    sources at B_max would hide exactly what censoring exists to show."""
    results = [budget_to_ninety([0.95], [1], hv_star=1.0),
               budget_to_ninety([0.95], [1], hv_star=1.0),
               budget_to_ninety([0.10], [1], hv_star=1.0)]
    summary = summarize_b90(results)
    assert summary["n_uncensored"] == 2
    assert summary["n_censored"] == 1
    assert summary["censoring_rate"] == pytest.approx(1 / 3)
    assert summary["median_uncensored"] == 1.0


def test_hv_star_is_method_symmetric_a_strong_method_raises_its_own_bar():
    """The whole point of the pooled rule: a method that expands the frontier
    raises the threshold for everyone INCLUDING ITSELF, so no one can define the
    ceiling from their own performance."""
    without = {"greedy_pref": np.array([[5.0, 5.0]]),
               "unguided": np.array([[4.0, 4.0]])}
    with_strong = dict(without, verified_pref=np.array([[9.0, 1.0], [1.0, 9.0]]))
    hv_without = pooled_attainable_hypervolume(without, REFERENCE, UTOPIA)
    hv_with = pooled_attainable_hypervolume(with_strong, REFERENCE, UTOPIA)
    assert hv_with > hv_without
    # And the strong method now needs a HIGHER absolute HV to clear 90%.
    assert 0.9 * hv_with > 0.9 * hv_without


def test_the_pooled_union_must_match_its_declared_membership():
    """"Pooled" is meaningless without a membership list. A union that silently
    gains or loses a method changes every B_90 in the table while nothing in the
    table appears to change."""
    fronts = {m: np.array([[5.0, 5.0]]) for m in INTERNAL_METHOD_UNIVERSE}
    assert check_method_universe(
        fronts, INTERNAL_METHOD_UNIVERSE, label="internal")["n_members"] == 5

    fronts["a_method_nobody_declared"] = np.array([[9.0, 9.0]])
    with pytest.raises(MethodUniverseError):
        check_method_universe(fronts, INTERNAL_METHOD_UNIVERSE, label="internal")


def test_a_missing_member_is_also_caught_not_just_an_extra_one():
    fronts = {m: np.array([[5.0, 5.0]]) for m in INTERNAL_METHOD_UNIVERSE[:-1]}
    with pytest.raises(MethodUniverseError):
        check_method_universe(fronts, INTERNAL_METHOD_UNIVERSE, label="internal")


def test_region_coverage_punishes_one_excellent_cluster():
    """A method producing a single potency-heavy cluster must not score well
    just because its hypervolume is decent."""
    front = np.array([[0.0, 10.0], [10.0, 0.0]])
    clustered = np.array([[9.6, 0.4], [9.7, 0.3], [9.8, 0.2], [9.9, 0.1], [10.0, 0.0]])
    spread = np.array([[0.0, 10.0], [2.5, 7.5], [5.0, 5.0], [7.5, 2.5], [10.0, 0.0]])
    assert preference_region_coverage(clustered, front)["coverage"] == pytest.approx(0.2)
    assert preference_region_coverage(spread, front)["coverage"] == pytest.approx(1.0)


def test_region_coverage_flags_a_degenerate_front_instead_of_scoring_it():
    front = np.array([[5.0, 5.0], [5.0, 5.0]])
    out = preference_region_coverage(np.array([[5.0, 5.0]]), front)
    assert out["degenerate_front"] is True
    assert out["coverage"] == 0.0


# ---------------------------------------------------------------------------
# SMOKE SOURCE 000 -- the permanent regression case
#
# Real endpoints from the first completed held-in smoke source. Verified control
# improved the scalarized value for EVERY requested preference while the
# set-level hypervolume came out LOWER than greedy's. If a future refactor
# starts suppressing that comparison again, these fail loudly.
# ---------------------------------------------------------------------------

SRC000_NADIR = np.array([-1.26891193, -1.35472576])
SRC000_UTOPIA = np.array([2.73150485, 0.55541982])
SRC000_W = (0.1, 0.3, 0.5, 0.7, 0.9)
SRC000_GREEDY = np.array([[2.2626, 0.5925], [2.3472, 0.3812], [2.7389, 0.3673],
                          [2.7389, 0.3673], [2.7389, 0.3673]])
SRC000_VERIFIED = np.array([[2.7113, 0.5588], [2.6189, 0.5268], [2.6043, 0.5454],
                            [2.7525, 0.3673], [2.7525, 0.3673]])


def test_src000_verified_wins_every_per_preference_scalarized_value():
    """The guaranteed half. Policy improvement is pointwise in w."""
    sc = Scalarization(SRC000_UTOPIA)
    for w, g, v in zip(SRC000_W, SRC000_GREEDY, SRC000_VERIFIED):
        assert float(sc(v, w)[0]) <= float(sc(g, w)[0]) + 1e-9, f"w={w}"


def test_src000_verified_LOSES_on_set_level_hypervolume():
    """The falsifying half, and the whole point of the fixture.

    Five individually better points enclosed LESS dominated area than five worse
    but better-spread ones. A registry that marked this comparison sign-
    guaranteed would be asserting that this observed value cannot exist.
    """
    hv_g = normalized_hypervolume(SRC000_GREEDY, SRC000_NADIR, SRC000_UTOPIA)
    hv_v = normalized_hypervolume(SRC000_VERIFIED, SRC000_NADIR, SRC000_UTOPIA)
    assert hv_g == pytest.approx(1.0074, abs=5e-4)
    assert hv_v == pytest.approx(1.0060, abs=5e-4)
    assert hv_v < hv_g


def test_src000_forbids_suppressing_the_set_level_comparison():
    """The loud failure a future refactor must trip over."""
    assert not is_sign_guaranteed("verified_pref", "greedy_pref",
                                  "normalized_hypervolume")
    assert not is_sign_guaranteed("verified_pref", "greedy_pref",
                                  "set_level_hypervolume")
    assert is_sign_guaranteed("verified_pref", "greedy_pref",
                              "per_preference_scalarized_value")


def test_src000_also_exceeds_the_p99_scale_and_is_not_clipped():
    """Both arms put endpoints past z*, which is why HV lands above 1. The
    fixture pins that this is left alone."""
    assert (SRC000_GREEDY > SRC000_UTOPIA[None, :]).any()
    assert normalized_hypervolume(SRC000_GREEDY, SRC000_NADIR, SRC000_UTOPIA) > 1.0


# ---------------------------------------------------------------------------
# A guarantee must declare its SCOPE, not merely its existence
# ---------------------------------------------------------------------------

def test_every_guarantee_carries_a_written_reason_and_an_adversarial_metric():
    assert GUARANTEED_SIGN_REGISTRY, "registry must not be empty"
    for (arm, base), metrics in GUARANTEED_SIGN_REGISTRY.items():
        for metric, decl in metrics.items():
            assert decl.estimand.strip(), f"{arm} vs {base}: {metric}"
            assert len(decl.reason.split()) >= 20, (
                f"{arm} vs {base}: {metric} -- the reason must be a written "
                f"mathematical argument, not a label")
            assert decl.adversarial_metric.strip()
            assert decl.adversarial_reason.strip()
            # The adversarial metric must NOT itself be guaranteed on this pair.
            assert decl.adversarial_metric not in metrics, (
                f"{decl.adversarial_metric} cannot be both the adversarial "
                f"example and a guaranteed metric on the same arm pair")


def test_a_guarantee_without_an_adversarial_metric_is_rejected():
    with pytest.raises(ValueError):
        GuaranteedSign(estimand="x", reason="because " * 25,
                       adversarial_metric="  ", adversarial_reason="y")


def test_a_guarantee_without_a_reason_is_rejected():
    with pytest.raises(ValueError):
        GuaranteedSign(estimand="x", reason="   ",
                       adversarial_metric="m", adversarial_reason="y")


def test_the_declared_adversarial_metric_really_can_move_the_other_way():
    """Not a claim in a docstring -- demonstrated on the registered fixture."""
    decl = GUARANTEED_SIGN_REGISTRY[("verified_pref", "greedy_pref")][
        "per_preference_scalarized_value"]
    assert decl.adversarial_metric == "set_level_hypervolume"
    hv_g = normalized_hypervolume(SRC000_GREEDY, SRC000_NADIR, SRC000_UTOPIA)
    hv_v = normalized_hypervolume(SRC000_VERIFIED, SRC000_NADIR, SRC000_UTOPIA)
    assert hv_v < hv_g, "the adversarial metric must be shown moving the other way"


# ---------------------------------------------------------------------------
# Preference responsiveness -- ORDERED regions, not merely different SMILES
# ---------------------------------------------------------------------------

def test_ordering_rewards_a_correctly_ordered_fan():
    z = np.array([[1.0, 9.0], [3.0, 7.0], [5.0, 5.0], [7.0, 3.0], [9.0, 1.0]])
    out = preference_ordering(z, (0.1, 0.3, 0.5, 0.7, 0.9))
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert out["correct_adjacent_fraction"] == pytest.approx(1.0)
    assert out["monotone_non_decreasing"] is True


def test_ordering_punishes_distinct_but_UNORDERED_endpoints():
    """Five different molecules arranged arbitrarily is a controller responding
    to something, but not to the preference."""
    z = np.array([[5.0, 5.0], [9.0, 1.0], [1.0, 9.0], [7.0, 3.0], [3.0, 7.0]])
    out = preference_ordering(z, (0.1, 0.3, 0.5, 0.7, 0.9))
    assert out["distinct_values"] == 5
    assert out["correct_adjacent_fraction"] < 0.75
    assert out["monotone_non_decreasing"] is False


def test_ordering_does_not_credit_a_collapsed_arm():
    """Ties are neither correct nor incorrect adjacencies, so collapse cannot
    masquerade as perfect ordering."""
    z = np.repeat(np.array([[5.0, 5.0]]), 5, axis=0)
    out = preference_ordering(z, (0.1, 0.3, 0.5, 0.7, 0.9))
    assert out["distinct_values"] == 1
    assert out["correct_adjacent_fraction"] == pytest.approx(0.0)
    assert out["n_strict_adjacencies"] == 0


def test_src000_greedy_is_ordered_but_partially_collapsed():
    """The real fixture: correctly ordered where it moves, collapsed at the
    potency-heavy end -- which is the W4 watch item, visible in one source."""
    out = preference_ordering(SRC000_GREEDY, SRC000_W)
    assert out["monotone_non_decreasing"] is True
    assert out["distinct_values"] == 3
    assert out["n_strict_adjacencies"] == 2
