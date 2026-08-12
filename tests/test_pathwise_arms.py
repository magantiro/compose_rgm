"""The five arms, exercised locally against a synthetic successor graph.

The frozen R_theta only runs on Modal, but the POLICIES are pure functions of
an injected kernel, so their semantics are testable here: what the mask
removes, where an arm dead-ends, which trajectory endpoint-only selection
returns, and -- the point of the workstream -- that endpoint-only can return a
trajectory whose intermediates the pathwise arms could never have committed.

The synthetic graph below is hand-built so that exactly one route to the best
endpoint passes through a motif-violating state. If the arms are wired
correctly, `endpoint_only` takes it and `pathwise_*` cannot.
"""

from __future__ import annotations

import random

import pytest

from compose_v4.experiments.pathwise_arms import (
    ALL_ARMS,
    ARM_BUILDERS,
    STAGE_A,
    STAGE_B,
    ArmContext,
    best_of_n,
    greedy_path,
    stochastic_path,
    verified_path,
)
from compose_v4.experiments.pathwise_constraints import (
    derive_protected_motif,
    path_violation_summary,
)

# Protected motif: the benzene ring of the source.
SOURCE = "CCN1CCN(c2ccccc2)CC1"
MOTIF = derive_protected_motif(SOURCE).smarts

#: A tiny hand-built successor graph.
#:
#:   SOURCE ──► KEEP_1  ──► KEEP_2          (motif intact, mediocre utility)
#:        └───► BREAK   ──► RESTORED        (motif broken, then restored: BEST)
#:
#: RESTORED is the highest-utility endpoint and is motif-valid, but the only
#: route to it destroys the ring on the way. That is the whole experiment in
#: five molecules.
KEEP_1 = "CCN1CCN(c2ccccc2Cl)CC1"
KEEP_2 = "CCN1CCN(c2ccc(Cl)cc2Cl)CC1"
BREAK = "CCN1CCN(C2CCCCC2)CC1"          # dearomatised: VIOLATION
RESTORED = "CCN1CCN(c2ccccc2Br)CC1"     # ring back, best endpoint
DEAD = "CCN1CCN(C2CCCCC2Br)CC1"         # violating leaf

GRAPH: dict[str, list[tuple[str, float]]] = {
    SOURCE: [(KEEP_1, 0.5), (BREAK, 0.5)],
    KEEP_1: [(KEEP_2, 1.0)],
    KEEP_2: [],
    BREAK: [(RESTORED, 0.5), (DEAD, 0.5)],
    RESTORED: [],
    DEAD: [],
}

UTILITY: dict[str, float] = {
    SOURCE: 0.0,
    KEEP_1: 1.0,
    KEEP_2: 2.0,
    BREAK: 3.0,        # the trap: the violating state looks good to greedy
    RESTORED: 9.0,     # global best endpoint, motif-valid
    DEAD: 4.0,
}


def context(**overrides) -> ArmContext:
    base = {
        "successors": lambda key: list(GRAPH.get(key, [])),
        "utility": lambda key: (UTILITY[key], UTILITY[key]),
        "motif_smarts": MOTIF,
        "rng": random.Random(0),
        "horizon": 2,
        "rollouts": 6,
        "shortlist": 3,
    }
    base.update(overrides)
    return ArmContext(**base)


# ------------------------------------------------------------------ plumbing


def test_stage_partition_covers_every_arm_exactly_once():
    assert set(STAGE_A) | set(STAGE_B) == set(ALL_ARMS)
    assert not set(STAGE_A) & set(STAGE_B)
    assert set(ARM_BUILDERS) == set(ALL_ARMS)


def test_masking_costs_no_extra_kernel_calls():
    """A masked and an unconstrained arm visiting a state share one call."""
    calls: list[str] = []

    def counting(key: str):
        calls.append(key)
        return list(GRAPH.get(key, []))

    cache: dict[str, list] = {}

    def cached(key: str):
        if key not in cache:
            cache[key] = counting(key)
        return cache[key]

    ctx = context(successors=cached)
    greedy_path(ctx, SOURCE, mask=False)
    unconstrained_calls = len(calls)
    greedy_path(ctx, SOURCE, mask=True)
    # the masked arm re-walks states the unconstrained arm already enumerated
    assert len(calls) < 2 * unconstrained_calls


# ------------------------------------------------------------------ the mask


def test_mask_removes_the_violating_successor():
    ctx = context()
    assert [k for k, _ in ctx.support(SOURCE, mask=False)] == [KEEP_1, BREAK]
    assert [k for k, _ in ctx.support(SOURCE, mask=True)] == [KEEP_1]


def test_unconstrained_greedy_takes_the_forbidden_route():
    """Greedy on utility walks into the violating state. Not guaranteed --
    it is a property of this graph, and the test pins it so the fixture keeps
    exercising the case the workstream is about."""
    result = greedy_path(context(), SOURCE, mask=False)
    assert result["trajectory"] == [SOURCE, BREAK, RESTORED]
    audit = path_violation_summary(MOTIF, result["trajectory"])
    assert audit["endpoint_valid_path_invalid"] is True
    assert audit["first_violation_index"] == 1


def test_pathwise_greedy_cannot_reach_the_best_endpoint():
    """The guarantee has a price, and this is where it shows up."""
    result = greedy_path(context(), SOURCE, mask=True)
    assert result["trajectory"] == [SOURCE, KEEP_1, KEEP_2]
    audit = path_violation_summary(MOTIF, result["trajectory"])
    assert audit["any_violation"] is False
    assert UTILITY[result["trajectory"][-1]] < UTILITY[RESTORED]


def test_pathwise_arms_never_commit_a_violation():
    """BUG DETECTOR, not a finding: this is true by construction. It fails
    only if the mask leaks."""
    for name in ("pathwise_greedy", "pathwise_stochastic", "mask_only_sampling",
                 "pathwise_verified"):
        for seed in range(12):
            ctx = context(rng=random.Random(seed))
            result = ARM_BUILDERS[name](ctx, SOURCE)
            audit = path_violation_summary(MOTIF, result["trajectory"])
            assert audit["any_violation"] is False, (name, seed, result["trajectory"])


def test_masked_dead_end_is_recorded_not_silently_truncated():
    ctx = context(horizon=4)
    result = greedy_path(ctx, SOURCE, mask=True)
    assert result["dead_end_step"] == 2  # KEEP_2 has no successors
    assert result["trajectory"][-1] == KEEP_2


# ------------------------------------------------------- endpoint-only arm


def test_endpoint_only_returns_a_path_invalid_trajectory():
    """The central claim, in miniature: endpoint-only filtering accepts a
    molecule that got there through a state the constraint forbids."""
    ctx = context(rng=random.Random(1))
    result = best_of_n(ctx, SOURCE, mask=False, goal_aware=True,
                       endpoint_filter=True)
    assert result["selection_failed"] is False
    assert result["trajectory"][-1] == RESTORED
    audit = path_violation_summary(MOTIF, result["trajectory"])
    assert audit["endpoint_valid"] is True
    assert audit["endpoint_valid_path_invalid"] is True


def test_endpoint_only_keeps_the_discarded_rollouts_for_audit():
    ctx = context(rng=random.Random(2))
    result = best_of_n(ctx, SOURCE, mask=False, goal_aware=True,
                       endpoint_filter=True)
    assert len(result["rollouts"]) == ctx.rollouts
    assert all("audit" in roll for roll in result["rollouts"])
    assert result["rollouts_admissible"] <= result["rollouts_offered"]


def test_endpoint_filter_rejects_endpoint_invalid_rollouts():
    """A rollout landing on DEAD must never be selectable."""
    ctx = context(rng=random.Random(3))
    result = best_of_n(ctx, SOURCE, mask=False, goal_aware=True,
                       endpoint_filter=True)
    assert result["trajectory"][-1] != DEAD


def test_endpoint_only_can_fail_outright():
    """If no rollout has a valid endpoint the arm returns nothing. Reported,
    not silently replaced by an unfiltered best."""
    graph = {SOURCE: [(BREAK, 1.0)], BREAK: [(DEAD, 1.0)], DEAD: []}
    ctx = context(successors=lambda k: list(graph.get(k, [])))
    result = best_of_n(ctx, SOURCE, mask=False, goal_aware=True,
                       endpoint_filter=True)
    assert result["selection_failed"] is True
    assert result["rollouts_admissible"] == 0
    assert result["trajectory"] == [SOURCE]


def test_pathwise_stochastic_is_the_budget_matched_twin():
    """Same rollout count and same policy as endpoint_only; only the mask
    differs. That is what makes the utility gap attributable to the mask."""
    unconstrained = best_of_n(context(rng=random.Random(4)), SOURCE, mask=False,
                              goal_aware=True, endpoint_filter=True)
    masked = best_of_n(context(rng=random.Random(4)), SOURCE, mask=True,
                       goal_aware=True, endpoint_filter=False)
    assert unconstrained["rollouts_offered"] == masked["rollouts_offered"]
    assert path_violation_summary(MOTIF, masked["trajectory"])["any_violation"] is False


# ------------------------------------------------------------- sampling arm


def test_mask_only_sampling_ignores_the_goal():
    """It must be able to pick a low-utility successor -- otherwise it is just
    another goal-directed arm and cannot separate feasibility from control."""
    graph = {
        SOURCE: [(KEEP_1, 0.5), ("CCN1CCN(c2ccccc2F)CC1", 0.5)],
        KEEP_1: [],
        "CCN1CCN(c2ccccc2F)CC1": [],
    }
    utility = {SOURCE: 0.0, KEEP_1: 9.0, "CCN1CCN(c2ccccc2F)CC1": 0.1}
    landings = set()
    for seed in range(30):
        ctx = context(successors=lambda k: list(graph.get(k, [])),
                      utility=lambda k: (utility[k], utility[k]),
                      rng=random.Random(seed), horizon=1)
        landings.add(stochastic_path(ctx, SOURCE, mask=True,
                                     goal_aware=False)["trajectory"][-1])
    assert len(landings) == 2, "goal-free sampling collapsed onto the argmax"


def test_goal_aware_shortlist_of_one_is_exactly_greedy():
    """The stochastic family degenerates to greedy, so it is not a separate
    tunable policy."""
    for seed in range(6):
        stochastic = stochastic_path(context(rng=random.Random(seed), shortlist=1),
                                     SOURCE, mask=False, goal_aware=True)
        greedy = greedy_path(context(), SOURCE, mask=False)
        assert stochastic["trajectory"] == greedy["trajectory"]


# ---------------------------------------------------------- verified control


def test_verified_lookahead_avoids_the_greedy_trap_when_unmasked():
    """Greedy walks to BREAK because it scores 3 > 1. Under this graph both
    routes are still reachable, so lookahead should not be worse."""
    result = verified_path(context(), SOURCE, mask=False)
    assert UTILITY[result["trajectory"][-1]] >= UTILITY[
        greedy_path(context(), SOURCE, mask=False)["trajectory"][-1]
    ]


def test_verified_records_top1_disagreement_separately_from_the_gap():
    """`top1_disagreements` is the non-circular statistic. It must exist and be
    countable independently of whether the utility improved."""
    result = verified_path(context(), SOURCE, mask=False)
    assert "top1_disagreements" in result
    assert "overrides" in result
    assert result["overrides"] <= result["top1_disagreements"] + len(
        result["trajectory"])


def test_verified_under_the_mask_never_leaves_the_feasible_set():
    for seed in range(8):
        result = verified_path(context(rng=random.Random(seed)), SOURCE, mask=True)
        assert path_violation_summary(MOTIF, result["trajectory"])[
            "any_violation"] is False


def test_verified_lookahead_respects_the_mask_during_rollouts():
    """If the lookahead planned over the unmasked support it could prefer an
    action whose value depends on a route the mask forbids."""
    ctx = context(rng=random.Random(0))
    result = verified_path(ctx, SOURCE, mask=True)
    assert result["trajectory"] == [SOURCE, KEEP_1, KEEP_2]


def test_verified_at_least_matches_greedy_under_the_same_support():
    """Policy improvement. Asserted so a regression is caught -- and recorded
    here as the reason the greedy/verified gap is NOT a primary statistic."""
    for seed in range(8):
        rng_a, rng_b = random.Random(seed), random.Random(seed)
        verified = verified_path(context(rng=rng_a), SOURCE, mask=True)
        greedy = greedy_path(context(rng=rng_b), SOURCE, mask=True)
        assert UTILITY[verified["trajectory"][-1]] >= UTILITY[
            greedy["trajectory"][-1]]


# --------------------------------------------------------------- every arm


@pytest.mark.parametrize("name", ALL_ARMS)
def test_every_arm_produces_a_wellformed_trajectory(name):
    ctx = context(rng=random.Random(7))
    result = ARM_BUILDERS[name](ctx, SOURCE)
    trajectory = result["trajectory"]
    assert trajectory[0] == SOURCE
    assert len(trajectory) <= ctx.horizon + 1
    for previous, following in zip(trajectory, trajectory[1:], strict=False):
        assert following in [k for k, _ in GRAPH[previous]], (name, previous)


def test_arms_are_not_all_the_same_trajectory():
    """If every arm lands identically the comparison is empty. On the real
    panel this is a stop rule; here it guards the fixture."""
    landings = {
        name: ARM_BUILDERS[name](context(rng=random.Random(5)), SOURCE)[
            "trajectory"][-1]
        for name in ALL_ARMS
    }
    assert len(set(landings.values())) > 1, landings
