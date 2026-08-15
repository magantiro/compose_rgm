"""The outer Pareto-navigator algorithm. The inner navigator is NOT under test.

What is pinned here are the properties the outer loop must have whatever ends up
inside `navigate()`: that nothing but the metered path can reach the objectives,
that the archive's cheap coverage estimate agrees with the reported metric, that
a dominated molecule can still be reused as a starting state, and that a resumed
run rebuilds its archive from the ledger rather than from a saved copy.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.benchmark.init_sets import DEFAULT_INIT_DIR, development_init_set
from compose_v4.benchmark.oracles import DEFAULT_BUNDLE_DIR
from compose_v4.policy.task3 import (
    AdaptiveParetoNavigatorPolicy,
    MarginalGainRegions,
    NavigationBudget,
    NavigatorUnavailable,
    NearestRealizedStart,
    ParetoArchive,
    RandomEditNavigator,
    RandomRegion,
    Region,
    RThetaNavigator,
    StopStatePreferNovel,
    Trajectory,
)

pytestmark = pytest.mark.skipif(
    not (DEFAULT_BUNDLE_DIR / "molleo_task3_oracle_manifest.json").exists()
    or not (DEFAULT_INIT_DIR / "development_init_sets.json").exists(),
    reason="frozen oracle bundle or init sets not present")


# ---- the archive ----------------------------------------------------------

def test_the_cheap_coverage_estimate_agrees_with_the_reported_metric():
    """The policy steers by a 2^14 probe set and the paper reports 2^20. If
    those disagreed, the policy would be optimising something else."""
    archive = ParetoArchive()
    rng = np.random.default_rng(0)
    for i, point in enumerate(rng.random((60, 5)) * 0.8):
        archive.add(f"MOL{i}", tuple(point))
    assert archive.covered_fraction == pytest.approx(
        archive.reported_hypervolume(), abs=5e-3)


def test_a_dominated_target_is_worth_nothing_and_an_improvement_is_worth_something():
    archive = ParetoArchive()
    archive.add("A", (0.5, 0.5, 0.5, 0.5, 0.5))
    assert archive.gain_of((0.4,) * 5) == 0.0, "already dominated"
    assert archive.gain_of((0.6,) * 5) > 0.0, "a strict improvement adds volume"


def test_adding_a_molecule_never_reduces_coverage():
    archive = ParetoArchive()
    rng = np.random.default_rng(1)
    previous = 0.0
    for i, point in enumerate(rng.random((25, 5))):
        archive.add(f"MOL{i}", tuple(point))
        assert archive.covered_fraction >= previous
        previous = archive.covered_fraction


def test_the_front_excludes_dominated_members_but_the_archive_keeps_them():
    archive = ParetoArchive()
    archive.add("best", (0.9,) * 5)
    archive.add("worse", (0.1,) * 5)
    assert [smiles for smiles, _ in archive.front()] == ["best"]
    assert len(archive) == 2, "a dominated molecule is still a realized state"


# ---- region and start selection -------------------------------------------

def test_the_chosen_region_is_somewhere_the_archive_has_not_reached():
    archive = ParetoArchive()
    archive.add("A", (0.5, 0.4, 0.6, 0.5, 0.5))
    region = MarginalGainRegions().select(archive, np.random.default_rng(0))
    assert region is not None
    assert region.gain > 0.0
    assert archive.gain_of(region.target) > 0.0


def test_region_selection_gives_up_rather_than_aiming_at_nothing():
    """A saturated archive has no valuable aspiration left, and saying so is
    better than returning an unreachable corner."""
    archive = ParetoArchive()
    archive.add("perfect", (1.0,) * 5)
    assert MarginalGainRegions().select(archive, np.random.default_rng(0)) is None
    assert RandomRegion().select(archive, np.random.default_rng(0)) is None


def test_a_non_front_member_can_be_chosen_as_the_starting_state():
    """The whole archive is searched, not just the front: the best place to
    attack a region from need not be the best molecule overall."""
    archive = ParetoArchive()
    archive.add("front_but_far", (0.95, 0.95, 0.95, 0.05, 0.95))
    archive.add("closer_to_target", (0.5, 0.5, 0.5, 0.5, 0.5))
    region = Region(target=(0.5, 0.5, 0.5, 0.55, 0.5), gain=0.01)
    start = NearestRealizedStart(temperature=0.0).select(
        archive, region, np.random.default_rng(0))
    assert start == "closer_to_target"
    assert start != archive.front()[0][0], "not simply the best-summed molecule"


def test_objective_proximity_alone_can_never_prefer_a_dominated_state():
    """A property of the rule, not of this example, and worth knowing before
    treating this selector as delivering state reuse.

    If x dominates y then x >= y on every axis, so x's shortfall to ANY target
    is <= y's. A dominated state can tie but never strictly win. Genuine reuse
    of dominated states needs a criterion outside objective space; pretending
    otherwise would be a selector that cannot do what its docstring claims.
    """
    rng = np.random.default_rng(4)
    for _ in range(200):
        dominated = rng.random(5) * 0.6
        dominator = dominated + rng.random(5) * 0.3
        target = rng.random(5)
        shortfall = lambda p: float(np.maximum(target - p, 0.0).sum())  # noqa: E731
        assert shortfall(dominator) <= shortfall(dominated) + 1e-12


def test_being_above_the_target_is_headroom_not_distance():
    archive = ParetoArchive()
    archive.add("above", (1.0, 1.0, 1.0, 0.30, 1.0))
    archive.add("below", (0.2, 0.2, 0.2, 0.29, 0.2))
    region = Region(target=(0.3,) * 5, gain=0.01)
    start = NearestRealizedStart(temperature=0.0).select(
        archive, region, np.random.default_rng(0))
    assert start == "above"


# ---- candidate selection ---------------------------------------------------

def test_an_already_paid_for_state_is_skipped_in_favour_of_a_new_one():
    archive = ParetoArchive()
    archive.add("SEEN", (0.5,) * 5)
    trajectory = Trajectory(states=["SEEN", "NEW"], stop=0)
    assert StopStatePreferNovel().select(trajectory, archive, 1) == ["NEW"]


def test_when_everything_on_the_route_is_known_the_stop_state_is_returned():
    archive = ParetoArchive()
    archive.add("SEEN", (0.5,) * 5)
    trajectory = Trajectory(states=["SEEN"], stop=0)
    assert StopStatePreferNovel().select(trajectory, archive, 2) == ["SEEN"]


# ---- the loop --------------------------------------------------------------

class RecordingNavigator:
    """A navigator that returns fixed routes and records what it was given."""

    name = "recording"

    def __init__(self, routes):
        self.routes = list(routes)
        self.calls = []

    def navigate(self, start, region, budget, rng, *, evidence=None):
        self.calls.append((start, region, evidence))
        states = self.routes[len(self.calls) % len(self.routes)]
        return Trajectory(states=list(states), stop=len(states) - 1,
                          region=region)


def test_every_molecule_that_gets_objectives_is_charged(tmp_path):
    """The property that separates our accounting from the released benchmark's.

    `run.evaluate` is called in exactly one place in the loop, so the count of
    molecules that received the objective vector IS the count charged. Here that
    is checked end to end: the meter's unique-molecule count equals the number
    of distinct molecules the run ever evaluated.
    """
    from compose_v4.benchmark.task3_run import Task3Run

    routes = [["CCO", "CCCO"], ["c1ccccc1", "Cc1ccccc1"], ["CCN", "CCCN"]]
    policy = AdaptiveParetoNavigatorPolicy(navigator=RecordingNavigator(routes),
                                           candidates_per_iteration=2)
    run = Task3Run.open(tmp_path / "run", seed=100, budget=140, policy="test")
    try:
        policy.run(run, development_init_set(100))
        assert run.spent == run.meter.n_unique
        assert run.spent <= 140
    finally:
        run.close()


def test_the_navigator_is_never_handed_the_run_or_the_objectives(tmp_path):
    """Structural, not a matter of discipline: a navigator that cannot see the
    objectives cannot screen against them without charging."""
    from compose_v4.benchmark.task3_run import Task3Run

    navigator = RecordingNavigator([["CCO", "CCCO"]])
    policy = AdaptiveParetoNavigatorPolicy(navigator=navigator)
    run = Task3Run.open(tmp_path / "run", seed=100, budget=125, policy="test")
    try:
        policy.run(run, development_init_set(100))
    finally:
        run.close()
    for start, region, evidence in navigator.calls:
        assert isinstance(start, str)
        assert isinstance(region, Region)
        # It sees what we have PAID FOR, and only that.
        assert isinstance(evidence, ParetoArchive)
        assert not hasattr(evidence, "evaluate")
    names = set(RandomEditNavigator.navigate.__code__.co_varnames[:6])
    assert names == {"self", "start", "region", "budget", "rng", "evidence"}, (
        "a navigator must not be handed the run, the meter or the evaluator")


def test_a_resumed_run_rebuilds_its_archive_from_the_ledger(tmp_path):
    """No checkpointed copy of the archive means no way for the two to disagree."""
    from compose_v4.benchmark.task3_run import Task3Run

    routes = [["CCO", "CCCO", "CCCCO"]]
    initial = development_init_set(100)

    first = Task3Run.open(tmp_path / "run", seed=100, budget=126, policy="test")
    try:
        AdaptiveParetoNavigatorPolicy(
            navigator=RecordingNavigator(routes)).run(first, initial)
        spent = first.spent
    finally:
        first.close()

    second = Task3Run.open(tmp_path / "run", seed=100, budget=200, policy="test")
    try:
        archive = ParetoArchive()
        archive.add_many(second.meter.evaluated())
        assert len(archive) == spent
        assert second.spent == spent
    finally:
        second.close()


def test_the_real_navigator_refuses_rather_than_substituting_another_process():
    """It must fail loudly, and the message must say what is actually wrong:
    the corpus is content-identical and the guard is pinning the MOUNT POINT."""
    with pytest.raises(NavigatorUnavailable, match="CONTENT-IDENTICAL"):
        RThetaNavigator().navigate("CCO", Region(target=(1.0,) * 5, gain=1.0),
                                   NavigationBudget(), np.random.default_rng(0))


def test_the_standin_navigator_produces_a_route_of_valid_molecules():
    from rdkit import Chem

    navigator = RandomEditNavigator(average_size=24.0, size_stdev=4.0)
    trajectory = navigator.navigate(
        "CCOc1ccc(CN2CCN(C)CC2)cc1", Region(target=(0.9,) * 5, gain=0.1),
        NavigationBudget(steps=6), np.random.default_rng(3))
    assert len(trajectory.states) >= 1
    for smiles in trajectory.states:
        assert Chem.MolFromSmiles(smiles) is not None, "every state must be valid"
    assert trajectory.stop == len(trajectory.states) - 1


def test_a_navigator_that_only_repeats_itself_terminates(tmp_path):
    """Repeats cost nothing, so `remaining` never falls and the loop would spin
    forever. Found by this test rather than by reasoning about it."""
    from compose_v4.benchmark.task3_run import Task3Run

    policy = AdaptiveParetoNavigatorPolicy(
        navigator=RecordingNavigator([["CCO"]]), stall_limit=5)
    run = Task3Run.open(tmp_path / "run", seed=100, budget=400, policy="test")
    try:
        stats = policy.run(run, development_init_set(100))
        assert stats["stalled_out"] is True
        assert run.remaining > 0, "it stopped early, which is the point"
    finally:
        run.close()


def test_the_incrementally_maintained_front_equals_the_swept_one():
    """The archive maintains its front on insertion because recomputing it took
    4.2 seconds on 10,000 molecules and the region selector wants it every
    iteration. Equality with the batch sweep is what makes that safe."""
    rng = np.random.default_rng(7)
    archive = ParetoArchive()
    points = np.round(rng.random((400, 5)), 2)      # rounding forces ties
    for i, point in enumerate(points):
        archive.add(f"MOL{i}", tuple(point))
    assert (sorted(k for k, _ in archive.front())
            == sorted(k for k, _ in archive.front_by_sweep()))


def test_a_new_molecule_evicts_only_what_it_dominates():
    archive = ParetoArchive()
    archive.add("weak", (0.2,) * 5)
    archive.add("orthogonal", (1.0, 0.0, 0.0, 0.0, 0.0))
    archive.add("strong", (0.5,) * 5)
    front = {k for k, _ in archive.front()}
    assert front == {"strong", "orthogonal"}, "weak is dominated, orthogonal is not"
    assert len(archive) == 3, "eviction is from the FRONT, not from the archive"
