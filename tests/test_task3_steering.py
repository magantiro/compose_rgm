"""The two steering rules and the shared surrogate.

These are new algorithmic components, so what is pinned is behaviour that would
silently invalidate the A/B if it changed -- not numerical parity with anything.
The single most important one is the last test in the file: both rules expose
the same three methods, which is what makes "only the steering varies" true
rather than merely intended.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.benchmark.oracles import DEFAULT_BUNDLE_DIR
from compose_v4.policy.task3.archive import ParetoArchive
from compose_v4.policy.task3.interfaces import Region
from compose_v4.policy.task3.steering import AdaptiveRegion, FixedScalarization
from compose_v4.policy.task3.surrogate import TanimotoKNN

pytestmark = pytest.mark.skipif(
    not (DEFAULT_BUNDLE_DIR / "molleo_task3_oracle_manifest.json").exists(),
    reason="frozen oracle bundle not present")

TRAIN = ["CCO", "CCCO", "c1ccccc1", "Cc1ccccc1", "CC(=O)O"]
LABELS = [(0.1, 0.2, 0.3, 0.4, 0.5), (0.2, 0.2, 0.3, 0.4, 0.5),
          (0.9, 0.8, 0.7, 0.6, 0.5), (0.8, 0.8, 0.7, 0.6, 0.5),
          (0.3, 0.3, 0.3, 0.3, 0.3)]


# ---- the surrogate ---------------------------------------------------------

def test_a_molecule_it_was_trained_on_predicts_close_to_its_own_label():
    surrogate = TanimotoKNN()
    surrogate.update(TRAIN, LABELS)
    predicted = surrogate.predict(["c1ccccc1"])[0]
    assert predicted == pytest.approx(np.array(LABELS[2]), abs=0.25)


def test_an_unparseable_molecule_gets_the_worst_vector_not_a_neutral_one():
    """A surrogate that cannot see a molecule must not make it look attractive."""
    surrogate = TanimotoKNN()
    surrogate.update(TRAIN, LABELS)
    assert np.array_equal(surrogate.predict(["not a molecule"])[0], np.zeros(5))


def test_an_empty_surrogate_predicts_the_worst_vector():
    assert np.array_equal(TanimotoKNN().predict(["CCO"])[0], np.zeros(5))
    assert len(TanimotoKNN()) == 0


def test_updating_is_incremental_and_counts_only_usable_labels():
    surrogate = TanimotoKNN()
    assert surrogate.update(["CCO", "~~~junk~~~"], [LABELS[0], LABELS[1]]) == 1
    assert len(surrogate) == 1
    assert surrogate.update(["c1ccccc1"], [LABELS[2]]) == 1
    assert len(surrogate) == 2


def test_the_surrogate_works_while_the_oracle_is_locked_out():
    """This is the whole point of it: a navigator may reason about molecules
    during navigation, when evaluating one would raise."""
    from compose_v4.benchmark.oracles.task3 import navigation_lockout

    surrogate = TanimotoKNN()
    surrogate.update(TRAIN, LABELS)
    with navigation_lockout():
        assert surrogate.predict(["CCO"]).shape == (1, 5)


# ---- arm A -----------------------------------------------------------------

def test_fixed_scalarization_ranks_by_the_weighted_sum():
    arm = FixedScalarization()
    predicted = np.array([[1.0, 0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5, 0.5]])
    scores = arm.rank(predicted, arm.target(ParetoArchive(), np.random.default_rng(0)))
    assert scores[1] > scores[0]
    assert scores[1] == pytest.approx(0.5)


def test_fixed_scalarizations_target_never_moves():
    """The property under test: it cannot notice a missing part of the front."""
    arm = FixedScalarization()
    rng = np.random.default_rng(0)
    empty = arm.target(ParetoArchive(), rng)
    archive = ParetoArchive()
    archive.add("A", (0.9, 0.1, 0.9, 0.9, 0.9))
    assert arm.target(archive, rng).target == empty.target == (1.0,) * 5


def test_fixed_scalarization_starts_from_the_best_summed_molecule():
    archive = ParetoArchive()
    archive.add("balanced", (0.6,) * 5)
    archive.add("spiky", (1.0, 0.0, 0.0, 0.0, 0.0))
    arm = FixedScalarization()
    target = arm.target(archive, np.random.default_rng(0))
    assert arm.start(archive, target, np.random.default_rng(0)) == "balanced"


# ---- arm B -----------------------------------------------------------------

def test_adaptive_region_ranks_by_shortfall_to_the_aspiration():
    arm = AdaptiveRegion()
    region = Region(target=(0.5, 0.9, 0.5, 0.5, 0.5), gain=0.01)
    close = np.array([[0.5, 0.85, 0.5, 0.5, 0.5]])
    far = np.array([[0.9, 0.1, 0.9, 0.9, 0.9]])
    assert arm.rank(close, region)[0] > arm.rank(far, region)[0]


def test_being_above_the_aspiration_earns_no_extra_credit():
    """Otherwise arm B is a scalarization with extra steps, and the A/B is void."""
    arm = AdaptiveRegion()
    region = Region(target=(0.5,) * 5, gain=0.01)
    exactly_met = np.array([[0.5] * 5])
    far_above = np.array([[1.0] * 5])
    assert arm.rank(exactly_met, region)[0] == pytest.approx(
        arm.rank(far_above, region)[0])


def test_adaptive_region_moves_its_target_as_the_archive_fills():
    arm = AdaptiveRegion()
    rng = np.random.default_rng(0)
    archive = ParetoArchive()
    archive.add("A", (0.5, 0.1, 0.5, 0.5, 0.5))
    first = arm.target(archive, rng)
    archive.add("B", (0.5, 0.9, 0.5, 0.5, 0.5))
    second = arm.target(archive, rng)
    assert first is not None and second is not None
    assert first.target != second.target, (
        "the aspiration must follow what the archive is still missing")


# ---- the property that makes the A/B a controlled comparison ---------------

def test_both_rules_expose_exactly_the_same_interface():
    """The loop driving them is shared code, so anything one rule has and the
    other lacks would become an uncontrolled difference between the arms."""
    for method in ("target", "start", "rank"):
        assert callable(getattr(FixedScalarization(), method))
        assert callable(getattr(AdaptiveRegion(), method))
    assert FixedScalarization().name != AdaptiveRegion().name
