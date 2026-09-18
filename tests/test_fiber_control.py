"""Guards on the FiberControl runtime.

The objective is `P*(tau) ∝ P0(tau) * 1[tau queryable] * exp(beta * J(tau))`. Three
properties of that decomposition are load-bearing and each one is here because breaking
it silently produced a wrong campaign:

  * the gate is the SUPPORT, so `Fiber.check` must refuse every ineligible endpoint
    itself rather than leave it to the oracle -- an omitted `SA < 4` invalidated a
    whole 32-call experiment;
  * reward decides preference INSIDE that support, so the value model must survive the
    constant intercept column that made it singular three times;
  * STOP must not fire on an unfitted opinion -- the first version terminated a campaign
    at 17 of 72 calls while a far better molecule was known to exist.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from compose_v4.control.fiber_control import (
    ProgramValue,
    SearchState,
    acquisition,
    program_features,
    should_stop,
)
from compose_v4.experiments.t4_fiber_campaign import Fiber, expand

JAK2_ROOT = "COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34"


def _candidate(gain, bits, *, regions=1, families=("base",), parent_score=-9.0):
    record = {
        "parent_score": parent_score,
        "similarity": 0.62,
        "qed": 0.66,
        "sa": 3.4,
        "regions": regions,
        "created": 2,
        "deleted": 0,
        "delta": 0.6,
        "families": list(families),
    }
    state = SearchState(archive={"x": -9.0})
    return {
        **record,
        "smiles": f"c{gain}",
        "features": program_features(record, state),
        "fingerprint": set(bits),
    }


# ---- Support: the gate refuses what the benchmark would refuse ----


def test_the_fiber_admits_its_own_seed():
    admitted = Fiber(JAK2_ROOT, 0.6).check(JAK2_ROOT)
    assert admitted is not None and admitted["similarity"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "smiles, reason",
    [
        ("CCO", "far below the 18-atom floor and nothing like the seed"),
        ("COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34.O", "disconnected"),
        ("O=C(O)CCCCCCCCCCCCCCCCCC", "similarity far below delta"),
    ],
)
def test_the_fiber_refuses_ineligible_endpoints(smiles, reason):
    assert Fiber(JAK2_ROOT, 0.6).check(smiles) is None, reason


# Real endpoints the production program process emits from the JAK2 root, each one
# eligible on every BENCHMARK threshold but one. They exist because a threshold omitted
# from an earlier version of this fiber invalidated a 32-call experiment: only 11 of its
# 28 docked candidates passed `SA <= 4` and none of its six controls did.
#
# An aldehyde witness used to live in this list, and that was the very conflation this
# module now separates: an aldehyde violates no benchmark threshold. It moved to the
# legacy-screened test, where a medicinal-chemistry preference belongs.
GATE_WITNESSES = (
    ("COC(=O)CC1Nc2ccpcc2-c2ccnc3[nH]cc1c23", "SA 4.07 > 4.0", "sim 0.741, QED 0.706"),
    ("OC1Nc2ccccc2-c2ccnc3[nH]cc1c23", "QED 0.563 < 0.6", "sim 0.600, SA 3.15"),
)


@pytest.mark.parametrize("smiles, refused_for, otherwise", GATE_WITNESSES)
def test_every_benchmark_threshold_refuses_before_a_call_is_spent(smiles, refused_for, otherwise):
    assert Chem.MolFromSmiles(smiles) is not None, "the witness must be a real molecule"
    assert Fiber(JAK2_ROOT, 0.6).check(smiles) is None, f"{refused_for}, though {otherwise}"


def test_the_benchmark_fiber_is_the_benchmark_criterion_and_nothing_else():
    """Three ways the screened fiber was tighter than the task it claims to serve.

    The contract is "RETURNED only QED>=0.6, SA<=4, sim>=delta". Searching under anything
    narrower and then comparing to a published number obtained on the full criterion is
    not a comparison. The screened bundle refuses 7.0% of the benchmark's own lead set.
    """
    from compose_v4.experiments.t4_fiber_campaign import BENCHMARK, LEGACY_SCREENED

    official = Fiber(JAK2_ROOT, 0.6, support=BENCHMARK)
    screened = Fiber(JAK2_ROOT, 0.6, support=LEGACY_SCREENED)

    # A molecule the med-chem bundle refuses but the task accepts.
    aldehyde = "O=CCC1Nc2ccccc2-c2ccnc3[nH]cc1c23"
    assert official.check(aldehyde) is not None, "the task does not screen aldehydes"
    assert screened.check(aldehyde) is None, "the legacy bundle does"

    with pytest.raises(ValueError):
        Fiber(JAK2_ROOT, 0.6, support="something_else")


def test_the_gate_is_inclusive_at_its_own_thresholds():
    """QED >= 0.6 and SA <= 4, not strictly-greater and strictly-less.

    This is not pedantry: the delta=0.6 leaders carry a median similarity margin of
    +0.019 and several sit at exactly 0.600, so high reward concentrates ON the active
    constraint and a boundary-exclusive gate excludes exactly the interesting corner.
    """
    import inspect

    from compose_v4.experiments import t4_fiber_campaign

    source = inspect.getsource(t4_fiber_campaign.Fiber.check)
    assert "quality < QED_MIN" in source, "QED must be admitted AT the threshold"
    assert "access > SA_MAX" in source, "SA must be admitted AT the threshold"
    assert "18 <=" not in source, "the 18-heavy-atom floor is not a benchmark criterion"


def test_the_similarity_gate_is_the_delta_that_was_asked_for():
    # The same molecule may be feasible at 0.4 and infeasible at 0.6; the two lanes are
    # different problems and a molecule may not be carried between them.
    analogue = "COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34"
    loose = Fiber(analogue, 0.4)
    assert loose.check(analogue) is not None
    assert Fiber(analogue, 0.6).delta == 0.6


# ---- Preference: the value model, and the intercept that kept going singular ----


def test_the_value_model_survives_its_constant_column():
    rng = np.random.default_rng(0)
    features = [program_features({"parent_score": -8.0 - 0.1 * i}, SearchState()) for i in range(12)]
    improvements = rng.normal(size=12)
    value = ProgramValue()
    value.fit(features, improvements)  # the last feature is a hard-coded 1.0
    assert value.weights is not None
    assert np.isfinite(value.predict(np.asarray(features))).all()


def test_the_value_model_abstains_until_it_has_observations():
    value = ProgramValue()
    value.fit([np.ones(15)] * 3, [1.0, 2.0, 3.0])
    assert value.weights is None
    assert value.predict(np.ones((2, 15))).tolist() == [0.0, 0.0]


def test_the_value_model_recovers_a_linear_improvement_signal():
    rng = np.random.default_rng(1)
    records = [
        {"parent_score": -9.0, "similarity": 0.6 + 0.01 * i, "qed": 0.66, "sa": 3.4,
         "regions": 1, "created": 2, "deleted": 0, "delta": 0.6, "families": ["base"]}
        for i in range(40)
    ]
    features = [program_features(r, SearchState()) for r in records]
    # improvement grows with the similarity margin
    improvements = [8.0 * (r["similarity"] - 0.6) + 0.01 * rng.normal() for r in records]
    value = ProgramValue()
    value.fit(features, improvements)
    predicted = value.predict(np.asarray(features))
    assert np.corrcoef(predicted, improvements)[0, 1] > 0.9


# ---- Acquisition ----


def test_acquisition_returns_a_distinct_batch():
    candidates = [_candidate(i, range(i, i + 40)) for i in range(20)]
    picked = acquisition(candidates, ProgramValue(), SearchState(), np.random.default_rng(0), batch=8)
    assert len(picked) == 8 and len(set(picked)) == 8


def test_acquisition_spends_its_batch_on_distinct_structures():
    # Twelve structural twins and four genuinely different candidates. Measured docking
    # noise here is 0.1-0.2, so a batch of near-identical molecules buys one observation
    # at the price of several calls.
    twins = [_candidate(i, range(100)) for i in range(12)]
    distinct = [_candidate(100 + i, range(500 * (i + 1), 500 * (i + 1) + 40)) for i in range(4)]
    candidates = twins + distinct
    # Diversity only governs the MODEL's slots, so the model must be fitted and the
    # random quota closed for this to be a test of diversity rather than of luck.
    value = ProgramValue()
    value.fit([c["features"] for c in candidates] * 4, [0.0] * (len(candidates) * 4))
    picked = acquisition(
        candidates, value, SearchState(), np.random.default_rng(3),
        batch=4, diversity=50.0, exploration=0,
    )
    assert sum(1 for i in picked if i >= 12) >= 3


def test_acquisition_ranks_by_the_endpoint_score_not_by_the_gain():
    """The defect that cost an autonomous campaign its middle 24 calls.

    Two candidates with the SAME predicted gain, one hanging off a -11.0 parent and one
    off a -7.0 parent. Their endpoint scores differ by 4.0, so only the first can carry
    the objective; ranking on gain alone cannot tell them apart and in practice bought
    the easier improvement off the weaker parent.
    """

    class _Flat(ProgramValue):
        def __init__(self):
            super().__init__()
            self.weights, self.n = np.zeros(15), 100

        def predict(self, features):
            return np.zeros(np.asarray(features).shape[0])

    strong = _candidate(0, range(40), parent_score=-11.0)
    weak = _candidate(1, range(500, 540), parent_score=-7.0)
    rng = np.random.default_rng(0)
    first = [acquisition([strong, weak], _Flat(), SearchState(), rng, batch=1, exploration=0)[0]
             for _ in range(50)]
    assert all(pick == 0 for pick in first), "the strong parent's candidate must win"


def test_exploration_is_a_quota_of_picks_not_noise_on_the_scores():
    """The defect that made a live campaign's reward arm equal its own control.

    Gaussian noise at 0.5 -- the oracle's single-call reproducibility -- was added to
    predictions whose spread across a chosen batch was 0.10 to 0.21, so the exploration
    term decided the batch. A quota keeps the two roles separate: the model's picks are
    its ranking, plainly, and the random picks are visibly random.
    """
    strong = _candidate(0, range(40), parent_score=-11.0)
    middling = [_candidate(i, range(500 * i, 500 * i + 40), parent_score=-8.0)
                for i in range(1, 12)]
    candidates = [strong] + middling
    value = ProgramValue()
    value.fit([c["features"] for c in candidates] * 6, [0.0] * (len(candidates) * 6))
    rng = np.random.default_rng(0)

    # With four of six slots decided by the model, the -11.0 parent's candidate is
    # ranked first on endpoint score and must be taken every time.
    picked = [acquisition(candidates, value, SearchState(), rng, batch=6, exploration=2)
              for _ in range(50)]
    assert all(0 in batch for batch in picked), "the model's own top pick must survive"
    assert all(len(batch) == 6 for batch in picked)

    # And the quota really is spent at random rather than on the ranking's next choices.
    tail = {i for batch in picked for i in batch if i != 0}
    assert len(tail) > 6, "a random quota should reach most of the pool over 50 draws"


def test_an_unfitted_model_defers_entirely_to_the_random_quota():
    candidates = [_candidate(i, range(50 * i, 50 * i + 40)) for i in range(20)]
    rng = np.random.default_rng(1)
    picked = acquisition(candidates, ProgramValue(), SearchState(), rng, batch=8)
    assert len(picked) == len(set(picked)) == 8


def test_endpoint_utility_adds_the_parent_offset_back():
    from compose_v4.control.fiber_control import endpoint_utility

    strong = _candidate(0, range(40), parent_score=-11.0)
    weak = _candidate(1, range(500, 540), parent_score=-7.0)

    class _Flat(ProgramValue):
        def __init__(self):
            super().__init__()
            self.weights, self.n = np.zeros(15), 100

        def predict(self, features):
            return np.zeros(np.asarray(features).shape[0])

    utility = endpoint_utility([strong, weak], _Flat())
    assert utility[0] == pytest.approx(11.0) and utility[1] == pytest.approx(7.0)


# ---- STOP ----


def test_stop_does_not_fire_on_an_unfitted_model():
    # The regression: a campaign terminated at 17 of 72 calls on a 16-observation ridge
    # while -11.00 was known to exist.
    state = SearchState(archive={"a": -9.0}, budget=55)
    state.history = [{"round": r, "improved": False} for r in range(5)]
    value = ProgramValue()
    value.fit([np.ones(15) * 0.1] * 16, [0.0] * 16)
    value.n = 16
    assert not should_stop(state, [_candidate(0, range(10))], value)


def test_stop_needs_patience_as_well_as_a_fitted_model():
    state = SearchState(archive={"a": -9.0}, budget=55)
    rng = np.random.default_rng(2)
    features = [program_features({"parent_score": -9.0}, state) for _ in range(50)]
    value = ProgramValue()
    value.fit(features, list(-1.0 + 0.001 * rng.normal(size=50)))
    candidates = [_candidate(0, range(10))]
    state.history = [{"round": 1, "improved": True}]
    assert not should_stop(state, candidates, value)
    state.history = [{"round": r, "improved": False} for r in range(3)]
    assert should_stop(state, candidates, value)


def test_stop_asks_whether_anything_beats_the_INCUMBENT():
    """Not whether anything beats its own parent -- those are different questions.

    A pool of candidates predicted to land at about -9.0 is worth docking when the
    incumbent is -8.0 and worth nothing when the incumbent is -11.0, and a gain-only
    stop rule reads both cases identically.
    """
    rng = np.random.default_rng(5)
    features = [program_features({"parent_score": -9.0}, SearchState()) for _ in range(50)]
    value = ProgramValue()
    value.fit(features, list(0.001 * rng.normal(size=50)))  # predicts ~zero gain
    candidates = [_candidate(i, range(i, i + 10), parent_score=-9.0) for i in range(4)]
    history = [{"round": r, "improved": False} for r in range(3)]

    behind = SearchState(archive={"a": -8.0}, budget=40)
    behind.history = history
    assert not should_stop(behind, candidates, value)

    ahead = SearchState(archive={"a": -11.0}, budget=40)
    ahead.history = history
    assert should_stop(ahead, candidates, value)


def test_stop_is_immediate_when_there_is_nothing_left():
    state = SearchState(archive={"a": -9.0}, budget=0)
    assert should_stop(state, [_candidate(0, range(10))], ProgramValue())
    assert should_stop(SearchState(archive={"a": -9.0}, budget=9), [], ProgramValue())


# ---- Frontier ----


def test_the_frontier_keeps_the_incumbent_and_still_leaves_it():
    archive = {f"m{i}": -9.0 + 0.1 * i for i in range(20)}
    state = SearchState(archive=archive)
    parents = state.parents(limit=4, rng=np.random.default_rng(0))
    assert parents[0] == "m0"  # the incumbent is always expanded
    ordered = sorted(archive, key=archive.get)
    assert any(p not in ordered[:3] for p in parents), "passive best-first misses branches"


# ---- Horizon ----


def test_the_horizon_is_a_parameter_of_one_code_path():
    fiber = Fiber(JAK2_ROOT, 0.6)
    rng = np.random.default_rng(11)
    with pytest.raises(ValueError):
        expand(JAK2_ROOT, -8.0, fiber, rng, draws=1, horizon=0)
    for horizon in (1, 3):
        found = expand(JAK2_ROOT, -8.0, fiber, rng, draws=2, horizon=horizon)
        assert all(fiber.check(r["smiles"]) is not None for r in found)


# ---- Features ----


def test_features_describe_the_decision_not_the_molecule():
    state = SearchState(archive={"a": -11.0})
    single = program_features({"parent_score": -9.0, "regions": 1, "families": ["base"]}, state)
    joint = program_features({"parent_score": -9.0, "regions": 2, "families": ["scale", "element"]}, state)
    assert single.shape == joint.shape == (15,)
    assert single[1] == pytest.approx(2.0)  # parent minus incumbent
    assert single[6] == 0.0 and joint[6] == 1.0  # the multi-region indicator
