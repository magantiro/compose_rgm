"""Invariants for the online surrogate-guided PMO oracle allocator.

The load-bearing property is the NO-PRESCREEN rule: the surrogate may train only
on evaluations the current run has already purchased.  Several tests below exist
purely to make a violation of that rule fail loudly rather than silently produce
a better-looking benchmark number.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from compose_v4.experiments.pmo_surrogate_allocator import (
    SURROGATE_FACTORIES,
    BatchDecision,
    MorganFeaturizer,
    OnlineArchive,
    Prediction,
    SurrogateAllocator,
    acquisition_scores,
    build_surrogate,
    linear_kernel,
    tanimoto_kernel,
)

_SMILES = [
    "CCO", "CCCO", "c1ccccc1", "CC(=O)O", "CCN", "c1ccncc1",
    "CCCCCC", "CC(C)C", "CCOCC", "c1ccc(O)cc1", "CCCCO", "CN(C)C",
    "c1ccc(N)cc1", "CC(=O)N", "CCCCCCO", "c1ccc(C)cc1",
]


def _features(smiles=None):
    featurizer = MorganFeaturizer()
    X, kept = featurizer.featurize(smiles or _SMILES)
    return X, kept


# ---- Archive / no-prescreen ----


def test_archive_requires_provenance_for_every_purchased_row():
    archive = OnlineArchive()
    with pytest.raises(ValueError):
        archive.observe("CCO", 0.5, provenance="", step=0)
    assert len(archive) == 0


def test_archive_rejects_non_finite_score():
    archive = OnlineArchive()
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            archive.observe("CCO", bad, provenance="run", step=0)


def test_archive_deduplicates_and_reports_provenance():
    archive = OnlineArchive()
    assert archive.observe("CCO", 0.5, provenance="init", step=0) is True
    assert archive.observe("CCO", 0.9, provenance="round1", step=1) is False
    assert archive.observe("CCN", 0.2, provenance="round1", step=1) is True
    assert len(archive) == 2
    assert archive.training_provenance() == {"init": 1, "round1": 1}
    assert archive.best == pytest.approx(0.5)


def test_archive_has_no_bulk_loader_so_a_prescreen_cannot_slip_in():
    """Every row must pass through observe(); there is no path that imports a
    scored external database wholesale."""
    forbidden = {"load", "from_file", "extend", "load_csv", "from_dataframe"}
    assert forbidden.isdisjoint(set(dir(OnlineArchive)))
    # observe is the single mutator that appends to rows.
    assert OnlineArchive().rows == []


# ---- Kernels ----


def test_tanimoto_kernel_is_bounded_symmetric_and_unit_diagonal():
    X, _ = _features()
    K = tanimoto_kernel(X, X)
    assert K.shape == (len(X), len(X))
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    assert K.min() >= 0.0 and K.max() <= 1.0 + 1e-12


def test_tanimoto_of_all_zero_rows_is_one_not_nan():
    zero = np.zeros((2, 16))
    assert np.allclose(tanimoto_kernel(zero, zero), 1.0)


def test_linear_kernel_is_cosine_normalised():
    """Regression: dividing by the bit count instead sinks the signal below the
    noise floor and the GP returns its prior for every candidate."""
    X, _ = _features()
    K = linear_kernel(X, X)
    assert np.allclose(np.diag(K), 1.0)
    assert K.max() <= 1.0 + 1e-9


# ---- Surrogates ----


@pytest.mark.parametrize("name", sorted(SURROGATE_FACTORIES))
def test_surrogate_fit_predict_shapes_and_positive_sigma(name):
    X, _ = _features()
    y = np.linspace(0.05, 0.95, len(X))
    pred = build_surrogate(name).fit(X[:10], y[:10]).predict(X[10:])
    assert isinstance(pred, Prediction)
    assert pred.mean.shape == (len(X) - 10,)
    assert pred.sigma.shape == (len(X) - 10,)
    assert np.all(pred.sigma > 0.0)
    assert np.isfinite(pred.mean).all() and np.isfinite(pred.sigma).all()


@pytest.mark.parametrize("name", sorted(SURROGATE_FACTORIES))
def test_surrogate_survives_a_degenerate_constant_archive(name):
    """All-identical scores must not produce a zero-variance divide."""
    X, _ = _features()
    y = np.full(10, 0.4)
    pred = build_surrogate(name).fit(X[:10], y[:10]).predict(X[10:])
    assert np.isfinite(pred.mean).all()
    assert np.all(pred.sigma > 0.0)


def test_predict_before_fit_raises():
    X, _ = _features()
    for name in ("gp_tanimoto", "knn_tanimoto", "bagged_linear"):
        with pytest.raises(RuntimeError):
            build_surrogate(name).predict(X)


def test_mean_surrogate_is_flat_and_structural_models_are_not():
    """The null control must ignore structure; a structural model must not."""
    X, _ = _features()
    y = np.asarray([1.0 if "c1" in s else 0.0 for s in _SMILES])
    flat = build_surrogate("mean").fit(X[:12], y[:12]).predict(X[12:])
    assert np.allclose(flat.mean, flat.mean[0])
    structural = build_surrogate("gp_tanimoto").fit(X[:12], y[:12]).predict(X[12:])
    assert structural.mean.std() > 0.0


def test_structural_surrogate_beats_null_on_a_learnable_signal():
    """Aromatic molecules score high; a Tanimoto GP should recover that from a
    handful of points while the mean model cannot."""
    X, _ = _features()
    y = np.asarray([0.9 if "c1" in s else 0.1 for s in _SMILES])
    train, test = slice(0, 12), slice(12, None)
    gp = build_surrogate("gp_tanimoto").fit(X[train], y[train]).predict(X[test])
    null = build_surrogate("mean").fit(X[train], y[train]).predict(X[test])
    gp_mae = np.abs(gp.mean - y[test]).mean()
    null_mae = np.abs(null.mean - y[test]).mean()
    assert gp_mae < null_mae


def test_unknown_surrogate_name_raises():
    with pytest.raises(ValueError):
        build_surrogate("xgboost_definitely_not_installed")


# ---- Acquisition ----


def test_ucb_with_zero_beta_equals_greedy():
    pred = Prediction(np.asarray([0.1, 0.5]), np.asarray([0.3, 0.05]))
    assert np.allclose(
        acquisition_scores(pred, kind="ucb", beta=0.0),
        acquisition_scores(pred, kind="greedy"),
    )


def test_ucb_beta_promotes_the_uncertain_candidate():
    """The whole point of the uncertainty term: it must be able to overturn the
    predicted-mean ordering, or the acquisition can collapse onto a false
    predictor maximum."""
    pred = Prediction(np.asarray([0.50, 0.45]), np.asarray([0.01, 0.40]))
    assert np.argmax(acquisition_scores(pred, kind="ucb", beta=0.0)) == 0
    assert np.argmax(acquisition_scores(pred, kind="ucb", beta=2.0)) == 1


def test_expected_improvement_is_non_negative():
    pred = Prediction(np.asarray([0.1, 0.9]), np.asarray([0.2, 0.2]))
    assert np.all(acquisition_scores(pred, kind="ei", best=0.5) >= 0.0)


def test_unknown_acquisition_raises():
    pred = Prediction(np.asarray([0.1]), np.asarray([0.1]))
    with pytest.raises(ValueError):
        acquisition_scores(pred, kind="not_an_acquisition")


# ---- Allocator ----


def _seeded_archive(n, scores=None):
    archive = OnlineArchive()
    for i in range(n):
        value = 0.5 if scores is None else scores[i]
        archive.observe(_SMILES[i], value, provenance="init", step=0)
    return archive


def test_cold_start_reports_inactive_surrogate_and_does_not_rank():
    allocator = SurrogateAllocator(min_fit_rows=8, seed=0)
    archive = _seeded_archive(3)
    decision = allocator.select_batch(archive, _SMILES[3:], k=2)
    assert isinstance(decision, BatchDecision)
    assert decision.surrogate_active is False
    assert decision.exploit_slots == 0
    assert len(decision.selected) == 2


def test_surrogate_activates_once_enough_rows_are_purchased():
    allocator = SurrogateAllocator(min_fit_rows=8, explore_fraction=0.0, seed=0)
    y = [0.9 if "c1" in s else 0.1 for s in _SMILES]
    archive = _seeded_archive(10, y)
    decision = allocator.select_batch(archive, _SMILES[10:], k=2)
    assert decision.surrogate_active is True
    assert decision.exploit_slots == 2


def test_exploration_floor_reserves_at_least_one_call():
    """A confident-but-wrong surrogate must never take the whole batch."""
    y = [0.9 if "c1" in s else 0.1 for s in _SMILES]
    archive = _seeded_archive(10, y)
    for k in (2, 4, 8):
        allocator = SurrogateAllocator(explore_fraction=0.01, min_fit_rows=8, seed=0)
        decision = allocator.select_batch(archive, _SMILES[10:] * 4, k=k)
        assert decision.explore_slots >= 1
        assert decision.exploit_slots == k - decision.explore_slots


def test_zero_exploration_fraction_is_honoured_exactly():
    y = [0.9 if "c1" in s else 0.1 for s in _SMILES]
    archive = _seeded_archive(10, y)
    allocator = SurrogateAllocator(explore_fraction=0.0, min_fit_rows=8, seed=0)
    decision = allocator.select_batch(archive, _SMILES[10:] * 4, k=4)
    assert decision.explore_slots == 0


def test_already_purchased_candidates_are_never_reselected():
    """Re-charging a molecule the run already paid for wastes scarce budget."""
    archive = _seeded_archive(10)
    allocator = SurrogateAllocator(min_fit_rows=8, seed=0)
    decision = allocator.select_batch(archive, _SMILES, k=4)
    for index in decision.selected:
        assert not archive.contains(_SMILES[index])


def test_empty_pool_returns_no_selection_rather_than_raising():
    archive = _seeded_archive(12)
    allocator = SurrogateAllocator(min_fit_rows=8, seed=0)
    decision = allocator.select_batch(archive, _SMILES[:12], k=4)
    assert decision.selected == []


def test_selection_is_deterministic_under_a_fixed_seed():
    y = [0.9 if "c1" in s else 0.1 for s in _SMILES]
    archive = _seeded_archive(10, y)
    pool = _SMILES[10:] * 3
    first = SurrogateAllocator(min_fit_rows=8, seed=7).select_batch(archive, pool, k=3)
    second = SurrogateAllocator(min_fit_rows=8, seed=7).select_batch(archive, pool, k=3)
    assert first.selected == second.selected


def test_selection_reports_its_own_overhead_and_predictions():
    y = [0.9 if "c1" in s else 0.1 for s in _SMILES]
    archive = _seeded_archive(10, y)
    allocator = SurrogateAllocator(min_fit_rows=8, explore_fraction=0.0, seed=0)
    decision = allocator.select_batch(archive, _SMILES[10:], k=2)
    assert decision.rank_seconds >= 0.0
    assert len(decision.predicted_mean) == len(decision.selected)
    assert len(decision.predicted_sigma) == len(decision.selected)


def test_unparseable_smiles_are_dropped_not_scored_as_zero():
    featurizer = MorganFeaturizer()
    X, kept = featurizer.featurize(["CCO", "not_a_molecule", "c1ccccc1"])
    assert kept == [0, 2]
    assert X.shape[0] == 2


def test_allocator_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        SurrogateAllocator(explore_fraction=1.5)
    with pytest.raises(ValueError):
        SurrogateAllocator(min_fit_rows=1)
    with pytest.raises(ValueError):
        SurrogateAllocator(diversity_penalty=-0.1)
    with pytest.raises(ValueError):
        SurrogateAllocator().select_batch(OnlineArchive(), ["CCO"], k=0)


def test_diversity_penalty_reduces_within_batch_similarity():
    """Plain top-k on a fingerprint acquisition returns near-duplicates and
    wastes a scarce batch on one chemical neighbourhood."""
    pool = [
        "c1ccc(O)cc1", "c1ccc(O)cc1C", "c1ccc(O)cc1CC", "c1ccc(O)cc1CCC",
        "CCCCCCCCN", "O=C(O)CCCC", "C1CCNCC1", "CCOC(=O)C",
    ]
    y = [0.9 if "c1" in s else 0.1 for s in _SMILES]
    archive = _seeded_archive(12, y)
    featurizer = MorganFeaturizer()

    def mean_pair_similarity(selected):
        X, _ = featurizer.featurize([pool[i] for i in selected])
        K = tanimoto_kernel(X, X)
        off = K[~np.eye(len(X), dtype=bool)]
        return float(off.mean())

    plain = SurrogateAllocator(
        min_fit_rows=8, explore_fraction=0.0, diversity_penalty=0.0, seed=0
    ).select_batch(archive, pool, k=3)
    diverse = SurrogateAllocator(
        min_fit_rows=8, explore_fraction=0.0, diversity_penalty=1.0, seed=0
    ).select_batch(archive, pool, k=3)
    assert mean_pair_similarity(diverse.selected) < mean_pair_similarity(plain.selected)


# ---- Acquisition engagement (regression against a documented failure) ----


def _incumbent_setup():
    """An archive whose best value is far above the rest -- the regime in which
    an averaging surrogate can no longer nominate an improving candidate."""
    X, _ = _features()
    y = np.full(len(_SMILES), 0.10)
    y[2] = 0.72  # one strong molecule, as in the recorded jnk3/gsk3b archives
    model = build_surrogate("knn_tanimoto").fit(X[:12], y[:12])
    return model.predict(X[12:]), float(y[:12].max())


def test_averaging_surrogate_cannot_exceed_the_incumbent_on_its_mean():
    """diagnostics/task3_surrogate_acquisition_kind_mismatch.json: a convex
    combination of observed labels cannot exceed the largest of them, so a
    greedy/EI-style absolute test against the incumbent cannot fire."""
    pred, incumbent = _incumbent_setup()
    assert pred.mean.max() <= incumbent + 1e-9
    from compose_v4.experiments.pmo_surrogate_allocator import engagement_fraction

    assert engagement_fraction(pred, incumbent, kind="greedy") == 0.0


def test_uncertainty_term_restores_engagement():
    """The fix recorded alongside that failure: an optimistic estimate lets a
    candidate whose neighbours disagree clear the incumbent."""
    from compose_v4.experiments.pmo_surrogate_allocator import engagement_fraction

    pred, incumbent = _incumbent_setup()
    low = engagement_fraction(pred, incumbent, kind="ucb", beta=0.0)
    high = engagement_fraction(pred, incumbent, kind="ucb", beta=3.0)
    assert low == 0.0
    assert high > 0.0


def test_beta_is_selected_by_engagement_not_by_outcome():
    from compose_v4.experiments.pmo_surrogate_allocator import select_beta_by_engagement

    pred, incumbent = _incumbent_setup()
    report = select_beta_by_engagement(pred, incumbent, threshold=0.5)
    assert set(report) == {
        "sweep", "threshold", "selected_beta", "cleared_threshold", "selection_rule",
    }
    sweep = report["sweep"]
    betas = sorted(sweep)
    # Engagement is monotone non-decreasing in beta, which is what makes
    # "smallest beta that clears" well defined.
    assert all(sweep[a] <= sweep[b] + 1e-12 for a, b in itertools.pairwise(betas))
    if report["cleared_threshold"]:
        assert sweep[report["selected_beta"]] >= report["threshold"]
        smaller = [b for b in betas if b < report["selected_beta"]]
        assert all(sweep[b] < report["threshold"] for b in smaller)


def test_engagement_rejects_unknown_acquisition():
    from compose_v4.experiments.pmo_surrogate_allocator import engagement_fraction

    pred, incumbent = _incumbent_setup()
    with pytest.raises(ValueError):
        engagement_fraction(pred, incumbent, kind="hypervolume")
