"""Guards for the constructive proposal law and the prospective value model.

Several of these pin bugs that were silent in every headline number they corrupted, so
they are written to fail on the defect rather than to restate the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.control.constructive_features import (
    MODE_FEATURE_NAMES,
    SITE_FEATURE_NAMES,
    mode_identity,
    mode_matrix,
    site_features,
)
from compose_v4.control.constructive_policy import (
    PolicyShape,
    fit,
    objective,
    pair_scores,
    rank_of_truth,
    unpack,
)
from compose_v4.control.prospective_value import (
    capture_at_fraction,
    fit_logistic,
    fold_averaged,
    roc_auc,
    spearman,
    standardize,
)


def _benzene_with_a_pendant():
    """Toluene as a padded state: a six-ring plus one exocyclic carbon."""
    bonds = [[0, 1, 2], [1, 2, 1], [2, 3, 2], [3, 4, 1], [4, 5, 2], [5, 0, 1], [0, 6, 1]]
    return {
        "atom_types": [2] * 7 + [0] * 5,
        "bonds": bonds,
        "formal_charges": [0] * 12,
        "implicit_h_counts": [0, 1, 1, 1, 1, 1, 3] + [0] * 5,
        "n_slots": 12,
    }


def _coupled_examples(rng, n=400):
    """Data whose site and mode choices are genuinely coupled, not independent.

    The true law puts mass on (site feature 0 with mode feature 0) and on
    (site feature 1 with mode feature 1) -- a pattern no additive model can express,
    so a fit that leaves the interaction at zero will rank the truth no better than
    the additive one.
    """
    examples = []
    for _ in range(n):
        n_sites, n_modes = 6, 4
        sites = rng.normal(size=(n_sites, 3))
        modes = rng.normal(size=(n_modes, 3))
        truth = sites[:, :2] @ np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])[:, :2] @ modes[:, :2].T
        flat = int(np.argmax(truth.reshape(-1)))
        examples.append(
            {
                "sites": sites,
                "modes": modes,
                "site_index": flat // n_modes,
                "mode_index": flat % n_modes,
            }
        )
    return examples


# ---- The proposal law ----


def test_the_analytic_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    shape = PolicyShape(5, 4)
    examples = [
        {
            "sites": rng.normal(size=(6, 5)),
            "modes": rng.normal(size=(3, 4)),
            "site_index": 2,
            "mode_index": 1,
        }
    ]
    penalties = {"site": 1e-3, "mode": 1e-3, "interaction": 0.5}
    theta = rng.normal(size=shape.size)
    _, analytic = objective(theta, shape, examples, penalties)
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = 1e-6
        high, _ = objective(theta + step, shape, examples, penalties)
        low, _ = objective(theta - step, shape, examples, penalties)
        assert abs((high - low) / 2e-6 - analytic[i]) < 1e-6


def test_the_interaction_is_actually_fitted_and_not_left_at_the_origin():
    """Regression: a rank-k factorisation W = U^T V has a stationary point at U = V = 0.

    Fitted from a small random start the factors stayed at 4e-5, silently reducing the
    law to A + B -- the independent-marginal model the design exists to avoid -- and the
    held-out scores were byte-identical across rank 1 to 4 and two orders of penalty, so
    nothing downstream revealed it. The full-matrix form is convex and cannot do this.
    """
    rng = np.random.default_rng(1)
    examples = _coupled_examples(rng)
    model = fit(
        examples, PolicyShape(3, 3), penalties={"site": 1e-3, "mode": 1e-3, "interaction": 1e-2}
    )
    assert model["interaction_norm"] > 0.1


def test_the_interaction_earns_its_keep_on_genuinely_coupled_data():
    """The additive model must be beaten where the truth is a coupling, or the term is decorative."""
    rng = np.random.default_rng(2)
    examples = _coupled_examples(rng)
    train, test = examples[:300], examples[300:]
    shape = PolicyShape(3, 3)
    coupled = fit(train, shape, penalties={"site": 1e-3, "mode": 1e-3, "interaction": 1e-2})
    additive = fit(train, shape, penalties={"site": 1e-3, "mode": 1e-3, "interaction": 1e7})
    coupled_rank = np.mean([rank_of_truth(coupled, e)["joint_rank"] for e in test])
    additive_rank = np.mean([rank_of_truth(additive, e)["joint_rank"] for e in test])
    assert coupled_rank < additive_rank


def test_the_fit_is_deterministic_because_the_objective_is_convex():
    rng = np.random.default_rng(3)
    examples = _coupled_examples(rng, n=60)
    shape = PolicyShape(3, 3)
    first = fit(examples, shape)
    second = fit(examples, shape)
    assert np.allclose(first["theta"], second["theta"])


def test_the_interaction_may_not_be_shrunk_more_softly_than_the_pooled_terms():
    with pytest.raises(ValueError, match="at least as hard"):
        fit(
            _coupled_examples(np.random.default_rng(4), n=20),
            PolicyShape(3, 3),
            penalties={"site": 1.0, "mode": 1.0, "interaction": 1e-6},
        )


def test_pair_scores_decompose_exactly_into_the_three_declared_terms():
    rng = np.random.default_rng(5)
    shape = PolicyShape(4, 3)
    theta = rng.normal(size=shape.size)
    sites, modes = rng.normal(size=(5, 4)), rng.normal(size=(2, 3))
    a, b, w = unpack(theta, shape)
    expected = (sites @ a)[:, None] + (modes @ b)[None, :] + sites @ w @ modes.T
    assert np.allclose(pair_scores(theta, shape, sites, modes), expected)


# ---- Features ----


def test_ring_atoms_are_distinguished_from_the_pendant_substituent():
    matrix, slots = site_features(_benzene_with_a_pendant())
    index = {name: i for i, name in enumerate(SITE_FEATURE_NAMES)}
    assert slots == list(range(7))
    in_ring = matrix[:, index["in_ring"]]
    assert in_ring[:6].all(), "the six aromatic carbons are on a cycle"
    assert not in_ring[6], "the methyl carbon is not"
    assert matrix[6, index["ring_adjacent"]] == 1.0


def test_padding_and_scar_slots_are_not_offered_as_attachment_sites():
    state = _benzene_with_a_pendant()
    state["atom_types"] = [2] * 7 + [11] + [0] * 4  # a SCAR slot among the padding
    _, slots = site_features(state)
    assert slots == list(range(7))


def test_the_mode_vocabulary_is_the_medium_granularity_it_claims():
    mode = {
        "attachment_count": 1,
        "created_atoms": 4,
        "closes_ring": True,
        "opens_ring": False,
        "primitive_count": 9,
    }
    assert mode_identity(mode) == (1, 4, True)
    assert mode_matrix([mode]).shape == (1, len(MODE_FEATURE_NAMES))


# ---- Metrics ----


def test_roc_auc_agrees_with_the_rank_definition_and_handles_ties():
    assert roc_auc([3.0, 2.0, 1.0], [1, 0, 0]) == pytest.approx(1.0)
    assert roc_auc([1.0, 2.0, 3.0], [1, 0, 0]) == pytest.approx(0.0)
    assert roc_auc([1.0, 1.0], [1, 0]) == pytest.approx(0.5), "a tie is half a win"


def test_capture_measures_realized_mass_not_count():
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    realized = np.array([9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert capture_at_fraction(scores, realized, 0.1) == pytest.approx(0.9)


def test_fitting_one_feature_reproduces_that_features_own_ranking():
    """Regression: a rank metric is invariant to any monotone transform of the score.

    Fitting a logistic on a single feature IS such a transform, so its AUC must equal
    the raw feature's. Pooling per-fold scores broke this -- each fold standardises and
    fits separately, so concatenated scores sit on incomparable scales -- and it
    depressed every fitted arm while leaving raw columns alone, producing three readings
    where a model scored below a feature it contains.
    """
    rng = np.random.default_rng(6)
    feature = rng.normal(size=400)
    labels = (feature + rng.normal(scale=0.5, size=400) > 0).astype(int)
    design = np.column_stack([feature, np.ones_like(feature)])
    (scaled,) = standardize(design)
    fitted = fit_logistic(scaled, labels, penalty=1e-4)
    assert roc_auc(fitted.score(scaled), labels) == pytest.approx(
        roc_auc(feature, labels), abs=1e-9
    )


def test_fold_averaging_never_compares_two_folds_scores_against_each_other():
    combined = fold_averaged([(100, 0.9), (100, 0.5)])
    assert combined["value"] == pytest.approx(0.7)
    assert combined["per_fold_min"] == pytest.approx(0.5)
    weighted = fold_averaged([(300, 0.9), (100, 0.5)])
    assert weighted["value"] == pytest.approx(0.8)


def test_spearman_is_zero_for_a_column_with_no_variation():
    assert spearman(np.ones(10), np.arange(10.0)) == 0.0
