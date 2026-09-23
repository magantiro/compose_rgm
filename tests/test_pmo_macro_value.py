"""The PMO reward target, and the ways it must DISAGREE with parent-relative improvement."""

from __future__ import annotations

import math

import numpy as np

from compose_v4.control.pmo_macro_value import (
    FEATURE_NAMES,
    MacroValue,
    archive_gain,
    archive_threshold,
    macro_features,
)

FULL_ARCHIVE = [(f"m{i}", 0.40 + 0.01 * i) for i in range(10)]  # top ten spans 0.40 .. 0.49


def test_a_candidate_below_the_frontier_is_worth_nothing():
    """The whole point of the target: not beating the archive means adding nothing."""
    assert archive_gain(FULL_ARCHIVE, 0.30) == 0.0
    assert archive_gain(FULL_ARCHIVE, 0.39) == 0.0
    assert archive_gain(FULL_ARCHIVE, 0.55) > 0.0


def test_archive_gain_disagrees_with_parent_relative_delta():
    """The case that motivated the new target, stated as an executable contrast.

    A improves a strong parent slightly and enters the frontier. B improves a weak parent
    enormously and lands below the tenth best. Parent-relative learning prefers B by a wide
    margin; archive gain prefers A, and the benchmark scores the archive.
    """
    a_parent, a_child = 0.40, 0.505
    b_parent, b_child = 0.25, 0.39
    assert (b_child - b_parent) > (a_child - a_parent), "B wins on parent-relative delta"
    assert archive_gain(FULL_ARCHIVE, a_child) > archive_gain(FULL_ARCHIVE, b_child), (
        "archive gain must prefer the candidate that actually moves the frontier"
    )
    assert archive_gain(FULL_ARCHIVE, b_child) == 0.0


def test_an_empty_archive_gives_a_candidate_its_full_value():
    assert archive_gain([], 0.30) == 0.30


def test_threshold_is_open_until_the_archive_is_full():
    assert archive_threshold(FULL_ARCHIVE[:3], k=10) == -math.inf
    assert math.isclose(archive_threshold(FULL_ARCHIVE, k=10), 0.40, rel_tol=1e-9)


def test_features_describe_the_macro_not_only_the_molecule():
    """Two candidates with the SAME endpoint score but different macros must differ."""
    base = {
        "provenance": {
            "parent_measured_score": 0.25,
            "planner_channel": "structured",
            "retention_realized": 0.6,
            "retention_target": 0.6,
        },
        "program_size": {"primitive_edits": 14, "block_count": 3, "delta_from_measured_parent": -5},
        "program_rule_histogram": {"cycle_close": 2, "atom_insert": 4, "atom_delete": 9},
    }
    small = {
        "provenance": dict(base["provenance"], planner_channel="shallow"),
        "program_size": {"primitive_edits": 3, "block_count": 1, "delta_from_measured_parent": 1},
        "program_rule_histogram": {"atom_insert": 1},
    }
    big = macro_features(base, observations=FULL_ARCHIVE)
    little = macro_features(small, observations=FULL_ARCHIVE)
    assert len(big) == len(FEATURE_NAMES)
    assert not np.allclose(big, little), "a 14-primitive ring macro must not look like a 3-edit growth"
    names = dict(zip(FEATURE_NAMES, big, strict=True))
    assert names["primitive_edits_scaled"] > dict(zip(FEATURE_NAMES, little, strict=True))[
        "primitive_edits_scaled"
    ]
    assert names["is_structured_channel"] == 1.0 and names["is_shallow_channel"] == 0.0


def test_uncertainty_is_wide_without_evidence_and_narrows_with_it():
    """Exploration must come from what the model does not know, not a fixed constant."""
    model = MacroValue()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, len(FEATURE_NAMES)))
    y = x @ np.linspace(0.0, 1.0, len(FEATURE_NAMES))
    probe = x[:5]
    unfitted = model.spread(probe)
    assert np.all(unfitted > 0), "an unfitted model must still offer a positive bonus"

    model.fit(x[:8], y[:8])
    thin = float(np.mean(model.spread(probe)))
    model.fit(x, y)
    thick = float(np.mean(model.spread(probe)))
    assert thick < thin, "the bonus must shrink as evidence accumulates"


def test_a_constant_bonus_would_not_reorder_but_uncertainty_does():
    """Why the shipped optimism constant was inert: it applied to nearly every candidate.

    Adding the same number to every score cannot change a ranking. A spread that varies by
    candidate can, and must.
    """
    model = MacroValue()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, len(FEATURE_NAMES)))
    model.fit(x, x @ np.linspace(0.0, 1.0, len(FEATURE_NAMES)))
    probe = rng.normal(size=(12, len(FEATURE_NAMES))) * 3.0  # off-support, so spread varies
    spreads = model.spread(probe)
    assert spreads.std() > 0, "predictive spread must differ between candidates"
    plain = np.argsort(-model.predict(probe))
    explored = np.argsort(-model.acquire(probe, beta=5.0))
    assert not np.array_equal(plain, explored), "uncertainty must be able to reorder candidates"
