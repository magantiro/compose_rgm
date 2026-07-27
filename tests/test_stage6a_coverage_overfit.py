"""Stage 6A regression: every production-enabled operator family has POSITIVE current-recipe supervision,
and the model can overfit a verified operator-balanced coverage corpus (a failure to overfit is a NO-GO).

Exercises the real coverage builder + tiny-overfit from scripts/stage6a_coverage_smoke.py (a shorter
overfit here for speed). It checks that every enabled family has >=1 positive teacher target, a nonzero
family-head gradient, forced-sampling reachability, no masked target, and that a short overfit reduces the
loss substantially while raising every family's selected-mark probability.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from stage6a_coverage_smoke import ENABLED, run  # noqa: E402


def test_every_enabled_family_supervised_and_corpus_overfits() -> None:
    contract, overfit = run(overfit_steps=350, write=False)

    supervised = {row["family"] for row in contract if row["status"] == "SUPERVISED_CURRENT_RECIPE"}
    assert supervised == set(ENABLED), f"not every enabled family is supervised: {set(ENABLED) - supervised}"
    for row in contract:
        if row["status"] != "SUPERVISED_CURRENT_RECIPE":
            continue
        assert row["positive_training_targets"] >= 1, row["family"]
        assert row["gradient_observed"], f"{row['family']}: no positive-target family-head gradient"
        assert row["forced_sampling_reaches"], f"{row['family']}: not reachable under forced sampling"

    assert not overfit["missing_families"]
    # substantial loss decrease (converges well below 0.6x the initial by ~250 steps) and every family rises.
    assert overfit["loss_final"] < 0.6 * overfit["loss_initial"], overfit
    assert overfit["every_family_probability_rose"]
    assert overfit["gap_above_lower_bound"] < 0.3, overfit  # near the Poisson-Bregman floor
