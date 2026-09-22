"""Guards for the donor-transplant feasibility driver.

The measurement's whole claim rests on two properties of the cut draw: that the retentive
arm RE-RANKS a support it shares with the uniform arm rather than filtering it, and that
the floor it promises actually binds somewhere.  A floor asserted but never exercised is
the failure this repository has recorded twice, so it is tested where it binds.

The information regime is also a tested property, not a comment: the driver must read a
CHARGED run's ledger and must not read the prescreen bank, whose scores come from ~249,455
uncounted oracle calls.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _driver():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "pmo_donor_transplant_feasibility",
        ROOT / "scripts/pmo_donor_transplant_feasibility.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cuts(sizes: list[int]):
    from compose_v4.control.donor_program import PendantCut

    return tuple(PendantCut(0, 1, tuple(range(2, 2 + size))) for size in sizes)


def test_retentive_shares_the_uniform_support_and_never_filters() -> None:
    driver = _driver()
    cuts = _cuts([1, 3, 8, 20, 34])
    uniform = driver.cut_weights(cuts, arm="uniform", n_real=35)
    retentive = driver.cut_weights(cuts, arm="retentive", n_real=35)
    assert (uniform > 0).all()
    # Every cut the uniform arm can draw is still reachable. A difference between the arms
    # therefore cannot be an artifact of one arm reaching molecules the other cannot.
    assert (retentive > 0).all()
    assert np.isclose(retentive.sum(), 1.0)
    assert np.isclose(uniform.sum(), 1.0)


def test_the_floor_actually_binds_on_a_large_release() -> None:
    """A floor is only tested where it binds.

    `exp(-released / 0.25)` underflows the 0.05 floor for any release above ~0.75, so the
    34-of-35 cut is carried by the floor and by nothing else.
    """
    driver = _driver()
    cuts = _cuts([1, 34])
    retentive = driver.cut_weights(cuts, arm="retentive", n_real=35)
    raw = np.exp(-np.asarray([1 / 35, 34 / 35]) / 0.25)
    assert raw[1] < 0.05, "fixture no longer exercises the floor"
    # The floored weight is strictly greater than the un-floored one would have been.
    unfloored = raw / raw.sum()
    assert retentive[1] > unfloored[1]


def test_retentive_prefers_small_releases_and_uniform_does_not() -> None:
    driver = _driver()
    cuts = _cuts([1, 34])
    uniform = driver.cut_weights(cuts, arm="uniform", n_real=35)
    retentive = driver.cut_weights(cuts, arm="retentive", n_real=35)
    assert uniform[0] == pytest.approx(uniform[1]), "the uniform arm must not prefer anything"
    assert retentive[0] > retentive[1], "the retentive arm must prefer the small release"


def test_the_ledger_is_a_charged_run_and_not_the_prescreen_bank() -> None:
    driver = _driver()
    assert driver.LEDGER.name == "pmo_3x250_autopsy_v1.json"
    assert "banks" not in driver.LEDGER.name
    source = (ROOT / "scripts/pmo_donor_transplant_feasibility.py").read_text()
    # The prescreen bank must not be reachable from this driver at all.
    assert "pmo_banks_all" not in source.replace(
        "`diagnostics/pmo_banks_all.json`", ""
    ), "the driver must not read the prescreen bank"
    assert driver.SCORE_KEY == "charged_score"


def test_ledger_rows_are_charged_scored_and_ranked() -> None:
    driver = _driver()
    rows = driver.ledger_rows()
    assert len(rows) == 250
    assert all(row[driver.SCORE_KEY] is not None for row in rows)
    scores = [row[driver.SCORE_KEY] for row in rows]
    assert scores == sorted(scores, reverse=True), "the pool must be the top-scoring rows"


def test_summarize_counts_retention_over_the_right_threshold() -> None:
    driver = _driver()
    arm = {
        "arm": "t",
        "attempts": 4,
        "statuses": {"compiled": 3},
        "rows": [
            {"smiles": "CCO", "retained_fraction": 0.50, "steps": 3,
             "descriptors": {"qed": 0.4, "sa": 3.0, "heavy_atoms": 3}},
            {"smiles": "CCN", "retained_fraction": 0.49, "steps": 5,
             "descriptors": {"qed": 0.5, "sa": 2.0, "heavy_atoms": 3}},
            {"smiles": "CCC", "retained_fraction": 0.90, "steps": 7,
             "descriptors": {"qed": 0.6, "sa": 1.0, "heavy_atoms": 3}},
        ],
    }
    report = driver.summarize(arm, {"CCO"})
    # 0.50 is retained AT LEAST half, 0.49 is not.
    assert report["retained_at_least_half"] == 2
    assert report["novel_vs_scored_set"] == 2, "a product already in the scored set is not novel"
    assert report["compiled"] == 3
    assert report["compiled_rate"] == pytest.approx(0.75)
    assert report["retained_fraction"]["median"] == pytest.approx(0.50)


def test_the_compiler_preserves_the_retained_core_by_construction() -> None:
    """The property that separates this compiler from the refuted generic one.

    `compile_source_to_target` demolishes to null (retained_fraction 0.000 on 4 of 4 real
    pairs). `compile_transplant` cannot: it raises if replay changes a retained slot. This
    drives the LIVE function rather than re-deriving its guarantee.
    """
    from compose_v4.control.donor_program import compile_transplant, pendant_cuts
    from compose_v4.experiments.editing_v2_evaluation_semantics import (
        production_state_from_smiles,
    )

    source = production_state_from_smiles("CC(=O)Nc1ccc(O)cc1", max_atoms=48)
    donor = production_state_from_smiles("COc1ccc(CCN)cc1", max_atoms=48)
    source_cuts, donor_cuts = pendant_cuts(source), pendant_cuts(donor)
    assert source_cuts and donor_cuts

    compiled = 0
    for a in source_cuts:
        for b in donor_cuts:
            result = compile_transplant(source, donor, a, b)
            if result.get("status") != "compiled":
                continue
            compiled += 1
            # Preservation is by construction, so retention is strictly positive: a route
            # through the null state is impossible here by the compiler's own assertion.
            assert result["released_fraction"] < 1.0
            assert 0.0 < 1.0 - result["released_fraction"] <= 1.0
    assert compiled > 0, "the fixture must reach the compiled branch or it guards nothing"
