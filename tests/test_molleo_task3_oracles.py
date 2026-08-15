"""The frozen Task 3 oracles. An oracle that is wrong must fail here, loudly.

The expensive checks (all 400 reference molecules, both RDKit environments) live
in `scripts/molleo_task3_oracle_parity.py`. What is pinned HERE is everything
that could silently change under an edit: the objective orientation, the
transforms, the checksums, and the counting rule.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from compose_v4.benchmark.molleo_task3 import (
    N_OBJECTIVES,
    BudgetExceeded,
    OracleMeter,
)
from compose_v4.benchmark.oracles import (
    DEFAULT_BUNDLE_DIR,
    WORST_VECTOR,
    FrozenForest,
    RawScores,
    Task3Objectives,
    canonical,
)

pytestmark = pytest.mark.skipif(
    not (DEFAULT_BUNDLE_DIR / "molleo_task3_oracle_manifest.json").exists(),
    reason="frozen oracle bundle not present; run molleo_task3_oracle_extract.py")

#: The panel is 400 molecules; a slice keeps the suite fast while still covering
#: ZINC molecules and known actives of both kinases.
PANEL_SLICE = 60


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(
        (DEFAULT_BUNDLE_DIR / "molleo_task3_oracle_manifest.json").read_text())


@pytest.fixture(scope="module")
def oracles() -> Task3Objectives:
    return Task3Objectives()


def _panel(manifest: dict, group: str | None = None) -> list[dict]:
    rows = manifest["reference_scores"]
    if group is not None:
        rows = [r for r in rows if r["group"] == group]
    return rows


def test_the_forests_reproduce_the_original_estimators(manifest, oracles):
    """Bit-for-bit against sklearn's predict_proba, not merely close to it."""
    rows = _panel(manifest)[:PANEL_SLICE]
    scored = oracles.raw_many([r["smiles"] for r in rows])
    for name in ("jnk3", "gsk3b", "drd2"):
        expected = np.array([r[name] for r in rows])
        actual = np.array([getattr(s, name) for s in scored])
        assert np.abs(actual - expected).max() < 1e-9, (
            f"{name} no longer reproduces the estimator it was frozen from")


def test_a_flipped_kinase_objective_would_fail_here(manifest, oracles):
    """Known actives must score ABOVE the ZINC background.

    This is the test that catches an inverted objective. Agreement on a panel of
    inactives proves nothing -- a function returning 0.0 everywhere would pass
    that -- so the orientation is pinned against molecules that are known to be
    active.
    """
    background = [r["smiles"] for r in _panel(manifest, "zinc")][:PANEL_SLICE]
    background_scores = oracles.raw_many(background)
    for name, group in (("jnk3", "jnk3_active"), ("gsk3b", "gsk3b_active")):
        actives = [r["smiles"] for r in _panel(manifest, group)][:40]
        active_mean = np.mean([getattr(s, name) for s in oracles.raw_many(actives)])
        background_mean = np.mean([getattr(s, name) for s in background_scores])
        assert active_mean > background_mean + 0.25, (
            f"{name} does not separate known actives from ZINC: "
            f"{active_mean:.3f} vs {background_mean:.3f}. Either the objective "
            f"is inverted or this is the wrong model.")


def test_the_wrong_fingerprint_width_raises_instead_of_scoring(oracles):
    """The 1024-bit variant of these tasks is a DIFFERENT model.

    Handing it a 1024-wide fingerprint must fail, not produce a plausible number.
    """
    with pytest.raises(ValueError, match="2048 features"):
        oracles.jnk3.probabilities(np.zeros((1, 1024)))


def test_minimised_objectives_are_inverted_and_maximised_ones_are_not():
    """The transforms ARE the benchmark contract, so they are pinned as values."""
    raw = RawScores(qed=0.8, jnk3=0.9, sa=1.0, gsk3b=0.25, drd2=0.75)
    qed, jnk3, sa, gsk3b, drd2 = raw.normalized()
    assert qed == pytest.approx(0.8)                 # max: untouched
    assert jnk3 == pytest.approx(0.9)                # max: untouched
    assert sa == pytest.approx(1.0)                  # min: (10 - 1) / 9
    assert gsk3b == pytest.approx(0.75)              # min: 1 - 0.25
    assert drd2 == pytest.approx(0.25)               # min: 1 - 0.75

    worst = RawScores(qed=0.0, jnk3=0.0, sa=10.0, gsk3b=1.0, drd2=1.0)
    assert worst.normalized() == WORST_VECTOR
    best = RawScores(qed=1.0, jnk3=1.0, sa=1.0, gsk3b=0.0, drd2=0.0)
    assert best.normalized() == (1.0,) * N_OBJECTIVES


def test_a_high_sa_molecule_scores_worse_than_an_easy_one(oracles):
    """Direction check on real molecules, not just on the algebra."""
    easy = oracles.raw("c1ccccc1")                    # benzene: trivial
    hard = oracles.raw("O=C1NC(=O)C=2C=3C=4C(N(CCC=5N=C(N(C)C)C(CCCN6C=7C"
                       "(C(C21)=C6)=CC=CC7)=CC5)C3)=CC=CC4")
    assert easy.sa < hard.sa
    assert easy.normalized()[2] > hard.normalized()[2]


def test_an_unparseable_molecule_gets_the_worst_vector(oracles):
    """Never zero-by-accident: for the MINIMISED objectives zero is the BEST
    value, so a non-molecule scored as zeros everywhere would look ideal."""
    assert oracles.raw("not a molecule") is None
    assert oracles.evaluate_many(["not a molecule"])[0] == WORST_VECTOR
    assert oracles("C1CC") == WORST_VECTOR             # unclosed ring


def test_two_spellings_of_one_molecule_canonicalise_together():
    assert canonical("OCC") == canonical("CCO")
    assert canonical("~~~") is None


def test_a_tampered_bundle_refuses_to_load(tmp_path, manifest):
    """The manifest pins sha256 so a swapped npz cannot be scored against."""
    bundle = tmp_path / "bundle"
    shutil.copytree(DEFAULT_BUNDLE_DIR, bundle)
    target = bundle / manifest["jnk3"]["parameters_npz"]
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="Refusing to score"):
        Task3Objectives(bundle)


def test_a_missing_bundle_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="refuses to invent an oracle"):
        Task3Objectives(tmp_path)


def test_the_forest_walk_terminates_on_every_panel_molecule(manifest, oracles):
    """Every molecule must reach a leaf in every tree.

    The walk stops when no node is still internal; if a node table were corrupt
    it would raise rather than return a partial answer, so scoring the panel at
    all is the check.
    """
    rows = [r["smiles"] for r in _panel(manifest)[:PANEL_SLICE]]
    probabilities = oracles.jnk3.score_many(rows)
    assert probabilities.shape == (len(rows),)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_the_frozen_forest_is_the_2048_feature_model():
    forest = FrozenForest(DEFAULT_BUNDLE_DIR / "jnk3_forest.npz")
    assert forest.n_features == 2048, (
        "a 1024-feature forest is the HN-GFN variant, not TDC's oracle")
    assert forest.n_trees == 100
    assert forest.split_threshold == 0.5


def test_a_molecules_score_does_not_depend_on_what_it_was_scored_with(manifest,
                                                                     oracles):
    """Bit-identical alone and in a batch, for all five objectives.

    `docs/ORACLE_BATCH_INVARIANCE_DEFECT.md` records that the shared DRD2
    evaluator fails this by ~2e-14, because `kernel @ dual_coef` dispatches
    differently for one row than for many. Task 3 uses a batch-invariant
    reduction instead. Without this property the meter's cache and a re-run
    could disagree, and a resumed run would not reproduce the run it resumed.
    """
    panel = [r["smiles"] for r in _panel(manifest)[:40]]
    alone = [oracles.raw(s) for s in panel]
    together = oracles.raw_many(panel)
    chunked = [row for i in range(0, len(panel), 7)
               for row in oracles.raw_many(panel[i:i + 7])]
    for name in ("qed", "jnk3", "sa", "gsk3b", "drd2"):
        one = np.array([getattr(r, name) for r in alone])
        many = np.array([getattr(r, name) for r in together])
        seven = np.array([getattr(r, name) for r in chunked])
        assert np.array_equal(one, many), f"{name} depends on batch size"
        assert np.array_equal(one, seven), f"{name} depends on batch size"


# ---- the counting rule, exercised through the real oracle -----------------

def test_batched_and_single_metering_charge_identically(oracles):
    """A batch is a performance choice, never a budget choice."""
    molecules = ["CCO", "c1ccccc1", "CC(=O)O", "CCO", "OCC"]
    single = OracleMeter(oracles, budget=100, canonicalize=canonical)
    batched = OracleMeter(oracles, budget=100, canonicalize=canonical,
                          evaluate_many=oracles.evaluate_many)
    one_at_a_time = [single(s) for s in molecules]
    all_at_once = batched.batch(molecules)
    assert one_at_a_time == all_at_once
    # CCO and OCC are the same molecule, so three distinct molecules are charged.
    assert single.spent == batched.spent == 3


def test_an_unaffordable_batch_charges_nothing(oracles):
    """Partial charging would leave a run whose counter no longer describes it."""
    meter = OracleMeter(oracles, budget=2, canonicalize=canonical,
                        evaluate_many=oracles.evaluate_many)
    meter.batch(["CCO", "c1ccccc1"])
    assert meter.spent == 2
    with pytest.raises(BudgetExceeded, match="Nothing was charged"):
        meter.batch(["CC(=O)O", "CCC", "CCCC"])
    assert meter.spent == 2 and meter.n_unique == 2


def test_novel_in_counts_what_would_actually_be_charged(oracles):
    meter = OracleMeter(oracles, budget=100, canonicalize=canonical,
                        evaluate_many=oracles.evaluate_many)
    meter.batch(["CCO"])
    assert meter.novel_in(["CCO", "OCC"]) == 0        # both already paid for
    assert meter.novel_in(["CCO", "c1ccccc1", "CCC"]) == 2


def test_the_ledger_hook_sees_every_charged_molecule(oracles):
    seen: list[str] = []
    meter = OracleMeter(oracles, budget=100, canonicalize=canonical,
                        evaluate_many=oracles.evaluate_many,
                        on_evaluated=lambda smiles, values: seen.append(smiles))
    meter.batch(["CCO", "OCC", "c1ccccc1"])
    meter("CCO")
    assert seen == [canonical("CCO"), canonical("c1ccccc1")], (
        "the hook must fire once per CHARGED molecule, never for a cache hit")
