"""Parity of the free PMO re-derivations against real charged oracle rows.

These objectives are the scorer every offline allocation result in this arm is
measured against.  If they drift from the benchmark, every number built on them
is wrong, so parity is asserted against the recorded charged ledger rather than
against a hand-written expectation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from compose_v4.experiments.pmo_free_objectives import (
    CELECOXIB,
    FREE_OBJECTIVES,
    PERINDOPRIL,
    build_free_scorer,
    celecoxib_rediscovery,
    perindopril_mpo,
)

_AUTOPSY = Path(__file__).resolve().parents[1] / "diagnostics" / "pmo_3x250_autopsy_v1.json"


def _charged_rows():
    if not _AUTOPSY.exists():
        pytest.skip(f"charged-ledger fixture absent: {_AUTOPSY}")
    return json.loads(_AUTOPSY.read_text())["celecoxib_distance"]["all_rows"]


def test_celecoxib_reproduces_every_charged_row_exactly():
    """250 real charged calls from the completed 3x250 run.  Zero tolerance:
    this is the benchmark objective, not an approximation of it."""
    rows = _charged_rows()
    assert len(rows) == 250
    deltas = [abs(celecoxib_rediscovery(r["endpoint"]) - r["charged_score"]) for r in rows]
    assert max(deltas) == 0.0


def test_celecoxib_scores_itself_at_one():
    assert celecoxib_rediscovery(CELECOXIB) == pytest.approx(1.0)


def test_objectives_are_bounded_to_the_unit_interval():
    """pmo_top_ten_auc rejects any reward outside [0, 1]."""
    probes = [CELECOXIB, "CCO", "c1ccccc1", "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC"]
    for objective in FREE_OBJECTIVES.values():
        for smiles in probes:
            assert 0.0 <= objective(smiles) <= 1.0


def test_unparseable_smiles_scores_zero_rather_than_raising():
    for objective in FREE_OBJECTIVES.values():
        assert objective("this is not a molecule") == 0.0


def test_perindopril_reproduces_the_recorded_charged_candidates():
    """The 15 top charged candidates of the same run, including its best
    (0.4864578...).  Tolerance is float64 round-off only."""
    if not _AUTOPSY.exists():
        pytest.skip("charged-ledger fixture absent")
    rows = json.loads(_AUTOPSY.read_text())["tasks"]["perindopril_mpo"]["top_candidates"]
    assert len(rows) == 15
    deltas = [abs(perindopril_mpo(r["endpoint"]) - r["score"]) for r in rows]
    assert max(deltas) < 1e-12
    assert max(r["score"] for r in rows) == pytest.approx(0.4864578310337349)


def test_perindopril_is_an_mpo_not_a_bare_similarity():
    """Perindopril itself has ZERO aromatic rings, so the Gaussian ring term at
    mu=2 holds the reference molecule far below 1.0 even at similarity 1.0.
    A bare-similarity implementation would score it 1.0 and pass a naive test."""
    assert perindopril_mpo(PERINDOPRIL) == pytest.approx(math.exp(-4.0), rel=1e-9)
    assert perindopril_mpo(PERINDOPRIL) < 0.02


def test_asset_backed_oracles_are_refused_not_approximated():
    for task in ("gsk3b", "jnk3", "drd2"):
        with pytest.raises(ValueError):
            build_free_scorer(task)


def test_build_free_scorer_returns_the_matching_objective():
    assert build_free_scorer("celecoxib_rediscovery")(CELECOXIB) == pytest.approx(1.0)
