"""Adversarial fixtures: each metric must SAY SO when its intended conclusion is false.

Project rule — every new analysis metric ships with a fixture in which the
conclusion it is meant to support is FALSE, plus a test asserting the metric
reports that. Seven sign-fixed statistics have been caught across the lanes; the
failure mode is a statistic that can only come out one way.

For each metric below, ask: what value could this take if the hypothesis were
false? If the answer is "none", it is not a measurement.
"""

from __future__ import annotations

import pytest

from compose_v4.experiments.oracle_accounting import OracleAccountant

pytest.importorskip("rdkit")

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def _accountant(**kwargs) -> OracleAccountant:
    defaults = dict(
        evaluate=lambda smi: 1.0,
        budget=1000,
        budget_counter="unique_valid_canonical_evaluations",
    )
    defaults.update(kwargs)
    return OracleAccountant(**defaults)


# ---- demand ratio: intended conclusion "this method wastes oracle demand" ----


def test_demand_ratio_reports_NO_waste_when_there_is_none() -> None:
    """The adversarial case: a method that never repeats itself.

    The ratio is used to argue that a method demands more than it evaluates. If
    it could only ever exceed 1.0, it would be sign-fixed and worthless.
    """
    acct = _accountant()
    for smiles in ["CCO", "CCC", "CCCC", "CCCCC", "c1ccccc1"]:
        acct.score(smiles)
    assert acct.demand_ratio_requests_over_unique == pytest.approx(1.0), (
        "a method with zero duplicates and zero invalids must score exactly 1.0; "
        "anything above it would mean the metric cannot represent 'no waste'"
    )


def test_demand_ratio_separates_a_wasteful_method_from_a_clean_one() -> None:
    clean = _accountant()
    for smiles in ["CCO", "CCC", "CCCC"]:
        clean.score(smiles)

    wasteful = _accountant()
    for smiles in ["CCO"] * 9 + ["CCC"]:
        wasteful.score(smiles)

    assert clean.demand_ratio_requests_over_unique == pytest.approx(1.0)
    assert wasteful.demand_ratio_requests_over_unique == pytest.approx(5.0)


# ---- caching invariant: intended conclusion "a cache cannot hide waste" ----


def test_the_cache_invariant_would_be_visible_if_it_were_violated() -> None:
    """Adversarial: if oracle_requests tracked evaluator work, this would fail.

    The invariant claims a cache reduces evaluator_calls WITHOUT reducing
    oracle_requests. A fixture where the two move together is exactly the world
    the invariant denies, so the test pins both numbers rather than only the
    direction.
    """
    stream = ["CCO", "OCC", "C(C)O", "CCO"]  # all the same molecule
    cached = _accountant(cache=True)
    uncached = _accountant(cache=False)
    for smiles in stream:
        cached.score(smiles)
        uncached.score(smiles)

    assert cached.counts.evaluator_calls == 1
    assert uncached.counts.evaluator_calls == 4
    # The claim that would be FALSE if requests tracked evaluator work:
    assert cached.counts.oracle_requests == uncached.counts.oracle_requests == 4
    assert (
        cached.counts.unique_valid_canonical_evaluations
        == uncached.counts.unique_valid_canonical_evaluations
        == 1
    )


# ---- applicability: intended conclusion "the domain excludes real molecules" ----


def test_applicability_admits_a_panel_it_should_admit() -> None:
    """Adversarial: a filter that rejected everything would look 'safe' and be useless."""
    from compose_v4.experiments.graphxform_applicability import partition_panel

    ordinary = ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"]
    result = partition_panel(ordinary)
    assert result["n_inapplicable"] == 0, (
        "the domain must ADMIT ordinary organic molecules; a filter that rejects "
        "everything would trivially 'protect' the comparison while destroying it"
    )


def test_applicability_rejects_what_the_method_genuinely_cannot_represent() -> None:
    from compose_v4.experiments.graphxform_applicability import partition_panel

    result = partition_panel(["OB(O)c1ccccc1", "C[Si](C)(C)C", "C" * 60])
    assert result["n_applicable"] == 0
    assert set(result["inapplicable_reasons"]) == {
        "unsupported_element_or_charge_state",
        "exceeds_heavy_atom_ceiling_50",
    }


def test_applicability_never_silently_drops_a_molecule() -> None:
    """Every input must appear in exactly one bucket, or the count is a lie."""
    from compose_v4.experiments.graphxform_applicability import partition_panel

    panel = ["CCO", "OB(O)c1ccccc1", "not_a_molecule", "c1ccccc1"]
    result = partition_panel(panel)
    assert result["n_applicable"] + result["n_inapplicable"] == len(panel)
    assert len(result["inapplicable"]) == result["n_inapplicable"]


def test_applicability_uses_only_the_native_ceiling_with_no_margin() -> None:
    """The adversarial case: a margin that quietly makes the panel convenient.

    An undervable margin whose only observable effect is to admit our own cohort
    is exactly what a reviewer would attack, so there must be none. A molecule
    between the old margin (42) and the native ceiling (50) must be ADMITTED.
    """
    from compose_v4.experiments.graphxform_applicability import (
        HEADROOM_BASIS,
        HEAVY_ATOM_CEILING,
        MAX_NUM_ATOMS,
        REQUIRED_HEADROOM,
        check_applicability,
    )

    assert REQUIRED_HEADROOM == 0
    assert HEAVY_ATOM_CEILING == MAX_NUM_ATOMS == 50
    assert HEADROOM_BASIS == "NONE_NATIVE_CEILING_ONLY"
    # 45 heavy atoms sits in the band the withdrawn margin would have excluded.
    admitted = check_applicability("C" * 45)
    assert admitted.applicable and admitted.heavy_atoms == 45
    assert not check_applicability("C" * 55).applicable
