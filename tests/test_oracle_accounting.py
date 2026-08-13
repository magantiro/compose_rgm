from __future__ import annotations

import pytest

from compose_v4.experiments.oracle_accounting import (
    BudgetExhausted,
    OracleAccountant,
    canonical_smiles,
    silence_rdkit,
)


silence_rdkit()

BENZENE_VARIANTS = ["c1ccccc1", "C1=CC=CC=C1", "c1ccccc1"]
INVALID = "C1CC"  # unclosed ring; RDKit cannot parse


def _accountant(**kwargs) -> OracleAccountant:
    defaults = dict(
        evaluate=lambda smi: float(len(smi)),
        budget=100,
        budget_counter="unique_valid_canonical_evaluations",
    )
    defaults.update(kwargs)
    return OracleAccountant(**defaults)


def test_budget_counter_must_be_named_explicitly() -> None:
    with pytest.raises(ValueError, match="budget_counter"):
        OracleAccountant(evaluate=lambda s: 0.0, budget=10, budget_counter="whatever")


def test_equivalent_smiles_are_one_unique_evaluation_but_three_requests() -> None:
    acct = _accountant()
    values = [acct.score(s) for s in BENZENE_VARIANTS]
    assert len(set(values)) == 1, "the same molecule must score identically"
    counts = acct.counts.as_dict()
    assert counts["oracle_requests"] == 3
    assert counts["unique_valid_canonical_evaluations"] == 1
    assert counts["evaluator_calls"] == 1
    assert counts["duplicate_requests"] == 2
    assert counts["cache_hits"] == 2


def test_caching_cannot_erase_a_wasteful_request() -> None:
    """The conceptual invariant the three counters exist to encode."""
    cached = _accountant()
    uncached = _accountant(cache=False)
    for smiles in BENZENE_VARIANTS:
        cached.score(smiles)
        uncached.score(smiles)

    # The cache changes real work...
    assert cached.counts.evaluator_calls == 1
    assert uncached.counts.evaluator_calls == 3
    # ...and leaves both algorithmic demand and the benchmark-native number
    # untouched. That is the whole point of separating the three.
    assert cached.counts.oracle_requests == uncached.counts.oracle_requests == 3
    assert (
        cached.counts.unique_valid_canonical_evaluations
        == uncached.counts.unique_valid_canonical_evaluations
        == 1
    )


def test_invalid_molecule_charges_requests_only() -> None:
    acct = _accountant()
    assert acct.score(INVALID) == 0.0
    counts = acct.counts.as_dict()
    assert counts["oracle_requests"] == 1
    assert counts["unique_valid_canonical_evaluations"] == 0
    assert counts["evaluator_calls"] == 0
    assert counts["failed_proposals"] == 1


def test_counter_identity_holds_with_and_without_cache() -> None:
    stream = [*BENZENE_VARIANTS, INVALID, "CCO", "OCC", "not_a_molecule"]
    for cache in (True, False):
        acct = _accountant(cache=cache)
        for smiles in stream:
            acct.score(smiles)
        assert all(acct.manifest()["invariants"].values()), f"cache={cache}"


def test_budget_binds_the_named_counter_and_not_the_others() -> None:
    """Three duplicates must not exhaust a benchmark-native budget of 2."""
    acct = _accountant(budget=2, budget_counter="unique_valid_canonical_evaluations")
    for smiles in BENZENE_VARIANTS:
        acct.score(smiles)
    assert acct.counts.oracle_requests == 3
    assert not acct.exhausted

    demand = _accountant(budget=2, budget_counter="oracle_requests")
    demand.score(BENZENE_VARIANTS[0])
    demand.score(BENZENE_VARIANTS[1])
    assert demand.exhausted
    with pytest.raises(BudgetExhausted):
        demand.score(BENZENE_VARIANTS[2])


def test_evaluator_calls_can_also_bind_a_budget() -> None:
    acct = _accountant(budget=1, budget_counter="evaluator_calls")
    acct.score("CCO")
    assert acct.exhausted
    # A duplicate would have cost no evaluator work, but the budget is already
    # spent, so the driver must stop rather than sneak a free request through.
    with pytest.raises(BudgetExhausted):
        acct.score("OCC")


def test_score_many_stops_at_the_budget_without_raising() -> None:
    acct = _accountant(budget=2, budget_counter="unique_valid_canonical_evaluations")
    scored = acct.score_many(["CCO", "CCC", "CCCC", "CCCCC"])
    assert len(scored) == 2
    assert acct.exhausted


def test_demand_ratio_name_matches_what_it_divides() -> None:
    acct = _accountant()
    for smiles in BENZENE_VARIANTS:
        acct.score(smiles)
    counts = acct.counts.as_dict()
    expected = (
        counts["oracle_requests"] / counts["unique_valid_canonical_evaluations"]
    )
    assert acct.demand_ratio_requests_over_unique == pytest.approx(expected)
    assert acct.demand_ratio_requests_over_unique == pytest.approx(3.0)


def test_canonicalization_is_the_shared_key() -> None:
    assert canonical_smiles("C1=CC=CC=C1") == canonical_smiles("c1ccccc1")
    assert canonical_smiles(INVALID) is None
    assert canonical_smiles("") is None
