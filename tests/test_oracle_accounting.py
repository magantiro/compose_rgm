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
    defaults = dict(evaluate=lambda smi: float(len(smi)), budget=100,
                    budget_counter="benchmark_native")
    defaults.update(kwargs)
    return OracleAccountant(**defaults)


def test_budget_counter_must_be_named_explicitly() -> None:
    with pytest.raises(ValueError, match="budget_counter"):
        OracleAccountant(evaluate=lambda s: 0.0, budget=10, budget_counter="whatever")


def test_equivalent_smiles_are_one_benchmark_native_call_but_three_invocations() -> None:
    acct = _accountant()
    values = [acct.score(s) for s in BENZENE_VARIANTS]
    assert len(set(values)) == 1, "the same molecule must score identically"
    counts = acct.counts.as_dict()
    assert counts["raw_compute"] == 3
    assert counts["benchmark_native"] == 1
    assert counts["evaluator_calls"] == 1
    assert counts["cache_hits"] == 2


def test_invalid_molecule_charges_raw_compute_only() -> None:
    acct = _accountant()
    assert acct.score(INVALID) == 0.0
    counts = acct.counts.as_dict()
    assert counts["raw_compute"] == 1
    assert counts["benchmark_native"] == 0
    assert counts["failed_proposals"] == 1


def test_counter_identity_holds() -> None:
    acct = _accountant()
    for smiles in [*BENZENE_VARIANTS, INVALID, "CCO", "OCC", "not_a_molecule"]:
        acct.score(smiles)
    assert all(acct.manifest()["invariants"].values())


def test_budget_binds_the_named_counter_and_not_the_other() -> None:
    """Three duplicates must not exhaust a benchmark_native budget of 2."""
    acct = _accountant(budget=2, budget_counter="benchmark_native")
    for smiles in BENZENE_VARIANTS:
        acct.score(smiles)
    assert acct.counts.raw_compute == 3
    assert not acct.exhausted

    raw = _accountant(budget=2, budget_counter="raw_compute")
    raw.score(BENZENE_VARIANTS[0])
    raw.score(BENZENE_VARIANTS[1])
    assert raw.exhausted
    with pytest.raises(BudgetExhausted):
        raw.score(BENZENE_VARIANTS[2])


def test_score_many_stops_at_the_budget_without_raising() -> None:
    acct = _accountant(budget=2, budget_counter="benchmark_native")
    scored = acct.score_many(["CCO", "CCC", "CCCC", "CCCCC"])
    assert len(scored) == 2
    assert acct.exhausted


def test_disabling_the_cache_changes_evaluator_calls_but_not_benchmark_native() -> None:
    calls: list[str] = []

    def evaluate(smi: str) -> float:
        calls.append(smi)
        return 1.0

    acct = _accountant(evaluate=evaluate, cache=False)
    for smiles in BENZENE_VARIANTS:
        acct.score(smiles)
    # Without a cache every distinct-looking request reaches the evaluator, and
    # benchmark_native counts them all -- which is exactly why the fairness
    # contract requires caching with identical semantics for every method.
    assert len(calls) == 3
    assert acct.counts.benchmark_native == 3
    assert acct.counts.cache_hits == 0


def test_demand_ratio_reports_duplicate_pressure() -> None:
    acct = _accountant()
    for smiles in BENZENE_VARIANTS:
        acct.score(smiles)
    assert acct.demand_ratio == pytest.approx(3.0)


def test_canonicalization_is_the_shared_key() -> None:
    assert canonical_smiles("C1=CC=CC=C1") == canonical_smiles("c1ccccc1")
    assert canonical_smiles(INVALID) is None
    assert canonical_smiles("") is None
