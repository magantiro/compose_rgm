"""Harness tests. A dummy policy must pass the FULL contract with no COMPOSE."""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.benchmark.molleo_task3 import (
    N_OBJECTIVES,
    BudgetExceeded,
    CountingRule,
    OracleMeter,
    hypervolume,
)


def fake_eval(smi: str):
    r = np.random.default_rng(abs(hash(smi)) % (2**32))
    return tuple(r.random(N_OBJECTIVES))


def test_meter_raises_instead_of_overrunning():
    m = OracleMeter(fake_eval, budget=3)
    for i in range(3):
        m(f"C{i}")
    assert m.spent == 3 and m.remaining == 0
    with pytest.raises(BudgetExceeded, match="over the 3 budget"):
        m("Cnew")
    assert m.spent == 3, "a refused call must not be charged"


def test_repeat_queries_are_cached_and_charged_once():
    m = OracleMeter(fake_eval, budget=10)
    a, b = m("CCO"), m("CCO")
    assert a == b and m.spent == 1 and m.n_unique == 1


def test_per_objective_rule_is_five_times_stricter():
    per_mol = OracleMeter(fake_eval, budget=10,
                          counting_rule=CountingRule.PER_MOLECULE)
    per_obj = OracleMeter(fake_eval, budget=10,
                          counting_rule=CountingRule.PER_OBJECTIVE)
    for i in range(2):
        per_mol(f"C{i}"); per_obj(f"C{i}")
    assert per_mol.spent == 2 and per_obj.spent == 10
    with pytest.raises(BudgetExceeded):
        per_obj("C99")          # 10/10 spent after only 2 molecules
    per_mol("C99")              # still has room


def test_can_afford_lets_a_policy_stop_cleanly():
    m = OracleMeter(fake_eval, budget=5)
    assert m.can_afford(5) and not m.can_afford(6)


def test_wrong_objective_count_is_rejected():
    m = OracleMeter(lambda s: (0.1, 0.2), budget=10)
    with pytest.raises(ValueError, match="expected 5 objectives"):
        m("CCO")


def test_hypervolume_of_the_unit_corner_is_the_whole_box():
    assert hypervolume([[1.0] * 5]) == pytest.approx(1.0, abs=1e-9)


def test_hypervolume_is_monotone_under_a_dominating_point():
    lo = hypervolume([[0.5] * 5], samples=80_000)
    hi = hypervolume([[0.5] * 5, [0.9] * 5], samples=80_000)
    assert hi > lo


def test_dominated_points_do_not_change_hypervolume():
    a = hypervolume([[0.8] * 5], samples=80_000)
    b = hypervolume([[0.8] * 5, [0.3] * 5, [0.1] * 5], samples=80_000)
    assert b == pytest.approx(a, rel=0.02)


def test_empty_front_has_zero_hypervolume():
    assert hypervolume([]) == 0.0


def test_dummy_policy_completes_the_whole_contract():
    """End-to-end: random policy, real meter, real HV, budget respected."""
    rng = np.random.default_rng(0)
    m = OracleMeter(fake_eval, budget=200)
    archive = []
    i = 0
    while m.can_afford(1):
        archive.append(m(f"MOL{i}"))
        i += 1
    assert m.spent == 200
    hv = hypervolume(archive, samples=50_000)
    assert 0.0 < hv <= 1.0
    with pytest.raises(BudgetExceeded):
        m("one-too-many")


def test_two_spellings_of_one_molecule_cost_one_unit():
    """Canonicalization is part of the COUNTING RULE, not an optimization."""
    from rdkit import Chem

    def canon(s: str) -> str:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else s

    # Same molecule, different valid SMILES spellings.
    a, b = "OCC", "CCO"
    assert canon(a) == canon(b)

    metered = OracleMeter(fake_eval, budget=10, canonicalize=canon)
    va, vb = metered(a), metered(b)
    assert va == vb, "same molecule must return the same objective vector"
    assert metered.spent == 1, "two spellings of one molecule cost ONE unit"

    # Without a canonicalizer the benchmark would overcharge -- this is the
    # bug the rule exists to prevent, pinned so it cannot silently return.
    naive = OracleMeter(fake_eval, budget=10)
    naive(a); naive(b)
    assert naive.spent == 2


def test_the_reported_hypervolume_is_exact_enough_to_compare_policies():
    """The QMC estimator is the one runs report, so its error must be far below
    any difference worth arguing about."""
    from compose_v4.benchmark.molleo_task3 import hypervolume_qmc

    # One corner point dominates exactly prod(p) of the unit box.
    assert hypervolume_qmc([[0.5] * 5]) == pytest.approx(0.5 ** 5, abs=1e-5)
    assert hypervolume_qmc([[1.0] * 5]) == pytest.approx(1.0, abs=1e-9)
    assert hypervolume_qmc([]) == 0.0


def test_the_reported_hypervolume_is_deterministic():
    """No seed, no variance: the same front always returns the same number, so a
    result can be recomputed from an archive alone."""
    from compose_v4.benchmark.molleo_task3 import hypervolume_qmc

    front = np.random.default_rng(3).random((40, 5)) * 0.6
    assert hypervolume_qmc(front) == hypervolume_qmc(front)


def test_the_reported_hypervolume_rejects_unnormalised_objectives():
    """It integrates the unit box; a raw SA value of 3.2 would silently break it."""
    from compose_v4.benchmark.molleo_task3 import hypervolume_qmc

    with pytest.raises(ValueError, match="normalised to"):
        hypervolume_qmc([[0.5, 0.5, 3.2, 0.5, 0.5]])


def test_a_dominating_front_reports_more_hypervolume():
    from compose_v4.benchmark.molleo_task3 import hypervolume_qmc

    worse = [[0.4] * 5, [0.6, 0.2, 0.2, 0.2, 0.2]]
    better = worse + [[0.7] * 5]
    assert hypervolume_qmc(better) > hypervolume_qmc(worse)


@pytest.mark.parametrize("log2_samples", [10, 14, 17, 18])
def test_a_small_sample_request_still_integrates_something(log2_samples):
    """A chunk larger than the request would integrate nothing and return 0.0 --
    a wrong answer that looks like an empty front."""
    from compose_v4.benchmark.molleo_task3 import hypervolume_qmc

    assert hypervolume_qmc([[0.5] * 5], log2_samples=log2_samples) > 0.0
