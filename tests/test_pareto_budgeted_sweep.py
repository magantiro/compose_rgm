"""Q1-Q3: qualification of the budgeted preference sweep.

These run BEFORE any K=41 quality outcome is opened, per
`docs/BUDGETED_PREFERENCE_SWEEP_PREREGISTRATION.md`. This is potentially the
final Pareto controller, so the code is qualified before it is trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.experiments.pareto_budgeted_sweep import (
    EXPANSION_GUARD,
    K_SWEEP,
    ORACLE_BUDGET,
    OracleLedger,
    select_for_evaluation,
)


def _fiber(n: int, seed: int = 0) -> list[list[object]]:
    """A synthetic canonical fiber with distinct keys and R_theta weights."""
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    return [[f"K{i:05d}", float(p[i])] for i in range(n)]


# --------------------------------------------------------------------------
# The derivation itself
# --------------------------------------------------------------------------

def test_K_SWEEP_is_derived_from_frozen_constants_not_from_P0c_data():
    assert ORACLE_BUDGET == 10_000
    assert EXPANSION_GUARD == 240
    assert K_SWEEP == ORACLE_BUDGET // EXPANSION_GUARD == 41


def test_worst_case_guard_traversal_stays_under_budget_so_cap_never_binds():
    """The whole reason the guard was used instead of the observed median."""
    assert EXPANSION_GUARD * K_SWEEP == 9_840
    assert EXPANSION_GUARD * K_SWEEP <= ORACLE_BUDGET
    # The headroom the preregistration audits for hidden oracle calls.
    assert ORACLE_BUDGET - EXPANSION_GUARD * K_SWEEP == 160


# --------------------------------------------------------------------------
# Q1 -- full-fiber equivalence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("width", [1, 7, 41, 42, 600, 1877])
def test_Q1_when_K_covers_the_fiber_every_successor_is_scored(width):
    """With K >= |F(x)| the budgeted path must reproduce the full sweep exactly.

    If this fails, any later difference between budgeted and ceiling is a BUG,
    not a finding -- which is precisely what Q1 exists to rule out.
    """
    rows = _fiber(width)
    to_eval, subset = select_for_evaluation(rows, already_scored=[], k=width)
    all_keys = [str(r[0]) for r in rows]
    assert to_eval and len(to_eval) == width
    # Identical set AND identical order to the full sweep's candidate list.
    assert subset == all_keys


def test_Q1_partition_input_is_order_identical_to_the_full_sweep():
    """The partition must see candidates in the kernel's own fiber order."""
    rows = _fiber(50)
    _, subset = select_for_evaluation(rows, already_scored=[], k=1000)
    assert subset == [str(r[0]) for r in rows]


# --------------------------------------------------------------------------
# Q2 -- ranking correctness
# --------------------------------------------------------------------------

def test_Q2_shortlist_takes_the_highest_R_theta_successors():
    rows = _fiber(500, seed=3)
    to_eval, _ = select_for_evaluation(rows, already_scored=[], k=K_SWEEP)
    order = sorted(rows, key=lambda r: (-float(r[1]), str(r[0])))
    assert to_eval == [str(r[0]) for r in order[:K_SWEEP]]


def test_Q2_ties_break_deterministically_and_ignore_fiber_order():
    """Same discipline `_argmin_stable` applies to the controller's own choice."""
    rows = [["Kb", 0.5], ["Ka", 0.5], ["Kc", 0.5]]
    a, _ = select_for_evaluation(rows, already_scored=[], k=2)
    b, _ = select_for_evaluation(list(reversed(rows)), already_scored=[], k=2)
    assert a == b == ["Ka", "Kb"]


def test_Q2_cached_successors_are_free_and_never_displace_fresh_candidates():
    """Cached values cost nothing, so they must not consume shortlist slots."""
    rows = _fiber(300, seed=5)
    top = sorted(rows, key=lambda r: (-float(r[1]), str(r[0])))
    cached = [str(r[0]) for r in top[:10]]          # the 10 best are already known
    to_eval, subset = select_for_evaluation(rows, already_scored=cached, k=K_SWEEP)
    assert len(to_eval) == K_SWEEP                   # still a full fresh shortlist
    assert not (set(to_eval) & set(cached))          # nothing re-billed
    assert len(subset) == K_SWEEP + len(cached)      # cached ones still partitioned
    assert set(cached).issubset(set(subset))


def test_Q2_shortlist_never_exceeds_K_even_on_a_very_wide_fiber():
    rows = _fiber(1877)
    to_eval, _ = select_for_evaluation(rows, already_scored=[], k=K_SWEEP)
    assert len(to_eval) == K_SWEEP


# --------------------------------------------------------------------------
# Q3 -- worst-case budget fixture
# --------------------------------------------------------------------------

def test_Q3_worst_case_240_expansions_zero_overlap_bills_exactly_9840():
    """Synthetic guard-length tree, no cache overlap, 41 unseen at every state."""
    ledger = OracleLedger()
    for state in range(EXPANSION_GUARD):
        rows = [[f"S{state:04d}_{i:05d}", float(i)] for i in range(600)]
        to_eval, _ = select_for_evaluation(rows, already_scored=ledger._evaluated,
                                           k=K_SWEEP)
        billed = ledger.charge(to_eval)
        assert len(billed) == K_SWEEP
    assert ledger.spent == 9_840
    assert ledger.spent <= ORACLE_BUDGET
    assert ledger.binding is False, "the hard cap must NEVER bind under K_SWEEP"
    assert ledger.remaining == 160


def test_Q3_ledger_counts_unique_molecules_not_raw_requests():
    """Frozen semantics: the denominator is unique canonical molecules."""
    ledger = OracleLedger()
    ledger.charge(["A", "B", "C"])
    ledger.charge(["A", "B", "C"])       # repeat requests are free
    ledger.charge(["C", "D"])
    assert ledger.spent == 4


def test_Q3_hard_cap_refuses_to_cross_the_budget():
    ledger = OracleLedger(budget=10)
    ledger.charge([f"X{i}" for i in range(8)])
    billed = ledger.charge([f"Y{i}" for i in range(5)])
    assert ledger.spent == 10
    assert len(billed) == 2
    assert ledger.binding is True


def test_Q3_would_exceed_ignores_already_evaluated_keys():
    ledger = OracleLedger(budget=5)
    ledger.charge(["A", "B", "C", "D"])
    assert ledger.would_exceed(["E", "F"]) is True
    assert ledger.would_exceed(["A", "B", "E"]) is False   # only E is fresh
