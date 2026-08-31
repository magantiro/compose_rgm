"""Invariants for region-local stochastic rewriting.

Executor-free: a stub law and stub apply exercise the admissibility, invariant,
and likelihood machinery without the model.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from compose_v4.control.region_rewrite import (
    RewriteContext, admissible_indices, context_preserved, created_slot,
    graph_connected, propose, touched_slots,
)


@dataclass
class _Bond:
    a: int
    b: int
    order: int = 1


@dataclass
class _Insert:
    slot: int
    atom_type: int
    formal_charge: int = 0
    implicit_h_count: int = 0
    neighbors: tuple = ()


@dataclass
class _Del:
    v: int


@dataclass
class _Reroute:
    a: int
    b: int
    u: int
    v: int
    new_order: int = 1


class _State:
    def __init__(self, types, bonds):
        self.atom_types = np.array(types)
        self.formal_charges = np.zeros(len(types), dtype=int)
        self.implicit_h_counts = np.zeros(len(types), dtype=int)
        self.bonds = np.array(bonds)


def _chain(n):
    b = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        b[i][i + 1] = b[i + 1][i] = 1
    return _State([2] * n, b)


def test_touched_slots_covers_every_action_shape():
    assert touched_slots(_Bond(1, 2)) == {1, 2}
    assert touched_slots(_Del(4)) == {4}
    assert touched_slots(_Reroute(1, 2, 3, 4)) == {1, 2, 3, 4}
    assert touched_slots(_Insert(9, 2, neighbors=((3, 1),))) == {9, 3}


def test_created_slot_only_for_inserts():
    assert created_slot(_Insert(9, 2)) == 9
    assert created_slot(_Bond(1, 2)) is None
    assert created_slot(_Del(3)) is None


def _ctx(frozen, locus, terminals=(), interface="pendant", k=1):
    return RewriteContext(frozenset(frozen), frozenset(locus), tuple(terminals),
                          interface, k)


def test_actions_touching_frozen_context_are_rejected():
    ctx = _ctx(frozen={0, 1}, locus={2, 3})
    fams = ["bond_insert", "bond_insert", "atom_delete"]
    acts = [_Bond(0, 1), _Bond(2, 3), _Del(0)]
    idx, why = admissible_indices(fams, acts, ctx)
    assert idx == [1]
    assert why["frozen_touched"] == 2


def test_boundary_terminal_is_reachable_but_context_is_not():
    ctx = _ctx(frozen={0, 1}, locus={2}, terminals=((1, 2, 1.0),))
    fams = ["bond_insert", "bond_insert"]
    acts = [_Bond(1, 2), _Bond(0, 2)]        # terminal ok, non-terminal not
    idx, _ = admissible_indices(fams, acts, ctx)
    assert idx == [0]


def test_insert_attached_to_locus_is_admissible():
    ctx = _ctx(frozen={0}, locus={1})
    idx, _ = admissible_indices(["atom_insert"], [_Insert(5, 2, neighbors=((1, 1),))], ctx)
    assert idx == [0]


def test_context_preserved_detects_mutation():
    a = _chain(4)
    b = _chain(4)
    assert context_preserved(a, b, frozenset({0, 1}))
    b.atom_types[0] = 3
    assert not context_preserved(a, b, frozenset({0, 1}))


def test_context_preserved_detects_bond_change_inside_context():
    a, b = _chain(4), _chain(4)
    b.bonds[0][1] = b.bonds[1][0] = 2
    assert not context_preserved(a, b, frozenset({0, 1}))


def test_graph_connected():
    assert graph_connected(_chain(4))
    s = _chain(4)
    s.bonds[1][2] = s.bonds[2][1] = 0
    assert not graph_connected(s)


def test_propose_accumulates_exact_log_density():
    """conditional_path_logq is the log-density of the REALISED path.

    It is deliberately not the full proposal density: q(M|x), stochastic
    termination, and the overlapping-mechanism mixture are all still missing.
    """
    st = _chain(4)
    fams, acts = ["bond_insert", "bond_insert"], [_Bond(2, 3), _Bond(2, 3)]
    probs = np.array([0.25, 0.75])

    def enum_fn(_s):
        return fams, acts, probs

    def apply_fn(s, _j):
        return s

    ctx = _ctx(frozen={0, 1}, locus={2, 3})
    rng = np.random.default_rng(0)
    p = propose(enum_fn, apply_fn, ctx, st, schedule=[("grow_new", 3)], rng=rng)
    assert p.status == "OK"
    assert len(p.steps) == 3
    expected = sum(np.log(s["q"]) for s in p.steps)
    assert p.conditional_path_logq == pytest.approx(expected)
    assert all(s["q"] in (pytest.approx(0.25), pytest.approx(0.75)) for s in p.steps)


def test_propose_returns_explicit_unsat_not_a_retry():
    st = _chain(4)

    def enum_fn(_s):
        return ["bond_insert"], [_Bond(0, 1)], np.array([1.0])   # frozen only

    def apply_fn(s, _j):
        return s

    ctx = _ctx(frozen={0, 1}, locus={2, 3})
    p = propose(enum_fn, apply_fn, ctx, st, schedule=[("grow_new", 1)],
                rng=np.random.default_rng(0))
    assert p.status == "UNSAT"
    assert p.stage == "grow_new:no_admissible_action"
    assert p.endpoint is None


def test_propose_rejects_a_step_that_disconnects():
    st = _chain(4)

    def enum_fn(_s):
        return ["bond_delete"], [_Bond(2, 3)], np.array([1.0])

    def apply_fn(_s, _j):
        broken = _chain(4)
        broken.bonds[2][3] = broken.bonds[3][2] = 0
        return broken

    ctx = _ctx(frozen={0, 1}, locus={2, 3})
    p = propose(enum_fn, apply_fn, ctx, st, schedule=[("prune_old", 1)],
                rng=np.random.default_rng(0))
    assert p.status == "UNSAT" and p.stage.endswith("disconnected")


# ------------------------------------------------------------------ lineage
from compose_v4.control.region_rewrite import Lineage, handoff_satisfied


def test_lineage_survives_delete_and_slot_reuse():
    """The executor zeroes a slot in place and can refill it later, so slot
    identity alone would silently rename an atom."""
    lin = Lineage.initial([0, 1, 2])
    assert lin.id_of[1] == 1
    lin = lin.observe("atom_delete", _Del(1))
    assert 1 not in lin.id_of, "deleted slot still claims an identity"
    lin2 = lin.observe("atom_insert", _Insert(1, 2, neighbors=((0, 1),)))
    assert lin2.id_of[1] != 1, "refilled slot reused a retired lineage id"
    assert lin2.slot_of[lin2.id_of[1]] == 1


def test_lineage_preserved_by_non_atom_ops():
    lin = Lineage.initial([0, 1, 2])
    before = dict(lin.id_of)
    for fam, act in (("bond_insert", _Bond(0, 2)), ("bond_reorder", _Bond(0, 1, 2)),
                     ("bond_reroute", _Reroute(0, 1, 1, 2)),
                     ("atom_restate_semantic", _Del(0))):
        if fam == "atom_restate_semantic":
            continue
        assert lin.observe(fam, act).id_of == before


def test_handoff_fires_only_when_context_joins_without_old_region():
    """Make-before-break: the trigger is topological, not a step count."""
    st = _chain(4)                       # 0-1-2-3, region = {1,2}, context {0,3}
    ctx = _ctx(frozen={0, 3}, locus={1, 2}, terminals=((0, 1, 1.0), (3, 2, 1.0)),
               interface="bridge", k=2)
    lin = Lineage.initial([0, 1, 2, 3])
    old = frozenset({lin.id_of[1], lin.id_of[2]})
    assert not handoff_satisfied(st, ctx, lin, old), "old bridge must not count"
    st2 = _chain(4)
    st2.bonds[0][3] = st2.bonds[3][0] = 1   # a new direct connection appears
    assert handoff_satisfied(st2, ctx, lin, old)


def test_missing_handoff_is_an_explicit_failure_label():
    st = _chain(4)
    ctx = _ctx(frozen={0, 3}, locus={1, 2}, terminals=((0, 1, 1.0), (3, 2, 1.0)),
               interface="bridge", k=2)
    lin = Lineage.initial([0, 1, 2, 3])
    old = frozenset({lin.id_of[1], lin.id_of[2]})

    def enum_fn(_s):
        return ["bond_reorder"], [_Bond(1, 2, 2)], np.array([1.0])

    def apply_fn(s, _j):
        return s                          # never establishes a new path

    p = propose(enum_fn, apply_fn, ctx, st, schedule=[("grow_new", "until_handoff")],
                rng=np.random.default_rng(0), lineage=lin,
                original_region_ids=old, max_handoff_steps=3)
    assert p.status == "UNSAT"
    assert p.stage == "no_connectivity_preserving_handoff"


def test_terminals_may_change_h_but_interior_may_not():
    from compose_v4.control.region_rewrite import context_preserved
    a, b = _chain(4), _chain(4)
    b.implicit_h_counts[0] = 3
    assert context_preserved(a, b, frozenset({0, 3}), terminals=frozenset({0}))
    assert not context_preserved(a, b, frozenset({0, 3}), terminals=frozenset())


def test_action_entirely_inside_context_is_refused_even_between_terminals():
    """Both endpoints being boundary terminals does not license rewriting a
    context-context bond; this was half the first sentinel run's failures."""
    ctx = _ctx(frozen={0, 1, 2}, locus={3},
               terminals=((0, 3, 1.0), (2, 3, 1.0)), interface="bridge", k=2)
    fams = ["bond_insert", "bond_insert"]
    acts = [_Bond(0, 2), _Bond(0, 3)]      # terminal-terminal vs terminal-locus
    idx, why = admissible_indices(fams, acts, ctx)
    assert idx == [1]
    assert why["context_only"] == 1


def test_new_atom_on_a_terminal_is_admissible():
    """The opening move of make-before-break. Its reach after removing the new
    slot is just the terminal, which lies inside the frozen context -- the
    context-only guard must not swallow it."""
    ctx = _ctx(frozen={0, 9}, locus={4, 5},
               terminals=((0, 4, 1.0), (9, 5, 1.0)), interface="bridge", k=2)
    fams = ["atom_insert", "bond_insert"]
    acts = [_Insert(20, 2, neighbors=((0, 1),)), _Bond(0, 9)]
    idx, why = admissible_indices(fams, acts, ctx)
    assert 0 in idx, "new atom on a terminal was refused; handoff is unreachable"
    assert 1 not in idx, "terminal-terminal bond must still be refused"
    assert why["context_only"] == 1
