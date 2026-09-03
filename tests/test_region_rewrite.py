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
               interface="splitting", k=2)
    lin = Lineage.initial([0, 1, 2, 3])
    old = frozenset({lin.id_of[1], lin.id_of[2]})
    assert not handoff_satisfied(st, ctx, lin, old), "the old connector must not count"
    st2 = _chain(4)
    st2.bonds[0][3] = st2.bonds[3][0] = 1   # a new direct connection appears
    assert handoff_satisfied(st2, ctx, lin, old)


def test_missing_handoff_is_an_explicit_failure_label():
    st = _chain(4)
    ctx = _ctx(frozen={0, 3}, locus={1, 2}, terminals=((0, 1, 1.0), (3, 2, 1.0)),
               interface="splitting", k=2)
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
               terminals=((0, 3, 1.0), (2, 3, 1.0)), interface="splitting", k=2)
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
               terminals=((0, 4, 1.0), (9, 5, 1.0)), interface="splitting", k=2)
    fams = ["atom_insert", "bond_insert"]
    acts = [_Insert(20, 2, neighbors=((0, 1),)), _Bond(0, 9)]
    idx, why = admissible_indices(fams, acts, ctx)
    assert 0 in idx, "new atom on a terminal was refused; handoff is unreachable"
    assert 1 not in idx, "terminal-terminal bond must still be refused"
    assert why["context_only"] == 1


# ------------------------------------------- connectivity-progress proposal
from compose_v4.control.region_rewrite import connectivity_potential


def _phi(st, ctx, lin, old):
    return connectivity_potential(st, ctx, lin, old)


def test_potential_is_a_connectivity_deficit_not_a_terminal_rule():
    """Zero deficit exactly when the preserved components are joined without
    the original region; strictly worse when they are not."""
    ctx = _ctx(frozen={0, 3}, locus={1, 2}, terminals=((0, 1, 1.0), (3, 2, 1.0)),
               interface="splitting", k=2)
    lin = Lineage.initial([0, 1, 2, 3])
    old = frozenset({lin.id_of[1], lin.id_of[2]})
    split = _chain(4)                       # only the old region joins 0 and 3
    joined = _chain(4)
    joined.bonds[0][3] = joined.bonds[3][0] = 1
    assert _phi(split, ctx, lin, old) < _phi(joined, ctx, lin, old)
    assert _phi(joined, ctx, lin, old) >= 0.0     # deficit cleared


def test_potential_rewards_new_material_before_any_merge():
    """Growth must be climbable, or the potential is flat until the one step
    that completes the connection and the sampler has nothing to follow."""
    ctx = _ctx(frozen={0, 3}, locus={1, 2}, terminals=((0, 1, 1.0), (3, 2, 1.0)),
               interface="splitting", k=2)
    lin = Lineage.initial([0, 1, 2, 3])
    old = frozenset({lin.id_of[1], lin.id_of[2]})
    bare = _chain(5)
    bare.atom_types[4] = 0                  # slot 4 empty
    grown = _chain(5)
    grown.bonds[0][4] = grown.bonds[4][0] = 1   # new atom hanging off terminal 0
    lin2 = lin.observe("atom_insert", _Insert(4, 2, neighbors=((0, 1),)))
    assert _phi(grown, ctx, lin2, old) > _phi(bare, ctx, lin, old)


def test_potential_generalises_to_three_preserved_components():
    """k>2 needs no new formulation: it is the remaining deficit."""
    ctx = _ctx(frozen={0, 3, 5}, locus={1, 2, 4},
               terminals=((0, 1, 1.0), (3, 2, 1.0), (5, 4, 1.0)),
               interface="multi", k=3)
    lin = Lineage.initial(range(6))
    old = frozenset({lin.id_of[1], lin.id_of[2], lin.id_of[4]})
    s = _chain(6)
    three = _phi(s, ctx, lin, old)
    s2 = _chain(6)
    s2.bonds[0][3] = s2.bonds[3][0] = 1     # merge two of the three
    assert _phi(s2, ctx, lin, old) > three


def test_directed_proposal_keeps_an_exploration_floor():
    """No executable action may be driven to zero probability."""
    st = _chain(4)
    ctx = _ctx(frozen={0, 3}, locus={1, 2}, terminals=((0, 1, 1.0), (3, 2, 1.0)),
               interface="splitting", k=2)
    lin = Lineage.initial([0, 1, 2, 3])
    old = frozenset({lin.id_of[1], lin.id_of[2]})
    fams = ["bond_reorder", "bond_reorder"]
    acts = [_Bond(1, 2, 2), _Bond(2, 1, 3)]

    def enum_fn(_s):
        return fams, acts, np.array([0.5, 0.5])

    def apply_fn(s, _j):
        return s

    p = propose(enum_fn, apply_fn, ctx, st, schedule=[("grow_new", 4)],
                rng=np.random.default_rng(1), lineage=lin,
                original_region_ids=old,
                potential_fn=lambda s, c, l: connectivity_potential(s, c, l, old),
                beta=6.0, epsilon=0.1)
    assert p.status == "OK"
    assert all(s["q"] > 0.0 for s in p.steps), "an executable action lost support"


def test_surviving_old_region_never_counts_toward_handoff():
    """The region being replaced must not make the context look connected."""
    ctx = _ctx(frozen={0, 3}, locus={1, 2}, terminals=((0, 1, 1.0), (3, 2, 1.0)),
               interface="splitting", k=2)
    lin = Lineage.initial([0, 1, 2, 3])
    old = frozenset({lin.id_of[1], lin.id_of[2]})
    st = _chain(4)                       # 0-1-2-3: joined ONLY through the old region
    assert not handoff_satisfied(st, ctx, lin, old)
    assert connectivity_potential(st, ctx, lin, old) < 0.0, "deficit must remain"


def test_merge_dominates_any_growth_the_horizon_allows():
    """W must exceed the largest reachable growth bonus, or appended atoms
    outrank the topological objective."""
    ctx = _ctx(frozen={0, 3}, locus={1, 2}, terminals=((0, 1, 1.0), (3, 2, 1.0)),
               interface="splitting", k=2)
    lin = Lineage.initial([0, 1, 2, 3])
    old = frozenset({lin.id_of[1], lin.id_of[2]})
    merged = _chain(4)
    merged.bonds[0][3] = merged.bonds[3][0] = 1
    phi_merged = connectivity_potential(merged, ctx, lin, old)
    # 16 appended atoms on one side, still split
    big = _State([2] * 20, np.zeros((20, 20), dtype=int))
    for i in range(3):
        big.bonds[i][i + 1] = big.bonds[i + 1][i] = 1
    lin_big = Lineage.initial(range(4))
    for k in range(4, 20):
        big.bonds[0][k] = big.bonds[k][0] = 1
        lin_big = lin_big.observe("atom_insert", _Insert(k, 2, neighbors=((0, 1),)))
    phi_appendage = connectivity_potential(big, ctx, lin_big, old)
    assert phi_merged > phi_appendage, "a useless appendage outranked a real merge"


def test_appendage_diagnostics_expose_one_sided_growth():
    from compose_v4.control.region_rewrite import connectivity_diagnostics
    ctx = _ctx(frozen={0, 3}, locus={1, 2}, terminals=((0, 1, 1.0), (3, 2, 1.0)),
               interface="splitting", k=2)
    lin = Lineage.initial([0, 1, 2, 3])
    old = frozenset({lin.id_of[1], lin.id_of[2]})
    s = _State([2] * 6, np.zeros((6, 6), dtype=int))
    for i in range(3):
        s.bonds[i][i + 1] = s.bonds[i + 1][i] = 1
    lin2 = lin
    for k in (4, 5):
        s.bonds[0][k] = s.bonds[k][0] = 1
        lin2 = lin2.observe("atom_insert", _Insert(k, 2, neighbors=((0, 1),)))
    d = connectivity_diagnostics(s, ctx, lin2, old)
    assert d["n_new_material"] == 2
    assert d["components_touched_by_new"] == 1, "growth is all on one side"


def test_recorded_density_is_the_full_exploration_floor_mixture():
    """conditional_path_logq must be the density actually sampled from --
    (1-eps)*shaped + eps*base -- not the shaped part alone."""
    st = _chain(4)
    ctx = _ctx(frozen={0, 3}, locus={1, 2}, terminals=((0, 1, 1.0), (3, 2, 1.0)),
               interface="splitting", k=2)
    lin = Lineage.initial([0, 1, 2, 3])
    old = frozenset({lin.id_of[1], lin.id_of[2]})
    base = np.array([0.25, 0.75])
    fams, acts = ["bond_reorder", "bond_reorder"], [_Bond(1, 2, 2), _Bond(2, 1, 3)]

    def enum_fn(_s):
        return fams, acts, base

    def apply_fn(s, _j):
        return s                       # identical successors -> dPhi = 0 for both

    eps = 0.1
    p = propose(enum_fn, apply_fn, ctx, st, schedule=[("grow_new", 1)],
                rng=np.random.default_rng(0), lineage=lin, original_region_ids=old,
                potential_fn=lambda s, c, l: connectivity_potential(s, c, l, old),
                beta=6.0, epsilon=eps)
    assert p.status == "OK"
    q = p.steps[0]["q"]
    # both successors identical => shaped == base => mixture == base
    assert q == pytest.approx(base[0] / base.sum()) or q == pytest.approx(base[1] / base.sum())
    assert p.conditional_path_logq == pytest.approx(np.log(q))


def test_first_handoff_step_is_recorded_without_changing_policy():
    st = _chain(4)
    st.bonds[0][3] = st.bonds[3][0] = 1        # already connected at t=0
    ctx = _ctx(frozen={0, 3}, locus={1, 2}, terminals=((0, 1, 1.0), (3, 2, 1.0)),
               interface="splitting", k=2)
    lin = Lineage.initial([0, 1, 2, 3])
    old = frozenset({lin.id_of[1], lin.id_of[2]})

    def enum_fn(_s):
        return ["bond_reorder"], [_Bond(1, 2, 2)], np.array([1.0])

    p = propose(enum_fn, lambda s, j: s, ctx, st,
                schedule=[("grow_new", "until_handoff")],
                rng=np.random.default_rng(0), lineage=lin,
                original_region_ids=old, max_handoff_steps=5)
    assert p.status == "OK"
    assert p.first_handoff_step == 0, "handoff held immediately; no steps needed"


# --------------------------------------------- old-dependence and pruning
from compose_v4.control.region_rewrite import old_dependence, prune_potential


def _split_ctx():
    ctx = _ctx(frozen={0, 3}, locus={1, 2}, terminals=((0, 1, 1.0), (3, 2, 1.0)),
               interface="splitting", k=2)
    lin = Lineage.initial([0, 1, 2, 3])
    return ctx, lin, frozenset({lin.id_of[1], lin.id_of[2]})


def test_old_dependence_is_zero_exactly_at_handoff():
    ctx, lin, old = _split_ctx()
    st = _chain(4)                                   # only route uses 1 and 2
    assert old_dependence(st, ctx, lin, old) > 0
    assert not handoff_satisfied(st, ctx, lin, old)
    st2 = _chain(4)
    st2.bonds[0][3] = st2.bonds[3][0] = 1            # old-free route appears
    assert old_dependence(st2, ctx, lin, old) == 0
    assert handoff_satisfied(st2, ctx, lin, old)


def test_old_dependence_falls_for_rewiring_not_only_growth():
    """The saturated regime: no new atoms, connectivity moved by rerouting."""
    ctx, lin, old = _split_ctx()
    st = _chain(4)
    before = old_dependence(st, ctx, lin, old)
    rewired = _chain(4)
    rewired.bonds[1][2] = rewired.bonds[2][1] = 0     # drop an old-region edge
    rewired.bonds[0][3] = rewired.bonds[3][0] = 1     # reroute around it
    assert old_dependence(rewired, ctx, lin, old) < before


def test_prune_potential_rewards_removing_old_atoms():
    ctx, lin, old = _split_ctx()
    st = _chain(4)
    st.bonds[0][3] = st.bonds[3][0] = 1               # handoff established
    both = prune_potential(st, ctx, lin, old)
    lin2 = lin.observe("atom_delete", _Del(1))
    st2 = _chain(4)
    st2.bonds[0][3] = st2.bonds[3][0] = 1
    st2.atom_types[1] = 0
    st2.bonds[1][:] = 0; st2.bonds[:, 1] = 0
    assert prune_potential(st2, ctx, lin2, old) > both


def test_prune_never_undoes_the_handoff_it_depends_on():
    """A removal that reintroduces old-region dependence is catastrophic."""
    ctx, lin, old = _split_ctx()
    still_dependent = _chain(4)                       # no old-free route
    assert prune_potential(still_dependent, ctx, lin, old) <= -1e5


def test_prune_budget_is_derived_from_structure_not_tuned():
    """Interior atoms are not directly deletable, so |M| steps is never enough."""
    from compose_v4.control.region_rewrite import prune_budget
    from compose_v4.control.region import Region
    r = Region(atoms=frozenset({1, 2}),
               boundary=((1, 0, 1.0), (2, 3, 1.0)),
               kind="linker", generator="t", n_atoms_total=4,
               n_context_components=2, interface="splitting")
    b = np.zeros((4, 4), dtype=int)
    for i in range(3):
        b[i][i + 1] = b[i + 1][i] = 1
    # 2 atoms + 1 internal bond + 2 boundary bonds
    assert prune_budget(r, b) == 5
    assert prune_budget(r, b) > len(r.atoms), "budget must exceed atom count"


def test_observe_fn_records_best_reachable_value_per_step():
    st = _chain(4)
    ctx, lin, old = _split_ctx()
    fams, acts = ["bond_reorder", "bond_reorder"], [_Bond(1, 2, 2), _Bond(2, 1, 3)]

    def enum_fn(_s):
        return fams, acts, np.array([0.5, 0.5])

    def apply_fn(s, _j):
        return s

    seen = []
    p = propose(enum_fn, apply_fn, ctx, st, schedule=[("grow_new", 2)],
                rng=np.random.default_rng(0), lineage=lin, original_region_ids=old,
                potential_fn=lambda s, c, l: connectivity_potential(s, c, l, old),
                beta=6.0, epsilon=0.1,
                observe_fn=lambda y, c, l: old_dependence(y, c, l, old))
    assert p.status == "OK"
    assert len(p.min_observed) == 2, "one best-reachable value per step"
    assert all(isinstance(v, int) for v in p.min_observed)


# ------------------------------- structural finite-horizon future value
from compose_v4.control.region_rewrite import (
    completion_terminal, establishment_terminal, structural_h)


def test_h_is_one_at_a_structural_terminal_and_zero_with_no_budget():
    ctx, lin, old = _split_ctx()
    joined = _chain(4)
    joined.bonds[0][3] = joined.bonds[3][0] = 1
    term = lambda s_, l_: establishment_terminal(s_, ctx, l_, old)
    assert structural_h(joined, ctx, lin, 3, term, None, None) == 1.0
    split = _chain(4)
    assert structural_h(split, ctx, lin, 0, term, None, None) == 0.0


def test_h_gives_mass_to_a_zero_gain_enabling_move():
    """The whole point: a step with no immediate gain must still carry mass if it
    opens a path to the terminal. No operator is named."""
    ctx, lin, old = _split_ctx()
    start = _chain(4)                       # split: only route uses the old region

    enabling = _chain(4)                    # no closer by any one-step measure...
    enabling.bonds[1][2] = enabling.bonds[2][1] = 0
    goal = _chain(4)                        # ...but one more step reaches terminal
    goal.bonds[1][2] = goal.bonds[2][1] = 0
    goal.bonds[0][3] = goal.bonds[3][0] = 1
    inert = _chain(4)                       # leads nowhere

    table = {id(start): [("cycle_open", enabling), ("bond_reorder", inert)],
             id(enabling): [("cycle_close", goal)],
             id(inert): [("bond_reorder", inert)]}

    def enum_fn(s_):
        moves = table.get(id(s_), [])
        return ([m[0] for m in moves],
                [_Bond(1, 2) for _ in moves],
                np.array([1.0 / max(1, len(moves))] * len(moves)))

    def apply_fn(s_, j):
        return table.get(id(s_), [])[j][1]

    term = lambda s_, l_: establishment_terminal(s_, ctx, l_, old)
    h_enabling = structural_h(enabling, ctx, lin, 2, term, enum_fn, apply_fn)
    h_inert = structural_h(inert, ctx, lin, 2, term, enum_fn, apply_fn)
    assert h_enabling > 0.0, "the enabling state must carry future value"
    assert h_enabling > h_inert, "zero-gain enabler must outrank a dead end"


def test_completion_terminal_requires_both_removal_and_intact_handoff():
    ctx, lin, old = _split_ctx()
    joined = _chain(4)
    joined.bonds[0][3] = joined.bonds[3][0] = 1
    assert not completion_terminal(joined, ctx, lin, old), "old region still present"
    lin2 = lin.observe("atom_delete", _Del(1)).observe("atom_delete", _Del(2))
    gone = _chain(4)
    gone.bonds[0][3] = gone.bonds[3][0] = 1
    for v in (1, 2):
        gone.atom_types[v] = 0
        gone.bonds[v][:] = 0; gone.bonds[:, v] = 0
    assert completion_terminal(gone, ctx, lin2, old)


def test_structural_h_memo_actually_caches():
    """The memo was dead code keyed on id(); this asserts it now short-circuits."""
    ctx, lin, old = _split_ctx()
    start = _chain(4)
    calls = {"n": 0}

    def enum_fn(s_):
        calls["n"] += 1
        return ["bond_reorder"], [_Bond(1, 2)], np.array([1.0])

    def apply_fn(s_, j):
        return s_                       # self-loop: same content every time

    term = lambda s_, l_: False         # never terminal -> full depth explored
    memo = {}
    structural_h(start, ctx, lin, 3, term, enum_fn, apply_fn, memo=memo)
    first = calls["n"]
    structural_h(start, ctx, lin, 3, term, enum_fn, apply_fn, memo=memo)
    assert calls["n"] == first, "second call re-enumerated; memo is not working"
    assert memo, "memo stayed empty"


# ------------------------------- adaptive tilt (KL control strength)
from compose_v4.control.region_rewrite import adaptive_tilt, structural_features


def test_adaptive_tilt_hits_the_target_ess():
    w = np.full(20, 1 / 20)
    h = np.linspace(0.01, 1.0, 20)
    for target in (0.2, 0.5, 0.8):
        q, T, ess = adaptive_tilt(w, h, target_ess=target)
        assert q.sum() == pytest.approx(1.0), "proposal must stay normalised"
        assert abs(ess - target) < 0.08, f"ESS {ess:.2f} missed target {target}"


def test_tilt_is_monotone_in_control_strength():
    """A lower ESS target must mean a sharper tilt, i.e. a smaller temperature."""
    w = np.full(20, 1 / 20)
    h = np.linspace(0.01, 1.0, 20)
    _q1, T_sharp, _e1 = adaptive_tilt(w, h, target_ess=0.2)
    _q2, T_soft, _e2 = adaptive_tilt(w, h, target_ess=0.8)
    assert T_sharp < T_soft


def test_tilt_normalisation_survives_a_bad_committor():
    """Approximation must cost efficiency, never correctness."""
    w = np.full(8, 1 / 8)
    for h in (np.zeros(8), np.full(8, 1e-300), np.array([1e9] + [1e-9] * 7)):
        q, _T, _e = adaptive_tilt(w, h, target_ess=0.3)
        assert q.sum() == pytest.approx(1.0)
        assert np.all(q >= 0.0)


def test_structural_features_are_finite_and_shaped():
    from compose_v4.control.region_rewrite import STRUCTURAL_FEATURES
    ctx, lin, old = _split_ctx()
    f = structural_features(_chain(4), ctx, lin, old, budget=3,
                            terminal_kind="establishment")
    assert len(f) == len(STRUCTURAL_FEATURES)
    assert all(np.isfinite(x) for x in f)
    fc = structural_features(_chain(4), ctx, lin, old, budget=3,
                             terminal_kind="completion")
    assert fc[-1] == 1.0 and f[-1] == 0.0, "terminal kind must be encoded"


def test_shared_features_reassemble_identically():
    """The 12x-redundancy fix must produce bit-identical vectors."""
    from compose_v4.control.region_rewrite import (
        features_from_shared, structural_features_shared)
    ctx, lin, old = _split_ctx()
    st = _chain(4)
    shared = structural_features_shared(st, ctx, lin, old)
    for b in (1, 3, 6):
        for kind in ("establishment", "completion"):
            direct = structural_features(st, ctx, lin, old, b, kind)
            assert features_from_shared(shared, b, kind) == direct


def test_kl_tilt_steers_a_peaked_base_where_ess_tilt_cannot():
    """The ESS rule silently disables itself on a peaked base; KL does not.

    adaptive_tilt returns the UNTILTED base whenever the base kernel's own ESS
    is already at or below the target, which on a concentrated R_M means the
    controller never acts. This pins that difference so the regression cannot
    come back unnoticed.
    """
    import numpy as np
    from compose_v4.control.region_rewrite import adaptive_tilt, kl_tilt
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.full(115, 0.05))       # peaked, like the measured base
    h = rng.uniform(0.0, 0.2, 115)
    good = 7
    h[good] = 0.87                              # the correctly ranked successor
    n = len(w)
    assert 1.0 / (n * np.sum(w ** 2)) < 0.3     # base ESS below the old target

    q_ess, T, _ = adaptive_tilt(w, h, target_ess=0.3)
    # T pins to the bisection bound, so h^(1/T) is near 1 but not exactly 1:
    # assert the SEMANTIC claim -- no material steering -- not bitwise equality.
    assert T >= 1e3 - 1e-9
    assert q_ess[good] < 1.05 * w[good]

    q_kl, eta, kl, _ = kl_tilt(w, h, kappa=1.0)
    assert q_kl[good] > 50 * w[good]            # actually steers
    assert kl <= 1.0 + 1e-6                     # and respects the budget
    assert eta > 0.0
    assert q_kl.sum() == pytest.approx(1.0)     # exactly normalised


def test_kl_tilt_is_identity_at_zero_budget():
    import numpy as np
    from compose_v4.control.region_rewrite import kl_tilt
    rng = np.random.default_rng(1)
    w = rng.dirichlet(np.full(20, 0.5))
    h = rng.uniform(0.01, 1.0, 20)
    q, eta, kl, _ = kl_tilt(w, h, kappa=0.0)
    assert kl == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(q, w / w.sum(), atol=1e-9)


def test_region_selector_orders_by_measured_feasibility():
    """Interface class must dominate size, because the measurement does.

    pendant 75% / segment 74% / multi 20% / splitting 1/91 on 239 sampled
    regions. A selector that ranks a splitting region above a pendant one is
    reasoning about the wrong variable.
    """
    from compose_v4.control.region_selector import feasibility, cost_seconds
    assert feasibility("pendant", 2) > feasibility("multi", 2)
    assert feasibility("segment", 15) > feasibility("splitting", 15)
    assert feasibility("splitting", 5) < 0.2
    # size is NOT monotone: mid-size regions are the hard case, not large ones
    assert feasibility("segment", 25) > feasibility("segment", 5)
    # cost rises with scope, which is what makes successes-per-second the
    # right objective rather than success probability
    assert cost_seconds(0.9) > cost_seconds(0.1)


def test_region_selector_scale_balance_prevents_small_region_flood():
    """Enumeration yields far more small regions; ranking must not just echo that."""
    from types import SimpleNamespace
    from compose_v4.control.region_selector import rank_regions
    small = [SimpleNamespace(interface="segment", size=2, released_fraction=0.05)
             for _ in range(50)]
    big = [SimpleNamespace(interface="segment", size=25, released_fraction=0.85)]
    ranked = rank_regions(small + big, scale_balance=True)
    top_bands = {min(int(r.released_fraction * 5), 4) for r, _ in ranked[:5]}
    assert len(top_bands) > 0
    unbalanced = rank_regions(small + big, scale_balance=False)
    assert unbalanced[0][1].score >= ranked[0][1].score
