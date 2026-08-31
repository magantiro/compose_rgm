"""Region-local stochastic rewriting: the proposal IS the executable trajectory.

There is no "sample an endpoint y, then search for a path to it". A region
proposal is generated as an explicit primitive trajectory

    x = x_0 -> x_1 -> ... -> x_T = y,      a_t in A_M(x_t)

where A_M restricts the ordinary executor to the mutable locus, its newly
created descendants, and the declared boundary. Every intermediate is a
complete valid molecule and the preserved context C is byte-identical
throughout.

Likelihood is accumulated stepwise and explicitly:

    q(omega | x, M) = q(M | x) * prod_t q(a_t | x_t, M, phi_t)

`phi_t` is the phase (grow_new / handoff / prune_old / finish). UNSAT is an
explicit returned outcome, never a hidden retry-until-success loop -- an opaque
retry makes q(omega) incomputable and would sink the exact
importance-corrected story later.

Interface classes need different phase schedules:

    pendant  C connected, one boundary  -> prune_old then grow_new is safe
    segment  C connected, two boundaries -> same, the rest of C joins the anchors
    splitting C SPLITS into two          -> MAKE BEFORE BREAK. The old region is
             holding the molecule together, so a new connection must be
             established before the old one is relinquished.

Splitting is the k=2 case of connectivity-preserving k-terminal replacement; the
representation is deliberately k-general so k>2 needs no new compiler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


# ------------------------------------------------------------------- lineage
@dataclass
class Lineage:
    """Immutable atom identity, tracked across a trajectory.

    Measured from rewrite/operators.py: `apply_atom_delete` zeroes the slot IN
    PLACE (atom_types[v] = NULL_IDX) and `apply_atom_insert` writes into a
    specified free slot. So the executor never compacts or reindexes, and slot
    order is stable -- but a DELETED SLOT CAN BE REFILLED, and then "slot 7"
    names a different atom than it did before. Region membership and the frozen
    context are therefore held in lineage space and projected to slots.
    """

    slot_of: dict = field(default_factory=dict)   # lineage id -> current slot
    id_of: dict = field(default_factory=dict)     # current slot -> lineage id
    next_id: int = 0

    @classmethod
    def initial(cls, slots) -> "Lineage":
        lin = cls()
        for s in sorted(int(x) for x in slots):
            lin.slot_of[lin.next_id] = s
            lin.id_of[s] = lin.next_id
            lin.next_id += 1
        return lin

    def observe(self, family: str, action) -> "Lineage":
        """Update identity after one applied action."""
        slot_of, id_of = dict(self.slot_of), dict(self.id_of)
        nxt = self.next_id
        new = created_slot(action)
        if family == "atom_insert" and new is not None:
            prev = id_of.pop(new, None)          # slot reuse: retire the old id
            if prev is not None:
                slot_of.pop(prev, None)
            slot_of[nxt] = new
            id_of[new] = nxt
            nxt += 1
        elif family == "atom_delete":
            v = int(getattr(action, "v", -1))
            gone = id_of.pop(v, None)
            if gone is not None:
                slot_of.pop(gone, None)
        return Lineage(slot_of=slot_of, id_of=id_of, next_id=nxt)

    def slots(self, ids) -> frozenset:
        return frozenset(self.slot_of[i] for i in ids if i in self.slot_of)

    def ids(self, slots) -> frozenset:
        return frozenset(self.id_of[s] for s in slots if s in self.id_of)


# --------------------------------------------------------------- action reach
def touched_slots(action) -> frozenset:
    """Every atom slot an action reads or writes.

    Shapes come from rewrite/operators.py: bond ops carry (a, b) and reroute
    also (u, v); AtomInsert carries the new `slot` plus its neighbours; atom
    ops carry `v`.
    """
    s = set()
    for name in ("a", "b", "u", "v", "slot"):
        val = getattr(action, name, None)
        if isinstance(val, int):
            s.add(int(val))
    for nb in (getattr(action, "neighbors", ()) or ()):
        try:
            s.add(int(nb[0]))
        except (TypeError, IndexError):
            pass
    return frozenset(s)


def created_slot(action):
    """The slot an AtomInsert brings into existence, else None."""
    slot = getattr(action, "slot", None)
    if slot is None or getattr(action, "atom_type", None) is None:
        return None
    return int(slot)


# ------------------------------------------------------------------- context
@dataclass(frozen=True)
class RewriteContext:
    """A frozen preserved context and the locus a proposal may rewrite."""

    frozen: frozenset            # slots of C = x \ M, immutable
    locus: frozenset             # slots currently mutable (M + new descendants)
    terminals: tuple             # ((context_slot, region_slot, order), ...)
    interface: str               # 'pendant' | 'segment' | 'splitting' | 'multi'
    k_components: int            # connected components of C
    phase: str = "grow_new"

    @property
    def terminal_context_slots(self) -> frozenset:
        return frozenset(int(t[0]) for t in self.terminals)

    def with_locus(self, extra: int) -> "RewriteContext":
        return replace(self, locus=self.locus | {int(extra)})

    def with_phase(self, phase: str) -> "RewriteContext":
        return replace(self, phase=phase)


def context_from_region(region) -> RewriteContext:
    """Build the frozen context from a Region (see control/region.py)."""
    frozen = frozenset(range(region.n_atoms_total)) - frozenset(region.atoms)
    terminals = tuple((int(out), int(ins), float(o))
                      for (ins, out, o) in region.boundary)
    phase = "grow_new" if region.interface == "splitting" else "prune_old"
    return RewriteContext(frozen=frozen, locus=frozenset(region.atoms),
                          terminals=terminals, interface=region.interface,
                          k_components=region.n_context_components, phase=phase)


# -------------------------------------------------------------- admissibility
def admissible_indices(fams, acts, ctx: RewriteContext):
    """Indices of actions that touch only the locus, its descendants, or the
    declared boundary, and never mutate the preserved context.

    Returns (indices, rejection_counts) so the funnel is inspectable rather than
    a silent filter.
    """
    ok, why = [], {"frozen_touched": 0, "not_in_locus": 0, "context_only": 0}
    allowed = ctx.locus | ctx.terminal_context_slots
    for j, _f in enumerate(fams):
        a = acts[j]
        reach = touched_slots(a)
        new = created_slot(a)
        if new is not None:
            reach = reach - {new}          # a brand-new slot is not yet placed
        if reach & (ctx.frozen - ctx.terminal_context_slots):
            why["frozen_touched"] += 1
            continue
        if not reach or not (reach <= allowed):
            why["not_in_locus"] += 1
            continue
        # An action lying ENTIRELY inside the frozen context passes the checks
        # above whenever its atoms happen to be boundary terminals -- terminals
        # are in `allowed` -- yet a bond op between two terminals rewrites a
        # context-context bond, which the invariant forbids. Measured: this was
        # half the sentinel failures, reported as context_mutated after the
        # fact instead of being refused up front.
        # An action lying entirely inside the frozen context is refused -- but
        # ONLY if it creates nothing. An atom_insert attached to a single
        # terminal has reach {terminal} after its new slot is removed, which is
        # entirely inside the context, and that is exactly the opening move of
        # make-before-break. Refusing it made the handoff unreachable and
        # produced 12/12 no_connectivity_preserving_handoff.
        if new is None and reach and reach <= ctx.frozen:
            why["context_only"] += 1
            continue
        ok.append(j)
    return ok, why


# ---------------------------------------------------------------- invariants
def graph_connected(state) -> bool:
    import numpy as np
    b = np.asarray(state.bonds)
    n = int(b.shape[0])
    live = [i for i in range(n) if int(np.asarray(state.atom_types)[i]) != 0]
    if not live:
        return False
    seen, stack = set(), [live[0]]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        for w in range(n):
            if w not in seen and w in live and b[v][w] > 0:
                stack.append(w)
    return seen == set(live)


def context_preserved(before, after, frozen: frozenset,
                      terminals: frozenset = frozenset()) -> bool:
    """The INDUCED graph on C is preserved; boundary terminals may rewire.

    Requiring every attribute of every context atom to be byte-identical would
    make make-before-break impossible: attaching a new connector necessarily
    changes a terminal's implicit-H count. So:

      interior context atoms  identity AND their entire bond row are frozen
      boundary terminals      identity and bonds to OTHER context atoms are
                              frozen; bonds to the mutable/new region and the
                              resulting valence/H bookkeeping may change

    Executor guards still enforce that the complete molecule stays valid.
    """
    import numpy as np
    fs = sorted(int(i) for i in frozen)
    term = {int(t) for t in terminals}
    interior = [i for i in fs if i not in term]
    at_b, at_a = np.asarray(before.atom_types), np.asarray(after.atom_types)
    fc_b, fc_a = np.asarray(before.formal_charges), np.asarray(after.formal_charges)
    h_b, h_a = np.asarray(before.implicit_h_counts), np.asarray(after.implicit_h_counts)
    if at_a.shape[0] <= max(fs, default=-1):
        return False
    if not np.array_equal(at_b[fs], at_a[fs]):
        return False                      # element identity of C never changes
    if not np.array_equal(fc_b[fs], fc_a[fs]):
        return False
    if interior and not np.array_equal(h_b[interior], h_a[interior]):
        return False                      # only terminals may change H
    ab, bb = np.asarray(before.bonds), np.asarray(after.bonds)
    for i in fs:                          # every C-C bond is frozen
        for j in fs:
            if ab[i][j] != bb[i][j]:
                return False
    for i in interior:                    # interior gains no edge to anything new
        if not np.array_equal(ab[i], bb[i][: ab.shape[1]]):
            return False
    return True


def handoff_satisfied(state, ctx: RewriteContext, lineage: Lineage,
                      original_region_ids: frozenset) -> bool:
    """Do the preserved components already connect WITHOUT the old region?

    This is the make-before-break trigger, and it is topological rather than
    time-based: `grow_new` continues until the newly constructed material
    provides an alternative valid path joining every preserved component, and
    only then may the old region be relinquished.

    Stated over components rather than over a pair, so k > 2 needs no new
    compiler: the condition is simply that the new region spans all required
    preserved components.
    """
    import numpy as np
    b = np.asarray(state.bonds)
    types = np.asarray(state.atom_types)
    n = int(b.shape[0])
    old = {lineage.slot_of[i] for i in original_region_ids if i in lineage.slot_of}
    live = [i for i in range(n) if int(types[i]) != 0 and i not in old]
    ctx_slots = {int(c) for c in ctx.frozen} & set(live)
    if not ctx_slots:
        return False
    start = next(iter(ctx_slots))
    seen, stack = set(), [start]
    liveset = set(live)
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        for w in range(n):
            if w in liveset and w not in seen and b[v][w] > 0:
                stack.append(w)
    return ctx_slots <= seen


# ------------------------------------------------------------------- proposal
@dataclass
class Proposal:
    """One region-local trajectory, with its evaluable proposal density."""

    status: str                       # 'OK' | 'UNSAT'
    stage: str = ""
    conditional_path_logq: float = 0.0
    # sum_t log q(a_t | x_t, M, phi_t) ONLY. The complete proposal density also
    # needs q(M | x), any stochastic termination/phase probabilities, and the
    # mixture across overlapping proposal mechanisms. Do not call this q(omega).
    steps: list = field(default_factory=list)
    endpoint: object = None
    n_admissible_seen: list = field(default_factory=list)
    rejections: dict = field(default_factory=dict)
    lineage: object = None


def propose(enum_fn, apply_fn, ctx: RewriteContext, start_state, *,
            schedule, rng, gate_fn=None, lineage=None,
            original_region_ids=frozenset(), max_handoff_steps=24):
    """Run one region-local trajectory under an explicit phase schedule.

    `schedule` is a list of (phase, n_steps). `enum_fn(state) -> (fams, acts,
    probs)`; `apply_fn(state, j) -> state | None`. Each step samples ONE action
    from the R_theta-renormalised admissible set and records its exact
    probability, so `log_q` is the true proposal log-density of the realised
    path. Any dead end returns UNSAT with the phase that failed; nothing is
    retried silently.
    """
    import numpy as np
    st = start_state
    if lineage is None:
        live = [i for i in range(len(np.asarray(st.atom_types)))
                if int(np.asarray(st.atom_types)[i]) != 0]
        lineage = Lineage.initial(live)
    p = Proposal(status="OK")
    p.lineage = lineage
    for phase, n_steps in schedule:
        ctx = ctx.with_phase(phase)
        until_handoff = (n_steps == "until_handoff")
        budget = int(max_handoff_steps) if until_handoff else int(n_steps)
        for _i in range(budget):
            if until_handoff and handoff_satisfied(st, ctx, lineage,
                                                   original_region_ids):
                break
            fams, acts, probs = enum_fn(st)
            idx, why = admissible_indices(fams, acts, ctx)
            p.n_admissible_seen.append(len(idx))
            for k, v in why.items():
                p.rejections[k] = p.rejections.get(k, 0) + v
            if not idx:
                p.status, p.stage = "UNSAT", f"{phase}:no_admissible_action"
                return p
            w = np.array([float(probs[j]) for j in idx], float)
            if w.sum() <= 0:
                p.status, p.stage = "UNSAT", f"{phase}:zero_mass"
                return p
            w = w / w.sum()
            choice = int(rng.choice(len(idx), p=w))
            j = idx[choice]
            nxt = apply_fn(st, j)
            if nxt is None:
                p.status, p.stage = "UNSAT", f"{phase}:apply_failed"
                return p
            if not context_preserved(start_state, nxt, ctx.frozen,
                                     ctx.terminal_context_slots):
                p.status, p.stage = "UNSAT", f"{phase}:context_mutated"
                return p
            if not graph_connected(nxt):
                p.status, p.stage = "UNSAT", f"{phase}:disconnected"
                return p
            p.conditional_path_logq += math.log(max(float(w[choice]), 1e-300))
            new = created_slot(acts[j])
            if new is not None:
                ctx = ctx.with_locus(new)
            lineage = lineage.observe(fams[j], acts[j])
            p.lineage = lineage
            p.steps.append({"phase": phase, "family": fams[j],
                            "q": float(w[choice]), "n_admissible": len(idx)})
            st = nxt
        if until_handoff and not handoff_satisfied(st, ctx, lineage,
                                                   original_region_ids):
            p.status = "UNSAT"
            p.stage = "no_connectivity_preserving_handoff"
            return p
    if gate_fn is not None and not gate_fn(st):
        p.status, p.stage = "UNSAT", "gate"
        return p
    p.endpoint = st
    return p
