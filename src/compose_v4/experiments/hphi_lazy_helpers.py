"""Per-family masks, legality predicates and batch construction for the lazy sampler.

SINGLE SOURCE OF TRUTH. The benchmark and the SMC controller both call
`make_helpers`, so there is no second implementation to drift from the qualified
one. Everything here was verified by the gates in
`docs/LAZY_SAMPLER_RESULT.md`: superset containment per family, execution parity
on 48 coordinates across all eight families that carry legal marks, and analytic
full-law equality at 1.8e-15.

Three kinds of family, and the distinction is the whole point of the design:

  * EXACT AND CHEAP -- atom_delete, bond_reroute, bond_reorder, grow_connected.
    Their masks are graph arithmetic or come from `_graph_application_masks`, so
    the first weighted draw is legal and no resolver runs at all. These are 76%
    of realized family mass.
  * SUPERSET PLUS RESOLVER -- atom_restate and cycle_insert. A cheap structural
    superset, verified to contain every legal coordinate, with the frozen
    resolver deciding each drawn candidate.
  * BUILT ON DEMAND -- cycle_attach and ring_system_restate. Genuinely expensive
    to construct, but drawn ~2.4% and ~3.1% of the time, so the cost is paid on
    those draws rather than on every transition.

The atom-restate predicate includes the PRODUCTIVE clause -- the successor must
differ from the source -- taken from the enumerator that builds its mask today.
Omitting it would admit no-op restatements that the frozen law excludes.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch

__all__ = ["make_helpers"]

_MAXH = 4


def make_helpers(model, *, time_point: float, canonical_slots: int) -> dict[str, Any]:
    """Build the helper bundle the lazy sampler needs for one model."""
    from compose_v4.chem.molecular_graph import NULL_IDX, is_element
    from compose_v4.experiments.hphi_graph_encode import build_graph_only_batch
    from compose_v4.experiments.hphi_lazy_sampler import FALLBACK
    from compose_v4.experiments.production_successor_kernel import (
        MARK_RULE_NAMES, enumerate_factorized_marked_law,
    )
    from compose_v4.model import factorized_tracelet_rate_model as F
    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.rewrite.operators import (
        CycleCloseEdge, SemanticAtomRestate, resolve_cycle_close_edge,
        resolve_semantic_atom_restate_action,
    )
    from compose_v4.rewrite.semantic_atom_restate import (
        prepare_semantic_atom_restate_context,
    )
    from compose_v4.rewrite.semantic_cycle_close import (
        prepare_semantic_cycle_close_context,
    )

    macro_system = F.de_novo_rewrite_system()

    class Ctx:
        """Per-state scaffolding, each piece built the first time it is asked for."""

        __slots__ = ("st", "_graph", "_cc", "_ar", "_key")

        def __init__(self, st):
            self.st = st
            self._graph = self._cc = self._ar = self._key = None

        @property
        def graph_masks(self):
            if self._graph is None:
                topo, _c, _r = F.compute_topology_features(self.st)
                self._graph = F._graph_application_masks(
                    self.st, topo, compute_cyclic_graft=True)
            return self._graph

        @property
        def cc_context(self):
            if self._cc is None:
                self._cc = prepare_semantic_cycle_close_context(self.st)
            return self._cc

        @property
        def ar_context(self):
            if self._ar is None:
                self._ar = prepare_semantic_atom_restate_context(self.st)
            return self._ar

        @property
        def source_key(self):
            if self._key is None:
                self._key = canonical_state_key(self.st)
            return self._key

    ctxs: dict[int, Ctx] = {}

    def ctx_for(st):
        c = ctxs.get(id(st))
        if c is None:
            # Keyed by object identity and bounded, because a long SMC run would
            # otherwise accumulate one Ctx per visited state for the life of the
            # process. The scaffolding is cheap to rebuild; the leak is not.
            if len(ctxs) > 256:
                ctxs.clear()
            c = ctxs[id(st)] = Ctx(st)
        return c

    def build_batch(st, t):
        b = build_graph_only_batch([st], [float(t)])
        b.bonds = torch.from_numpy(np.asarray(st.bonds)).unsqueeze(0)
        b.formal_charges = torch.from_numpy(
            np.asarray(st.formal_charges)).unsqueeze(0)
        b.states = (st,)
        b.ring_topology_local_support_log_mass = None
        b.graft_mask = None
        b.graft_remove_neighbors = None
        return b

    def family_mask(model, table, st, batch, pair, glob, node):
        c = ctx_for(st)
        real = torch.from_numpy(np.asarray(is_element(st.atom_types), dtype=bool))
        n = real.shape[0]
        bonds = torch.from_numpy(np.asarray(st.bonds)).long()
        hyd = torch.from_numpy(np.asarray(st.implicit_h_counts)).long()
        charged = torch.from_numpy(np.asarray(st.formal_charges) != 0)
        czero = ~charged
        upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        orders = torch.arange(1, 4)

        if table == "atom_delete":
            m = torch.from_numpy(
                np.asarray(F.process_v2_atom_delete_mask(st), dtype=bool))
            if charged.any():
                nb = ((bonds != 0) & charged.unsqueeze(0)).any(dim=-1)
                m = m & czero & ~nb
            return m.unsqueeze(0)

        if table == "bond_reroute":
            (_dm, _ce, _cp, graft, removed, gsucc) = c.graph_masks
            m = torch.from_numpy(np.asarray(graft, dtype=bool))
            rem = torch.from_numpy(np.asarray(removed)).long()
            # The scorer's relational residual and the executor both read these,
            # so they are attached rather than recomputed downstream.
            batch.graft_mask = m.unsqueeze(0)
            batch.graft_remove_neighbors = rem.unsqueeze(0)
            batch.graft_successor_groups = (gsucc,)
            if charged.any():
                cz2 = czero.unsqueeze(1) & czero.unsqueeze(0)
                m = m & cz2 & (rem >= 0) & ~charged[rem.clamp_min(0)]
            return m.unsqueeze(0)

        if table == "bond_reorder":
            (_dm, cyc_edge, _cp, _g, _rn, _gs) = c.graph_masks
            ce = torch.from_numpy(np.asarray(cyc_edge, dtype=bool))
            old = bonds.unsqueeze(-1)
            new = orders.view(1, 1, 3)
            delta = new - old
            m = (upper.unsqueeze(-1) & (old >= 1) & (old <= 3) & (new != old)
                 & ~ce.unsqueeze(-1)
                 & ((hyd.unsqueeze(1).unsqueeze(-1) - delta) >= 0)
                 & ((hyd.unsqueeze(1).unsqueeze(-1) - delta) <= _MAXH)
                 & ((hyd.unsqueeze(0).unsqueeze(-1) - delta) >= 0)
                 & ((hyd.unsqueeze(0).unsqueeze(-1) - delta) <= _MAXH))
            if charged.any():
                m = m & (czero.unsqueeze(1) & czero.unsqueeze(0)).unsqueeze(-1)
            return m.unsqueeze(0)

        if table == "grow_connected":
            nnull = int(
                (torch.from_numpy(np.asarray(st.atom_types)) == NULL_IDX).sum())
            th = model.cnof_valences.view(1, 1, -1) - orders.view(1, -1, 1)
            m = (real.view(-1, 1, 1) & torch.tensor(nnull > 0)
                 & (hyd.view(-1, 1, 1) >= orders.view(1, -1, 1))
                 & (th >= 0) & (th <= _MAXH))
            if charged.any():
                m = m & czero.view(-1, 1, 1)
            return m.unsqueeze(0)

        if table == "atom_restate":
            m = real.view(-1, 1).expand(n, len(model.atom_vocabulary)).clone()
            if charged.any():
                m = m & czero.view(-1, 1)
            return m.unsqueeze(0)

        if table == "cycle_insert":
            m = torch.zeros(n, n, 3, dtype=torch.bool)
            pr = real.view(-1, 1) & real.view(1, -1)
            for k, o in enumerate((1, 2, 3)):
                m[:, :, k] = (upper & pr & (bonds == 0)
                              & (hyd.view(-1, 1) >= o) & (hyd.view(1, -1) >= o))
            if charged.any():
                m = m & (czero.unsqueeze(1) & czero.unsqueeze(0)).unsqueeze(-1)
            return m.unsqueeze(0)

        if table == "cycle_attach":
            m = torch.from_numpy(
                np.asarray(F._semantic_cycle_open_admission_mask(st), dtype=bool))
            if m.dim() == 2:
                m = m & upper
                if charged.any():
                    m = m & (czero.unsqueeze(1) & czero.unsqueeze(0))
            return m.unsqueeze(0)

        if table in ("ring_system_grow", "ring_system_delete", "grow_root"):
            # Structurally disabled or unreachable under the frozen
            # configuration: compute_ring_grow_support and
            # compute_ring_system_delete are both False, and grow_root applies
            # only to the null state. The family audit found ZERO legal
            # coordinates for all three across every state tested, and an empty
            # mask makes the sampler reject the family and redraw -- which is
            # the exact behaviour, since a family with no legal action carries
            # no probability under the hierarchical law.
            return torch.zeros(1, 1, dtype=torch.bool)

        if table == "ring_system_restate":
            try:
                g = F.enumerate_ring_restate_semantic_groups(st, system=macro_system)
                batch.ring_restate_actions = (g.actions,)
                batch.ring_restate_successor_group_ids = (g.successor_group_ids,)
                batch.ring_restate_successor_group_descriptors = (g.group_descriptors,)
                batch.ring_restate_successor_group_multiplicities = (
                    g.group_multiplicities,)
                _lg, rm = model._ring_restate_logits(batch, pair, glob)
                return rm
            except Exception:  # noqa: BLE001
                return FALLBACK

        return None

    def legal_atom_restate(st, coord):
        r = resolve_semantic_atom_restate_action(
            st, SemanticAtomRestate(int(coord[0]), int(coord[1])),
            context=ctx_for(st).ar_context)
        return bool(r is not None and r.admitted and r.successor is not None
                    and canonical_state_key(r.successor) != ctx_for(st).source_key)

    def legal_cycle_insert(st, coord):
        r = resolve_cycle_close_edge(
            st, CycleCloseEdge(int(coord[0]), int(coord[1]), int(coord[2]) + 1),
            context=ctx_for(st).cc_context)
        return bool(r is not None and r.admitted)

    def eager_fallback(st, table, rng):
        """Exact law restricted to the ALREADY-CHOSEN family.

        Family-CONDITIONED. Sampling the global law here would redraw a family
        that has already been drawn and bias toward whichever families dominate
        globally -- a distribution change, not a slow path.
        """
        law = enumerate_factorized_marked_law(model, st, float(time_point))
        marks = [m for m in law.marks if m.table_name == table]
        if not marks:
            return (None, None)
        pr = np.array([float(np.exp(m.log_probability)) for m in marks])
        mk = marks[int(rng.choice(len(pr), p=pr / pr.sum()))]
        return (mk.table_name, tuple(int(v) for v in mk.coordinate))

    # STATE-CONTEXT CACHE, the lazy equivalent of the old law cache.
    #
    # A lazy draw produces no law, so the law cache stops being populated. What
    # IS reusable is the encoder output: ~55 ms of a ~79 ms transition, and a
    # pure function of the state. A revisited state should therefore cost ~24 ms.
    #
    # Sized in STATES, not megabytes: the pair tensor is 48*48*256 floats, about
    # 2.4 MB, so 64 entries is ~154 MB against a 4.5 GiB container.
    #
    # 12 entries was tried first and returned only 7% (98.7 -> 91.7 s on the
    # extinction sentinel) although the eager law cache hits 425 of 768
    # transitions on the same source. So the repeats are NOT clustered enough
    # for a handful of slots -- particles revisit states many steps after first
    # seeing them -- and the cache has to span the working set rather than the
    # recent burst.
    ENCODE_CACHE_MAX = 64
    enc_cache: OrderedDict[Any, Any] = OrderedDict()

    def encode(st, t):
        from compose_v4.model.factorized_tracelet_rate_model import (
            molecular_state_cache_key,
        )

        key = molecular_state_cache_key(st)
        hit = enc_cache.get(key)
        if hit is not None:
            enc_cache.move_to_end(key)
            batch, node, glob, pair = hit
            # The batch carries per-family fields attached during a previous
            # draw. They are pure functions of the same state, so reusing them
            # is correct and is most of the point.
            return batch, node, glob, pair, True
        batch = build_batch(st, t)
        with torch.no_grad():
            node, glob, pair = model._encode_batch(batch)
        enc_cache[key] = (batch, node, glob, pair)
        if len(enc_cache) > ENCODE_CACHE_MAX:
            enc_cache.popitem(last=False)
        return batch, node, glob, pair, False

    return {
        "build_batch": build_batch,
        "encode": encode,
        "family_names": list(MARK_RULE_NAMES),
        "family_mask": family_mask,
        "legality": {"atom_restate": legal_atom_restate,
                     "cycle_insert": legal_cycle_insert},
        "eager_fallback": eager_fallback,
        "ctx_for": ctx_for,
    }
