"""Exact partial enumeration of the frozen marked law, one family at a time.

WHY
---
`build_ring_system_exact` is DETERMINISTIC: at each step it compiles a handful
of candidate descriptors and takes the top-ranked legal one. But it obtains them
by calling `enumerate_factorized_marked_law`, which constructs and validates
EVERY legal mark in EVERY family. Measured on real production payloads
(diagnostics/enum_profile.json):

    ring build wall     9.3-27.7 s
    full enumerations   9-10 per build
    enumeration share   98.9-99.6 % of build wall
    marks per call      140 -> 702, growing as atoms are added

So essentially the entire cost of a ring build is computing distributions that
are discarded. `cycle_close` alone is 56.5% of law-build time while carrying
0.53% of realized family mass (docs/LAZY_SAMPLER_RESULT.md).

EXACTNESS
---------
Under the frozen hierarchical factorization, verified analytically to 1.8e-15
over 15,179 marks:

    p(f, a) = p(f) * softmax over LEGAL coordinates within f

p(f) is the family base categorical restricted to families that HAVE a legal
action. Determining that alive set exactly is the expensive part, because for a
superset-masked family it requires resolving coordinates until one is proven
legal.

We never need it. The builder uses `probs` ONLY to rank candidates, and

    p(f1,a1) / p(f2,a2) = [exp(base_f1) * q_f1(a1)] / [exp(base_f2) * q_f2(a2)]

-- the alive-set normalizer is a common factor and cancels in every ratio. So
returning the UNNORMALIZED weight

    w(f, a) = exp(base_f) * q_f(a)

induces exactly the eager law's ordering, over any subset of families, without
building the families the consumer never asks about.

WHAT DIFFERS FROM THE EAGER PATH
--------------------------------
Ordering, selection and the resulting molecule are identical. The `rank`
integers the builder writes into its trace are NOT: eager ranks against the full
law, this ranks against the requested subset. Parity is therefore asserted on
`status` and `smiles`, never on trace rank annotations.

Nothing here reimplements chemistry. Masks, legality resolvers and the
coordinate->action map are the frozen production implementations, reached
through `hphi_lazy_helpers.make_helpers` -- the same bundle the GriDDD wave runs
on.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["enum_family_marks", "ring_phase_enumerator", "GROWTH_TABLES",
           "CLOSURE_TABLES", "RESTATE_TABLES"]

# Tables the ring builder actually consumes, by phase.
GROWTH_TABLES = ("grow_connected",)
CLOSURE_TABLES = ("cycle_insert", "cycle_attach", "bond_reorder")
RESTATE_TABLES = ("ring_system_restate", "atom_restate", "bond_reorder")


def enum_family_marks(model, helpers, state, time, want_tables, *,
                      encoded=None) -> tuple[list, list, np.ndarray]:
    """(families, actions, weights) for `want_tables`, in exact eager order.

    `weights` are proportional to the eager law's probabilities, not normalized
    to 1 -- see the module docstring. Ranking by them is exact.
    """
    import torch
    from compose_v4.experiments.hphi_lazy_family_scores import LAZY_SCORERS
    from compose_v4.experiments.hphi_lazy_sampler import FALLBACK, _FAMILY_TABLE
    from compose_v4.experiments.production_successor_kernel import _coordinate_action

    if encoded is None:
        batch, node, glob, pair, _c = helpers["encode"](state, float(time))
    else:
        batch, node, glob, pair = encoded
    with torch.no_grad():
        base = model._family_base_logits(batch, glob)[0].double().numpy()

    names = helpers["family_names"]
    want = set(want_tables)
    fams: list[str] = []
    acts: list[Any] = []
    ws: list[float] = []

    for fi, fam in enumerate(names):
        table = _FAMILY_TABLE.get(fam)
        if table is None or table not in want:
            continue
        scorer = LAZY_SCORERS.get(table)
        if scorer is None:
            continue
        mask = helpers["family_mask"](model, table, state, batch, pair, glob, node)
        if mask is None or mask is FALLBACK or not bool(mask.any()):
            continue
        with torch.no_grad():
            logits = scorer(model, node, glob, pair, batch)[0].double()
        shape = tuple(logits.shape)
        flat_mask = mask.reshape(-1).numpy().astype(bool)
        flat_log = logits.reshape(-1).numpy()
        idx = np.flatnonzero(flat_mask)
        if idx.size == 0:
            continue

        legality = helpers["legality"].get(table)
        coords, keep = [], []
        for j in idx:
            coord = tuple(int(v) for v in np.unravel_index(int(j), shape))
            # A superset mask must be resolved BEFORE the within-family softmax:
            # the eager law normalizes over LEGAL coordinates, so including
            # illegal ones would shift every weight in the family.
            if legality is not None and not legality(state, coord):
                continue
            coords.append(coord); keep.append(int(j))
        if not coords:
            continue
        lg = flat_log[np.asarray(keep)]
        q = np.exp(lg - lg.max()); q = q / q.sum()
        wf = float(np.exp(base[fi]))
        for coord, qi in zip(coords, q):
            try:
                rule, action = _coordinate_action(
                    model, state, batch, family_name=fam, table_name=table,
                    coordinate=coord)
            except Exception:
                continue
            fams.append(rule); acts.append(action); ws.append(wf * float(qi))

    return fams, acts, np.asarray(ws, dtype=float)


def ring_phase_enumerator(model, helpers, time, size: int):
    """An `enum_full_fn` for build_ring_system_exact that builds only the
    families that phase needs.

    PHASE IS DERIVED FROM STATE, NEVER FROM CALL ORDER. The builder's own
    `apply_fn` re-enters the enumerator to resolve an action, so a call counter
    is incremented twice per construction step and drifts out of phase; measured
    consequence was growth halting after ~3 atoms and every ring failing to
    close (propyl chains instead of rings, 0/6 parity). Deriving the phase from
    the state is idempotent and order-free:

        atoms added < size            -> GROWTH
        atoms added == size, no new ring -> CLOSURE
        atoms added == size, ring present -> electronic step
    """
    from compose_v4.chem.molecular_graph import molecular_graph_to_smiles

    ref = {"n0": None, "rings0": None}
    cache: dict[str, tuple] = {}

    def _n_real(st):
        from compose_v4.chem.molecular_graph import NULL_IDX
        return int((np.asarray(st.atom_types) != NULL_IDX).sum())

    def _rings(smi):
        if not smi:
            return 0
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        return int(m.GetRingInfo().NumRings()) if m is not None else 0

    def enum(st):
        try:
            key = molecular_graph_to_smiles(st)
        except Exception:
            key = None
        n = _n_real(st)
        r = _rings(key)
        if ref["n0"] is None:
            ref["n0"], ref["rings0"] = n, r
        added = n - int(ref["n0"])
        if added < int(size):
            tables = GROWTH_TABLES
        elif r <= int(ref["rings0"]):
            tables = CLOSURE_TABLES
        else:
            tables = RESTATE_TABLES
        ck = f"{key}|{','.join(tables)}"
        if key is not None and ck in cache:
            return cache[ck]
        out = enum_family_marks(model, helpers, st, time, tables)
        if key is not None:
            cache[ck] = out
        return out

    return enum
