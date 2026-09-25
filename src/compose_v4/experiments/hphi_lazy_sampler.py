"""Family-first lazy sampler: one exact draw from the frozen marked law.

THE ARCHITECTURAL POINT
-----------------------
The eager path builds EVERY family's action table and then samples one mark:

    build 8 families  ->  prove thousands of unused actions  ->  choose one

Cycle-close alone is 56.5% of that construction time and carries 0.53% of
realized family mass, so it was being built roughly two hundred times for every
draw that used it. This path instead does

    encode once  ->  draw the family  ->  build ONLY that family  ->  draw a
    coordinate

which is exact under the frozen `hierarchical` factorization, where

    p(f, a) = p(f) * softmax over LEGAL coordinates within f

verified analytically to 1.8e-15 over 15,179 marks.

WHY THIS IS EXACT
-----------------
Two rejection arguments, both standard and both needed:

* FAMILY. p(f) is the base categorical restricted to families that HAVE a legal
  action. Drawing from the unrestricted base categorical and redrawing when the
  chosen family turns out empty reproduces that conditional exactly. A family is
  only ever declared empty by exhausting its candidates, never by a budget.

* COORDINATE. Within a family, drawing candidates in weighted random order and
  taking the first legal one is exactly the weighted categorical conditioned on
  the legal subset -- the Plackett-Luce first-element property. Sampling WITHOUT
  replacement also means a rejected coordinate is never paid for twice, and
  exhausting the list is what proves the family empty.

The raw scores come from `hphi_lazy_family_scores`, each verified BIT-FOR-BIT
against `_action_tables` on real states, including the two residuals that the
runtime subclasses add and that are invisible from the base class.

WHAT IS NOT CLAIMED
-------------------
This does NOT reproduce the eager sampler's RNG trajectory. It is a different
exact algorithm for the same law; the contract is
p_lazy(f, a | x) == p_eager(f, a | x), not identical random-number consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = [
    "FALLBACK",
    "LazyDraw",
    "fixed_size_allowed_families",
    "sample_one_transition",
]


_SIZE_CHANGING_FAMILIES = frozenset(
    {
        "grow_root",
        "atom_insert",
        "grow_connected",
        "atom_delete",
        "ring_system_grow",
        "ring_system_delete",
    }
)


def fixed_size_allowed_families(family_names: tuple[str, ...]) -> frozenset[str]:
    """Retain all deployed primitive families proven to preserve atom cardinality.

    The executor-level check in the H40 controller remains the final guard: an
    unexpectedly size-changing mark from a retained family must fail loudly.
    """
    if len(family_names) != len(set(family_names)):
        raise ValueError("model family names must be unique")
    unknown = set(family_names).difference(_FAMILY_TABLE)
    if unknown:
        raise ValueError(f"unknown model families: {sorted(unknown)}")
    return frozenset(family_names) - _SIZE_CHANGING_FAMILIES


class _Fallback:
    """Sentinel: this family needs the exact eager law, not an empty verdict."""

    def __repr__(self) -> str:
        return "FALLBACK"


_FALLBACK = _Fallback()
FALLBACK = _FALLBACK

#: Families in `MARK_RULE_NAMES` order carry a table of the same name except
#: where the historical table name differs from the family name.
#: FAMILY name -> TABLE name. These differ: the family is `atom_insert` while
#: its action table is `grow_connected`. Getting this wrong does not degrade
#: gracefully -- it raised on 25.9% of draws, the second-largest family.
_FAMILY_TABLE = {
    "grow_root": "grow_root",
    "atom_insert": "grow_connected",
    "grow_connected": "grow_connected",
    "atom_delete": "atom_delete",
    "atom_restate": "atom_restate",
    "bond_reorder": "bond_reorder",
    "bond_reroute": "bond_reroute",
    "cycle_insert": "cycle_insert",
    "cycle_attach": "cycle_attach",
    "ring_system_grow": "ring_system_grow",
    "ring_system_delete": "ring_system_delete",
    "ring_system_restate": "ring_system_restate",
}


@dataclass
class LazyDraw:
    """One sampled mark plus the accounting the benchmark needs."""

    table: str | None
    coordinate: tuple[int, ...] | None
    family_redraws: int = 0
    coordinate_rejections: int = 0
    resolver_calls: int = 0
    seconds_encode: float = 0.0
    seconds_family: float = 0.0
    seconds_mask: float = 0.0
    seconds_resolver: float = 0.0
    seconds_total: float = 0.0
    used_fallback: bool = False
    encode_cached: bool = False
    empty_families: list = field(default_factory=list)


def _plackett_luce_order(weights: np.ndarray, rng) -> np.ndarray:
    """Weighted random order. The first LEGAL element is the exact draw.

    Implemented by the Gumbel top-k trick, which is equivalent to repeated
    weighted sampling without replacement: adding independent Gumbel noise to
    log-weights and sorting descending yields a Plackett-Luce ordering. The
    first legal element of that ordering is distributed proportional to weight
    restricted to the legal subset, which is the distribution we need -- and it
    costs one sort rather than a resample per rejection.
    """
    g = rng.gumbel(size=weights.shape)
    return np.argsort(-(np.log(np.maximum(weights, 0.0) + 1e-300) + g))


def sample_one_transition(
    model, state, time, rng, *, helpers, allowed_families: frozenset[str] | None = None
) -> LazyDraw:
    """Draw ONE mark from the frozen law, building only what is needed.

    `helpers` supplies the per-family mask and legality callables so this module
    stays free of chemistry imports; the caller wires them from the frozen
    implementations, never from reimplementations. An optional family set
    removes excluded families before the categorical is normalized; the
    default retains the unrestricted deployed law.
    """
    import time as _t

    from compose_v4.experiments.hphi_lazy_family_scores import LAZY_SCORERS

    out = LazyDraw(table=None, coordinate=None)
    t_start = _t.perf_counter()

    # ---- encode once ---------------------------------------------------
    t0 = _t.perf_counter()
    enc = helpers.get("encode")
    if enc is not None:
        batch, node, glob, pair, cached = enc(state, time)
        out.encode_cached = cached
    else:
        batch = helpers["build_batch"](state, time)
        with torch.no_grad():
            node, glob, pair = model._encode_batch(batch)
    out.seconds_encode = _t.perf_counter() - t0

    # ---- family draw, with EXACT empty-family rejection -----------------
    t0 = _t.perf_counter()
    with torch.no_grad():
        base = model._family_base_logits(batch, glob)[0].double()
    names = helpers["family_names"]
    if allowed_families is not None:
        unknown = allowed_families.difference(names)
        if unknown:
            raise ValueError(f"allowed_families contains unknown model families: {sorted(unknown)}")
    alive = np.asarray(
        [allowed_families is None or name in allowed_families for name in names],
        dtype=bool,
    )
    out.seconds_family = _t.perf_counter() - t0

    while alive.any():
        t0 = _t.perf_counter()
        lg = base.clone()
        lg[~torch.from_numpy(alive)] = float("-inf")
        p = torch.softmax(lg, dim=-1).numpy()
        fi = int(rng.choice(len(p), p=p / p.sum()))
        out.seconds_family += _t.perf_counter() - t0

        family = names[fi]
        table = _FAMILY_TABLE.get(family)
        scorer = LAZY_SCORERS.get(table)
        if table is None:
            raise RuntimeError(f"no table registered for family {family!r}")
        if scorer is None:
            # No scorer: ask the helper whether the family is empty. It may
            # legitimately be structurally disabled. Never assume either way --
            # assuming empty removes probability mass, assuming non-empty
            # crashes mid-trajectory as ring_system_grow did.
            probe = helpers["family_mask"](
                model, table, state, batch, pair, glob, node)
            if probe is None or (probe is not _FALLBACK and not bool(probe.any())):
                alive[fi] = False
                out.family_redraws += 1
                out.empty_families.append(family)
                continue
            raise RuntimeError(
                f"family {family!r} has legal actions but no lazy scorer")

        # ---- build ONLY this family ------------------------------------
        t0 = _t.perf_counter()
        mask = helpers["family_mask"](model, table, state, batch, pair, glob, node)
        out.seconds_mask += _t.perf_counter() - t0
        if mask is _FALLBACK:
            # This family has no lazy construction yet. Fall back to the exact
            # eager law rather than treating it as empty: dropping a family that
            # HAS legal actions would remove real probability mass and silently
            # change the law. Slow and correct beats fast and wrong.
            out.used_fallback = True
            fb = helpers["eager_fallback"](state, table, rng)
            if fb[0] is None:
                # No legal mark in this family: it is EMPTY, so reject the
                # family and redraw. Returning "no mark" here would instead be
                # read by the controller as a dead end and would kill the
                # particle -- a silent trajectory change, far worse than a raise.
                alive[fi] = False
                out.family_redraws += 1
                out.empty_families.append(family)
                continue
            out.table, out.coordinate = fb
            break
        if mask is None or not bool(mask.any()):
            alive[fi] = False
            out.family_redraws += 1
            out.empty_families.append(family)
            continue

        with torch.no_grad():
            logits = scorer(model, node, glob, pair, batch)[0].double()

        flat_mask = mask.reshape(-1).numpy().astype(bool)
        flat_logits = logits.reshape(-1).numpy()
        idx = np.flatnonzero(flat_mask)
        w = np.exp(flat_logits[idx] - flat_logits[idx].max())

        legality = helpers["legality"].get(table)
        if legality is None:
            # The mask IS exact for this family, so the first weighted draw is
            # legal by construction and no resolver call is needed.
            chosen = int(idx[int(rng.choice(len(idx), p=w / w.sum()))])
            out.table = table
            out.coordinate = tuple(
                int(v) for v in np.unravel_index(chosen, tuple(logits.shape)))
            break

        # The mask is a SUPERSET: draw in weighted order and resolve exactly.
        order = _plackett_luce_order(w, rng)
        found = None
        for pos in order:
            coord = tuple(int(v) for v in np.unravel_index(
                int(idx[pos]), tuple(logits.shape)))
            t0 = _t.perf_counter()
            ok = legality(state, coord)
            out.seconds_resolver += _t.perf_counter() - t0
            out.resolver_calls += 1
            if ok:
                found = coord
                break
            out.coordinate_rejections += 1
        if found is None:
            # Exhausted: the family is PROVEN empty, not assumed so.
            alive[fi] = False
            out.family_redraws += 1
            out.empty_families.append(family)
            continue
        out.table = table
        out.coordinate = found
        break

    out.seconds_total = _t.perf_counter() - t_start
    return out
