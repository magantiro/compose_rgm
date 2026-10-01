"""Structural geometry of an executable trajectory.

COMPOSE defines local and global from WHAT CHANGED IN THE MOLECULAR GRAPH, not
from fingerprint distance. Morgan Tanimoto stays a benchmark constraint and a
descriptor-space diagnostic, but it does not decide move scale: a molecule can
gain a whole phenyl ring and keep high Tanimoto on a large scaffold, while a few
atom-type changes can perturb many bits without touching the recognisable
scaffold. Neither tells you whether a change was one coherent region replacement,
a ring expansion, a linker swap, or ten scattered edits.

Three complementary quantities, all computed in STATE SPACE against persistent
lineage, so they describe the executed trajectory rather than a re-matching
heuristic:

  1 how much of the original molecule changed   changed_fraction, inserted, deleted
  2 was it coherent or scattered                n_changed_regions, largest region
  3 what topology changed                       rings, ring systems, bridges, size

Because every COMPOSE intermediate is itself a complete molecule with
provenance, a trajectory also has an EXCURSION -- max_t D(x_0, x_t) -- which
separates "small refinement throughout" from "restructured heavily, then
returned somewhere endpoint-similar". Endpoint displacement alone cannot see
that, and it is exactly where pathwise constraints bite.

The second axis is kinetic: D_R(omega) = -sum_t log R_theta(x_{t+1} | x_t).
Structural displacement and kinetic difficulty are independent, giving four
regimes; ring expansion is structurally moderate but kinetically remote, while a
half-scaffold swap is structurally global.
"""

from __future__ import annotations

import numpy as np


def _live(state) -> list:
    t = np.asarray(state.atom_types)
    return [i for i in range(int(t.shape[0])) if int(t[i]) != 0]


def _edges(state) -> set:
    b = np.asarray(state.bonds)
    live = set(_live(state))
    out = set()
    for i in live:
        for j in live:
            if i < j and b[i][j] > 0:
                out.add((i, j))
    return out


def atom_environment(state, slot: int, radius: int = 1) -> tuple:
    """Identity plus local bonding environment of one atom.

    radius 0 is the atom alone; radius 1 adds the sorted multiset of
    (neighbour element, bond order), which is what makes "this atom's
    environment changed" mean something chemical rather than positional.
    """
    at = np.asarray(state.atom_types)
    fc = np.asarray(state.formal_charges)
    hh = np.asarray(state.implicit_h_counts)
    b = np.asarray(state.bonds)
    core = (int(at[slot]), int(fc[slot]), int(hh[slot]))
    if radius <= 0:
        return core
    nbrs = sorted((int(at[j]), int(b[slot][j]))
                  for j in range(int(at.shape[0]))
                  if b[slot][j] > 0 and int(at[j]) != 0)
    return core + (tuple(nbrs),)


def _bridges(state) -> set:
    """Edges whose removal disconnects; the non-bridge subgraph carries rings."""
    adj = {v: set() for v in _live(state)}
    for (i, j) in _edges(state):
        adj[i].add(j); adj[j].add(i)
    disc, low, out = {}, {}, set()
    counter = [0]

    def dfs(u, parent):
        disc[u] = low[u] = counter[0]; counter[0] += 1
        for w in adj[u]:
            if w == parent:
                continue
            if w not in disc:
                dfs(w, u)
                low[u] = min(low[u], low[w])
                if low[w] > disc[u]:
                    out.add((min(u, w), max(u, w)))
            else:
                low[u] = min(low[u], disc[w])

    for v in adj:
        if v not in disc:
            dfs(v, None)
    return out


def topology(state) -> dict:
    """Coarse structural descriptors, straight off the adjacency."""
    live = _live(state)
    edges = _edges(state)
    bridges = _bridges(state)
    ring_edges = edges - bridges
    adj = {v: set() for v in live}
    for (i, j) in ring_edges:
        adj[i].add(j); adj[j].add(i)
    seen, systems = set(), 0
    for v in live:
        if v in seen or not adj[v]:
            continue
        systems += 1
        stack = [v]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj[u] - seen)
    return {"n_heavy": len(live), "n_bonds": len(edges),
            "cycle_rank": len(edges) - len(live) + 1,
            "n_ring_systems": systems, "n_ring_atoms": len(seen),
            "n_bridges": len(bridges)}


def structural_displacement(s0, st, lin0, lint, radius: int = 1) -> dict:
    """What changed between two states of one lineage-tracked trajectory."""
    surviving = [i for i in lin0.slot_of if i in lint.slot_of]
    changed_ids = [i for i in surviving
                   if atom_environment(s0, lin0.slot_of[i], radius)
                   != atom_environment(st, lint.slot_of[i], radius)]
    deleted = [i for i in lin0.slot_of if i not in lint.slot_of]
    inserted = [i for i in lint.slot_of if i not in lin0.slot_of]
    n0 = max(1, len(lin0.slot_of))

    # coherence: connected components of the changed induced subgraph in x_t,
    # counting changed originals together with everything newly inserted
    focus = {lint.slot_of[i] for i in changed_ids} | {lint.slot_of[i] for i in inserted}
    b = np.asarray(st.bonds)
    seen, comps = set(), []
    for v in focus:
        if v in seen:
            continue
        comp, stack = set(), [v]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u); seen.add(u)
            for w in focus:
                if w not in comp and b[u][w] > 0:
                    stack.append(w)
        comps.append(comp)
    largest = max((len(c) for c in comps), default=0)

    t0, t1 = topology(s0), topology(st)
    return {
        "changed_fraction": len(changed_ids) / n0,
        "n_changed_originals": len(changed_ids),
        "n_inserted": len(inserted), "n_deleted": len(deleted),
        "n_changed_regions": len(comps),
        "largest_changed_region": largest,
        "largest_changed_fraction": largest / n0,
        "coherence": (largest / len(focus)) if focus else 1.0,
        "d_heavy": t1["n_heavy"] - t0["n_heavy"],
        "d_cycle_rank": t1["cycle_rank"] - t0["cycle_rank"],
        "d_ring_systems": t1["n_ring_systems"] - t0["n_ring_systems"],
        "d_ring_atoms": t1["n_ring_atoms"] - t0["n_ring_atoms"],
        "topology_changed": any(t0[k] != t1[k] for k in
                                ("cycle_rank", "n_ring_systems", "n_ring_atoms")),
    }


def trajectory_excursion(states, lineages, radius: int = 1) -> dict:
    """max_t D(x_0, x_t) versus the endpoint, over a lineage-tracked path.

    A large excursion with a small endpoint displacement means the trajectory
    restructured heavily and came back -- invisible to endpoint-only analysis,
    and precisely the case pathwise constraints exist for.
    """
    if not states:
        return {}
    per_t = [structural_displacement(states[0], s, lineages[0], l, radius)
             for s, l in zip(states, lineages)]
    end = per_t[-1]
    return {
        "endpoint_changed_fraction": end["changed_fraction"],
        "excursion_changed_fraction": max(d["changed_fraction"] for d in per_t),
        "endpoint_largest_region": end["largest_changed_fraction"],
        "excursion_largest_region": max(d["largest_changed_fraction"] for d in per_t),
        "topology_changed_at_any_t": any(d["topology_changed"] for d in per_t),
        "topology_changed_at_endpoint": end["topology_changed"],
        "per_step": per_t,
    }


def kinetic_difficulty(step_log_r) -> float:
    """D_R(omega) = -sum_t log R_theta(x_{t+1} | x_t).

    The second axis. A transformation can be structurally modest and still be
    effectively unreachable: the measured 7->8 ring expansion sits at 1e-8 to
    1e-14 over two or three valid primitive edits.
    """
    return -float(sum(step_log_r))
