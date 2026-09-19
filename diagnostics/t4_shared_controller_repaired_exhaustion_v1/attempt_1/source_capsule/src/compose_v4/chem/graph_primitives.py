"""
Pure graph-theory primitives over the heavy-atom MolecularGraph — no generative
or chemistry semantics. This is the lowest layer of the framework (below the
operator ISA); it depends only on the molecular-graph state structure, so the
ISA and process layers can import it without creating a cycle.
"""

from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    is_element,
    is_occupied,
)


def count_components(mg: MolecularGraph) -> int:
    """Count connected components in the heavy-atom graph (over OCCUPIED slots).

    Vectorized via scipy.sparse.csgraph.connected_components. ~50× faster than
    the previous Python-BFS implementation; used heavily during noising/sampling
    when connectivity preservation is enabled.

    Uses `is_occupied` (element OR scar), NOT `is_element`: a degree-2 SCAR carries
    connectivity through itself (u–SCAR–w keeps u and w in one component), so a scar
    must count as an occupied node here or scarring would spuriously fragment the
    graph. (This is the connectivity side of the is_element/is_occupied split.)

    Strategy:
      - Build sparse adjacency over occupied (non-NULL, includes scars) atoms only
      - scipy returns (n_components_in_subgraph, labels_per_node)
      - return that count
    """
    occ = is_occupied(mg.atom_types)
    n_real = int(occ.sum())
    if n_real == 0:
        return 0
    if n_real == 1:
        return 1
    real_indices = np.where(occ)[0]
    # Sub-bond matrix on real atoms only
    sub_bonds = mg.bonds[np.ix_(real_indices, real_indices)] > 0
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n_comp, _ = connected_components(
        csgraph=csr_matrix(sub_bonds.astype(np.int8)),
        directed=False,
        return_labels=True,
    )
    return int(n_comp)


# Ring-size buckets for topology features: 0=no-ring, 1=3, 2=4, 3=5, 4=6, 5=7+ (six classes).
RING_SIZE_BUCKETS = 6

# Fused-system-size-if-bonded buckets (per-PAIR): how many rings the fused SYSTEM would
# have if i,j were bonded. 0=no ring formed (different component / already bonded), 1/2/3
# = small fused system (allowed: benzene, naphthalene, fused-3), 4=>=4-ring system
# (EXCESS fusion -> the polycyclic cages we want to suppress). Five classes. This is the
# look-ahead form of the fusion signal: per-atom fusion-degree failed because the close
# decision is per-pair; this reads "would THIS close create a 4+ ring pile-up".
FUSED_SIZE_BUCKETS = 5


def _ring_size_to_bucket(sizes: np.ndarray) -> np.ndarray:
    """Map raw ring sizes -> bucket index {0:none/0, 1:3, 2:4, 3:5, 4:6, 5:7+}."""
    out = np.zeros_like(sizes, dtype=np.int64)
    out[sizes == 3] = 1
    out[sizes == 4] = 2
    out[sizes == 5] = 3
    out[sizes == 6] = 4
    out[sizes >= 7] = 5
    return out


def _fused_size_to_bucket(fused: np.ndarray) -> np.ndarray:
    """Map fused-system-size-if-bonded (ring count) -> {0:0, 1:1, 2:2, 3:3, >=4:4}."""
    out = np.zeros_like(fused, dtype=np.int64)
    out[fused == 1] = 1
    out[fused == 2] = 2
    out[fused == 3] = 3
    out[fused >= 4] = 4
    return out


def _smallest_ring_per_atom(adj: np.ndarray) -> np.ndarray:
    """Smallest ring containing each atom = shortest cycle THROUGH each vertex (girth-
    through-vertex), via BFS. O(V*(V+E)); robust to chemically-invalid graphs (pure
    topology). For each v, BFS records depth + which first-step branch each node hangs
    off; a non-tree edge joining two DIFFERENT branches closes a cycle through v of
    length depth[x]+depth[y]+1, and the min over such edges is the smallest ring.
    """
    from collections import deque

    n = adj.shape[0]
    nbrs = [np.nonzero(adj[v])[0].tolist() for v in range(n)]
    smallest = np.zeros(n, dtype=np.int64)
    for v in range(n):
        if len(nbrs[v]) < 2:  # leaf / isolated atom can't sit in a ring
            continue
        depth = {v: 0}
        branch = {v: -1}
        parent = {v: -1}
        best = 0
        dq = deque([v])
        while dq:
            x = dq.popleft()
            for y in nbrs[x]:
                if y == parent[x]:
                    continue
                if y not in depth:
                    depth[y] = depth[x] + 1
                    parent[y] = x
                    branch[y] = y if x == v else branch[x]
                    dq.append(y)
                elif branch[x] != branch[y] and branch[x] != -1 and branch[y] != -1:
                    clen = depth[x] + depth[y] + 1  # cycle through v across two branches
                    if best == 0 or clen < best:
                        best = clen
        smallest[v] = best
    return smallest


def compute_topology_features(
    mg: MolecularGraph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Topology features for the x̂0 predictor (Phase B). THREE bucketed arrays the model
    embeds as extra input channels:

      - atom_topo: (N,) int64 -- SMALLEST-RING-per-atom (ring size each atom sits in;
        bucket via `_ring_size_to_bucket`; 0 = not in a ring). Load-bearing for
        aromatization ("this atom is in a clean 6-ring -> aromatize candidate").
      - bond_topo: (N, N) int64 -- cycle-size-if-bonded: size of the smallest ring that
        would FORM if i,j were bonded (= heavy-graph distance + 1; `_ring_size_to_bucket`;
        0 = already bonded / unreachable). The macrocycle fix.
      - fused_bond_topo: (N, N) int64 -- fused-system-size-if-bonded: how many rings the
        fused SYSTEM would have if i,j were bonded (`_fused_size_to_bucket`, FUSED_SIZE_
        BUCKETS=5; bucket 4 = 4+ ring pile-up). Targets polycyclic OVER-fusion: per-pair
        look-ahead so the model can avoid closes that grow a system past 3 rings while
        still allowing good 2-3-ring fusion (naphthalene/quinoline).

    Pure graph-theory, all cheap: scipy all-pairs shortest path (bond_topo + component
    test), BFS girth (atom_topo), networkx bridge-finding + union-find (ring systems for
    fused_bond_topo). NO Horton minimum_cycle_basis, NO per-pair BFS. NULL atoms excluded.
    """
    n = mg.n_atoms
    # SCAR-aware (scar-restoration §3-G step 14): ring/aromaticity perception must
    # treat a SCAR as INERT — an in-ring scar is NOT a real cycle, and the
    # cycle-size / fused look-ahead ignore scar atoms. So the ring adjacency is
    # built over is_element (element only), NOT `!= NULL_IDX` (which would count a
    # scar as a real ring atom). `count_components` is the exception: it uses
    # is_occupied (a deg-2 scar DOES carry connectivity) — handled in that fn.
    is_real = is_element(mg.atom_types)
    adj = ((mg.bonds > 0) & is_real[:, None] & is_real[None, :]).astype(np.int8)

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    dist = shortest_path(csr_matrix(adj), directed=False, unweighted=True)  # (n,n); inf if unreachable

    # bond_topo: cycle-size-if-bonded = shortest heavy-graph distance + 1
    cycle_size = np.where(np.isfinite(dist) & (dist >= 2.0), dist + 1.0, 0.0).astype(np.int64)
    bond_topo = _ring_size_to_bucket(cycle_size)

    # atom_topo: smallest ring per atom (girth through vertex)
    atom_topo = _ring_size_to_bucket(_smallest_ring_per_atom(adj))

    # fused_bond_topo: per-pair fused-system-size-if-bonded.
    # Precompute per-atom ring-SYSTEM membership + the system's ring count (cyclomatic).
    sys_id = np.full(n, -1, dtype=np.int64)     # ring-system id per atom (-1 = not in any ring)
    sys_rings = np.zeros(n, dtype=np.int64)     # # independent rings in the atom's ring-system
    if int(adj.sum()) > 0:
        import networkx as nx

        graph = nx.from_numpy_array(adj)
        bridges = {frozenset(e) for e in nx.bridges(graph)}
        ring_graph = nx.Graph()
        ring_graph.add_nodes_from(range(n))
        for i, j in graph.edges():
            if frozenset((i, j)) not in bridges:  # non-bridge = ring bond
                ring_graph.add_edge(i, j)
        cid = 0
        for comp in nx.connected_components(ring_graph):
            sub = ring_graph.subgraph(comp)
            e_sys = sub.number_of_edges()
            if e_sys == 0:  # isolated atom, not a ring system
                continue
            cyclomatic = e_sys - sub.number_of_nodes() + 1  # independent rings in this fused system
            for a in comp:
                sys_id[a] = cid
                sys_rings[a] = cyclomatic
            cid += 1

    si, sj = sys_rings[:, None], sys_rings[None, :]
    idi, idj = sys_id[:, None], sys_id[None, :]
    in_i, in_j = (idi != -1), (idj != -1)
    same_sys = in_i & (idi == idj)                 # both in the SAME ring system -> grow it by 1
    both_diff = in_i & in_j & (idi != idj)         # in two different systems -> merge + 1
    one_only = in_i ^ in_j                          # exactly one in a system -> that one + 1
    fused = np.where(same_sys, si + 1, 0)
    fused = np.where(both_diff, si + sj + 1, fused)
    fused = np.where(one_only, np.where(in_i, si, sj) + 1, fused)
    fused = np.where(~(in_i | in_j), 1, fused)     # both in chains -> new isolated ring
    # a ring forms only if i,j are in the same component and not already bonded (i!=j)
    valid = np.isfinite(dist) & (~np.eye(n, dtype=bool)) & (adj == 0)
    fused = np.where(valid, fused, 0).astype(np.int64)
    fused_bond_topo = _fused_size_to_bucket(fused)

    return atom_topo, bond_topo, fused_bond_topo
