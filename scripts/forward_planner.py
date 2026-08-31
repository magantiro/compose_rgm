"""Exact target-conditioned planner: seed -> IVG winner, forward, all legal.

This is program synthesis, not molecular optimization. The destination is known,
so the heuristic is EXACT GRAPH DIFFERENCE in operator units -- not Tanimoto,
not shared bits, not docking, not R_theta. A* tolerates a step that temporarily
increases the discrepancy, which greedy search cannot, and which the witness
route provably requires (it opens the scaffold before rebuilding it).

    D(x, x*) = (atoms to add) + (atoms to remove)
             + (bonds to add) + (bonds to remove)

computed from the maximum common substructure, so it is a real operator-unit
deficit rather than a proxy.

Cost control: the full legal fiber is ~5-17s per state, and MCS is ~0.1-1s per
candidate, so candidates are pre-ranked by a cheap counting proxy and only the
top-K get an exact MCS evaluation.
"""
import json, sys, time, heapq
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, 'src')
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
RDLogger.DisableLog('rdApp.*')
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.fiber import enumerate_action_fiber
from compose_v4.rewrite.kernel import de_novo_rewrite_system

CELL = sys.argv[1] if len(sys.argv) > 1 else "parp1_s0"
MAX_EXPAND = int(sys.argv[2]) if len(sys.argv) > 2 else 120
TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 24

cc = json.load(open('diagnostics/route_cores.json'))[CELL]
seed, target = cc["seed"], cc["target"]
m_t = Chem.MolFromSmiles(target)
T_can = Chem.MolToSmiles(m_t)
T_heavy, T_bonds = m_t.GetNumHeavyAtoms(), m_t.GetNumBonds()
sysm = de_novo_rewrite_system()

def cheap(m):
    """Counting proxy: how far off are atom and bond counts."""
    return abs(m.GetNumHeavyAtoms() - T_heavy) + abs(m.GetNumBonds() - T_bonds)

def exact_D(m):
    """Operator-unit deficit via MCS against the target."""
    try:
        r = rdFMCS.FindMCS([m, m_t], timeout=2, ringMatchesRingOnly=False,
                           completeRingsOnly=False,
                           atomCompare=rdFMCS.AtomCompare.CompareElements,
                           bondCompare=rdFMCS.BondCompare.CompareOrder)
        na, nb = r.numAtoms, r.numBonds
    except Exception:
        return cheap(m) * 2
    return ((m.GetNumHeavyAtoms() - na) + (T_heavy - na)
            + (m.GetNumBonds() - nb) + (T_bonds - nb))

def _D_worker(smi):
    return exact_D(Chem.MolFromSmiles(smi))


def succ(smi):
    """DEDUPED successors. The fiber returns one MarkedTransition per (rule,
    action) coordinate, so the same successor molecule appears many times over:
    measured 1861 transitions collapsing to 157 unique states on the seed.
    Without dedup the top-K selection filled up with duplicates of one molecule
    and A* degenerated into a single chain (|open| stuck at 1)."""
    st = pad_molecular_graph(smiles_to_molecular_graph(smi), 48)
    return sorted({mt.successor_key for mt in enumerate_action_fiber(st, system=sysm)
                   if mt.successor_key and mt.successor_key != smi})

s0 = Chem.MolToSmiles(Chem.MolFromSmiles(seed))
h0 = exact_D(Chem.MolFromSmiles(s0))
print(f"  {CELL}: seed D={h0} -> target (heavy {T_heavy}, bonds {T_bonds})")
W_ASTAR = 6
openq = [(W_ASTAR * h0, 0, s0, [s0])]
best_seen = {s0: 0}
t0 = time.time(); expanded = 0; best = (h0, s0, [s0])
while openq and expanded < MAX_EXPAND:
    f, g, cur, path = heapq.heappop(openq)
    if cur == T_can:
        print(f"\n  *** REACHED TARGET in {g} legal forward edits ***")
        best = (0, cur, path); break
    expanded += 1
    kids = succ(cur)
    scored = sorted(((cheap(Chem.MolFromSmiles(k)), k) for k in kids
                     if Chem.MolFromSmiles(k)))[:TOPK]
    # exact MCS deficit for the shortlist, in parallel across cores
    # threads, not processes: RDKit + fork is unsafe on macOS
    # (BrokenProcessPool), and the MCS call releases the GIL.
    with ThreadPoolExecutor(max_workers=8) as ex:
        ds = list(ex.map(_D_worker, [k for _, k in scored]))
    for (_, k), d in zip(scored, ds):
        g2 = g + 1
        if k in best_seen and best_seen[k] <= g2:
            continue
        best_seen[k] = g2
        # WEIGHTED A*: f = g + W*h. With W = 1 the search behaved like BFS --
        # D fell 45 -> 43 over 15 expansions while |open| grew to 308, because
        # hundreds of nodes shared the same f. We need A path, not the optimal
        # one, so weighting the heuristic dives toward the target instead of
        # exhausting each depth layer.
        heapq.heappush(openq, (g2 + W_ASTAR * d, g2, k, path + [k]))
        if d < best[0]:
            best = (d, k, path + [k])
    print(f"  expand {expanded:>3}: g={g} f={f} bestD={best[0]} "
          f"|open|={len(openq)}  ({time.time()-t0:.0f}s)", flush=True)
print(f"\n  expanded {expanded}, best remaining deficit D={best[0]}, "
      f"path length {len(best[2])-1} edits  ({time.time()-t0:.0f}s)")
json.dump({"cell": CELL, "reached": best[0] == 0, "final_D": best[0],
           "path": best[2], "n_edits": len(best[2]) - 1},
          open(f'diagnostics/forward_plan_{CELL}.json', 'w'), indent=2)
