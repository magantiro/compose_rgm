"""Derive the FORWARD build order from a BACKWARD elimination order.

Greedy forward construction saturated its anchors: target atom 28 had to bond to
two atoms that were both degree-3 with zero implicit H by the time we reached it,
so no insert action could attach it. The dependency order has to be planned, not
discovered.

Deletion is the easy direction (the deletion fiber costs 0.034s and never needs
the model), so:

    eliminate target-only atoms from the winner, leaves first  ->  a_n ... a_1
    reverse it                                                 ->  build a_1 ... a_n

A leaf-first elimination reversed is an attach-to-existing-structure build, which
is exactly the guarantee we need: an atom is always placed while its anchor still
has free valence.
"""
import json, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
RDLogger.DisableLog('rdApp.*')

CELL = sys.argv[1] if len(sys.argv) > 1 else "parp1_s0"
cc = json.load(open('diagnostics/route_cores.json'))[CELL]
m_t = Chem.MolFromSmiles(cc["target"])
core = Chem.MolFromSmarts(cc["core_smarts"])
keep = set(m_t.GetSubstructMatch(core))      # shared core: never eliminated
print(f"  target {m_t.GetNumHeavyAtoms()} atoms; shared core {len(keep)} atoms; "
      f"{m_t.GetNumHeavyAtoms()-len(keep)} to eliminate")

alive = set(a.GetIdx() for a in m_t.GetAtoms())
order = []
while True:
    # a target-only atom is removable when it has <=1 neighbour still alive
    # (a leaf of the remaining graph)
    cands = []
    for i in sorted(alive - keep):
        deg = sum(1 for nb in m_t.GetAtomWithIdx(i).GetNeighbors()
                  if nb.GetIdx() in alive)
        cands.append((deg, i))
    cands.sort()
    if not cands:
        break
    deg, i = cands[0]
    order.append((i, m_t.GetAtomWithIdx(i).GetSymbol(), deg))
    alive.discard(i)
build = list(reversed(order))
print(f"  eliminated {len(order)} atoms; leftover non-core alive: "
      f"{sorted(alive - keep)}")
print(f"\n  BUILD ORDER (reverse elimination):")
for n, (i, sym, deg) in enumerate(build):
    nbrs = [nb.GetIdx() for nb in m_t.GetAtomWithIdx(i).GetNeighbors()]
    print(f"    {n+1:>2}. target atom {i:>2} ({sym})  neighbours {nbrs}  "
          f"(was a degree-{deg} leaf at elimination)")
json.dump({"cell": CELL, "build_order": [[i, s] for i, s, _ in build],
           "core": sorted(keep)},
          open(f'diagnostics/build_order_{CELL}.json','w'), indent=2)
print(f"\n  wrote diagnostics/build_order_{CELL}.json")
