"""Exact local program synthesis for the ONE unresolved forward transition.

The backward witness gives 18 legal steps; reversed, 14 are directly legal
forward and 3 of the remaining 4 already have known 2-edit decompositions. Only
step 9 is unsolved. Rebuilding all 33 atoms from scratch to recover a route we
already mostly possess is the wrong move.

Here we search ONLY between the exact x_before and x_after of that step, over
legal COMPOSE actions, checking EXACT canonical equality -- not an MCS score.
Guidance is the exact graph difference to x_after, so a state is kept only if it
reduces (or holds) the number of atom/bond discrepancies.
"""
import json, sys, time, heapq
sys.path.insert(0, 'src')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.factorized_fiber import _factorized_candidates
from compose_v4.experiments.production_successor_kernel import canonical_state_key
from compose_v4.rewrite.kernel import de_novo_rewrite_system

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 9
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 5
WIDTH = int(sys.argv[3]) if len(sys.argv) > 3 else 30

r = json.load(open('diagnostics/parp1_witness_route.json'))
fs = r['forward_states']
a, b = fs[STEP - 1], fs[STEP]
m_b = Chem.MolFromSmiles(b); B_can = Chem.MolToSmiles(m_b)
B_h, B_bd = m_b.GetNumHeavyAtoms(), m_b.GetNumBonds()
sysm = de_novo_rewrite_system()
print(f"  step {STEP}:  {a}")
print(f"          ->  {b}")
print(f"  target: {B_h} heavy, {B_bd} bonds\n")

def dist(smi):
    """Exact discrepancy to x_after: atoms + bonds not in common (strict)."""
    m = Chem.MolFromSmiles(smi)
    if m is None: return 10**6
    if Chem.MolToSmiles(m) == B_can: return 0
    from rdkit.Chem import rdFMCS
    try:
        q = rdFMCS.FindMCS([m, m_b], timeout=2, ringMatchesRingOnly=False,
                           completeRingsOnly=False,
                           atomCompare=rdFMCS.AtomCompare.CompareElements,
                           bondCompare=rdFMCS.BondCompare.CompareOrderExact)
        na, nb = q.numAtoms, q.numBonds
    except Exception:
        return 10**6
    return ((m.GetNumHeavyAtoms() - na) + (B_h - na)
            + (m.GetNumBonds() - nb) + (B_bd - nb))

def _touched(act):
    """Atom slots an action touches."""
    s = set()
    for f in ("v", "a", "b", "u", "slot"):
        x = getattr(act, f, None)
        if isinstance(x, int): s.add(x)
    for n in (getattr(act, "neighbors", ()) or ()):
        s.add(n[0] if isinstance(n, (tuple, list)) else n)
    return s


def succ(smi, local=None):
    """Successors from LEGAL ACTIONS ONLY, restricted to a local atom set.

    enumerate_action_fiber applies every action (~15s/state), which made depth-4
    search impossible. _factorized_candidates yields actions lazily, so we filter
    to those touching the affected region and apply only those.
    """
    st = pad_molecular_graph(smiles_to_molecular_graph(smi), 48)
    out = set()
    for rule, act in _factorized_candidates(st, allow_bond_reroute=True):
        if local is not None and not (_touched(act) & local):
            continue
        try:
            y = canonical_state_key(sysm.apply(st, rule, act))
        except Exception:
            continue
        if y and y != smi:
            out.add(y)
    return sorted(out)

# affected region: atoms whose environment differs between the endpoints, plus
# 1-hop neighbours. Everything outside is irrelevant to this transition.
_ma = Chem.MolFromSmiles(a)
LOCAL = set(range(_ma.GetNumHeavyAtoms()))
if _ma.GetNumHeavyAtoms() > 12:
    from rdkit.Chem import rdFMCS as _F
    _q = _F.FindMCS([_ma, m_b], timeout=5, ringMatchesRingOnly=False,
                    completeRingsOnly=False,
                    atomCompare=_F.AtomCompare.CompareElements,
                    bondCompare=_F.BondCompare.CompareOrderExact)
    _mm = Chem.MolFromSmarts(_q.smartsString)
    _shared = set(_ma.GetSubstructMatch(_mm)) if _mm is not None else set()
    _diff = set(range(_ma.GetNumHeavyAtoms())) - _shared
    LOCAL = set(_diff)
    for i in list(_diff):
        for nb in _ma.GetAtomWithIdx(i).GetNeighbors():
            LOCAL.add(nb.GetIdx())
print(f"  affected region: {len(LOCAL)} of {_ma.GetNumHeavyAtoms()} atoms")

t0 = time.time()
d0 = dist(a)
print(f"  starting discrepancy: {d0}")
frontier = [(d0, a, [a])]
seen = {a}
for depth in range(1, DEPTH + 1):
    nxt = []
    for _, s, path in frontier[:WIDTH]:
        for y in succ(s, LOCAL):
            if y in seen: continue
            seen.add(y)
            dy = dist(y)
            if dy == 0:
                print(f"\n  *** SOLVED at depth {depth} in {time.time()-t0:.0f}s ***")
                for i, p in enumerate(path + [y]):
                    print(f"    {i}: {p}")
                json.dump({"step": STEP, "program": path + [y],
                           "n_edits": len(path)},
                          open(f'diagnostics/micro_step{STEP}.json','w'), indent=2)
                sys.exit(0)
            nxt.append((dy, y, path + [y]))
    if not nxt:
        print(f"  depth {depth}: no new states"); break
    nxt.sort(key=lambda x: x[0])
    frontier = nxt
    print(f"  depth {depth}: {len(nxt)} states, best discrepancy {nxt[0][0]}  "
          f"({time.time()-t0:.0f}s)", flush=True)
print(f"\n  NOT SOLVED within depth {DEPTH}; best discrepancy "
      f"{frontier[0][0] if frontier else '-'}  ({time.time()-t0:.0f}s)")
