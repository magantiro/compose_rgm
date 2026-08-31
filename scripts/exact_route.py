"""LOCAL executor-only route search. Seconds per iteration, no Modal.

The search needs no model: reachability is a property of the executor. Running
it locally removes a 3-5 minute image-build and cold-start per attempt, which
had grown to ~300x the cost of the search itself.

Fast path uses the deletion fiber (10 actions, 0.03s). The full action fiber
(1698 transitions, 17s) is invoked ONLY at a plateau, where ring-opening moves
are needed because the remaining excess atoms sit in ring systems that cannot be
shed atom-by-atom without destroying the shared core.
"""
import json, sys, time
sys.path.insert(0, 'src')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.production_successor_kernel import canonical_state_key
from compose_v4.rewrite.process_v2_atom_delete import enumerate_process_v2_atom_deletes
from compose_v4.rewrite.fiber import enumerate_action_fiber
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from rdkit.Chem import rdFMCS

CELL = sys.argv[1] if len(sys.argv) > 1 else "parp1_s0"
MAXS = int(sys.argv[2]) if len(sys.argv) > 2 else 80
PDEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 3

cc = json.load(open('diagnostics/route_cores.json'))[CELL]
core_q = Chem.MolFromSmarts(cc["core_smarts"])
h_seed = cc["h_seed"]
seed_canon = Chem.MolToSmiles(Chem.MolFromSmiles(cc["seed"]))
system = de_novo_rewrite_system()

def st_of(smi):
    return pad_molecular_graph(smiles_to_molecular_graph(smi), 48)

m_seed = Chem.MolFromSmiles(cc["seed"])
nb_seed = m_seed.GetNumBonds()

def _mcs_to_seed(m):
    """Atoms shared with the SEED. Used once heavy-atom excess hits 0, where the
    remaining difference is atom identity/connectivity rather than count and the
    excess objective is exhausted (it reads 0 for every candidate)."""
    try:
        r = rdFMCS.FindMCS([m, m_seed], timeout=5, ringMatchesRingOnly=False,
                           completeRingsOnly=False,
                           atomCompare=rdFMCS.AtomCompare.CompareElements,
                           bondCompare=rdFMCS.BondCompare.CompareOrder)
        return r.numAtoms, r.numBonds
    except Exception:
        return 0, 0

def D(smi, fine=False):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    if core_q is not None and not m.HasSubstructMatch(core_q): return None
    ex = m.GetNumHeavyAtoms() - h_seed
    if not fine:
        return ex
    # FINE objective: operator deficit over ATOMS **AND BONDS**. An atom-only
    # MCS reads 0 while ring-closure bonds are still missing -- measured: the
    # search reached a 19-atom molecule matching the seed element-for-element
    # but with the fused tricycle opened (O=C(NCc1cccc(CN2CC2)c1)c1ccc[nH]1 vs
    # CN(C)Cc1ccc2c(c1)CNC(=O)c1cccn1-2), and declared deficit 0. Ring closure
    # is invisible to an atom-count objective.
    na, nb = _mcs_to_seed(m)
    return ((m.GetNumHeavyAtoms() - na) + (h_seed - na)
            + (m.GetNumBonds() - nb) + (nb_seed - nb))

def dels(smi):
    try: st = st_of(smi)
    except Exception: return []
    out = []
    for a in enumerate_process_v2_atom_deletes(st):
        try: y = canonical_state_key(system.apply(st, "atom_delete", a))
        except Exception: continue
        if y and y != smi: out.append(y)
    return out

def allmoves(smi):
    try: st = st_of(smi)
    except Exception: return []
    try: return [mt.successor_key for mt in enumerate_action_fiber(st, system=system)
                 if mt.successor_key and mt.successor_key != smi]
    except Exception: return []

cur = Chem.MolToSmiles(Chem.MolFromSmiles(cc["target"]))
route = [cur]; t0 = time.time(); obstruction = None
FINE_LATCHED = False   # once the coarse excess hits 0 the remaining difference
                       # is atom identity, so the fine objective must STAY on:
                       # recomputing it each step let one fine move drop the
                       # heavy count to 18 and flip the flag back off.
print(f"  {CELL}: winner {Chem.MolFromSmiles(cur).GetNumHeavyAtoms()} heavy, "
      f"seed {h_seed}, core {cc['core_atoms']}  -> excess {D(cur)}")
for step in range(MAXS):
    if cur == seed_canon:
        break
    if not FINE_LATCHED and D(cur) == 0:
        FINE_LATCHED = True
        print(f"  --- heavy count matched; switching to exact graph deficit ---")
    FINE = FINE_LATCHED
    d_cur = D(cur, FINE)
    # under the fine objective the seed needs atoms ADDED BACK, which the
    # deletion fiber cannot supply, so use the full legal move set.
    cand = [(y, D(y, FINE)) for y in (allmoves(cur) if FINE else dels(cur))]
    cand = [(y, d) for y, d in cand if d is not None and d >= 0]
    best = min(cand, key=lambda p: p[1], default=None)
    if best and d_cur is not None and best[1] < d_cur:
        cur = best[0]; route.append(cur)
        print(f"  del {step:>2}: excess {d_cur} -> {best[1]}   ({time.time()-t0:.1f}s)")
        continue
    # PLATEAU: full legal fiber, then bounded depth
    print(f"  plateau at {'graph-deficit' if FINE else 'excess'} {d_cur}: "
          f"opening the full action fiber...", flush=True)
    seen = {cur}; frontier = [cur]; found = None
    for depth in range(1, PDEPTH + 1):
        nxt = []
        for s in frontier[:6]:
            for y in allmoves(s):
                if y in seen: continue
                seen.add(y); dy = D(y, FINE)
                if dy is None or dy < 0: continue
                nxt.append((dy, y))
                if dy < d_cur: found = y; break
            if found: break
        if found: break
        frontier = [y for _, y in sorted(nxt)[:6]]
        print(f"     depth {depth}: {len(nxt)} core-preserving states, "
              f"best excess {min([d for d,_ in nxt], default='-')}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if not frontier: break
    if found:
        cur = found; route.append(cur)
        print(f"  fib {step:>2}: excess {d_cur} -> {D(cur)} via full fiber "
              f"({time.time()-t0:.0f}s)")
        continue
    obstruction = f"no move of depth<={PDEPTH} reduces excess below {d_cur} keeping the core"
    print(f"  OBSTRUCTION: {obstruction}")
    break
out = {"cell": CELL, "reached_seed": cur == seed_canon, "final_excess": D(cur),
       "n_states": len(route), "obstruction": obstruction,
       "forward_states": list(reversed(route)), "wall_s": time.time() - t0}
json.dump(out, open(f'diagnostics/exact_route_{CELL}.json', 'w'), indent=2)
print(f"\n  reached_seed={out['reached_seed']} final_excess={out['final_excess']} "
      f"states={out['n_states']} in {out['wall_s']:.0f}s")
