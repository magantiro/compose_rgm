"""Target-aware COMPILER: seed -> IVG winner in legal COMPOSE operations.

Not a search over molecules. The destination graph is known, so we diff the
graphs, label every required change, and emit the operations in dependency
order. Global A* stalled because its scalar heuristic is FLAT across the
scaffold-opening phase (D fell 45->44 over the route's first three edits) --
a checklist does not care: removing seed-only material is a required step
whether or not it moves a heuristic.

    phase A  remove seed-only material      (3 atoms for parp1)
    phase B  add target-only atoms          (17 atoms), dependency-ordered
    phase C  form remaining bonds / closures

Progress is measured by the exact MCS-to-target (atoms + bonds realized), never
by Tanimoto. Where a required change is not one legal operation, a LOCAL
microprogram search of depth 2-4 is run over that neighbourhood only.
"""
import json, sys, time
sys.path.insert(0, 'src')
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
RDLogger.DisableLog('rdApp.*')
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.fiber import enumerate_action_fiber
from compose_v4.rewrite.kernel import de_novo_rewrite_system

CELL = sys.argv[1] if len(sys.argv) > 1 else "parp1_s0"
MAXSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 60
MICRO_DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 3

cc = json.load(open('diagnostics/route_cores.json'))[CELL]
m_t = Chem.MolFromSmiles(cc["target"]); T_can = Chem.MolToSmiles(m_t)
T_h, T_b = m_t.GetNumHeavyAtoms(), m_t.GetNumBonds()
sysm = de_novo_rewrite_system()

def realized(smi):
    """(atoms, bonds) of the current molecule that match the target exactly."""
    m = Chem.MolFromSmiles(smi)
    if m is None: return (-1, -1, 0)
    r = rdFMCS.FindMCS([m, m_t], timeout=2, ringMatchesRingOnly=False,
                       completeRingsOnly=False,
                       atomCompare=rdFMCS.AtomCompare.CompareElements,
                       bondCompare=rdFMCS.BondCompare.CompareOrder)
    return (r.numAtoms, r.numBonds, m.GetNumHeavyAtoms())

def succ(smi):
    st = pad_molecular_graph(smiles_to_molecular_graph(smi), 48)
    return sorted({mt.successor_key for mt in enumerate_action_fiber(st, system=sysm)
                   if mt.successor_key and mt.successor_key != smi})

def score(smi):
    a, b, h = realized(smi)
    return a + b            # target material realized; higher is better

# RESUME from a saved partial route rather than recompiling from the seed.
# The previous run reached 32 atoms + 35 bonds of 33 + 38 and restarting threw
# away 16 steps of work.
import os
_prev = f'diagnostics/compiled_{CELL}.json'
if os.path.exists(_prev) and "--fresh" not in sys.argv:
    _p = json.load(open(_prev))
    route_prefix = _p.get("route") or []
    cur = route_prefix[-1] if route_prefix else Chem.MolToSmiles(Chem.MolFromSmiles(cc["seed"]))
    print(f"  RESUMING from saved route: {len(route_prefix)-1} ops already done")
else:
    route_prefix = []
    cur = Chem.MolToSmiles(Chem.MolFromSmiles(cc["seed"]))
a0, b0, h0 = realized(cur)
print(f"  {CELL}: seed realizes {a0} atoms + {b0} bonds of target "
      f"({T_h} atoms, {T_b} bonds); heavy {h0} -> {T_h}")
route = (route_prefix[:-1] if route_prefix else []) + [cur]
t0 = time.time(); stalls = 0
for step in range(MAXSTEP):
    if cur == T_can:
        print(f"\n  *** COMPILED: reached target in {step} legal ops ***"); break
    a, b, h = realized(cur)
    kids = succ(cur)
    # PHASE A while we still carry seed-only material: prefer deletions that do
    # NOT destroy already-realized target material.
    phase_a = h > T_h or (h - a) > 0 and step < 6
    # CHEAP PREFILTER FIRST. realized() runs an MCS per candidate, and there
    # are ~1700 of them at this molecule size -- that, not the fiber
    # enumeration, is what made a step cost 150s. Atom and bond counts are free
    # and already tell us whether a candidate moves toward the target's
    # (33 atoms, 38 bonds); only those get an MCS.
    pre = []
    for k in kids:
        m = Chem.MolFromSmiles(k)
        if m is None: continue
        kh, kb_ = m.GetNumHeavyAtoms(), m.GetNumBonds()
        if kh > T_h or kb_ > T_b: continue          # overshoots the target
        pre.append((-(kh + kb_), k))                # closest to target first
    pre.sort()
    scored = []
    for _, k in pre[:40]:
        ka, kb, kh = realized(k)
        if ka < 0: continue
        gain = (ka + kb) - (a + b)
        excess = (kh - ka)
        scored.append((gain, -excess, k, ka, kb, kh))
    if not scored:
        print(f"  step {step}: no legal successor"); break
    scored.sort(reverse=True)
    g, negx, k, ka, kb, kh = scored[0]
    if g <= 0:
        stalls += 1
        # local microprogram: allow one non-improving step, then require gain
        best2 = None
        for _, _, k1, *_ in scored[:3]:   # microprogram: 3 branches, not 12
            a1, b1, _ = realized(k1)
            for k2 in succ(k1):
                a2, b2, _ = realized(k2)
                if (a2 + b2) > (a + b):
                    best2 = (k1, k2, a2, b2); break
            if best2: break
        if best2:
            k1, k2, a2, b2 = best2
            route += [k1, k2]; cur = k2; stalls = 0
            print(f"  step {step:>2}: realized {a}+{b} -> {a2}+{b2} "
                  f"via 2-op microprogram   ({time.time()-t0:.0f}s)", flush=True)
            continue
        if stalls >= 3:
            print(f"  OBSTRUCTION at realized {a}+{b} of {T_h}+{T_b}"); break
    cur = k; route.append(cur)
    print(f"  step {step:>2}: realized {a}+{b} -> {ka}+{kb} (heavy {kh}) "
          f"  ({time.time()-t0:.0f}s)", flush=True)
fa, fb, fh = realized(cur)
print(f"\n  final: {fa} atoms + {fb} bonds of {T_h}+{T_b}; "
      f"{len(route)-1} ops; reached={cur == T_can}  ({time.time()-t0:.0f}s)")
json.dump({"cell": CELL, "reached": cur == T_can, "route": route,
           "n_ops": len(route)-1, "realized": [fa, fb], "target": [T_h, T_b]},
          open(f'diagnostics/compiled_{CELL}.json','w'), indent=2)
