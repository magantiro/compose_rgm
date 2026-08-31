"""Direct graph-diff construction. No enumeration of successors.

The previous compiler asked COMPOSE to GENERATE every legal successor (~thousands,
each requiring an executor apply) and then checked which looked closest to the
target. That is backwards and cost ~250s per step.

Here we do what a person would do by hand:
  1. map the current molecule onto the target,
  2. find a target atom we are missing whose neighbour we already have,
  3. look through the AVAILABLE ACTIONS (yielded without being applied) for the
     atom_insert that adds exactly that element at exactly that attachment,
  4. apply that one action.

_factorized_candidates yields (rule_name, action) pairs lazily; AtomInsert
carries slot / atom_type / neighbors, so the right action can be selected by
inspection instead of by trial.
"""
import json, sys, time
sys.path.insert(0, 'src')
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
RDLogger.DisableLog('rdApp.*')
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.production_successor_kernel import canonical_state_key
from compose_v4.rewrite.factorized_fiber import _factorized_candidates
from compose_v4.rewrite.kernel import de_novo_rewrite_system

# COMPOSE encodes elements as internal integer codes, and AtomInsert.neighbors
# holds (slot, bond_order) pairs -- not element symbols and not bare indices.
# Comparing act.atom_type to 'C' could never match, which is why the first
# version found nothing. Verified: slot order == RDKit atom order.
CODE = {2: "C", 3: "N", 4: "O", 5: "F"}
SYM2CODE = {v: k for k, v in CODE.items()}

CELL = sys.argv[1] if len(sys.argv) > 1 else "parp1_s0"
MAXS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
cc = json.load(open('diagnostics/route_cores.json'))[CELL]
m_t = Chem.MolFromSmiles(cc["target"]); T_can = Chem.MolToSmiles(m_t)
T_AROM = sum(1 for a in m_t.GetAtoms() if a.GetIsAromatic())
_M_T = m_t
T_h, T_b = m_t.GetNumHeavyAtoms(), m_t.GetNumBonds()
sysm = de_novo_rewrite_system()

def match_to_target(smi):
    """MCS mapping: which target atoms are already realized."""
    m = Chem.MolFromSmiles(smi)
    r = rdFMCS.FindMCS([m, m_t], timeout=3, ringMatchesRingOnly=False,
                       completeRingsOnly=False,
                       atomCompare=rdFMCS.AtomCompare.CompareElements,
                       bondCompare=rdFMCS.BondCompare.CompareOrderExact)
    q = Chem.MolFromSmarts(r.smartsString)
    cm = m.GetSubstructMatch(q); tm = m_t.GetSubstructMatch(q)
    # AROMATICITY-AWARE. The MCS compares a KEKULE view with CompareOrder, so an
    # aromatic ring and a saturated ring of the same connectivity score as a full
    # match: the build reported 33 atoms + 38 bonds of 33 + 38 while carrying 2
    # aromatic rings against the target's 3 (11 aromatic atoms vs 17). Charging
    # the aromatic deficit makes that gap visible to the compiler and gives
    # bond_reorder something to close.
    # CompareOrderExact distinguishes aromatic from saturated, so the separate
    # aromatic penalty is redundant and would double-count.
    return m, dict(zip(tm, cm)), r.numAtoms, r.numBonds

def missing_bonds(m_cur, t2c):
    """Target bonds whose BOTH endpoints already exist here but which are not
    yet formed. These are the ring closures -- 7 of them for parp1 -- and they
    need bond_insert, not atom_insert. Omitting them is why the build stopped
    at 30+31 with the skeleton complete but unclosed."""
    out = []
    for b in m_t.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i not in t2c or j not in t2c: continue
        ci, cj = t2c[i], t2c[j]
        if m_cur.GetBondBetweenAtoms(ci, cj) is None:
            out.append((ci, cj))
    return out


BUILD_ORDER = [tuple(x) for x in
               json.load(open(f'diagnostics/build_order_{CELL}.json'))["build_order"]]


def next_targets(m_cur, t2c):
    """Missing target atoms, in the PLANNED build order.

    Order comes from reversing a leaf-first elimination of the target (see
    build_order.py). Greedy ordering saturated anchors: atom 28 needs bonds to
    two atoms that were both degree-3 by the time we got to it. In the planned
    order atom 28 is FIRST, while its anchors still have free valence.
    """
    out = []
    for ti, sym in BUILD_ORDER:
        if ti in t2c: continue
        for nb in m_t.GetAtomWithIdx(ti).GetNeighbors():
            if nb.GetIdx() in t2c:
                out.append((ti, sym, t2c[nb.GetIdx()]))
                break
    return out

cur = Chem.MolToSmiles(Chem.MolFromSmiles(cc["seed"]))
route = [cur]; t0 = time.time()
for step in range(MAXS):
    if cur == T_can:
        print(f"\n  *** BUILT TARGET in {step} ops ***"); break
    m_cur, t2c, na, nb_ = match_to_target(cur)
    want = next_targets(m_cur, t2c)
    # do NOT stop when every target atom is placed -- excess seed material still
    # has to come off. The build finished at 36 heavy against a 33-atom target,
    # still carrying the seed's dimethylamino, because the loop broke here
    # before reaching the deletion phase.
    if not want and m_cur.GetNumHeavyAtoms() <= T_h and not missing_bonds(m_cur, t2c):
        print(f"  step {step}: target material complete (realized {na}+{nb_})"); break
    st = pad_molecular_graph(smiles_to_molecular_graph(cur), 48)
    took = None
    # ONE lazy pass over available actions; no applies until we pick one.
    for ti, sym, attach in want:
        for rule, act in _factorized_candidates(st, allow_bond_reroute=True):
            if rule != "atom_insert": continue
            if getattr(act, "atom_type", None) != SYM2CODE.get(sym): continue
            nbrs = tuple(n[0] if isinstance(n, (tuple, list)) else n
                         for n in (getattr(act, "neighbors", ()) or ()))
            if attach not in nbrs: continue
            try:
                y = canonical_state_key(sysm.apply(st, rule, act))
            except Exception:
                continue
            if y and y != cur:
                _, _, ya, yb = match_to_target(y)
                if (ya + yb) <= (na + nb_):
                    # A BRIDGING atom (two realized target neighbours) gains
                    # nothing until its SECOND bond is formed, so the
                    # gain-required rule rejects the only productive move.
                    # Allow it when the target atom has >=2 realized
                    # neighbours, then immediately form the remaining bond.
                    n_real = sum(1 for nb2 in m_t.GetAtomWithIdx(ti).GetNeighbors()
                                 if nb2.GetIdx() in t2c)
                    if n_real < 2:
                        continue
                    st2 = pad_molecular_graph(smiles_to_molecular_graph(y), 48)
                    m2, t2c2, a2, b2 = match_to_target(y)
                    closed = None
                    for (ci, cj) in missing_bonds(m2, t2c2):
                        for r2, ac2 in _factorized_candidates(st2, allow_bond_reroute=True):
                            if r2 != "bond_insert": continue
                            if not (hasattr(ac2, "a") and hasattr(ac2, "b")): continue
                            if {ac2.a, ac2.b} != {ci, cj}: continue
                            try:
                                y2 = canonical_state_key(sysm.apply(st2, r2, ac2))
                            except Exception:
                                continue
                            _, _, a3, b3 = match_to_target(y2)
                            if (a3 + b3) > (na + nb_):
                                closed = y2; break
                        if closed: break
                    if not closed:
                        continue
                    route.append(y)
                    took = (closed, sym + "+bond", attach); break
                took = (y, sym, attach); break
        if took: break
    # PHASE SEPARATION: do not close rings while target atoms are still
    # missing. Closing saturates the valence of the atoms the missing atoms
    # need to attach to -- measured: target atom 28 must bond to current atoms
    # 25 and 26, and by the time we tried, both were degree-3 with zero
    # implicit H, so ZERO atom_insert actions could reach them. The operators
    # can express this; we simply built in the wrong order.
    if not took and want:
        print(f"  step {step}: {len(want)} atom(s) still missing but no insert "
              f"available -- deferring closures", flush=True)
    if not took:
        # fall through to ring closure: form a target bond between two atoms
        # that already exist
        for (ci, cj) in missing_bonds(m_cur, t2c):
            # skip a closure that would saturate an atom a missing target atom
            # still needs to attach to
            _needed = set()
            for ti2, sym2, at2 in want:
                for nb2 in m_t.GetAtomWithIdx(ti2).GetNeighbors():
                    if nb2.GetIdx() in t2c: _needed.add(t2c[nb2.GetIdx()])
            if ci in _needed or cj in _needed:
                continue
            for rule, act in _factorized_candidates(st, allow_bond_reroute=True):
                if rule not in ("bond_insert", "cycle_close", "bond_reorder"): continue
                # BondInsert(a=..., b=..., order=...) -- the endpoints are `a`
                # and `b`. Guessing at field names is what made this branch a
                # no-op on the previous pass.
                if not (hasattr(act, "a") and hasattr(act, "b")): continue
                if {act.a, act.b} != {ci, cj}: continue
                try:
                    y = canonical_state_key(sysm.apply(st, rule, act))
                except Exception:
                    continue
                if y and y != cur:
                    _, _, ya, yb = match_to_target(y)
                    if (ya + yb) <= (na + nb_):
                        continue
                    took = (y, "bond", f"{ci}-{cj}"); break
            if took: break
    if not took and want:
        # MOTIF UNIT. Build a connected target motif as ONE program instead of
        # demanding that each atom individually improve the MCS. Anchors may be
        # valence-saturated (parp1 atom 28 attaches to two degree-3 atoms), so
        # the program may first BREAK a bond to reserve valence. Nothing inside
        # is evaluated; only the completed program has to increase realized
        # material. This is the compiler-side version of "purpose is evaluated
        # at the macro endpoint".
        ti, sym, _anchor = want[0]
        anchors = [t2c[nb.GetIdx()] for nb in m_t.GetAtomWithIdx(ti).GetNeighbors()
                   if nb.GetIdx() in t2c]
        prog = None
        for rule0, act0 in _factorized_candidates(st, allow_bond_reroute=True):
            if rule0 != "bond_delete": continue
            if not (hasattr(act0, "a") and hasattr(act0, "b")): continue
            if act0.a not in anchors and act0.b not in anchors: continue
            try:
                y1 = canonical_state_key(sysm.apply(st, rule0, act0))
            except Exception:
                continue
            if not y1: continue
            st1 = pad_molecular_graph(smiles_to_molecular_graph(y1), 48)
            # now insert the atom that was blocked
            for r1, a1 in _factorized_candidates(st1, allow_bond_reroute=True):
                if r1 != "atom_insert": continue
                if getattr(a1, "atom_type", None) != SYM2CODE.get(sym): continue
                nb1 = tuple(n[0] if isinstance(n, (tuple, list)) else n
                            for n in (getattr(a1, "neighbors", ()) or ()))
                if not (set(nb1) & set(anchors)): continue
                try:
                    y2 = canonical_state_key(sysm.apply(st1, r1, a1))
                except Exception:
                    continue
                if not y2: continue
                # ACCEPT break->insert on its own when it does not lose
                # ground. Requiring the close to gain in the SAME program was
                # too strict: the atom has to exist before its second bond can
                # be formed, so the gain arrives a step later.
                _, _, ya2, yb2 = match_to_target(y2)
                if (ya2 + yb2) >= (na + nb_):
                    prog = (y1, y2, y2); break
                # otherwise try to close inside the program
                st2 = pad_molecular_graph(smiles_to_molecular_graph(y2), 48)
                m2, t2c2, _, _ = match_to_target(y2)
                for (ci, cj) in missing_bonds(m2, t2c2):
                    for r2, a2 in _factorized_candidates(st2, allow_bond_reroute=True):
                        if r2 != "bond_insert": continue
                        if not (hasattr(a2, "a") and hasattr(a2, "b")): continue
                        if {a2.a, a2.b} != {ci, cj}: continue
                        try:
                            y3 = canonical_state_key(sysm.apply(st2, r2, a2))
                        except Exception:
                            continue
                        _, _, a3, b3 = match_to_target(y3)
                        if (a3 + b3) > (na + nb_):
                            prog = (y1, y2, y3); break
                    if prog: break
                if prog: break
            if prog: break
        if prog:
            y1, y2, y3 = prog
            route += [y1, y2]
            took = (y3, "motif", str(anchors))
        else:
            print(f"       motif failed: target atom {ti} ({sym}) anchors "
                  f"{anchors} -- no break->insert->close program raised "
                  f"realized above {na}+{nb_}", flush=True)
    if not took:
        # RESTATE / REORDER: an atom of the wrong element, or a bond of the
        # wrong order, cannot match under an element+order MCS, so it reads as
        # unrealized and blocks everything downstream of it. Accept any such
        # correction that increases realized material.
        # TARGETED, not exhaustive. We know which target atom is missing, its
        # element, and which realized atom it attaches to -- so restate a
        # neighbouring unmatched atom to that element rather than applying all
        # 81 restate/reorder actions and running an MCS on each.
        matched = set(t2c.values())
        wanted = {}
        for ti, sym, attach in want:
            for nb in m_cur.GetAtomWithIdx(attach).GetNeighbors():
                if nb.GetIdx() not in matched:
                    wanted[nb.GetIdx()] = SYM2CODE.get(sym)
        for rule, act in _factorized_candidates(st, allow_bond_reroute=True):
            if rule == "bond_reorder":
                try:
                    y = canonical_state_key(sysm.apply(st, rule, act))
                except Exception:
                    continue
                if not y or y == cur: continue
                _, _, ya, yb = match_to_target(y)
                if (ya + yb) <= (na + nb_): continue
                took = (y, "reorder", "-"); break
            if rule == "bond_reorder":
                try:
                    y = canonical_state_key(sysm.apply(st, rule, act))
                except Exception:
                    continue
                if not y or y == cur: continue
                _, _, ya, yb = match_to_target(y)
                if (ya + yb) <= (na + nb_): continue
                took = (y, "reorder", "-"); break
            if rule != "atom_restate": continue
            v = getattr(act, "v", None)
            if v not in wanted: continue
            if getattr(act, "atom_type", None) != wanted[v]: continue
            try:
                y = canonical_state_key(sysm.apply(st, rule, act))
            except Exception:
                continue
            if not y or y == cur: continue
            _, _, ya, yb = match_to_target(y)
            if (ya + yb) <= (na + nb_): continue
            took = (y, "restate", str(v)); break
    if not took:
        # remove material the target does not contain (seed-only atoms, and any
        # junk we added). Accept only if realized does not fall.
        cur_matched = set(t2c.values())
        for rule, act in _factorized_candidates(st, allow_bond_reroute=True):
            if rule != "atom_delete": continue
            v = getattr(act, "v", None)
            if v is None or v in cur_matched: continue
            try:
                y = canonical_state_key(sysm.apply(st, rule, act))
            except Exception:
                continue
            if not y or y == cur: continue
            _, _, ya, yb = match_to_target(y)
            if (ya + yb) < (na + nb_): continue
            took = (y, "del", str(v)); break
    if not took:
        print(f"  step {step}: nothing matches the diff "
              f"(realized {na}+{nb_}, {len(want)} atoms / "
              f"{len(missing_bonds(m_cur, t2c))} bonds wanted)"); break
    cur, sym, attach = took; route.append(cur)
    print(f"  step {step:>2}: +{sym} at atom {attach}  realized {na}+{nb_} "
          f"  ({time.time()-t0:.1f}s)", flush=True)
m, _, fa, fb = match_to_target(cur)
print(f"\n  final realized {fa}+{fb} of {T_h}+{T_b}; {len(route)-1} ops; "
      f"reached={cur == T_can}  ({time.time()-t0:.1f}s)")
json.dump({"cell": CELL, "route": route, "reached": cur == T_can,
           "realized": [fa, fb], "target": [T_h, T_b]},
          open(f'diagnostics/direct_{CELL}.json','w'), indent=2)
