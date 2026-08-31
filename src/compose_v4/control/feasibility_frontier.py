"""Cheap constraint satisfaction, separated from expensive purpose optimisation.

T4 has exactly this structure: THREE CHEAP constraints (QED, SA, similarity,
plus validity) and ONE EXPENSIVE objective (docking). So solve them with
different machinery instead of asking one learned controller to do both.

    seed feasible?  -> optimise docking immediately
    seed infeasible -> build a cheap feasible frontier first, THEN optimise

9 of the 15 T4 seeds start below the QED gate, so the second branch is the
common case, not a corner case. braf_s10 starts at QED 0.346 against 0.6 and
gave 0 feasible / 0 docked over ten full controller rounds -- no docking
feedback can exist before the feasible set is entered, so the controller had
nothing to learn from.

This phase spends ZERO oracle calls and uses the SAME generic executable
transitions; it does not encode a repair recipe. Measured on braf_s10 it finds
113 distinct feasible endpoints at depth 5-6 through programs it discovers
itself (bond_insert, atom_delete, atom_restate).

BREADTH, NOT GREED. A frontier kept by lowest current shortfall deletes
legitimate routes: of those 113 successful repairs, 17% dip BELOW the
similarity threshold and recover, 10% have non-monotone total shortfall, and
the deepest valley's first edit is strictly WORSE than the seed
(0.254 -> 0.275, sim 0.576) yet reaches a feasible endpoint. Intermediates are
therefore never required to satisfy the endpoint constraints -- only the
returned lead is.
"""
from __future__ import annotations

import os
import sys

CHEAP_ONLY = True          # this module must never import a docking oracle


def _props(smiles, seed_fp, delta, gm, sascorer):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    q = float(QED.qed(m))
    sa = float(sascorer.calculateScore(m))
    sim = float(DataStructs.TanimotoSimilarity(seed_fp, gm.GetFingerprint(m)))
    short = max(0.0, 0.6 - q) + max(0.0, sa - 4.0) + max(0.0, float(delta) - sim)
    return dict(qed=q, sa=sa, sim=sim, short=short,
                heavy=m.GetNumHeavyAtoms())


def is_feasible(smiles, seed_smiles, delta, qed_min=0.6, sa_max=4.0):
    """The PUBLISHED benchmark criterion only -- no med-chem screen."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    m = Chem.MolFromSmiles(smiles); s = Chem.MolFromSmiles(seed_smiles)
    if m is None or s is None:
        return False
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    sim = float(DataStructs.TanimotoSimilarity(gm.GetFingerprint(s),
                                               gm.GetFingerprint(m)))
    return (float(QED.qed(m)) >= qed_min
            and float(sascorer.calculateScore(m)) <= sa_max
            and sim >= float(delta))


def build_frontier(seed_smiles, delta, target=100, max_depth=7, beam=140,
                   diversity_frac=0.35, slots=48, seed_rng=0, log=None):
    """Distinct feasible molecules reachable from `seed_smiles`. No oracle.

    `diversity_frac` of each beam is kept WITHOUT reference to shortfall, so
    valley routes survive. Returns (feasible: dict smiles->props, stats).
    """
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import RDConfig, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.chem.molecular_graph import (smiles_to_molecular_graph,
        molecular_graph_to_smiles, AtomVocabulary, ATOM_VALENCE_CLASSES)
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.rewrite.factorized_fiber import _factorized_candidates
    from compose_v4.rewrite.tracelets import _micro_runtime
    from compose_v4.gates.med_chem_gate import is_executable

    voc = AtomVocabulary(ATOM_VALENCE_CLASSES)
    system = _micro_runtime()
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rng = np.random.default_rng(seed_rng)
    sfp = gm.GetFingerprint(Chem.MolFromSmiles(seed_smiles))
    st0 = pad_molecular_graph(smiles_to_molecular_graph(seed_smiles), slots)
    p0 = _props(seed_smiles, sfp, delta, gm, sascorer)
    seed_key = Chem.MolToSmiles(Chem.MolFromSmiles(seed_smiles))

    frontier = [(p0["short"], seed_smiles, st0, 0)]
    seen = {seed_key}
    feasible, expanded = {}, 0
    for depth in range(int(max_depth)):
        nxt = []
        for _sh, smi, st, _d in frontier:
            try:
                cands = list(_factorized_candidates(st, allow_bond_reroute=True,
                                                    vocabulary=voc))
            except Exception:
                continue
            for rule, act in cands:
                try:
                    y = system.apply(st, rule, act)
                    ys = molecular_graph_to_smiles(y)
                except Exception:
                    continue
                if not ys:
                    continue
                mm = Chem.MolFromSmiles(ys)
                if mm is None:
                    continue
                k = Chem.MolToSmiles(mm)
                if k in seen or not is_executable(ys):
                    continue
                seen.add(k); expanded += 1
                pr = _props(ys, sfp, delta, gm, sascorer)
                if pr is None:
                    continue
                if pr["short"] <= 0.0 and k != seed_key:
                    feasible[k] = dict(pr, depth=depth + 1)
                nxt.append((pr["short"], ys, y, depth + 1))
        if not nxt:
            break
        nxt.sort(key=lambda t: t[0])
        n_keep = int(beam)
        n_div = int(n_keep * float(diversity_frac))
        kept = nxt[: n_keep - n_div]
        rest = nxt[n_keep - n_div:]
        if rest and n_div > 0:
            idx = rng.choice(len(rest), size=min(n_div, len(rest)), replace=False)
            kept += [rest[int(i)] for i in idx]
        frontier = kept
        if log:
            log(f"  depth {depth+1}: expanded {expanded} kept {len(frontier)} "
                f"feasible {len(feasible)} best_short {nxt[0][0]:.3f}")
        if len(feasible) >= int(target):
            break
    return feasible, dict(expanded=expanded, depth_reached=depth + 1,
                          seed_shortfall=p0["short"])
