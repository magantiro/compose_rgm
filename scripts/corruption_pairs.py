#!/usr/bin/env python3
"""C: valid-corruption source->target pairs (data-gen for the Stage-B edit flow).

The source-conditioned edit flow (memo) trains on (G_source, G_target) pairs where
G_source is a valid CORRUPTION of a data molecule, produced by inverse rewrite
actions, and the model learns the reverse executable program. This is the longest-
lead-time item on the editing critical path and is data-only (no GPU), so it can run
in parallel. This script is the feasibility de-risk: can we generate valid, connected
corrupted sources from real leads at controlled depth, each with a fiber-reachable
path back to the target?

Corruption = apply depth-k legal rewrites that REDUCE molecular complexity
(heteroatoms + rings), walking the target toward the carbon-tree prior it was built
from. Every intermediate is a valid molecule (fiber closure), so the reverse program
(source->target) lives in the same legal fiber. Reports success rate, achieved depth,
complexity drop, and example pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from compose_v4.chem.molecular_graph import (molecular_graph_to_smiles,
                                             smiles_to_molecular_graph)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.fiber import ActionFiberSpec, _candidate_actions
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system

RDLogger.DisableLog("rdApp.*")

_ROOT = Path(__file__).resolve().parents[1]
LEADS = str(_ROOT / "configs" / "benchmarks" / "cnof_leads.json")
OUTDIAG = _ROOT / "diagnostics" / "composition"

# Reductive operators = the inverse of the build-up moves (atom_insert / bond_insert).
# Corruption walks DOWN these so the reverse (source->target) edit program is guaranteed to
# live in the legal fiber (its inverse is an insert, which is abundant). They are ~0.3% of the
# full fiber, so we enumerate + filter to these classes -- random-sampling the fiber misses them.
REDUCTIVE_RULES = ("atom_delete", "bond_delete")


def complexity(mol):
    return sum(1 for a in mol.GetAtoms() if a.GetSymbol() != "C") + 2 * Descriptors.RingCount(mol)


def _mol(node):
    return Chem.MolFromSmiles(molecular_graph_to_smiles(node) or "")


def corrupt(node, depth, spec, rewrite, rng):
    """Apply up to `depth` complexity-reducing legal rewrites, drawn from the REDUCTIVE
    operator classes (the inverse of build-up). Enumerate the reductive candidates and take
    a complexity-reducing, connected successor; reductive moves are ~0.3% of the full fiber,
    so filtering to their classes first (vs random-sampling the whole fiber) is what makes
    this fire. Returns (source, program) where program is the executed corruption path."""
    program = []
    for _ in range(depth):
        m = _mol(node)
        if m is None:
            break
        cur_c = complexity(m)
        base_rings = Descriptors.RingCount(m)          # preserve ring systems: realistic, ring-intact sources
        cur_key = canonical_state_key(node)
        cands = _candidate_actions(node, spec)
        cands = cands if isinstance(cands, list) else list(cands)
        reductive = [(rn, a) for (rn, a) in cands if rn in REDUCTIVE_RULES]
        moved = False
        for idx in rng.permutation(len(reductive)):    # vary which reductive move fires
            rule_name, action = reductive[int(idx)]
            try:
                succ = rewrite.apply(node, rule_name, action)
            except Exception:  # noqa: BLE001
                continue
            succ_smi = molecular_graph_to_smiles(succ)
            sm = Chem.MolFromSmiles(succ_smi or "")
            if sm is None or canonical_state_key(succ) == cur_key:
                continue
            if "." in (succ_smi or ""):                # keep the source a single connected molecule
                continue
            if Descriptors.RingCount(sm) != base_rings:  # peripheral trimming only -> SE sidesteps rings
                continue
            if complexity(sm) < cur_c:                 # a corrupting (simplifying) move
                node, moved = succ, True
                program.append((rule_name, succ_smi))
                break
        if not moved:
            break
    return node, program


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-leads", type=int, default=8)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    rewrite = de_novo_rewrite_system()
    spec = ActionFiberSpec.neutral_cnof()
    leads = json.load(open(LEADS))[: args.n_leads]
    rng = np.random.default_rng(0)

    rows, ok = [], 0
    print(f"generating valid corruption pairs (depth {args.depth}):", flush=True)
    for i, (_, smi, _) in enumerate(leads):
        tgt = Chem.MolFromSmiles(smi)
        if tgt is None:
            continue
        state = pad_molecular_graph(smiles_to_molecular_graph(smi), 40)
        src, program = corrupt(state, args.depth, spec, rewrite, rng)
        src_smi = molecular_graph_to_smiles(src)
        sm = Chem.MolFromSmiles(src_smi or "")
        valid = sm is not None and len(program) > 0
        drop = complexity(tgt) - (complexity(sm) if sm is not None else complexity(tgt))
        ok += int(valid)
        rows.append({"target": smi, "source": src_smi, "depth_reached": len(program),
                     "target_complexity": complexity(tgt),
                     "source_complexity": complexity(sm) if sm is not None else None,
                     "complexity_drop": int(drop), "valid": bool(valid)})
        if i < 6:
            print(f"  {smi[:34]:<36} -> {str(src_smi)[:30]:<32} depth={len(program)} drop={drop}", flush=True)

    summary = {"experiment": "corruption_pairs_feasibility", "n_leads": len(rows),
               "success_rate": ok / max(len(rows), 1),
               "mean_depth_reached": float(np.mean([r["depth_reached"] for r in rows])),
               "mean_complexity_drop": float(np.mean([r["complexity_drop"] for r in rows])),
               "target_depth": args.depth, "pairs": rows}
    print(f"\nvalid corruption pairs: {ok}/{len(rows)}  mean depth {summary['mean_depth_reached']:.1f}"
          f"  mean complexity drop {summary['mean_complexity_drop']:.1f}", flush=True)
    print("=> Stage-B data-gen is feasible" if summary["success_rate"] > 0.8
          else "=> corruption needs work (low success)", flush=True)
    OUTDIAG.mkdir(parents=True, exist_ok=True)
    out = args.output or (OUTDIAG / "corruption_pairs_feasibility.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
