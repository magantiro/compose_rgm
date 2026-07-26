#!/usr/bin/env python3
"""Sparse shared-Murcko-scaffold analogue graph -- candidate longer-range pairs for the A2.3 stage.

A shared Bemis-Murcko scaffold is a LOOSE analogue signal; a naive within-group all-pairs join is
misleading because big groups dominate through C(n,2). This builds the SPARSE analogue graph instead:
inside each scaffold group, connect each molecule only to its k nearest neighbours by MCS-based
distance (atoms outside the maximum common substructure, ring-matches-ring / complete-rings-only), so
edges scale ~k*n, not n^2. A cheap Morgan-Tanimoto pre-rank narrows the MCS-refine candidates for
speed; the final ranking is MCS distance. Three caps keep the set balanced and non-redundant: per
SOURCE molecule (=k), per SCAFFOLD group (--max-per-scaffold), and per TRANSFORMATION signature
(--max-per-transformation, the unordered pair of MCS-variable fragments, global).

These are CANDIDATES ONLY -- they are NOT run through the one-cut compiler (their variable regions can
open rings / cross multiple attachments); they are recorded for the later A2.3 general scaffold-pair
compiler + the longer-range curriculum. Output: analogue_murcko_graph_candidates.jsonl (one edge per
line) + analogue_murcko_graph_summary.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFMCS, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from compose_v4.chem.molecular_graph import MolecularGraphError, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph

RDLogger.DisableLog("rdApp.*")

_ROOT = Path(__file__).resolve().parents[1]
CORPUS = _ROOT / "results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles"
OUT_DIR = _ROOT / "diagnostics" / "composition"
_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fingerprint(mol):
    return _FP.GetFingerprint(mol)


def _mcs_edge(mol_a, mol_b, *, timeout: int) -> dict | None:
    """MCS-based distance + transformation signature for an ordered (source, target) pair. Distance =
    heavy atoms outside the MCS on both sides (an operator-distance proxy). Returns None if no MCS."""
    result = rdFMCS.FindMCS([mol_a, mol_b], timeout=timeout, ringMatchesRingOnly=True,
                            completeRingsOnly=True, matchValences=False)
    if result.numAtoms == 0 or not result.smartsString:
        return None
    pattern = Chem.MolFromSmarts(result.smartsString)
    if pattern is None:
        return None
    match_a = set(mol_a.GetSubstructMatch(pattern))
    match_b = set(mol_b.GetSubstructMatch(pattern))
    var_a = [i for i in range(mol_a.GetNumAtoms()) if i not in match_a]
    var_b = [i for i in range(mol_b.GetNumAtoms()) if i not in match_b]
    var_a_smi = Chem.MolFragmentToSmiles(mol_a, atomsToUse=var_a, canonical=True) if var_a else ""
    var_b_smi = Chem.MolFragmentToSmiles(mol_b, atomsToUse=var_b, canonical=True) if var_b else ""
    ring_atoms = (sum(mol_a.GetAtomWithIdx(i).IsInRing() for i in var_a)
                  + sum(mol_b.GetAtomWithIdx(i).IsInRing() for i in var_b))
    return {
        "mcs_num_atoms": int(result.numAtoms),
        "mcs_distance": int(len(var_a) + len(var_b)),
        "mcs_timed_out": bool(result.canceled),
        "source_variable_heavy": int(len(var_a)),
        "target_variable_heavy": int(len(var_b)),
        "variable_ring_atoms": int(ring_atoms),
        "transformation_signature": " >> ".join(sorted((var_a_smi, var_b_smi))),
    }


def _nearest_neighbours(source, members, fps, *, k, mcs_candidates, timeout):
    """k nearest neighbours of `source` within `members` by MCS distance, Morgan-Tanimoto pre-ranked
    to the top `mcs_candidates` for the (expensive) MCS refine. Deterministic tie-breaking on SMILES."""
    others = [m for m in members if m != source]
    ranked = sorted(others, key=lambda o: (-DataStructs.TanimotoSimilarity(fps[source], fps[o]), o))
    scored = []
    for other in ranked[:mcs_candidates]:
        edge = _mcs_edge(Chem.MolFromSmiles(source), Chem.MolFromSmiles(other), timeout=timeout)
        if edge is not None:
            scored.append((edge["mcs_distance"], other, edge))
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--limit", type=int, default=5000, help="max corpus molecules to scan")
    parser.add_argument("--n-slots", type=int, default=48)
    parser.add_argument("--k", type=int, default=3, help="nearest neighbours per source molecule")
    parser.add_argument("--max-per-scaffold", type=int, default=24)
    parser.add_argument("--max-per-transformation", type=int, default=6)
    parser.add_argument("--mcs-candidates", type=int, default=8,
                        help="Tanimoto-pre-ranked candidates per source that get the MCS refine")
    parser.add_argument("--max-group-size", type=int, default=60,
                        help="downsample larger scaffold groups (deterministic) to bound MCS cost")
    parser.add_argument("--mcs-timeout", type=int, default=2, help="per-pair rdFMCS timeout (s)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    raw = []
    with open(args.corpus) as handle:
        for line in handle:
            token = line.strip().split()[0] if line.strip() else ""
            if token:
                raw.append(token)
    raw = raw[:args.limit]

    supported, skipped = [], 0
    for smi in raw:
        try:
            pad_molecular_graph(smiles_to_molecular_graph(smi), args.n_slots)
        except (MolecularGraphError, ValueError):
            skipped += 1
            continue
        supported.append(smi)

    groups: dict[str, list[str]] = defaultdict(list)
    for smi in supported:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:  # noqa: BLE001
            continue
        if scaffold:
            groups[scaffold].append(smi)
    multi = {s: sorted(set(m)) for s, m in groups.items() if len(set(m)) >= 2}
    all_pairs_clique = sum(comb(len(m), 2) for m in multi.values())
    print(f"corpus: {len(supported)} supported ({skipped} skipped) | {len(groups)} scaffolds | "
          f"{len(multi)} multi-member groups | {all_pairs_clique} all-pairs (avoided)", flush=True)

    fps: dict[str, object] = {}
    for members in multi.values():
        for smi in members:
            if smi not in fps:
                fps[smi] = _fingerprint(Chem.MolFromSmiles(smi))

    directed = []
    transformation_counts: Counter = Counter()
    for scaffold in sorted(multi):
        members = multi[scaffold]
        if len(members) > args.max_group_size:
            members = members[:args.max_group_size]
        scaffold_count = 0
        for source in members:
            if scaffold_count >= args.max_per_scaffold:
                break
            for distance, target, edge in _nearest_neighbours(
                    source, members, fps, k=args.k, mcs_candidates=args.mcs_candidates,
                    timeout=args.mcs_timeout):
                if scaffold_count >= args.max_per_scaffold:
                    break
                signature = edge["transformation_signature"]
                if transformation_counts[signature] >= args.max_per_transformation:
                    continue
                transformation_counts[signature] += 1
                scaffold_count += 1
                directed.append({"source_smiles": source, "target_smiles": target,
                                 "murcko_scaffold": scaffold, "stage": "A2.3_candidate", **edge})

    seen, candidates = set(), []
    for edge in directed:
        key = frozenset((edge["source_smiles"], edge["target_smiles"]))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(edge)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.out_dir / "analogue_murcko_graph_candidates.jsonl"
    summary_path = args.out_dir / "analogue_murcko_graph_summary.json"
    with open(candidates_path, "w") as handle:
        for edge in candidates:
            handle.write(json.dumps(edge) + "\n")

    distances = [edge["mcs_distance"] for edge in candidates]
    ring_bearing = sum(1 for edge in candidates if edge["variable_ring_atoms"] > 0)
    group_sizes = Counter(len(m) for m in multi.values())
    summary = {
        "experiment": "murcko_analogue_graph_sparse",
        "corpus": str(args.corpus),
        "corpus_supported": len(supported),
        "scaffold_groups_total": len(groups),
        "scaffold_groups_multi_member": len(multi),
        "all_pairs_clique_would_be": all_pairs_clique,
        "directed_knn_edges": len(directed),
        "unique_candidate_pairs": len(candidates),
        "sparsity_ratio": len(candidates) / max(all_pairs_clique, 1),
        "ring_bearing_variable_fraction": ring_bearing / max(len(candidates), 1),
        "mcs_distance_min": min(distances) if distances else None,
        "mcs_distance_median": float(np.median(distances)) if distances else None,
        "mcs_distance_mean": float(np.mean(distances)) if distances else None,
        "mcs_distance_max": max(distances) if distances else None,
        "mcs_distance_hist": {str(k): v for k, v in sorted(Counter(distances).items())},
        "group_size_hist": {str(k): v for k, v in sorted(group_sizes.items())},
        "distinct_transformations": len(transformation_counts),
        "caps": {"k": args.k, "max_per_scaffold": args.max_per_scaffold,
                 "max_per_transformation": args.max_per_transformation,
                 "mcs_candidates": args.mcs_candidates, "max_group_size": args.max_group_size},
        "outputs": {"candidates": str(candidates_path), "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== sparse Murcko analogue graph (k={args.k} MCS-nearest neighbours) ===", flush=True)
    print(f"directed kNN edges {len(directed)} | unique candidate pairs {len(candidates)} | "
          f"sparsity {summary['sparsity_ratio']:.3f} of the {all_pairs_clique}-pair clique", flush=True)
    if distances:
        print(f"MCS distance (variable atoms) min={summary['mcs_distance_min']} median="
              f"{summary['mcs_distance_median']:.0f} mean={summary['mcs_distance_mean']:.1f} "
              f"max={summary['mcs_distance_max']} | hist {summary['mcs_distance_hist']}", flush=True)
    print(f"ring-bearing variable region: {100 * summary['ring_bearing_variable_fraction']:.1f}% "
          f"| distinct transformations {len(transformation_counts)}", flush=True)
    print(f"wrote {candidates_path}\n      {summary_path}", flush=True)


if __name__ == "__main__":
    main()
