"""Celecoxib target-distance diagnostic for the PMO 3x250 scored run (zero oracle calls).

Celecoxib rediscovery is the one task in the run whose answer is known, so "is the search
becoming target-like?" is decidable rather than a matter of opinion.  This driver measures
three things per charged call, offline:

  1. the oracle's OWN metric -- ECFP4 COUNT-fingerprint Tanimoto to celecoxib, verified
     elsewhere to reproduce all 250 charged scores exactly;
  2. the binary Morgan r=2 / 2048-bit Tanimoto, which is the controller's internal
     ``fiber_fingerprint`` convention and a stricter read of structural resemblance -- a
     candidate can gain on (1) by REPEATING common fragments while losing on (2);
  3. presence of the four structural pieces celecoxib is made of, as SMARTS.

Divergence between (1) and (2) is the whole point: it separates "closing on the target"
from "accumulating fragment multiplicity that the count metric rewards".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

CELECOXIB = "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F"

# The four pharmacophoric pieces celecoxib is built from, plus graded relaxations so a
# near-miss is distinguishable from an absence.  ``core_diarylpyrazole`` is the Bemis-Murcko
# scaffold expressed as SMARTS: matching it means the required scaffold jump was made.
PIECES = {
    "pyrazole_ring": "c1cc[nX2]n1",
    "n_aryl_pyrazole": "[c]1[c][c][nX2][n]1-[c;R]",
    "core_diarylpyrazole": "[c;R]-[c]1[c][c][nX2][n]1-[c;R]",
    "any_sulfonamide": "[SX4](=O)(=O)[NX3,NX2-]",
    "aryl_sulfonamide": "[SX4](=O)(=O)([c])[NX3,NX2-]",
    "primary_sulfonamide": "[NX3;H2][SX4](=O)(=O)",
    "benzenesulfonamide": "[NX3;H2][SX4](=O)(=O)c1ccccc1",
    "trifluoromethyl": "[CX4](F)(F)F",
    "tolyl": "[CH3][c]1[c][c][c]([*])[c][c]1",
    "biaryl_bond": "[c;R]-[c;R]",
    "three_aromatic_rings": None,  # handled numerically
}
# Scored out of four: pyrazole, sulfonamide, CF3, aryl-aryl link.
COMPOSITE_PIECES = ("pyrazole_ring", "any_sulfonamide", "trifluoromethyl", "biaryl_bond")
EXACT_PIECES = ("n_aryl_pyrazole", "benzenesulfonamide", "trifluoromethyl", "tolyl")


def analyse(smiles: str, target_count, target_bits, target_scaffold) -> dict:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {"parseable": False}
    row = {
        "parseable": True,
        "oracle_metric_ecfp4_count_tanimoto": float(DataStructs.TanimotoSimilarity(
            target_count, AllChem.GetMorganFingerprint(molecule, 2))),
        "morgan2_2048bit_tanimoto": float(DataStructs.TanimotoSimilarity(
            target_bits, AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048))),
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "rings": int(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False),
    }
    row["scaffold_equals_target"] = row["scaffold"] == target_scaffold
    for name, smarts in PIECES.items():
        if smarts is None:
            continue
        pattern = Chem.MolFromSmarts(smarts)
        row[f"has_{name}"] = bool(pattern is not None and molecule.HasSubstructMatch(pattern))
    row["has_three_aromatic_rings"] = row["aromatic_rings"] >= 3
    row["pieces_present"] = sum(bool(row.get(f"has_{name}")) for name in COMPOSITE_PIECES)
    row["exact_pieces_present"] = sum(bool(row.get(f"has_{name}")) for name in EXACT_PIECES)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    target = Chem.MolFromSmiles(CELECOXIB)
    target_count = AllChem.GetMorganFingerprint(target, 2)
    target_bits = AllChem.GetMorganFingerprintAsBitVect(target, 2, nBits=2048)
    target_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=target, includeChirality=False)

    rows = []
    for path in sorted(Path(arguments.task_dir, "oracle").glob("query_*/result.json")):
        record = json.loads(path.read_text())
        rows.append({
            "index": record["index"],
            "call": record["index"] + 1,
            "role": record["role"],
            "endpoint": record["endpoint"],
            "charged_score": float(record["score"]),
            **analyse(record["endpoint"], target_count, target_bits, target_scaffold),
        })
    rows.sort(key=lambda row: row["index"])

    parseable = [r for r in rows if r.get("parseable")]
    # Does the CHARGED score agree with the offline oracle metric on every row?
    agreement = all(
        abs(r["charged_score"] - r["oracle_metric_ecfp4_count_tanimoto"]) < 1e-9
        for r in parseable
    )

    # Trend: is the population moving toward the target over the run?
    blocks = []
    for start in range(0, len(rows), 50):
        block = [r for r in rows[start:start + 50] if r.get("parseable")]
        if not block:
            continue
        blocks.append({
            "calls": f"{start + 1}-{min(start + 50, len(rows))}",
            "mean_oracle_metric": float(np.mean(
                [r["oracle_metric_ecfp4_count_tanimoto"] for r in block])),
            "max_oracle_metric": float(np.max(
                [r["oracle_metric_ecfp4_count_tanimoto"] for r in block])),
            "mean_2048bit_tanimoto": float(np.mean(
                [r["morgan2_2048bit_tanimoto"] for r in block])),
            "max_2048bit_tanimoto": float(np.max(
                [r["morgan2_2048bit_tanimoto"] for r in block])),
            "mean_heavy_atoms": float(np.mean([r["heavy_atoms"] for r in block])),
            "mean_pieces_present": float(np.mean([r["pieces_present"] for r in block])),
            "rows_with_any_piece": sum(1 for r in block if r["pieces_present"] > 0),
            "mean_exact_pieces": float(np.mean([r["exact_pieces_present"] for r in block])),
            "rows_with_pyrazole": sum(1 for r in block if r.get("has_pyrazole_ring")),
            "rows_with_sulfonamide": sum(1 for r in block if r.get("has_any_sulfonamide")),
        })

    piece_counts = {
        name: sum(1 for r in parseable if r.get(f"has_{name}"))
        for name in list(PIECES) + ["three_aromatic_rings"]
        if any(f"has_{name}" in r for r in parseable)
    }

    top = sorted(parseable, key=lambda r: -r["charged_score"])[:20]

    payload = {
        "schema_version": "pmo_3x250_celecoxib_distance_v1",
        "oracle_calls_made_by_this_analysis": 0,
        "target": {
            "smiles": CELECOXIB,
            "canonical": Chem.MolToSmiles(target),
            "scaffold": target_scaffold,
            "heavy_atoms": int(target.GetNumHeavyAtoms()),
            "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(target)),
        },
        "offline_metric_reproduces_charged_score": agreement,
        "rows_scored": len(rows),
        "structural_piece_counts_over_all_charged_molecules": piece_counts,
        "molecules_with_target_scaffold": sum(
            1 for r in parseable if r["scaffold_equals_target"]),
        "block_trend_per_50_calls": blocks,
        "top20_by_charged_score": top,
        "best_initialization": max(
            (r for r in parseable if r["role"] == "initialization"),
            key=lambda r: r["charged_score"]),
        "best_candidate": max(
            (r for r in parseable if r["role"] == "candidate"),
            key=lambda r: r["charged_score"]),
        "all_rows": rows,
    }
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.output).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
