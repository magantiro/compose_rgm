"""Freeze the prescreen table, emit per-task seed banks, and record the structural diagnostic.

SEED RULE (deterministic, documented, no target information): rank the frozen ZINC250k
prescreen table by the task's own official oracle score, canonical-deduplicate, take the
top `POOL`.  The controller's existing `count` then draws from that bank exactly as it
draws from the task-independent bank today, so no controller hyperparameter moves.

The structural diagnostic is NOT reduced to one number.  Troglitazone already showed that
"correct but incomplete" scores worse under a scalar topology distance than a wrong
molecule of about the right size, so each seed carries its scaffold, ring profile, MCS
coverage and substructure containment separately.  These reference-derived fields are
DIAGNOSTIC ONLY and must never reach a proposal law.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
OUT = Path("diagnostics/pmo_prescreen_v1")
POOL = 40


def signature(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    info = mol.GetRingInfo()
    rings = [set(r) for r in info.AtomRings()]
    return {
        "ring_sizes": sorted(len(r) for r in info.AtomRings()),
        "ring_heteroatoms": sum(
            1 for r in info.AtomRings() for i in r
            if mol.GetAtomWithIdx(i).GetSymbol() != "C"
        ),
        "fused_pairs": sum(
            1 for i in range(len(rings)) for j in range(i + 1, len(rings))
            if len(rings[i] & rings[j]) >= 2
        ),
        "aromatic_rings": sum(
            1 for r in info.AtomRings()
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
        ),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
    }


def scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles)
    except (ValueError, RuntimeError):
        return ""


def mcs_atoms(a: str, b: str) -> int:
    x, y = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if x is None or y is None:
        return 0
    try:
        return rdFMCS.FindMCS([x, y], timeout=10, ringMatchesRingOnly=True).numAtoms
    except (ValueError, RuntimeError):
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--pool", type=int, default=POOL)
    args = ap.parse_args()

    canonical = {}
    with (OUT / "zinc250k_canonical_v1.csv").open() as fh:
        for row in csv.DictReader(fh):
            canonical[int(row["source_row_id"])] = row["canonical_smiles"]

    refs = json.loads(Path("/tmp/tdc_refs.json").read_text())
    reference = {
        "albuterol_similarity": refs["albuterol_similarity"],
        "celecoxib_rediscovery": refs["celecoxib_rediscovery"],
        "mestranol_similarity": refs["mestranol_similarity"],
        "thiothixene_rediscovery": refs["thiothixene_rediscovery"],
        "troglitazone_rediscovery": refs["troglitazone_rediscovery"],
        "amlodipine_mpo": refs["amlodipine_smiles"],
        "fexofenadine_mpo": refs["fexofenadine_smiles"],
        "osimertinib_mpo": refs["osimertinib_smiles"],
        "perindopril_mpo": refs["perindopril_smiles"],
        "ranolazine_mpo": refs["ranolazine_smiles"],
        "sitagliptin_mpo": refs["sitagliptin_smiles"],
        "zaleplon_mpo": refs["zaleplon_smiles"],
        "valsartan_smarts": refs["valsartan_smarts"],
        "deco_hop": refs["pharmacophor_smiles"],
        "scaffold_hop": refs["pharmacophor_smiles"],
        "median1": refs["camphor_smiles"],
        "median2": refs["tadalafil_smiles"],
    }

    seeds_dir = OUT / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    diagnostic = {}
    for path in sorted(Path(args.scores_dir).glob("*.csv")):
        task = path.stem
        rows = []
        with path.open() as fh:
            for row in csv.DictReader(fh):
                rows.append((float(row["oracle_score"]), int(row["source_row_id"])))
        if len(rows) != len(canonical):
            print(f"  SKIP {task}: {len(rows):,} of {len(canonical):,} rows")
            continue
        rows.sort(key=lambda r: (-r[0], r[1]))
        seen, top = set(), []
        for score, row_id in rows:
            smiles = canonical[row_id]
            if smiles in seen:
                continue
            seen.add(smiles)
            top.append((score, row_id, smiles))
            if len(top) >= args.pool:
                break

        bank = {
            "n": len(top),
            "rule": (
                "top-N distinct canonical ZINC250k molecules by this task's official TDC "
                "oracle, from the frozen prescreen table; ties broken by source_row_id. "
                "No target structure, fingerprint or topology enters this ranking."
            ),
            "smiles": [s for _, _, s in top],
        }
        out_path = seeds_dir / f"{task}.json"
        out_path.write_text(json.dumps(bank, indent=1) + "\n")

        ref = reference.get(task)
        entries = []
        for score, row_id, smiles in top:
            entry = {
                "source_row_id": row_id, "oracle_score": score, "smiles": smiles,
                "scaffold": scaffold(smiles), "signature": signature(smiles),
            }
            if ref:
                entry["mcs_atoms_with_reference"] = mcs_atoms(smiles, ref)
                rmol, smol = Chem.MolFromSmiles(ref), Chem.MolFromSmiles(smiles)
                entry["scaffold_equals_reference"] = scaffold(smiles) == scaffold(ref)
                entry["is_reference_substructure"] = bool(
                    rmol is not None and smol is not None
                    and rmol.HasSubstructMatch(smol)
                )
            entries.append(entry)
        diagnostic[task] = {
            "reference": ref, "reference_signature": signature(ref) if ref else None,
            "reference_scaffold": scaffold(ref) if ref else None,
            "pool": entries,
            "flags": {
                "exact_scaffold_supplied": sum(
                    1 for e in entries if e.get("scaffold_equals_reference")),
                "reference_substructure_supplied": sum(
                    1 for e in entries if e.get("is_reference_substructure")),
                "distinct_scaffolds_in_pool": len({e["scaffold"] for e in entries}),
                "max_mcs_with_reference": max(
                    (e.get("mcs_atoms_with_reference", 0) for e in entries), default=0),
            },
        }
        f = diagnostic[task]["flags"]
        print(f"  {task:26s} pool={len(top):3d} best={top[0][0]:.4f} "
              f"scaffolds={f['distinct_scaffolds_in_pool']:2d} "
              f"exact_scaffold={f['exact_scaffold_supplied']:2d} "
              f"ref_substruct={f['reference_substructure_supplied']:2d} "
              f"maxMCS={f['max_mcs_with_reference']:2d}")

    (OUT / "structural_diagnostic_v1.json").write_text(
        json.dumps(diagnostic, indent=1) + "\n")
    print(f"\nwrote {len(diagnostic)} seed banks to {seeds_dir}")
    print("wrote structural_diagnostic_v1.json (DIAGNOSTIC ONLY -- never a proposal input)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
