"""How far is COMPOSE's initialization from a DECLARED target? Zero oracle calls.

Sizes the transport the planner must perform, before the planner is built. The
declared target is read from the goal-specification audit artifact, so no molecular
content is hand-written here.

Reported per task: best Tanimoto from the init bank, the largest common substructure
(ring-complete, so a partial ring is not counted as retained), and the implied
retain / delete / install split -- which is exactly the
`(retain core, R_delete, H_install)` the planner must emit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdFMCS

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def probe(target_smiles: str, bank: list[str], *, top: int = 40) -> dict:
    target = Chem.MolFromSmiles(target_smiles)
    if target is None:
        raise ValueError(f"declared target does not parse: {target_smiles!r}")
    target_heavy = target.GetNumHeavyAtoms()
    target_fp = _GEN.GetFingerprint(target)
    scored = []
    for smiles in bank:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        scored.append(
            (DataStructs.TanimotoSimilarity(target_fp, _GEN.GetFingerprint(mol)), smiles, mol)
        )
    scored.sort(key=lambda row: row[0], reverse=True)

    def mcs_atoms(mol):
        # completeRingsOnly: a partial ring is not a retained core, and counting it as
        # one would understate the install work by a whole ring system.
        return rdFMCS.FindMCS(
            [mol, target], timeout=20, completeRingsOnly=True, ringMatchesRingOnly=True
        ).numAtoms

    best = max(scored[:top], key=lambda row: mcs_atoms(row[2]))
    retained = mcs_atoms(best[2])
    return {
        "target_smiles": target_smiles,
        "target_heavy_atoms": target_heavy,
        "bank_size": len(scored),
        "best_tanimoto": round(scored[0][0], 4),
        "median_tanimoto": round(sorted(row[0] for row in scored)[len(scored) // 2], 4),
        "best_mcs_source": best[1],
        "retained_atoms": retained,
        "retained_fraction_of_target": round(retained / target_heavy, 4),
        "atoms_to_install": target_heavy - retained,
        "atoms_to_delete": best[2].GetNumHeavyAtoms() - retained,
        "implied_primitive_transport": (target_heavy - retained)
        + (best[2].GetNumHeavyAtoms() - retained),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["celecoxib_rediscovery"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    audit = json.loads(
        (ROOT / "diagnostics/pmo_discovery_v1/goal_specification_audit_v1.json").read_text()
    )
    rows = {row["task"]: row for row in audit["rows"]}
    bank = json.loads((ROOT / "docs/PMO_INIT_BANK.json").read_text())
    if isinstance(bank, dict):
        bank = bank.get("smiles") or bank.get("molecules")

    results = []
    for task in args.tasks:
        row = rows.get(task)
        declared = [d for d in (row or {}).get("declared", []) if d["kind"] == "smiles"]
        if not declared:
            print(f"{task}: no declared SMILES target; skipped")
            continue
        result = probe(declared[0]["value"], bank)
        result["task"] = task
        results.append(result)
        print(
            f"{task}: target {result['target_heavy_atoms']} heavy; "
            f"best Tanimoto {result['best_tanimoto']} (median {result['median_tanimoto']}); "
            f"retain {result['retained_atoms']} "
            f"({result['retained_fraction_of_target']:.0%} of target), "
            f"delete {result['atoms_to_delete']}, install {result['atoms_to_install']} "
            f"-> ~{result['implied_primitive_transport']} primitives"
        )
    report = {
        "schema_version": "pmo_transport_distance_probe_v1",
        "oracle_calls": 0,
        "init_bank": "docs/PMO_INIT_BANK.json",
        "results": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
