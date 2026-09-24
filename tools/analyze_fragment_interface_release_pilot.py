"""Decompose QED/SA and molecular size on all committed pilot endpoints.

This reads frozen unit artifacts; it never generates molecules or selects a
controller. The pinned IVG implementation supplies the quality components.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from compose_v4.benchmark.fragment_official_metrics import _official_module_path

ROOT = Path(__file__).resolve().parents[1]
PILOT = Path("diagnostics/fragment_interface_release_pilot_v1")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    RDLogger.DisableLog("rdApp.warning")
    _official_module_path()  # fetch and hash-verify before import
    from in_virtuo_gen.utils.mol import compute_single_property

    summary = json.loads((PILOT / "summary.json").read_text())
    by_group: dict[str, dict[str, list[dict]]] = {}
    unit_hashes: dict[str, str] = {}
    for unit_name in summary["units"]:
        path = Path(unit_name)
        unit_hashes[str(path)] = sha256(path)
        unit = json.loads(path.read_text())
        if unit["identity"] != summary["identity"]:
            raise RuntimeError(f"identity mismatch in {path}")
        arm, task, drug = unit["arm"], unit["task"], unit["drug"]
        row = unit["result"]["per_drug"][drug][0]
        records = by_group.setdefault(task, {}).setdefault(arm, [])
        for attempt in row["attempt_records"]:
            smiles = attempt["committed_smiles"]
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                raise AssertionError(f"invalid committed molecule: {path}/{smiles}")
            sa, qed = compute_single_property(smiles)
            records.append(
                {
                    "drug": drug,
                    "attempt_index": attempt["attempt_index"],
                    "smiles": smiles,
                    "qed": float(qed),
                    "sa": float(sa),
                    "qed_pass": bool(qed >= 0.6),
                    "sa_pass": bool(sa <= 4.0),
                    "joint_pass": bool(qed >= 0.6 and sa <= 4.0),
                    "heavy_atoms": mol.GetNumHeavyAtoms(),
                    "exact_mw": float(Descriptors.ExactMolWt(mol)),
                    "formal_charge": sum(a.GetFormalCharge() for a in mol.GetAtoms()),
                    "fluorine_atoms": sum(
                        a.GetAtomicNum() == 9 for a in mol.GetAtoms()
                    ),
                    "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                }
            )
    aggregate = {}
    for task, by_arm in by_group.items():
        aggregate[task] = {}
        for arm, records in by_arm.items():
            n = len(records)
            unique = {record["smiles"]: record for record in records}
            aggregate[task][arm] = {
                "committed": n,
                "unique_committed": len(unique),
                "qed_mean": sum(x["qed"] for x in records) / n,
                "sa_mean": sum(x["sa"] for x in records) / n,
                "qed_pass": sum(x["qed_pass"] for x in records),
                "sa_pass": sum(x["sa_pass"] for x in records),
                "joint_pass": sum(x["joint_pass"] for x in records),
                "unique_joint_pass": sum(x["joint_pass"] for x in unique.values()),
                "heavy_atoms_mean": sum(x["heavy_atoms"] for x in records) / n,
                "exact_mw_mean": sum(x["exact_mw"] for x in records) / n,
                "fluorine_atoms_mean": sum(x["fluorine_atoms"] for x in records) / n,
                "records": records,
            }
    output = {
        "schema": "fragment_interface_release_qualitative_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_identity": summary["identity"],
        "summary_sha256": sha256(PILOT / "summary.json"),
        "unit_sha256": unit_hashes,
        "analysis_sha256": sha256(Path(__file__)),
        "thresholds": {"qed_min": 0.6, "sa_max": 4.0},
        "aggregates": aggregate,
    }
    path = PILOT / "qualitative_descriptors.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True))
    temporary.replace(path)
    for task, rows in aggregate.items():
        for arm, values in rows.items():
            print(
                f"{task}/{arm}: n={values['committed']}, "
                f"QED mean={values['qed_mean']:.3f}, SA mean={values['sa_mean']:.3f}, "
                f"QED-pass={values['qed_pass']}, SA-pass={values['sa_pass']}, "
                f"joint={values['joint_pass']}, "
                f"unique-joint={values['unique_joint_pass']}, "
                f"heavy={values['heavy_atoms_mean']:.1f}"
            )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
