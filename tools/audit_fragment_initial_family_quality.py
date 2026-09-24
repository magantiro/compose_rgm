"""Decompose quality and inspect molecules from the frozen seed-10 pilot.

This is a diagnostic on development outputs. It does not select a controller or
change the official metric; per-molecule properties use the pinned IVG code.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import rdkit
from fetch_official_fragment_evaluator import OFFICIAL_BLOBS, verify_only
from rdkit import Chem
from rdkit.Chem import Descriptors

from compose_v4.benchmark.fragment_official_metrics import _official_module_path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "diagnostics/fragment_initial_family_dev_v1"
ARMS = ("baseline", "conditioned", "conditioned_strict")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    verified = verify_only()
    evaluator_hash = OFFICIAL_BLOBS["in_virtuo_gen/train_utils/metrics.py"][1]
    mol_hash = OFFICIAL_BLOBS["in_virtuo_gen/utils/mol.py"][1]
    if evaluator_hash not in verified.values() or mol_hash not in verified.values():
        raise RuntimeError("official metric/property implementation not verified")
    _official_module_path()
    from in_virtuo_gen.utils.mol import compute_single_property

    records = []
    inputs = {}
    for arm in ARMS:
        path = FOLDER / f"{arm}_seed10_n20_all10.json"
        inputs[str(path.relative_to(ROOT))] = sha256(path)
        payload = json.loads(path.read_text())
        for drug, [entry] in sorted(
            payload["results"]["superstructure_generation"]["per_drug"].items()
        ):
            seen = set()
            passes = 0
            for attempt in entry["attempt_records"]:
                smiles = attempt["emitted_smiles"]
                if not smiles or smiles in seen:
                    continue
                seen.add(smiles)
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise RuntimeError(f"invalid emitted molecule: {arm}/{drug}")
                sa, qed = compute_single_property(smiles)
                quality = bool(qed >= 0.6 and sa <= 4.0)
                passes += quality
                records.append(
                    {
                        "arm": arm,
                        "drug": drug,
                        "attempt_index": attempt["attempt_index"],
                        "smiles": smiles,
                        "sa": float(sa),
                        "qed": float(qed),
                        "quality_pass": quality,
                        "heavy_atoms": mol.GetNumHeavyAtoms(),
                        "molecular_weight": float(Descriptors.ExactMolWt(mol)),
                        "rings": int(mol.GetRingInfo().NumRings()),
                        "aromatic_atoms": sum(atom.GetIsAromatic() for atom in mol.GetAtoms()),
                        "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
                    }
                )
            if abs(100.0 * passes / 20 - entry["official"]["quality"]) > 1e-10:
                raise RuntimeError(f"quality reconstruction mismatch: {arm}/{drug}")
    groups = defaultdict(list)
    for record in records:
        groups[(record["arm"], record["drug"])].append(record)
    summary = []
    for (arm, drug), cohort in sorted(groups.items()):
        n = len(cohort)
        summary.append(
            {
                "arm": arm,
                "drug": drug,
                "unique_emitted": n,
                "qed_pass": sum(row["qed"] >= 0.6 for row in cohort),
                "sa_pass": sum(row["sa"] <= 4.0 for row in cohort),
                "joint_pass": sum(row["quality_pass"] for row in cohort),
                "mean_qed": sum(row["qed"] for row in cohort) / n,
                "mean_sa": sum(row["sa"] for row in cohort) / n,
                "mean_heavy_atoms": sum(row["heavy_atoms"] for row in cohort) / n,
                "mean_rings": sum(row["rings"] for row in cohort) / n,
            }
        )
    result = {
        "schema": "fragment_initial_family_quality_audit_v1",
        "evidence_role": "Post-hoc molecule-level diagnosis of matched development outputs; not selection evidence",
        "input_sha256": inputs,
        "official_evaluator_sha256": evaluator_hash,
        "official_property_sha256": mol_hash,
        "auditor_sha256": sha256(Path(__file__)),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "software": {"python": sys.version.split()[0], "rdkit": rdkit.__version__},
        "summary": summary,
        "molecules": records,
    }
    output = FOLDER / "quality_seed10_n20_all10.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=FOLDER, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, output)
    for row in summary:
        if row["drug"] in {"ERLOTINIB", "FUTIBATINIB", "SPIRAPRIL"}:
            print(
                row["drug"],
                row["arm"],
                "QED",
                row["qed_pass"],
                "SA",
                row["sa_pass"],
                "joint",
                row["joint_pass"],
                "heavy",
                round(row["mean_heavy_atoms"], 1),
            )


if __name__ == "__main__":
    main()
