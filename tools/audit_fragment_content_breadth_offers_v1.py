"""Post-lock quality diagnosis of the matched decoration content-breadth pilot.

Quality is read only from immutable completed panels. This module is not used
by proposal construction or endpoint selection.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import QED

from compose_v4.benchmark.fragment_constrained import sascorer
from compose_v4.benchmark.training_attachment_fragments import physical_sha256

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "diagnostics/fragment_content_breadth_pilot_v1"
ARMS = ("frozen", "uniform_within_cell")


def descriptor(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"locked decoration product does not parse: {smiles}")
    qed = float(QED.qed(mol))
    sa = float(sascorer.calculateScore(mol))
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "qed_pass": qed >= 0.6,
        "sa_pass": sa <= 4.0,
        "quality_pass": qed >= 0.6 and sa <= 4.0,
    }


def audit() -> dict:
    summary_path = PILOT / "summary.json"
    manifest_path = PILOT / "manifest.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("schema") != "fragment_content_breadth_pilot_result_v1":
        raise ValueError(f"unexpected or incomplete pilot: {summary_path}")
    if summary["attempts_per_arm"] != 200 or summary["outputs"] != dict.fromkeys(ARMS, 200):
        raise ValueError("decoration pilot lacks all 400 locked outputs")
    if summary["manifest_sha256"] != physical_sha256(manifest_path):
        raise ValueError("decoration pilot manifest changed")

    input_hashes = {
        str(summary_path.relative_to(ROOT)): physical_sha256(summary_path),
        str(manifest_path.relative_to(ROOT)): physical_sha256(manifest_path),
    }
    prompt_rows = []
    for arm in ARMS:
        row_paths = sorted((PILOT / "rows" / arm).glob("*.json"))
        if len(row_paths) != 10:
            raise ValueError(f"incomplete {arm} prompt rows")
        for row_path in row_paths:
            relative_row = str(row_path.relative_to(PILOT))
            row_hash = physical_sha256(row_path)
            if summary["row_sha256"].get(relative_row) != row_hash:
                raise ValueError(f"decoration row changed: {row_path}")
            input_hashes[str(row_path.relative_to(ROOT))] = row_hash
            row = json.loads(row_path.read_text())
            drug = row["drug"]
            if row["arm"] != arm or row["attempts"] != 20 or row["outputs"] != 20:
                raise ValueError(f"incomplete or mismatched prompt row: {row_path}")
            lock_path = PILOT / "locks" / arm / f"{drug}.json"
            if row["lock_sha256"] != physical_sha256(lock_path):
                raise ValueError(f"decoration lock changed: {lock_path}")
            input_hashes[str(lock_path.relative_to(ROOT))] = physical_sha256(lock_path)
            lock = json.loads(lock_path.read_text())
            if len(lock["samples"]) != 20 or len(lock["attempt_hashes"]) != 20:
                raise ValueError(f"incomplete decoration lock: {lock_path}")
            selected_pass = 0
            selected_qed_fail = 0
            selected_sa_fail = 0
            any_pass = 0
            supported_offers = 0
            passing_offers = 0
            selected_atoms = 0
            all_offer_atoms = 0
            for index, selected_smiles in enumerate(lock["samples"]):
                relative_attempt = f"attempts/{arm}/{drug}_{index:03d}.json"
                attempt_path = PILOT / relative_attempt
                digest = physical_sha256(attempt_path)
                if lock["attempt_hashes"].get(relative_attempt) != digest:
                    raise ValueError(f"decoration attempt changed: {attempt_path}")
                input_hashes[str(attempt_path.relative_to(ROOT))] = digest
                attempt = json.loads(attempt_path.read_text())
                if (
                    attempt["arm"] != arm
                    or attempt["drug"] != drug
                    or attempt["attempt_index"] != index
                ):
                    raise ValueError(f"mismatched decoration attempt: {attempt_path}")
                if attempt["panel"]["selected_smiles"] != selected_smiles:
                    raise ValueError(f"selected endpoint differs from lock: {attempt_path}")
                selected = descriptor(selected_smiles)
                selected_pass += selected["quality_pass"]
                selected_qed_fail += not selected["qed_pass"]
                selected_sa_fail += not selected["sa_pass"]
                selected_atoms += selected["heavy_atoms"]
                offers = [
                    descriptor(offer["endpoint"])
                    for offer in attempt["panel"]["offered"]
                    if offer["status"] == "model_supported"
                ]
                supported_offers += len(offers)
                passing_offers += sum(offer["quality_pass"] for offer in offers)
                any_pass += any(offer["quality_pass"] for offer in offers)
                all_offer_atoms += sum(offer["heavy_atoms"] for offer in offers)
            if selected_pass < row["metrics"]["quality"] / 5:
                raise ValueError(f"official quality exceeds raw passing outputs: {row_path}")
            prompt_rows.append(
                {
                    "arm": arm,
                    "drug": drug,
                    "attempts": 20,
                    "official_quality_percent": row["metrics"]["quality"],
                    "official_diversity": row["metrics"]["diversity"],
                    "selected_quality_pass": selected_pass,
                    "selected_qed_fail": selected_qed_fail,
                    "selected_sa_fail": selected_sa_fail,
                    "attempts_with_any_quality_offer": any_pass,
                    "model_supported_offers": supported_offers,
                    "quality_passing_offers": passing_offers,
                    "selected_mean_heavy_atoms": selected_atoms / 20,
                    "supported_offer_mean_heavy_atoms": (
                        all_offer_atoms / supported_offers if supported_offers else None
                    ),
                }
            )
    input_hashes[str(Path(__file__).relative_to(ROOT))] = physical_sha256(Path(__file__))
    return {
        "schema": "fragment_content_breadth_offer_audit_v1",
        "role": "post-lock development diagnosis; never proposal-time quality guidance",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "input_sha256": dict(sorted(input_hashes.items())),
        "prompts": prompt_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = audit()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".breadth-offers-") as stage:
        temporary = Path(stage) / "result.json"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.rename(output)
    print(json.dumps({"prompts": len(result["prompts"]), "output": str(output)}))


if __name__ == "__main__":
    main()
