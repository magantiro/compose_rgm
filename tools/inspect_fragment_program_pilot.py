"""Inspect every completed prompt in one frozen pilot, never selecting outputs."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import QED, Draw
from run_fragment_attachment_library_pilot import _atomic_json

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts, sascorer
from compose_v4.benchmark.training_attachment_fragments import physical_sha256


def describe(records, metrics):
    rows, seen = [], set()
    for index, record in enumerate(records):
        smiles = record["committed_smiles"]
        if not smiles:
            rows.append({"attempt": index, "smiles": None, "group": "no_output"})
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError("invalid/disconnected committed endpoint")
        qed, sa = QED.qed(molecule), sascorer.calculateScore(molecule)
        a, b = qed >= 0.6, sa <= 4
        rows.append(
            {
                "attempt": index,
                "smiles": smiles,
                "duplicate": smiles in seen,
                "qed": qed,
                "sa": sa,
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "rings": molecule.GetRingInfo().NumRings(),
                "fluorines": sum(atom.GetAtomicNum() == 9 for atom in molecule.GetAtoms()),
                "group": "both_pass"
                if a and b
                else "sa_fail"
                if a
                else "qed_fail"
                if b
                else "both_fail",
                "program": record.get("selected_capabilities"),
            }
        )
        seen.add(smiles)
    unique_pass = sum(row["group"] == "both_pass" and not row["duplicate"] for row in rows)
    quality = 100 * unique_pass / len(records)
    if abs(quality - metrics["quality"]) > 1e-10:
        raise ValueError("diagnostic decomposition does not reproduce official quality")
    return {
        "attempts": len(records),
        "groups": dict(Counter(row["group"] for row in rows)),
        "unique_joint_pass": unique_pass,
        "official": metrics,
        "molecules": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["motif_extension", "scaffold_decoration"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    prompts_path = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
    prompts = [p for p in load_genmol_prompts(prompts_path) if p.task == FragmentTask(args.task)]
    row_paths = [args.pilot_dir / "rows" / f"{p.task.value}_{p.drug_name}.json" for p in prompts]
    included = [p.is_file() for p in row_paths]
    if not any(included) or (not args.allow_partial and not all(included)):
        raise ValueError("required completed prompt rows not available")
    inputs = {
        str(p.resolve()): physical_sha256(p)
        for p in (Path(__file__), prompts_path, args.pilot_dir / "manifest.json")
    }
    results, images = [], {}
    for prompt, row_path in zip(prompts, row_paths, strict=True):
        if not row_path.is_file():
            continue
        row = json.loads(row_path.read_text())
        baseline_path = (
            args.baseline_dir / "shards" / f"{prompt.task.value}__{prompt.drug_name}__baseline.json"
        )
        paths = [
            args.pilot_dir / "attempts" / f"{prompt.task.value}_{prompt.drug_name}_{i:03d}.json"
            for i in range(20)
        ]
        attempts = [json.loads(p.read_text()) for p in paths]
        if not all(a["complete"] for a in attempts) or row["attempts"] != 20:
            raise ValueError("refusing to inspect unfinished prompt as a completed row")
        inputs.update(
            {str(p.resolve()): physical_sha256(p) for p in [row_path, baseline_path, *paths]}
        )
        baseline = json.loads(baseline_path.read_text())
        result = {"drug": prompt.drug_name}
        for arm, records in (("baseline", baseline["attempt_records"]), ("new", attempts)):
            result[arm] = describe(records, row[arm])
            molecules, legends = [], []
            for r in result[arm]["molecules"]:
                molecules.append(Chem.MolFromSmiles(r["smiles"]) if r["smiles"] else None)
                legends.append(
                    f"#{r['attempt']} {r['group']}"
                    + (f" Q={r['qed']:.2f} SA={r['sa']:.2f}" if r["smiles"] else "")
                )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            image_path = args.output_dir / f"{prompt.drug_name}_{arm}.png"
            Draw.MolsToGridImage(
                molecules, molsPerRow=4, subImgSize=(360, 280), legends=legends
            ).save(image_path)
            images[image_path.name] = physical_sha256(image_path)
        results.append(result)
    output = {
        "schema": "fragment_complete_program_pilot_inspection_v1",
        "role": "post-generation development diagnostic; no selection or sampler change",
        "partial": len(results) != 10,
        "task": args.task,
        "completed_prompts": len(results),
        "expected_prompts": 10,
        "inputs": inputs,
        "image_hashes": images,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "mean_metrics_completed_prompts_only": {
            arm: {
                metric: float(np.mean([r[arm]["official"][metric] for r in results]))
                for metric in ("quality", "uniqueness", "diversity", "validity")
            }
            for arm in ("baseline", "new")
        },
        "prompts": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "inspection.json", output)
    print(
        json.dumps(
            {
                k: output[k]
                for k in (
                    "task",
                    "partial",
                    "completed_prompts",
                    "mean_metrics_completed_prompts_only",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
