"""Publish compact hash-bound census receipts and optional fixed saved-molecule grids."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw
from run_fragment_attachment_library_pilot import _atomic_json

from compose_v4.benchmark.fragment_constrained import _fragment_spec, load_genmol_prompts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_census(data):
    result = {k: v for k, v in data.items() if k != "prompts"}
    result["source_schema"] = result["schema"]
    result["schema"] = "fragment_completion_census_compact_v1"
    result["prompts"] = []
    for row in data["prompts"]:
        result["prompts"].append(
            {
                **{k: row[k] for k in ("task", "drug", "core")},
                **{
                    arm: {k: v for k, v in row[arm].items() if k != "molecules"}
                    for arm in ("baseline", "new")
                },
            }
        )
    if sum(p["new"]["attempts"] for p in result["prompts"]) != result["attempts_per_arm"]:
        raise ValueError("compact receipt attempt denominator disagrees")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-maribavir", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    data = json.loads(args.census.read_text())
    result = compact_census(data)
    result["source_census"] = {"path": str(args.census.resolve()), "sha256": sha(args.census)}
    result["summarizer_sha256"] = sha(Path(__file__))
    args.output_dir.mkdir(parents=True)
    if args.render_maribavir:
        prompt_path = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
        prompts = {(p.task.value, p.drug_name): p for p in load_genmol_prompts(prompt_path)}
        images = {}

        def render(rows, filename):
            molecules, legends, highlights = [], [], []
            for row, molecule in rows:
                mol = Chem.MolFromSmiles(molecule["smiles"]) if molecule["smiles"] else None
                molecules.append(mol)
                core = _fragment_spec(prompts[(row["task"], row["drug"])].fragments[0]).core
                highlights.append(list(mol.GetSubstructMatch(core)) if mol else [])
                legends.append(
                    f"{row['drug']} #{molecule['attempt']} "
                    + (
                        f"HA {molecule['heavy_atoms']} Q {molecule['qed']:.2f} SA {molecule['sa']:.2f}"
                        if mol
                        else "no output"
                    )
                )
            path = args.output_dir / filename
            Draw.MolsToGridImage(
                molecules,
                legends=legends,
                highlightAtomLists=highlights,
                molsPerRow=4,
                subImgSize=(360, 290),
            ).save(path)
            images[filename] = sha(path)

        maribavir = next(
            r
            for r in data["prompts"]
            if r["task"] == "scaffold_decoration" and r["drug"] == "MARIBAVIR"
        )
        for arm in ("baseline", "new"):
            render([(maribavir, m) for m in maribavir[arm]["molecules"]], f"maribavir_{arm}.png")
        render(
            [
                (r, r["new"]["molecules"][0])
                for r in data["prompts"]
                if r["task"] == "scaffold_decoration"
            ],
            "all_decoration_first_saved_attempt.png",
        )
        result["visualization"] = {
            "selection": "all twenty saved Maribavir attempts per arm; first saved attempt for every decoration prompt; no quality selection",
            "core_highlight": "retained dummy-stripped prompt substructure",
            "prompt_sha256": sha(prompt_path),
            "images_sha256": images,
        }
    _atomic_json(args.output_dir / "receipt.json", result)
    print(
        json.dumps(
            {
                "completed_prompts": result["completed_prompts"],
                "partial": result["partial"],
                "output": str(args.output_dir / "receipt.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
