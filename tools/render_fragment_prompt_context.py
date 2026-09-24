"""Show an exact supplied prompt beside saved outputs; no generation or selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import Draw

from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    _fragment_spec,
    load_genmol_prompts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--drug", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
    data = json.loads(args.inspection.read_text())
    prompt = next(
        p
        for p in load_genmol_prompts(source)
        if p.drug_name == args.drug and p.task == FragmentTask(data["task"])
    )
    if len(prompt.fragments) != 1:
        raise ValueError("this visualization requires a single-core prompt")
    spec = _fragment_spec(prompt.fragments[0])
    rows = next(p for p in data["prompts"] if p["drug"] == args.drug)
    supplied = Chem.MolFromSmiles(prompt.fragments[0])
    Chem.RemoveStereochemistry(supplied)
    for atom in supplied.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetProp("atomLabel", f"R{atom.GetIsotope()}")
            atom.SetIsotope(0)
    molecules = [supplied]
    legends = [
        f"SUPPLIED CORE: {spec.core.GetNumHeavyAtoms()} heavy atoms\nR1/R2: decoration sites; stereo omitted"
    ]
    highlights = [[a.GetIdx() for a in supplied.GetAtoms() if a.GetAtomicNum() != 0]]
    for arm in ("baseline", "new"):
        row = rows[arm]["molecules"][0]
        molecule = Chem.MolFromSmiles(row["smiles"])
        if molecule is None:
            raise ValueError("first saved attempt has no valid output")
        matches = molecule.GetSubstructMatches(spec.core, useChirality=False)
        if not matches:
            raise ValueError("saved molecule lacks the supplied core")
        molecules.append(molecule)
        highlights.append(list(min(matches)))
        legends.append(
            f"{arm.upper()} attempt 0: {row['heavy_atoms']} heavy atoms\n"
            f"QED={row['qed']:.3f}; SA={row['sa']:.3f}"
        )
    args.output_dir.mkdir(parents=True)
    picture = args.output_dir / "prompt_and_completions.png"
    Draw.MolsToGridImage(
        molecules,
        molsPerRow=3,
        subImgSize=(520, 420),
        legends=legends,
        highlightAtomLists=highlights,
    ).save(picture)
    files = (source, args.inspection, Path(__file__), picture)
    receipt = {
        "schema": "fragment_prompt_context_visual_v1",
        "role": "diagnostic only; first saved attempt per arm, no quality selection",
        "prompt": prompt.fragments[0],
        "core_heavy_atoms": spec.core.GetNumHeavyAtoms(),
        "attachment_requirements": spec.attachment_requirements,
        "stereo": "not asserted; omitted in prompt drawing to match current model support",
        "inputs_and_image_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
