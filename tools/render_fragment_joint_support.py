"""Draw frozen support endpoints without evaluating or selecting by quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import Draw

from compose_v4.benchmark.fragment_constrained import _fragment_spec, load_genmol_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
    summary_path = args.support_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["mode"] != "support":
        raise ValueError("expected completed support-only run")
    prompts = {(p.task.value, p.drug_name): p for p in load_genmol_prompts(source)}
    inputs = [source, summary_path, args.support_dir / "manifest.json", Path(__file__)]
    groups, records = {}, []
    for name, digest in sorted(summary["row_sha256"].items()):
        path = args.support_dir / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"support row changed: {path}")
        inputs.append(path)
        row = json.loads(path.read_text())
        prompt = prompts[row["task"], row["drug"]]
        spec = _fragment_spec(prompt.fragments[0])
        supplied = Chem.MolFromSmiles(prompt.fragments[0])
        Chem.RemoveStereochemistry(supplied)
        for atom in supplied.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetProp("atomLabel", f"R{atom.GetIsotope()}")
                atom.SetIsotope(0)
        item = next((a for a in row["attempts"] if a["committed_smiles"]), None)
        molecule = Chem.MolFromSmiles(item["committed_smiles"]) if item else None
        match = molecule.GetSubstructMatch(spec.core) if molecule else ()
        if molecule and not match:
            raise ValueError(f"missing visual core: {row['task']} {row['drug']}")
        entry = groups.setdefault(row["task"], {"mols": [], "legends": [], "highlights": []})
        entry["mols"].extend([supplied, molecule])
        entry["highlights"].extend(
            [[a.GetIdx() for a in supplied.GetAtoms() if a.GetAtomicNum() != 0], list(match)]
        )
        detail = (
            f"attempt {item['attempt_index']}: "
            f"{item['selected_structure']['heavy_atoms']} atoms, "
            f"{item['selected_structure']['rings']} rings"
            if item
            else "NO OUTPUT"
        )
        entry["legends"].extend([f"{row['drug']}: supplied core", detail])
        records.append(
            {
                "task": row["task"],
                "drug": row["drug"],
                "attempt": item["attempt_index"] if item else None,
            }
        )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent, prefix=".support-view-") as tmp:
        stage = Path(tmp) / "complete"
        stage.mkdir()
        images = {}
        for task, group in groups.items():
            image = stage / f"{task}.png"
            Draw.MolsToGridImage(
                group["mols"],
                molsPerRow=2,
                subImgSize=(520, 360),
                legends=group["legends"],
                highlightAtomLists=group["highlights"],
            ).save(image)
            images[image.name] = hashlib.sha256(image.read_bytes()).hexdigest()
        receipt = {
            "schema": "fragment_joint_support_visual_v1",
            "role": "first emitted attempt per prompt; no quality calculation or selection",
            "stereo": "not asserted; supplied prompt stereo omitted to match model support",
            "records": records,
            "input_sha256": {
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
            },
            "image_sha256": images,
            "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        }
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stage.rename(args.output_dir)
    print(json.dumps({"prompts": len(records), "images": images}, sort_keys=True))


if __name__ == "__main__":
    main()
