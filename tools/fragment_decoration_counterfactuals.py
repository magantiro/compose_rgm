"""Diagnostic manual simplifications, never a sampler or benchmark improvement."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import QED, Draw

from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    _fragment_spec,
    check_fragment_constraint,
    load_genmol_prompts,
    sascorer,
)


def simplify(smiles, prompt, *, all_decorations=False):
    """Replace largest pendant (or all pendants) by methyl, preserving core graph."""
    mol = Chem.MolFromSmiles(smiles)
    spec = _fragment_spec(prompt.fragments[0])
    matches = mol.GetSubstructMatches(spec.core, useChirality=False, uniquify=False)
    if not matches:
        raise ValueError("missing supplied core")
    for match in sorted(matches):
        core = set(match)
        allowed = {match[i]: count for i, count in spec.attachment_requirements}
        outside = set(range(mol.GetNumAtoms())) - core
        components = []
        while outside:
            pending, component = [min(outside)], set()
            while pending:
                node = pending.pop()
                if node in component:
                    continue
                component.add(node)
                pending.extend(
                    a.GetIdx()
                    for a in mol.GetAtomWithIdx(node).GetNeighbors()
                    if a.GetIdx() in outside and a.GetIdx() not in component
                )
            outside -= component
            contacts = [
                (a, n.GetIdx())
                for a in sorted(component)
                for n in mol.GetAtomWithIdx(a).GetNeighbors()
                if n.GetIdx() in core
            ]
            if len(contacts) != 1 or contacts[0][1] not in allowed:
                break
            child, anchor = contacts[0]
            if mol.GetBondBetweenAtoms(child, anchor).GetBondType() != Chem.BondType.SINGLE:
                break
            components.append((component, anchor))
        else:
            if components:
                break
    else:
        raise ValueError("outside regions are not single-boundary authorized pendants")
    components.sort(key=lambda x: (-len(x[0]), match.index(x[1])))
    selected = components if all_decorations else components[:1]
    edit = Chem.RWMol(mol)
    for atom in edit.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    removed = set().union(*(c for c, _ in selected))
    for idx in sorted(removed, reverse=True):
        edit.RemoveAtom(idx)
    new_positions = {a.GetAtomMapNum() - 1: a.GetIdx() for a in edit.GetAtoms()}
    for _, anchor in selected:
        methyl = edit.AddAtom(Chem.Atom(6))
        edit.AddBond(new_positions[anchor], methyl, Chem.BondType.SINGLE)
    result = edit.GetMol()
    Chem.SanitizeMol(result)
    for old in core:
        a, b = mol.GetAtomWithIdx(old), result.GetAtomWithIdx(new_positions[old])
        if (a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsAromatic()) != (
            b.GetAtomicNum(),
            b.GetFormalCharge(),
            b.GetIsAromatic(),
        ):
            raise ValueError("counterfactual altered locked atom")
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in core and b in core:
            new_bond = result.GetBondBetweenAtoms(new_positions[a], new_positions[b])
            if new_bond is None or new_bond.GetBondType() != bond.GetBondType():
                raise ValueError("counterfactual altered locked bond")
    for atom in result.GetAtoms():
        atom.SetAtomMapNum(0)
    text = Chem.MolToSmiles(result)
    if len(Chem.GetMolFrags(result)) != 1 or not check_fragment_constraint(prompt, text).satisfied:
        raise ValueError("counterfactual violated connectedness or prompt interfaces")
    return text


def metrics(smiles):
    mol = Chem.MolFromSmiles(smiles)
    qed, sa = QED.qed(mol), sascorer.calculateScore(mol)
    return {
        "smiles": smiles,
        "qed": qed,
        "sa": sa,
        "joint_pass": qed >= 0.6 and sa <= 4,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--drug", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    data = json.loads(args.inspection.read_text())
    source = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
    prompt = next(
        p
        for p in load_genmol_prompts(source)
        if p.drug_name == args.drug and p.task == FragmentTask(data["task"])
    )
    rows = next(p for p in data["prompts"] if p["drug"] == args.drug)["new"]["molecules"]
    results = []
    for row in rows:
        if not row["smiles"]:
            results.append({"attempt": row["attempt"], "refusal": "no saved output"})
            continue
        item = {"attempt": row["attempt"], "original": metrics(row["smiles"])}
        for name, both in (("largest_to_methyl", False), ("all_to_methyl", True)):
            try:
                item[name] = metrics(simplify(row["smiles"], prompt, all_decorations=both))
            except ValueError as error:
                item[name] = {"refusal": str(error)}
        results.append(item)
    summary = {}
    for name in ("original", "largest_to_methyl", "all_to_methyl"):
        values = [r[name] for r in results if name in r and "refusal" not in r[name]]
        summary[name] = {
            "completed": len(values),
            "attempts": len(results),
            "joint_pass": sum(v["joint_pass"] for v in values),
            "unique": len({v["smiles"] for v in values}),
            "mean_qed": sum(v["qed"] for v in values) / len(values),
            "mean_sa": sum(v["sa"] for v in values) / len(values),
        }
    inputs = (args.inspection, source, Path(__file__), Path(QED.__file__), Path(sascorer.__file__))
    output = {
        "schema": "manual_decoration_counterfactual_v1",
        "results": results,
        "summary": summary,
        "role": "manual post-generation diagnosis only; RDKit edited, not COMPOSE executed; no benchmark claim, no inference selection",
        "rule": "methyl replacement fixed before scoring; largest non-core single-boundary pendant or all such pendants; no drug rules",
        "limitations": "changes size and chemical content jointly; no isolated size causality; current non-stereo model scope",
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
        "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent, prefix=".manual-edit-") as temp:
        stage = Path(temp) / "complete"
        stage.mkdir()
        first = results[0]
        names = ("original", "largest_to_methyl", "all_to_methyl")
        mols = [Chem.MolFromSmiles(first[n]["smiles"]) for n in names]
        spec = _fragment_spec(prompt.fragments[0])
        image = stage / "first_attempt.png"
        Draw.MolsToGridImage(
            mols,
            molsPerRow=3,
            subImgSize=(520, 420),
            highlightAtomLists=[list(m.GetSubstructMatch(spec.core)) for m in mols],
            legends=[f"{n}\nQED {first[n]['qed']:.3f}; SA {first[n]['sa']:.3f}" for n in names],
        ).save(image)
        output["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
        (stage / "result.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        stage.rename(args.output_dir)
    print(json.dumps({"summary": summary, "first": results[0]}, indent=2))


if __name__ == "__main__":
    main()
