"""Decompose saved QED scores without changing or selecting generated outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import QED, Draw

from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    _fragment_spec,
    load_genmol_prompts,
    sascorer,
)


def decompose(molecule):
    properties = QED.properties(molecule)._asdict()
    terms = {
        key: weight * math.log(QED.ads(value, QED.adsParameters[key])) / sum(QED.WEIGHT_MEAN)
        for (key, value), weight in zip(properties.items(), QED.WEIGHT_MEAN, strict=True)
    }
    reconstructed = math.exp(sum(terms.values()))
    if abs(reconstructed - QED.qed(molecule)) > 1e-12:
        raise ValueError("QED decomposition parity failed")
    return {"properties": properties, "log_qed_terms": terms, "qed": reconstructed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--drug", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    data = json.loads(args.inspection.read_text())
    row = next(p for p in data["prompts"] if p["drug"] == args.drug)
    source = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
    prompt = next(
        p
        for p in load_genmol_prompts(source)
        if p.drug_name == args.drug and p.task == FragmentTask(data["task"])
    )
    spec = _fragment_spec(prompt.fragments[0])
    results = {}
    for arm in ("baseline", "new"):
        results[arm] = []
        for item in row[arm]["molecules"]:
            if not item["smiles"]:
                results[arm].append({"attempt": item["attempt"], "no_output": True})
                continue
            mol = Chem.MolFromSmiles(item["smiles"])
            if mol is None or not mol.HasSubstructMatch(spec.core):
                raise ValueError("invalid or missing-core saved output")
            detail = decompose(mol)
            if abs(detail["qed"] - item["qed"]) > 1e-12:
                raise ValueError("saved QED does not reproduce")
            results[arm].append({**item, **detail})
    means = {
        arm: {
            key: sum(m["log_qed_terms"][key] for m in values if not m.get("no_output"))
            / sum(not m.get("no_output", False) for m in values)
            for key in QED.WEIGHT_MEAN._fields
        }
        for arm, values in results.items()
    }
    difference = {key: means["new"][key] - means["baseline"][key] for key in means["new"]}
    examples = []
    for arm, passed in (("baseline", True), ("new", True), ("new", False)):
        examples.append(
            (
                arm,
                next(
                    m
                    for m in results[arm]
                    if not m.get("no_output") and (m["group"] == "both_pass") == passed
                ),
            )
        )
    molecules = [Chem.MolFromSmiles(m["smiles"]) for _, m in examples]
    legends = [
        f"{arm.upper()} #{m['attempt']}: {m['group']}\nQED {m['qed']:.3f}; SA {m['sa']:.3f}; HA {m['heavy_atoms']}"
        for arm, m in examples
    ]
    inputs = (args.inspection, source, Path(__file__), Path(QED.__file__), Path(sascorer.__file__))
    result = {
        "schema": "fragment_qed_decomposition_v1",
        "role": "post-generation diagnostic only; not score guidance or sampler selection",
        "example_selection": "first joint pass in baseline; first joint pass in new; first failure in new; illustration, not representative frequency",
        "scope": {
            "drug": args.drug,
            "task": data["task"],
            "attempts": {a: len(v) for a, v in results.items()},
        },
        "molecules": results,
        "mean_log_qed_terms": means,
        "new_minus_baseline_log_qed": difference,
        "interpretation": "Exact descriptive score decomposition, not causal effects of molecular edits.",
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
        "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent, prefix=".qed-audit-") as temp:
        stage = Path(temp) / "complete"
        stage.mkdir()
        picture = stage / "pass_and_fail.png"
        Draw.MolsToGridImage(
            molecules,
            legends=legends,
            molsPerRow=3,
            subImgSize=(520, 420),
            highlightAtomLists=[list(m.GetSubstructMatch(spec.core)) for m in molecules],
        ).save(picture)
        result["image_sha256"] = hashlib.sha256(picture.read_bytes()).hexdigest()
        (stage / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        stage.rename(args.output_dir)
    print(
        json.dumps(
            {"delta_log_qed": difference, "examples": [{"arm": a, **m} for a, m in examples]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
