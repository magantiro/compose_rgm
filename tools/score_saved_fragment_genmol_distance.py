"""Evaluate GenMol's exact 1024-bit original-drug distance on a sealed pilot."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_official_metrics import (
    assert_emission_invariants,
    official_unique_valid,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_function(source):
    # Isolate only the inspected pure function; do not import its model/CLI code.
    if sha(source) != "157be112b3beb76541c1915d78cea03c9b7a826ed4158122ff2a2d023aad456e":
        raise ValueError("GenMol metric source differs from pinned public revision")
    nodes = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "get_distance"
    ]
    if len(nodes) != 1:
        raise ValueError("expected exactly one official distance function")
    namespace = {"Chem": Chem, "DataStructs": DataStructs, "AllChem": AllChem, "np": np}
    # Only the exact SHA-bound, inspected pure evaluator is executable here.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    return namespace["get_distance"]


def evaluate(samples, reference, function):
    assert_emission_invariants(samples)
    unique = official_unique_valid(samples)
    value = float(function(reference, pd.DataFrame({"smiles": unique}))) if unique else None
    return {
        "attempts": len(samples),
        "outputs": sum(bool(s) for s in samples),
        "unique_valid": len(unique),
        "distance": value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--genmol-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    function = official_function(args.genmol_source)
    if evaluate(["CCO", "CCO", ""], "CCO", function)["distance"] != 0.0:
        raise ValueError("exact reference/duplicate/failure fixture failed")
    prompt_path = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
    summary_path = args.pilot_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    paths = {
        Path(__file__),
        prompt_path,
        summary_path,
        args.genmol_source,
        args.pilot_dir / "manifest.json",
        args.baseline_dir / "manifest.json",
    }
    rows = []
    for prompt in load_genmol_prompts(prompt_path):
        if prompt.task not in (FragmentTask.MOTIF_EXTENSION, FragmentTask.SCAFFOLD_DECORATION):
            continue
        sample_paths = [
            args.pilot_dir / "attempts" / f"{prompt.task.value}_{prompt.drug_name}_{i:03d}.json"
            for i in range(20)
        ]
        samples = []
        for path in sample_paths:
            if summary["attempt_hashes"][str(path.relative_to(args.pilot_dir))] != sha(path):
                raise ValueError(f"sealed attempt changed: {path}")
            attempt = json.loads(path.read_text())
            if not attempt["complete"]:
                raise ValueError("unfinished saved attempt")
            samples.append(attempt["committed_smiles"] or "")
        baseline_path = (
            args.baseline_dir / "shards" / f"{prompt.task.value}__{prompt.drug_name}__baseline.json"
        )
        baseline = json.loads(baseline_path.read_text())
        old_samples = [a["committed_smiles"] or "" for a in baseline["attempt_records"]]
        if len(old_samples) != 20:
            raise ValueError("baseline does not contain twenty attempts")
        paths.update([*sample_paths, baseline_path])
        rows.append(
            {
                "task": prompt.task.value,
                "drug": prompt.drug_name,
                "reference": prompt.original_smiles,
                "baseline": evaluate(old_samples, prompt.original_smiles, function),
                "full_program": evaluate(samples, prompt.original_smiles, function),
            }
        )
    result = {
        "schema": "genmol_distance_saved_panel_v1",
        "role": "post-generation development metric only; original drug never enters proposals",
        "definition": "pinned GenMol get_distance: mean Tanimoto distance, Morgan radius2 1024 bits, original drug, distinct valid outputs",
        "rows": rows,
        "macro_means": {
            task: {
                arm: float(
                    np.mean(
                        [
                            r[arm]["distance"]
                            for r in rows
                            if r["task"] == task and r[arm]["distance"] is not None
                        ]
                    )
                )
                for arm in ("baseline", "full_program")
            }
            for task in sorted({r["task"] for r in rows})
        },
        "empty_prompts": [
            f"{r['task']}:{r['drug']}:{a}"
            for r in rows
            for a in ("baseline", "full_program")
            if r[a]["distance"] is None
        ],
        "input_sha256": {str(p.resolve()): sha(p) for p in sorted(paths)},
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed_role": "no generation; consume exact saved seed-zero attempts",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=args.output_dir.parent, prefix=".genmol-distance-"
    ) as temp:
        stage = Path(temp) / "complete"
        stage.mkdir()
        (stage / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        stage.rename(args.output_dir)
    print(
        json.dumps({"macro_means": result["macro_means"], "empty_prompts": result["empty_prompts"]})
    )


if __name__ == "__main__":
    main()
