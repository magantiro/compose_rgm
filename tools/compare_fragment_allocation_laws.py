"""Compare exact old/new structural cell laws, without sampling or scoring molecules."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from run_fragment_attachment_library_pilot import _atomic_json

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import load_genmol_prompts
from compose_v4.benchmark.fragment_program_adapter import ProgramConstraint
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior, JointCompletionSampler
from compose_v4.benchmark.training_attachment_fragments import atom_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--old-census", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    prompt_path = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
    paths = [
        args.catalog,
        args.prior,
        args.old_census,
        prompt_path,
        Path(__file__),
        Path("src/compose_v4/benchmark/joint_completion_prior.py"),
        Path("src/compose_v4/benchmark/fragment_conditioned_sampler.py"),
        Path("src/compose_v4/benchmark/fragment_program_adapter.py"),
        Path("src/compose_v4/benchmark/training_attachment_fragments.py"),
    ]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    prior = json.loads(args.prior.read_text())
    catalog = json.loads(args.catalog.read_text())
    old = json.loads(args.old_census.read_text())
    if (
        prior["catalog_sha256"] != hashes[str(args.catalog.resolve())]
        or old["inputs_sha256"][str(args.catalog.resolve())] != hashes[str(args.catalog.resolve())]
    ):
        raise ValueError("catalog lineage mismatch")
    sampler = JointCompletionSampler(catalog["entries"], JointCompletionPrior.from_dict(prior))
    prompts = {(p.task.value, p.drug_name): p for p in load_genmol_prompts(prompt_path)}
    rows = []
    for row in old["prompts"]:
        context = build_prompt_context(prompts[(row["task"], row["drug"])])
        constraint = ProgramConstraint.from_context(context)
        core = Chem.MolFromSmiles(context.start_smiles)
        keys = tuple(
            atom_context(core.GetAtomWithIdx(site))
            for site, count in constraint.requirements
            for _ in range(count)
        )
        if (
            sorted(keys) != sorted(row["contexts"])
            or context.start_state.n_real_atoms != row["core_atoms"]
            or core.GetRingInfo().NumRings() != row["core_rings"]
        ):
            raise ValueError("old census descriptors differ from actual runtime prompt context")
        table = sampler.plan_table(keys, row["core_atoms"], row["core_rings"])
        old_cells = {
            (c["added_atoms"] + row["core_atoms"], c["added_rings"] + row["core_rings"])
            for c in row["cells"]
        }
        if (
            old_cells != set(table.cells)
            or not np.all(table.probabilities > 0)
            or abs(float(table.probabilities.sum()) - 1) > 1e-12
        ):
            raise ValueError("new structural cell law changes support or has invalid probabilities")
        cells = np.asarray(table.cells)
        rows.append(
            {
                "task": row["task"],
                "drug": row["drug"],
                "core_atoms": row["core_atoms"],
                "core_rings": row["core_rings"],
                "contexts": keys,
                "reachable_cells_old": len(old_cells),
                "reachable_cells_new": len(table.cells),
                "support_equal": True,
                "old": {
                    k: row[k]
                    for k in (
                        "mean_total_atoms",
                        "mean_total_rings",
                        "probability_at_40_atoms",
                        "probability_at_least_38_atoms",
                    )
                },
                "new": {
                    "mean_total_atoms": float(table.probabilities @ cells[:, 0]),
                    "mean_total_rings": float(table.probabilities @ cells[:, 1]),
                    "probability_at_40_atoms": float(table.probabilities[cells[:, 0] == 40].sum()),
                    "probability_at_least_38_atoms": float(
                        table.probabilities[cells[:, 0] >= 38].sum()
                    ),
                },
                "new_cells": [
                    {"atoms": int(a), "rings": int(r), "probability": float(p)}
                    for (a, r), p in zip(table.cells, table.probabilities, strict=True)
                ],
            }
        )
    if hashes != {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:
        raise ValueError("input changed during census")
    output = {
        "schema": "fragment_allocation_law_comparison_v1",
        "role": "pre-compilation structural allocation law comparison; no candidate generation, checkpoint loading, quality scoring or sampler change",
        "inputs_sha256": hashes,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "configuration": {
            "precision": "float64",
            "seed": "none; exact deterministic dynamic programming",
            "device": "cpu",
            "workers": 1,
            "split": prior["split"],
            "training_molecules": prior["training_molecules"],
        },
        "all_runtime_prompt_contexts_match": True,
        "all_cell_supports_identical": True,
        "prompts": rows,
    }
    args.output_dir.mkdir(parents=True)
    _atomic_json(args.output_dir / "comparison.json", output)
    print(json.dumps({"prompts": len(rows), "all_cell_supports_identical": True}))


if __name__ == "__main__":
    main()
