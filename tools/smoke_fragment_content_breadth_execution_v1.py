"""Matched, zero-quality exact-execution smoke for decoration content breadth."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_pendant_programs import propose_pendant_decoration
from compose_v4.benchmark.fragment_program_adapter import ProgramConstraint
from compose_v4.benchmark.joint_mass_pendant_policy import JointMassPendantSampler
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
SCHEMA = "fragment_content_breadth_execution_smoke_v1"
PROMPT_SHA = "a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9"
CATALOG_SHA = "fad4dd91495b46d08a66f219c0887f0f8acdaf9f245613e1202904891b9211be"
MASS_SHA = "288a012905cd461c4f31e18e3c6cd6dbc43248f74b764ccda4877a9fbc6da3bd"
ARMS = ("frozen", "uniform_within_cell")
DRAWS = 3


def seed(drug: str, index: int) -> int:
    digest = hashlib.sha256(f"{SCHEMA}:{drug}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def run(catalog_path: Path, mass_path: Path) -> dict:
    materials = ((PROMPTS, PROMPT_SHA), (catalog_path, CATALOG_SHA), (mass_path, MASS_SHA))
    for path, expected in materials:
        if physical_sha256(path) != expected:
            raise ValueError(f"decoration smoke input hash mismatch: {path}")
    catalog = json.loads(catalog_path.read_text())
    mass = json.loads(mass_path.read_text())
    samplers = {
        "frozen": JointMassPendantSampler(catalog, mass),
        "uniform_within_cell": JointMassPendantSampler(
            catalog, mass, content_allocation="uniform_within_cell"
        ),
    }
    prompts = tuple(
        prompt
        for prompt in load_genmol_prompts(PROMPTS)
        if prompt.task is FragmentTask.SCAFFOLD_DECORATION
    )
    if len(prompts) != 10:
        raise ValueError("content-breadth execution smoke requires ten decoration prompts")
    rows = []
    for prompt in prompts:
        context = build_prompt_context(prompt)
        constraint = ProgramConstraint.from_context(context)
        lock = constraint.lock(context.start_state)
        for arm, sampler in samplers.items():
            for index in range(DRAWS):
                try:
                    candidate = propose_pendant_decoration(
                        context, sampler, np.random.default_rng(seed(prompt.drug_name, index))
                    )
                except ValueError as error:
                    rows.append(
                        {
                            "drug": prompt.drug_name,
                            "arm": arm,
                            "draw": index,
                            "status": "exact_compiler_abstention",
                            "reason": str(error),
                        }
                    )
                    continue
                if not (
                    is_valid_state(candidate.endpoint)
                    and is_connected_or_null(candidate.endpoint)
                    and constraint.complete(candidate.endpoint)
                    and all(
                        lock.permits(decode_state(state)) for state in candidate.trace["states"]
                    )
                ):
                    raise RuntimeError(
                        f"invalid/nonfaithful decoration endpoint: {arm}/{prompt.drug_name}/{index}"
                    )
                rows.append(
                    {
                        "drug": prompt.drug_name,
                        "arm": arm,
                        "draw": index,
                        "status": "valid_exact_prompt_faithful",
                        "smiles": candidate.smiles,
                        "trace": candidate.trace,
                        "provenance": candidate.provenance,
                    }
                )
    inputs = (
        *tuple(path for path, _ in materials),
        ROOT / "src/compose_v4/benchmark/pendant_completion_policy.py",
        ROOT / "src/compose_v4/benchmark/joint_mass_pendant_policy.py",
        ROOT / "src/compose_v4/benchmark/fragment_pendant_programs.py",
        ROOT / "src/compose_v4/benchmark/fragment_program_adapter.py",
        Path(__file__),
    )
    return {
        "schema": SCHEMA,
        "role": "matched zero-quality exact-execution support gate",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_sha256": {str(path.resolve()): physical_sha256(path) for path in inputs},
        "training_split": catalog["split"],
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "device": "cpu",
        "precision": "float64 proposal probabilities",
        "workers": 1,
        "seed_derivation": "first 64 bits of SHA-256(schema:drug:draw)",
        "draws_per_prompt_and_arm": DRAWS,
        "rows": rows,
        "status_counts": {
            arm: dict(sorted(Counter(row["status"] for row in rows if row["arm"] == arm).items()))
            for arm in ARMS
        },
        "qed_sa_evaluations": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--mass-prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(
        subprocess.run(command, cwd=ROOT, check=False).returncode
        for command in (("git", "diff", "--quiet"), ("git", "diff", "--cached", "--quiet"))
    ):
        raise ValueError("decoration execution smoke requires clean tracked source")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = run(args.catalog, args.mass_prior)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".content-breadth-smoke-") as stage:
        temporary = Path(stage) / "result.json"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.rename(output)
    print(json.dumps(result["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
