"""Matched zero-quality exact-execution smoke for conditional pendant content."""

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

from compose_v4.benchmark.conditional_content_pendant_policy import (
    ConditionalContentPendantSampler,
)
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
CATALOG = ROOT / "diagnostics/fragment_training_pendant_catalog_v1/catalog.json"
MASS = ROOT / "diagnostics/fragment_training_decoration_mass_prior_v1/prior.json"
CONTENT = ROOT / "diagnostics/fragment_training_decoration_content_context_v1/prior.json"
SCHEMA = "conditional_content_decoration_execution_smoke_v1"
DRAWS = 3


def seed(drug: str, index: int) -> int:
    payload = f"{SCHEMA}|{drug}|{index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def run() -> dict:
    catalog = json.loads(CATALOG.read_text())
    mass = json.loads(MASS.read_text())
    content = json.loads(CONTENT.read_text())
    samplers = {
        "frozen_joint_mass": JointMassPendantSampler(catalog, mass),
        "conditional_content": ConditionalContentPendantSampler(
            catalog, mass, content, catalog_sha256=physical_sha256(CATALOG)
        ),
    }
    prompts = [
        prompt
        for prompt in load_genmol_prompts(PROMPTS)
        if prompt.task is FragmentTask.SCAFFOLD_DECORATION
    ]
    if len(prompts) != 10:
        raise ValueError("exact-execution smoke requires ten decoration prompts")
    rows = []
    for prompt in prompts:
        context = build_prompt_context(prompt)
        constraint = ProgramConstraint.from_context(context)
        lock = constraint.lock(context.start_state)
        for arm, sampler in samplers.items():
            attempts = []
            for index in range(DRAWS):
                try:
                    candidate = propose_pendant_decoration(
                        context,
                        sampler,
                        np.random.default_rng(seed(prompt.drug_name, index)),
                    )
                except ValueError as error:
                    attempts.append(
                        {"draw": index, "status": "exact_compiler_abstention", "reason": str(error)}
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
                        f"invalid or nonfaithful exact decoration endpoint: {arm}/{prompt.drug_name}/{index}"
                    )
                attempts.append(
                    {
                        "draw": index,
                        "status": "valid_exact_prompt_faithful",
                        "smiles": candidate.smiles,
                        "trace": candidate.trace,
                        "provenance": candidate.provenance,
                    }
                )
            rows.append({"drug": prompt.drug_name, "arm": arm, "attempts": attempts})
    inputs = (
        PROMPTS,
        CATALOG,
        MASS,
        CONTENT,
        ROOT / "src/compose_v4/benchmark/conditional_content_pendant_policy.py",
        ROOT / "src/compose_v4/benchmark/joint_mass_pendant_policy.py",
        ROOT / "src/compose_v4/benchmark/fragment_pendant_programs.py",
        ROOT / "src/compose_v4/benchmark/fragment_program_adapter.py",
        Path(__file__),
    )
    return {
        "schema": SCHEMA,
        "role": "matched zero-quality exact-execution support gate",
        "draws_per_prompt_and_arm": DRAWS,
        "input_sha256": {str(path): physical_sha256(path) for path in inputs},
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "rows": rows,
        "status_counts": {
            arm: dict(
                sorted(
                    Counter(
                        attempt["status"]
                        for row in rows
                        if row["arm"] == arm
                        for attempt in row["attempts"]
                    ).items()
                )
            )
            for arm in samplers
        },
        "qed_sa_evaluations": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = run()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".conditional-content-execution-"
    ) as stage:
        temporary = Path(stage) / "result.json"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.rename(output)
    print(json.dumps(result["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
