"""Compare unscored decoration plans under frozen and conditional content laws."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

from compose_v4.benchmark.conditional_content_pendant_policy import (
    ConditionalContentPendantSampler,
)
from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_program_adapter import ProgramConstraint
from compose_v4.benchmark.joint_mass_pendant_policy import JointMassPendantSampler
from compose_v4.benchmark.training_attachment_fragments import atom_context, physical_sha256

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "diagnostics/fragment_training_pendant_catalog_v1/catalog.json"
MASS = ROOT / "diagnostics/fragment_training_decoration_mass_prior_v1/prior.json"
CONTENT = ROOT / "diagnostics/fragment_training_decoration_content_context_v1/prior.json"
PROMPTS = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
PLANS_PER_PROMPT = 200
SEED = 20260924


def sample_prompt(context, sampler, *, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    constraint = ProgramConstraint.from_context(context)
    core = Chem.MolFromSmiles(context.start_smiles)
    if core is None:
        raise ValueError(f"invalid decoration core: {context.prompt.drug_name}")
    slots = [site for site, count in constraint.requirements for _ in range(count)]
    plan_keys, masses, ring_counts, conditional_mass = [], [], [], []
    for _ in range(PLANS_PER_PROMPT):
        shuffled = [slots[int(index)] for index in rng.permutation(len(slots))]
        contexts = tuple(atom_context(core.GetAtomWithIdx(site)) for site in shuffled)
        entries, receipt = sampler.sample(contexts, 40 - context.start_state.n_real_atoms, rng)
        if len(entries) != len(shuffled) or receipt["planned_heavy_atoms"] > 40:
            raise ValueError("conditional content changed exact interface or atom support")
        plan_keys.append(
            tuple(
                sorted(
                    (site, entry["rooted_smiles"])
                    for site, entry in zip(shuffled, entries, strict=True)
                )
            )
        )
        masses.append(receipt["planned_decoration_mass"])
        ring_counts.append(sum(entry["ring_count"] for entry in entries))
        conditional_mass.extend(
            draw.get("conditional_training_mass_in_group", 0.0) for draw in receipt["draws"]
        )
    return {
        "plans": PLANS_PER_PROMPT,
        "unique_site_bound_content_plans": len(set(plan_keys)),
        "mean_added_atoms": float(np.mean(masses)),
        "mean_pendant_ring_count": float(np.mean(ring_counts)),
        "ring_containing_plan_fraction": float(np.mean(np.asarray(ring_counts) > 0)),
        "conditional_group_draws_with_train_mass": sum(value > 0 for value in conditional_mass),
        "total_group_draws": len(conditional_mass),
    }


def analyze() -> dict:
    catalog = json.loads(CATALOG.read_text())
    mass = json.loads(MASS.read_text())
    content = json.loads(CONTENT.read_text())
    baseline = JointMassPendantSampler(catalog, mass)
    conditional = ConditionalContentPendantSampler(
        catalog, mass, content, catalog_sha256=physical_sha256(CATALOG)
    )
    prompts = [
        prompt
        for prompt in load_genmol_prompts(PROMPTS)
        if prompt.task is FragmentTask.SCAFFOLD_DECORATION
    ]
    if len(prompts) != 10:
        raise ValueError("expected ten released decoration prompts")
    rows = []
    for index, prompt in enumerate(prompts):
        context = build_prompt_context(prompt)
        rows.append(
            {
                "drug": prompt.drug_name,
                "baseline": sample_prompt(context, baseline, seed=SEED + index),
                "conditional": sample_prompt(context, conditional, seed=SEED + index),
            }
        )
    script = Path(__file__)
    return {
        "schema": "conditional_content_pendant_support_v1",
        "role": "zero-oracle plan support; no molecular quality or validity claim",
        "plans_per_prompt_and_arm": PLANS_PER_PROMPT,
        "seed": SEED,
        "input_sha256": {
            str(path): physical_sha256(path)
            for path in (
                CATALOG,
                MASS,
                CONTENT,
                PROMPTS,
                script,
                ROOT / "src/compose_v4/benchmark/conditional_content_pendant_policy.py",
            )
        },
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "prompts": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = analyze()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".conditional-content-support-"
    ) as stage:
        temporary = Path(stage) / "complete"
        temporary.mkdir()
        (temporary / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.rename(output)
    print(json.dumps({"prompts": len(result["prompts"]), "output": str(output)}))


if __name__ == "__main__":
    main()
