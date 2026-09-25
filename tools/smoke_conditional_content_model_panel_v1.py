"""Zero-quality model-panel support gate for conditional decoration content."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase
from run_fragment_constrained_suite import prompt_rng_seed

from compose_v4.benchmark.conditional_content_pendant_policy import (
    ConditionalContentPendantSampler,
)
from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_pendant_programs import sample_pendant_panel
from compose_v4.benchmark.fragment_program_adapter import ProgramConstraint
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
CATALOG = ROOT / "diagnostics/fragment_training_pendant_catalog_v1/catalog.json"
MASS = ROOT / "diagnostics/fragment_training_decoration_mass_prior_v1/prior.json"
CONTENT = ROOT / "diagnostics/fragment_training_decoration_content_context_v1/prior.json"
CHECKPOINT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")
SCHEMA = "conditional_content_model_panel_smoke_v1"
ATTEMPTS_PER_PROMPT = 2


def run() -> dict:
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    torch.set_num_threads(1)
    catalog = json.loads(CATALOG.read_text())
    mass = json.loads(MASS.read_text())
    content = json.loads(CONTENT.read_text())
    sampler = ConditionalContentPendantSampler(
        catalog, mass, content, catalog_sha256=physical_sha256(CATALOG)
    )
    model, _ = load_factorized_rollout_checkpoint(CHECKPOINT)
    prompts = [
        prompt
        for prompt in load_genmol_prompts(PROMPTS)
        if prompt.task is FragmentTask.SCAFFOLD_DECORATION
    ]
    if len(prompts) != 10:
        raise ValueError("model support smoke requires all ten decoration prompts")
    rows = []
    for prompt in prompts:
        context = build_prompt_context(prompt)
        constraint = ProgramConstraint.from_context(context)
        lock = constraint.lock(context.start_state)
        rng = np.random.default_rng(prompt_rng_seed(prompt.drug_name, prompt.task.value, 3))
        attempts = []
        for index in range(ATTEMPTS_PER_PROMPT):
            panel = sample_pendant_panel(context, sampler, model, rng)
            selected = panel.selected
            valid = bool(
                selected
                and is_valid_state(selected.endpoint)
                and is_connected_or_null(selected.endpoint)
            )
            faithful = bool(
                selected
                and constraint.complete(selected.endpoint)
                and all(lock.permits(decode_state(state)) for state in selected.trace["states"])
            )
            if selected and not (valid and faithful):
                raise RuntimeError(
                    f"model-selected conditional decoration violates constraints: {prompt.drug_name}/{index}"
                )
            attempts.append(
                {
                    "attempt_index": index,
                    "panel": panel.receipt,
                    "selected_valid_connected": valid,
                    "selected_constraint_fidelity": faithful,
                }
            )
        rows.append({"drug": prompt.drug_name, "attempts": attempts})
    inputs = (
        PROMPTS,
        CATALOG,
        MASS,
        CONTENT,
        CHECKPOINT,
        ROOT / "src/compose_v4/benchmark/conditional_content_pendant_policy.py",
        ROOT / "src/compose_v4/benchmark/fragment_pendant_programs.py",
        Path(__file__),
    )
    statuses = Counter(
        offer["status"]
        for row in rows
        for attempt in row["attempts"]
        for offer in attempt["panel"]["offered"]
    )
    return {
        "schema": SCHEMA,
        "role": "zero-quality model support gate; not benchmark scores",
        "input_sha256": {str(path): physical_sha256(path) for path in inputs},
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "rows": rows,
        "candidate_status": dict(sorted(statuses.items())),
        "attempts": 20,
        "outputs": sum(item["panel"]["output_count"] for row in rows for item in row["attempts"]),
        "quality_evaluations": 0,
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
        dir=output.parent, prefix=".conditional-content-model-"
    ) as stage:
        temporary = Path(stage) / "result.json"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.rename(output)
    print(
        json.dumps(
            {
                "attempts": result["attempts"],
                "outputs": result["outputs"],
                "candidate_status": result["candidate_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
