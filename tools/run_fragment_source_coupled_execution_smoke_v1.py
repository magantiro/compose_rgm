"""Zero-quality, model-scored execution smoke for source-coupled decoration.

This small test does not compute benchmark quality, diversity, or uniqueness.
It checks whether the train-only content proposal reaches exact, model-supported,
prompt-faithful completions under the existing eight-offer selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_pendant_programs import sample_pendant_panel
from compose_v4.benchmark.fragment_program_adapter import ProgramConstraint
from compose_v4.benchmark.joint_mass_pendant_policy import JointMassPendantSampler
from compose_v4.benchmark.source_coupled_pendant_policy import SourceCoupledPendantSampler
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path(
    "/Users/rmaganti/compose_rgm_git/.worktrees/fragment-attachment-library-20260924/"
    "diagnostics/fragment_training_pendant_catalog_v1/catalog.json"
)
PRIOR = Path(
    "/Users/rmaganti/compose_rgm_git/.worktrees/fragment-attachment-library-20260924/"
    "diagnostics/fragment_training_decoration_mass_prior_v1/prior.json"
)
CHECKPOINT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")
PROMPTS = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
OUTPUT = ROOT / "diagnostics/fragment_source_coupled_execution_smoke_v1"
MATERIAL_HASHES = {
    str(CATALOG): "fad4dd91495b46d08a66f219c0887f0f8acdaf9f245613e1202904891b9211be",
    str(PRIOR): "288a012905cd461c4f31e18e3c6cd6dbc43248f74b764ccda4877a9fbc6da3bd",
    str(CHECKPOINT): "24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4",
    str(PROMPTS): "a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9",
}
ARMS = ("frozen", "source_coupled")
DRUGS = (
    "BARICITINIB",
    "CYCLOTHIAZIDE",
    "ELIGLUSTAT",
    "ERLOTINIB",
    "FUTIBATINIB",
    "LESINURAD",
    "LIOTHYRONINE",
    "LOVASTATIN",
    "MARIBAVIR",
    "SPIRAPRIL",
)
ATTEMPTS_PER_PROMPT_ARM = 3


def _seed(drug: str, arm: str) -> int:
    digest = hashlib.sha256(f"source-coupled-execution-v1:{drug}:{arm}:7".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def preflight() -> tuple[tuple, dict, dict, str]:
    if OUTPUT.exists():
        raise FileExistsError(f"execution smoke output already exists: {OUTPUT}")
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError("source-coupled smoke stopped at five-GiB disk floor")
    for path, expected in MATERIAL_HASHES.items():
        if physical_sha256(Path(path)) != expected:
            raise ValueError(f"source-coupled smoke input changed: {path}")
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode:
        raise ValueError("source-coupled smoke tracked source is dirty")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False).returncode:
        raise ValueError("source-coupled smoke index is dirty")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prompts = tuple(
        prompt
        for prompt in load_genmol_prompts(PROMPTS)
        if prompt.task is FragmentTask.SCAFFOLD_DECORATION
    )
    if tuple(prompt.drug_name for prompt in prompts) != DRUGS:
        raise ValueError("source-coupled smoke prompt identity changed")
    catalog = json.loads(CATALOG.read_text())
    prior = json.loads(PRIOR.read_text())
    JointMassPendantSampler(catalog, prior)
    SourceCoupledPendantSampler(catalog, prior, coupled_probability=0.5)
    return prompts, catalog, prior, revision


def run() -> dict:
    prompts, catalog, prior, revision = preflight()
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    torch.set_num_threads(1)
    model, _ = load_factorized_rollout_checkpoint(CHECKPOINT)
    samplers = {
        "frozen": JointMassPendantSampler(catalog, prior),
        "source_coupled": SourceCoupledPendantSampler(catalog, prior, coupled_probability=0.5),
    }
    input_hashes = {
        **MATERIAL_HASHES,
        str(Path(__file__)): physical_sha256(Path(__file__)),
        str(ROOT / "src/compose_v4/benchmark/source_coupled_pendant_policy.py"): physical_sha256(
            ROOT / "src/compose_v4/benchmark/source_coupled_pendant_policy.py"
        ),
        str(ROOT / "src/compose_v4/benchmark/fragment_pendant_programs.py"): physical_sha256(
            ROOT / "src/compose_v4/benchmark/fragment_pendant_programs.py"
        ),
    }
    results = []
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUTPUT.parent, prefix=".source-coupled-smoke-") as stage:
        staged = Path(stage) / "complete"
        staged.mkdir()
        attempt_hashes = {}
        for prompt in prompts:
            context = build_prompt_context(prompt)
            constraint = ProgramConstraint.from_context(context)
            protected = constraint.lock(context.start_state)
            for arm in ARMS:
                rng = np.random.default_rng(_seed(prompt.drug_name, arm))
                counts = {
                    "attempts": 0,
                    "outputs": 0,
                    "exact_compiled_offers": 0,
                    "model_supported_offers": 0,
                    "coupled_plans": 0,
                    "plans_with_shared_training_row": 0,
                    "invalid_outputs": 0,
                    "nonfaithful_outputs": 0,
                }
                for index in range(ATTEMPTS_PER_PROMPT_ARM):
                    panel = sample_pendant_panel(context, samplers[arm], model, rng)
                    selected = panel.selected
                    valid = bool(
                        selected
                        and is_valid_state(selected.endpoint)
                        and is_connected_or_null(selected.endpoint)
                    )
                    faithful = bool(
                        selected
                        and constraint.complete(selected.endpoint)
                        and all(
                            protected.permits(decode_state(state))
                            for state in selected.trace["states"]
                        )
                    )
                    counts["attempts"] += 1
                    counts["outputs"] += selected is not None
                    counts["invalid_outputs"] += bool(selected and not valid)
                    counts["nonfaithful_outputs"] += bool(selected and not faithful)
                    counts["exact_compiled_offers"] += panel.receipt["exact_compiled_count"]
                    counts["model_supported_offers"] += panel.receipt["model_supported_count"]
                    for offer in panel.receipt["offered"]:
                        plan = offer.get("provenance", {}).get("pendant_plan")
                        if plan is None:
                            continue
                        counts["coupled_plans"] += bool(plan.get("source_coupling_used"))
                        counts["plans_with_shared_training_row"] += bool(
                            plan.get("shared_training_rows", 0)
                        )
                    relative = Path("attempts") / arm / f"{prompt.drug_name}_{index:03d}.json"
                    path = staged / relative
                    _json(
                        path,
                        {
                            "arm": arm,
                            "drug": prompt.drug_name,
                            "attempt_index": index,
                            "selected_valid": valid,
                            "selected_prompt_fidelity": faithful,
                            "panel": panel.receipt,
                        },
                    )
                    attempt_hashes[str(relative)] = physical_sha256(path)
                results.append({"drug": prompt.drug_name, "arm": arm, **counts})
                print(json.dumps({"completed": f"{arm}/{prompt.drug_name}", **counts}), flush=True)
        if len(results) != 20 or any(row["attempts"] != 3 for row in results):
            raise ValueError("source-coupled smoke did not complete all cells")
        result = {
            "schema": "fragment_source_coupled_execution_smoke_v1",
            "role": "zero-quality model-scored execution smoke; not benchmark evidence",
            "code_revision": revision,
            "input_sha256": dict(sorted(input_hashes.items())),
            "attempt_sha256": dict(sorted(attempt_hashes.items())),
            "versions": {
                "python": platform.python_version(),
                "rdkit": rdBase.rdkitVersion,
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
            "device": "cpu",
            "precision": "float32 model, float64 proposal probabilities",
            "configuration": {
                "task": "scaffold_decoration",
                "arms": list(ARMS),
                "drugs": list(DRUGS),
                "attempts_per_prompt_arm": ATTEMPTS_PER_PROMPT_ARM,
                "offers_per_attempt": 8,
                "source_coupling_probability": 0.5,
                "seed_derivation": "first 64 SHA-256 bits of source-coupled-execution-v1:drug:arm:7",
                "quality_oracle_calls": 0,
                "scored_benchmark_metrics": False,
            },
            "rows": results,
        }
        _json(staged / "result.json", result)
        os.replace(staged, OUTPUT)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = run()
    print(json.dumps({"completed": len(result["rows"]), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
