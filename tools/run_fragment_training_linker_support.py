"""Prepare/run the bounded 20-attempt, no-quality linker panel support gate.

The proposal and scorer are the existing stochastic linker panel without a
scientific-policy override. A whole eight-draw attempt is the recovery unit.
Completed units are reused; an interrupted started unit fails closed.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from audit_fragment_training_linker import CATALOG_SHA, CHECKPOINT_SHA, PROMPT_SHA, ROOT
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
from rdkit import rdBase
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_constrained_suite import prompt_rng_seed

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_linker_assembly import linker_fidelity
from compose_v4.benchmark.fragment_linker_sampler import load_linker_catalog, sample_linker_panel
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state


def support_summary(rows):
    """Frozen success rule; every attempted output remains in the denominator."""
    identities = [(row["drug"], row["attempt_index"]) for row in rows]
    drugs = sorted({name for name, _ in identities})
    if (
        len(rows) != 20
        or len(set(identities)) != 20
        or len(drugs) != 10
        or set(identities) != {(name, i) for name in drugs for i in range(2)}
    ):
        raise ValueError("support gate requires exactly two attempts for all ten prompts")
    if any(
        row["panel"]["offered_count"] != 8
        or [record["draw"] for record in row["panel"]["offered"]] != list(range(8))
        for row in rows
    ):
        raise ValueError("support gate requires eight recorded candidate draws per attempt")
    selected = [row for row in rows if row["panel"]["output_count"]]
    if any(row["panel"]["output_count"] not in (0, 1) for row in rows):
        raise ValueError("an attempt can emit at most one completed endpoint")
    per_prompt = {drug: sum(row["drug"] == drug for row in selected) for drug in drugs}
    statuses = Counter(record["status"] for row in rows for record in row["panel"]["offered"])
    valid = sum(row["selected_valid_connected"] for row in selected)
    fidelity = sum(row["selected_exact_core_path_fidelity"] for row in selected)
    checks = {
        "output_at_least_90_percent": len(selected) >= 18,
        "every_prompt_emits": all(per_prompt.values()),
        "all_selected_chemically_valid_connected": valid == len(selected),
        "all_selected_exact_core_path_fidelity": fidelity == len(selected),
    }
    return {
        "attempts": len(rows),
        "outputs": len(selected),
        "output_coverage": len(selected) / len(rows),
        "connected_validity_precision": valid / len(selected) if selected else None,
        "exact_core_path_fidelity_precision": fidelity / len(selected) if selected else None,
        "unique_selected_endpoints": len({row["panel"]["selected_smiles"] for row in selected}),
        "per_prompt_outputs": per_prompt,
        "offered_candidate_draws": 160,
        "candidate_status_counts": dict(sorted(statuses.items())),
        "exact_compiled_candidate_draws": sum(row["panel"]["exact_compiled_count"] for row in rows),
        "model_supported_candidate_draws": sum(
            row["panel"]["model_supported_count"] for row in rows
        ),
        "checks": checks,
        "support_pass": all(checks.values()),
        "quality_evaluations": 0,
        "oracle_calls": 0,
        "official_evaluation_status": "not_run",
    }


def restore_completed_attempt(row, *, drug, attempt_index, rng):
    if (
        row["drug"] != drug
        or row["attempt_index"] != attempt_index
        or row["panel"]["rng_state_before"] != rng.bit_generator.state
        or row["panel"]["offered_count"] != 8
        or [record["draw"] for record in row["panel"]["offered"]] != list(range(8))
    ):
        raise ValueError("completed support attempt identity/RNG/draw chain changed")
    rng.bit_generator.state = row["panel"]["rng_state_after"]


def manifest_for(checkpoint):
    catalog_path = ROOT / "diagnostics/fragment_training_region_catalog_v1/catalog.json"
    prior_path = ROOT / "diagnostics/fragment_joint_completion_prior_v1/prior.json"
    prompt_path = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    for path, expected in (
        (catalog_path, CATALOG_SHA),
        (checkpoint, CHECKPOINT_SHA),
        (prompt_path, PROMPT_SHA),
    ):
        if physical_sha256(path) != expected:
            raise ValueError(f"frozen support input changed: {path}")
    prior_data = json.loads(prior_path.read_text())
    if prior_data["catalog_sha256"] != CATALOG_SHA:
        raise ValueError("joint prior and connector catalog training identities disagree")
    JointCompletionPrior.from_dict(prior_data)
    prompts = tuple(
        p for p in load_genmol_prompts(prompt_path) if p.task is FragmentTask.LINKER_DESIGN
    )
    if len(prompts) != 10 or len({p.drug_name for p in prompts}) != 10:
        raise ValueError("frozen linker prompt census changed")
    paths = {
        catalog_path,
        prior_path,
        prompt_path,
        checkpoint,
        Path(__file__).resolve(),
        ROOT / "scripts/evaluate_tracelet_rollouts.py",
        ROOT / "tools/audit_fragment_training_linker.py",
        ROOT / "tools/run_fragment_constrained_suite.py",
        ROOT / "tools/run_fragment_attachment_library_pilot.py",
        ROOT / "docs/FRAGMENT_LINKER_ASSEMBLY_GATE_2026-09-24.md",
    }
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("compose_v4") and filename and filename.endswith(".py"):
            paths.add(Path(filename).resolve())
    manifest = {
        "schema": "fragment_training_linker_panel_support_v1",
        "inputs": {str(path.resolve()): physical_sha256(path) for path in sorted(paths)},
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "configuration": {
            "prompts": [p.drug_name for p in prompts],
            "task": FragmentTask.LINKER_DESIGN.value,
            "attempts_per_prompt": 2,
            "candidate_draws_per_attempt": 8,
            "seed": 0,
            "seed_derivation": "existing prompt_rng_seed(drug, task, 0), sequential two attempts",
            "model_score": "shared learned_program_scores mean native log mark at time zero",
            "selection": "shared select_learned_program unit-temperature canonical-panel softmax",
            "max_active_atoms": 40,
            "slots": 48,
            "max_primitives": 32,
            "max_blocks": 8,
            "workers": 1,
            "threads": 1,
            "device": "cpu",
            "recovery_unit": "one complete eight-draw attempt; interrupted starts fail closed",
        },
        "gate": {
            "minimum_outputs": 18,
            "attempts": 20,
            "every_prompt_emits": True,
            "selected_chemical_connected_precision": 1.0,
            "selected_exact_core_path_fidelity_precision": 1.0,
            "quality_metric": None,
        },
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "role": "zero-oracle development support; not an official linker/morphing row",
        "oracle_calls": 0,
        "quality_evaluations": 0,
    }
    return manifest, prompts, catalog_path, prior_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest, prompts, catalog_path, prior_data = manifest_for(args.checkpoint)
    manifest_path = args.output_dir / "manifest.json"
    if args.phase == "prepare":
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        _atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {"status": "prepared_only", "manifest_sha256": physical_sha256(manifest_path)}
            )
        )
        return
    if not manifest_path.exists():
        raise ValueError("prepare the frozen support manifest before running")
    prepared = json.loads(manifest_path.read_text())
    # A later commit with byte-identical dependencies does not alter science.
    if {k: v for k, v in prepared.items() if k != "code_revision"} != {
        k: v for k, v in manifest.items() if k != "code_revision"
    }:
        raise ValueError("prepared support identity changed before run/resume")
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("completed support summary already exists; do not rerun")
    catalog = load_linker_catalog(catalog_path, expected_sha256=CATALOG_SHA)
    prior = JointCompletionPrior.from_dict(prior_data)
    model = None
    rows, hashes = [], {}
    for prompt in prompts:
        rng = np.random.default_rng(prompt_rng_seed(prompt.drug_name, prompt.task.value, 0))
        for index in range(2):
            name = f"{prompt.drug_name}_{index:03d}"
            path = args.output_dir / "attempts" / f"{name}.json"
            start_path = args.output_dir / "starts" / f"{name}.json"
            if path.exists():
                row = json.loads(path.read_text())
                restore_completed_attempt(row, drug=prompt.drug_name, attempt_index=index, rng=rng)
            else:
                if start_path.exists():
                    raise ValueError(f"interrupted attempt {name}; no automatic regeneration/retry")
                if model is None:
                    model, _ = load_factorized_rollout_checkpoint(args.checkpoint)
                _atomic_json(
                    start_path,
                    {
                        "drug": prompt.drug_name,
                        "attempt_index": index,
                        "rng_state_before": rng.bit_generator.state,
                        "manifest_sha256": physical_sha256(manifest_path),
                    },
                )
                started = time.monotonic()
                result = sample_linker_panel(prompt, catalog, prior, model, rng)
                selected = result.selected
                fidelity = linker_fidelity(prompt, selected.smiles) if selected else None
                row = {
                    "drug": prompt.drug_name,
                    "attempt_index": index,
                    "panel": result.receipt,
                    "selected_valid_connected": bool(
                        selected
                        and is_valid_state(selected.endpoint)
                        and is_connected_or_null(selected.endpoint)
                    ),
                    "selected_exact_core_path_fidelity": bool(
                        selected
                        and fidelity["satisfied"]
                        and selected.provenance["exact_mapped_core_identity_checked"]
                        and selected.provenance["source_core_locked_all_states"]
                    ),
                    "selected_fidelity": fidelity,
                    "wall_seconds": time.monotonic() - started,
                }
                _atomic_json(path, row)
            rows.append(row)
            hashes[str(path.relative_to(args.output_dir))] = physical_sha256(path)
            progress = {
                "completed_attempts": len(rows),
                "total_attempts": 20,
                "outputs": sum(row["panel"]["output_count"] for row in rows),
                "last_prompt": prompt.drug_name,
            }
            _atomic_json(args.output_dir / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    if any(
        physical_sha256(Path(path)) != expected for path, expected in prepared["inputs"].items()
    ):
        raise ValueError("frozen support dependency changed during execution")
    summary = {
        "schema": manifest["schema"],
        "manifest_sha256": physical_sha256(manifest_path),
        "execution_code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "attempt_hashes": dict(sorted(hashes.items())),
        "candidate_and_selection_wall_seconds": sum(row["wall_seconds"] for row in rows),
        **support_summary(rows),
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
