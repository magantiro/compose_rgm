"""One seeded linker draw per prompt; at most two native-model support checks.

This diagnostic is not an eight-draw panel, quality pilot or official task row.
The first compiled ring-free and first compiled ring-containing connectors in
frozen prompt order are the only allowed model examples. Missing types abstain.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
from rdkit import rdBase
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_constrained_suite import prompt_rng_seed

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_linker_sampler import (
    LinkerProposalAbstention,
    compatible_linker_cells,
    load_linker_catalog,
    propose_linker_completion,
)
from compose_v4.benchmark.fragment_program_adapter import learned_program_scores
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior
from compose_v4.benchmark.training_attachment_fragments import physical_sha256

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA = "24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4"
CATALOG_SHA = "a8775b466f479c2594ff5d8afd81049a57969211d3152035c93e9db51e8b1347"
PROMPT_SHA = "a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.monotonic()
    torch.set_num_threads(1)
    catalog_path = ROOT / "diagnostics/fragment_training_region_catalog_v1/catalog.json"
    prior_path = ROOT / "diagnostics/fragment_joint_completion_prior_v1/prior.json"
    prompt_path = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    for path, expected in ((args.checkpoint, CHECKPOINT_SHA), (prompt_path, PROMPT_SHA)):
        if physical_sha256(path) != expected:
            raise ValueError(f"frozen diagnostic input changed: {path}")
    catalog = load_linker_catalog(catalog_path, expected_sha256=CATALOG_SHA)
    prior_data = json.loads(prior_path.read_text())
    if prior_data["catalog_sha256"] != CATALOG_SHA:
        raise ValueError("joint prior was not fitted from this frozen catalog")
    prior = JointCompletionPrior.from_dict(prior_data)
    material = {catalog_path, prior_path, prompt_path, args.checkpoint, Path(__file__).resolve()}
    material.add(ROOT / "scripts/evaluate_tracelet_rollouts.py")
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("compose_v4") and path and path.endswith(".py"):
            material.add(Path(path).resolve())
    identities = {str(path.resolve()): physical_sha256(path) for path in sorted(material)}
    manifest = {
        "schema": "fragment_training_linker_smoke_v1",
        "role": "structural proposal/compiler/model support; no quality or panel success claim",
        "inputs": identities,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "configuration": {
            "seed": 0,
            "seed_derivation": "existing prompt_rng_seed(drug, linker_design, 0)",
            "prompts": 10,
            "candidate_draws_per_prompt": 1,
            "maximum_model_scored_programs": 2,
            "model_selection": "first compiled ring-free/with-ring connector in frozen prompt order",
            "native_score_batch_size": 16,
            "max_active_atoms": 40,
            "slots": 48,
            "max_primitives": 32,
            "max_blocks": 8,
            "workers": 1,
            "threads": 1,
        },
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": "cpu",
            "precision": "checkpoint native float32",
        },
        "oracle_calls": 0,
        "quality_evaluations": 0,
    }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    prompts = tuple(
        p for p in load_genmol_prompts(prompt_path) if p.task is FragmentTask.LINKER_DESIGN
    )
    if len(prompts) != 10:
        raise ValueError("frozen linker prompt census changed")
    rows, examples = [], {}
    for prompt in prompts:
        rng_seed = prompt_rng_seed(prompt.drug_name, prompt.task.value, 0)
        rng = np.random.default_rng(rng_seed)
        grouped, census = compatible_linker_cells(prompt, catalog)
        cells = tuple(grouped)
        probabilities = prior.cell_probabilities(cells) if cells else ()
        row = {
            "drug": prompt.drug_name,
            "fragments": prompt.fragments,
            "seed": rng_seed,
            "census": census,
            "structural_cell_probabilities": [
                {"cell": list(cell), "probability": float(probability)}
                for cell, probability in zip(cells, probabilities, strict=True)
            ],
        }
        candidate_start = time.monotonic()
        try:
            candidate = propose_linker_completion(prompt, catalog, prior, rng)
        except ValueError as error:
            row.update(status="abstained", reason=str(error))
            if isinstance(error, LinkerProposalAbstention):
                row["proposal_receipt"] = error.receipt
        else:
            row.update(
                status="exact_compiled",
                endpoint=candidate.smiles,
                provenance=candidate.provenance,
                trace=candidate.trace,
            )
            proposal = candidate.provenance["training_connector"]
            added_rings = candidate.provenance["final_cell"][1] - proposal["core_ring_count"]
            example_type = "ring_containing_connector" if added_rings else "ring_free_connector"
            if example_type not in examples:
                examples[example_type] = (candidate, row)
        row["compile_wall_seconds"] = time.monotonic() - candidate_start
        row["rng_state_after"] = rng.bit_generator.state
        rows.append(row)
        _atomic_json(args.output_dir / "progress.json", {"rows": rows, "model_calls": 0})
    checks = []
    if examples:
        model, _ = load_factorized_rollout_checkpoint(args.checkpoint)
        for kind, (candidate, row) in examples.items():
            score_start = time.monotonic()
            check = {
                "connector_type": kind,
                "drug": row["drug"],
                "primitives": len(candidate.trace["actions"]),
            }
            try:
                [score] = learned_program_scores(model, (candidate,))
            except (ValueError, KeyError) as error:
                check.update(status="model_abstained", reason=str(error))
            else:
                if np.isfinite(score):
                    check.update(status="model_supported", mean_native_log_mark=float(score))
                else:
                    check.update(status="model_abstained", reason="nonfinite_native_probability")
            check["wall_seconds"] = time.monotonic() - score_start
            checks.append(check)
            _atomic_json(args.output_dir / "progress.json", {"rows": rows, "model_checks": checks})
    if any(physical_sha256(Path(path)) != expected for path, expected in identities.items()):
        raise RuntimeError("material diagnostic inputs changed during execution")
    complete = sum(row["status"] == "exact_compiled" for row in rows)
    result = {
        "schema": manifest["schema"],
        "manifest_sha256": physical_sha256(args.output_dir / "manifest.json"),
        "attempts": len(rows),
        "exact_compiled": complete,
        "exact_compiler_coverage": complete / len(rows),
        "exact_execution_precision": 1.0 if complete else None,
        "two_boundary_catalog_entries": catalog.two_boundary_entries,
        "rows": rows,
        "model_checks": checks,
        "model_scored_programs": len(checks),
        "model_supported_programs": sum(c["status"] == "model_supported" for c in checks),
        "missing_model_example_types": sorted(
            {"ring_free_connector", "ring_containing_connector"} - set(examples)
        ),
        "official_evaluation_status": "not_run",
        "oracle_calls": 0,
        "quality_evaluations": 0,
        "wall_seconds": time.monotonic() - started,
        "peak_process_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    }
    _atomic_json(args.output_dir / "result.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("rows",)}))


if __name__ == "__main__":
    main()
