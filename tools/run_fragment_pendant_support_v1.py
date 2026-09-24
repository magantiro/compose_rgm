"""Freeze and run the no-quality all-prompt pendant-decoration support gate."""

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
from rdkit import rdBase
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_constrained_suite import prompt_rng_seed
from run_fragment_training_linker_support import restore_completed_attempt

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_pendant_programs import sample_pendant_panel
from compose_v4.benchmark.fragment_program_adapter import (
    ProgramConstraint,
    verify_prompt_endpoint,
)
from compose_v4.benchmark.pendant_completion_policy import PendantCompletionSampler
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/fragment_pendant_support_v1.json"
CATALOG = ROOT / "diagnostics/fragment_training_pendant_catalog_v1/catalog.json"
PROMPTS = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
CHECKPOINT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")


def support_summary(rows: list[dict], contract: dict) -> dict:
    prompts = contract["prompts"]
    identities = {(row["drug"], row["attempt_index"]) for row in rows}
    if (
        len(prompts) != 10
        or len(set(prompts)) != 10
        or identities != {(drug, index) for drug in prompts for index in range(2)}
        or len(rows) != 20
    ):
        raise ValueError("pendant support requires two attempts for each of ten prompts")
    offered = [entry for row in rows for entry in row["panel"]["offered"]]
    if len(offered) != 160 or any(
        row["panel"]["offered_count"] != 8
        or [entry["draw"] for entry in row["panel"]["offered"]] != list(range(8))
        for row in rows
    ):
        raise ValueError("pendant support has an incomplete eight-draw candidate panel")
    outputs = [row for row in rows if row["panel"]["output_count"]]
    if any(row["panel"]["output_count"] not in (0, 1) for row in rows):
        raise ValueError("pendant attempt emitted more than one endpoint")
    supported = [entry for entry in offered if entry["status"] == "model_supported"]
    one_atom = sum(
        any(draw["heavy_atoms"] == 1 for draw in entry["provenance"]["pendant_plan"]["draws"])
        for entry in supported
    )
    ring_bearing = sum(
        any(draw["rings"] > 0 for draw in entry["provenance"]["pendant_plan"]["draws"])
        for entry in supported
    )
    per_prompt = {drug: sum(row["drug"] == drug for row in outputs) for drug in prompts}
    valid = sum(row["selected_valid_connected"] for row in outputs)
    faithful = sum(row["selected_constraint_fidelity"] for row in outputs)
    checks = {
        "output_at_least_90_percent": len(outputs) / 20 >= contract["required_output_coverage"],
        "each_prompt_has_output": all(per_prompt.values()),
        "all_outputs_valid_connected": valid == len(outputs),
        "all_outputs_exact_constraint_fidelity": faithful == len(outputs),
        "one_atom_model_supported": one_atom >= contract["required_one_atom_model_supported"],
        "ring_bearing_model_supported": ring_bearing
        >= contract["required_ring_bearing_model_supported"],
    }
    return {
        "schema": "fragment_pendant_support_result_v1",
        "attempts": 20,
        "outputs": len(outputs),
        "valid_connected_outputs": valid,
        "exact_constraint_fidelity_outputs": faithful,
        "per_prompt_outputs": per_prompt,
        "offered_draws": len(offered),
        "candidate_status": dict(sorted(Counter(entry["status"] for entry in offered).items())),
        "model_supported_one_atom_draws": one_atom,
        "model_supported_ring_bearing_draws": ring_bearing,
        "checks": checks,
        "support_pass": all(checks.values()),
        "quality_evaluations": 0,
        "oracle_calls": 0,
    }


def preflight() -> tuple[dict, tuple]:
    contract = json.loads(CONTRACT.read_text())
    if (
        contract["schema"] != "fragment_pendant_support_contract_v1"
        or contract["task"] != "scaffold_decoration"
        or contract["prompt_count"] != 10
        or contract["attempts_per_prompt"] != 2
        or contract["candidate_draws_per_attempt"] != 8
        or contract["seed"] != 1
        or contract["max_workers"] != 1
        or contract["threads_per_worker"] != 1
        or contract["required_committed_validity"] != 1.0
        or contract["required_prompt_fidelity"] != 1.0
        or contract["quality_evaluation"] is not False
    ):
        raise ValueError("pendant contract changes the predeclared support gate")
    for path, expected in (
        (CATALOG, contract["catalog_sha256"]),
        (CHECKPOINT, contract["checkpoint_sha256"]),
        (PROMPTS, contract["prompt_sha256"]),
    ):
        if physical_sha256(path) != expected:
            raise ValueError(f"frozen pendant input changed: {path}")
    prompts = tuple(
        prompt
        for prompt in load_genmol_prompts(PROMPTS)
        if prompt.task is FragmentTask.SCAFFOLD_DECORATION
    )
    if [prompt.drug_name for prompt in prompts] != contract["prompts"]:
        raise ValueError("pendant prompt identity/order changed")
    return contract, prompts


def manifest_for(contract: dict) -> dict:
    paths = {
        CONTRACT,
        CATALOG,
        CHECKPOINT,
        PROMPTS,
        Path(__file__),
        ROOT / "src/compose_v4/benchmark/fragment_pendant_programs.py",
        ROOT / "src/compose_v4/benchmark/pendant_completion_policy.py",
        ROOT / "src/compose_v4/benchmark/fragment_program_adapter.py",
        ROOT / "src/compose_v4/benchmark/fragment_conditioned_sampler.py",
        ROOT / "docs/FRAGMENT_PENDANT_DECORATION_DEV_2026-09-24.md",
        ROOT / "scripts/evaluate_tracelet_rollouts.py",
        ROOT / "tools/run_fragment_attachment_library_pilot.py",
        ROOT / "tools/run_fragment_constrained_suite.py",
        ROOT / "tools/run_fragment_training_linker_support.py",
    }
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("compose_v4") and filename and filename.endswith(".py"):
            paths.add(Path(filename).resolve())
    return {
        "schema": "fragment_pendant_support_manifest_v1",
        "contract": contract,
        "inputs": {str(path.resolve()): physical_sha256(path) for path in sorted(paths)},
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "device": "cpu",
        "precision": "float32",
        "threads": 1,
        "role": "training-only structural support; no official quality evaluation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract, prompts = preflight()
    manifest = manifest_for(contract)
    output = args.output_dir.resolve()
    expected = (ROOT / contract["output_dir"]).resolve()
    if output != expected:
        raise ValueError("pendant support output differs from frozen contract")
    manifest_path = output / "manifest.json"
    if args.phase == "prepare":
        if output.exists():
            raise FileExistsError(output)
        _atomic_json(manifest_path, manifest)
        print(
            json.dumps({"prepared_only": True, "manifest_sha256": physical_sha256(manifest_path)})
        )
        return
    if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("pendant support manifest changed before run/resume")
    if (output / "summary.json").exists():
        raise FileExistsError("pendant support result already complete")
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    sampler = PendantCompletionSampler(json.loads(CATALOG.read_text()))
    model, _ = load_factorized_rollout_checkpoint(CHECKPOINT)
    rows, hashes = [], {}
    for prompt in prompts:
        context = build_prompt_context(prompt)
        rng = np.random.default_rng(prompt_rng_seed(prompt.drug_name, prompt.task.value, 1))
        for index in range(2):
            target = output / "attempts" / f"{prompt.drug_name}_{index:03d}.json"
            start = output / "starts" / f"{prompt.drug_name}_{index:03d}.json"
            if target.exists():
                row = json.loads(target.read_text())
                restore_completed_attempt(row, drug=prompt.drug_name, attempt_index=index, rng=rng)
            else:
                if start.exists():
                    raise ValueError(f"interrupted pendant panel; no automatic redraw: {start}")
                _atomic_json(
                    start,
                    {
                        "drug": prompt.drug_name,
                        "attempt_index": index,
                        "rng_state_before": rng.bit_generator.state,
                        "manifest_sha256": physical_sha256(manifest_path),
                    },
                )
                began = time.monotonic()
                panel = sample_pendant_panel(context, sampler, model, rng)
                selected = panel.selected
                fidelity = False
                if selected:
                    constraint = ProgramConstraint.from_context(context)
                    lock = constraint.lock(context.start_state)
                    fidelity = constraint.complete(selected.endpoint) and all(
                        lock.permits(decode_state(state)) for state in selected.trace["states"]
                    )
                    verify_prompt_endpoint(context, selected)
                row = {
                    "drug": prompt.drug_name,
                    "attempt_index": index,
                    "panel": panel.receipt,
                    "wall_seconds": time.monotonic() - began,
                    "selected_valid_connected": bool(
                        selected
                        and is_valid_state(selected.endpoint)
                        and is_connected_or_null(selected.endpoint)
                    ),
                    "selected_constraint_fidelity": fidelity,
                }
                if selected and not (
                    row["selected_valid_connected"] and row["selected_constraint_fidelity"]
                ):
                    raise RuntimeError("pendant committed invalid or constraint-breaking endpoint")
                _atomic_json(target, row)
            rows.append(row)
            hashes[str(target.relative_to(output))] = physical_sha256(target)
            _atomic_json(
                output / "progress.json",
                {
                    "completed_attempts": len(rows),
                    "total_attempts": 20,
                    "outputs": sum(item["panel"]["output_count"] for item in rows),
                    "drug": prompt.drug_name,
                    "attempt_index": index,
                },
            )
            print(
                json.dumps(
                    {
                        "drug": prompt.drug_name,
                        "attempt": index,
                        "output": bool(row["panel"]["output_count"]),
                    }
                ),
                flush=True,
            )
    if any(physical_sha256(Path(path)) != digest for path, digest in manifest["inputs"].items()):
        raise ValueError("pendant material changed during support gate")
    summary = support_summary(rows, contract)
    summary.update(
        {
            "manifest_sha256": physical_sha256(manifest_path),
            "attempt_sha256": dict(sorted(hashes.items())),
        }
    )
    _atomic_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "support_pass": summary["support_pass"],
                "outputs": summary["outputs"],
                "checks": summary["checks"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
