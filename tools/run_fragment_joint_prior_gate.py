"""One-worker resumable support gate and matched pilot for joint completion."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_complete_program_gate import CHECKPOINT_SHA, TASKS, capabilities
from run_fragment_constrained_suite import prompt_rng_seed
from run_fragment_program_matched_pilot import restore_attempt_rng, restore_candidates

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import load_genmol_prompts
from compose_v4.benchmark.fragment_joint_programs import propose_joint_panel_member
from compose_v4.benchmark.fragment_program_adapter import (
    learned_program_scores,
    select_learned_program,
    verify_prompt_endpoint,
)
from compose_v4.benchmark.fragment_t4_programs import T4_LANES
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior, JointCompletionSampler
from compose_v4.benchmark.training_attachment_fragments import physical_sha256


def collect_summary(rows, mode):
    summary = {"schema": "fragment_joint_prior_summary_v1", "mode": mode, "tasks": {}}
    for task in TASKS:
        subset = [r for r in rows if r["task"] == task.value]
        attempts = [a for r in subset for a in r["attempts"]]
        caps = [a["selected_capabilities"] for a in attempts if a["committed_smiles"]]
        offered = [c for a in attempts for c in a["offered"]]
        metrics = {
            "attempts": len(attempts),
            "outputs": len(caps),
            "candidate_attempts": len(offered),
            "candidate_status": dict(Counter(c["status"] for c in offered)),
            "model_supported_t4_lanes": dict(
                Counter(
                    c["capabilities"]["t4_lane"]
                    for c in offered
                    if c["status"] == "model_supported" and c["capabilities"]["t4_lane"]
                )
            ),
            "selected_multi_primitive_fraction": sum(c["multi_primitive"] for c in caps)
            / max(1, len(caps)),
            "selected_new_ring_fraction": sum(c["new_ring"] for c in caps) / max(1, len(caps)),
            "selected_t4_fraction": sum(c["t4_refinement"] for c in caps) / max(1, len(caps)),
            "prompt_fidelity": sum(a["prompt_fidelity"] for a in attempts),
            "mean_selected_heavy_atoms": float(
                np.mean(
                    [
                        a["selected_structure"]["heavy_atoms"]
                        for a in attempts
                        if a["committed_smiles"]
                    ]
                )
            )
            if caps
            else None,
            "mean_selected_rings": float(
                np.mean(
                    [a["selected_structure"]["rings"] for a in attempts if a["committed_smiles"]]
                )
            )
            if caps
            else None,
        }
        if mode == "pilot":
            metrics["official"] = {
                name: float(np.mean([r["official"][name] for r in subset]))
                for name in ("validity", "uniqueness", "quality", "diversity")
            }
        summary["tasks"][task.value] = metrics
    summary["every_prompt_has_output"] = len(rows) == 20 and all(
        any(a["committed_smiles"] for a in r["attempts"]) for r in rows
    )
    summary["native_t4_support_pass"] = all(
        sum(m["model_supported_t4_lanes"].get(lane, 0) for m in summary["tasks"].values()) > 0
        for lane in T4_LANES
    )
    summary["support_pass"] = (
        summary["every_prompt_has_output"]
        and summary["native_t4_support_pass"]
        and all(
            m["attempts"] > 0
            and m["outputs"] / m["attempts"] >= 0.9
            and m["prompt_fidelity"] == m["outputs"]
            and m["selected_multi_primitive_fraction"] > 0.5
            and m["selected_new_ring_fraction"] >= 0.1
            and m["selected_t4_fraction"] > 0
            for m in summary["tasks"].values()
        )
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-support-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("support", "pilot"), default="support")
    parser.add_argument("--qualified-support-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = json.loads((args.base_support_dir / "manifest.json").read_text())
    material = dict(base["inputs"])
    for path, digest in material.items():
        if physical_sha256(Path(path)) != digest:
            raise ValueError(f"frozen base input changed: {path}")
    if physical_sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("not the frozen molecular checkpoint")
    catalog_path = args.catalog_dir / "catalog.json"
    prior_data = json.loads(args.prior.read_text())
    if prior_data["catalog_sha256"] != physical_sha256(catalog_path):
        raise ValueError("joint prior and region library differ")
    for p in (
        Path(__file__),
        args.prior,
        Path("src/compose_v4/benchmark/joint_completion_prior.py"),
        Path("src/compose_v4/benchmark/fragment_joint_programs.py"),
        Path("docs/FRAGMENT_JOINT_COMPLETION_PRIOR_DEV_2026-09-24.md"),
        Path("tools/run_fragment_program_matched_pilot.py"),
    ):
        material[str(p.resolve())] = physical_sha256(p)
    if args.mode == "pilot":
        if args.qualified_support_dir is None:
            raise ValueError("pilot requires same-code qualified support")
        previous = json.loads((args.qualified_support_dir / "manifest.json").read_text())
        gate = json.loads((args.qualified_support_dir / "summary.json").read_text())
        if previous["inputs"] != material or not gate["support_pass"]:
            raise ValueError("support gate failed or material input changed")
        for name, digest in gate["row_sha256"].items():
            if physical_sha256(args.qualified_support_dir / name) != digest:
                raise ValueError("qualification row changed")
    # Import/hash verification before molecular work, without evaluating quality.
    from fetch_official_fragment_evaluator import verify_only

    verify_only()
    from compose_v4.benchmark.fragment_official_metrics import (
        FAILED_SAMPLE_PLACEHOLDER,
        official_prompt_metrics,
    )

    samples = 2 if args.mode == "support" else 20
    manifest = {
        "schema": "fragment_joint_prior_gate_v1",
        "mode": args.mode,
        "inputs": material,
        "attempts_per_prompt": samples,
        "candidate_attempts": 8,
        "seed": 0,
        "seed_derivation": "existing BLAKE2b drug|task|seed",
        "quality_guidance": False,
        "source_density": "joint training whole-molecule atoms/rings; exact conditional allocation",
        "workers": 1,
        "threads": 1,
        "device": "cpu",
        "precision": "float32",
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("resume manifest differs")
    _atomic_json(manifest_path, manifest)
    entries = json.loads(catalog_path.read_text())["entries"]
    sampler = JointCompletionSampler(entries, JointCompletionPrior.from_dict(prior_data))
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, _ = load_factorized_rollout_checkpoint(args.checkpoint)
    prompt_path = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
    prompts = [p for p in load_genmol_prompts(prompt_path) if p.task in TASKS]
    rows = []
    for prompt in prompts:
        context = build_prompt_context(prompt)
        initial_rings = Chem.MolFromSmiles(context.start_smiles).GetRingInfo().NumRings()
        rng = np.random.default_rng(prompt_rng_seed(prompt.drug_name, prompt.task.value, 0))
        attempts = []
        for index in range(samples):
            target = (
                args.output_dir
                / "attempts"
                / f"{prompt.task.value}_{prompt.drug_name}_{index:03d}.json"
            )
            if target.exists():
                attempt = json.loads(target.read_text())
                restore_attempt_rng(attempt, index, rng)
                if attempt["complete"]:
                    attempts.append(attempt)
                    continue
            else:
                attempt = {
                    "attempt_index": index,
                    "rng_state_before": rng.bit_generator.state,
                    "rng_state_after": rng.bit_generator.state,
                    "complete": False,
                    "offered": [],
                }
            for draw in range(len(attempt["offered"]), 8):
                started = time.monotonic()
                record = {"draw": draw}
                try:
                    candidate = propose_joint_panel_member(context, sampler, rng, draw)
                except ValueError as error:
                    record.update(status="compiler_or_constraint_abstention", reason=str(error))
                    if hasattr(error, "receipt"):
                        record["refusal_receipt"] = error.receipt
                else:
                    record.update(
                        endpoint=candidate.smiles,
                        actions=candidate.trace["actions"],
                        provenance=candidate.provenance,
                        capabilities=capabilities(candidate, initial_rings),
                    )
                    try:
                        [score] = learned_program_scores(model, (candidate,))
                    except (ValueError, KeyError) as error:
                        record.update(status="model_support_abstention", reason=str(error))
                    else:
                        if np.isfinite(score):
                            record.update(status="model_supported", mean_log_mark=score)
                        else:
                            record.update(
                                status="model_support_abstention",
                                reason="nonfinite_native_probability",
                            )
                record["wall_seconds"] = time.monotonic() - started
                attempt["offered"].append(record)
                attempt["rng_state_after"] = rng.bit_generator.state
                _atomic_json(target, attempt)
            candidates, scores = restore_candidates(context.start_state, attempt["offered"])
            if candidates:
                selected, receipt = select_learned_program(candidates, scores, rng)
                verify_prompt_endpoint(context, selected)
                mol = Chem.MolFromSmiles(selected.smiles)
                attempt.update(
                    committed_smiles=selected.smiles,
                    prompt_fidelity=True,
                    selection=receipt,
                    selected_capabilities=capabilities(selected, initial_rings),
                    selected_structure={
                        "heavy_atoms": mol.GetNumHeavyAtoms(),
                        "rings": mol.GetRingInfo().NumRings(),
                    },
                    joint_plan=selected.provenance["joint_plan"],
                )
            else:
                attempt.update(
                    committed_smiles=None,
                    prompt_fidelity=False,
                    selected_capabilities=None,
                    selection={"reason": "empty_model_supported_panel"},
                )
            attempt.update(complete=True, rng_state_after=rng.bit_generator.state)
            _atomic_json(target, attempt)
            attempts.append(attempt)
            print(
                json.dumps(
                    {
                        "task": prompt.task.value,
                        "drug": prompt.drug_name,
                        "attempt": index,
                        "output": bool(attempt["committed_smiles"]),
                        "structure": attempt.get("selected_structure"),
                    }
                ),
                flush=True,
            )
        row = {
            "task": prompt.task.value,
            "drug": prompt.drug_name,
            "attempts": attempts,
            "start_smiles": context.start_smiles,
        }
        if args.mode == "pilot":
            row["official"] = official_prompt_metrics(
                [a["committed_smiles"] or FAILED_SAMPLE_PLACEHOLDER for a in attempts],
                expected_samples=20,
            )
        _atomic_json(args.output_dir / "rows" / f"{prompt.task.value}_{prompt.drug_name}.json", row)
        rows.append(row)
    if any(physical_sha256(Path(p)) != digest for p, digest in material.items()):
        raise RuntimeError("material input changed during generation")
    summary = collect_summary(rows, args.mode)
    summary["row_sha256"] = {
        f"rows/{p.task.value}_{p.drug_name}.json": physical_sha256(
            args.output_dir / "rows" / f"{p.task.value}_{p.drug_name}.json"
        )
        for p in prompts
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
