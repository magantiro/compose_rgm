"""Frozen-checkpoint support gate, then one explicitly selected matched pilot.

No objective values are evaluated during generation. Every candidate/refusal and
selected exact program is persisted. The support-only mode does not calculate
QED/SA; it cannot select per-prompt quality settings.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_constrained_suite import prompt_rng_seed

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_program_adapter import (
    learned_program_scores,
    select_learned_program,
)
from compose_v4.benchmark.fragment_t4_programs import T4_LANES, propose_program_panel_member
from compose_v4.benchmark.training_attachment_fragments import physical_sha256

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA = "24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4"
TASKS = (FragmentTask.MOTIF_EXTENSION, FragmentTask.SCAFFOLD_DECORATION)


def capabilities(candidate, initial_rings):
    mol = Chem.MolFromSmiles(candidate.smiles)
    actions = candidate.trace["actions"]
    rings = mol.GetRingInfo().NumRings()
    return {
        "primitives": len(actions),
        "multi_primitive": len(actions) > 1,
        "ring_containing_endpoint": rings > 0,
        "new_ring": rings > initial_rings,
        "whole_region_program": True,
        "t4_refinement": candidate.provenance.get("t4_changed_endpoint", False),
        "t4_lane": candidate.provenance.get("t4_lane"),
        "program_blocks": len(candidate.provenance["program_block_lengths"]),
        "dependency_edges": len(candidate.provenance.get("dependencies", ())),
        "conflict_edges": len(candidate.provenance.get("conflicts", ())),
        "two_boundary_program": any(
            r["boundary_arity"] == 2 for r in candidate.provenance["regions"]
        ),
        "one_step": len(actions) == 1,
        "primitive_families": dict(sorted(Counter(a["executor_rule"] for a in actions).items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("support", "pilot"), default="support")
    parser.add_argument("--qualified-support-dir", type=Path)
    args = parser.parse_args()
    if physical_sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("not the frozen paired checkpoint")
    catalog_path = args.catalog_dir / "catalog.json"
    catalog_manifest_path = args.catalog_dir / "manifest.json"
    catalog_manifest = json.loads(catalog_manifest_path.read_text())
    if physical_sha256(catalog_path) != catalog_manifest["catalog_sha256"]:
        raise ValueError("catalog hash mismatch")
    catalog = json.loads(catalog_path.read_text())
    if (
        catalog["schema"] != "split_first_training_region_catalog_v1"
        or catalog["split"]["partition"] != "train"
    ):
        raise ValueError("not the split-first broad-region library")
    prompt_path = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    material = [
        Path(__file__).resolve(),
        prompt_path,
        args.checkpoint,
        catalog_path,
        catalog_manifest_path,
    ]
    material += [
        ROOT / p
        for p in (
            "src/compose_v4/benchmark/fragment_program_adapter.py",
            "src/compose_v4/benchmark/fragment_t4_programs.py",
            "src/compose_v4/control/progressive_structured_sampler.py",
            "src/compose_v4/control/dynamic_program_synthesis.py",
            "src/compose_v4/control/dynamic_program_synthesis_v1.py",
            "src/compose_v4/control/ring_program.py",
            "src/compose_v4/benchmark/fragment_constrained.py",
            "src/compose_v4/benchmark/fragment_conditioned_sampler.py",
            "src/compose_v4/control/structural_subgoal.py",
            "src/compose_v4/control/structural_subgoal_realizer.py",
            "src/compose_v4/control/edit_program.py",
            "src/compose_v4/control/edit_program_graph.py",
            "src/compose_v4/model/factorized_tracelet_rate_model.py",
            "scripts/evaluate_tracelet_rollouts.py",
            "src/compose_v4/benchmark/fragment_official_metrics.py",
            "src/compose_v4/rewrite/kernel.py",
            "src/compose_v4/rewrite/operators.py",
        )
    ]
    hashes = {str(p.resolve()): physical_sha256(p) for p in material}
    if args.mode == "pilot":
        if args.qualified_support_dir is None:
            raise ValueError("pilot requires a passed same-code support gate")
        qualification = json.loads((args.qualified_support_dir / "summary.json").read_text())
        frozen = json.loads((args.qualified_support_dir / "manifest.json").read_text())
        if not qualification["support_pass"] or frozen["inputs"] != hashes:
            raise ValueError("support gate failed or material inputs changed")
    samples = 2 if args.mode == "support" else 20
    torch.set_num_threads(1)
    manifest = {
        "schema": "fragment_complete_program_gate_v1",
        "mode": args.mode,
        "inputs": hashes,
        "sample_count_per_prompt": samples,
        "panel_attempts": 8,
        "seed": 0,
        "seed_derivation": "existing BLAKE2b drug|task|seed",
        "runtime_scoring": "mean native log-mark probability, time 0, unit-temperature panel softmax",
        "quality_guidance": False,
        "candidate_source": "observed split-first training regions",
        "candidate_lanes": [
            "region",
            "region+t4_shallow",
            "region+t4_structured",
            "region+t4_anchored_replacement",
        ]
        * 2,
        "qualification": {
            "minimum_output_fraction_per_task": 0.9,
            "every_prompt_has_output": True,
            "multi_primitive_selected_fraction_above": 0.5,
            "new_ring_selected_fraction_at_least": 0.1,
            "every_native_t4_lane_model_supported": True,
            "native_t4_selected_in_each_task": True,
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rdkit": rdBase.rdkitVersion,
        "torch": torch.__version__,
        "device": "cpu",
        "threads": 1,
        "precision": "float32",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "scope": "motif/decoration only; no official linker/morphing claim",
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("resume manifest mismatch")
    else:
        _atomic_json(manifest_path, manifest)
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, _ = load_factorized_rollout_checkpoint(args.checkpoint)
    entries = tuple(e for e in catalog["entries"] if len(e["contexts"]) == 1)
    prompts = [p for p in load_genmol_prompts(prompt_path) if p.task in TASKS]
    if len(prompts) != 20:
        raise ValueError("expected all ten motif and ten decoration prompts")
    rows = []
    for prompt in prompts:
        shard = args.output_dir / "shards" / f"{prompt.task.value}_{prompt.drug_name}.json"
        if shard.exists():
            rows.append(json.loads(shard.read_text()))
            continue
        started = time.monotonic()
        context = build_prompt_context(prompt)
        core = Chem.MolFromSmiles(context.start_smiles)
        initial_rings = core.GetRingInfo().NumRings()
        seed = prompt_rng_seed(prompt.drug_name, prompt.task.value, 0)
        rng = np.random.default_rng(seed)
        attempts = []
        for attempt_index in range(samples):
            before = rng.bit_generator.state
            candidates, offered, scores = [], [], []
            for draw in range(8):
                record = {"draw": draw}
                candidate_started = time.monotonic()
                try:
                    candidate = propose_program_panel_member(context, entries, rng, draw)
                except ValueError as error:
                    record.update(status="compiler_or_constraint_abstention", reason=str(error))
                    if hasattr(error, "receipt"):
                        record["refusal_receipt"] = error.receipt
                    record["generation_seconds"] = time.monotonic() - candidate_started
                    offered.append(record)
                    continue
                record["generation_seconds"] = time.monotonic() - candidate_started
                record.update(
                    endpoint=candidate.smiles,
                    provenance=candidate.provenance,
                    actions=candidate.trace["actions"],
                    capabilities=capabilities(candidate, initial_rings),
                )
                scoring_started = time.monotonic()
                try:
                    [score] = learned_program_scores(model, (candidate,))
                except (ValueError, KeyError) as error:
                    record.update(status="model_support_abstention", reason=str(error))
                    record["scoring_seconds"] = time.monotonic() - scoring_started
                    offered.append(record)
                    continue
                record["scoring_seconds"] = time.monotonic() - scoring_started
                if not np.isfinite(score):
                    record.update(
                        status="model_support_abstention", reason="nonfinite_native_probability"
                    )
                    offered.append(record)
                    continue
                record.update(
                    status="model_supported", mean_log_mark=score, candidate_index=len(candidates)
                )
                offered.append(record)
                candidates.append(candidate)
                scores.append(score)
            if candidates:
                selected, receipt = select_learned_program(candidates, scores, rng)
                endpoint, cap = selected.smiles, capabilities(selected, initial_rings)
            else:
                endpoint, receipt, cap = None, {"reason": "empty_model_supported_panel"}, None
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "rng_state_before": before,
                    "offered": offered,
                    "committed_smiles": endpoint,
                    "selection": receipt,
                    "selected_capabilities": cap,
                }
            )
        row = {
            "task": prompt.task.value,
            "drug": prompt.drug_name,
            "rng_seed": seed,
            "start_smiles": context.start_smiles,
            "attempts": attempts,
            "wall_seconds": time.monotonic() - started,
        }
        # Calculate official metrics only after all candidates and selections are locked.
        _atomic_json(shard, row)
        rows.append(row)
        print(
            json.dumps(
                {
                    "task": row["task"],
                    "drug": row["drug"],
                    "outputs": sum(a["committed_smiles"] is not None for a in attempts),
                    "attempts": samples,
                    "seconds": round(row["wall_seconds"], 2),
                }
            ),
            flush=True,
        )
    if hashes != {str(p.resolve()): physical_sha256(p) for p in material}:
        raise RuntimeError("material code/data changed while the gate was running")
    summary = {
        "schema": "fragment_complete_program_gate_summary_v1",
        "mode": args.mode,
        "tasks": {},
    }
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task.value]
        attempts = [a for r in task_rows for a in r["attempts"]]
        selected = [a["selected_capabilities"] for a in attempts if a["selected_capabilities"]]
        offered = [c for a in attempts for c in a["offered"]]
        metrics = {
            "attempts": len(attempts),
            "outputs": len(selected),
            "candidate_attempts": len(offered),
            "candidate_status": dict(Counter(c["status"] for c in offered)),
            "model_supported_t4_lanes": dict(
                Counter(
                    c["capabilities"]["t4_lane"]
                    for c in offered
                    if c["status"] == "model_supported" and c["capabilities"]["t4_lane"]
                )
            ),
            "selected_t4_lanes": dict(Counter(c["t4_lane"] for c in selected if c["t4_lane"])),
            "selected_t4_fraction": sum(c["t4_refinement"] for c in selected)
            / max(1, len(selected)),
            "selected_multi_primitive_fraction": sum(c["multi_primitive"] for c in selected)
            / max(1, len(selected)),
            "selected_new_ring_fraction": sum(c["new_ring"] for c in selected)
            / max(1, len(selected)),
            "selected_two_boundary_fraction": sum(c["two_boundary_program"] for c in selected)
            / max(1, len(selected)),
        }
        if args.mode == "pilot":
            from compose_v4.benchmark.fragment_official_metrics import (
                FAILED_SAMPLE_PLACEHOLDER,
                official_prompt_metrics,
            )

            official = [
                official_prompt_metrics(
                    [a["committed_smiles"] or FAILED_SAMPLE_PLACEHOLDER for a in r["attempts"]],
                    expected_samples=20,
                )
                for r in task_rows
            ]
            metrics["official"] = {
                name: float(np.mean([r[name] for r in official]))
                for name in ("validity", "uniqueness", "quality", "diversity")
            }
        summary["tasks"][task.value] = metrics
    summary["support_pass"] = all(
        m["outputs"] / m["attempts"] >= 0.9
        and m["selected_multi_primitive_fraction"] > 0.5
        and m["selected_new_ring_fraction"] >= 0.1
        and m["selected_t4_fraction"] > 0
        for m in summary["tasks"].values()
    )
    summary["native_t4_support_pass"] = all(
        sum(m["model_supported_t4_lanes"].get(lane, 0) for m in summary["tasks"].values()) > 0
        for lane in T4_LANES
    )
    summary["every_prompt_has_output"] = all(
        any(a["committed_smiles"] is not None for a in row["attempts"]) for row in rows
    )
    summary["support_pass"] &= (
        summary["native_t4_support_pass"] and summary["every_prompt_has_output"]
    )
    summary["shard_hashes"] = {
        str(p.relative_to(args.output_dir)): physical_sha256(p)
        for p in sorted((args.output_dir / "shards").glob("*.json"))
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["tasks"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
