"""One resumable all-prompt complete-program pilot, gated by same-code support.

Each candidate draw and output attempt is checkpointed. Official metric values
are computed only after an entire prompt's samples are durably locked. Frozen
baseline endpoints are reused, not pooled with other runs or silently regenerated.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_complete_program_gate import ROOT, TASKS, capabilities
from run_fragment_constrained_suite import prompt_reference_smiles, prompt_rng_seed

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import load_genmol_prompts
from compose_v4.benchmark.fragment_official_metrics import (
    FAILED_SAMPLE_PLACEHOLDER,
    official_distance,
    official_prompt_metrics,
)
from compose_v4.benchmark.fragment_program_adapter import (
    CompleteProgram,
    learned_program_scores,
    select_learned_program,
    verify_prompt_endpoint,
)
from compose_v4.benchmark.fragment_t4_programs import propose_program_panel_member
from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.experiments.whole_ring_plan import execute_program


def restore_candidates(source, records):
    candidates, scores = [], []
    for record in records:
        if record["status"] != "model_supported":
            continue
        endpoint, trace = execute_program(source, record["actions"])
        candidate = CompleteProgram(endpoint, trace, record["provenance"])
        if candidate.smiles != record["endpoint"]:
            raise ValueError("saved pilot candidate changed on exact replay")
        candidates.append(candidate)
        scores.append(record["mean_log_mark"])
    return candidates, scores


def restore_attempt_rng(attempt, index, rng):
    if attempt["attempt_index"] != index or attempt["rng_state_before"] != rng.bit_generator.state:
        raise ValueError("attempt resume identity/RNG chain changed")
    records = attempt["offered"]
    if len(records) > 8 or [r["draw"] for r in records] != list(range(len(records))):
        raise ValueError("attempt contains missing, reordered or extra candidate draws")
    if attempt["complete"] and len(records) != 8:
        raise ValueError("completed output attempt lacks its eight candidate records")
    rng.bit_generator.state = attempt["rng_state_after"]


def require_qualified_support(support, support_manifest):
    if (
        support_manifest["mode"] != "support"
        or support_manifest["sample_count_per_prompt"] != 2
        or support_manifest["panel_attempts"] != 8
        or support_manifest["seed"] != 0
        or not support["support_pass"]
        or not support["native_t4_support_pass"]
        or not support["every_prompt_has_output"]
        or len(support["shard_hashes"]) != 20
        or set(support["tasks"]) != {t.value for t in TASKS}
        or any(t["attempts"] != 20 for t in support["tasks"].values())
    ):
        raise ValueError("all-prompt same-code support gate is not complete/passed")


def assigned_prompt(index, shard_index, shard_count):
    """Keep each drug's motif/decoration pair together; preserve per-prompt RNG."""
    if shard_count not in (1, 2, 3, 4) or not 0 <= shard_index < shard_count:
        raise ValueError("pilot permits one to four bounded CPU shards")
    return (index // 2) % shard_count == shard_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    assigned_prompt(0, args.shard_index, args.shard_count)
    support = json.loads((args.support_dir / "summary.json").read_text())
    support_manifest = json.loads((args.support_dir / "manifest.json").read_text())
    require_qualified_support(support, support_manifest)
    for raw, expected in support_manifest["inputs"].items():
        if physical_sha256(Path(raw)) != expected:
            raise ValueError(f"material support-gate input changed: {raw}")
    for relative, expected in support["shard_hashes"].items():
        if physical_sha256(args.support_dir / relative) != expected:
            raise ValueError(f"support shard changed: {relative}")
    baseline_manifest = json.loads((args.baseline_dir / "manifest.json").read_text())
    prompt_path = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    evaluator = ROOT / "src/compose_v4/benchmark/fragment_official_metrics.py"
    if (
        baseline_manifest["prompt_sha256"] != physical_sha256(prompt_path)
        or baseline_manifest["checkpoint_sha256"] != physical_sha256(args.checkpoint)
        or baseline_manifest["evaluator_sha256"] != physical_sha256(evaluator)
        or baseline_manifest["samples_per_prompt"] != 20
        or baseline_manifest["seed"] != 0
    ):
        raise ValueError("frozen baseline is not the same checkpoint/prompts/evaluator/attempts")
    catalog_path = args.catalog_dir / "catalog.json"
    if support_manifest["inputs"].get(str(catalog_path.resolve())) != physical_sha256(catalog_path):
        raise ValueError("pilot catalog differs from qualified support catalog")
    entries = tuple(
        e for e in json.loads(catalog_path.read_text())["entries"] if len(e["contexts"]) == 1
    )
    prompts = [p for p in load_genmol_prompts(prompt_path) if p.task in TASKS]
    baseline_paths = [
        args.baseline_dir / "shards" / f"{p.task.value}__{p.drug_name}__baseline.json"
        for p in prompts
    ]
    material = {
        **support_manifest["inputs"],
        str(Path(__file__).resolve()): physical_sha256(Path(__file__)),
        str((args.support_dir / "summary.json").resolve()): physical_sha256(
            args.support_dir / "summary.json"
        ),
    }
    material.update({str(p.resolve()): physical_sha256(p) for p in baseline_paths})
    material[str((args.baseline_dir / "manifest.json").resolve())] = physical_sha256(
        args.baseline_dir / "manifest.json"
    )
    manifest = {
        "schema": "fragment_complete_program_matched_pilot_v1",
        "inputs": material,
        "same_seed": 0,
        "attempts_per_prompt": 20,
        "candidate_attempts_per_output": 8,
        "prompts": 20,
        "baseline_settings": baseline_manifest["baseline_attachment_control"],
        "scoring": support_manifest["runtime_scoring"],
        "qed_sa_guidance": False,
        "role": "development bundle comparison; not isolated proof of coordination causality",
        "checkpointed_unit": "candidate draw and selected output attempt",
        "execution": {"cpu_workers": args.shard_count, "assignment": "drug-pair round robin"},
        "hardware": {
            key: support_manifest[key]
            for key in ("device", "threads", "precision", "torch", "python", "rdkit")
        },
    }
    path = args.output_dir / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("pilot resume identity mismatch")
    _atomic_json(path, manifest)
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    torch.set_num_threads(1)
    model, _ = load_factorized_rollout_checkpoint(args.checkpoint)
    rows = []
    for prompt_index, (prompt, baseline_path) in enumerate(
        zip(prompts, baseline_paths, strict=True)
    ):
        if not assigned_prompt(prompt_index, args.shard_index, args.shard_count):
            continue
        context = build_prompt_context(prompt)
        core = Chem.MolFromSmiles(context.start_smiles)
        initial_rings = core.GetRingInfo().NumRings()
        rng = np.random.default_rng(prompt_rng_seed(prompt.drug_name, prompt.task.value, 0))
        attempt_rows = []
        for index in range(20):
            target = (
                args.output_dir
                / "attempts"
                / f"{prompt.task.value}_{prompt.drug_name}_{index:03d}.json"
            )
            if target.exists():
                attempt = json.loads(target.read_text())
                restore_attempt_rng(attempt, index, rng)
                if attempt["complete"]:
                    attempt_rows.append(attempt)
                    continue
            else:
                attempt = {
                    "attempt_index": index,
                    "rng_state_before": rng.bit_generator.state,
                    "rng_state_after": rng.bit_generator.state,
                    "complete": False,
                    "offered": [],
                }
            records = attempt["offered"]
            # All completed draws, including refusals, retain their RNG advancement.
            # A crash during a draw reruns just that draw from its saved pre-state.
            for draw in range(len(records), 8):
                started = time.monotonic()
                record = {"draw": draw}
                try:
                    candidate = propose_program_panel_member(context, entries, rng, draw)
                except ValueError as error:
                    record.update(status="compiler_or_constraint_abstention", reason=str(error))
                    if hasattr(error, "receipt"):
                        record["refusal_receipt"] = error.receipt
                else:
                    record.update(
                        endpoint=candidate.smiles,
                        provenance=candidate.provenance,
                        actions=candidate.trace["actions"],
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
                records.append(record)
                attempt["rng_state_after"] = rng.bit_generator.state
                _atomic_json(target, attempt)
            candidates, scores = restore_candidates(context.start_state, records)
            if candidates:
                candidate, receipt = select_learned_program(candidates, scores, rng)
                verify_prompt_endpoint(context, candidate)
                attempt.update(
                    committed_smiles=candidate.smiles,
                    prompt_fidelity=True,
                    selection=receipt,
                    selected_capabilities=capabilities(candidate, initial_rings),
                )
            else:
                attempt.update(
                    committed_smiles=None,
                    prompt_fidelity=False,
                    selection={"reason": "empty_model_supported_panel"},
                    selected_capabilities=None,
                )
            attempt.update(complete=True, rng_state_after=rng.bit_generator.state)
            _atomic_json(target, attempt)
            attempt_rows.append(attempt)
        samples = [a["committed_smiles"] or FAILED_SAMPLE_PLACEHOLDER for a in attempt_rows]
        metrics = official_prompt_metrics(samples, expected_samples=20)
        distance = official_distance(samples, prompt_reference_smiles(prompt))
        metrics["distance"] = None if np.isnan(distance) else distance
        baseline = json.loads(baseline_path.read_text())
        row = {
            "task": prompt.task.value,
            "drug": prompt.drug_name,
            "new": metrics,
            "baseline": baseline["metrics"],
            "outputs": sum(a["committed_smiles"] is not None for a in attempt_rows),
            "prompt_fidelity_numerator": sum(a["prompt_fidelity"] for a in attempt_rows),
            "attempts": 20,
            "candidate_attempts": sum(len(a["offered"]) for a in attempt_rows),
            "wall_seconds": sum(c["wall_seconds"] for a in attempt_rows for c in a["offered"]),
        }
        _atomic_json(args.output_dir / "rows" / f"{prompt.task.value}_{prompt.drug_name}.json", row)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    if material != {raw: physical_sha256(Path(raw)) for raw in material}:
        raise RuntimeError("pilot inputs changed during generation")
    expected_rows = [
        args.output_dir / "rows" / f"{p.task.value}_{p.drug_name}.json" for p in prompts
    ]
    if not all(p.is_file() for p in expected_rows):
        print(
            json.dumps({"shard_complete": args.shard_index, "all_rows_complete": False}), flush=True
        )
        return
    rows = [json.loads(p.read_text()) for p in expected_rows]
    if len(rows) != 20 or any(r["attempts"] != 20 for r in rows):
        raise ValueError("matched summary requires every exact twenty-attempt prompt row")
    result = {"schema": "fragment_complete_program_matched_summary_v1", "tasks": {}}
    for task in TASKS:
        subset = [r for r in rows if r["task"] == task.value]
        result["tasks"][task.value] = {
            arm: {
                name: float(np.mean([r[arm][name] for r in subset]))
                for name in ("validity", "uniqueness", "quality", "diversity")
            }
            for arm in ("baseline", "new")
        }
        task_result = result["tasks"][task.value]
        task_result["attempts"] = sum(r["attempts"] for r in subset)
        task_result["outputs"] = sum(r["outputs"] for r in subset)
        task_result["prompt_fidelity_numerator"] = sum(
            r["prompt_fidelity_numerator"] for r in subset
        )
        task_result["candidate_attempts"] = sum(r["candidate_attempts"] for r in subset)
        task_result["wall_seconds"] = sum(r["wall_seconds"] for r in subset)
    result["row_hashes"] = {
        str(p.relative_to(args.output_dir)): physical_sha256(p) for p in expected_rows
    }
    result["attempt_hashes"] = {
        str(p.relative_to(args.output_dir)): physical_sha256(p)
        for p in [
            args.output_dir
            / "attempts"
            / f"{prompt.task.value}_{prompt.drug_name}_{index:03d}.json"
            for prompt in prompts
            for index in range(20)
        ]
    }
    _atomic_json(args.output_dir / "summary.json", result)


if __name__ == "__main__":
    main()
