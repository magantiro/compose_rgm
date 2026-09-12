"""Answer-known proposal ablation: coordination versus uninterrupted execution.

This deliberately uses the supplied winning route to prepare a program bank. It
tests whether the controller can assign useful mass to that demonstrated class;
it is NOT source-held-out learning, autonomous discovery or a docking benchmark.
The generator never reads the target endpoint. Target comparison follows locking.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.control.edit_program_policy import ProgramEntry
from compose_v4.control.multi_site_program_policy import MODES, generate_program_candidates
from compose_v4.experiments.continuation_profile import ExecutorMeter
from compose_v4.experiments.t4_winner_refinement import property_scorer
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.ivg_winner_paths import implementation_closure, publish, sha


def run(args):
    if (args.output / "result.json").exists():
        raise ValueError("complete proposal probe already exists; reuse rather than overwrite")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("proposal probe requires pinned RDKit 2024.03.5")
    started = perf_counter()
    saved = json.loads(args.route.read_text())
    source = decode_state(saved["source_state"])
    original_seed = canonical_state_key(source)
    score = property_scorer(original_seed)
    stages = saved["attempts"][0]["stages"]
    summaries, input_programs = [], []
    meter = ExecutorMeter(None)
    with meter.instrument():
        for context, offset in (("original_seed", 0), ("post_linker", 1)):
            if context not in args.contexts:
                continue
            parent = decode_state(stages[offset]["states"][0])
            local = {}
            for stage in stages[offset:]:
                program, _ = extract_program(decode_state(stage["states"][0]), [stage])
                local.setdefault(program.program_id, ProgramEntry(program, (original_seed,)))
            joint, _ = extract_program(parent, stages[offset:])
            # Give the controls the same complete peripheral branch and core
            # branch, not only a menu of single-stage fragments. Core-first exact
            # replay supplies its parent-state binding without any slot recovery.
            peripheral, _ = extract_program(parent, stages[offset:-1])
            local.setdefault(peripheral.program_id, ProgramEntry(peripheral, (original_seed,)))
            _, binding = extract_program(parent, stages[offset:])
            core_index = len(joint.blocks) - 1
            _, core_first = execute_program_graph(
                parent,
                compile_program_graph(joint),
                binding,
                priority=(core_index, *range(core_index)),
            )
            length = joint.blocks[-1].stop - joint.blocks[-2].stop
            core_stage = {
                "name": "core",
                "actions": core_first["actions"][:length],
                "states": core_first["states"][: length + 1],
                "endpoint": canonical_state_key(decode_state(core_first["states"][length])),
            }
            core, _ = extract_program(parent, [core_stage])
            local.setdefault(core.program_id, ProgramEntry(core, (original_seed,)))
            input_programs.append(
                {
                    "context": context,
                    "parent_state": stages[offset]["states"][0],
                    "local_programs": [entry.program.payload() for entry in local.values()],
                    "joint_program": joint.payload(),
                }
            )
            for mode in MODES:
                before_calls = meter.calls
                pool = generate_program_candidates(
                    parent,
                    tuple(local.values()),
                    mode=mode,
                    seed=args.seed,
                    attempts=args.attempts,
                    sites=2,
                    max_primitives=24,
                    max_blocks=len(joint.blocks),
                    wall_seconds=120,
                    joint_entries=(ProgramEntry(joint, (original_seed,)),),
                )
                path = args.output / f"{context}_{mode}.json.gz"
                publish(path, pool, compressed=True)  # Lock before property/target diagnostics.
                unique = list(pool["unique_endpoint_first_attempt"])
                properties = [score({"smiles": smi}) for smi in unique]
                counts = Counter(row["status"] for row in pool["rows"])
                failures = Counter(
                    row["reason_code"] for row in pool["rows"] if row["status"] != "complete"
                )
                sites = Counter(
                    str(row["receipt"]["actual_changes"]["changed_site_count"])
                    for row in pool["rows"]
                    if row["status"] == "complete"
                )
                summary = {
                    "context": context,
                    "arm": mode,
                    "pool_sha256": sha(path),
                    "pool_path": path.name,
                    "status_counts": dict(counts),
                    "failures": dict(failures),
                    "actual_site_counts": dict(sites),
                    "unique_endpoints": len(unique),
                    "unique_eligible_endpoints": sum(p["oracle_eligible"] for p in properties),
                    "exact_demonstrated_endpoint_proposed": saved["target"] in unique,
                    "exact_endpoint_attempts": sum(
                        row["status"] == "complete"
                        and row["receipt"]["endpoint"] == saved["target"]
                        for row in pool["rows"]
                    ),
                    "executor_calls": meter.calls - before_calls,
                    "seconds": pool["proposal_seconds"],
                    "attempts_unstarted": pool["attempts_unstarted"],
                    "endpoint_properties": properties,
                }
                summaries.append(summary)
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in summary.items()
                            if key != "endpoint_properties"
                        }
                    ),
                    flush=True,
                )
    publish(args.output / "programs.json.gz", {"contexts": input_programs}, compressed=True)
    result = {
        "schema_version": "multi_site_proposal_probe_v1",
        "input": {"path": str(args.route.resolve()), "sha256": sha(args.route)},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "implementation_sha256": implementation_closure(Path(__file__).resolve()),
        "programs_sha256": sha(args.output / "programs.json.gz"),
        "configuration": {
            "seed": args.seed,
            "seed_derivation": "same integer in paired arms",
            "attempts": args.attempts,
            "sites": 2,
            "max_primitives": 24,
            "max_blocks_by_context": {"original_seed": 5, "post_linker": 4},
            "contexts": args.contexts,
            "wall_seconds_per_arm_context": 120,
        },
        "arms": summaries,
        "costs": {
            "seconds": perf_counter() - started,
            "executor_calls": meter.calls,
            "new_oracle_calls": 0,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {"machine": platform.machine(), "device": "cpu"},
        "evidence": "winner-informed in-demonstration proposal diagnostic, not transfer or autonomous discovery",
        "split": "explicit answer-known development; not added to source-held-out training or evaluation",
        "limitations": [
            "one demonstrated route, two correlated contexts",
            "no task-value fitting",
            "shared work ceilings, differing actual work",
            "winner-specific preparation is task information even without endpoint at inference",
        ],
    }
    publish(args.output / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route", type=Path, default=Path("diagnostics/t4_whole_ring_plan/result.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--attempts", type=int, default=32)
    parser.add_argument(
        "--contexts",
        nargs="+",
        choices=("original_seed", "post_linker"),
        default=["original_seed", "post_linker"],
    )
    run(parser.parse_args())
