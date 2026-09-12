"""Verify the exact saved multi-site route under distinct legal block schedules."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.experiments.continuation_profile import ExecutorMeter
from compose_v4.rewrite.trace_shard import decode_state
from tools.ivg_winner_paths import implementation_closure, publish, sha


def run(args):
    if (args.output / "result.json").exists():
        raise ValueError("completed route probe exists; inspect it instead of overwriting")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("route probe requires pinned RDKit 2024.03.5")
    started = perf_counter()
    saved = json.loads(args.route.read_text())
    source = decode_state(saved["source_state"])
    stages = saved["attempts"][0]["stages"]
    meter = ExecutorMeter(None)
    with meter.instrument():
        program, assignment = extract_program(source, stages)
        dag = compile_program_graph(program)
        receipts = {}
        for arm, priority in (("recorded", (0, 1, 2, 3, 4)), ("core_first", (4, 0, 1, 2, 3))):
            _, receipt = execute_program_graph(source, dag, assignment, priority=priority)
            if receipt["endpoint"] != saved["target"]:
                raise ValueError(f"{arm}: reordered executable route has a different endpoint")
            receipts[arm] = receipt
    payload = {"graph": dag.payload(), "assignment": assignment, "receipts": receipts}
    publish(args.output / "replay.json.gz", payload, compressed=True)
    result = {
        "schema_version": "multi_site_route_probe_v1",
        "input": {"path": str(args.route.resolve()), "sha256": sha(args.route)},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "implementation_sha256": implementation_closure(Path(__file__).resolve()),
        "outputs_sha256": {"replay.json.gz": sha(args.output / "replay.json.gz")},
        "configuration": {
            "max_primitives": 24,
            "max_blocks": 5,
            "seed": None,
            "randomization": "deterministic schedules",
        },
        "counts": {
            "sources": 1,
            "schedules": 2,
            "primitive_edits_per_schedule": len(program.marks),
        },
        "dependencies": dag.dependencies,
        "conflicts": dag.conflicts,
        "serialization_edges": dag.serialization_edges,
        "arms": {
            arm: {
                key: receipt[key]
                for key in ("block_order", "endpoint", "actual_changes", "capacity_timeline")
            }
            for arm, receipt in receipts.items()
        },
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
        "evidence": "answer-known exact replay, not autonomous search or reference probability equivalence",
        "limitations": [
            "one inspected development route",
            "only two schedules tested",
            "footprints do not prove global guard independence",
        ],
    }
    publish(args.output / "result.json", result)
    print(json.dumps({key: result[key] for key in ("counts", "dependencies", "arms", "costs")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route", type=Path, default=Path("diagnostics/t4_whole_ring_plan/result.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
