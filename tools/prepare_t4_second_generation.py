"""Prepare measured-archive descendants and a matched score-blind control, no queries."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer
from compose_v4.control.edit_program_graph import compile_program_graph, program_size_profile
from compose_v4.experiments.continuation_profile import ExecutorMeter, publish_json, verify_file
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_winner_refinement import property_scorer
from tools.adaptive_program_search import summarize
from tools.ivg_winner_paths import implementation_closure, sha

ROOT = Path(__file__).resolve().parents[1]
PAID = ROOT / "diagnostics/t4_program_curriculum/attempt_1"
CELLS = ("jak2_1", "fa7_0", "braf_1", "5ht1b_0")
ARMS = ("score_rank", "score_blind")


def allocation(search):
    keys, weights = search.selection()
    rows = []
    for key, weight in zip(keys, weights, strict=True):
        entry = search.entries[key]
        score = float(
            np.mean(
                [
                    r["score"]
                    for r in search.observations.values()
                    if r["endpoint"] == entry["endpoint"]
                ]
            )
        )
        rows.append(
            {
                "entry_id": key,
                "endpoint": entry["endpoint"],
                "measured_score": score,
                "probability": float(weight),
                "program_size": program_size_profile(
                    compile_program_graph(search._program(entry)),
                    search._source(entry).n_real_atoms,
                ),
            }
        )
    return {
        "parents": rows,
        "expected_measured_parent_score": sum(r["probability"] * r["measured_score"] for r in rows),
        "best_parent_mass": sum(
            r["probability"]
            for r in rows
            if r["measured_score"] == min(x["measured_score"] for x in rows)
        ),
    }


def prepare(output):
    if output.exists():
        raise ValueError("second-generation output exists; preserve its candidate locks")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("second generation requires the paid archive's RDKit 2024.03.5")
    review = unseal(PAID / "review.json")
    archives, input_hashes = {}, {str(PAID / "review.json"): sha(PAID / "review.json")}
    for cell in CELLS:
        name = f"{cell}_archive.json"
        path = PAID / name
        verify_file(path, review["outputs_sha256"][name])
        archives[cell] = json.loads(path.read_text())
        input_hashes[str(path)] = sha(path)
    closure = implementation_closure(Path(__file__).resolve())
    for name in ("tools/adaptive_program_search.py", "tools/ivg_winner_paths.py"):
        closure[name] = sha(ROOT / name)
    context = {
        "schema_version": "t4_second_generation_preparation_v1",
        "inputs_sha256": input_hashes,
        "implementation_sha256": closure,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "configuration": {
            "cells": CELLS,
            "arms": ARMS,
            "seed": 20260914,
            "attempts_per_arm_cell": 128,
            "candidates_per_arm_cell": 8,
            "wall_seconds_per_arm_cell": 60,
            "new_oracle_calls": 0,
            "source": "same eight genuinely measured programs per cell, no new winner mining",
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "hardware": {"machine": platform.machine(), "device": "cpu", "omp_threads": 1},
        "split": "same four inspected development cells; no held-out claim",
        "seed_derivation": "new explicit seed shared across paired arms; saved predecessor RNG is retained as provenance, not silently resumed",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    contract_hash = seal(output / "contract.json", context)
    started, cells = perf_counter(), []
    for cell in CELLS:
        archive = archives[cell]
        records = []
        endpoints = {}
        for arm in ARMS:
            search = ProgramOptimizer.restore(archive["optimizer"])
            if search.pending is not None or len(search.observations) != 8:
                raise ValueError(f"{cell}: expected completed eight-observation predecessor")
            # Explicit derived experiment, not mutation of a pending/resumed run.
            search.config = replace(search.config, parent_allocation=arm, seed=20260914)
            search.rng = np.random.default_rng(search.config.seed)
            before = allocation(search)
            meter = ExecutorMeter(None)
            with meter.instrument():
                batch = search.propose_batch(
                    property_scorer(archive["oracle_domain"]["original_seed"])
                )
            name = f"{cell}_{arm}"
            batch_hash = publish_json(output / f"{name}_batch.json", batch)
            snapshot_hash = publish_json(
                output / f"{name}_snapshot.json",
                {
                    "contract_sha256": contract_hash,
                    "predecessor_snapshot_id": archive["optimizer"]["snapshot_id"],
                    "configuration_transition": {
                        "new_seed": search.config.seed,
                        "parent_allocation": arm,
                        "exploration_floor_after_exhaustion": True,
                    },
                    "optimizer": search.snapshot(),
                },
            )
            row = {
                "cell": cell,
                "arm": arm,
                "configuration": asdict(search.config),
                "allocation_before_proposal": before,
                "summary": summarize(batch),
                "executor_calls": meter.calls,
                "batch_path": f"{name}_batch.json",
                "batch_sha256": batch_hash,
                "snapshot_path": f"{name}_snapshot.json",
                "snapshot_sha256": snapshot_hash,
                "new_endpoint_size_changes_from_measured_parent": dict(
                    Counter(
                        str(c["provenance"]["program_size"]["delta_from_measured_parent"])
                        for c in batch["candidates"]
                    )
                ),
            }
            endpoints[arm] = {c["endpoint"] for c in batch["candidates"]}
            records.append(row)
            print(json.dumps({"cell": cell, "arm": arm, **row["summary"]}), flush=True)
        cells.append(
            {
                "cell": cell,
                "arms": records,
                "cross_arm_unique_candidates": len(set.union(*endpoints.values())),
                "cross_arm_shared_candidates": len(set.intersection(*endpoints.values())),
            }
        )
        publish_json(
            output / "progress.json",
            {"complete_cells": [c["cell"] for c in cells], "total_cells": 4},
        )
    result = {
        **context,
        "contract_sha256": contract_hash,
        "cells": cells,
        "costs": {
            "seconds": perf_counter() - started,
            "executor_calls": sum(a["executor_calls"] for c in cells for a in c["arms"]),
            "new_oracle_calls": 0,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "maximum_distinct_new_dockings_for_both_arms": sum(
            c["cross_arm_unique_candidates"] for c in cells
        ),
        "decision": "candidate and feedback-allocation evidence only; no second-generation docking quality observed",
        "limitations": [
            "no remote broad callback in local preparation; unavailable draws retained",
            "one paired seed per cell",
            "size is structural accounting, not a docking predictor or shrinkage reward",
        ],
    }
    seal(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "diagnostics/t4_second_generation/attempt_1"
    )
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps(result["costs"], indent=2))


if __name__ == "__main__":
    main()
