"""Zero-oracle throughput assay on exact, already-scored T4 warm archives."""

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
from time import perf_counter, process_time
from unittest.mock import patch

import numpy as np
from rdkit import rdBase

from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer, ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.program_task import ProgramTask
from compose_v4.experiments.continuation_profile import publish_json
from compose_v4.experiments.parent_edit_cycles import load_contract, warm_archive
from compose_v4.rewrite import RewriteSystem
from tools.ivg_winner_paths import implementation_closure, sha

ROOT = Path(__file__).resolve().parents[1]


def compare_profiles(before_path, after_path, output, *, require_equivalent=True):
    """Compare retained exact outputs, not timings alone; no oracle evaluation."""
    if output.exists():
        raise ValueError("comparison output already exists")
    reports, inputs = [], {}
    for folder in (before_path, after_path):
        path = folder / "result.json"
        report = json.loads(path.read_text())
        if (
            identity({k: v for k, v in report.items() if k != "result_sha256"})
            != report["result_sha256"]
        ):
            raise ValueError(f"corrupt profile: {path}")
        reports.append(report)
        inputs[str(path)] = sha(path)
    old, new = ({r["unit"]: r for r in report["units"]} for report in reports)
    if set(old) != set(new):
        raise ValueError("profile unit census differs")
    comparisons = []
    for name in sorted(old):
        configurations = [
            asdict(ProgramSearchConfig(**r[name]["configuration"])) for r in (old, new)
        ]
        for config in configurations:
            config.pop("proposal_cache_entries")
            if not require_equivalent:
                config.pop("mutation_sampling")
                config.pop("composition_probability")
        if configurations[0] != configurations[1]:
            raise ValueError(f"unmatched proposal/work recipe: {name}")
        records = []
        for folder, summary in ((before_path, old[name]), (after_path, new[name])):
            path = folder / name / "batch.json"
            if sha(path) != summary["batch_sha256"]:
                raise ValueError(f"batch changed: {path}")
            inputs[str(path)] = sha(path)
            records.append(json.loads(path.read_text()))
        exact = [
            [
                {k: c[k] for k in ("endpoint", "program", "assignment", "source_state")}
                | {"states": c["trace"]["states"], "actions": c["trace"]["actions"]}
                for c in batch["candidates"]
            ]
            for batch in records
        ]
        attempted = [
            [
                {
                    k: r.get(k)
                    for k in ("attempt", "entry_id", "channel", "status", "endpoint", "reason")
                }
                for r in batch["attempts"]
            ]
            for batch in records
        ]
        equivalent = exact[0] == exact[1] and attempted[0] == attempted[1]
        if require_equivalent and not equivalent:
            raise ValueError(f"optimization changed exact candidate/attempt semantics: {name}")
        endpoint_sets = [{c["endpoint"] for c in batch["candidates"]} for batch in records]
        channel_counts = [
            dict(Counter(c["provenance"]["channel"] for c in batch["candidates"]))
            for batch in records
        ]
        comparisons.append(
            {
                "unit": name,
                "exact_candidate_and_attempt_equivalence": equivalent,
                "before_seconds": old[name]["proposal_wall_seconds"],
                "after_seconds": new[name]["proposal_wall_seconds"],
                "before_attempts": old[name]["attempts"],
                "after_attempts": new[name]["attempts"],
                "before_new_candidates": old[name]["eligible_new_candidates"],
                "after_new_candidates": new[name]["eligible_new_candidates"],
                "before_eligible_by_channel": channel_counts[0],
                "after_eligible_by_channel": channel_counts[1],
                "shared_candidate_endpoints": len(endpoint_sets[0] & endpoint_sets[1]),
                "new_only_candidate_endpoints": len(endpoint_sets[1] - endpoint_sets[0]),
            }
        )
    before_seconds = sum(r["before_seconds"] for r in comparisons)
    after_seconds = sum(r["after_seconds"] for r in comparisons)
    composition_endpoints = sum(
        row["after_eligible_by_channel"].get("program_composition", 0) for row in comparisons
    )
    composition_targets = sorted(
        {
            row["unit"].rsplit("_", 1)[0]
            for row in comparisons
            if row["after_eligible_by_channel"].get("program_composition", 0)
        }
    )
    time_ratio = after_seconds / before_seconds
    structural_gate = {
        "required_targets": ["braf_1", "jak2_1"],
        "composition_eligible_endpoints": composition_endpoints,
        "targets_with_composition_endpoint": composition_targets,
        "proposal_time_ratio": time_ratio,
        "passed": composition_endpoints >= 4
        and set(composition_targets) == {"braf_1", "jak2_1"}
        and time_ratio <= 1.5,
    }
    structural_gate["decision"] = (
        "ADVANCE_TO_LOCKED_DOCKING"
        if structural_gate["passed"]
        else "NO_DOCKING_INCONCLUSIVE_STRUCTURAL_GATE"
        if composition_endpoints
        else "NO_DOCKING_NEGATIVE_STRUCTURAL_GATE"
    )
    body = {
        "schema_version": "fast_program_comparison_v1",
        "inputs_sha256": inputs,
        "units": comparisons,
        "before_seconds": before_seconds,
        "after_seconds": after_seconds,
        "wall_time_reduction_fraction": 1 - after_seconds / before_seconds,
        "structural_gate": structural_gate,
        "require_equivalent": require_equivalent,
        "new_oracle_calls": 0,
        "software": reports[1]["software"],
        "hardware": reports[1]["hardware"],
        "analysis_sha256": sha(Path(__file__).resolve()),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "limitations": "Single timing per configuration/source/seed, local dirty-source development with recorded source hashes. Not a docking comparison or a speedup relative to remote mixed runs.",
    }
    publish_json(output, {**body, "result_sha256": identity(body)})
    print(json.dumps(comparisons, indent=2))


def run(args):
    if args.output.exists():
        raise ValueError("use a fresh output directory; previous results are immutable")
    contract = load_contract(ROOT)
    if rdBase.rdkitVersion != contract["rdkit"]:
        raise ValueError("saved exact programs require the frozen RDKit version")
    configs = [
        (
            cell,
            replace(
                ProgramSearchConfig.program_only_recipe(seed=seed),
                attempts_per_batch=args.attempts,
                candidates_per_batch=args.candidates,
                wall_seconds=args.seconds,
                proposal_cache_entries=args.cache_entries,
                mutation_sampling=args.mutation_sampling,
                composition_probability=args.composition_probability,
                max_composed_programs=args.max_composed_programs,
                composition_stop_probability=args.composition_stop_probability,
                composition_donor_trials=args.composition_donor_trials,
                composition_execution_trials=args.composition_execution_trials,
            ),
        )
        for cell in args.cells
        for seed in args.seeds
    ]
    material = [
        ROOT / "configs/parent_edit_cycles.json",
        *[ROOT / f"diagnostics/parent_edit_cycles/prepared/{c}_warm.json" for c in args.cells],
    ]
    recipe = {
        "schema_version": "fast_program_profile_recipe_v1",
        "configurations": [{"cell": cell, **asdict(config)} for cell, config in configs],
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in material},
        "implementation_sha256": implementation_closure(Path(__file__).resolve()),
        "seed_derivation": "explicit seed per cell; same seed reused across comparison modes",
        "grouping": "two inspected development sources; not held-out transfer",
        "new_oracle_calls": 0,
        "reference_law_calls": 0,
    }
    recipe["implementation_sha256"]["tools/ivg_winner_paths.py"] = sha(
        ROOT / "tools/ivg_winner_paths.py"
    )
    publish_json(args.output / "recipe.json", {**recipe, "recipe_sha256": identity(recipe)})
    results = []
    original = RewriteSystem.apply
    for cell, config in configs:
        calls = 0

        def measured(system, state, rule, action):
            nonlocal calls
            calls += 1
            return original(system, state, rule, action)

        with patch.object(RewriteSystem, "apply", measured):
            begin = perf_counter()
            saved = warm_archive(ROOT, cell, config, None)
            search = ProgramOptimizer.restore(saved)
            admission_seconds, admission_calls = perf_counter() - begin, calls
            domain = contract["t4_protocols"][cell]
            task = ProgramTask(cell, search.oracle_protocol, "t4", domain["original_seed"], 0.4)
            begin, cpu = perf_counter(), process_time()
            batch = search.propose_batch(task.endpoint_evaluator())
            seconds, cpu_seconds = perf_counter() - begin, process_time() - cpu
        unit = f"{cell}_{config.seed}"
        batch_sha = publish_json(args.output / unit / "batch.json", batch)
        snapshot_sha = publish_json(args.output / unit / "snapshot.json", search.snapshot())
        row = {
            "unit": unit,
            "configuration": asdict(config),
            "archive_programs": len(search.entries),
            "archive_endpoints": len({r["endpoint"] for r in search.observations.values()}),
            "attempts": len(batch["attempts"]),
            "status_counts": dict(Counter(r["status"] for r in batch["attempts"])),
            "failure_reasons": dict(
                Counter(r["reason"] for r in batch["attempts"] if "reason" in r)
            ),
            "eligible_new_candidates": len(batch["candidates"]),
            "eligible_by_channel": dict(
                Counter(c["provenance"]["channel"] for c in batch["candidates"])
            ),
            "proposal_wall_seconds": seconds,
            "proposal_cpu_seconds": cpu_seconds,
            "proposal_executor_calls": calls - admission_calls,
            "work_cache": search.work_cache.report(),
            "admission_seconds": admission_seconds,
            "admission_executor_calls": admission_calls,
            "batch_sha256": batch_sha,
            "snapshot_sha256": snapshot_sha,
        }
        results.append(row)
        publish_json(args.output / "progress.json", results)
        print(json.dumps(row), flush=True)
    result = {
        "schema_version": "fast_program_profile_v1",
        "recipe_sha256": identity(recipe),
        "units": results,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {
            "device": "cpu",
            "machine": platform.machine(),
            "workers": 1,
            "precision": "float64 probabilities, integer executor state",
            "peak_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_unit": "bytes" if platform.system() == "Darwin" else "KiB",
        },
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "new_oracle_calls": 0,
        "new_training_calls": 0,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "limitations": [
            "Unscored development candidates, not improved docking scores.",
            "Qualified reference assets are remote; historical mixed timing is unmatched hardware/streams, not a speedup estimate.",
            "Time cap checked between attempts; primitive/binding caps still apply.",
        ],
    }
    publish_json(args.output / "result.json", {**result, "result_sha256": identity(result)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--proposal-change", action="store_true")
    parser.add_argument(
        "--cells", nargs="+", choices=("braf_1", "jak2_1"), default=["braf_1", "jak2_1"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260921, 20260922])
    parser.add_argument("--attempts", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--cache-entries", type=int, default=0)
    parser.add_argument("--mutation-sampling", choices=("random", "untried"), default="random")
    parser.add_argument("--composition-probability", type=float, default=0.0)
    parser.add_argument("--max-composed-programs", type=int, default=2)
    parser.add_argument("--composition-stop-probability", type=float, default=0.5)
    parser.add_argument("--composition-donor-trials", type=int, default=8)
    parser.add_argument("--composition-execution-trials", type=int, default=16)
    args = parser.parse_args()
    if args.compare:
        compare_profiles(*args.compare, args.output, require_equivalent=not args.proposal_change)
    else:
        run(args)


if __name__ == "__main__":
    main()
