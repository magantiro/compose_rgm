"""Prepare or advance a locked adaptive program batch using existing paid labels.

This local developer entry point makes no oracle or network calls. The required
local-mode flag records unavailable learned-reference draws as failed attempts;
it does not reallocate them to the program channel or emulate the reference.
Production callers instantiate ProgramOptimizer with their authenticated
MolecularHierarchy. Supplied outcome files must contain genuine oracle receipts.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer, ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import ExecutorMeter, publish_json, verify_file
from compose_v4.experiments.t4_winner_refinement import property_scorer
from tools.ivg_winner_paths import implementation_closure, sha

ROOT = Path(__file__).resolve().parents[1]


def initialize(bank, config):
    if bank["schema_version"] != "scored_attachment_program_replay_v1":
        raise ValueError("unsupported measured-program replay schema")
    groups = {r["source_group"] for r in bank["records"]}
    if len(groups) != 1:
        raise ValueError("select exactly one identified source group before initialization")
    search = ProgramOptimizer(
        config, source_group=next(iter(groups)), oracle_protocol=identity(bank["oracle_protocol"])
    )
    for row in bank["records"]:
        record = {
            **row,
            "trace": row["executed_trace"],
            "oracle_protocol": search.oracle_protocol,
        }
        # All decompositions of the same paid endpoint bind the same observation.
        receipt = identity(
            {
                "replay_inputs": bank["inputs"],
                "protocol": search.oracle_protocol,
                "endpoint": row["endpoint"],
                "oracle_label": row["oracle_label"],
            }
        )
        search.add_measured_program(
            record, receipt_id=receipt, score=row["oracle_label"]["docking_score"]
        )
    return search


def summarize(batch):
    attempts = batch["attempts"]
    return {
        "attempts": len(attempts),
        "status_counts": dict(Counter(r["status"] for r in attempts)),
        "channel_counts": dict(Counter(r["channel"] for r in attempts)),
        "failure_reasons": dict(
            Counter(r["reason"] for r in attempts if r["status"] == "execution_rejected")
        ),
        "new_eligible_endpoints": len(batch["candidates"]),
        "completed_unique_endpoints": len({r["endpoint"] for r in attempts if "endpoint" in r}),
        "new_eligible_by_channel": dict(
            Counter(c["provenance"]["channel"] for c in batch["candidates"])
        ),
        "completed_mutation_types": dict(
            Counter(m["kind"] for r in attempts for m in r.get("metadata", {}).get("mutations", ()))
        ),
        "actual_changed_sites": dict(
            Counter(
                str(c["trace"]["actual_changes"]["changed_site_count"]) for c in batch["candidates"]
            )
        ),
        "proposal_seconds": batch["proposal_seconds"],
        "seconds_per_new_eligible_endpoint": batch["proposal_seconds"] / len(batch["candidates"])
        if batch["candidates"]
        else None,
    }


def run(args):
    if args.output.exists():
        raise ValueError("output already exists; use a new batch directory, not an overwrite")
    if not args.local_program_development:
        raise ValueError(
            "local mode must be explicit; this command cannot load remote reference assets"
        )
    if bool(args.snapshot) != bool(args.outcomes):
        raise ValueError(
            "resume requires both the prior snapshot and complete outcome receipt file"
        )
    if args.snapshot and args.static:
        raise ValueError("resume uses its frozen configuration; --static is initialization-only")
    started = perf_counter()
    bank = json.loads(args.replay.read_text())
    if rdBase.rdkitVersion != bank["oracle_protocol"]["required_rdkit"]:
        raise ValueError("local chemistry differs from the paid replay protocol")
    if bank["oracle_protocol"]["task"]["delta"] != 0.4:
        raise ValueError("this pinned local T4 evaluator is explicitly the delta=0.4 recipe")
    # The scored conversion and its exact-state pools remain immutable inputs.
    verify_file(Path(bank["inputs"]["review_path"]), bank["inputs"]["review_sha256"])
    for path, digest in bank["inputs"]["pool_sha256"].items():
        verify_file(ROOT / path, digest)
    closure = implementation_closure(Path(__file__).resolve())
    closure["tools/ivg_winner_paths.py"] = sha(ROOT / "tools/ivg_winner_paths.py")
    software = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rdkit": rdBase.rdkitVersion,
    }
    context = {
        "input": {"path": str(args.replay.resolve()), "sha256": sha(args.replay)},
        "implementation_sha256": closure,
        "software": software,
        "oracle_protocol": bank["oracle_protocol"],
    }
    meter = ExecutorMeter(None)
    with meter.instrument():
        if args.snapshot:
            saved = json.loads(args.snapshot.read_text())
            if saved["context"] != context:
                raise ValueError(
                    "resume changes replay, implementation, software or oracle protocol"
                )
            search = ProgramOptimizer.restore(saved["optimizer"])
            labels = json.loads(args.outcomes.read_text())
            search.observe_batch(labels["batch_id"], labels["outcomes"])
        else:
            config = ProgramSearchConfig(
                seed=args.seed,
                adaptive=not args.static,
                attempts_per_batch=args.attempts,
                candidates_per_batch=args.candidates,
                wall_seconds=args.wall_seconds,
                max_primitives=args.max_primitives,
                max_blocks=args.max_blocks,
                require_broad_runtime=False,
            )
            search = initialize(bank, config)
        prepare_seconds, prepare_calls = perf_counter() - started, meter.calls
        batch = search.propose_batch(property_scorer(bank["oracle_protocol"]["task"]["smiles"]))
    snapshot = {"context": context, "optimizer": search.snapshot()}
    snapshot_sha = publish_json(args.output / "snapshot.json", snapshot)
    batch_sha = publish_json(args.output / "batch.json", batch)
    result = {
        "schema_version": "adaptive_program_local_batch_v1",
        **context,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "configuration": asdict(search.config),
        "seed_derivation": "explicit seed; exact numpy RNG state retained across batches",
        "counts": summarize(batch),
        "archive": {
            "programs": len(search.entries),
            "measured_endpoints": len({r["endpoint"] for r in search.observations.values()}),
            "observations": len(search.observations),
        },
        "outputs": {"snapshot.json": snapshot_sha, "batch.json": batch_sha},
        "resume_inputs": {str(p.resolve()): sha(p) for p in (args.snapshot, args.outcomes) if p},
        "costs": {
            "prepare_seconds": prepare_seconds,
            "prepare_executor_calls": prepare_calls,
            "proposal_executor_calls": meter.calls - prepare_calls,
            "seconds": perf_counter() - started,
            "new_oracle_calls": 0,
            "new_model_training_calls": 0,
        },
        "hardware": {
            "machine": platform.machine(),
            "device": "cpu",
            "workers": 1,
            "precision": "numpy float64 probabilities; executor integer state",
            "peak_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_unit": "bytes" if platform.system() == "Darwin" else "KiB",
        },
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "evidence_regime": "winner-informed one-source development; structural yield, not new docking performance",
        "split": "all original-source derivatives remain inspected development",
        "limitations": [
            "No oracle called; new endpoints are unscored and do not establish improvement.",
            "Learned-reference assets are remote; broad draws fail explicitly in this local mode.",
            "Initial static/adaptive distributions are identical until new scores arrive.",
            "Wall limit is checked between bounded program attempts, not interrupting an executor call.",
        ],
    }
    publish_json(args.output / "result.json", result)
    print(json.dumps({"counts": result["counts"], "costs": result["costs"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay",
        type=Path,
        default=ROOT / "diagnostics/t4_program_pool/attempt_1/scored_program_replay.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-program-development", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--attempts", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--wall-seconds", type=float, default=60)
    parser.add_argument("--max-primitives", type=int, default=32)
    parser.add_argument("--max-blocks", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
