"""Prepare first query batches on a fixed four-cell development curriculum.

Reuse exact source tensors and the same shared PARP1 program logic. The recipient
winner routes and scores are not passed to the proposal. No docking calls.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import extract_program
from compose_v4.control.program_transfer import initial_program_batch, shared_program_library
from compose_v4.experiments.continuation_profile import ExecutorMeter, publish_json, verify_file
from compose_v4.experiments.t4_winner_refinement import property_scorer
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.adaptive_program_search import summarize
from tools.ivg_winner_paths import implementation_closure, sha

ROOT = Path(__file__).resolve().parents[1]
CELLS = (("jak2", 1), ("fa7", 0), ("braf", 1), ("5ht1b", 0))


def add_verified_winner_programs(directory):
    """Mine complete supported routes into one shared library, not per-task banks."""
    records, exclusions, inputs, seen = [], [], {}, set()
    for path in sorted(directory.glob("*.json.gz")):
        row = json.loads(gzip.decompress(path.read_bytes()))
        if identity(row["payload"]) != row["payload_sha256"]:
            raise ValueError(f"corrupt winner-route input: {path}")
        inputs[str(path)] = sha(path)
        pair_id = row["pair"]["pair_id"]
        if pair_id in seen:
            exclusions.append({"path": str(path), "reason": "duplicate_pair_receipt"})
            continue
        seen.add(pair_id)
        saved = row["payload"]["path"]
        if saved["status"] != "witness_found":
            exclusions.append({"path": str(path), "reason": saved["status"]})
            continue
        if len(saved["actions"]) > 32:
            exclusions.append(
                {
                    "path": str(path),
                    "reason": "route_exceeds_declared_32_primitive_recipe",
                    "primitives": len(saved["actions"]),
                }
            )
            continue
        source = decode_state(saved["source_state"])
        stage = {
            "name": "compiled_complete_transformation",
            "actions": saved["actions"],
            "states": saved["states"],
            "endpoint": canonical_state_key(decode_state(saved["states"][-1])),
        }
        program, _ = extract_program(source, [stage])
        for target in sorted({r["target"] for r in row["pair"]["references"]}):
            records.append(
                {
                    "source_group": identity(
                        {"original_benchmark_seed": row["pair"]["source"], "target": target}
                    ),
                    "program": program.payload(),
                    "origin": str(path),
                }
            )
    return records, {
        "input_sha256": inputs,
        "exclusions": exclusions,
        "admitted_records": len(records),
        "label_use": "winner selection informs structural supervision; external numeric scores are not new-target production labels",
    }


def sources_from_saved_states(directory, registry):
    """Read only recipient identity and its exact initial tensor from saved receipts."""
    # Registry idx is global (0..14); public result source_idx is within target.
    expected = {
        (target, local): row["smiles"]
        for target in sorted({r["target"] for r in registry})
        for local, row in enumerate(r for r in registry if r["target"] == target)
    }
    sources = {}
    for path in sorted(directory.glob("*.json.gz")):
        row = json.loads(gzip.decompress(path.read_bytes()))
        matching = {(r["target"], r["source_idx"]) for r in row["pair"]["references"]} & set(CELLS)
        for cell in sorted(matching - set(sources)):
            exact = row["payload"]["path"].get("source_state")
            if exact is None:
                continue
            if identity(row["payload"]) != row["payload_sha256"]:
                raise ValueError(f"corrupt saved source receipt: {path}")
            if row["pair"]["source"] != expected[cell]:
                raise ValueError(f"saved source identity differs from seed registry: {cell}")
            sources[cell] = {
                "state": exact,
                "path": str(path),
                "sha256": sha(path),
                "smiles": expected[cell],
            }
        if len(sources) == len(CELLS):
            break
    if set(sources) != set(CELLS):
        raise ValueError(f"missing exact initial tensors for: {set(CELLS) - set(sources)}")
    return sources


def run(args):
    if args.output.exists():
        raise ValueError("transfer output already exists; use a new immutable round")
    started = perf_counter()
    bank = json.loads(args.replay.read_text())
    if rdBase.rdkitVersion != bank["oracle_protocol"]["required_rdkit"]:
        raise ValueError("transfer requires the original pinned RDKit version")
    verify_file(Path(bank["inputs"]["review_path"]), bank["inputs"]["review_sha256"])
    for path, digest in bank["inputs"]["pool_sha256"].items():
        verify_file(ROOT / path, digest)
    registry_path = ROOT / "docs/GENMOL_T4_SEEDS.json"
    expected_seed_hash = bank["oracle_protocol"]["task"]["expected_input_sha256"]["seed_manifest"]
    verify_file(registry_path, expected_seed_hash)
    sources = sources_from_saved_states(
        ROOT / "diagnostics/ivg_winner_paths/pairs", json.loads(registry_path.read_text())
    )
    extra, winner_evidence = ([], None)
    if args.include_verified_winners:
        extra, winner_evidence = add_verified_winner_programs(
            ROOT / "diagnostics/ivg_winner_paths/pairs"
        )
    entries = shared_program_library([*bank["records"], *extra])
    library = [{"program": e.program.payload(), "source_groups": e.source_groups} for e in entries]
    library_hash = publish_json(args.output / "shared_library.json", library)
    config = ProgramSearchConfig(
        seed=20260913,
        attempts_per_batch=128,
        candidates_per_batch=8,
        max_primitives=32,
        max_blocks=8,
        wall_seconds=60,
        require_broad_runtime=False,
    )
    summaries, meter = [], ExecutorMeter(None)
    with meter.instrument():
        for target, index in CELLS:
            saved = sources[(target, index)]
            source = decode_state(saved["state"])
            # This is a pending oracle identity, not a claim to have measured the
            # recipient with PARP1 receptor/preparation settings.
            protocol = {
                "schema_version": "unscored_t4_recipient_v1",
                "target": target,
                "source_idx": index,
                "original_seed": saved["smiles"],
                "delta": 0.4,
                "production_docking_manifest": None,
            }
            source_group = identity({"original_benchmark_seed": saved["smiles"], "target": target})
            before = meter.calls
            batch = initial_program_batch(
                source,
                entries,
                config,
                source_group=source_group,
                oracle_protocol=identity(protocol),
                eligibility=property_scorer(saved["smiles"]),
            )
            name = f"{target}_{index}"
            batch_hash = publish_json(args.output / f"{name}.json", batch)
            summary = {
                "cell": name,
                "source": saved,
                "protocol": protocol,
                "batch_sha256": batch_hash,
                "counts": summarize(batch),
                "executor_calls": meter.calls - before,
            }
            summaries.append(summary)
            print(json.dumps({"cell": name, "counts": summary["counts"]}), flush=True)
    closure = implementation_closure(Path(__file__).resolve())
    for path in ("tools/adaptive_program_search.py", "tools/ivg_winner_paths.py"):
        closure[path] = sha(ROOT / path)
    result = {
        "schema_version": "shared_t4_program_preparation_v1",
        "input": {"path": str(args.replay), "sha256": sha(args.replay)},
        "seed_registry_sha256": sha(registry_path),
        "library_sha256": library_hash,
        "winner_program_evidence": winner_evidence,
        "programs": len(entries),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "implementation_sha256": closure,
        "configuration": config.__dict__,
        "cells": summaries,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {"device": "cpu", "machine": platform.machine(), "workers": 1},
        "costs": {
            "seconds": perf_counter() - started,
            "executor_calls": meter.calls,
            "new_oracle_calls": 0,
        },
        "split": "four previously inspected recipient development cells; all-task winner-derived shared library"
        if extra
        else "four previously inspected recipient development cells; PARP1-derived shared library",
        "decision": "The architecture is retained; eligible locked endpoints are ready for an identified target-specific score protocol.",
        "limitations": [
            "No new docking results or adaptation gain is established.",
            "Broad draws fail explicitly in local mode.",
            "A paid driver must replace pending oracle identities with a frozen receptor/preparation manifest before scoring.",
        ],
    }
    publish_json(args.output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay",
        type=Path,
        default=ROOT / "diagnostics/t4_program_pool/attempt_1/scored_program_replay.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-verified-winners", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
