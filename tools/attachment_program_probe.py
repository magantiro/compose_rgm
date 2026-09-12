"""Prepare transferable programs and run a paired zero-oracle attachment assay.

Reuses exact inverse-derived traces and their pre-existing source roles. Public
candidate rows are inventoried with immutable upstream identities, never turned
into trajectories or treated as interchangeable production docking labels.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.edit_program import (
    EditProgram,
    ProgramExecutionError,
    execute_bound_program,
    extract_program,
)
from compose_v4.control.edit_program_policy import ProgramEntry, propose_programs
from compose_v4.experiments.continuation_profile import ExecutorMeter
from compose_v4.rewrite.trace_shard import decode_state
from tools.ivg_t4_census import REVISION, source_inventory
from tools.ivg_winner_paths import digest, implementation_closure, publish, sha

ROOT = Path(__file__).resolve().parents[1]


def load_verified(path, expected=None):
    if expected is not None and sha(path) != expected:
        raise ValueError(f"{path}: input hash mismatch")
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def prepare(directory, output):
    result = load_verified(directory / "result.json")
    rows = load_verified(directory / "demonstrations.json.gz", result["demonstrations_sha256"])[
        "rows"
    ]
    roles = load_verified(directory / "roles.json", result["roles_sha256"])
    sources = defaultdict(set)
    for row in rows:
        sources[row["target_used_for_preparation_only"]].add((row["source_id"], row["role"]))
    bank, examples, exclusions, inputs = (
        {},
        [],
        [],
        {
            str((directory / name).resolve()): sha(directory / name)
            for name in ("result.json", "demonstrations.json.gz", "roles.json")
        },
    )
    for unit in result["units"]:
        path = directory / unit["path"]
        payload = load_verified(path, unit["sha256"])
        if digest(payload["payload"]) != payload["payload_sha256"]:
            raise ValueError(f"{path}: corrupt payload")
        inputs[str(path.resolve())] = unit["sha256"]
        target = unit["target"]
        role = roles["winner_roles"][target]
        groups = tuple(
            sorted(source for source, group_role in sources[target] if group_role == role)
        )
        if not groups:
            exclusions.append({"target": target, "reason": "no_eligible_source_roles"})
            continue
        for index, example in enumerate(payload["payload"]["examples"]):
            stages = [{"name": example["option"], **example["forward"]}]
            if example["redecoration"]["actions"]:
                stages.append({"name": "terminal_restoration", **example["redecoration"]})
            program, assignment = extract_program(decode_state(example["precursor_state"]), stages)
            if len(program.marks) > 24 or len(program.blocks) > 5:
                exclusions.append({"target": target, "example": index, "reason": "program_budget"})
                continue
            _, replay = execute_bound_program(
                decode_state(example["precursor_state"]), program, assignment
            )
            if replay["endpoint"] != target:
                raise ValueError("extracted program lost the demonstrated endpoint")
            key = program.program_id
            record = {
                "example_id": f"{payload['identity']}:{index}",
                "program_id": key,
                "program": program.payload(),
                "source_groups": groups,
                "role": role,
                "source_state": example["precursor_state"],
                "source_smiles": example["precursor"],
                "target_for_evaluation_only": target,
                "observed_assignment": assignment,
            }
            examples.append(record)
            if role == "train":
                stored = bank.setdefault(key, {"program": program.payload(), "sources": set()})
                stored["sources"].update(groups)
    train_sources = {source for row in bank.values() for source in row["sources"]}
    excluded_sources = {s for r in examples if r["role"] != "train" for s in r["source_groups"]}
    if train_sources & excluded_sources:
        raise ValueError("source roles leak across the program bank")
    held_precursors = {r["source_smiles"] for r in examples if r["role"] != "train"}
    # Existing upstream preprocessing exclusions are retained; exact precursor
    # collisions require withholding the training example rather than relabeling it.
    overlap = [
        r for r in examples if r["role"] == "train" and r["source_smiles"] in held_precursors
    ]
    if overlap:
        raise ValueError(
            "exact precursor overlap requires an explicit exclusion before preparation"
        )
    payload = {
        "schema_version": "attachment_program_bank_v1",
        "programs": [
            {"program_id": key, "program": row["program"], "source_groups": sorted(row["sources"])}
            for key, row in sorted(bank.items())
        ],
        "examples": examples,
        "exclusions": exclusions,
        "inputs_sha256": inputs,
        "upstream": result["access_basis_and_upstream"],
        "roles": {
            "train": sorted(train_sources),
            "excluded_source_diagnostic": sorted(excluded_sources),
        },
        "evidence": "winner-informed structural programs; all examples previously inspected",
    }
    publish(output / "prepared.json.gz", payload, compressed=True)
    return payload


def public_inventory(output):
    cache = ROOT / ".cache/ivg_t4" / REVISION
    manifest = source_inventory(cache, download=False)
    census = []
    for asset in manifest:
        if not asset["path"].endswith(".csv"):
            continue
        with (cache / asset["path"]).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        magnitude = [float(r["docking score"]) for r in rows]
        if not np.isfinite(magnitude).all():
            raise ValueError(f"{asset['path']}: nonfinite external observation")
        census.append(
            {
                **asset,
                "rows": len(rows),
                "distinct_reported_smiles": len({r["smiles"] for r in rows}),
                "run_seeds": sorted({int(r["seed"]) for r in rows}),
                "score_domain": "external_quickvina_positive_magnitude; not COMPOSE labels",
            }
        )
    payload = {
        "schema_version": "public_scored_program_source_census_v1",
        "revision": REVISION,
        "sources": manifest,
        "cells": census,
        "rows": sum(r["rows"] for r in census),
    }
    publish(output / "public_candidate_census.json", payload)
    return payload


def run(args):
    if (args.output / "result.json").exists():
        raise ValueError("completed probe exists; inspect/reuse it, do not overwrite")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("requires pinned chemistry RDKit 2024.03.5")
    started = perf_counter()
    meter = ExecutorMeter(None)
    with meter.instrument():
        prepared = prepare(args.input, args.output)
        entries = tuple(
            ProgramEntry(EditProgram.from_payload(r["program"]), tuple(r["source_groups"]))
            for r in prepared["programs"]
        )
        parents = {}
        for row in prepared["examples"]:
            if row["role"] != "train":
                parents.setdefault(row["source_smiles"], row)
        output_rows, summaries = [], defaultdict(Counter)
        for parent_index, (_, example) in enumerate(sorted(parents.items())):
            graph = decode_state(example["source_state"])
            # Cache actual executor outcomes shared by the two arms; matching
            # results never depend on a docking or reconstruction target.
            cache = {}
            for arm in ("unconditioned", "contextual"):
                selection = propose_programs(
                    graph,
                    entries,
                    seed=args.seed + parent_index,
                    count=args.draws,
                    contextual=arm == "contextual",
                )
                products, attempts = {}, []
                for draw in selection["draws"]:
                    index, assignment = draw["program_index"], tuple(draw["assignment"])
                    key = (index, assignment)
                    if key not in cache:
                        try:
                            _, receipt = execute_bound_program(
                                graph, entries[index].program, assignment
                            )
                            cache[key] = {"status": "complete", "receipt": receipt}
                        except ProgramExecutionError as error:
                            cache[key] = {
                                "status": "executor_rejected",
                                "reason": str(error),
                                "prefix": error.receipt,
                            }
                    attempt = {**draw, **cache[key]}
                    attempts.append(attempt)
                    summaries[arm][attempt["status"]] += 1
                    if attempt["status"] == "complete":
                        products.setdefault(attempt["receipt"]["endpoint"], attempt["receipt"])
                summaries[arm]["parents"] += 1
                summaries[arm]["unique_products_sum"] += len(products)
                summaries[arm]["exact_reconstruction_parents"] += int(
                    example["target_for_evaluation_only"] in products
                )
                output_rows.append(
                    {
                        "parent_id": example["example_id"],
                        "arm": arm,
                        "selection": selection,
                        "attempts": attempts,
                        "unique_endpoints": sorted(products),
                        "target_for_evaluation_only": example["target_for_evaluation_only"],
                    }
                )
            if (parent_index + 1) % 4 == 0:
                print(
                    json.dumps(
                        {
                            "parents_completed": parent_index + 1,
                            "total": len(parents),
                            "seconds": perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    publish(args.output / "proposals.json.gz", {"rows": output_rows}, compressed=True)
    external = public_inventory(args.output)
    counts = Counter(r["role"] for r in prepared["examples"])
    body = {
        "schema_version": "attachment_program_probe_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        ),
        "configuration": {
            "seed": args.seed,
            "draws_per_arm_parent": args.draws,
            "max_primitives": 24,
            "max_blocks": 5,
            "max_bindings": 32,
            "max_visits": 4096,
        },
        "implementation_sha256": implementation_closure(Path(__file__).resolve()),
        "inputs_sha256": prepared["inputs_sha256"],
        "outputs_sha256": {
            name: sha(args.output / name)
            for name in ("prepared.json.gz", "proposals.json.gz", "public_candidate_census.json")
        },
        "counts": {
            "programs": len(entries),
            "examples_by_role": dict(counts),
            "excluded_sources": len(prepared["roles"]["excluded_source_diagnostic"]),
            "parents": len(parents),
        },
        "arms": {arm: dict(values) for arm, values in summaries.items()},
        "costs": {
            "seconds": perf_counter() - started,
            "executor_calls": meter.calls,
            "new_oracle_calls": 0,
            "reference_law_evaluations": 0,
            "external_rows_inventoried": external["rows"],
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "device": "cpu",
        },
        "evidence": "retrospective cross-source program transfer, not autonomous docking improvement",
        "limitations": [
            "capped binding pool",
            "context heuristic not learned pointer policy",
            "external scores inventoried, not used for guidance",
            "existing broad reference branch unchanged; isolated program-channel comparison",
        ],
    }
    publish(args.output / "result.json", body)
    print(
        json.dumps({"counts": body["counts"], "arms": body["arms"], "costs": body["costs"]}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "diagnostics/inverse_ring_proposals/attempt_1"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--draws", type=int, default=32)
    run(parser.parse_args())
