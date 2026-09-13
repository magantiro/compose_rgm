"""Compare mutated cold starts with bounded direct program retrieval on T4."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_policy import ProgramEntry
from compose_v4.control.program_transfer import initial_program_batch
from compose_v4.experiments.continuation_profile import ExecutorMeter, sha256_file
from compose_v4.experiments.t4_frozen_program_benchmark import strict_endpoint_scorer
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
CENSUS = ROOT / "diagnostics/ivg_t4_census/census.json"
IMPLEMENTATION = (
    "src/compose_v4/control/adaptive_program_optimizer.py",
    "src/compose_v4/control/edit_program_policy.py",
    "src/compose_v4/control/program_transfer.py",
    "tools/t4_program_retrieval_probe.py",
)


def _winner_index(census):
    winners = {}
    for row in census["cells"]:
        if row["delta"] != 0.4:
            continue
        cell = f"{row['target']}_{row['source_idx']}"
        values = {}
        for run in row["runs"]:
            for winner in run["winners"]:
                endpoint = winner["canonical_smiles"]
                values[endpoint] = min(
                    values.get(endpoint, float("inf")), run["reported_docking_score"]
                )
        winners[cell] = values
    if len(winners) != 15:
        raise ValueError("public census does not contain all fifteen delta=0.4 cells")
    return winners


def _summary(batch, known):
    endpoints = {row["endpoint"] for row in batch["candidates"]}
    recovered = sorted(
        endpoints & set(known), key=lambda endpoint: (known[endpoint], endpoint)
    )
    return {
        "batch_id": batch["batch_id"],
        "candidates": len(endpoints),
        "attempts": len(batch["attempts"]),
        "proposal_seconds": batch["proposal_seconds"],
        "candidate_channels": dict(
            sorted(
                Counter(
                    row["provenance"]["channel"] for row in batch["candidates"]
                ).items()
            )
        ),
        "attempt_status": dict(
            sorted(Counter(row["status"] for row in batch["attempts"]).items())
        ),
        "known_winners_recovered": [
            {"endpoint": endpoint, "external_reported_score": known[endpoint]}
            for endpoint in recovered
        ],
        "best_recovered_external_score": None if not recovered else known[recovered[0]],
    }


def run(output: Path):
    if output.exists():
        raise ValueError(
            f"output already exists; use a new immutable attempt: {output}"
        )
    contract = unseal(CONTRACT)
    if contract["schema_version"] != "t4_frozen_program_benchmark_v2":
        raise ValueError("unexpected T4 source contract")
    library_path = ROOT / contract["library_path"]
    library_rows = json.loads(library_path.read_text())
    entries = tuple(
        ProgramEntry(
            EditProgram.from_payload(row["program"]), tuple(row["source_groups"])
        )
        for row in library_rows
    )
    if len(entries) != contract["library_programs"]:
        raise ValueError("shared program library size changed")
    winners = _winner_index(json.loads(CENSUS.read_text()))
    config_payload = dict(contract["controller"])
    config_payload["channel_probabilities"] = tuple(
        config_payload["channel_probabilities"]
    )
    control = ProgramSearchConfig(**config_payload)
    retrieval = replace(control, cold_start_retrieval_candidates=8)
    output.mkdir(parents=True)
    began, rows = perf_counter(), []
    meter = ExecutorMeter(None)
    with meter.instrument():
        for cell in sorted(contract["cells"]):
            saved = contract["cells"][cell]
            source = decode_state(saved["source_state"])
            unit = next(
                row
                for row in contract["units"]
                if row["cell"] == cell and row["replicate"] == 0
            )
            common = {
                "source_group": identity(
                    {
                        "target": saved["target"],
                        "source_idx": saved["source_idx"],
                        "seed": saved["original_seed"],
                    }
                ),
                "oracle_protocol": unit["oracle_protocol"],
                "eligibility": strict_endpoint_scorer(saved["original_seed"]),
            }
            measured = {}
            for arm, config in (("control", control), ("retrieval", retrieval)):
                before = meter.calls
                batch = initial_program_batch(source, entries, config, **common)
                if arm == "retrieval":
                    repeated = initial_program_batch(source, entries, config, **common)
                    if repeated["batch_id"] != batch["batch_id"]:
                        raise ValueError(
                            f"retrieval cold start is nondeterministic: {cell}"
                        )
                seal(output / arm / f"{cell}.json", batch)
                measured[arm] = {
                    **_summary(batch, winners[cell]),
                    "executor_calls": meter.calls - before,
                    "artifact": f"{arm}/{cell}.json",
                    "artifact_sha256": sha256_file(output / arm / f"{cell}.json"),
                }
            control_endpoints = {
                row["endpoint"]
                for row in unseal(output / "control" / f"{cell}.json")["candidates"]
            }
            retrieval_endpoints = {
                row["endpoint"]
                for row in unseal(output / "retrieval" / f"{cell}.json")["candidates"]
            }
            known = set(winners[cell])
            rows.append(
                {
                    "cell": cell,
                    **measured,
                    "shared_endpoints": len(control_endpoints & retrieval_endpoints),
                    "retrieval_only_endpoints": len(
                        retrieval_endpoints - control_endpoints
                    ),
                    "control_only_endpoints": len(
                        control_endpoints - retrieval_endpoints
                    ),
                    "new_winner_recovery": sorted(
                        (retrieval_endpoints - control_endpoints) & known
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "cell": cell,
                        "control_candidates": measured["control"]["candidates"],
                        "retrieval_candidates": measured["retrieval"]["candidates"],
                        "new_winner_recovery": len(rows[-1]["new_winner_recovery"]),
                    }
                ),
                flush=True,
            )
    control_seconds = sum(row["control"]["proposal_seconds"] for row in rows)
    retrieval_seconds = sum(row["retrieval"]["proposal_seconds"] for row in rows)
    cells_with_new_winner = sum(bool(row["new_winner_recovery"]) for row in rows)
    all_nonempty = all(row["retrieval"]["candidates"] > 0 for row in rows)
    ratio = retrieval_seconds / control_seconds
    passed = all_nonempty and cells_with_new_winner >= 5 and ratio <= 1.5
    body = {
        "schema_version": "t4_program_retrieval_probe_v1",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            )
        ),
        "inputs_sha256": {
            str(CONTRACT.relative_to(ROOT)): sha256_file(CONTRACT),
            str(CENSUS.relative_to(ROOT)): sha256_file(CENSUS),
            str(library_path.relative_to(ROOT)): sha256_file(library_path),
        },
        "implementation_sha256": {
            path: sha256_file(ROOT / path) for path in IMPLEMENTATION
        },
        "configuration": {
            "control": control.__dict__,
            "retrieval": retrieval.__dict__,
            "direct_candidate_target": 8,
            "new_oracle_calls": 0,
        },
        "rows": rows,
        "summary": {
            "cells": len(rows),
            "control_candidates": sum(row["control"]["candidates"] for row in rows),
            "retrieval_candidates": sum(row["retrieval"]["candidates"] for row in rows),
            "cells_with_new_winner_recovery": cells_with_new_winner,
            "new_winners_recovered": sum(
                len(row["new_winner_recovery"]) for row in rows
            ),
            "control_proposal_seconds": control_seconds,
            "retrieval_proposal_seconds": retrieval_seconds,
            "proposal_time_ratio": ratio,
            "all_retrieval_batches_nonempty": all_nonempty,
            "structural_gate_passed": passed,
            "decision": (
                "POSITIVE_RETAIN_FOR_SUCCESSOR_CONTROLLER"
                if passed
                else (
                    "NEGATIVE_NO_SUPPORT_GAIN"
                    if cells_with_new_winner == 0
                    and sum(row["retrieval_only_endpoints"] for row in rows) == 0
                    else "INCONCLUSIVE_KEEP_EXPERIMENTAL"
                )
            ),
        },
        "evidence": {
            "candidate_endpoints": "computed with exact executor and strict endpoint gates",
            "winner_overlap": "answer-known comparison to public IVG development inputs",
            "external_scores": "reported public values; not new COMPOSE docking observations",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {"device": "cpu", "workers": 1, "machine": platform.machine()},
        "costs": {
            "wall_seconds": perf_counter() - began,
            "executor_calls": meter.calls,
            "new_oracle_calls": 0,
        },
    }
    seal(output / "result.json", body)
    print(json.dumps(body["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
