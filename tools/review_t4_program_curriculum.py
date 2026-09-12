"""Verify the four-target receipts and retain measured programs without new queries."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from pathlib import Path

from rdkit import rdBase

from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer, ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_program_curriculum import LOCK
from compose_v4.rewrite.trace_shard import decode_state
from tools.ivg_winner_paths import implementation_closure

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = "diagnostics/t4_shared_program_controller/attempt_2"
CENSUS = "diagnostics/ivg_t4_census/census.json"
CENSUS_SHA256 = "2c06fdfe65fc31327d865381ccb8f4d2c40553c3925d9dc0a03386c22fda05bb"


def validate_outcomes(lock, result):
    """Bind every score or failure to its original locked target and molecule."""
    rows = result["rows"]
    if len(rows) != len(lock["take"]) or len(rows) != result["new_oracle_calls"]:
        raise ValueError("missing or extra charged docking outcomes")
    for index, (expected, observed) in enumerate(zip(lock["take"], rows, strict=True)):
        if observed["index"] != index:
            raise ValueError(f"docking row {index}: changed order or duplicate receipt")
        for field in ("candidate_id", "cell", "target", "role", "smiles", "oracle_protocol"):
            if expected[field] != observed[field]:
                raise ValueError(f"docking row {index}: mismatched {field}")
        if observed["docking_seed"] != lock["docking_seed"]:
            raise ValueError(f"docking row {index}: changed docking seed")
        if observed["ds"] is not None and not math.isfinite(observed["ds"]):
            raise ValueError(f"docking row {index}: nonfinite score")
        if (observed["ds"] is None) != (observed["failure"] == "oracle_failed"):
            raise ValueError(f"docking row {index}: missing or inconsistent failure status")


def review(folder):
    result_path, launch_path = folder / "remote_result.json", folder / "launch.json"
    result, lock = unseal(result_path), unseal(ROOT / LOCK)
    launch = json.loads(launch_path.read_text())
    if result["task"] != launch["task"]:
        raise ValueError("returned task differs from the durable launch receipt")
    verify_file(ROOT / LOCK, result["task"]["files_sha256"][LOCK])
    validate_outcomes(lock, result)
    verify_file(ROOT / CENSUS, CENSUS_SHA256)
    census = json.loads((ROOT / CENSUS).read_text())
    prep_path = ROOT / PREPARATION / "result.json"
    verify_file(prep_path, lock["preparation_sha256"])
    verify_file(ROOT / PREPARATION / "shared_library.json", lock["shared_library_sha256"])
    for name, digest in lock["input_sha256"].items():
        verify_file(ROOT / name, digest)
    preparation = json.loads(prep_path.read_text())
    if rdBase.rdkitVersion != lock["required_rdkit"]:
        raise ValueError("archive replay must use the qualified chemistry version")
    result_hash = sha256_file(result_path)
    inputs = {
        LOCK: sha256_file(ROOT / LOCK),
        str(result_path): result_hash,
        str(launch_path): sha256_file(launch_path),
        CENSUS: CENSUS_SHA256,
        str(prep_path): lock["preparation_sha256"],
        **lock["input_sha256"],
    }
    closure = implementation_closure(Path(__file__).resolve())
    software = {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion}
    archive_payloads, cells = {}, []
    for cell in lock["cells"]:
        batch_path = ROOT / PREPARATION / f"{cell}.json"
        batch = json.loads(batch_path.read_text())
        candidates = {c["candidate_id"]: c for c in batch["candidates"]}
        config = dict(preparation["configuration"])
        config["channel_probabilities"] = tuple(config["channel_probabilities"])
        protocol = identity(lock["oracle_protocols"][cell])
        optimizer = ProgramOptimizer(
            ProgramSearchConfig(**config),
            source_group=batch["source_group"],
            oracle_protocol=protocol,
        )
        measured, failures, seed = [], [], None
        for row, original in zip(result["rows"], lock["take"], strict=True):
            if row["cell"] != cell:
                continue
            if row["role"] == "seed_control":
                seed = row
                continue
            candidate = candidates[original["source_candidate_id"]]
            if (
                original["source_batch_id"] != batch["batch_id"]
                or candidate["endpoint"] != row["smiles"]
                or candidate["trace"] != original["trace"]
                or row["oracle_protocol"] != protocol
            ):
                raise ValueError(f"{cell}: scored candidate differs from its exact program lock")
            receipt = identity({"result_sha256": result_hash, "row": row})
            if row["ds"] is None:
                failures.append(row)
                optimizer.failed_endpoints.add(row["smiles"])
                continue
            record = {
                **candidate,
                "oracle_protocol": protocol,
                "score": row["ds"],
                "structural_proposal_context": candidate["oracle_protocol"],
                "scored_receipt": {
                    "receipt_id": receipt,
                    "result_sha256": result_hash,
                    "index": row["index"],
                },
            }
            optimizer.add_measured_program(record, receipt_id=receipt, score=row["ds"])
            states = candidate["trace"]["states"]
            source_atoms = decode_state(states[0]).n_real_atoms
            endpoint_atoms = decode_state(states[-1]).n_real_atoms
            measured.append(
                {
                    "smiles": row["smiles"],
                    "ds": row["ds"],
                    "receipt_id": receipt,
                    "properties": candidate["provenance"]["properties"],
                    "source_heavy_atoms": source_atoms,
                    "endpoint_heavy_atoms": endpoint_atoms,
                    "heavy_atom_delta": endpoint_atoms - source_atoms,
                    "actual_changes": candidate["trace"]["actual_changes"],
                }
            )
        if seed is None:
            raise ValueError(f"{cell}: no charged seed control")
        external = next(
            c
            for c in census["cells"]
            if c["target"] == seed["target"]
            and c["source_idx"] == lock["oracle_protocols"][cell]["source_idx"]
            and c["delta"] == 0.4
        )
        best = min(measured, key=lambda r: (r["ds"], r["smiles"]), default=None)
        scores = [r["ds"] for r in measured]
        summary = {
            "cell": cell,
            "candidate_calls": len(candidates),
            "seed_control_calls": 1,
            "oracle_failures": len(failures),
            "seed_ds": seed["ds"],
            "best": best,
            "candidate_scores_in_locked_order": scores,
            "better_than_seed": None if seed["ds"] is None else sum(s < seed["ds"] for s in scores),
            "ivg_reported_run_bests": [r["reported_docking_score"] for r in external["runs"]],
            "external_comparison": "released scores, not redocked controls or a matched-budget benchmark win",
            "program_curve": next(c["program_curve"] for c in result["cells"] if c["cell"] == cell),
            "observations": len(optimizer.observations),
        }
        cells.append(summary)
        archive_payloads[f"{cell}_archive.json"] = {
            "schema_version": "qualified_program_curriculum_archive_v1",
            "optimizer": optimizer.snapshot(),
            "inputs": inputs,
            "implementation_sha256": closure,
            "software": software,
            "oracle_domain": lock["oracle_protocols"][cell],
            "initialization_receipts": [r for r in result["rows"] if r["cell"] == cell],
            "failures": failures,
            "new_oracle_calls_in_conversion": 0,
        }
    # Nothing is published until every input/receipt/program passes validation.
    output_hashes = {
        name: publish_json(folder / name, body) for name, body in archive_payloads.items()
    }
    report = {
        "schema_version": "t4_program_curriculum_review_v1",
        "inputs_sha256": inputs,
        "analysis_implementation_sha256": closure,
        "analysis_base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "analysis_revision_note": "exact reducer content is hash-bound separately from its pre-commit base revision",
        "software": software,
        "randomness": "no sampled analysis; exact archived search seeds/RNG in snapshots",
        "cells": cells,
        "outputs_sha256": output_hashes,
        "assay_new_oracle_calls": result["new_oracle_calls"],
        "conversion_new_oracle_calls": 0,
        "driver_seconds": result["seconds"],
        "worker_seconds": sum(r["seconds"] for r in result["rows"]),
        "completed_at_utc": result["completed_at_utc"],
        "capacity_census": {
            "distinct_released_winners": census["all_targets"]["unique_structures"],
            "above_40": census["all_targets"]["above_40_heavy_atoms"],
        },
        "limitations": [
            "one small shared-program development batch per target; no replicate confirmation",
            "library includes public winner routes; external numeric scores are not recipient labels",
            "no measured matched broad/static/adaptive comparison in this initialization round",
            "local archive records unavailable broad-runtime draws explicitly",
            "CPU seconds are not provider-billed dollars",
        ],
        "next_decision": "retain the four genuinely measured archives; prioritize shared program adaptation/replacement without raising the 40-atom cap; further paid rounds need their own bounded authorization",
    }
    seal(folder / "review.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder", type=Path, default=ROOT / "diagnostics/t4_program_curriculum/attempt_1"
    )
    args = parser.parse_args()
    report = review(args.folder)
    print(
        json.dumps(
            {
                "calls": report["assay_new_oracle_calls"],
                "cells": [
                    {
                        "cell": c["cell"],
                        "seed": c["seed_ds"],
                        "best": None if c["best"] is None else c["best"]["ds"],
                        "observations": c["observations"],
                    }
                    for c in report["cells"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
