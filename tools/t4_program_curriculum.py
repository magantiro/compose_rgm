"""Freeze or launch the bounded first scoring round on four T4 development cells."""

import argparse
import json
import subprocess
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_program_curriculum import APP, LOCK

ROOT = Path(__file__).resolve().parents[1]
MOOD_REVISION = "ac9730818325a72f68d748802721762ab6134552"


def prepare(args):
    if (ROOT / LOCK).exists():
        raise ValueError("the curriculum lock already exists; do not relock candidates")
    folder = ROOT / "diagnostics/t4_shared_program_controller/attempt_2"
    report = json.loads((folder / "result.json").read_text())
    rows, inputs, protocols = [], {}, {}
    prior = json.loads((ROOT / "configs/t4_winner_refinement.json").read_text())
    runtime_inputs = {"/opt/dock/qvina02": prior["expected_input_sha256"]["qvina02"]}
    for cell in report["cells"]:
        path = folder / f"{cell['cell']}.json"
        verify_file(path, cell["batch_sha256"])
        inputs[str(path.relative_to(ROOT))] = cell["batch_sha256"]
        batch = json.loads(path.read_text())
        protocol = cell["protocol"]
        target = protocol["target"]
        digest = sha256_file(args.receptors / f"{target}.pdbqt")
        runtime_inputs[f"/opt/dock/receptors/{target}.pdbqt"] = digest
        domain = {
            "target": target,
            "source_idx": protocol["source_idx"],
            "original_seed": protocol["original_seed"],
            "delta": 0.4,
            "qed_min": 0.6,
            "sa_max": 4.0,
            "receptor_sha256": digest,
            "docking": prior["docking"],
            "qvina02_sha256": runtime_inputs["/opt/dock/qvina02"],
            "box_definition": "unchanged BOXES in hash-bound modal_apps/genmol_t4_opt_app.py",
        }
        protocols[cell["cell"]] = domain
        common = {
            "cell": cell["cell"],
            "target": target,
            "original_seed": protocol["original_seed"],
            "oracle_protocol": identity(domain),
        }
        control = {
            **common,
            "role": "seed_control",
            "smiles": protocol["original_seed"],
            "state": cell["source"]["state"],
        }
        rows.append({**control, "candidate_id": identity(control)})
        if len(batch["candidates"]) != 8:
            raise ValueError("each curriculum cell must have its already locked eight candidates")
        for candidate in batch["candidates"]:
            actual = {
                **common,
                "role": "candidate",
                "smiles": candidate["endpoint"],
                "trace": candidate["trace"],
                "source_batch_id": batch["batch_id"],
                "source_candidate_id": candidate["candidate_id"],
            }
            rows.append({**actual, "candidate_id": identity(actual)})
    if len(rows) != 36:
        raise ValueError("curriculum must contain four complete nine-query cells")
    lock = {
        "schema_version": "t4_program_curriculum_lock_v1",
        "cells": [c["cell"] for c in report["cells"]],
        "take": rows,
        "input_sha256": inputs,
        "preparation_sha256": sha256_file(folder / "result.json"),
        "shared_library_sha256": report["library_sha256"],
        "oracle_protocols": protocols,
        "source_registry": {
            "path": "docs/GENMOL_T4_SEEDS.json",
            "sha256": sha256_file(ROOT / "docs/GENMOL_T4_SEEDS.json"),
        },
        "runtime_input_sha256": runtime_inputs,
        "required_rdkit": "2024.03.5",
        "docking_seed": 1701,
        "source_origin": {
            "repository": "https://github.com/SeulLee05/MOOD",
            "revision": MOOD_REVISION,
            "receptor_directory": "scorer/receptors",
            "access_basis": "public MIT-licensed source",
        },
        "new_oracle_call_limit": 36,
        "prior_followon_calls": 28,
        "cumulative_followon_maximum": 64,
        "authorization_ceiling": 66,
        "compute": {
            "workers": 8,
            "driver": 1,
            "cpu_per_worker": 1,
            "gpu": False,
            "expected_wall_seconds": [60, 1200],
            "reserved_usd": 20,
            "worker_timeout_seconds": 600,
            "driver_timeout_seconds": 3600,
            "restart_unit": "sealed per-row result; no retry after an ambiguous start",
            "automatic_retries": 0,
        },
        "selection": "all eight already locked candidates per cell plus the prescribed seed; no surrogate selection",
        "interpretation": "shared-program curriculum initialization; broad/static/adaptive comparison follows measured initialization",
    }
    seal(ROOT / LOCK, lock)
    print("Locked 36 calls: 32 candidates + 4 seed controls; cumulative follow-on cap 64/66.")


def launch(args):
    import modal

    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    preflight = assert_synced(strict=True)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("curriculum launch requires clean committed source")
    lock = unseal(ROOT / LOCK)
    if lock["cumulative_followon_maximum"] > lock["authorization_ceiling"]:
        raise ValueError("curriculum exceeds the total authorized follow-on calls")
    task = {
        "files_sha256": {
            p: sha256_file(ROOT / p) for p in (LOCK, APP, "modal_apps/genmol_t4_opt_app.py")
        },
        "image_revision": local_image_revision(expected_commit=preflight["commit"]),
    }
    task["run_id"] = identity(task)
    call = modal.Function.from_name("compose-t4-program-curriculum", "run_curriculum").spawn(task)
    receipt = {
        "task": task,
        "call_id": call.object_id,
        "new_oracle_call_limit": 36,
        "volume": "compose-v4-artifacts",
        "volume_path": f"t4_program_curriculum/{task['run_id']}",
    }
    publish_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch"))
    parser.add_argument("--receptors", type=Path)
    parser.add_argument(
        "--receipt", type=Path, default=ROOT / "diagnostics/t4_program_curriculum_spawn.json"
    )
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.receptors is None:
            raise ValueError("preparation requires the downloaded pinned receptor directory")
        prepare(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
