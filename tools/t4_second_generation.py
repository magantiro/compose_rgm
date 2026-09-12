"""Bind prepared paired pools, launch once, and ingest only each arm's own labels."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer
from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_second_generation import APP, KIND, LOCK, summarize

ROOT = Path(__file__).resolve().parents[1]
PREP = "diagnostics/t4_second_generation/attempt_2"
PAID = "diagnostics/t4_program_curriculum/attempt_1"


def prepare():
    if (ROOT / LOCK).exists():
        raise ValueError("second-generation scoring lock already exists; do not relock")
    report = unseal(ROOT / PREP / "result.json")
    prior = unseal(ROOT / "configs/t4_program_curriculum_lock.json")
    reviewed = unseal(ROOT / PAID / "review.json")
    old_result = unseal(ROOT / PAID / "remote_result.json")
    queries, memberships, inputs, incumbents = {}, [], {}, {}
    for path in (
        f"{PREP}/result.json",
        f"{PAID}/review.json",
        f"{PAID}/remote_result.json",
        "configs/t4_program_curriculum_lock.json",
    ):
        inputs[path] = sha256_file(ROOT / path)
    for path, digest in reviewed["inputs_sha256"].items():
        # Existing review binds absolute local receipt paths as well as repo paths.
        verify_file(Path(path) if Path(path).is_absolute() else ROOT / path, digest)
    for cell in report["cells"]:
        name = cell["cell"]
        domain = prior["oracle_protocols"][name]
        common = {"cell": name, "target": domain["target"], "oracle_protocol": identity(domain)}
        seen = {r["smiles"] for r in old_result["rows"] if r["cell"] == name}
        for arm in cell["arms"]:
            path = f"{PREP}/{arm['batch_path']}"
            snapshot_path = f"{PREP}/{arm['snapshot_path']}"
            for relative, digest in (
                (path, arm["batch_sha256"]),
                (snapshot_path, arm["snapshot_sha256"]),
            ):
                verify_file(ROOT / relative, digest)
                inputs[relative] = digest
            batch = json.loads((ROOT / path).read_text())
            snapshot = json.loads((ROOT / snapshot_path).read_text())
            search = ProgramOptimizer.restore(snapshot["optimizer"])
            if (
                search.pending["batch_id"] != batch["batch_id"]
                or search.oracle_protocol != common["oracle_protocol"]
            ):
                raise ValueError("prepared arm disagrees with its qualified snapshot/domain")
            if search.config.parent_allocation != arm["arm"]:
                raise ValueError("prepared arm's allocation mode changed")
            if (
                identity({k: v for k, v in search.pending.items() if k != "batch_id"})
                != batch["batch_id"]
            ):
                raise ValueError("prepared batch identity changed")
            if search.pending["candidates"] != batch["candidates"]:
                raise ValueError("candidate file disagrees with the pending lock")
            for candidate in batch["candidates"]:
                if candidate["endpoint"] in seen:
                    raise ValueError(
                        "new pool contains an already queried first-generation endpoint"
                    )
                body = {**common, "smiles": candidate["endpoint"], "docking_seed": 1701}
                query_id = identity(body)
                if query_id not in queries:
                    queries[query_id] = {
                        **body,
                        "candidate_id": query_id,
                        "original_seed": domain["original_seed"],
                        "trace": candidate["trace"],
                    }
                memberships.append(
                    {
                        "cell": name,
                        "arm": arm["arm"],
                        "query_id": query_id,
                        "candidate_id": candidate["candidate_id"],
                        "batch_id": batch["batch_id"],
                        "batch_path": path,
                        "snapshot_path": snapshot_path,
                    }
                )
        best = next(c["best"] for c in reviewed["cells"] if c["cell"] == name)
        construction = next(
            r for r in prior["take"] if r["cell"] == name and r["smiles"] == best["smiles"]
        )
        incumbents[name] = {
            **common,
            "original_seed": domain["original_seed"],
            "smiles": best["smiles"],
            "historical_ds": best["ds"],
            "trace": construction["trace"],
            "historical_receipt_id": best["receipt_id"],
        }
    if len(queries) != 49 or len(memberships) != 63:
        raise ValueError("locked attempt_2 pool census changed")
    lock = {
        "schema_version": "t4_second_generation_lock_v1",
        "cells": prior["cells"],
        "take": list(queries.values()),
        "memberships": memberships,
        "incumbents": incumbents,
        "inputs_sha256": inputs,
        "oracle_protocols": prior["oracle_protocols"],
        "runtime_input_sha256": prior["runtime_input_sha256"],
        "required_rdkit": prior["required_rdkit"],
        "new_oracle_call_limit": 73,
        "first_calls": 49,
        "maximum_confirmation_calls": 24,
        "compute": {
            "workers": 8,
            "driver": 1,
            "cpu_per_worker": 1,
            "gpu": False,
            "worker_memory_mib": 4096,
            "driver_memory_mib": 2048,
            "worker_timeout_seconds": 600,
            "driver_timeout_seconds": 1200,
            "automatic_retries": 0,
            "reserved_usd": 20,
            "expected_wall_seconds": [60, 300],
            "expected_cost": "cents-scale CPU/RAM at previous throughput; excludes deployment/startup and is not a billed amount",
            "restart_unit": "sealed stage/row; ambiguous prior start is never retried",
        },
        "selection": "all locked candidates, no regeneration or surrogate veto; score ties by canonical SMILES",
        "confirmation": "two fresh seeds 1702/1703 for each within-arm champion and each predeclared incumbent; identical physical queries shared",
        "label_access": "each arm receives only its own previously requested candidate labels; no cross-target scores",
        "comparison": "same paid histories and program support; score-ranked versus score-blind allocation, development only",
        "preserved_failures": "FA7 score_rank filled seven of eight candidate slots",
        "source_origin": prior["source_origin"],
    }
    seal(ROOT / LOCK, lock)
    print("Locked 49 first queries and at most 24 fresh confirmations; no oracle called.")


def launch(receipt):
    import modal

    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    if receipt.exists():
        raise ValueError("launch receipt already exists; monitor the existing call")
    preflight = assert_synced(strict=True)
    revision = local_image_revision(expected_commit=preflight["commit"])
    lock = unseal(ROOT / LOCK)
    for path, digest in lock["inputs_sha256"].items():
        verify_file(ROOT / path, digest)
    if len(lock["take"]) != 49 or lock["new_oracle_call_limit"] != 73:
        raise ValueError("launch exceeds or differs from the approved allocation")
    task = {
        "image_revision": revision,
        "files_sha256": {
            p: sha256_file(ROOT / p) for p in (LOCK, APP, "modal_apps/genmol_t4_opt_app.py")
        },
    }
    task["run_id"] = identity(task)
    call = modal.Function.from_name("compose-t4-second-generation", "run_comparison").spawn(task)
    body = {
        "task": task,
        "call_id": call.object_id,
        "new_oracle_call_limit": 73,
        "volume": "compose-v4-artifacts",
        "volume_path": f"{KIND}/{task['run_id']}",
    }
    publish_json(receipt, body)
    print(json.dumps({k: v for k, v in body.items() if k != "task"}, indent=2))


def review(folder):
    result_path = folder / "remote_result.json"
    result, lock = unseal(result_path), unseal(ROOT / LOCK)
    launch_receipt = json.loads((folder / "launch.json").read_text())
    if result["task"] != launch_receipt["task"]:
        raise ValueError("returned task differs from launch")
    verify_file(ROOT / LOCK, result["task"]["files_sha256"][LOCK])
    for path, digest in lock["inputs_sha256"].items():
        verify_file(ROOT / path, digest)
    cells = summarize(lock, result["rows"], result["confirmation"])
    if cells != result["cells"] or result["new_oracle_calls"] != len(result["rows"]) + len(
        result["confirmation"]
    ):
        raise ValueError("remote reduction/accounting changed")
    result_hash = sha256_file(result_path)
    scored = {r["candidate_id"]: r for r in result["rows"]}
    archives = {}
    for cell in lock["cells"]:
        for arm in ("score_rank", "score_blind"):
            members = [m for m in lock["memberships"] if m["cell"] == cell and m["arm"] == arm]
            wrapper = json.loads((ROOT / members[0]["snapshot_path"]).read_text())
            search = ProgramOptimizer.restore(wrapper["optimizer"])
            outcomes = []
            for member in members:
                row = scored[member["query_id"]]
                outcomes.append(
                    {
                        "candidate_id": member["candidate_id"],
                        "oracle_protocol": row["oracle_protocol"],
                        "score": row["ds"],
                        "failure": row["failure"],
                        "receipt_id": identity(
                            {"result_sha256": result_hash, "stage": "first", "row": row}
                        ),
                    }
                )
            search.observe_batch(members[0]["batch_id"], outcomes)
            for row in result["confirmation"]:
                if row["cell"] != cell or row["ds"] is None:
                    continue
                matching = [e for e in search.entries.values() if e["endpoint"] == row["smiles"]]
                if not matching:
                    continue  # A different arm's exclusive champion is not this arm's label.
                search.add_measured_program(
                    matching[0],
                    receipt_id=identity(
                        {"result_sha256": result_hash, "stage": "confirmation", "row": row}
                    ),
                    score=row["ds"],
                    static_score=matching[0]["static_score"],
                )
            archives[f"{cell}_{arm}_archive.json"] = {
                "schema_version": "t4_second_generation_archive_v1",
                "optimizer": search.snapshot(),
                "predecessor_snapshot_id": wrapper["optimizer"]["snapshot_id"],
                "result_sha256": result_hash,
                "lock_sha256": sha256_file(ROOT / LOCK),
                "oracle_domain": lock["oracle_protocols"][cell],
                "new_oracle_calls_in_conversion": 0,
            }
    hashes = {name: publish_json(folder / name, body) for name, body in archives.items()}
    review_body = {
        "schema_version": "t4_second_generation_review_v1",
        "cells": cells,
        "inputs_sha256": {
            **lock["inputs_sha256"],
            LOCK: sha256_file(ROOT / LOCK),
            str(result_path): result_hash,
        },
        "outputs_sha256": hashes,
        "new_oracle_calls": result["new_oracle_calls"],
        "driver_seconds": result["seconds"],
        "worker_seconds": sum(r["seconds"] for r in [*result["rows"], *result["confirmation"]]),
        "analysis_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "analysis_implementation_sha256": sha256_file(Path(__file__)),
        "software": result["software"],
        "completed_at_utc": result["completed_at_utc"],
        "interpretation": "paired development and two-repeat confirmation; no external matched benchmark win",
    }
    seal(folder / "review.json", review_body)
    print(
        json.dumps(
            {
                "calls": review_body["new_oracle_calls"],
                "cells": [
                    {
                        "cell": c["cell"],
                        "incumbent_repeats": c["incumbent_repeat_scores"],
                        "arms": [
                            {
                                "arm": a["arm"],
                                "best": None if a["best"] is None else a["best"]["ds"],
                                "repeats": a["repeat_scores"],
                                "repeat_improvement": a["repeat_improvement_over_incumbent"],
                            }
                            for a in c["arms"]
                        ],
                    }
                    for c in cells
                ],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch", "review"))
    parser.add_argument(
        "--folder", type=Path, default=ROOT / "diagnostics/t4_second_generation/scoring_1"
    )
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "launch":
        launch(args.receipt or args.folder / "launch.json")
    else:
        review(args.folder)


if __name__ == "__main__":
    main()
