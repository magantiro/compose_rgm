"""Prepare, launch, or retrieve the single bounded fast PMO development run."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.fast_pmo_run import (
    APP,
    APP_NAME,
    CONTRACT,
    KIND,
    configuration,
    load_contract,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    if (ROOT / CONTRACT).exists():
        raise ValueError("frozen fast PMO contract already exists")
    prior = unseal(ROOT / "configs/parent_edit_cycles.json")
    initialization = "diagnostics/parent_edit_cycles/prepared/init_20260921.json"
    library = prior["library_path"]
    inputs = {p: prior["inputs"][p] for p in (initialization, library)}
    for path, digest in inputs.items():
        verify_file(ROOT / path, digest)
    body = {
        "schema_version": "fast_pmo_run_v1",
        "run": {
            "name": "albuterol_similarity",
            "seed": 20260921,
            "budget": 128,
            "rounds": 7,
            "queries_per_round": 16,
            "initialization_count": 16,
            "initialization_mode": "all_scored_pool",
            "initial_parent_fraction": 0.2,
            "max_seconds": 420,
        },
        "configuration": asdict(configuration()),
        "compute": {
            "cpu": 1,
            "memory_mib": 4096,
            "containers": 1,
            "timeout_seconds": 600,
            "gpu": False,
            "retries": 0,
            "reserved_usd": 1,
        },
        "compute_estimate": {
            "expected_worker_wall_seconds": [60, 420],
            "basis": "local four-pool program-only profile; no cloud speed-equivalence claim",
            "inherited_tariff_source": prior["compute"]["price_source"],
            "tariff_checked": prior["compute"]["price_checked"],
            "cpu_core_second_usd": prior["compute"]["cpu_core_second_usd"],
            "gib_second_usd": prior["compute"]["gib_second_usd"],
            "timeout_compute_estimate_usd": 600
            * (prior["compute"]["cpu_core_second_usd"] + 4 * prior["compute"]["gib_second_usd"]),
            "build_storage_headroom_in_reserved_usd": True,
        },
        "oracle_protocol": prior["pmo_protocols"]["albuterol_similarity"],
        "initialization_path": initialization,
        "library_path": library,
        "inputs": inputs,
        "information_regime": {
            "initialization": "reused task-independent exact structure lock, all queries charged",
            "library": "shared 146-program T4 winner-route-derived development library",
            "pmo_history": "task previously inspected; no historical PMO scores loaded",
            "reference": "not loaded; optimization proposal, not frozen-reference sampling",
            "previous_unresolved_query": "unchanged charged reservation, not recovered here",
            "comparison": "single development execution, no matched superiority claim",
        },
        "decision": {
            "positive": "multiple productive rounds and score progress justify a matched fast-control comparison",
            "negative": "empty pools or exhausted support direct repair to recorded proposal failures",
            "inconclusive": "runtime/oracle failure requires receipt audit, not automatic grid or retry",
        },
        "metrics": "per-query best/top-ten curve; budget-128 AUC diagnostic, not official 10k PMO result",
    }
    seal(ROOT / CONTRACT, body)
    load_contract(ROOT)
    print(json.dumps({"contract_sha256": sha256_file(ROOT / CONTRACT), "queries": 0}))


def launch(receipt):
    import modal

    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    intent = receipt.with_suffix(".intent.json")
    if receipt.exists() or intent.exists():
        raise ValueError(
            "launch already attempted; inspect receipt/intent instead of spawning again"
        )
    c = load_contract(ROOT)
    revision = local_image_revision(expected_commit=assert_synced(strict=True)["commit"])
    body = {
        "image_revision": revision,
        "files_sha256": {p: sha256_file(ROOT / p) for p in (CONTRACT, APP)},
    }
    task = {**body, "run_id": identity(body)}
    publish_json(intent, {"task": task, "automatic_retry": False})
    call = modal.Function.from_name(APP_NAME, "run").spawn(task)
    saved = {
        "task": task,
        "call_id": call.object_id,
        "volume": "compose-v4-artifacts",
        "volume_path": f"{KIND}/{task['run_id']}",
        "reserved_usd": c["compute"]["reserved_usd"],
    }
    publish_json(receipt, saved)
    print(json.dumps({k: v for k, v in saved.items() if k != "task"}, indent=2))


def retrieve(receipt, output):
    import modal

    saved = json.loads(receipt.read_text())
    result = modal.FunctionCall.from_id(saved["call_id"]).get(timeout=0)
    seal(output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "oracle_calls",
                    "remaining_queries",
                    "stop_reason",
                    "history",
                    "seconds",
                    "oracle_seconds",
                    "proposal_seconds",
                    "pmo_top10_auc_with_flat_tail",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "retrieve"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "launch" and args.receipt:
        launch(args.receipt)
    elif args.action == "retrieve" and args.receipt and args.output:
        retrieve(args.receipt, args.output)
    else:
        parser.error("launch needs --receipt; retrieve also needs --output")
