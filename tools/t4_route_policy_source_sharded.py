#!/usr/bin/env python3
"""Launch, monitor, collect and merge durable per-source T4 policy shards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_route_policy_comparison import predeclared_source_folds
from modal_apps.run_process_v2_p50_app import local_image_revision
from tools.t4_program_vocabulary_audit import source_group_map
from tools.t4_route_policy_comparison import _combined_metrics
from tools.t4_route_policy_select import actor_selection_gate

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "compose-t4-route-policy-source-sharded"
VOLUME_NAME = "compose-t4-route-distilled-artifacts"
REMOTE_OUTPUT_ROOT = "/t4_route_policy_source_sharded/attempt_1"
LOCAL = ROOT / "diagnostics/t4_route_policy_source_sharded/attempt_1"
LAUNCH = LOCAL / "launch.json"
OUTPUT = ROOT / "diagnostics/t4_route_policy_selection/attempt_1/result.json"
ACTOR = ROOT / "diagnostics/t4_route_distillation/attempt_3/actor.json.gz"
CONTRACT = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"
MATERIAL_FILES = (
    "AGENTS.md",
    "configs/t4_frozen_program_benchmark_v2.json",
    "modal_apps/t4_route_policy_source_sharded_app.py",
    "src/compose_v4/experiments/t4_route_policy_comparison.py",
    "tools/t4_route_policy_source_sharded.py",
    "tools/t4_route_policy_comparison.py",
    "tools/t4_program_vocabulary_audit.py",
    "tools/t4_route_distillation.py",
    "docs/GENMOL_T4_SEEDS.json",
    "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json",
)


def _clean_revision():
    from tools.preflight import assert_synced

    commit = assert_synced(strict=True)["commit"]
    return commit, local_image_revision(expected_commit=commit, repo_root=ROOT)


def _source_rows() -> list[dict]:
    metadata = source_group_map(unseal(CONTRACT), json.loads(SEEDS.read_text()))
    folds = predeclared_source_folds(metadata)
    rows = []
    for split in folds:
        for source_index, source_id in enumerate(split["test_sources"]):
            rows.append(
                {
                    "fold": split["fold"],
                    "source_index": source_index,
                    "source_id": source_id,
                    "cell": metadata[source_id]["cell"],
                }
            )
    if len(rows) != 15 or len({row["source_id"] for row in rows}) != 15:
        raise ValueError("source-sharded T4 census must contain 15 unique sources")
    return rows


def _tasks(image_revision):
    files = {relative: sha256_file(ROOT / relative) for relative in MATERIAL_FILES}
    result = []
    for row in _source_rows():
        output = f"{REMOTE_OUTPUT_ROOT}/fold_{row['fold']}/{row['cell']}"
        body = {
            "schema_version": "t4_route_policy_source_shard_lock_v1",
            **row,
            "output": f"/artifacts{output}",
            "oracle_calls": 0,
            "automatic_retry": 0,
            "image_revision": image_revision,
            "files_sha256": files,
        }
        result.append({**body, "run_id": identity(body)})
    return result


def launch():
    import modal

    if LAUNCH.exists():
        raise ValueError(
            "source-sharded T4 launch exists; monitor rather than duplicate"
        )
    commit, image_revision = _clean_revision()
    tasks = _tasks(image_revision)
    worker = modal.Function.from_name(APP_NAME, "worker")
    calls = {task["cell"]: worker.spawn(task).object_id for task in tasks}
    body = {
        "schema_version": "t4_route_policy_source_sharded_launch_v1",
        "commit": commit,
        "tasks": tasks,
        "call_ids": calls,
        "modal_volume": VOLUME_NAME,
        "concurrent_single_cpu_workers": 15,
        "restart_unit": "one held-out source",
        "oracle_calls_authorized": 0,
        "automatic_retry": 0,
    }
    receipt = {**body, "receipt_sha256": identity(body)}
    publish_json(LAUNCH, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _receipt():
    value = json.loads(LAUNCH.read_text())
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if identity(body) != value.get("receipt_sha256"):
        raise ValueError("source-sharded T4 launch receipt changed")
    if len(value.get("tasks", ())) != 15 or len(value.get("call_ids", {})) != 15:
        raise ValueError("source-sharded T4 launch census changed")
    return value


def _status_rows(receipt):
    import modal

    rows = []
    for cell, call_id in sorted(receipt["call_ids"].items()):
        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            rows.append({"cell": cell, "status": "running"})
        except (ValueError, RuntimeError, ImportError, modal.exception.Error) as error:
            rows.append(
                {
                    "cell": cell,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        else:
            rows.append({"cell": cell, "status": "complete", "result": result})
    return rows


def status():
    receipt = _receipt()
    rows = _status_rows(receipt)
    counts = {
        name: sum(row["status"] == name for row in rows)
        for name in ("running", "complete", "failed")
    }
    print(json.dumps({"counts": counts, "sources": rows}, indent=2, sort_keys=True))


def collect():
    import modal

    receipt = _receipt()
    volume = modal.Volume.from_name(receipt["modal_volume"])
    completed = {
        row["cell"] for row in _status_rows(receipt) if row["status"] == "complete"
    }
    collected, preserved = [], []
    for task in receipt["tasks"]:
        if task["cell"] not in completed:
            continue
        folder = LOCAL / f"fold_{task['fold']}" / task["cell"]
        destinations = [folder / name for name in ("result.json", "models.json.gz")]
        if all(path.exists() for path in destinations):
            preserved.append(task["cell"])
            continue
        if any(path.exists() for path in destinations):
            raise ValueError(f"partial local source artifact: {task['cell']}")
        for name in ("result.json", "models.json.gz"):
            destination = folder / name
            remote = f"{REMOTE_OUTPUT_ROOT}/fold_{task['fold']}/{task['cell']}/{name}"
            raw = b"".join(volume.read_file(remote))
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(raw)
            temporary.replace(destination)
        collected.append(task["cell"])
    print(
        json.dumps(
            {
                "status": "incremental_collection_complete",
                "newly_collected": collected,
                "already_preserved": preserved,
                "durable_sources": len(collected) + len(preserved),
                "oracle_calls": 0,
            }
        )
    )


def _load_envelope(path: Path, *, compressed=False):
    raw = gzip.decompress(path.read_bytes()) if compressed else path.read_bytes()
    envelope = json.loads(raw)
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid policy artifact envelope: {path}")
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"policy artifact payload changed: {path}")
    return envelope["payload"]


def merge():
    receipt = _receipt()
    if OUTPUT.exists():
        raise ValueError("route-policy selection exists; do not reselect")
    reports, model_rows, inputs = [], [], {}
    models_by_fold = {}
    observed_sources = set()
    for task in receipt["tasks"]:
        folder = LOCAL / f"fold_{task['fold']}" / task["cell"]
        result_path, model_path = folder / "result.json", folder / "models.json.gz"
        result = _load_envelope(result_path)
        models = _load_envelope(model_path, compressed=True)
        fold = task["fold"]
        source_id = task["source_id"]
        fold_reports = result.get("fold_reports", ())
        model_rows_for_source = models.get("fold_models", ())
        if (
            result.get("costs") != {"oracle_calls": 0, "docking_calls": 0}
            or result.get("implementation", {}).get("revision") != receipt["commit"]
            or result.get("implementation", {}).get("working_tree_dirty") is not False
            or result.get("predeclared_split", {}).get("evaluated_fold_ids") != [fold]
            or result.get("predeclared_split", {}).get("evaluated_test_sources")
            != [source_id]
            or len(fold_reports) != 1
            or fold_reports[0].get("split", {}).get("test_sources") != [source_id]
            or len(model_rows_for_source) != 1
            or model_rows_for_source[0].get("fold") != fold
        ):
            raise ValueError(
                f"source shard is incomplete or incompatible: {task['cell']}"
            )
        if source_id in observed_sources:
            raise ValueError(f"source shard repeated: {source_id}")
        observed_sources.add(source_id)
        reports.append(fold_reports[0])
        model = model_rows_for_source[0]
        model_rows.append(model)
        checkpoint_identity = identity(model)
        prior_identity = models_by_fold.setdefault(fold, checkpoint_identity)
        if checkpoint_identity != prior_identity:
            raise ValueError(f"fold {fold} model identity differs across source shards")
        for path in (result_path, model_path):
            inputs[str(path.relative_to(ROOT))] = sha256_file(path)
    expected_sources = {task["source_id"] for task in receipt["tasks"]}
    if observed_sources != expected_sources or len(observed_sources) != 15:
        raise ValueError("source shards do not cover the sealed 15-source census")
    autonomous = _combined_metrics(reports, "autonomous_generation")
    shared = _combined_metrics(reports, "shared_panel_reranking")
    actor_gate = actor_selection_gate(autonomous)
    selected = all(actor_gate.values())
    actor_payload = _load_envelope(ACTOR, compressed=True)
    body = {
        "schema_version": "t4_route_policy_selection_v1",
        "decision": (
            "route_distilled_actor_selected_for_scored_qualification"
            if selected
            else "route_distilled_actor_failed_zero_oracle_selection_gate"
        ),
        "selected_policy": "context_module_prototype" if selected else None,
        "selection_gate": actor_gate,
        "selection_rule": {
            "primary": (
                "held-source autonomous transformation recall@128 must not fall "
                "and transformation MRR must strictly improve over generic marginal"
            ),
            "safeguards": (
                "execution precision and unique endpoint yield must each retain at "
                "least half of generic marginal"
            ),
            "shared_panel_role": "reported reranking evidence only",
        },
        "aggregate_results": {
            "autonomous_generation": autonomous,
            "shared_panel_reranking": shared,
        },
        "folds": reports,
        "model_census": [
            {
                "fold": fold,
                "marginal_training_identity": row["marginal"]["training_identity"],
                "actor_training_identity": row["actor"]["training_identity"],
                "hybrid_training_identity": row["hybrid"]["training_identity"],
            }
            for fold, row in sorted({row["fold"]: row for row in model_rows}.items())
        ],
        "selected_all_route_checkpoint": (
            {
                "path": str(ACTOR.relative_to(ROOT)),
                "sha256": sha256_file(ACTOR),
                "payload_sha256": identity(actor_payload),
                "training_identity": actor_payload["training_identity"],
            }
            if selected
            else None
        ),
        "inputs": inputs,
        "source_sharded_execution": {
            "launch": str(LAUNCH.relative_to(ROOT)),
            "launch_sha256": sha256_file(LAUNCH),
            "commit": receipt["commit"],
            "sources": 15,
            "fold_model_identities": models_by_fold,
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0},
        "evidence": "answer-known T4 route-policy development with held-source folds",
        "limitations": [
            "The selected actor is subsequently fit on all 77 public T4 teacher routes.",
            "Transformation equivalence is a bounded structural diagnostic, not task utility.",
            "The graph contrastive model is reported as a candidate ranker, not a decoder.",
            "Source sharding repeats fold fitting but preserves fold training and test policy.",
        ],
    }
    body["input_set_sha256"] = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    publish_json(OUTPUT, body)
    print(
        json.dumps(
            {
                "decision": body["decision"],
                "selected_policy": body["selected_policy"],
                "selection_gate": body["selection_gate"],
                "sources": 15,
                "costs": body["costs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "status", "collect", "merge"))
    args = parser.parse_args()
    {"launch": launch, "status": status, "collect": collect, "merge": merge}[
        args.action
    ]()


if __name__ == "__main__":
    main()
