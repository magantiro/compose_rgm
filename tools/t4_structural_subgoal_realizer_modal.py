#!/usr/bin/env python3
"""Launch, query, collect and reduce the structural-realizer Modal fan-out."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from modal_apps.run_process_v2_p50_app import local_image_revision
from modal_apps.t4_structural_subgoal_realizer_app import APP_NAME, MATERIAL_FILES, VOLUME_NAME
from tools.t4_structural_subgoal_audit import source_groups

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = ROOT / "diagnostics/t4_structural_subgoal_realizer/attempt_2"
LAUNCH = LOCAL_ROOT / "launch.json"
SUMMARY = LOCAL_ROOT / "summary.json"
REMOTE_ROOT = "/t4_structural_subgoal_realizer/attempt_2"


def _clean_revision():
    from tools.preflight import assert_synced

    commit = assert_synced(strict=True)["commit"]
    return commit, local_image_revision(expected_commit=commit, repo_root=ROOT)


def _tasks(image_revision):
    files = {relative: sha256_file(ROOT / relative) for relative in MATERIAL_FILES}
    rows = []
    for source_group in source_groups():
        body = {
            "schema_version": "t4_structural_subgoal_realizer_source_lock_v2",
            "source_group": source_group,
            "output": f"/artifacts{REMOTE_ROOT}/{source_group}/result.json",
            "oracle_calls": 0,
            "automatic_retry": 0,
            "image_revision": image_revision,
            "files_sha256": files,
        }
        rows.append({**body, "run_id": identity(body)})
    return rows


def launch():
    import modal

    if LAUNCH.exists():
        raise ValueError("structural-realizer launch exists; query it instead")
    commit, image_revision = _clean_revision()
    tasks = _tasks(image_revision)
    worker = modal.Function.from_name(APP_NAME, "worker")
    calls = {task["source_group"]: worker.spawn(task).object_id for task in tasks}
    body = {
        "schema_version": "t4_structural_subgoal_realizer_launch_v2",
        "commit": commit,
        "tasks": tasks,
        "call_ids": calls,
        "modal_volume": VOLUME_NAME,
        "concurrent_single_cpu_workers": 15,
        "heartbeat_seconds": 30,
        "oracle_calls_authorized": 0,
        "automatic_retry": 0,
    }
    publish_json(LAUNCH, {**body, "receipt_sha256": identity(body)})
    print(json.dumps({"status": "launched", "workers": len(calls), "commit": commit}))


def _receipt():
    receipt = json.loads(LAUNCH.read_text())
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if identity(body) != receipt.get("receipt_sha256"):
        raise ValueError("structural-realizer launch receipt changed")
    return receipt


def _read_volume_json(volume, path: str):
    try:
        return json.loads(b"".join(volume.read_file(path)))
    except FileNotFoundError:
        return None


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _status_rows(receipt):
    import modal

    volume = modal.Volume.from_name(receipt["modal_volume"])
    rows = []
    for source_group, call_id in sorted(receipt["call_ids"].items()):
        progress = _read_volume_json(volume, f"{REMOTE_ROOT}/{source_group}/progress.json")
        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            rows.append({"source_group": source_group, "status": "running", "progress": progress})
        except (ValueError, RuntimeError, ImportError, modal.exception.Error) as error:
            rows.append(
                {
                    "source_group": source_group,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "progress": progress,
                }
            )
        else:
            rows.append(
                {
                    "source_group": source_group,
                    "status": "complete",
                    "result": result,
                    "progress": progress,
                }
            )
    return rows


def status():
    rows = _status_rows(_receipt())
    counts = {
        state: sum(row["status"] == state for row in rows)
        for state in ("running", "complete", "failed")
    }
    running_etas = [
        row["progress"].get("estimated_remaining_seconds_from_completed_work")
        for row in rows
        if row["status"] == "running" and row.get("progress")
    ]
    running_etas = [value for value in running_etas if value is not None]
    print(
        json.dumps(
            {
                "counts": counts,
                "estimated_wall_seconds_remaining": max(running_etas, default=None),
                "sources": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


def collect():
    import modal

    receipt = _receipt()
    rows = _status_rows(receipt)
    complete = {row["source_group"] for row in rows if row["status"] == "complete"}
    volume = modal.Volume.from_name(receipt["modal_volume"])
    downloaded = []
    for source_group in sorted(complete):
        destination = LOCAL_ROOT / "sources" / source_group / "result.json"
        if destination.exists():
            continue
        raw = b"".join(volume.read_file(f"{REMOTE_ROOT}/{source_group}/result.json"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(raw)
        temporary.replace(destination)
        downloaded.append(source_group)
    print(json.dumps({"complete": len(complete), "downloaded": downloaded}))


def merge():
    receipt = _receipt()
    if SUMMARY.exists():
        raise ValueError("structural-realizer summary already exists")
    rows, inputs = [], {}
    for source_group in sorted(receipt["call_ids"]):
        path = LOCAL_ROOT / "sources" / source_group / "result.json"
        if not path.exists():
            raise ValueError(f"missing structural-realizer source shard: {source_group}")
        row = json.loads(path.read_text())
        if (
            row.get("schema_version") != "t4_structural_subgoal_realizer_source_audit_v2"
            or row.get("source_group") != source_group
            or row.get("implementation", {}).get("revision") != receipt["commit"]
            or any(
                row.get("costs", {}).get(name) != 0
                for name in ("oracle_calls", "docking_calls", "gpu_seconds")
            )
        ):
            raise ValueError(f"incompatible structural-realizer shard: {source_group}")
        rows.append(row)
        inputs[str(path.relative_to(ROOT))] = sha256_file(path)
    routes = sum(row["census"]["routes"] for row in rows)
    subgoals = sum(row["census"]["subgoals"] for row in rows)
    route_realized = sum(row["gates"]["route_realization_coverage"]["covered"] for row in rows)
    route_exact = sum(row["gates"]["route_exact_endpoint_precision"]["exact"] for row in rows)
    subgoal_realized = sum(
        row["gates"]["subgoal_target_precision_within_realized_complete_goals"][
            "containing_route_realized"
        ]
        for row in rows
    )
    subgoal_exact = sum(
        row["gates"]["subgoal_target_coverage_within_complete_goals"]["covered"] for row in rows
    )
    fallback = sum(row["gates"]["teacher_action_fallback"]["used"] for row in rows)
    subgoal_ids = {subgoal["subgoal_id"] for row in rows for subgoal in row["subgoals"]}
    route_times = [route["elapsed_seconds"] for row in rows for route in row["routes"]]
    route_expansions = [route["expanded"] for row in rows for route in row["routes"]]
    route_attempts = [route["attempted"] for row in rows for route in row["routes"]]
    route_primitives = [
        route["realized_primitive_count"] for row in rows for route in row["routes"]
    ]
    status_counts = Counter()
    for row in rows:
        status_counts.update(row["census"]["route_status_counts"])
    body = {
        "schema_version": "t4_structural_subgoal_realizer_summary_v2",
        "evidence": "answer-known zero-oracle conditional-realizer audit",
        "implementation_revision": receipt["commit"],
        "launch_receipt_sha256": sha256_file(LAUNCH),
        "source_shards": inputs,
        "census": {
            "sources": len(rows),
            "routes": routes,
            "subgoals": subgoals,
            "unique_subgoal_ids": len(subgoal_ids),
            "status_counts": dict(sorted(status_counts.items())),
            "search_expansions": sum(row["census"]["search_expansions"] for row in rows),
            "action_attempts": sum(row["census"]["action_attempts"] for row in rows),
        },
        "gates": {
            "route_realization_coverage": {
                "covered": route_realized,
                "denominator": routes,
                "coverage": route_realized / routes,
            },
            "route_exact_endpoint_precision": {
                "exact": route_exact,
                "realized": route_realized,
                "precision": route_exact / max(1, route_realized),
            },
            "subgoal_target_coverage_within_complete_goals": {
                "covered": subgoal_exact,
                "denominator": subgoals,
                "coverage": subgoal_exact / subgoals,
            },
            "subgoal_target_precision_within_realized_complete_goals": {
                "exact": subgoal_exact,
                "containing_route_realized": subgoal_realized,
                "precision": subgoal_exact / max(1, subgoal_realized),
            },
            "teacher_action_fallback": {
                "used": fallback,
                "route_denominator": routes,
                "subgoal_denominator": subgoals,
            },
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0, "gpu_seconds": 0},
        "timing": {
            "route_seconds_median": _percentile(route_times, 0.5),
            "route_seconds_p95": _percentile(route_times, 0.95),
            "route_seconds_maximum": max(route_times, default=0.0),
        },
        "compiler_cost": {
            metric: {
                "median": _percentile(values, 0.5),
                "p95": _percentile(values, 0.95),
                "maximum": max(values, default=0),
            }
            for metric, values in {
                "route_seconds": route_times,
                "expansions": route_expansions,
                "action_attempts": route_attempts,
                "realized_primitives": route_primitives,
            }.items()
        },
        "decision": (
            "pass_conditional_realizer_gate"
            if (
                route_realized == routes
                and route_exact == route_realized
                and subgoal_realized == subgoals
                and subgoal_exact == subgoal_realized
                and fallback == 0
            )
            else "fail_conditional_realizer_gate"
        ),
        "limitations": [
            "Targets and audited bindings are answer-known training data.",
            "Subgoal targets are checked inside their coordinated complete-goal endpoint; shared boundary roles make isolated-subgoal execution a different problem.",
            "The bounded compiler uses at most 12 children per expansion; this empirical gate does not prove complete search support for every representable goal.",
            "This does not measure autonomous subgoal proposal or task utility.",
        ],
    }
    publish_json(SUMMARY, body)
    print(json.dumps(body, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "status", "collect", "merge"))
    args = parser.parse_args()
    {"launch": launch, "status": status, "collect": collect, "merge": merge}[args.action]()


if __name__ == "__main__":
    main()
