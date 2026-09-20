#!/usr/bin/env python3
"""Zero-oracle dependency-region audit for the PMO route corpus.

This mirrors the T4 route abstraction: primitive traces are represented as
connected dependency regions with ordered microstreams and exact executor
replay.  The audit only consumes frozen route supervision; it never loads an
oracle, task scores, or controller outputs.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from compose_v4.control.dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "diagnostics/pmo_route_distillation/attempt_1/training_dataset.json.gz"
OUT = ROOT / "diagnostics/pmo_route_distillation/dependency_region_audit_v1/result.json"
CONFIG = DependencyRegionConfig(runtime_maximum_primitives=32, runtime_maximum_components=8)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_records(manifest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    by_source: dict[str, set[str]] = collections.defaultdict(set)
    for route in manifest:
        for member in route.get("members", []):
            by_source[member["source_path"]].add(member["member_id"])

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for source_path in sorted(by_source):
        path = ROOT / source_path
        try:
            payload = _load_json(path).get("payload", _load_json(path))
        except Exception as error:  # pragma: no cover - frozen source validation
            errors.append({"source_path": source_path, "stage": "load", "error": str(error)})
            continue
        wanted = by_source[source_path]
        candidates: list[tuple[str, str | None, dict[str, Any]]] = []
        tasks = payload.get("tasks")
        if isinstance(tasks, dict):
            for task_name in sorted(tasks):
                for program in tasks[task_name].get("programs", []):
                    candidate_id = program.get("candidate_id")
                    if candidate_id in wanted:
                        candidates.append((str(candidate_id), task_name, program))
        if isinstance(payload.get("programs"), list):
            for program in payload["programs"]:
                candidate_id = program.get("candidate_id") or program.get("variant_id")
                if candidate_id in wanted:
                    candidates.append((str(candidate_id), program.get("target"), program))
        source = _load_json(path)
        if "result" in source and not candidates:
            candidates.append(("current_best", source.get("source_id"), source["result"]))

        seen: set[tuple[str, str | None]] = set()
        for member_id, task_name, program in candidates:
            key = (member_id, task_name)
            if key in seen:
                continue
            seen.add(key)
            receipt = program.get("receipt", program)
            states = receipt.get("states")
            actions = receipt.get("actions")
            if states is None or actions is None:
                errors.append({"source_path": source_path, "member_id": member_id, "stage": "missing_trace"})
                continue
            try:
                region = dependency_region_program(tuple(states), tuple(actions), CONFIG)
            except Exception as error:  # pragma: no cover - frozen source validation
                errors.append({
                    "source_path": source_path,
                    "member_id": member_id,
                    "stage": "decompose",
                    "error": f"{type(error).__name__}: {error}",
                })
                continue
            records.append({
                "source_path": source_path,
                "member_id": member_id,
                "task": task_name,
                "primitive_transitions": region["primitive_transitions"],
                "component_count": region["component_count"],
                "component_sizes": [item["primitive_count"] for item in region["components"]],
                "component_run_count": region["component_run_count"],
                "component_reentries": region["component_reentries"],
                "created_dependency_edges": len(region["created_dependency_edges"]),
                "cycle_dependency_edges": len(region["cycle_dependency_edges"]),
                "cross_component_created_dependency_edges": region["cross_component_created_dependency_edges"],
                "cross_component_cycle_dependency_edges": region["cross_component_cycle_dependency_edges"],
                "exact_replay": region["exact_replay"],
                "runtime_length_supported": region["runtime_length_supported"],
                "complete_representation_supported": region["complete_representation_supported"],
                "abstention_reason": region["abstention_reason"],
            })
    return records, errors


def _distribution(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(collections.Counter(str(row[field]) for row in records).items(), key=lambda item: int(item[0])))


def run() -> dict[str, Any]:
    dataset_hash = _sha(DATASET)
    with gzip.open(DATASET, "rt") as handle:
        dataset = json.load(handle)
    payload = dataset["payload"]
    records, errors = _resolve_records(payload["route_manifest"])
    summary = {
        "routes_in_manifest": len(payload["route_manifest"]),
        "resolved_routes": len(records),
        "decomposition_errors": len(errors),
        "exact_replay_routes": sum(row["exact_replay"] for row in records),
        "runtime_length_supported_routes": sum(row["runtime_length_supported"] for row in records),
        "complete_representation_supported_routes": sum(row["complete_representation_supported"] for row in records),
        "primitive_transition_distribution": _distribution(records, "primitive_transitions"),
        "component_count_distribution": _distribution(records, "component_count"),
        "component_run_count_distribution": _distribution(records, "component_run_count"),
        "component_reentry_routes": sum(row["component_reentries"] > 0 for row in records),
        "created_dependency_edges": sum(row["created_dependency_edges"] for row in records),
        "cycle_dependency_edges": sum(row["cycle_dependency_edges"] for row in records),
        "cross_component_created_dependency_edges": sum(row["cross_component_created_dependency_edges"] for row in records),
        "cross_component_cycle_dependency_edges": sum(row["cross_component_cycle_dependency_edges"] for row in records),
    }
    artifact = {
        "schema_version": "pmo_dependency_region_audit_v1",
        "status": "ZERO_ORACLE_COMPUTED",
        "claim": "Primitive PMO routes can be represented as exact dependency regions within declared runtime support.",
        "input": {
            "dataset_path": str(DATASET.relative_to(ROOT)),
            "dataset_sha256": dataset_hash,
            "dataset_payload_sha256": dataset.get("payload_sha256"),
            "config": {
                "runtime_maximum_primitives": CONFIG.runtime_maximum_primitives,
                "runtime_maximum_components": CONFIG.runtime_maximum_components,
                "join_lifetime_neighbors": CONFIG.join_lifetime_neighbors,
            },
        },
        "summary": summary,
        "routes": sorted(records, key=lambda row: (row["source_path"], row["member_id"], row["task"] or "")),
        "errors": sorted(errors, key=lambda row: (row["source_path"], row.get("member_id", ""), row["stage"])),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(OUT)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    global OUT
    OUT = args.output if args.output.is_absolute() else ROOT / args.output
    artifact = run()
    print(json.dumps(artifact["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
