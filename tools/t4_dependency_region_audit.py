"""Audit the effective dependency-region horizon of the 77 T4 teacher routes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.control.dependency_region_program import (
    SCHEMA as PROGRAM_SCHEMA,
)
from compose_v4.control.dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
    dependency_region_summary,
)
from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_matched_pilot import unseal
from tools.t4_program_vocabulary_audit import source_group_map
from tools.t4_route_distillation import _teacher_traces

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_dependency_region_audit_v1.json"
T4_CONTRACT = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
LIBRARY = (
    ROOT / "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json"
)
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"
OUTPUT = ROOT / "diagnostics/t4_dependency_regions/attempt_1/result.json"
RESULT_SCHEMA = "t4_dependency_region_audit_result_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sealed(path: Path) -> dict:
    raw = (
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )
    envelope = json.loads(raw)
    payload = envelope.get("payload")
    claimed = envelope.get("payload_sha256", envelope.get("contract_sha256"))
    if not isinstance(payload, dict) or claimed != identity(payload):
        raise ValueError(f"artifact is not self-hashed: {path}")
    return payload


def publish(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite dependency-region artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def git_revision() -> tuple[str, bool]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    return revision, dirty


def _load_inputs() -> tuple[dict, list[dict], dict[str, dict]]:
    contract = load_sealed(CONTRACT)
    if contract["oracle"]["calls_authorized"] != 0:
        raise ValueError("dependency-region audit unexpectedly authorizes oracle calls")
    for name, path in (
        ("teacher_library", LIBRARY),
        ("source_registry", SEEDS),
        ("t4_contract", T4_CONTRACT),
    ):
        if sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"dependency-region input hash changed: {path}")
    library = json.loads(LIBRARY.read_text())
    traces, exclusions = _teacher_traces(library)
    if exclusions or len(traces) != contract["census"]["teacher_routes"]:
        raise RuntimeError(
            f"teacher census changed: {len(traces)} admitted, {exclusions}"
        )
    metadata = source_group_map(unseal(T4_CONTRACT), json.loads(SEEDS.read_text()))
    return contract, traces, metadata


def _annotated_routes(
    traces: list[dict],
    metadata: dict[str, dict],
    *,
    join_lifetime_neighbors: bool,
    maximum_regions: int,
) -> list[dict]:
    config = DependencyRegionConfig(
        runtime_maximum_primitives=32,
        runtime_maximum_components=maximum_regions,
        join_lifetime_neighbors=join_lifetime_neighbors,
    )
    rows = []
    for teacher in traces:
        trace = teacher["trace"]
        program = dependency_region_program(
            tuple(trace["states"]), tuple(trace["actions"]), config
        )
        source = metadata[teacher["source_group"]]
        rows.append(
            {
                "program_id": teacher["program_id"],
                "source_group": teacher["source_group"],
                "cell": source["cell"],
                "target": source["target"],
                "source_idx": source["source_idx"],
                "teacher_receipt": teacher["receipt"],
                "teacher_receipt_sha256": teacher["receipt_sha256"],
                "dependency_region_program": program,
            }
        )
    return rows


def _group_summaries(routes: list[dict], field: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for route in routes:
        groups[str(route[field])].append(route)
    return {
        key: dependency_region_summary(rows) for key, rows in sorted(groups.items())
    }


def _compression(summary: dict) -> dict:
    routes = summary["routes"]
    distribution = summary["component_count_distribution"]

    def fraction_at_most(maximum: int) -> float:
        return (
            sum(count for value, count in distribution.items() if int(value) <= maximum)
            / routes
        )

    return {
        "total_primitives_per_region": (
            summary["primitive_transitions"] / summary["components"]
        ),
        "routes_at_most_1_region": fraction_at_most(1),
        "routes_at_most_2_regions": fraction_at_most(2),
        "routes_at_most_3_regions": fraction_at_most(3),
        "routes_at_most_4_regions": fraction_at_most(4),
        "routes_at_most_8_regions": fraction_at_most(8),
    }


def run(output: Path) -> None:
    contract, traces, metadata = _load_inputs()
    maximum_regions = int(contract["definitions"]["maximum_regions"])
    primary = _annotated_routes(
        traces,
        metadata,
        join_lifetime_neighbors=True,
        maximum_regions=maximum_regions,
    )
    strict = _annotated_routes(
        traces,
        metadata,
        join_lifetime_neighbors=False,
        maximum_regions=maximum_regions,
    )
    primary_summary = dependency_region_summary(primary)
    strict_summary = dependency_region_summary(strict)
    if (
        primary_summary["exact_replay_precision"]
        != contract["gates"]["exact_replay_precision"]
        or strict_summary["exact_replay_precision"]
        != contract["gates"]["exact_replay_precision"]
    ):
        raise RuntimeError("dependency-region exact replay gate failed")
    revision, dirty = git_revision()
    payload = {
        "schema_version": RESULT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "program_schema_version": PROGRAM_SCHEMA,
        "contract": {
            "path": str(CONTRACT.relative_to(ROOT)),
            "sha256": sha256(CONTRACT),
            "payload_sha256": identity(contract),
        },
        "inputs": {
            "teacher_library": {
                "path": str(LIBRARY.relative_to(ROOT)),
                "sha256": sha256(LIBRARY),
            },
            "source_registry": {
                "path": str(SEEDS.relative_to(ROOT)),
                "sha256": sha256(SEEDS),
            },
            "t4_contract": {
                "path": str(T4_CONTRACT.relative_to(ROOT)),
                "sha256": sha256(T4_CONTRACT),
            },
            "teacher_receipts": {
                route["teacher_receipt"]: route["teacher_receipt_sha256"]
                for route in primary
            },
        },
        "software": {
            "git_revision": revision,
            "git_dirty": dirty,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "scope": {
            "evidence": "retrospective_answer_known_zero_oracle_diagnostic",
            "new_oracle_calls": 0,
            "model_training": False,
            "decoder_changed": False,
            "live_runs_changed": False,
        },
        "definitions": contract["definitions"],
        "primary_lifetime_neighbor": {
            "summary": primary_summary,
            "compression": _compression(primary_summary),
            "by_target": _group_summaries(primary, "target"),
            "by_source": _group_summaries(primary, "cell"),
            "routes": primary,
        },
        "strict_shared_handle": {
            "summary": strict_summary,
            "compression": _compression(strict_summary),
            "by_target": _group_summaries(strict, "target"),
            "by_source": _group_summaries(strict, "cell"),
            "routes": strict,
        },
        "interpretation_boundary": (
            "Dependency connectivity is a computed horizon proxy, not evidence "
            "that a learned subgoal policy will generate or optimize these regions."
        ),
    }
    publish(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
