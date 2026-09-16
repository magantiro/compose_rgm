"""Zero-oracle gates for the complete dependency-region T4 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from compose_v4.control.complete_region_program import (
    CompleteRegionProgram,
    execute_complete_region_program,
    program_from_structural_goal,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal import extract_structural_goal
from compose_v4.control.structural_subgoal_realizer import RealizerConfig
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.t4_structural_subgoal_audit import teacher_traces

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/t4_complete_region_runtime_v1.json"
DEFAULT_OUTPUT = ROOT / "diagnostics/t4_complete_region_runtime/attempt_1"
SUPPORT_SCHEMA = "t4_complete_region_runtime_support_v1"
ROUTE_SCHEMA = "t4_complete_region_runtime_route_result_v1"
SUMMARY_SCHEMA = "t4_complete_region_runtime_summary_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite complete-region artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def _unseal(path: Path) -> dict:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != identity(
        payload
    ):
        raise ValueError(f"sealed payload identity changed: {path}")
    return payload


def load_contract(path: Path) -> dict:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("contract_sha256") != identity(
        payload
    ):
        raise ValueError(f"complete-region contract is not self-hashed: {path}")
    if any(payload["costs"][name] != 0 for name in payload["costs"]):
        raise ValueError("complete-region contract authorizes a nonzero external cost")
    for row in payload["inputs"].values():
        source = ROOT / row["path"]
        if sha256_file(source) != row["sha256"]:
            raise ValueError(f"complete-region input hash changed: {row['path']}")
    structural = json.loads(
        (ROOT / payload["inputs"]["structural_goal_summary"]["path"]).read_text()
    )
    for relative, expected in structural["source_shards"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"structural source shard hash changed: {relative}")
    return payload


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean() -> str:
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError(
            "complete-region authoritative gates require clean committed source"
        )
    return _revision()


def _program(trace: dict) -> tuple[CompleteRegionProgram, tuple[tuple[int, ...], ...]]:
    goal, bindings, _ = extract_structural_goal(
        tuple(trace["states"]), tuple(trace["actions"])
    )
    program = program_from_structural_goal(goal)
    if CompleteRegionProgram.from_payload(program.payload()) != program:
        raise AssertionError("complete-region program payload is not round-trip stable")
    forbidden = {
        "actions",
        "absolute_atom_address",
        "endpoint",
        "executor_rule",
        "route_id",
        "source_graph",
        "target_id",
    }
    serialized_keys = set()

    def visit(value) -> None:
        if isinstance(value, dict):
            serialized_keys.update(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(program.payload())
    overlap = forbidden & serialized_keys
    if overlap:
        raise AssertionError(
            f"runtime program leaked forbidden fields: {sorted(overlap)}"
        )
    return program, bindings


def support_gate(output: Path, *, contract_path: Path) -> dict:
    contract = load_contract(contract_path)
    revision = _require_clean()
    rows = []
    counts = Counter()
    for teacher in sorted(teacher_traces(), key=lambda row: row["program_id"]):
        program, bindings = _program(teacher["trace"])
        controls = [row.control_after for row in program.decisions]
        supported = (
            1 <= len(program.decisions) <= 4
            and controls[-1] == "stop"
            and all(value == "continue" for value in controls[:-1])
            and len(bindings) == len(program.decisions)
        )
        counts[len(program.decisions)] += 1
        rows.append(
            {
                "teacher_program_id": teacher["program_id"],
                "teacher_receipt_sha256": teacher["receipt_sha256"],
                "runtime_program_id": program.program_id,
                "region_decisions": len(program.decisions),
                "controls": controls,
                "runtime_support": supported,
                "runtime_program": program.payload(),
                "resolved_binding_rows": len(bindings),
                "teacher_primitive_actions_in_runtime_program": 0,
            }
        )
    covered = sum(row["runtime_support"] for row in rows)
    payload = {
        "schema_version": SUPPORT_SCHEMA,
        "evidence": "computed answer-known zero-oracle runtime-support gate",
        "contract": {
            "path": str(contract_path.relative_to(ROOT)),
            "sha256": sha256_file(contract_path),
            "payload_sha256": identity(contract),
        },
        "implementation_revision": revision,
        "census": {
            "routes": len(rows),
            "regions": sum(row["region_decisions"] for row in rows),
            "region_count_distribution": {
                str(key): value for key, value in sorted(counts.items())
            },
        },
        "gate": {
            "covered": covered,
            "denominator": len(rows),
            "coverage": covered / len(rows),
            "precision": 1.0 if covered else None,
        },
        "rows": rows,
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "network_calls": 0,
            "gpu_seconds": 0,
        },
    }
    _publish(output, payload)
    return payload


def _route_result(teacher: dict, *, revision: str, config_payload: dict) -> dict:
    trace = teacher["trace"]
    program, bindings = _program(trace)
    source = decode_state(trace["states"][0])
    target = decode_state(trace["states"][-1])
    config = RealizerConfig(**config_payload)
    execution = execute_complete_region_program(
        source,
        program,
        resolved_bindings=bindings,
        config=config,
    )
    committed = execution["committed_endpoint_state"]
    exact = committed is not None and canonical_state_key(
        decode_state(committed)
    ) == canonical_state_key(target)
    return {
        "schema_version": ROUTE_SCHEMA,
        "evidence": "answer-known zero-oracle complete-region runtime realization",
        "implementation_revision": revision,
        "teacher_program_id": teacher["program_id"],
        "teacher_receipt_sha256": teacher["receipt_sha256"],
        "runtime_program_id": program.program_id,
        "runtime_program": program.payload(),
        "teacher_primitive_count": len(trace["actions"]),
        "region_decisions": len(program.decisions),
        "resolved_binding_membership_checked": True,
        "resolved_binding_sha256": identity([list(row) for row in bindings]),
        "execution": execution,
        "exact_endpoint_reconstruction": exact,
        "realization_precision": (
            float(exact) if execution["committed_endpoint_count"] else None
        ),
        "configuration": config_payload,
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "network_calls": 0,
            "gpu_seconds": 0,
        },
    }


def _route_worker(arguments: tuple[dict, str, dict, str]) -> dict:
    teacher, revision, config_payload, output_text = arguments
    output = Path(output_text)
    result = _route_result(teacher, revision=revision, config_payload=config_payload)
    _publish(output, result)
    return {
        "teacher_program_id": teacher["program_id"],
        "status": result["execution"]["status"],
        "exact": result["exact_endpoint_reconstruction"],
        "path": output_text,
        "sha256": sha256_file(output),
    }


def realize_all(
    output_root: Path,
    *,
    contract_path: Path,
    workers: int,
    config_payload: dict,
) -> list[dict]:
    load_contract(contract_path)
    revision = _require_clean()
    if workers < 1:
        raise ValueError("workers must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    teachers = sorted(teacher_traces(), key=lambda row: row["program_id"])
    outputs = [output_root / f"{row['program_id']}.json" for row in teachers]
    if any(path.exists() for path in outputs):
        raise ValueError("complete-region route output already exists")
    arguments = [
        (teacher, revision, config_payload, str(output))
        for teacher, output in zip(teachers, outputs, strict=True)
    ]
    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_route_worker, row) for row in arguments]
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            elapsed = time.monotonic() - started
            rate = completed / elapsed
            print(
                json.dumps(
                    {
                        "event": "complete_region_route_finished",
                        "completed": completed,
                        "total": len(futures),
                        "exact": row["exact"],
                        "status": row["status"],
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": (len(futures) - completed)
                        / rate,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return sorted(rows, key=lambda row: row["teacher_program_id"])


def aggregate(output: Path, *, route_root: Path, contract_path: Path) -> dict:
    contract = load_contract(contract_path)
    revision = _revision()
    rows = []
    for path in sorted(route_root.glob("*.json")):
        payload = _unseal(path)
        rows.append(
            {
                "teacher_program_id": payload["teacher_program_id"],
                "runtime_program_id": payload["runtime_program_id"],
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "payload_sha256": identity(payload),
                "status": payload["execution"]["status"],
                "exact_endpoint_reconstruction": payload[
                    "exact_endpoint_reconstruction"
                ],
                "primitive_teacher_actions_used": payload["execution"][
                    "primitive_teacher_actions_used"
                ],
                "region_decisions": payload["region_decisions"],
                "realized_primitive_count": payload["execution"].get(
                    "realized_primitive_count", 0
                ),
            }
        )
    if len(rows) != 77 or len({row["teacher_program_id"] for row in rows}) != 77:
        raise ValueError(f"expected 77 distinct route results, found {len(rows)}")
    committed = sum(row["status"] == "committed" for row in rows)
    exact = sum(row["exact_endpoint_reconstruction"] for row in rows)
    no_teacher = sum(row["primitive_teacher_actions_used"] == 0 for row in rows)
    payload = {
        "schema_version": SUMMARY_SCHEMA,
        "evidence": "computed zero-oracle complete-region runtime gate",
        "contract": {
            "path": str(contract_path.relative_to(ROOT)),
            "sha256": sha256_file(contract_path),
            "payload_sha256": identity(contract),
        },
        "implementation_revision": revision,
        "census": {
            "routes": len(rows),
            "regions": sum(row["region_decisions"] for row in rows),
            "status_counts": dict(
                sorted(Counter(row["status"] for row in rows).items())
            ),
        },
        "gates": {
            "runtime_realization_coverage": {
                "covered": committed,
                "denominator": len(rows),
                "coverage": committed / len(rows),
            },
            "exact_endpoint_precision": {
                "exact": exact,
                "committed": committed,
                "precision": exact / max(1, committed),
            },
            "zero_teacher_action_fallback": {
                "covered": no_teacher,
                "denominator": len(rows),
                "coverage": no_teacher / len(rows),
            },
        },
        "route_results": rows,
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "network_calls": 0,
            "gpu_seconds": 0,
        },
        "deferred": {
            "learned_factor_rank": "requires a later split-first complete-patch modeling contract",
            "autonomous_generation": "requires a later split-first complete-region generator contract",
        },
    }
    _publish(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    commands = parser.add_subparsers(dest="command", required=True)
    support = commands.add_parser("support")
    support.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "support.json")
    realize = commands.add_parser("realize-all")
    realize.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT / "routes")
    realize.add_argument("--workers", type=int, default=4)
    realize.add_argument("--maximum-expansions", type=int, default=16_384)
    realize.add_argument("--children-per-expansion", type=int, default=12)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--route-root", type=Path, default=DEFAULT_OUTPUT / "routes")
    combine.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "result.json")
    arguments = parser.parse_args()
    contract_path = arguments.contract.resolve()
    if arguments.command == "support":
        result = support_gate(arguments.output.resolve(), contract_path=contract_path)
        print(
            json.dumps(
                {"census": result["census"], "gate": result["gate"]}, sort_keys=True
            )
        )
    elif arguments.command == "realize-all":
        realize_all(
            arguments.output_root.resolve(),
            contract_path=contract_path,
            workers=arguments.workers,
            config_payload={
                "maximum_primitives": 32,
                "maximum_active_atoms": 40,
                "maximum_expansions": arguments.maximum_expansions,
                "children_per_expansion": arguments.children_per_expansion,
            },
        )
    else:
        result = aggregate(
            arguments.output.resolve(),
            route_root=arguments.route_root.resolve(),
            contract_path=contract_path,
        )
        print(
            json.dumps(
                {"census": result["census"], "gates": result["gates"]}, sort_keys=True
            )
        )


if __name__ == "__main__":
    main()
