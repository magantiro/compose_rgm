#!/usr/bin/env python3
"""Plan, map, and reduce the validation-only cycle-open equivalence audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compose_v4.data.active8_trace_inventory import (  # noqa: E402
    load_active8_trace_admission,
)
from compose_v4.data.immutable_artifact import write_bytes_if_absent  # noqa: E402
from compose_v4.data.packed_charge_policy_audit import (  # noqa: E402
    resolve_unified_manifest_shards,
)
from compose_v4.data.packed_trace_store import (  # noqa: E402
    read_frozen_source_addressed_packed_shard,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (  # noqa: E402
    CycleOpenComponentEquivalenceError,
    audit_one_addressed_shard,
    file_sha256,
    load_contract,
    reduce_shard_receipts,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence_runtime import (  # noqa: E402
    active8_partition_census,
    build_plan,
    implementation_identity,
    load_json_artifact,
    pretty_json_bytes,
    validate_parent_admission,
    validate_plan,
)


DEFAULT_CONTRACT = ROOT / "configs/editing_cycle_open_component_equivalence_v1.json"


def _git_revision() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CycleOpenComponentEquivalenceError(
            "cannot bind equivalence audit to a Git revision"
        ) from error
    if dirty:
        raise CycleOpenComponentEquivalenceError(
            "authoritative equivalence audit requires a clean committed tree"
        )
    return {"commit": commit, "tree_dirty": False}


def _common_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--active8-inventory", type=Path, required=True)
    parser.add_argument("--unified-manifest", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="freeze positive-support validation shard tasks")
    _common_sources(plan)
    plan.add_argument("--output", type=Path, required=True)

    shard = commands.add_parser("shard", help="audit one exact planned physical shard")
    _common_sources(shard)
    shard.add_argument("--plan", type=Path, required=True)
    shard.add_argument("--task-index", type=int, required=True)
    shard.add_argument("--receipt-dir", type=Path, required=True)

    reducer = commands.add_parser("reduce", help="reduce the exact planned receipt set")
    reducer.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    reducer.add_argument("--active8-inventory", type=Path, required=True)
    reducer.add_argument("--plan", type=Path, required=True)
    reducer.add_argument("--receipt-dir", type=Path, required=True)
    reducer.add_argument("--output", type=Path, required=True)
    return parser


def _load_contract_and_admission(contract_path: Path, inventory_path: Path):
    contract = load_contract(contract_path)
    parent = contract["parent_active8_identity"]
    admission = load_active8_trace_admission(
        inventory_path,
        expected_manifest_file_sha256=parent["inventory_manifest_file_sha256"],
        expected_inventory_sha256=parent["inventory_sha256"],
        expected_effective_source_corpus_cache_sha256=parent[
            "effective_source_corpus_cache_sha256"
        ],
        expected_support_contract_sha256=parent["support_contract_sha256"],
    )
    validate_parent_admission(contract, admission)
    return contract, admission


def _plan_inputs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "contract_path": str(args.contract),
        "active8_inventory_path": str(args.active8_inventory),
        "unified_manifest_path": str(args.unified_manifest),
        "audit_root": str(args.audit_root),
        "mmp_root": str(args.mmp_root),
    }


def _build_live_plan(args: argparse.Namespace):
    contract, admission = _load_contract_and_admission(args.contract, args.active8_inventory)
    _manifest, declared = resolve_unified_manifest_shards(
        args.unified_manifest,
        audit_root=args.audit_root,
        mmp_root=args.mmp_root,
    )
    plan = build_plan(
        contract,
        admission,
        declared,
        unified_manifest_file_sha256=file_sha256(args.unified_manifest),
        inputs=_plan_inputs(args),
        code_revision=_git_revision(),
        implementation=implementation_identity(repo_root=ROOT),
    )
    return contract, admission, declared, plan


def _matching_shard(declared, task):
    matching = tuple(
        shard
        for shard in declared
        if (
            shard.envelope_layer == task["layer"]
            and shard.partition == task["partition"]
            and shard.relative_path == task["relative_path"]
            and shard.path.name == task["packed_shard_name"]
        )
    )
    if len(matching) != 1:
        raise CycleOpenComponentEquivalenceError(
            "planned task does not resolve to exactly one physical shard"
        )
    return matching[0]


def _plan_command(args: argparse.Namespace) -> None:
    _contract, _admission, _declared, plan = _build_live_plan(args)
    write_bytes_if_absent(args.output, pretty_json_bytes(plan))
    print(
        json.dumps(
            {
                "phase": "cycle_open_component_equivalence_plan_complete",
                "plan_sha256": plan["plan_sha256"],
                "task_count": plan["task_count"],
                "zero_support_shards": len(plan["zero_cycle_attach_support_shards"]),
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _shard_command(args: argparse.Namespace) -> None:
    contract, admission, declared, live_plan = _build_live_plan(args)
    frozen_plan = validate_plan(
        load_json_artifact(args.plan, description="cycle-open equivalence plan"),
        expected_contract_sha256=contract["contract_sha256"],
    )
    if frozen_plan != live_plan:
        raise CycleOpenComponentEquivalenceError(
            "frozen plan differs from live immutable inputs or implementation"
        )
    if args.task_index < 0 or args.task_index >= frozen_plan["task_count"]:
        raise CycleOpenComponentEquivalenceError("task index lies outside the plan")
    task = frozen_plan["tasks"][args.task_index]
    shard = _matching_shard(declared, task)
    rows = read_frozen_source_addressed_packed_shard(
        shard.path,
        expected_shard_sha256=task["packed_shard_content_sha256"],
        expected_manifest_sha256=task["packed_manifest_sha256"],
        expected_overlay_sha256=task["packed_provenance_overlay_sha256"],
        verify_fraction=0.0,
    )
    receipt = audit_one_addressed_shard(
        task,
        rows,
        admission,
        plan_sha256=frozen_plan["plan_sha256"],
        implementation_sha256=frozen_plan["implementation"]["implementation_sha256"],
        maximum_oracle_structures=contract["comparison_policy"]["maximum_oracle_structures"],
    )
    destination = args.receipt_dir / task["receipt_filename"]
    write_bytes_if_absent(destination, pretty_json_bytes(receipt))
    print(
        json.dumps(
            {
                "phase": "cycle_open_component_equivalence_shard_complete",
                "task_index": args.task_index,
                "unique_sources": len(receipt["source_records"]),
                "receipt_sha256": receipt["receipt_sha256"],
                "output": str(destination),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _reduce_command(args: argparse.Namespace) -> None:
    contract, admission = _load_contract_and_admission(args.contract, args.active8_inventory)
    plan = validate_plan(
        load_json_artifact(args.plan, description="cycle-open equivalence plan"),
        expected_contract_sha256=contract["contract_sha256"],
    )
    if (
        plan["code_revision"] != _git_revision()
        or plan["implementation"] != implementation_identity(repo_root=ROOT)
        or plan["active8_partition_census"] != active8_partition_census(admission)
    ):
        raise CycleOpenComponentEquivalenceError(
            "reducer revision, implementation, or Active8 census differs from the plan"
        )
    expected_names = {task["receipt_filename"] for task in plan["tasks"]}
    observed_names = (
        {path.name for path in args.receipt_dir.iterdir() if path.suffix == ".json"}
        if args.receipt_dir.is_dir()
        else set()
    )
    if observed_names != expected_names:
        raise CycleOpenComponentEquivalenceError(
            "receipt directory differs from exact plan: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unexpected={sorted(observed_names - expected_names)}"
        )
    receipts = [
        load_json_artifact(
            args.receipt_dir / task["receipt_filename"],
            description=f"cycle-open equivalence receipt {task['task_index']}",
        )
        for task in plan["tasks"]
    ]
    evidence_identity = {
        "contract_sha256": contract["contract_sha256"],
        "parent_active8_identity": contract["parent_active8_identity"],
        "partitions": ["validation"],
        "excluded_partitions": ["train", "controller_validation", "test"],
        "source_scope": contract["source_scope"],
        "edge_scope": contract["edge_scope"],
    }
    result = reduce_shard_receipts(
        receipts,
        expected_tasks=plan["tasks"],
        plan_sha256=plan["plan_sha256"],
        implementation_sha256=plan["implementation"]["implementation_sha256"],
        evidence_identity=evidence_identity,
    )
    write_bytes_if_absent(args.output, pretty_json_bytes(result))
    print(
        json.dumps(
            {
                "phase": "cycle_open_component_equivalence_reduce_complete",
                "passed": result["equivalence_gate"]["passed"],
                "unique_sources": result["unique_source_count"],
                "semantic_edges": result["totals"]["semantic_aromatic_edge_count"],
                "result_sha256": result["result_sha256"],
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan":
        _plan_command(args)
    elif args.command == "shard":
        _shard_command(args)
    elif args.command == "reduce":
        _reduce_command(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
