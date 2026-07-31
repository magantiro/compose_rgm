#!/usr/bin/env python3
"""Plan, map, and reduce the frozen Active8 aromatic cycle-open impact audit."""

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
from compose_v4.experiments.editing_active8_aromatic_cycle_open_impact import (  # noqa: E402
    AromaticCycleOpenImpactAuditError,
    active8_partition_census,
    audit_one_shard,
    build_audit_plan,
    file_sha256,
    implementation_identity,
    load_audit_contract,
    load_json_artifact,
    pretty_json_bytes,
    reduce_shard_receipts,
    validate_audit_plan,
    validate_parent_admission,
)


DEFAULT_CONTRACT = ROOT / "configs/editing_active8_aromatic_cycle_open_impact_audit_v1.json"


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
        raise AromaticCycleOpenImpactAuditError(
            "cannot bind impact audit to a git revision"
        ) from error
    if dirty:
        raise AromaticCycleOpenImpactAuditError(
            "authoritative impact audit requires a clean committed scientific tree"
        )
    return {"commit": commit, "tree_dirty": False}


def _common_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--active8-inventory", type=Path, required=True)
    parser.add_argument("--unified-manifest", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read immutable validation-only Active8 shards and characterize "
            "aromatic bond_delete teachers under the non-authorizing semantic resolver."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="freeze one task per physical shard")
    _common_sources(plan)
    plan.add_argument("--output", type=Path, required=True)

    shard = commands.add_parser("shard", help="audit exactly one planned shard")
    _common_sources(shard)
    shard.add_argument("--plan", type=Path, required=True)
    shard.add_argument("--task-index", type=int, required=True)
    shard.add_argument("--receipt-dir", type=Path, required=True)

    reduce_parser = commands.add_parser(
        "reduce", help="reduce the exact immutable planned receipt set"
    )
    reduce_parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    reduce_parser.add_argument("--active8-inventory", type=Path, required=True)
    reduce_parser.add_argument("--plan", type=Path, required=True)
    reduce_parser.add_argument("--receipt-dir", type=Path, required=True)
    reduce_parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_contract_and_admission(contract_path: Path, inventory_path: Path):
    contract = load_audit_contract(contract_path)
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
    plan = build_audit_plan(
        contract,
        admission,
        declared,
        unified_manifest_file_sha256=file_sha256(args.unified_manifest),
        inputs=_plan_inputs(args),
        code_revision=_git_revision(),
        implementation=implementation_identity(repo_root=ROOT),
    )
    return contract, admission, declared, plan


def _plan_command(args: argparse.Namespace) -> None:
    _contract, _admission, _declared, plan = _build_live_plan(args)
    write_bytes_if_absent(args.output, pretty_json_bytes(plan))
    print(
        json.dumps(
            {
                "phase": "aromatic_cycle_open_impact_plan_complete",
                "plan_sha256": plan["plan_sha256"],
                "task_count": plan["task_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _shard_command(args: argparse.Namespace) -> None:
    contract, admission, declared, live_plan = _build_live_plan(args)
    frozen_plan = validate_audit_plan(
        load_json_artifact(args.plan, description="impact audit plan"),
        expected_contract_sha256=contract["contract_sha256"],
    )
    if frozen_plan != live_plan:
        raise AromaticCycleOpenImpactAuditError(
            "frozen impact plan differs from live immutable inputs or implementation"
        )
    if args.task_index < 0 or args.task_index >= frozen_plan["task_count"]:
        raise AromaticCycleOpenImpactAuditError("task index lies outside the plan")
    task = frozen_plan["tasks"][args.task_index]
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
        raise AromaticCycleOpenImpactAuditError(
            "planned task does not resolve to exactly one physical shard"
        )
    shard = matching[0]
    rows = read_frozen_source_addressed_packed_shard(
        shard.path,
        expected_shard_sha256=task["packed_shard_content_sha256"],
        expected_manifest_sha256=task["packed_manifest_sha256"],
        expected_overlay_sha256=task["packed_provenance_overlay_sha256"],
        verify_fraction=0.0,
    )
    receipt = audit_one_shard(
        task,
        rows,
        admission,
        contract=contract,
        plan_sha256=frozen_plan["plan_sha256"],
        implementation_sha256=frozen_plan["implementation"]["implementation_sha256"],
    )
    destination = args.receipt_dir / task["receipt_filename"]
    write_bytes_if_absent(destination, pretty_json_bytes(receipt))
    print(
        json.dumps(
            {
                "phase": "aromatic_cycle_open_shard_receipt_complete",
                "task_index": args.task_index,
                "receipt_sha256": receipt["receipt_sha256"],
                "affected_rows": receipt["counts"].get(
                    "perceived_aromatic_bond_delete_teacher_rows", 0
                ),
                "output": str(destination),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _reduce_command(args: argparse.Namespace) -> None:
    contract, admission = _load_contract_and_admission(args.contract, args.active8_inventory)
    plan = validate_audit_plan(
        load_json_artifact(args.plan, description="impact audit plan"),
        expected_contract_sha256=contract["contract_sha256"],
    )
    if plan["parent_active8_identity"] != contract["parent_active8_identity"] or plan[
        "active8_partition_census"
    ] != active8_partition_census(admission):
        raise AromaticCycleOpenImpactAuditError(
            "reducer plan Active8 parent or partition census disagrees"
        )
    if plan["code_revision"] != _git_revision() or plan["implementation"] != (
        implementation_identity(repo_root=ROOT)
    ):
        raise AromaticCycleOpenImpactAuditError(
            "reducer implementation or revision differs from the frozen plan"
        )
    for partition in contract["partitions"]:
        admission.assert_partition_shards(
            partition,
            (
                (task["layer"], task["partition"], task["packed_shard_name"])
                for task in plan["tasks"]
                if task["partition"] == partition
            ),
        )
    expected_names = {task["receipt_filename"] for task in plan["tasks"]}
    try:
        observed_names = {
            path.name for path in args.receipt_dir.iterdir() if path.suffix == ".json"
        }
    except OSError as error:
        raise AromaticCycleOpenImpactAuditError(
            f"cannot inspect receipt directory: {args.receipt_dir}"
        ) from error
    if observed_names != expected_names:
        raise AromaticCycleOpenImpactAuditError(
            "receipt directory differs from exact plan: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unexpected={sorted(observed_names - expected_names)}"
        )
    receipts = [
        load_json_artifact(
            args.receipt_dir / task["receipt_filename"],
            description=f"shard receipt {task['task_index']}",
        )
        for task in plan["tasks"]
    ]
    result = reduce_shard_receipts(
        plan,
        receipts,
        contract=contract,
        implementation_sha256=plan["implementation"]["implementation_sha256"],
    )
    write_bytes_if_absent(args.output, pretty_json_bytes(result))
    print(
        json.dumps(
            {
                "phase": "aromatic_cycle_open_impact_reduce_complete",
                "result_sha256": result["result_sha256"],
                "affected_rows": result["denominators"][
                    "perceived_aromatic_bond_delete_teacher_rows"
                ],
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
    else:  # pragma: no cover - argparse guarantees the command set
        raise AromaticCycleOpenImpactAuditError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
