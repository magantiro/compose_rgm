#!/usr/bin/env python3
"""Run the prospective validation-only complete cycle-open equivalence audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.experiments.editing_cycle_open_global_equivalence import (
    atomic_write_if_absent,
    audit_exact_validation_shard,
    build_plan,
    load_contract,
    load_json,
    pretty_json_bytes,
    validate_v1_receipt,
    validate_v1_result,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACT = ROOT / "configs/editing_cycle_open_global_equivalence_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--v1-receipt", type=Path, required=True)
    parser.add_argument("--v1-result", type=Path, required=True)
    parser.add_argument("--packed-shard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_contract(args.contract)
    raw_v1_receipt = load_json(args.v1_receipt, description="V1 equivalence receipt")
    v1_receipt = validate_v1_receipt(raw_v1_receipt, contract=contract)
    raw_v1_result = load_json(args.v1_result, description="V1 equivalence result")
    validate_v1_result(raw_v1_result, contract=contract)
    plan = build_plan(
        contract,
        contract_path=args.contract,
        v1_receipt_path=args.v1_receipt,
        v1_result_path=args.v1_result,
        packed_shard_path=args.packed_shard,
        repo_root=ROOT,
    )
    receipt, result = audit_exact_validation_shard(
        contract,
        plan,
        v1_receipt,
        packed_shard_path=args.packed_shard,
    )

    output = args.output_dir.resolve()
    if output == ROOT.resolve():
        raise ValueError("output directory must name a task-specific child")
    plan_path = output / f"plan.{plan['plan_sha256']}.json"
    receipt_path = output / f"receipt.{plan['plan_sha256']}.json"
    result_path = output / f"result.{plan['plan_sha256']}.json"
    atomic_write_if_absent(plan_path, pretty_json_bytes(plan))
    atomic_write_if_absent(receipt_path, pretty_json_bytes(receipt))
    atomic_write_if_absent(result_path, pretty_json_bytes(result))
    print(
        json.dumps(
            {
                "phase": "cycle_open_global_equivalence_complete",
                "training_launched": False,
                "plan_path": str(plan_path),
                "plan_sha256": plan["plan_sha256"],
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt["receipt_sha256"],
                "result_path": str(result_path),
                "result_sha256": result["result_sha256"],
                "passed": result["equivalence_gate"]["passed"],
                "unique_sources": result["unique_source_count"],
                "semantic_edges": result["totals"]["semantic_aromatic_edge_count"],
                "mismatch_edges": result["totals"]["mismatch_edge_count"],
                "oracle_overflow_edges": result["totals"]["oracle_overflow_edge_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
