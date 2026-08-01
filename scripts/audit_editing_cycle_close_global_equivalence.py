#!/usr/bin/env python3
"""Run the validation-only complete semantic cycle-close equivalence audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.experiments.editing_cycle_close_global_equivalence import (
    atomic_write_if_absent,
    audit_exact_validation_shard,
    build_plan,
    load_contract,
    load_json,
    pretty_json_bytes,
    validate_source_evidence,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACT = ROOT / "configs/editing_cycle_close_global_equivalence_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--packed-shard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_contract(args.contract)
    source_receipt, _ = validate_source_evidence(
        load_json(args.source_receipt, description="cycle-open V2 receipt"),
        load_json(args.source_result, description="cycle-open V2 result"),
        contract=contract,
    )
    plan = build_plan(
        contract,
        contract_path=args.contract,
        source_receipt_path=args.source_receipt,
        source_result_path=args.source_result,
        packed_shard_path=args.packed_shard,
        repo_root=ROOT,
    )
    receipt, result = audit_exact_validation_shard(
        contract,
        plan,
        source_receipt,
        packed_shard_path=args.packed_shard,
    )

    output = args.output_dir.resolve()
    if output == ROOT.resolve():
        raise ValueError("output directory must name a task-specific child")
    paths = {
        "plan": output / f"plan.{plan['plan_sha256']}.json",
        "receipt": output / f"receipt.{plan['plan_sha256']}.json",
        "result": output / f"result.{plan['plan_sha256']}.json",
    }
    atomic_write_if_absent(paths["plan"], pretty_json_bytes(plan))
    atomic_write_if_absent(paths["receipt"], pretty_json_bytes(receipt))
    atomic_write_if_absent(paths["result"], pretty_json_bytes(result))
    print(
        json.dumps(
            {
                "phase": "cycle_close_global_equivalence_complete",
                "training_launched": False,
                "plan_path": str(paths["plan"]),
                "plan_sha256": plan["plan_sha256"],
                "receipt_path": str(paths["receipt"]),
                "receipt_sha256": receipt["receipt_sha256"],
                "result_path": str(paths["result"]),
                "result_sha256": result["result_sha256"],
                "passed": result["equivalence_gate"]["passed"],
                "unique_sources": result["unique_source_count"],
                "candidate_count": result["totals"]["candidate_count"],
                "admitted_count": result["totals"]["admitted_count"],
                "ambiguous_rejection_count": result["totals"][
                    "ambiguous_rejection_count"
                ],
                "mismatch_count": result["totals"]["mismatch_count"],
                "oracle_overflow_source_count": result["oracle_overflow_source_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
