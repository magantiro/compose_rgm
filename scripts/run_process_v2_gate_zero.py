"""Run the Process-V2 Gate-0 reducer over a published Active8 run.

Gate 0 is a small deterministic post-Active8 reducer, so this driver is a
one-shot local command: bind the frozen contracts, resolve every role's decision
metadata, read the decision-eligible shards once each, and publish
``DECISION.json``.  It launches nothing remote and opens no molecular cache
chunk.

    PYTHONPATH=src:scripts python scripts/run_process_v2_gate_zero.py \
        --active8-run-root <root> --gate-zero-root <root>

Exits 0 on PASS, 1 on FAIL, 2 on a refusal.  A FAIL is a completed result: the
decision is published either way, and no outcome authorizes anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compose_v4.data.editing_v2_process_v2_gate_zero import (
    ProcessV2GateZeroError,
    run_gate_zero,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active8-run-root", type=Path, required=True)
    parser.add_argument("--gate-zero-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--print-decision",
        action="store_true",
        help="print the whole decision instead of its summary",
    )
    args = parser.parse_args(argv)

    try:
        decision = run_gate_zero(
            args.active8_run_root,
            gate_zero_root=args.gate_zero_root,
            repo_root=args.repo_root,
        )
    except ProcessV2GateZeroError as error:
        print(f"gate-0 refused: {error}", file=sys.stderr)
        return 2

    if args.print_decision:
        print(json.dumps(decision, indent=2, sort_keys=True))
    else:
        summary = {
            "decision": decision["decision"],
            "decision_sha256": decision["decision_sha256"],
            "accounting": decision["accounting"],
            "missing_active8_families": decision["missing_active8_families"],
            "missing_required_cells": decision["missing_required_cells"],
            "total_violations": decision["total_violations"],
            "failed_checks": sorted(
                name for name, value in decision["checks"].items() if value is not True
            ),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if decision["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
