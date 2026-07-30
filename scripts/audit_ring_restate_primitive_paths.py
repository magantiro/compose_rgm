"""Run the charge-clean whole-corpus ring-restatement primitive-path audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.experiments.ring_restate_primitive_path_audit import (
    RingRestateAuditInputs,
    audit_unified_ring_restate_teachers,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSFER_ROOT = Path(
    "/private/tmp/claude-502/"
    "-Users-rmaganti-Documents-Codex-2026-07-14-ok-so-"
    "compose-rgm-claude-generators/"
    "6d6fc94f-1f64-41f3-8db5-e0589f315b48/scratchpad/transfer"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every addressable packed ring_system_restate teacher against "
            "the production macro executor and authoritative primitive lowering."
        )
    )
    parser.add_argument("--transfer-root", type=Path, default=DEFAULT_TRANSFER_ROOT)
    parser.add_argument(
        "--charge-policy-audit",
        type=Path,
        default=(
            ROOT / "diagnostics" / "coherence" / "packed_charge_policy_audit_v1_2026-07-30.json"
        ),
    )
    parser.add_argument(
        "--charge-policy-exclusions",
        type=Path,
        default=(
            ROOT
            / "diagnostics"
            / "coherence"
            / "packed_charge_policy_exclusions_v1_2026-07-30.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "diagnostics"
            / "coherence"
            / "ring_restate_primitive_path_audit_v1_2026-07-30.json"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress one progress record per packed shard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = RingRestateAuditInputs.from_transfer_root(
        args.transfer_root,
        charge_policy_audit=args.charge_policy_audit,
        charge_policy_exclusions=args.charge_policy_exclusions,
    )

    def progress(row: dict[str, object]) -> None:
        if not args.quiet:
            print(
                json.dumps(
                    {
                        "phase": "ring_restate_primitive_path_audit",
                        **row,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    report = audit_unified_ring_restate_teachers(
        inputs,
        progress_callback=progress,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact_sha256": report["artifact_sha256"],
                "output": str(args.output),
                "coverage": report["coverage"],
                "all_partitions": report["all_partitions"],
                "all_hard_checks_pass": report["all_hard_checks_pass"],
                "training_authorized": report["training_authorized"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["all_hard_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
