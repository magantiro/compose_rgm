#!/usr/bin/env python3
"""Run the exact frozen Active8 train atom-restatement orbit audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.experiments.atom_restate_neural_orbit_full_corpus import (
    FullCorpusAuditInputs,
    run_full_corpus_audit,
)


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--driver-contract",
        type=Path,
        default=ROOT / "configs/editing_atom_restate_neural_orbit_full_corpus_v1.json",
    )
    parser.add_argument(
        "--orbit-contract",
        type=Path,
        default=ROOT / "configs/editing_atom_restate_neural_orbit_audit_v1.json",
    )
    parser.add_argument("--active8-inventory", type=Path, required=True)
    parser.add_argument("--unified-manifest", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--mmp-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-problem-address-examples",
        type=int,
        default=256,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run_full_corpus_audit(
        FullCorpusAuditInputs(
            driver_contract_path=args.driver_contract,
            orbit_contract_path=args.orbit_contract,
            active8_inventory_path=args.active8_inventory,
            unified_manifest_path=args.unified_manifest,
            audit_root=args.audit_root,
            mmp_root=args.mmp_root,
            output_dir=args.output_dir,
            repository_root=ROOT,
            max_problem_address_examples=args.max_problem_address_examples,
        )
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(args.output_dir.resolve()),
                "row_evidence_sha256": summary["row_evidence"]["file_sha256"],
                "row_count": summary["row_evidence"]["row_count"],
                "summary_sha256": summary["summary_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
