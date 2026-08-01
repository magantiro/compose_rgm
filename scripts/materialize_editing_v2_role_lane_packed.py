#!/usr/bin/env python3
"""Materialize the frozen Editing-V2 split into 20 pre-Active8 packed shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compose_v4.data.editing_v2_role_lane_packed_materializer import (
    DEFAULT_MAX_SOURCE_ROW_BYTES,
    OUTPUT_NAMESPACE,
    materialize_editing_v2_role_lane_packed,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-materialization-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidate-provenance-bridge-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidate-provenance-registry",
        type=Path,
        required=True,
    )
    parser.add_argument("--split-assignment", type=Path, required=True)
    parser.add_argument(
        "--editing-corpus-contract",
        type=Path,
        required=True,
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument(
        "--output-artifact-prefix",
        default=OUTPUT_NAMESPACE,
    )
    parser.add_argument(
        "--max-source-row-bytes",
        type=int,
        default=DEFAULT_MAX_SOURCE_ROW_BYTES,
    )
    args = parser.parse_args()

    result = materialize_editing_v2_role_lane_packed(
        candidate_materialization_dir=(args.candidate_materialization_dir),
        candidate_provenance_bridge_dir=(args.candidate_provenance_bridge_dir),
        candidate_provenance_registry_path=(args.candidate_provenance_registry),
        split_assignment_path=args.split_assignment,
        editing_corpus_contract_path=args.editing_corpus_contract,
        artifact_root=args.artifact_root,
        code_revision=args.code_revision,
        output_artifact_prefix=args.output_artifact_prefix,
        max_source_row_bytes=args.max_source_row_bytes,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_identity_sha256": result["run_identity_sha256"],
                "run_artifact_root": result["run_artifact_root"],
                "output_shards": result["totals"]["output_shards"],
                "output_records": result["totals"]["output_records"],
                "active8_admission_status": result["active8_admission_status"],
                "training_authorized": result["training_authorized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
