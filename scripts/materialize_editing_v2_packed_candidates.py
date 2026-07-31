#!/usr/bin/env python3
"""Materialize deterministic Editing-V2 candidate headers from packed caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.data.editing_v2_packed_candidate_materializer import (
    materialize_packed_candidate_headers,
    validate_packed_candidate_materialization,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--routing-policy", type=Path, required=True)
    parser.add_argument("--editing-corpus-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--max-entry-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()

    built = materialize_packed_candidate_headers(
        source_manifest_path=args.source_manifest,
        artifact_root=args.artifact_root,
        routing_policy_path=args.routing_policy,
        editing_corpus_contract_path=args.editing_corpus_contract,
        output_dir=args.output_dir,
        code_revision=args.code_revision,
        max_entry_bytes=args.max_entry_bytes,
    )
    validate_packed_candidate_materialization(
        args.output_dir,
        expected_manifest_sha256=built["manifest_sha256"],
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "manifest_sha256": built["manifest_sha256"],
                "rows": built["rows"]["totals"]["rows"],
                "routed_rows": built["rows"]["totals"]["routed_rows"],
                "rejected_rows": built["rows"]["totals"]["rejected_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
