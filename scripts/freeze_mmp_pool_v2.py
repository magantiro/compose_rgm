#!/usr/bin/env python3
"""Validate and immutably publish one completed V2 MMP mining run.

This command performs no mining, compilation, packing or remote work.  It accepts only a completed
``mining_summary_full.json`` produced by the V2 reducer and writes a content-addressed pool plus the
``MMP_POOL_COMPLETE.json`` contract consumed by ``pack_mmp_pool_app.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.data.mmp_pool_freezer import freeze_reduced_mmp_pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reduction-summary",
        type=Path,
        required=True,
        help="mounted filesystem path to the completed V2 mining summary",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="filesystem mount corresponding exactly to the artifact namespace /artifacts",
    )
    parser.add_argument(
        "--output-parent-artifact-path",
        default="/artifacts/mmp_pool_frozen_v2",
        help="V2-only artifact namespace; the content-addressed directory is created below it",
    )
    args = parser.parse_args()
    result = freeze_reduced_mmp_pool(
        args.reduction_summary,
        artifact_root=args.artifact_root,
        output_parent_artifact_path=args.output_parent_artifact_path,
    )
    print(json.dumps({
        "freeze_identity": result["freeze_identity"],
        "namespace": result["namespace"],
        "pool_path": result["pool_path"],
        "pool_sha256": result["pool_sha256"],
        "pool_records": result["pool_records"],
        "contract_path": result["contract_path"],
        "contract_sha256": result["contract_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
