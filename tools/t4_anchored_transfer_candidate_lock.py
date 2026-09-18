"""Publish the zero-oracle JAK2 seed-0 anchored-transfer utility lock."""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
from pathlib import Path

from rdkit import rdBase

from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_anchored_replacement_lock import unique_lane_candidates
from compose_v4.experiments.t4_anchored_transfer_lock import (
    build_transfer_lock,
    verify_transfer_lock,
)
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transfer-result",
        type=Path,
        default=ROOT / "diagnostics/t4_anchored_replacement_transfer_v1/result.json",
    )
    parser.add_argument(
        "--transfer-ledger",
        type=Path,
        default=ROOT / "diagnostics/t4_anchored_replacement_transfer_v1/result.jsonl.gz",
    )
    parser.add_argument(
        "--historical-corpus",
        type=Path,
        default=ROOT / "diagnostics/t4_proposal_prior/dataset_v1/records.jsonl.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "diagnostics/t4_anchored_replacement_transfer_pilot_v1/candidate_lock.json",
    )
    args = parser.parse_args()

    transfer = unseal(args.transfer_result)
    with gzip.open(args.transfer_ledger, "rt") as handle:
        attempts = [json.loads(line) for line in handle]
    with gzip.open(args.historical_corpus, "rt") as handle:
        historical = [json.loads(line) for line in handle]
    candidates = unique_lane_candidates(
        (row for row in attempts if row.get("cell") == "jak2_0"),
        lane="anchored_replacement",
    )
    expected = transfer["summaries"]["jak2_0"]["anchored_replacement"]
    if len(candidates) != expected["unique_eligible_endpoints"]:
        raise ValueError("seed-0 anchored census disagrees with the transfer result")
    inputs = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (args.transfer_result, args.transfer_ledger, args.historical_corpus)
    }
    config = transfer["configuration"]
    assessment = build_transfer_lock(
        candidates,
        historical,
        cell="jak2_0",
        delta=float(config["delta"]),
        root_smiles=str(config["cells"]["jak2_0"]["root_smiles"]),
        docking_seed=20260919,
        selection_limit=12,
        input_sha256=inputs,
    )
    verify_transfer_lock(assessment["lock"])
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    payload = {
        "schema_version": "t4_anchored_transfer_lock_publication_v1",
        "assessment": assessment,
        "code_revision": revision,
        "runtime": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "completed_at_utc": _stamp(),
        "new_oracle_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seal(args.output, payload)
    print(
        json.dumps(
            {
                "lock_id": assessment["lock"]["lock_id"],
                "pool": assessment["candidate_endpoints"],
                "historical_hits": assessment["exact_historical_rediscoveries"],
                "best_historical": assessment["best_historical_rediscovery"],
                "novel": assessment["prospective_novel_endpoints"],
                "locked": assessment["lock"]["charged_calls"],
                "new_oracle_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
