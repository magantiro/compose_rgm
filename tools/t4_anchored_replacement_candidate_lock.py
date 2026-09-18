"""Assess the anchored JAK2 support pool and publish a score-blind lock.

Zero oracle calls.  Publishing this artifact does not launch docking.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
from pathlib import Path

from rdkit import rdBase

from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_anchored_replacement_lock import (
    build_exact_assessment_and_lock,
    unique_lane_candidates,
    verify_lock,
)
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal

ROOT = Path(__file__).resolve().parents[1]


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _report(payload: dict) -> str:
    assessment = payload["assessment"]
    hits = assessment["historical_hits"]
    lines = [
        "# Anchored-replacement JAK2 candidate lock",
        "",
        "## Outcome",
        "",
        (
            f"The zero-oracle pool contains {assessment['eligible_anchored_basin_endpoints']} "
            "unique eligible amide-plus-diamine endpoints. Exact identity matching found "
            f"{assessment['exact_historical_rediscoveries']} previously charged endpoint; "
            f"the remaining {assessment['prospective_novel_endpoints']} endpoints are locked "
            "for a prospective pilot."
        ),
        "",
        "No oracle calls were made. No structural-distance surrogate was used.",
        "",
        "## Exact historical rediscoveries",
        "",
        "| score | endpoint | arm | record |",
        "| ---: | --- | --- | --- |",
    ]
    for hit in hits:
        lines.append(
            f"| {hit['historical_score']:.2f} | `{hit['endpoint']}` | "
            f"{','.join(hit['arms'])} | `{hit['record_id']}` |"
        )
    if not hits:
        lines.append("| none | none | none | none |")
    lines += [
        "",
        "## Prospective lock",
        "",
        f"Lock ID: `{assessment['lock']['lock_id']}`",
        "",
        f"Charged-call ceiling: {assessment['lock']['charged_calls']}",
        "",
        "Every novel basin endpoint is included. The order is endpoint digest only, and",
        "there is no replacement, backfill, or automatic retry after scores are seen.",
        "",
        "## Claim boundary",
        "",
        "The historical hit is evidence that the proposal family can reach one molecule",
        "already measured at useful JAK2 docking utility. It is not a new result and does",
        "not establish the utility of the novel endpoints. The locked prospective panel is",
        "answer-known development evidence, not held-out evidence or a benchmark result.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--support-result",
        type=Path,
        default=ROOT / "diagnostics/t4_anchored_replacement_support_v1/result.json",
    )
    parser.add_argument(
        "--support-ledger",
        type=Path,
        default=ROOT / "diagnostics/t4_anchored_replacement_support_v1/result.jsonl.gz",
    )
    parser.add_argument(
        "--historical-corpus",
        type=Path,
        default=ROOT / "diagnostics/t4_proposal_prior/dataset_v1/records.jsonl.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "diagnostics/t4_anchored_replacement_pilot_v1/candidate_lock.json",
    )
    args = parser.parse_args()

    support = unseal(args.support_result)
    with gzip.open(args.support_ledger, "rt") as handle:
        attempts = [json.loads(line) for line in handle]
    with gzip.open(args.historical_corpus, "rt") as handle:
        historical = [json.loads(line) for line in handle]
    candidates = unique_lane_candidates(attempts, lane="anchored_replacement")
    expected = support["arms"]["anchored_replacement"]
    if len(candidates) != expected["unique_eligible_endpoints"]:
        raise ValueError("ledger endpoint census disagrees with the sealed support result")

    inputs = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (args.support_result, args.support_ledger, args.historical_corpus)
    }
    assessment = build_exact_assessment_and_lock(
        candidates,
        historical,
        cell="jak2_1",
        delta=float(support["configuration"]["delta"]),
        root_smiles=str(support["configuration"]["root_smiles"]),
        docking_seed=20260918,
        input_sha256=inputs,
    )
    expected_basin = expected["motif_counts_over_unique_eligible"]["basin"]
    if assessment["eligible_anchored_basin_endpoints"] != expected_basin:
        raise ValueError("basin census disagrees with the sealed support result")
    verify_lock(assessment["lock"])
    payload = {
        "schema_version": "t4_anchored_replacement_lock_publication_v1",
        "assessment": assessment,
        "code_revision": _revision(),
        "runtime": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "completed_at_utc": _stamp(),
        "new_oracle_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seal(args.output, payload)
    (args.output.parent / "REPORT.md").write_text(_report(payload))
    print(
        json.dumps(
            {
                "lock_id": assessment["lock"]["lock_id"],
                "basin": assessment["eligible_anchored_basin_endpoints"],
                "rediscovered": assessment["exact_historical_rediscoveries"],
                "best_rediscovered": (
                    assessment["historical_hits"][0]["historical_score"]
                    if assessment["historical_hits"]
                    else None
                ),
                "prospective": assessment["prospective_novel_endpoints"],
                "new_oracle_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
