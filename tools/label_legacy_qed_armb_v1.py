"""Label the pre-fix arm-B records; do NOT let them stand in as the corrected result.

~/qed_griddd_armb_v1/run holds per-source records produced under the MISCONFIGURED
interface, in which the similarity indicator lived in the REWARD
(score = QED * 1[sim >= 0.4]) rather than in the support.  Under that wiring an
endpoint below the similarity floor was still admitted as a candidate and still
consumed a charged query, scoring 0.0.

This tool re-scores those records under the corrected metric so the defect is
quantified rather than described, and stamps them as superseded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QED_TARGET = 0.90
SIM_FLOOR = 0.40


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = []
    for path in sorted(Path(args.records).glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except Exception:  # noqa: BLE001, S112 - truncated record skipped
            continue
        if record.get("status") == "complete":
            records.append(record)

    solved = charged = zero_scored = inadmissible_charged = proposed = 0
    indices = []
    config = {}
    for record in records:
        indices.append(record["index"])
        if not config:
            config = {
                key: record.get(key)
                for key in ("budget", "rounds", "queries_per_round")
            }
            config["proposal_pool_per_round"] = record.get("queries_per_round")
            config["property_evaluated_in_proposal_path"] = False
        scored = record.get("scored", [])
        charged += len(scored)
        zero_scored += sum(1 for row in scored if row["score"] <= 0.0)
        inadmissible_charged += sum(1 for row in scored if row["sim"] < SIM_FLOOR)
        proposed += int(record.get("proposed_evaluations") or 0)
        distinct, seen = [], set()
        for row in sorted(scored, key=lambda r: -r["score"]):
            if row["smiles"] in seen:
                continue
            seen.add(row["smiles"])
            distinct.append(row)
        if any(
            row["qed"] >= QED_TARGET and row["sim"] >= SIM_FLOOR for row in distinct[: args.k]
        ):
            solved += 1

    report = {
        "schema_version": "qed_legacy_armb_label_v1",
        "LABEL": "SUPERSEDED -- produced under the MISCONFIGURED interface",
        "do_not_reuse_as": "the corrected dedicated-task result",
        "what_was_wrong": [
            (
                "the similarity floor lived in the REWARD (score = QED * indicator), so "
                "endpoints below the floor were admitted as candidates and consumed "
                "charged queries while scoring 0.0"
            ),
            (
                "the task was ProgramTask(kind='pmo'), a dispatch label, not a dedicated "
                "QED-editing task with contract-governed thresholds"
            ),
        ],
        "search_configuration": config,
        "records": len(records),
        "source_index_range": [min(indices), max(indices)] if indices else None,
        "charged_scored_endpoints": charged,
        "zero_scored_charged_endpoints": zero_scored,
        "zero_scored_fraction_of_budget": round(zero_scored / charged, 4) if charged else None,
        "charged_endpoints_below_similarity_floor": inadmissible_charged,
        "wasted_fraction_of_budget": round(inadmissible_charged / charged, 4) if charged else None,
        "proposed_evaluations_total": proposed,
        "solved_at_k_under_corrected_metric": solved,
        "rate": round(solved / len(records), 4) if records else None,
        "note": "the rate is reported only so the corrected run can be compared against its "
                "own predecessor; it is NOT a controller result and must not be quoted as one",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
