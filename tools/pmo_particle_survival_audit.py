"""Audit saved particle ancestry and option outcomes without new scoring."""

import argparse
import json
from collections import Counter
from pathlib import Path

from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.inference_package import software


def summarize_proposals(proposals):
    return {
        "completed": len(proposals),
        "improving_parent": sum(p["score"] > p["parent_score"] for p in proposals),
        "equal_parent": sum(p["score"] == p["parent_score"] for p in proposals),
        "worse_than_parent": sum(p["score"] < p["parent_score"] for p in proposals),
        "inserted_atoms": sum(p["structural_change"]["n_inserted"] for p in proposals),
        "deleted_atoms": sum(p["structural_change"]["n_deleted"] for p in proposals),
        "ring_constructions": sum(
            p["bundle"]["option"].startswith("construct:") for p in proposals
        ),
        "options": dict(sorted(Counter(p["bundle"]["option"] for p in proposals).items())),
    }


def analyze(path, receipt_path):
    receipt = json.loads(receipt_path.read_text())
    verify_file(path, receipt["input_paths_sha256"][str(path)])
    r = json.loads(path.read_text())
    if r["run_id"] != receipt["run_id"] or not receipt["particle_decisions_match_exactly"]:
        raise ValueError("survival audit requires the verified particle result")
    n = r["configuration"]["particles"]
    roots = r["initial_parents"]
    if len(roots) != n or len(r["rounds"]) != r["configuration"]["boundaries"]:
        raise ValueError("particle or boundary census mismatch")
    reference = [row["arms"]["reference"] for row in r["rounds"]]
    if any(a["resampled"] or a["indices"] != list(range(n)) for a in reference):
        raise ValueError("reference suffix comparison requires independent reference trajectories")
    rows = []
    for arm in r["configuration"]["arms"]:
        current, origin = roots, list(range(n))
        boundaries, proposals_all, discarded = [], [], []
        for t, round_row in enumerate(r["rounds"]):
            a = round_row["arms"][arm]
            proposed = a["proposals"]
            for p, parent in zip(proposed, current, strict=True):
                if p is not None and (
                    parent is None
                    or p["parent_smiles"] != parent["smiles"]
                    or p["parent_score"] != parent["score"]
                    or p["chain"] != parent["chain"] + [p["id"]]
                ):
                    raise ValueError(f"saved parent/score/ancestry mismatch: {arm}/{t + 1}")
            chosen = [proposed[i] for i in a["indices"]]
            next_origin = [origin[i] for i in a["indices"]]
            census = Counter(
                roots[i]["id"] for i, p in zip(next_origin, chosen, strict=True) if p is not None
            )
            for slot, p in enumerate(proposed):
                if p is None or slot in a["indices"]:
                    continue
                # A suffix is available only for the exact shared reference
                # proposal. This is one realized trajectory, not a bound on
                # unobserved continuations or a causal counterfactual estimate.
                matches = [
                    i
                    for i, q in enumerate(reference[t]["proposals"])
                    if q is not None and q["id"] == p["id"] and q["chain"] == p["chain"]
                ]
                suffix = [
                    q
                    for i in matches
                    for stage in reference[t:]
                    if (q := stage["proposals"][i]) is not None
                ]
                discarded.append(
                    {
                        "boundary": t + 1,
                        "smiles": p["smiles"],
                        "option": p["bundle"]["option"],
                        "parent_score": p["parent_score"],
                        "score": p["score"],
                        "origin_id": roots[origin[slot]]["id"],
                        "saved_reference_suffix_max": max(
                            (q["score"] for q in suffix), default=None
                        ),
                        "saved_suffix_count": len(suffix),
                    }
                )
            completed = [p for p in proposed if p is not None]
            proposals_all.extend(completed)
            boundaries.append(
                {
                    "boundary": t + 1,
                    "resampled": a["resampled"],
                    "ess": a["ess"],
                    "retained_start_census": dict(sorted(census.items())),
                    "distinct_retained_starts": len(census),
                    "live_particles": sum(census.values()),
                    "best": a["best"],
                    **summarize_proposals(completed),
                }
            )
            current, origin = chosen, next_origin
        rows.append(
            {
                "arm": arm,
                "boundaries": boundaries,
                "all_proposals": summarize_proposals(proposals_all),
                "after_first_boundary": summarize_proposals(
                    [
                        p
                        for row in r["rounds"][1:]
                        for p in row["arms"][arm]["proposals"]
                        if p is not None
                    ]
                ),
                "discarded_continuations": discarded,
            }
        )
    return {
        "schema_version": "particle_survival_audit_v1",
        "run_id": r["run_id"],
        "input_sha256": {str(p): sha256_file(p) for p in (path, receipt_path)},
        "analyzer_sha256": sha256_file(Path(__file__)),
        "software": software(),
        "executed_revision": r["image_revision"],
        "configuration": r["configuration"],
        "initial_starts": [
            {"id": p["id"], "smiles": p["smiles"], "score": p["score"]} for p in roots
        ],
        "initial_best": r["initial_metrics"]["best"],
        "arms": rows,
        "new_oracle_calls": 0,
        "new_model_calls": 0,
        "claim_boundary": "retrospective warm-development ancestry and observed outcome audit; saved suffixes do not bound unsampled futures; no new optimizer comparison",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--verified-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.result, args.verified_report)
    publish_json(args.output, result)
    for row in result["arms"]:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
