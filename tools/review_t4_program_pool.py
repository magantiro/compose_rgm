"""Audit the fully scored program pool without new molecular or oracle work."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import platform
import subprocess
from collections import Counter
from pathlib import Path

from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.t4_matched_pilot import seal, unseal

ROOT = Path(__file__).resolve().parents[1]
LOCK = "configs/t4_program_pool_lock.json"


def arm_summary(pool, properties, scores, winner):
    """Repeated proposals reuse one measured endpoint score, not independent labels."""
    rows = pool["rows"]
    completed = [r["receipt"]["endpoint"] for r in rows if r["status"] == "complete"]
    eligible = [s for s in completed if properties[s]["oracle_eligible"]]
    missing = sorted(set(eligible) - scores.keys())
    if missing:
        raise ValueError(f"eligible endpoints have no docking receipt: {missing}")
    measured = [s for s in eligible if scores[s] is not None]
    unique = sorted(set(eligible))
    new = [s for s in unique if s != winner and scores[s] is not None]
    thresholds = (-10.6, -12.0, -13.0, -13.6)
    return {
        "attempts": len(rows),
        "unstarted": pool["attempts_unstarted"],
        "completed_attempts": len(completed),
        "eligible_attempts": len(eligible),
        "known_winner_attempts": eligible.count(winner),
        "unique_endpoints": len(set(completed)),
        "unique_eligible": len(unique),
        "unique_new_scored": len(new),
        "unique_failed_dockings": sum(scores[s] is None for s in unique),
        "best_new_score": min((scores[s] for s in new), default=None),
        "best_including_reused_winner": min((scores[s] for s in measured), default=None),
        "threshold_counts_inclusive": {
            str(t): {
                "attempts_including_reused_winner": sum(scores[s] <= t for s in measured),
                "unique_new_endpoints": sum(scores[s] <= t for s in new),
            }
            for t in thresholds
        },
        "failures": dict(
            sorted(Counter(r["reason_code"] for r in rows if r["status"] != "complete").items())
        ),
        "proposal_seconds": pool["proposal_seconds"],
        "new_scored_endpoints": [{"smiles": s, "ds": scores[s]} for s in new],
    }


def review(root, result_path):
    lock = unseal(root / LOCK)
    result = unseal(result_path)
    if result["lock_sha256"] != sha256_file(root / LOCK):
        raise ValueError("result does not bind the current immutable candidate lock")
    rows = result["rows"]
    expected = [r["smiles"] for r in lock["take"]]
    if [r["smiles"] for r in rows] != expected or len(set(expected)) != len(expected):
        raise ValueError("missing, duplicated, or reordered locked docking rows")
    if [r["index"] for r in rows] != list(range(len(rows))):
        raise ValueError("docking indices differ from the complete candidate lock")
    controls_path = root / lock["winner_controls"]["path"]
    verify_file(controls_path, lock["winner_controls"]["sha256"])
    controls = unseal(controls_path)
    winner = controls[0]["smiles"]
    if len(controls) != 3 or any(r["smiles"] != winner for r in controls):
        raise ValueError("winner controls differ from the declared three evaluations")
    if any(r["ds"] is not None and not math.isfinite(r["ds"]) for r in [*rows, *controls]):
        raise ValueError("nonfinite docking score")
    winner_scores = [r["ds"] for r in controls]
    if any(v is None for v in winner_scores):
        raise ValueError("missing winner-control score")
    scores = {r["smiles"]: r["ds"] for r in rows}
    if winner in scores:
        raise ValueError("winner was charged as a new candidate")
    # Use the matching first seed, not a selected or retrospectively favorable control.
    scores[winner] = winner_scores[0]
    repeats = result["confirmation"]
    if result["best"] is not None and (
        len(repeats) != 2 or any(r["smiles"] != result["best"]["smiles"] for r in repeats)
    ):
        raise ValueError("champion confirmation does not contain both planned repeats")
    calls = len(rows) + len(repeats)
    if calls != result["new_oracle_calls"] or calls > lock["compute"]["new_oracle_call_limit"]:
        raise ValueError("oracle accounting differs from locked allocation")
    inputs = {LOCK: sha256_file(root / LOCK), str(result_path): sha256_file(result_path)}
    inputs[lock["winner_controls"]["path"]] = sha256_file(controls_path)
    for name, digest in lock["inputs_sha256"].items():
        verify_file(root / name, digest)
        inputs[name] = digest
    summaries = []
    for attempt, context in ((2, "original_seed"), (3, "post_linker")):
        folder = root / f"diagnostics/multi_site_proposal_probe/attempt_{attempt}"
        report = json.loads((folder / "result.json").read_text())
        for arm in report["arms"]:
            if arm["context"] != context:
                continue
            path = folder / arm["pool_path"]
            verify_file(path, arm["pool_sha256"])
            pool = json.loads(gzip.decompress(path.read_bytes()))
            properties = {p["smiles"]: p for p in arm["endpoint_properties"]}
            summaries.append(
                {
                    "context": context,
                    "arm": arm["arm"],
                    "pool_sha256": sha256_file(path),
                    "executor_calls": arm["executor_calls"],
                    "actual_site_counts": arm["actual_site_counts"],
                    **arm_summary(pool, properties, scores, winner),
                }
            )
    return {
        "schema_version": "t4_program_pool_review_v1",
        "input_sha256": inputs,
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "analysis_revision_note": "analysis was produced before its evidence commit; the git revision is the base and the exact reducer content is bound separately above",
        "analysis_code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "software": {"python": platform.python_version()},
        "randomness": "none; deterministic reduction of locked proposal streams and saved scores",
        "scientific_source_revision": result["task"]["image_revision"],
        "completed_at_utc": result["completed_at_utc"],
        "new_oracle_calls_in_assay": calls,
        "new_oracle_calls_in_review": 0,
        "reused_control_calls": len(controls),
        "assay_wall_seconds": result["seconds"],
        "docking_worker_seconds": sum(r["docking_seconds"] for r in [*rows, *repeats]),
        "winner": {"smiles": winner, "scores": winner_scores},
        "best": None
        if result["best"] is None
        else {k: result["best"][k] for k in ("smiles", "ds", "ancestry", "sim", "qed", "sa")},
        "repeat_comparison": {
            **result["comparison"],
            "interpretation": "winner-informed seed/post-linker program pool; first evaluation selected the candidate; not benchmark superiority",
            "raw_receipt_label_note": "the shared replicate-summary helper calls this winner-initialized refinement; that label belongs to the other assay, not these starting states",
        },
        "arms": summaries,
        "limitations": [
            "winner-derived parameter-bound programs; not autonomous or held-out discovery",
            "two contexts share one source and target, not independent replicates",
            "each unique endpoint evaluated once; repeated proposals reuse that observation",
            "only the overall new champion has two fresh repeats",
            "different pool sizes and work; descriptive thresholds, not matched benchmark claims",
            "no surrogate selected the fully scored pools; no prospective selector comparison",
            "cost reports wall and worker seconds, not a provider-billed dollar total",
        ],
        "benchmark_claim": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = review(ROOT, args.result)
    seal(args.output, summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "new_oracle_calls_in_assay",
                    "assay_wall_seconds",
                    "best",
                    "repeat_comparison",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
