"""Verify every lock/worker receipt in the completed saved-prefix T4 diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from rdkit import Chem

from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_warm_continuation import payload_hash
from tools.t4_warm_audit import candidate_summary, counts


def report(root: Path, source: Path) -> dict:
    inventory = json.loads((root / "remote_inventory.json").read_text())
    for relative, row in inventory["files"].items():
        verify_file(root / relative, row["sha256"])
    result = json.loads((root / "result.json").read_text())
    docking = unseal(root / "docking.json")
    # The common remote wrapper replaces its own completion timestamp after
    # publication/profiling. Scientific fields must match exactly; these two
    # operational timestamps must be ordered, not equal.
    if any(result[key] != value for key, value in docking.items() if key != "completed_at_utc"):
        raise ValueError("result differs from sealed docking reduction")
    if docking["completed_at_utc"] > result["completed_at_utc"]:
        raise ValueError("wrapper completion precedes docking reduction")
    contract = result["configuration"]
    for relative, digest in contract["source"]["files"].items():
        verify_file(source / relative, digest)
    warm = unseal(source / "warm_start.json")
    lock = unseal(root / "candidate_lock.json")
    digest = sha256_file(root / "candidate_lock.json")
    started = json.loads((root / "docking_started.json").read_text())
    if result["candidate_lock_sha256"] != digest or started["candidate_lock_sha256"] != digest:
        raise ValueError("batch/lock hash mismatch")
    if lock["source_warm_start_sha256"] != payload_hash(warm):
        raise ValueError("wrong warm archive")
    if (
        lock["oracle_calls"] != 0
        or not lock["locked_at_utc"] <= started["started_at_utc"]
        or lock["software"]["rdkit"] != contract["required_rdkit"]
    ):
        raise ValueError("pre-oracle lock or pinned-runtime invariant violated")
    previous = {Chem.MolToSmiles(Chem.MolFromSmiles(c["smiles"])) for c in warm["archive"]}
    parents = {c["smiles"]: c for c in warm["archive"]}
    seed_counts = counts(warm["archive"][0]["state"])
    n = len(lock["take"])
    if not n == len(result["docked"]) == started["attempts"] <= 13:
        raise ValueError("partial batch accounting mismatch")
    if (
        result["cumulative_guided_calls"] != 20 + n
        or result["remaining_new_call_allowance"] != 40 - n
    ):
        raise ValueError("cumulative allowance mismatch")
    actual_row_paths = sorted(root.glob("rows/*/result.json"))
    if len(actual_row_paths) != n or len(list(root.glob("rows/*/started.json"))) != n:
        raise ValueError("missing or additional worker attempts")
    seen, records, events = set(), [], []
    for index, (candidate, docked) in enumerate(zip(lock["take"], result["docked"], strict=True)):
        row = unseal(root / "rows" / f"{index:02d}" / "result.json")
        attempt = json.loads((root / "rows" / f"{index:02d}" / "started.json").read_text())
        if (
            any(docked[k] != v for k, v in candidate.items())
            or any(docked[k] != v for k, v in row.items())
            or any(row[k] != v for k, v in attempt.items())
            or row["index"] != index
            or row["candidate_lock_sha256"] != digest
        ):
            raise ValueError("worker/candidate identity mismatch")
        if (
            not started["started_at_utc"]
            <= row["started_at_utc"]
            <= row["completed_at_utc"]
            <= result["completed_at_utc"]
        ):
            raise ValueError("worker chronology violated")
        smiles = candidate["smiles"]
        if smiles in seen or smiles in previous or smiles not in contract["authorized_smiles"]:
            raise ValueError("duplicate, previously evaluated, or unauthorized candidate")
        seen.add(smiles)
        limits = lock["constraints"]
        if not (
            candidate["oracle_eligible"]
            and candidate["v"] == 0
            and candidate["qed"] >= limits["qed_min"]
            and candidate["sa"] <= limits["sa_max"]
            and candidate["sim"] >= limits["delta"]
        ):
            raise ValueError("ineligible oracle candidate")
        summary = candidate_summary(docked, lock, parents, seed_counts)
        parent_score = parents[candidate["parent"]]["ds"]
        records.append(
            {
                **summary,
                "parent_docking_score": parent_score,
                "score_delta_from_parent": None
                if docked["ds"] is None
                else docked["ds"] - parent_score,
            }
        )
        events.extend([(row["started_at_utc"], 1), (row["completed_at_utc"], -1)])
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    if peak > contract["compute"]["docking_containers"]:
        raise ValueError("worker overlap exceeded the four-container cap")
    return {
        "schema_version": "t4_partial_review_v1",
        "verification": "passed",
        "source_revision": result["code_revision"],
        "audit_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "input_inventory_sha256": sha256_file(root / "remote_inventory.json"),
        "source_files": contract["source"]["files"],
        "selected_options": dict(sorted(Counter(c["option"] for c in records).items())),
        "distinct_canonical": len(seen),
        "distinct_bundles": len({c["bundle_id"] for c in records}),
        "maximum_observed_worker_overlap": peak,
        "best_source_score": result["best_source"]["ds"],
        "best_batch_score": result["best_batch"]["ds"] if result["best_batch"] else None,
        "best_so_far_score": result["best_so_far"]["ds"],
        "new_oracle_attempts": n,
        "cumulative_guided_calls": result["cumulative_guided_calls"],
        "oracle_failures": result["oracle_failures"],
        "docking_seconds": result["docking_seconds"],
        "total_seconds": result["elapsed_seconds"],
        "candidates": records,
        "interpretation_scope": result["interpretation_scope"],
        "limitations": [
            "No matched continuation; partial round and inspected development cell.",
            "Unseeded oracle; parent score differences are not causal affinity estimates.",
            "Endpoint feasibility and ring construction do not prove synthesizability.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.root, args.source)
    print(publish_json(args.output, result))
    print(
        json.dumps({k: v for k, v in result.items() if k != "candidates"}, sort_keys=True, indent=2)
    )
