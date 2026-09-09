"""Deterministic reduction of saved winner-path receipts, with no molecular search."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

from compose_v4.control.region_rewrite import touched_slots
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.tracelets import RingSystemRestate

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def review(directory):
    script = Path(__file__).resolve()
    relative = script.relative_to(ROOT).as_posix()
    if (
        subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        != script.read_bytes()
    ):
        raise ValueError("commit the exact review implementation before producing evidence")
    audit_path = directory / "audit.json"
    audit = json.loads(audit_path.read_text())
    by_target, by_cell, operations = {}, {}, Counter()
    paths, unresolved, timings, footprints = [], [], [], []
    hashes = {"audit.json": sha(audit_path)}
    inspected_source = {}
    for relative_source in (
        "src/compose_v4/control/region_rewrite.py",
        "src/compose_v4/rewrite/action_codec_v4.py",
        "src/compose_v4/rewrite/tracelets.py",
    ):
        actual = sha(ROOT / relative_source)
        if actual != audit["implementation_sha256"][relative_source]:
            raise ValueError(f"{relative_source}: source differs from audited implementation")
        inspected_source[relative_source] = actual
    for pair in audit["pairs"]:
        receipt_path = directory / pair["receipt"]
        if sha(receipt_path) != pair["receipt_sha256"]:
            raise ValueError(f"{receipt_path}: physical receipt hash mismatch")
        hashes[pair["receipt"]] = pair["receipt_sha256"]
        receipt = json.loads(gzip.decompress(receipt_path.read_bytes()))
        payload = receipt["payload"]
        if hashlib.sha256(canonical(payload)).hexdigest() != receipt["payload_sha256"]:
            raise ValueError(f"{receipt_path}: payload hash mismatch")
        if receipt["scientific_identity"] != audit["scientific_identity"]:
            raise ValueError(f"{receipt_path}: scientific identity mismatch")
        path, steps = payload["path"], payload["annotations"]
        if path["status"] != pair["status"]:
            raise ValueError(f"{receipt_path}: status disagrees with aggregate")
        for target in {r["target"] for r in pair["references"]}:
            by_target.setdefault(target, Counter())[pair["status"]] += 1
        for cell in {r["cell"] for r in pair["references"]}:
            by_cell.setdefault(cell, Counter())[pair["status"]] += 1
        attempts = path.get("attempts", [])
        meta = {
            "pair_id": pair["pair_id"],
            "cells": sorted({r["cell"] for r in pair["references"]}),
            "primitive_lower_bound": path.get("primitive_lower_bound"),
            "mapping_timeouts": sum(a["mapping"]["timed_out"] for a in attempts),
            "truncated_mapping_enumerations": sum(
                a["mapping"].get("match_enumeration_truncated", False) for a in attempts
            ),
        }
        if pair["status"] != "witness_found":
            unresolved.append(
                {
                    **meta,
                    "status": pair["status"],
                    "attempt_statuses": [a["status"] for a in attempts],
                    "best_residuals": [a.get("best_residual") for a in attempts],
                }
            )
            continue
        if len(steps) != len(path["actions"]) + 1 or len(path["states"]) != len(steps):
            raise ValueError(f"{receipt_path}: action/state/annotation length mismatch")
        for state, step in zip(path["states"], steps, strict=True):
            if hashlib.sha256(canonical(state)).hexdigest() != step["state_sha256"]:
                raise ValueError(f"{receipt_path}: annotation state mismatch")
        counts = Counter(a["executor_rule"] for a in path["actions"])
        for index, mark in enumerate(path["actions"]):
            family, action = decode_action(mark)
            if isinstance(action, RingSystemRestate):
                expected = sorted({v for change in action.changes for v in (change.a, change.b)})
                observed = sorted(touched_slots(action))
                footprints.append(
                    {
                        "pair_id": pair["pair_id"],
                        "step": index,
                        "executor_rule": family,
                        "bond_change_endpoints": expected,
                        "reported_touch_slots": observed,
                        "equal": expected == observed,
                        "regions_passing_touch_filter": steps[index]["next_edit"][
                            "touch_filter_regions"
                        ],
                    }
                )
        operations.update(counts)
        trajectories = {}
        for delta in steps[0]["failed_constraints_by_delta"]:
            failed = [r["failed_constraints_by_delta"][delta] for r in steps]
            trajectories[delta] = {
                "feasible_steps": [i for i, failures in enumerate(failed) if not failures],
                "infeasible_intermediate_steps": [
                    i for i, failures in enumerate(failed) if 0 < i < len(steps) - 1 and failures
                ],
                "endpoint_failures": failed[-1],
                "constraints_encountered": sorted(
                    {failure for failures in failed for failure in failures}
                ),
            }
        paths.append(
            {
                **meta,
                "witness_steps": path["witness_steps"],
                "operations": dict(sorted(counts.items())),
                "zero_touch_filter_steps": [
                    r["step"] for r in steps[:-1] if not r["next_edit"]["touch_filter_regions"]
                ],
                "constraints": trajectories,
                "source": steps[0],
                "endpoint": steps[-1],
                "minimum_similarity": min(r["sim"] for r in steps),
                "minimum_qed": min(r["qed"] for r in steps),
                "maximum_sa": max(r["sa"] for r in steps),
                "canonical_path": [r["smiles"] for r in steps],
                "rule_sequence": [a["executor_rule"] for a in path["actions"]],
            }
        )
        equiv = receipt.get("equivalence")
        while equiv is not None:
            original = directory / equiv["prior_receipt"]
            if sha(original) != equiv["prior_sha256"]:
                raise ValueError(f"{original}: conversion source hash mismatch")
            hashes[equiv["prior_receipt"]] = equiv["prior_sha256"]
            if "fast_annotation_seconds" in equiv:
                timings.append(
                    {
                        "pair_id": pair["pair_id"],
                        "original_annotation_seconds": payload["total_seconds"]
                        - payload["search_seconds"],
                        "equivalent_fast_annotation_seconds": equiv["fast_annotation_seconds"],
                    }
                )
            equiv = equiv.get("previous_equivalence")
    result = {
        "schema_version": "ivg_winner_path_review_v1",
        "input_directory": str(directory.resolve()),
        "input_sha256": dict(sorted(hashes.items())),
        "producer_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "producer_sha256": sha(script),
        "inspected_source_sha256": inspected_source,
        "python": platform.python_version(),
        "configuration": {
            "seed": None,
            "seed_reason": "deterministic reduction, no new molecular computation",
        },
        "evidence": audit["evidence"],
        "software": audit["software"],
        "snapshot": audit["snapshot"],
        "n_pairs": audit["n_pairs"],
        "n_cell_run_winners": audit["n_cell_run_winners"],
        "n_replayed_witnesses": len(paths),
        "n_within_16_witnesses": sum(p["witness_steps"] <= 16 for p in paths),
        "n_certified_lower_bound_above_16": sum(
            p.get("primitive_lower_bound", 0) > 16 for p in audit["pairs"]
        ),
        "by_target": {k: dict(sorted(v.items())) for k, v in sorted(by_target.items())},
        "by_cell": {k: dict(sorted(v.items())) for k, v in sorted(by_cell.items())},
        "witness_step_range": [
            min(p["witness_steps"] for p in paths),
            max(p["witness_steps"] for p in paths),
        ]
        if paths
        else None,
        "paths_with_feasibility_valleys": sum(
            any(
                c["infeasible_intermediate_steps"] and not c["endpoint_failures"]
                for c in p["constraints"].values()
            )
            for p in paths
        ),
        "paths_with_zero_touch_filter_steps": sum(
            bool(p["zero_touch_filter_steps"]) for p in paths
        ),
        "operations": dict(sorted(operations.items())),
        "composite_action_footprints": footprints,
        "saved_pair_search_seconds": sum(p["search_seconds"] for p in audit["pairs"]),
        "saved_pair_total_seconds_including_original_slow_annotations": sum(
            p["total_seconds"] for p in audit["pairs"]
        ),
        "paths": paths,
        "unresolved": unresolved,
        "annotation_speedup_equivalence": timings,
        "execution": audit["execution"],
        "new_oracle_calls": 0,
        "new_search_executor_calls": 0,
        "limitations": audit["limitations"],
    }
    output = directory / "review.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_bytes(canonical(result) + b"\n")
    temporary.replace(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = review(parser.parse_args().directory)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("n_pairs", "n_replayed_witnesses", "n_within_16_witnesses", "by_target")
            },
            sort_keys=True,
        )
    )
