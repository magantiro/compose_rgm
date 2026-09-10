"""Reduce the frozen four-case diagnostic; no generation, fitting, or docking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def summarize(root: Path):
    hashes = {}

    def read(relative, *, sealed=False):
        path = root / relative
        raw = path.read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw)
        if sealed:
            if digest(value["payload"]) != value["payload_sha256"]:
                raise ValueError(f"corrupt sealed input: {path}")
            return value["payload"]
        return value

    launch = read("launch.json")
    cases, snapshots, all_products, result_pools = [], [], [], []
    for case in range(4):
        prefix = f"case_{case}/"
        result = read(prefix + "result.json")
        lock = read(prefix + "proposal_lock.json", sealed=True)
        snapshot = read(prefix + "task_snapshot.json", sealed=True)
        cache = read(prefix + "cache_inventory.json", sealed=True)
        gate = read(prefix + "runtime_gate.json")
        heartbeat = read(prefix + "heartbeat.json")
        assert result["case_index"] == case
        assert result["code_revision"] == launch["image_revision"]["commit"]
        assert gate == result["runtime_gate"]
        assert gate["input_sha256"] == result["configuration"]["expected_input_sha256"]
        assert heartbeat["phase"] == "complete"
        assert result["oracle_calls"] == lock["oracle_calls"] == 0
        assert lock["winner_or_task_value_used"] is False
        assert len(lock["attempts"]) == result["proposal_attempts"] == 4
        assert all(a["status"] in ("complete", "support_dead_end") for a in lock["attempts"])
        assert sum(a["status"] == "complete" for a in lock["attempts"]) == len(result["products"])
        snapshots.append(snapshot)
        products = {p["attempt"]: p for p in result["products"]}
        known = result["known_continuation"]
        source = result["source"]
        unique = {p["smiles"]: p for p in products.values()}
        eligible = {s: p for s, p in unique.items() if p["oracle_eligible"]}
        focal = [i for i, a in enumerate(lock["attempts"]) if a["stratum"] == "focal"]
        assert len(focal) == 2
        cases.append(
            {
                "case_index": case,
                "option": result["option"],
                "applicable_option_count": len(lock["options"]),
                "focal_option_balanced_prior": lock["option_prior"][
                    lock["options"].index(lock["focal_option"])
                ],
                "generic_option_balanced_prior": lock["option_prior"][
                    lock["options"].index("generic")
                ],
                "focal_attempts": len(focal),
                "focal_completed": sum(i in products for i in focal),
                "focal_known_canonical_matches": sum(
                    i in products and products[i]["smiles"] == known["smiles"] for i in focal
                ),
                "attempt_options": [a["option"] for a in lock["attempts"]],
                "completed": len(products),
                "unique_completed": len(unique),
                "eligible_attempts": sum(p["oracle_eligible"] for p in products.values()),
                "unique_eligible": len(eligible),
                "known_eligible": known["oracle_eligible"],
                "known_rank_among_unique_eligible": 1
                + sum(p["desirability"] > known["desirability"] for p in eligible.values()),
                "known_prediction_change_from_source": known["predicted_docking"]
                - source["predicted_docking"],
                "known_topology_change": {
                    key: known["topology"][key] - value for key, value in source["topology"].items()
                },
                "known_path_support": result["known_path_support"],
                "elapsed_seconds_including_initialization_and_support_audit": result[
                    "elapsed_seconds"
                ],
                "initialization_seconds": gate["initialization_seconds"],
                "post_initialization_seconds_including_support_audit": result["elapsed_seconds"]
                - gate["initialization_seconds"],
                "fresh_law_enumerations": result["law_work"]["fresh_laws"],
                "fresh_law_seconds": result["law_work"]["law_seconds"],
                "law_cache_hits": result["law_work"]["cached_laws"],
                "old_run_law_receipts_reused": len(cache["used_law_sha256"]),
                "public_executor_calls_including_support_audit": result["public_executor_calls"],
                "peak_rss_kib_linux": result["peak_rss_native_units"],
            }
        )
        all_products.extend(products.values())
        result_pools.append([source, known, *products.values()])
    # Independent float64 solves differed at the last bit on one worker. Keep
    # that deviation explicit; do not call the snapshots byte-identical.
    # The fixed kernel lies in [0,1], hence |delta prediction| <= ||delta a||_1.
    # An observed pairwise gap > 2 * this bound cannot reverse under head reuse.
    reference = snapshots[0]
    equivalence = []
    for case, snapshot in enumerate(snapshots):
        structural_fields = set(reference) - {"coefficients", "snapshot_sha256"}
        assert all(snapshot[k] == reference[k] for k in structural_fields)
        differences = [
            abs(a - b)
            for a, b in zip(reference["coefficients"], snapshot["coefficients"], strict=True)
        ]
        bound = sum(differences)
        pool = list({p["smiles"]: p for p in result_pools[case]}.values())
        gaps = [
            abs(a["predicted_docking"] - b["predicted_docking"])
            for i, a in enumerate(pool)
            for b in pool[i + 1 :]
        ]
        equivalence.append(
            {
                "case_index": case,
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "same_recipe_training_rows_and_features": True,
                "max_coefficient_difference_from_case_0": max(differences),
                "prediction_error_upper_bound_from_case_0": bound,
                "minimum_distinct_molecule_prediction_gap": min(gaps),
                "pairwise_prediction_order_stable_under_bound": min(gaps) > 2 * bound,
            }
        )
        assert equivalence[-1]["pairwise_prediction_order_stable_under_bound"]
    return {
        "schema_version": "option_decision_summary_v1",
        "run_id": launch["run_id"],
        "generation_code_revision": launch["image_revision"]["commit"],
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashes,
        "byte_identical_value_snapshots": len({s["snapshot_sha256"] for s in snapshots}) == 1,
        "value_snapshot_equivalence": equivalence,
        "cases": cases,
        "total_attempts": 16,
        "total_completed": len(all_products),
        "total_unique_completed": len({p["smiles"] for p in all_products}),
        "total_eligible_attempts": sum(p["oracle_eligible"] for p in all_products),
        "total_focal_known_canonical_matches": sum(
            c["focal_known_canonical_matches"] for c in cases
        ),
        "new_docking_calls": 0,
        "limitations": [
            "four answer-known starting states and preselected focal option types",
            "attachment sites and known continuations not supplied to proposal generation",
            "conditional decision diagnosis, not seed-to-winner autonomous recovery",
            "two focal trials per case, not a precise recovery-rate estimate",
            "surrogate ordering is not observed docking ordering",
            "reported timing includes post-lock support diagnosis; pure proposal time not isolated",
            "the local/global region-selection policy is not evaluated by this fully mutable assay",
        ],
    }


if __name__ == "__main__":
    result = summarize(Path(__file__).parent / "attempt_1")
    destination = Path(__file__).parent / "summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(json.dumps({k: v for k, v in result.items() if k.startswith("total_")}, indent=2))
