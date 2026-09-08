"""Hash-bound offline report of full or interrupted guided T4 continuation."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.control.graph_geometry import topology
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_warm_continuation import (
    advance_archive,
    endpoint,
    payload_hash,
    verify_round,
)
from compose_v4.rewrite.trace_shard import decode_state


def counts(encoded: dict) -> dict:
    state = decode_state(encoded)
    result = topology(state)
    if result["n_heavy"] != state.n_real_atoms:
        raise ValueError("topology helper encountered a non-element occupied slot")
    return {k: result[k] for k in ("n_heavy", "cycle_rank", "n_ring_systems")}


def candidate_summary(candidate: dict, lock: dict, parents: dict, seed_counts: dict) -> dict:
    # Lazy import keeps the existing ring-quality auditor's use of this helper
    # acyclic during module initialization. These are structural diagnostics,
    # not new oracle eligibility rules.
    from tools.t4_ring_quality_audit import ring_descriptors

    actual = counts(endpoint(lock, candidate))
    parent = parents[candidate["parent"]]
    before = counts(parent["state"])
    for field, metric in (
        ("d_heavy", "n_heavy"),
        ("d_cycle_rank", "cycle_rank"),
        ("d_ring_systems", "n_ring_systems"),
    ):
        if actual[metric] - before[metric] != candidate[field]:
            raise ValueError(f"candidate {field} differs from its exact-state counts")
    return {
        **{k: v for k, v in candidate.items() if k != "state"},
        "topology": actual,
        "cumulative_delta_from_seed": {k: actual[k] - seed_counts[k] for k in actual},
        "ancestors": [*parent["ancestors"], parent["smiles"]],
        "parent_round": parent["round"],
        "parent_docking_score": parent["ds"],
        "ring_descriptors": ring_descriptors(candidate["smiles"]),
        "parent_ring_descriptors": ring_descriptors(parent["smiles"]),
        "new_cycle_this_round": candidate["d_cycle_rank"] > 0,
        "system_split_without_cycle_gain": (
            candidate["d_ring_systems"] > 0 and candidate["d_cycle_rank"] <= 0
        ),
    }


def summarize(records: list[dict]) -> dict:
    return {
        "count": len(records),
        "unique_canonical": len({c["smiles"] for c in records}),
        "distinct_bundles": len({c["bundle_id"] for c in records}),
        "options": dict(sorted(Counter(c["option"] for c in records).items())),
        "new_cycle_relative_to_parent": sum(c["d_cycle_rank"] > 0 for c in records),
        "two_or_more_extra_cycles_from_seed": sum(
            c["cumulative_delta_from_seed"]["cycle_rank"] >= 2 for c in records
        ),
        "system_splits_without_cycle_gain": sum(
            c["system_split_without_cycle_gain"] for c in records
        ),
        "intended_release_min_median_max": [
            float(f([c["r_release"] for c in records])) for f in (np.min, np.median, np.max)
        ]
        if records
        else None,
        "realized_coherent_min_median_max": [
            float(f([c["r_coherent"] for c in records])) for f in (np.min, np.median, np.max)
        ]
        if records
        else None,
        "feasible": sum(c["v"] <= 0 for c in records),
    }


def report(root: Path) -> dict:
    warm = unseal(root / "warm_start.json")
    seed_counts = counts(warm["archive"][0]["state"])
    curve, rounds = [], []

    def score_point(archive):
        feasible = [c for c in archive["archive"] if c["ds"] is not None and c["v"] <= 0]
        best = min(feasible, key=lambda c: c["ds"]) if feasible else None
        return {
            "round": archive["round"],
            "cumulative_calls": archive["oracle_attempts"],
            "best_score": best["ds"] if best else None,
            "best_smiles": best["smiles"] if best else None,
        }

    curve.append(score_point(warm))
    # A feedback episode can start after the audited partial event 2. Do not
    # look for a fictitious completed round 2 in the new run's namespace.
    for rd in range(warm["round"] + 1, warm["round"] + 3):
        directory = root / f"round_{rd}"
        lock_path = directory / "candidate_lock.json"
        if not lock_path.exists():
            units = [unseal(p) for p in sorted((directory / "parents").glob("*.json"))]
            if units:
                rounds.append(
                    {
                        "round": rd,
                        "status": "partial_preparation_not_an_oracle_pool",
                        "completed_parent_units": len(units),
                        "checkpointed_bundles": [b for u in units for b in u["bundles"]],
                        "checkpointed_executor_calls": max(
                            u["cumulative_executor_calls"] for u in units
                        ),
                    }
                )
            break
        lock = unseal(lock_path)
        if lock["task"]["warm_start_sha256"] != payload_hash(warm):
            raise ValueError("round did not consume the expected complete archive")
        verify_round(lock, warm, lock["task"])
        parents = {c["smiles"]: c for c in warm["archive"]}
        pool = [candidate_summary(c, lock, parents, seed_counts) for c in lock["pool"]]
        selected = [candidate_summary(c, lock, parents, seed_counts) for c in lock["take"]]
        trace = [t for w in lock["work"] for t in w["sampled_transitions"]]
        if any(
            not 0 < t["sample_probability"] <= 1 or not -1e-12 <= t["kl"] <= 1 + 1e-8 for t in trace
        ):
            raise ValueError("sample probability or KL outside the frozen bounds")
        row = {
            "round": rd,
            "status": "locked",
            "bundles": lock["bundles"],
            "selected_options": dict(sorted(Counter(b["option"] for b in lock["bundles"]).items())),
            "pool": summarize(pool),
            "selected": summarize(selected),
            "pool_candidates": pool,
            "selected_candidates": selected,
            "pool_diversity": lock["candidate_diversity"],
            "selected_diversity": lock["selected_diversity"],
            "proposal_seconds": lock["proposal_seconds"],
            "prepare_wall_seconds": lock["prepare_wall_seconds"],
            "executor_calls": lock["total_public_executor_calls"],
            "law_enumerations": sum(w["law_enumerations"] for w in lock["work"]),
            "sampled_transitions": len(trace),
            "parent_budgets": [w.get("parent_budget") for w in lock["work"]],
            "unselected_region_draws": sum(
                w.get("unselected_region_draws", 0) for w in lock["work"]
            ),
        }
        ledger_path = directory / "executor_attempts_0.json"
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            attempts = ledger["attempts"]
            if ledger["prior_calls"] != 0 or sorted(a["call_index"] for a in attempts) != list(
                range(lock["total_public_executor_calls"])
            ):
                raise ValueError("public executor ledger does not reconcile with the round")
            row["executor_receipt_counts"] = dict(
                sorted(Counter(a["status"] for a in attempts).items())
            )
        docking_path = directory / "docking.json"
        if docking_path.exists():
            docking = unseal(docking_path)
            started = json.loads((directory / "docking_started.json").read_text())
            digest = sha256_file(lock_path)
            if any(r["candidate_lock_sha256"] != digest for r in (started, docking)):
                raise ValueError("oracle receipt differs from candidate lock")
            if (
                not lock["locked_at_utc"]
                <= started["started_at_utc"]
                <= docking["completed_at_utc"]
            ):
                raise ValueError("oracle/lock chronology violated")
            warm = advance_archive(warm, lock, docking)
            if warm != unseal(directory / "archive.json"):
                raise ValueError("archive differs from deterministic docking reduction")
            row.update(
                status="docked",
                docking_seconds=docking["docking_seconds"],
                oracle_failures=docking["oracle_failures"],
                docked_candidates=[
                    candidate_summary(c, lock, parents, seed_counts) for c in docking["docked"]
                ],
            )
            curve.append(score_point(warm))
        rounds.append(row)
        if row["status"] != "docked":
            break
    inputs = {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted(root.rglob("*.json"))
        if p.name not in ("review.json", "verification.json")
    }
    result_path, failure_path = root / "result.json", root / "failure.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    failure = json.loads(failure_path.read_text()) if failure_path.exists() else None
    if result and result["cumulative_oracle_attempts"] != warm["oracle_attempts"]:
        raise ValueError("result accounting differs from full archive")
    return {
        "schema_version": "t4_warm_review_v1",
        "status": result["status"] if result else "incomplete",
        "failure": failure,
        "source_revision": result["code_revision"] if result else None,
        "review_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "review_script_sha256": sha256_file(Path(__file__)),
        "input_sha256": inputs,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "configuration": result["configuration"] if result else None,
        "seed_topology": seed_counts,
        "best_so_far_curve": curve,
        "rounds": rounds,
        "cumulative_oracle_attempts": warm["oracle_attempts"],
        "elapsed_seconds": result["elapsed_seconds"] if result else None,
        "limitations": [
            "guided-only inspected development cell; no matched continuation",
            "unseeded docking; score differences can include oracle noise",
            "coherent displacement is per parent, not cumulative atom lineage",
            "cycle count alone does not identify fused versus pendant construction",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.root)
    if args.output:
        digest = publish_json(args.output, result)
        print(json.dumps({"output": str(args.output), "sha256": digest}))
    else:
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
