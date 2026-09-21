"""TEST A: can the compiler reach a SUPPLIED destination from a NEW source?

The recorded teacher witnesses already prove that routes exist for the
particular (source, destination) pairs they were compiled for.  The missing
evidence is TRANSFER: whether the same compiler can construct a program for a
destination the atlas supplies and a source the live optimizer actually meets.

Every attempt here CONSTRUCTS a program for its own pair.  No recorded absolute
atom-address sequence is replayed, and no recorded correspondence is reused:
``winner_paths.find_path`` recomputes its own MCS correspondence per pair.

The decisive split:
    fails even with the destination supplied -> construction / support / compiler
    succeeds                                 -> transport exists, and autonomous
                                                goal selection is the open problem

Costs ZERO objective evaluations.  DEVELOPMENT_INFORMED_DIAGNOSTIC.

Usage
-----
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
      python scripts/pmo_atlas_transport.py --output diagnostics/pmo_atlas_v1/test_a_transport.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

from compose_v4.experiments.pmo_atlas_routes import (
    DEVELOPMENT_INFORMED_LABEL,
    REGIME_STATEMENT,
    assert_not_scored_consumable,
    file_sha256,
    load_atlas,
    payload_sha256,
)
from compose_v4.experiments.winner_paths import PathConfig, find_path

INIT_BANK = "docs/PMO_INIT_BANK.json"
LIVE_PARENTS = "diagnostics/pmo_population_live_parent_gate_v1/scheduler_result_realizer.json"


def _destinations(repo_root: Path) -> list[dict[str, Any]]:
    """One (task, destination, recorded source) row per distinct compiled spine."""

    dossier = load_atlas(repo_root)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for route in dossier.spines():
        if not route.destination_smiles:
            continue
        key = (route.task, route.destination_smiles)
        rows.setdefault(
            key,
            {
                "task": route.task,
                "destination_smiles": route.destination_smiles,
                "destination_role": route.destination_role,
                "recorded_source_smiles": route.source_smiles,
                "recorded_primitive_steps": route.primitive_steps,
                "artifact": route.artifact,
            },
        )
    return [rows[key] for key in sorted(rows)]


def _sources(repo_root: Path, per_pool: int) -> dict[str, list[dict[str, str]]]:
    bank = json.loads((repo_root / INIT_BANK).read_text())
    # Deterministic, order-preserving spread across the bank; no objective is read.
    smiles = bank["smiles"]
    stride = max(1, len(smiles) // per_pool)
    init = [
        {"smiles": smiles[i * stride], "origin": f"{INIT_BANK}:{i * stride}"}
        for i in range(per_pool)
    ]
    gate = json.loads((repo_root / LIVE_PARENTS).read_text())
    gate = gate.get("payload", gate)
    parents = [
        {"smiles": parent["endpoint"], "origin": parent["source_id"]}
        for parent in gate["parents"][:per_pool]
    ]
    return {"init_bank": init, "live_parents": parents}


def _attempt_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Where the search spent itself, from the compiler's own attempt records."""

    attempts = result.get("attempts") or []
    expanded = sum(int(a.get("expanded") or 0) for a in attempts)
    proposals = sum(int(a.get("attempts") or 0) for a in attempts)
    rejections: Counter[str] = Counter()
    best_residual: int | None = None
    for attempt in attempts:
        for reason, count in (attempt.get("rejections") or {}).items():
            rejections[str(reason)] += int(count)
        residual = attempt.get("best_residual")
        if residual is not None:
            residual = int(residual)
            best_residual = residual if best_residual is None else min(best_residual, residual)
    return {
        "mappings_tried": len(attempts),
        "nodes_expanded": expanded,
        "candidate_actions_proposed": proposals,
        "executor_rejections": dict(sorted(rejections.items(), key=lambda kv: -kv[1])),
        "best_residual": best_residual,
        "mapping_statuses": sorted({str(a.get("status")) for a in attempts}),
    }


def _verify_witness(result: dict[str, Any]) -> dict[str, Any]:
    """Independently replay a returned witness through the production executor."""

    from compose_v4.rewrite.action_codec_v4 import decode_action
    from compose_v4.rewrite.kernel import (
        canonical_state_key,
        editing_v2_semantic_rewrite_system,
    )
    from compose_v4.rewrite.trace_shard import decode_state

    system = editing_v2_semantic_rewrite_system()
    graph = decode_state(result["source_state"])
    for record in result["actions"]:
        family, action = decode_action(record)
        graph = system.apply(graph, family, action)
    endpoint = canonical_state_key(graph)
    return {
        "replayed": True,
        "endpoint_smiles": endpoint,
        "exact_destination_recovery": endpoint == result["target_2d"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="diagnostics/pmo_atlas_v1/test_a_transport.json")
    parser.add_argument("--per-pool", type=int, default=8)
    parser.add_argument(
        "--rescue-from",
        default=None,
        help=(
            "a prior transport artifact; re-runs ONLY its search_unresolved pairs at "
            "the new budget, which separates 'the search ran out' from 'no route exists "
            "inside the declared support'"
        ),
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        nargs="+",
        default=[128],
        help="node budgets to sweep; the first is the production default",
    )
    args = parser.parse_args(argv)

    assert_not_scored_consumable("atlas_transport_diagnostic")
    repo_root = Path(args.repo_root).resolve()
    started = time.time()

    destinations = _destinations(repo_root)
    pools = _sources(repo_root, args.per_pool)

    rescue_pairs: list[dict[str, Any]] | None = None
    if args.rescue_from:
        prior = json.loads((repo_root / args.rescue_from).read_text())
        prior = prior.get("payload", prior)
        rescue_pairs = [
            {
                "task": row["task"],
                "destination_smiles": row["destination_smiles"],
                "destination_role": row["destination_role"],
                "recorded_primitive_steps": row["recorded_primitive_steps"],
                "source_pool": row["source_pool"],
                "source_smiles": row["source_smiles"],
                "source_origin": row["source_origin"],
                "prior_status": row["status"],
                "prior_max_expansions": row["max_expansions"],
                "prior_best_residual": (row.get("search") or {}).get("best_residual"),
            }
            for row in prior["attempts"]
            if row["status"] == "search_unresolved"
        ]
        print(f"rescue mode: {len(rescue_pairs)} unresolved pairs from {args.rescue_from}")

    records: list[dict[str, Any]] = []
    for budget in args.max_expansions:
        config = PathConfig(max_expansions=budget)
        if rescue_pairs is not None:
            plan = [
                (
                    dict(pair),
                    {
                        "pool": pair["source_pool"],
                        "smiles": pair["source_smiles"],
                        "origin": pair["source_origin"],
                    },
                )
                for pair in rescue_pairs
            ]
        else:
            plan = []
            for destination in destinations:
                trials = [
                    {
                        "pool": "recorded_source",
                        "smiles": destination["recorded_source_smiles"],
                        "origin": destination["artifact"],
                    }
                ]
                for pool_name, entries in pools.items():
                    trials.extend({"pool": pool_name, **entry} for entry in entries)
                plan.extend((destination, trial) for trial in trials)

        if True:
            for destination, trial in plan:
                begin = time.time()
                try:
                    result = find_path(trial["smiles"], destination["destination_smiles"], config)
                    error = None
                except (ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
                    result = {"status": f"compiler_exception:{type(exc).__name__}"}
                    error = str(exc)[:300]
                elapsed = time.time() - begin

                status = result.get("status")
                record: dict[str, Any] = {
                    "task": destination["task"],
                    "destination_smiles": destination["destination_smiles"],
                    "destination_role": destination["destination_role"],
                    "recorded_primitive_steps": destination["recorded_primitive_steps"],
                    "prior_status": destination.get("prior_status"),
                    "prior_max_expansions": destination.get("prior_max_expansions"),
                    "prior_best_residual": destination.get("prior_best_residual"),
                    "source_pool": trial["pool"],
                    "source_smiles": trial["smiles"],
                    "source_origin": trial["origin"],
                    "max_expansions": budget,
                    "status": status,
                    "reason": result.get("reason"),
                    "error": error,
                    "seconds": round(elapsed, 3),
                    "primitive_lower_bound": result.get("primitive_lower_bound"),
                    "lower_bound_reason": result.get("lower_bound_reason"),
                    "witness_steps": result.get("witness_steps"),
                    "within_16_edit_witness": result.get("within_16_edit_witness"),
                }
                if result.get("attempts") is not None:
                    record["search"] = _attempt_summary(result)
                if status == "witness_found":
                    record["verification"] = _verify_witness(result)
                    record["family_counts"] = dict(
                        sorted(Counter(a["executor_rule"] for a in result["actions"]).items())
                    )
                records.append(record)
                print(
                    f"{destination['task'][:22]:22s} {trial['pool'][:14]:14s} "
                    f"b={budget:4d} {status!s:28s} steps={result.get('witness_steps')} "
                    f"{elapsed:6.2f}s",
                    flush=True,
                )

    by_status = Counter(r["status"] for r in records)
    by_pool: dict[str, Any] = {}
    for pool in sorted({r["source_pool"] for r in records}):
        subset = [r for r in records if r["source_pool"] == pool]
        found = [r for r in subset if r["status"] == "witness_found"]
        by_pool[pool] = {
            "attempts": len(subset),
            "witness_found": len(found),
            "rate": round(len(found) / len(subset), 4) if subset else None,
            "status_counts": dict(Counter(r["status"] for r in subset)),
            "exact_recovery": sum(
                1 for r in found if r.get("verification", {}).get("exact_destination_recovery")
            ),
            "median_witness_steps": (
                sorted(r["witness_steps"] for r in found)[len(found) // 2] if found else None
            ),
            "total_seconds": round(sum(r["seconds"] for r in subset), 2),
        }
    by_task: dict[str, Any] = {}
    for task in sorted({r["task"] for r in records}):
        subset = [r for r in records if r["task"] == task]
        transfer = [r for r in subset if r["source_pool"] != "recorded_source"]
        by_task[task] = {
            "attempts": len(subset),
            "witness_found": sum(1 for r in subset if r["status"] == "witness_found"),
            "recorded_source_status": next(
                (r["status"] for r in subset if r["source_pool"] == "recorded_source"), None
            ),
            "transfer_attempts": len(transfer),
            "transfer_witness_found": sum(
                1 for r in transfer if r["status"] == "witness_found"
            ),
            "transfer_status_counts": dict(Counter(r["status"] for r in transfer)),
        }

    payload = {
        "schema_version": "pmo_atlas_test_a_transport_v1",
        "information_regime": DEVELOPMENT_INFORMED_LABEL,
        "information_regime_statement": REGIME_STATEMENT,
        "question": (
            "Given the destination, can the production compiler construct a program "
            "for a source-destination pair it has not seen?"
        ),
        "new_oracle_calls": 0,
        "new_docking_calls": 0,
        "compiler": {
            "module": "compose_v4.experiments.winner_paths",
            "entry_point": "find_path",
            "config_defaults": PathConfig().__dict__,
            "correspondence": "recomputed per pair by rdFMCS (elements and topology modes)",
            "replayed_recorded_addresses": False,
        },
        "inputs": {
            INIT_BANK: file_sha256(repo_root / INIT_BANK),
            LIVE_PARENTS: file_sha256(repo_root / LIVE_PARENTS),
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": __import__("rdkit").__version__,
            "numpy": __import__("numpy").__version__,
        },
        "summary": {
            "attempts": len(records),
            "destinations": len(destinations),
            "status_counts": dict(by_status),
            "elapsed_seconds": round(time.time() - started, 2),
        },
        "by_pool": by_pool,
        "by_task": by_task,
        "destinations": destinations,
        "attempts": records,
    }
    document = {"payload": payload, "payload_sha256": payload_sha256(payload)}
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=1, sort_keys=True))

    print("\n--- by pool ---")
    for pool, info in by_pool.items():
        print(f"{pool:16s} {info['witness_found']:3d}/{info['attempts']:3d} "
              f"rate={info['rate']} statuses={info['status_counts']}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
