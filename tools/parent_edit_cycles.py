"""Prepare, launch and retrieve the approved matched learning cycles."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict
from pathlib import Path

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.parent_edit_model import MUTATION_RECIPE
from compose_v4.control.program_task import initialization_lock
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.parent_edit_cycles import (
    APP,
    APP_NAME,
    CONTRACT,
    KIND,
    PMO_ORACLE_SHA256,
    PREPARED,
    configured,
    fit_measured_edits,
    load_contract,
    zero_query_dispatch_failures,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.trace_shard import encode_state

ROOT = Path(__file__).resolve().parents[1]
PMO_TASKS = ("albuterol_similarity", "isomers_c7h8n2o2", "perindopril_mpo", "scaffold_hop")


def prepare():
    if (ROOT / CONTRACT).exists():
        raise ValueError("campaign contract already frozen; do not relock")
    initial = unseal(ROOT / "configs/t4_program_curriculum_lock.json")
    previous = unseal(ROOT / "configs/t4_second_generation_lock.json")
    inputs = {}
    for path in (
        "configs/t4_program_curriculum_lock.json",
        "configs/t4_second_generation_lock.json",
        "docs/PMO_INIT_BANK.json",
        "docs/INVGNN_FIBER_MOLECULES.json",
        "docs/PARENT_EDIT_LEARNING_CYCLES.md",
        "modal_apps/genmol_t4_opt_app.py",
    ):
        inputs[path] = sha256_file(ROOT / path)
    for cell in ("braf_1", "jak2_1"):
        entries, observations, histories = {}, {}, []
        for arm in ("score_rank", "score_blind"):
            path = f"diagnostics/t4_second_generation/scoring_1/{cell}_{arm}_archive.json"
            inputs[path] = sha256_file(ROOT / path)
            snapshot = json.loads((ROOT / path).read_text())["optimizer"]
            if (
                identity({k: v for k, v in snapshot.items() if k != "snapshot_id"})
                != snapshot["snapshot_id"]
            ):
                raise ValueError("corrupt measured history")
            entries.update(snapshot["entries"])
            for receipt, row in snapshot["observations"].items():
                if receipt in observations and row != observations[receipt]:
                    raise ValueError("warm archive observation conflict")
                observations[receipt] = row
            histories.append(
                {"snapshot_id": snapshot["snapshot_id"], "history": snapshot["history"]}
            )
        path = f"{PREPARED}/{cell}_warm.json"
        payload = {
            "source_group": snapshot["source_group"],
            "oracle_protocol": snapshot["oracle_protocol"],
            "entries": entries,
            "observations": observations,
            "historical_work": histories,
            "label_access": "same union of already paid development outcomes for both arms",
        }
        inputs[path] = publish_json(ROOT / path, payload)
    bank_path = "docs/PMO_INIT_BANK.json"
    bank = json.loads((ROOT / bank_path).read_text())
    records, exclusions = [], []
    for index, smiles in enumerate(bank["smiles"]):
        try:
            # Fresh initialization coordinates, NOT recovery of old slot-addressed traces.
            state = pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)
            if state.n_real_atoms > 40:
                raise ValueError("outside the declared 40-atom executor support")
            execute_program(state, [])
            records.append({"source_id": f"{bank_path}:{index}", "state": encode_state(state)})
        except ValueError as error:
            exclusions.append({"row": index, "smiles": smiles, "reason": str(error)})
    for seed in (20260921, 20260922, 20260923):
        initialized = initialization_lock(
            records, count=16, seed=seed, source_sha256=inputs[bank_path]
        )
        path = f"{PREPARED}/init_{seed}.json"
        inputs[path] = publish_json(ROOT / path, initialized)
    inputs[f"{PREPARED}/initialization_audit.json"] = publish_json(
        ROOT / PREPARED / "initialization_audit.json",
        {
            "source": bank_path,
            "source_sha256": inputs[bank_path],
            "admitted": len(records),
            "excluded": exclusions,
            "role": "task-independent fresh imports; no prior slot replay claimed",
            "new_oracle_calls": 0,
            "source_selection_rule": bank["rule"],
        },
    )
    library_path = "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json"
    inputs[library_path] = sha256_file(ROOT / library_path)
    # Immutable authenticated reference package and probability-parity artifacts.
    reference = json.loads((ROOT / "configs/t4_winner_refinement.json").read_text())
    for path in (
        "configs/t4_winner_refinement.json",
        reference["qualified_inference"]["qualification"]["path"],
        reference["qualified_inference"]["reference_law"]["path"],
    ):
        inputs[path] = sha256_file(ROOT / path)
    units = []
    for kind, names, seeds in (
        ("t4", ("braf_1", "jak2_1"), (20260921, 20260922)),
        ("pmo", PMO_TASKS, (20260921, 20260922, 20260923)),
    ):
        for name in names:
            for seed in seeds:
                for arm in ("score_blind", "learned"):
                    units.append(
                        {
                            "unit_id": f"{kind}_{name}_{seed}_{arm}",
                            "kind": kind,
                            "name": name,
                            "seed": seed,
                            "arm": arm,
                            "budget": 12 if kind == "t4" else 1000,
                            "rounds": 3 if kind == "t4" else 64,
                            "batch_queries": 4 if kind == "t4" else 16,
                        }
                    )
    config = {
        "schema_version": "parent_edit_learning_cycles_v1",
        "units": units,
        "max_docking_calls": 108,
        "max_pmo_queries": 24000,
        "reserved_usd": 20,
        "container_limit": 30,
        "rdkit": "2024.03.5",
        "inputs": inputs,
        "library_path": library_path,
        "t4_protocols": {c: initial["oracle_protocols"][c] for c in ("braf_1", "jak2_1")},
        "incumbents": {c: previous["incumbents"][c] for c in ("braf_1", "jak2_1")},
        "docking_inputs": {
            p: h
            for p, h in initial["runtime_input_sha256"].items()
            if any(k in p for k in ("qvina02", "braf", "jak2"))
        },
        "pmo_protocols": {
            n: {
                "task": n,
                "implementation": "PyTDC==0.3.6",
                "rdkit": "2024.03.5",
                "direction": "maximize",
                "range": [0, 1],
                "prescreen": False,
                "calls_include_initialization": True,
            }
            for n in PMO_TASKS
        },
        "update_recipe": MUTATION_RECIPE,
        "update_rule_id": identity(
            {
                "recipe": MUTATION_RECIPE,
                "source": sha256_file(ROOT / "src/compose_v4/experiments/parent_edit_cycles.py"),
            }
        ),
        "engine_configurations": {
            k: asdict(configured(next(r for r in units if r["kind"] == k))) for k in ("t4", "pmo")
        },
        "compute": {
            "worker_max_containers": 12,
            "group_drivers": 2,
            "confirmation_workers": 2,
            "cpu_per_worker": 1,
            "worker_memory_mib": 8192,
            "worker_timeout_seconds": 2100,
            "driver_timeout_seconds": 7200,
            "campaign_soft_seconds": 1800,
            "retries": 0,
            "gpu": False,
            "reservation_usd": 20,
            "price_source": "https://modal.com/pricing",
            "price_checked": "2026-09-12",
            "cpu_core_second_usd": 0.0000131,
            "gib_second_usd": 0.00000222,
            "all_32_worker_timeout_reservation_usd": 32 * 2100 * (0.0000131 + 8 * 0.00000222),
            "driver_timeout_reservation_usd": 2 * 7200 * (0.0000131 + 2 * 0.00000222),
            "build_confirmation_storage_headroom_usd": 10,
            "expected_wall_seconds": [300, 7200],
            "expected_cost_usd": [0.2, 5],
            "estimate_basis": "previous 69 calls in 102 driver seconds; proposal-heavy PMO not yet throughput-qualified",
            "restart_unit": "completed unit/round; ambiguous oracle reservations require manual recovery",
        },
        "information_regime": "T4 warm public-winner-informed development; PMO no task prescreen, shared T4-derived structural programs",
        "confirmation": "two fresh 1702/1703 evaluations per cell/arm champion across search seeds and each predeclared incumbent; <=12 calls",
        "no_benchmark_superiority_claim": True,
    }
    seal(ROOT / CONTRACT, config)
    print(
        json.dumps(
            {
                "units": len(units),
                "init_admitted": len(records),
                "init_excluded": len(exclusions),
                "new_oracle_calls": 0,
                "contract_sha256": sha256_file(ROOT / CONTRACT),
            }
        )
    )


def bind_runtime():
    """Finalize code/oracle identities without regenerating chemical inputs."""
    c = load_contract(ROOT)
    old_hash = sha256_file(ROOT / CONTRACT)
    old = ROOT / PREPARED / f"prelaunch_contract_{old_hash}.json"
    if not old.exists():
        seal(old, c)
    c["prelaunch_predecessor_sha256"] = old_hash
    c["update_rule_id"] = identity(
        {"recipe": MUTATION_RECIPE, "function_source": inspect.getsource(fit_measured_edits)}
    )
    c["update_function_source"] = inspect.getsource(fit_measured_edits)
    for domain in c["pmo_protocols"].values():
        domain["oracle_module_sha256"] = PMO_ORACLE_SHA256
    seal(ROOT / CONTRACT, c)
    print(
        json.dumps(
            {
                "contract_sha256": sha256_file(ROOT / CONTRACT),
                "chemical_inputs_recomputed": False,
                "new_oracle_calls": 0,
            }
        )
    )


def launch(group, receipt, recover_from=None):
    import modal

    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    if receipt.exists():
        raise ValueError("launch receipt exists; monitor it, do not spawn again")
    c = load_contract(ROOT)
    revision = local_image_revision(expected_commit=assert_synced(strict=True)["commit"])
    body = {
        "image_revision": revision,
        "files_sha256": {
            p: sha256_file(ROOT / p) for p in (CONTRACT, APP, "modal_apps/genmol_t4_opt_app.py")
        },
    }
    if recover_from is not None:
        if group != "pmo":
            raise ValueError("only identified zero-query PMO dispatch failures are recoverable")
        prior = json.loads(recover_from.read_text())
        previous = modal.FunctionCall.from_id(prior["call_id"]).get(timeout=0)
        units = zero_query_dispatch_failures(previous, body["files_sha256"][CONTRACT])
        body["recovery"] = {
            "run_id": previous["task"]["run_id"],
            "result_sha256": identity(previous),
            "units": units,
        }
    task = {**body, "run_id": identity(body), "group": group}
    call = modal.Function.from_name(APP_NAME, "run").spawn(task)
    result = {
        "task": task,
        "call_id": call.object_id,
        "volume": "compose-v4-artifacts",
        "volume_path": f"{KIND}/{task['run_id']}",
        "group": group,
        "reservation_usd": c["reserved_usd"],
    }
    publish_json(receipt, result)
    print(json.dumps({k: v for k, v in result.items() if k != "task"}, indent=2))


def retrieve(receipt, output):
    import modal

    saved = json.loads(receipt.read_text())
    result = modal.FunctionCall.from_id(saved["call_id"]).get(timeout=0)
    seal(output, result)
    print(
        json.dumps(
            {
                "group": result["group"],
                "oracle_calls": result["oracle_calls"],
                "units": [
                    {
                        "id": r["unit"]["unit_id"],
                        "queries": r["oracle_calls"],
                        "last": r.get("history", [None])[-1] if r.get("history") else None,
                        "error": r.get("error"),
                    }
                    for r in result["results"]
                ],
            },
            indent=2,
        )
    )


def review_t4(output):
    """Reproduce the completed T4 comparison from locked receipts and snapshots."""
    import platform
    import subprocess
    from collections import Counter
    from statistics import mean

    result_path = ROOT / "diagnostics/parent_edit_cycles/t4_result.json"
    result = unseal(result_path)
    inputs = {str(result_path.relative_to(ROOT)): sha256_file(result_path)}
    attempts, reports, repetitions = Counter(), [], {}
    for unit in result["results"]:
        path = (
            ROOT / "diagnostics/parent_edit_cycles/t4_units" / (unit["unit"]["unit_id"] + ".json")
        )
        inputs[str(path.relative_to(ROOT))] = sha256_file(path)
        full = unseal(path)
        if full["rows"] != unit["rows"] or full["unit"] != unit["unit"]:
            raise ValueError(f"unit receipt disagrees with group result: {path}")
        for history in full["snapshot"]["history"]:
            attempts.update((a["channel"], a["status"]) for a in history["batch"]["attempts"])
        best = min(unit["rows"], key=lambda r: (r["score"], r["endpoint"]))
        entries = [
            e for e in full["snapshot"]["entries"].values() if e["endpoint"] == best["endpoint"]
        ]
        reports.append(
            {
                "unit": unit["unit"],
                "first_calls": unit["oracle_calls"],
                "best_new_score": best["score"],
                "best_new_endpoint": best["endpoint"],
                "best_program_provenance": [e.get("provenance", {}) for e in entries],
                "proposal_seconds": sum(h["proposal_seconds"] for h in unit["history"]),
                "oracle_seconds": sum(r["seconds"] for r in unit["rows"]),
                "worker_seconds": unit["seconds"],
                "law_seconds": full["law_work"]["law_seconds"],
                "rounds": len(unit["history"]),
                "empty_rounds": sum(h["pool_size"] == 0 for h in unit["history"]),
                "rounds_with_selection_headroom": sum(
                    h["pool_size"] > unit["unit"]["batch_queries"] for h in unit["history"]
                ),
                "history": unit["history"],
            }
        )
    for row in result["confirmations"]:
        for role in row["roles"]:
            group = repetitions.setdefault(row["cell"], {}).setdefault(role, [])
            group.append({k: row[k] for k in ("endpoint", "seed", "score", "status")})
    first_calls = sum(r["first_calls"] for r in reports)
    confirmations = len(result["confirmations"])
    if first_calls + confirmations != result["oracle_calls"]:
        raise ValueError("T4 call accounting does not reconcile")
    report = {
        "schema_version": "parent_edit_t4_review_v1",
        "inputs_sha256": inputs,
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "analysis_git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "run_task": result["task"],
        "first_calls": first_calls,
        "confirmation_calls": confirmations,
        "total_calls": result["oracle_calls"],
        "oracle_failures": sum(
            r["status"] != "complete" for u in result["results"] for r in u["rows"]
        )
        + sum(r["status"] != "complete" for r in result["confirmations"]),
        "first_oracle_seconds": sum(r["oracle_seconds"] for r in reports),
        "mean_first_oracle_seconds": sum(r["oracle_seconds"] for r in reports) / first_calls,
        "summed_proposal_seconds": sum(r["proposal_seconds"] for r in reports),
        "summed_search_worker_seconds": sum(r["worker_seconds"] for r in reports),
        "summed_law_seconds": sum(r["law_seconds"] for r in reports),
        "attempt_counts": [
            {"channel": c, "status": s, "count": n} for (c, s), n in sorted(attempts.items())
        ],
        "units": reports,
        "fresh_confirmations": repetitions,
        "repeat_only_means": {
            cell: {
                role: mean(r["score"] for r in rows if r["status"] == "complete")
                for role, rows in roles.items()
            }
            for cell, roles in repetitions.items()
        },
        "benchmark_claim": False,
        "comparison_limits": [
            "two search seeds per target",
            "same proposal recipe but wall-limited pool sizes differ",
            "two fresh repeats per selected arm champion and predeclared incumbent",
            "warm paid archives and winner-derived structural development information declared",
        ],
    }
    seal(output, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "total_calls",
                    "mean_first_oracle_seconds",
                    "summed_proposal_seconds",
                    "summed_law_seconds",
                    "repeat_only_means",
                )
            },
            indent=2,
        )
    )


def review_pmo(output):
    """Reconcile interrupted PMO receipts without inventing completion or rewards."""
    import platform
    import subprocess

    path = ROOT / "diagnostics/parent_edit_cycles/pmo_reconciliation.json"
    data = unseal(path)
    rows = []
    for name, unit in sorted(data["units"].items()):
        completed = unit["completed"]
        for query, row in completed.items():
            if (
                query not in unit["reservations"]
                or identity({k: v for k, v in row.items() if k != "receipt_id"})
                != row["receipt_id"]
            ):
                raise ValueError(f"unreconciled PMO receipt: {name}/{query}")
        scores = [r["score"] for r in completed.values() if r["status"] == "complete"]
        rows.append(
            {
                "unit": name,
                "charged": len(unit["reservations"]),
                "completed": len(scores),
                "unresolved": sorted(set(unit["reservations"]) - set(completed)),
                "best_partial_score": max(scores) if scores else None,
                "oracle_seconds": sum(r.get("seconds", 0) for r in completed.values()),
                "error": unit.get("failure.json", {}).get("error"),
            }
        )
    report = {
        "schema_version": "parent_edit_pmo_review_v1",
        "input_sha256": sha256_file(path),
        "input_path": str(path.relative_to(ROOT)),
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "analysis_git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "launch": data["launch"],
        "units": rows,
        "charged": sum(r["charged"] for r in rows),
        "completed": sum(r["completed"] for r in rows),
        "unresolved": sum(len(r["unresolved"]) for r in rows),
        "oracle_seconds": sum(r["oracle_seconds"] for r in rows),
        "benchmark_claim": False,
        "completed_comparison": False,
    }
    if (report["charged"], report["completed"], report["unresolved"]) != (
        data["reserved_calls"],
        data["completed_calls"],
        data["unresolved_calls"],
    ):
        raise ValueError("PMO reconciliation changed during analysis")
    seal(output, report)
    print(
        json.dumps(
            {k: report[k] for k in ("charged", "completed", "unresolved", "oracle_seconds")},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "bind", "launch", "retrieve", "review-t4", "review-pmo")
    )
    parser.add_argument("--group", choices=("t4", "pmo"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recover-from", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "bind":
        bind_runtime()
    elif args.mode == "launch":
        if not args.group or args.receipt is None:
            parser.error("launch requires --group and --receipt")
        launch(args.group, args.receipt, args.recover_from)
    elif args.mode == "review-t4":
        if args.output is None:
            parser.error("review-t4 requires --output")
        review_t4(args.output)
    elif args.mode == "review-pmo":
        if args.output is None:
            parser.error("review-pmo requires --output")
        review_pmo(args.output)
    else:
        if args.receipt is None or args.output is None:
            parser.error("retrieve requires --receipt and --output")
        retrieve(args.receipt, args.output)
