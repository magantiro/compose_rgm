#!/usr/bin/env python3
"""Prepare, preflight and run the four-task PMO Dynamic-v2.1 diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from rdkit import Chem

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import (
    initial_dynamic_program_batch,
    synthesize_dynamic_program,
)
from compose_v4.control.dynamic_program_synthesis_v2 import (
    SHALLOW_CHANNEL,
    synthesize_structured_program,
)
from compose_v4.control.dynamic_program_synthesis_v21 import (
    CHANNELS,
    DynamicV21ProgramOptimizer,
    arbitrate_candidates,
    initial_allocator_state,
    initial_dynamic_program_batch_v21,
    validate_allocator_state,
)
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.program_task import ProgramTask
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.pmo_dynamic_v21 import (
    ARMS,
    CONTRACT,
    EMPTY_LIBRARY,
    GSK3B_ASSET,
    INITIALIZATION,
    LAUNCH,
    MAX_WORKERS,
    OFFLINE_COMPARATORS,
    ORACLE_ENVIRONMENT,
    ORACLE_SOURCE_SHA256,
    PARITY_RESULT,
    PREFLIGHT,
    RESULT,
    SEARCH_SEED,
    TASK_SEEDS,
    TASKS,
    TOTAL_QUERY_CEILING,
    aggregate_offline,
    configuration,
    load_contract,
    native_oracle,
    policy_payload,
    run_scored_task,
    validate_launch_ready,
    verify_runtime_environment,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]


def _material_inputs() -> tuple[str, ...]:
    return (
        INITIALIZATION,
        EMPTY_LIBRARY,
        "docs/PMO_INIT_BANK.json",
        "docs/PMO_DYNAMIC_V21.md",
        "src/compose_v4/control/adaptive_program_optimizer.py",
        "src/compose_v4/control/dynamic_program_synthesis.py",
        "src/compose_v4/control/dynamic_program_synthesis_v1.py",
        "src/compose_v4/control/dynamic_program_synthesis_v2.py",
        "src/compose_v4/control/dynamic_program_synthesis_v21.py",
        "src/compose_v4/control/edit_program_graph.py",
        "src/compose_v4/control/program_campaign.py",
        "src/compose_v4/control/program_task.py",
        "src/compose_v4/experiments/pmo_dynamic_v21.py",
        "src/compose_v4/experiments/pmo_ivg_oracle_parity.py",
        "tools/pmo_dynamic_v21.py",
    )


def build_offline_comparators(root: Path) -> dict:
    source_path = root / PARITY_RESULT
    source = unseal(source_path)
    if source.get("schema_version") != "pmo_ivg_oracle_parity_result_v1":
        raise ValueError("unexpected PMO parity result schema")
    tasks = {}
    for task in TASKS:
        row = source["results"][task]
        tasks[task] = {
            "locked_compose_best_score": row["best_score"],
            "locked_compose_final_top10": row["final_top10"],
            "locked_compose_auc_top10_official_10k": row["auc_top10_official_10k"],
            "ivg_no_prescreen_auc_top10": row["ivg_no_prescreen_auc_top10"],
            "ivg_prescreen_auc_top10": row["ivg_prescreen_auc_top10"],
        }
    return {
        "schema_version": "pmo_dynamic_v21_offline_comparators_v1",
        "source_path": PARITY_RESULT,
        "source_sha256": sha256_file(source_path),
        "tasks": tasks,
        "runtime_selection_access": False,
        "interpretation": (
            "post-run arithmetic only; locked COMPOSE candidates are task/panel-informed "
            "development evidence and never initialize or select this run"
        ),
    }


def prepare(root: Path = ROOT) -> dict:
    contract_path = root / CONTRACT
    comparator_path = root / OFFLINE_COMPARATORS
    if contract_path.exists() or comparator_path.exists():
        raise ValueError(
            "PMO Dynamic-v2.1 contract artifacts already exist; do not relock"
        )
    comparator = build_offline_comparators(root)
    seal(comparator_path, comparator)
    initialized = json.loads((root / INITIALIZATION).read_text())
    contract = {
        "schema_version": "pmo_dynamic_v21_matched_contract_v1",
        "scientific_problem": (
            "compare the unchanged route-free Dynamic-v0 shallow proposer with the "
            "Dynamic-v2.1 pooled shallow/structured proposer on four PMO objectives"
        ),
        "primary_output": (
            "matched arm-specific complete exact-replay edit programs, charged PMO "
            "rewards and best/top-ten trajectories"
        ),
        "central_claim": (
            "post-filter structured competition can improve over the same pure shallow "
            "Dynamic-v0 search law without task-informed routes"
        ),
        "validation_setting": (
            "one predeclared seed and two matched arms on four answer-known PMO tasks"
        ),
        "support": (
            "complete supported molecular graphs under the existing 40-active-atom, "
            "32-primitive, eight-block exact executor; PMO validity only; stereochemistry "
            "and formal-charge changes remain outside support"
        ),
        "dynamic_v21_development": policy_payload(),
        "controller": json.loads(json.dumps(configuration().__dict__)),
        "initialization": {
            "path": INITIALIZATION,
            "sha256": sha256_file(root / INITIALIZATION),
            "lock_sha256": initialized["lock_sha256"],
            "source_sha256": initialized["source_sha256"],
            "count": initialized["count"],
            "task_independent": True,
            "all_scores_charged": True,
        },
        "library": {
            "path": EMPTY_LIBRARY,
            "sha256": sha256_file(root / EMPTY_LIBRARY),
            "programs": 0,
            "task_informed_curriculum": False,
        },
        "oracle": {
            "tasks": list(TASKS),
            "adapter": "native PyTDC Oracle(name=task)",
            "environment": ORACLE_ENVIRONMENT,
            "source_files": ORACLE_SOURCE_SHA256,
            "gsk3b_asset": GSK3B_ASSET,
            "direction": "maximize",
            "range": [0, 1],
            "prescreen": False,
            "calls_include_initialization": True,
        },
        "compute": {
            "workers": MAX_WORKERS,
            "cpu_per_worker": 1,
            "gpu": False,
            "automatic_retries": 0,
            "charged_units": len(ARMS) * len(TASKS),
            "restart_unit": (
                "none automatically; any started task or unresolved query requires audit"
            ),
        },
        "postrun_only_comparator": {
            "path": OFFLINE_COMPARATORS,
            "sha256": sha256_file(comparator_path),
            "available_to_runtime": False,
        },
        "information_regime": {
            "answer_known_development_tasks": True,
            "matched_arms": list(ARMS),
            "identical_initialization_and_seeds_by_arm": True,
            "task_or_panel_informed_programs": False,
            "initial_task_specific_complete_routes": 0,
            "prior_pmo_outcomes_available_to_runtime": False,
            "ivg_values_available_to_runtime": False,
            "held_out_claim": False,
            "general_pmo_claim": False,
            "t4_affected": False,
        },
        "inputs": {path: sha256_file(root / path) for path in _material_inputs()},
    }
    seal(contract_path, contract)
    load_contract(root)
    return {
        "contract": CONTRACT,
        "contract_sha256": sha256_file(contract_path),
        "offline_comparators": OFFLINE_COMPARATORS,
        "offline_comparators_sha256": sha256_file(comparator_path),
        "tasks": list(TASKS),
        "new_oracle_calls": 0,
    }


def _replay(candidate: dict, config, *, label: str) -> None:
    program = EditProgram.from_payload(candidate["program"])
    _, replay = execute_program_graph(
        decode_state(candidate["source_state"]),
        compile_program_graph(program),
        tuple(candidate["assignment"]),
        max_primitives=config.max_primitives,
        max_blocks=config.max_blocks,
    )
    if replay != candidate["trace"] or replay["endpoint"] != candidate["endpoint"]:
        raise ValueError(f"PMO Dynamic-v2.1 exact replay changed: {label}")


def _representative_channel(channel: str) -> dict:
    from compose_v4.experiments.editing_v2_evaluation_semantics import (
        production_state_from_smiles,
    )

    source = production_state_from_smiles("C1CCCCC1CCO", max_atoms=48)
    synthesizer = (
        synthesize_dynamic_program
        if channel == SHALLOW_CHANNEL
        else synthesize_structured_program
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([SEARCH_SEED, 311, CHANNELS.index(channel)])
    )
    for attempt in range(64):
        try:
            _, program, binding, trace, metadata = synthesizer(
                source,
                rng,
                max_modules=3,
                max_primitives=32,
                max_blocks=8,
            )
        except ValueError:
            continue
        _, replay = execute_program_graph(
            source,
            compile_program_graph(program),
            binding,
            max_primitives=32,
            max_blocks=8,
        )
        if replay != trace:
            raise ValueError(f"PMO representative {channel} replay changed")
        return {
            "channel": channel,
            "attempt": attempt,
            "endpoint": trace["endpoint"],
            "program_id": program.program_id,
            "metadata": metadata,
        }
    raise ValueError(f"PMO representative {channel} produced no legal program")


def build_structural_preflight(root: Path = ROOT) -> dict:
    contract = load_contract(root)
    initialized = json.loads((root / INITIALIZATION).read_text())
    validity = {}
    for row in initialized["candidates"]:
        source = decode_state(row["state"])
        if source.n_real_atoms > 40:
            raise ValueError("PMO initialization exceeds 40-active-atom support")
        molecule = Chem.MolFromSmiles(row["endpoint"])
        if molecule is None or Chem.MolToSmiles(molecule) != row["endpoint"]:
            raise ValueError("PMO initialization endpoint lost canonical validity")
        for task_name in TASKS:
            evaluated = ProgramTask(
                task_name, identity(_protocol_for_preflight(contract, task_name)), "pmo"
            ).endpoint_evaluator()({"smiles": row["endpoint"]})
            if evaluated != {"smiles": row["endpoint"], "oracle_eligible": True}:
                raise ValueError("PMO validity inherited a non-validity endpoint gate")
        validity[row["endpoint"]] = True

    representative = {channel: _representative_channel(channel) for channel in CHANNELS}
    rows = []
    for task_name in TASKS:
        seed = TASK_SEEDS[task_name]
        parent_rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 31]))
        source_index = int(parent_rng.integers(len(initialized["candidates"])))
        source = decode_state(initialized["candidates"][source_index]["state"])
        config = replace(
            configuration(seed),
            attempts_per_batch=2,
            candidates_per_batch=4,
            wall_seconds=3600,
        )
        task = ProgramTask(
            task_name, identity(_protocol_for_preflight(contract, task_name)), "pmo"
        )
        kwargs = {
            "source_group": initialized["lock_sha256"],
            "oracle_protocol": task.oracle_protocol,
            "eligibility": task.endpoint_evaluator(),
        }
        v0 = initial_dynamic_program_batch(source, (), config, **kwargs)
        v0_repeat = initial_dynamic_program_batch(source, (), config, **kwargs)
        v21 = initial_dynamic_program_batch_v21(source, (), config, **kwargs)
        v21_repeat = initial_dynamic_program_batch_v21(source, (), config, **kwargs)
        if v0["batch_id"] != v0_repeat["batch_id"]:
            raise ValueError(
                f"PMO Dynamic-v0 initial pool is nondeterministic: {task_name}"
            )
        if v21["batch_id"] != v21_repeat["batch_id"]:
            raise ValueError(
                f"PMO Dynamic-v2.1 initial pool is nondeterministic: {task_name}"
            )
        validate_allocator_state(v21["allocator_state_after_preparation"])
        for arm, batch in (("dynamic_v0", v0), ("dynamic_v21", v21)):
            for candidate in batch["candidates"]:
                _replay(candidate, config, label=f"{arm}/{task_name}")
        shallow = [
            row
            for row in v21["proposal_pool"]["candidates"]
            if row["provenance"]["planner_channel"] == SHALLOW_CHANNEL
        ]
        common = min(len(v0["candidates"]), len(shallow))
        if common and [row["endpoint"] for row in shallow[:common]] != [
            row["endpoint"] for row in v0["candidates"][:common]
        ]:
            raise ValueError(f"PMO Dynamic-v2.1 changed shallow v0 prefix: {task_name}")
        rows.append(
            {
                "task": task_name,
                "search_seed": seed,
                "initialization_source_index": source_index,
                "dynamic_v0_batch_id": v0["batch_id"],
                "dynamic_v21_batch_id": v21["batch_id"],
                "dynamic_v0_attempts": len(v0["attempts"]),
                "dynamic_v0_eligible": len(v0["candidates"]),
                "dynamic_v21_attempts": len(v21["attempts"]),
                "dynamic_v21_eligible_union": len(v21["proposal_pool"]["candidates"]),
                "dynamic_v21_selected": len(v21["candidates"]),
                "shallow_v0_prefix_checked": common,
                "identical_initialization_lock": initialized["lock_sha256"],
                "identical_configuration": asdict(configuration(seed)),
                "identical_oracle_protocol": task.oracle_protocol,
            }
        )

    left = DynamicV21ProgramOptimizer(
        configuration(17),
        source_group="preflight",
        oracle_protocol="zero-oracle",
        hierarchy=None,
    )
    right = DynamicV21ProgramOptimizer(
        configuration(17),
        source_group="preflight",
        oracle_protocol="zero-oracle",
        hierarchy=None,
    )
    right.structured_rng.random(100)
    independent = (
        left.shallow_rng.bit_generator.state == right.shallow_rng.bit_generator.state
    )
    if not independent:
        raise ValueError("PMO structured RNG advanced the shallow stream")

    fallback = {}
    for available_channel in CHANNELS:
        candidates = [
            {
                "candidate_id": f"{available_channel}-{index}",
                "provenance": {"planner_channel": available_channel},
            }
            for index in range(3)
        ]
        selected, receipt = arbitrate_candidates(
            candidates,
            limit=2,
            state=initial_allocator_state(score_direction="maximize"),
            rng=np.random.default_rng(9),
        )
        if len(selected) != 2:
            raise ValueError("PMO empty-channel fallback lost available candidates")
        fallback[available_channel] = receipt
    return {
        "schema_version": "pmo_dynamic_v21_matched_preflight_v1",
        "passed": len(rows) == len(TASKS),
        "contract_sha256": sha256_file(root / CONTRACT),
        "implementation_sha256": contract["inputs"],
        "tasks": rows,
        "task_independent_initialization": {
            "path": INITIALIZATION,
            "lock_sha256": initialized["lock_sha256"],
            "count": len(validity),
            "all_scores_charged": True,
        },
        "representative_legal_execution": representative,
        "rng_streams_independent": independent,
        "empty_channel_fallback": fallback,
        "initial_route_archive_rows": 0,
        "initial_route_archive_rows_by_arm": {arm: 0 for arm in ARMS},
        "task_informed_curriculum_rows": 0,
        "runtime_comparison_inputs": 0,
        "t4_similarity_qed_sa_gates": 0,
        "new_oracle_calls": 0,
    }


def _protocol_for_preflight(contract: dict, task_name: str) -> dict:
    asset = (
        {"gsk3b_current.pkl": contract["oracle"]["gsk3b_asset"]["sha256"]}
        if task_name == "gsk3b"
        else {}
    )
    return {
        "task": task_name,
        "implementation": "native PyTDC Oracle",
        "environment": contract["oracle"]["environment"],
        "source_sha256": contract["oracle"]["source_files"],
        "asset_sha256": asset,
        "direction": "maximize",
        "range": [0, 1],
        "prescreen": False,
        "calls_include_initialization": True,
    }


def preflight(root: Path = ROOT) -> dict:
    path = root / PREFLIGHT
    if path.exists():
        raise ValueError("PMO Dynamic-v2.1 preflight already exists; do not overwrite")
    body = _preflight_body(root)
    seal(path, body)
    return body


def _preflight_body(root: Path) -> dict:
    began = __import__("time").perf_counter()
    body = build_structural_preflight(root)
    contract = load_contract(root)
    body["oracle_environment"] = verify_runtime_environment(root, contract)
    adapters = {}
    for task_name in TASKS:
        with native_oracle(root, contract, task_name):
            adapters[task_name] = "constructed_without_evaluation"
    body["native_oracle_construction"] = adapters
    body["seconds"] = __import__("time").perf_counter() - began
    body["code_revision_at_preflight"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return body


def refresh_prequery(root: Path = ROOT) -> dict:
    """Reseal implementation hashes before scoring, preserving zero-call lineage."""
    if (root / LAUNCH).exists():
        raise ValueError("cannot refresh PMO Dynamic-v2.1 after a launch exists")
    path = root / CONTRACT
    contract = unseal(path)
    if contract.get("dynamic_v21_development") != policy_payload() or contract.get(
        "controller"
    ) != json.loads(json.dumps(configuration().__dict__)):
        raise ValueError("PMO Dynamic-v2.1 scientific policy changed before refresh")
    prior_contract = sha256_file(path)
    prior_preflight = (
        sha256_file(root / PREFLIGHT) if (root / PREFLIGHT).exists() else None
    )
    contract["inputs"] = {
        relative: sha256_file(root / relative) for relative in _material_inputs()
    }
    contract.setdefault("prequery_repairs", []).append(
        {
            "reason": "reseal formatting-only or focused pre-oracle implementation identities",
            "prior_contract_sha256": prior_contract,
            "prior_preflight_sha256": prior_preflight,
            "oracle_calls": 0,
            "scientific_policy_changed": False,
        }
    )
    seal(path, contract)
    load_contract(root)
    body = _preflight_body(root)
    seal(root / PREFLIGHT, body)
    return {
        "status": "prequery_contract_and_preflight_refreshed",
        "prior_contract_sha256": prior_contract,
        "prior_preflight_sha256": prior_preflight,
        "contract_sha256": sha256_file(path),
        "preflight_sha256": sha256_file(root / PREFLIGHT),
        "new_oracle_calls": 0,
    }


def _worker(root: str, launch: dict, arm: str, task_name: str) -> dict:
    return run_scored_task(Path(root), launch, arm, task_name)


def launch(root: Path = ROOT, *, workers: int = MAX_WORKERS) -> dict:
    if (root / LAUNCH).exists():
        raise ValueError("PMO Dynamic-v2.1 launch exists; do not relaunch")
    _, preflight_row = validate_launch_ready(root, workers=workers)
    body = {
        "schema_version": "pmo_dynamic_v21_matched_launch_v1",
        "contract_sha256": sha256_file(root / CONTRACT),
        "preflight_sha256": sha256_file(root / PREFLIGHT),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "tasks": list(TASKS),
        "arms": list(ARMS),
        "units": [{"arm": arm, "task": task} for task in TASKS for arm in ARMS],
        "task_seeds": TASK_SEEDS,
        "configuration": asdict(configuration()),
        "configuration_id": identity(asdict(configuration())),
        "workers": workers,
        "cpu_per_worker": 1,
        "charged_query_ceiling": TOTAL_QUERY_CEILING,
        "automatic_retries": 0,
        "preflight_schema": preflight_row["schema_version"],
    }
    launch_row = {**body, "run_id": identity(body)}
    seal(root / LAUNCH, launch_row)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_worker, str(root), launch_row, arm, task_name): (
                arm,
                task_name,
            )
            for task_name in TASKS
            for arm in ARMS
        }
        for future in as_completed(futures):
            results.append(future.result())
    aggregate = aggregate_offline(root, launch_row, results)
    seal(root / RESULT, aggregate)
    return aggregate


def _live_curve(folder: Path) -> tuple[list[dict], dict]:
    charged, scores, failures, unresolved = 0, [], 0, 0
    for started in sorted((folder / "oracle").glob("query_*/started.json")):
        charged += 1
        result = started.with_name("result.json")
        if not result.exists():
            unresolved += 1
            continue
        row = json.loads(result.read_text())
        if row.get("status") == "complete":
            scores.append(float(row["score"]))
        else:
            failures += 1
    curve, top_values = [], []
    for count in range(1, len(scores) + 1):
        top_values.append(scores[count - 1])
        top_values = sorted(top_values, reverse=True)[:10]
        curve.append(
            {
                "completed_queries": count,
                "best_score": top_values[0],
                "top10_score": sum(top_values) / len(top_values),
            }
        )
    if curve:
        curve[-1]["auc_top10_development_1000"] = _auc_from_top10_curve(
            curve, budget=1000
        )
        curve[-1]["auc_top10_official_10k"] = _auc_from_top10_curve(curve, budget=10000)
    accounting = {
        "actual_charged_queries": charged,
        "completed_queries": len(scores),
        "failed_queries": failures,
        "unresolved_queries": unresolved,
        "cache_hits_are_uncharged": True,
    }
    return curve, accounting


def _auc_from_top10_curve(curve: list[dict], *, budget: int) -> float:
    count = len(curve)
    checkpoints = [*range(100, count, 100), count]
    previous_x, previous_y, area = 0, 0.0, 0.0
    for checkpoint in checkpoints:
        current_y = curve[checkpoint - 1]["top10_score"]
        area += (checkpoint - previous_x) * (current_y + previous_y) / 2
        previous_x, previous_y = checkpoint, current_y
    area += (budget - count) * previous_y
    return area / budget


def _proposal_status(folder: Path, arm: str) -> dict:
    statuses, channels, selected = Counter(), Counter(), Counter()
    last_allocation, pending_rounds, complete_rounds = None, [], []
    for pending in sorted((folder / "campaign").glob("round_*/pending.json")):
        batch = json.loads(pending.read_text())["batch"]
        round_name = pending.parent.name
        complete = pending.with_name("complete.json")
        (complete_rounds if complete.exists() else pending_rounds).append(round_name)
        for attempt in batch.get("attempts", ()):
            statuses[attempt.get("status", "unknown")] += 1
            channel = attempt.get("planner_channel", attempt.get("channel", "unknown"))
            channels[channel] += 1
        for candidate in batch.get("candidates", ()):
            channel = candidate.get("provenance", {}).get(
                "planner_channel", "dynamic_v0_shallow"
            )
            selected[channel] += 1
        if arm == "dynamic_v21":
            last_allocation = batch.get("allocation", last_allocation)
    return {
        "attempt_status_counts": dict(sorted(statuses.items())),
        "attempt_channel_counts": dict(sorted(channels.items())),
        "selected_channel_counts": dict(sorted(selected.items())),
        "latest_v21_allocation": last_allocation,
        "completed_rounds": complete_rounds,
        "pending_rounds": pending_rounds,
        "latest_resume_snapshot": (
            None
            if not complete_rounds
            else str(folder / "campaign" / complete_rounds[-1] / "complete.json")
        ),
        "pending_round_recovery": "manual audit; no automatic retry",
    }


def _live_unit_status(root: Path, launch_row: dict, arm: str, task_name: str) -> dict:
    folder = (
        root
        / "diagnostics/pmo_dynamic_v21/runs"
        / launch_row["run_id"]
        / arm
        / task_name
    )
    curve, accounting = _live_curve(folder)
    row = {
        "arm": arm,
        "task": task_name,
        "search_seed": launch_row["task_seeds"][task_name],
        "contract_sha256": launch_row["contract_sha256"],
        "code_revision": launch_row["code_revision"],
        "configuration": launch_row["configuration"],
        "configuration_id": launch_row["configuration_id"],
        **accounting,
        "score_curve": curve,
        "current": None if not curve else curve[-1],
        "proposal": _proposal_status(folder, arm),
    }
    if (folder / "result.json").exists():
        result = unseal(folder / "result.json")
        return {**row, "status": "complete", "termination": result["termination"]}
    if (folder / "failure.json").exists():
        failure = unseal(folder / "failure.json")
        return {**row, "status": "failed", "error": failure["error"]}
    if (folder / "progress.json").exists():
        row["latest_progress"] = json.loads((folder / "progress.json").read_text())
    row["status"] = "started" if (folder / "started.json").exists() else "pending"
    return row


def status(root: Path = ROOT) -> dict:
    if not (root / LAUNCH).exists():
        return {"status": "not_launched", "new_oracle_calls": 0}
    launch_row = unseal(root / LAUNCH)
    rows = {
        arm: {task: _live_unit_status(root, launch_row, arm, task) for task in TASKS}
        for arm in ARMS
    }
    return {"status": "launched", "launch": launch_row, "arms": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "preflight", "refresh-prequery", "launch", "status"),
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    if args.action == "prepare":
        row = prepare()
    elif args.action == "preflight":
        row = preflight()
    elif args.action == "refresh-prequery":
        row = refresh_prequery()
    elif args.action == "launch":
        row = launch(workers=args.workers)
    else:
        row = status()
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
