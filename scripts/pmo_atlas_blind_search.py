"""TEST C, arm 1: run the optimizer BLIND and record its whole trajectory.

The optimizer under test is the same frozen controller Test B started inside a
teacher region, given the production task-independent initialization and
nothing else.  The atlas is loaded here only so the run can be REFUSED if any
answer-known molecule reaches the optimizer's inputs; it is never passed in.

What is recorded is the full charged trajectory -- every molecule in ledger
order with its score, its parent, its proposal channel and its 48-slot endpoint
state -- because the question "did it enter a productive region, and by what
route" cannot be answered from a score curve.

DEVELOPMENT_INFORMED_DIAGNOSTIC: nothing produced here may seed or train a
scored no-prescreen run.  Every objective evaluation is a counted diagnostic
call and is reported.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v21 import (
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask
from compose_v4.experiments.pmo_atlas_discovery import (
    BLIND_INITIALIZATION_PATH,
    assert_optimizer_blind,
    atlas_molecules,
    load_blind_initialization,
    read_blind_trajectory,
)
from compose_v4.experiments.pmo_atlas_objectives import (
    ReferenceExpectation,
    assert_reference_panel,
    build_oracle,
)
from compose_v4.experiments.pmo_atlas_routes import (
    DEVELOPMENT_INFORMED_LABEL,
    REGIME_STATEMENT,
    assert_not_scored_consumable,
    file_sha256,
    load_atlas,
    payload_sha256,
)
from compose_v4.experiments.pmo_population_v1 import configuration

CHECKPOINTS = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"


def _jump_checkpoint(repo_root: Path) -> dict[str, Any]:
    envelope = json.loads((repo_root / CHECKPOINTS).read_text())
    payload = envelope["payload"]["checkpoints"]["shared_all_routes"]
    if payload.get("fit_scope") != "shared_all_routes":
        raise ValueError("the PMO runtime checkpoint must be shared and task blind")
    return payload


def _run_one(
    repo_root: Path,
    folder: Path,
    task_name: str,
    initialization: dict[str, Any],
    atlas: frozenset[str],
    oracle,
    budget: int,
    queries_per_round: int,
    seed: int,
) -> dict[str, Any]:
    optimizer_kwargs = {
        "jump_checkpoint": _jump_checkpoint(repo_root),
        "enable_online_memory": True,
    }
    # The blindness guard runs BEFORE the first charged call, so a leaked
    # answer-known input cannot manufacture a plausible trajectory.
    blindness = assert_optimizer_blind(
        initialization=initialization,
        optimizer_kwargs=optimizer_kwargs,
        library=(),
        atlas=atlas,
    )
    protocol = identity(
        {
            "role": "pmo_atlas_blind_discovery_diagnostic",
            "task": task_name,
            "initialization": BLIND_INITIALIZATION_PATH,
            "budget": budget,
            "information_regime": DEVELOPMENT_INFORMED_LABEL,
        }
    )
    task = ProgramTask(task_name, protocol, "pmo")
    ledger = ProgramQueryLedger(folder / "oracle", task, oracle, budget=budget)
    rounds = max(1, -(-budget // queries_per_round))
    began = time.time()
    run_program_campaign(
        output=folder / "campaign",
        task=task,
        config=configuration(seed),
        initialization=initialization,
        library=(),
        ledger=ledger,
        rounds=rounds,
        queries_per_round=queries_per_round,
        hierarchy=None,
        fit_model=None,
        stagnation_rounds=None,
        bootstrap_rounds=1,
        initialization_mode="all_scored_pool",
        initial_parent_fraction=0.2,
        optimizer_type=PmoPopulationController,
        optimizer_kwargs=optimizer_kwargs,
        initial_batch_fn=initial_dynamic_program_batch_v21,
    )
    elapsed = time.time() - began

    trajectory = read_blind_trajectory(folder)
    scores = [row.score for row in trajectory]
    ranked = sorted(scores, reverse=True)
    top_ten = ranked[:10]
    return {
        "task": task_name,
        "charged_calls": len(trajectory),
        "budget": budget,
        "distinct_molecules": len({row.endpoint for row in trajectory}),
        "best_score": max(scores) if scores else None,
        "top_ten_mean": sum(top_ten) / len(top_ten) if top_ten else None,
        "initialization_best": max(
            (row.score for row in trajectory if row.role == "initialization"), default=None
        ),
        "seconds": round(elapsed, 2),
        "blindness": blindness,
        "oracle_ledger": oracle.ledger(),
        "run_dir": str(folder),
        "trajectory": [
            {
                "index": row.index,
                "endpoint": row.endpoint,
                "score": row.score,
                "role": row.role,
                "round_index": row.round_index,
                "parent_endpoint": row.parent_endpoint,
                "parent_score": row.parent_score,
                "parent_label_disagrees": row.parent_label_disagrees,
                "planner_channel": row.planner_channel,
                "mode": row.mode,
                "program_primitives": row.program_primitives,
                "heavy_atoms": row.heavy_atoms,
            }
            for row in trajectory
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--queries-per-round", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-panel", default="diagnostics/pmo_atlas_v1/reference_panel.json")
    args = parser.parse_args(argv)

    assert_not_scored_consumable("atlas_blind_discovery_diagnostic")
    repo_root = Path(args.repo_root).resolve()
    work = (
        Path(args.work_dir).expanduser()
        if args.work_dir
        else Path.home() / "compose_pmo_atlas_runs" / "test_c_blind"
    )
    work.mkdir(parents=True, exist_ok=True)

    dossier = load_atlas(repo_root)
    tasks = args.tasks or list(dossier.tasks)
    initialization = load_blind_initialization(repo_root)

    panel_document = json.loads((repo_root / args.reference_panel).read_text())
    panel_payload = panel_document.get("payload", panel_document)

    started = time.time()
    runs: list[dict[str, Any]] = []
    controls: dict[str, Any] = {}

    def _payload(partial: bool) -> dict[str, Any]:
        return {
            "schema_version": "pmo_atlas_test_c_blind_search_v1",
            "partial": partial,
            "information_regime": DEVELOPMENT_INFORMED_LABEL,
            "information_regime_statement": REGIME_STATEMENT,
            "question": (
                "Started from the production task-independent bank with no atlas, what "
                "does the frozen controller actually visit, and by what route?"
            ),
            "controller": {
                "class": "PmoPopulationController",
                "enable_online_memory": True,
                "arm": "B_memory",
                "jump_checkpoint": CHECKPOINTS,
                "jump_checkpoint_sha256": file_sha256(repo_root / CHECKPOINTS),
                "atlas_supplied_to_optimizer": False,
            },
            "initialization": {
                "path": BLIND_INITIALIZATION_PATH,
                "file_sha256": file_sha256(repo_root / BLIND_INITIALIZATION_PATH),
                "task_independent": True,
                "count": initialization["count"],
            },
            "budget_per_task": args.budget,
            "queries_per_round": args.queries_per_round,
            "seed": args.seed,
            "tasks_probed": tasks,
            "diagnostic_oracle_calls": sum(run["charged_calls"] for run in runs),
            "reference_panel": {
                "artifact": args.reference_panel,
                "file_sha256": file_sha256(repo_root / args.reference_panel),
                "controls": controls,
            },
            "software": {
                "python": platform.python_version(),
                "rdkit": __import__("rdkit").__version__,
                "numpy": __import__("numpy").__version__,
                "pytdc": __import__("importlib.metadata", fromlist=["version"]).version("PyTDC"),
            },
            "elapsed_seconds": round(time.time() - started, 2),
            "runs": runs,
        }

    def _write(partial: bool) -> dict[str, Any]:
        payload = _payload(partial)
        document = {"payload": payload, "payload_sha256": payload_sha256(payload)}
        output = repo_root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=1, sort_keys=True))
        return payload

    for task_name in tasks:
        atlas = atlas_molecules(dossier, task=task_name)
        oracle = build_oracle(task_name)
        panel = tuple(
            ReferenceExpectation(item["smiles"], item["value"], item["label"])
            for item in panel_payload["panels"][task_name]
        )
        controls[task_name] = assert_reference_panel(oracle, panel)
        record = _run_one(
            repo_root,
            work / f"{task_name}__blind_{args.budget}",
            task_name,
            initialization,
            atlas,
            oracle,
            args.budget,
            args.queries_per_round,
            args.seed,
        )
        runs.append(record)
        _write(partial=True)
        print(
            f"{task_name:26s} calls={record['charged_calls']:4d} "
            f"best={record['best_score']:.4f} top10={record['top_ten_mean']:.4f} "
            f"init_best={record['initialization_best']:.4f} {record['seconds']:.0f}s",
            flush=True,
        )

    payload = _write(partial=False)
    print(f"\ndiagnostic oracle calls: {payload['diagnostic_oracle_calls']}")
    print(f"wrote {repo_root / args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
