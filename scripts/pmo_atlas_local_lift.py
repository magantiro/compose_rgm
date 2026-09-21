"""TEST B: does landing in a teacher region make local optimization strong?

The assumption under test is that reaching a good region is the hard part and
local search then fills a strong top-ten set.  PMO scores a top-ten AUC curve,
not a champion, so "preserves one good seed" and "fills a strong top ten" are
different outcomes with the same best-score headline.

Protocol
--------
For each task, take structural checkpoints along the recorded spine, start the
SAME frozen local controller from each with the SAME small budget, and measure
what it produces.  After initialization the controller is given the checkpoint
molecule and nothing else: no destination graph, no teacher suffix, no atlas.

Reported per run: best score, the seed's own score, how many DISTINCT molecules
beat the seed, and the mean of the ten best NEW candidates (the seed excluded,
because a run that merely preserves its seed must not score as a lift).

Costs are real diagnostic oracle evaluations and are counted per run.
DEVELOPMENT_INFORMED_DIAGNOSTIC: checkpoint positions come from answer-known
routes, so nothing here may inform a scored no-prescreen run.
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
    route_checkpoints,
)
from compose_v4.experiments.pmo_population_v1 import configuration

CHECKPOINTS = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"

#: Positions probed on every spine.  ``early`` is the far control: if a lift is
#: a property of the region rather than of the controller, ``early`` must be
#: measurably worse than ``anchor``.
PROBE_POSITIONS: tuple[tuple[str, float], ...] = (
    ("early", 0.25),
    ("near_anchor", 0.85),
    ("anchor", 1.0),
)


def _jump_checkpoint(repo_root: Path) -> dict[str, Any]:
    envelope = json.loads((repo_root / CHECKPOINTS).read_text())
    payload = envelope["payload"]["checkpoints"]["shared_all_routes"]
    if payload.get("fit_scope") != "shared_all_routes":
        raise ValueError("the PMO runtime checkpoint must be shared and task blind")
    return payload


def _initialization(smiles: str, state: dict[str, Any], origin: str) -> dict[str, Any]:
    body = {
        "candidates": [{"endpoint": smiles, "source_id": origin, "state": state}],
        "count": 1,
        "task_independent": False,
        "provenance": origin,
        "information_regime": DEVELOPMENT_INFORMED_LABEL,
    }
    return {**body, "lock_sha256": identity(body)}


def _run_one(
    repo_root: Path,
    folder: Path,
    task_name: str,
    checkpoint,
    oracle,
    budget: int,
    queries_per_round: int,
    seed: int,
) -> dict[str, Any]:
    protocol = identity(
        {
            "role": "pmo_atlas_local_lift_diagnostic",
            "task": task_name,
            "checkpoint": checkpoint.label,
            "seed_smiles": checkpoint.smiles,
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
        initialization=_initialization(
            checkpoint.smiles, checkpoint.state, checkpoint.program_id
        ),
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
        optimizer_kwargs={
            "jump_checkpoint": _jump_checkpoint(repo_root),
            "enable_online_memory": True,
        },
        initial_batch_fn=initial_dynamic_program_batch_v21,
    )
    elapsed = time.time() - began

    rows = list(ledger.rows)
    scores = {row["endpoint"]: float(row["score"]) for row in rows}
    seed_score = scores.get(checkpoint.smiles)
    new = {smiles: value for smiles, value in scores.items() if smiles != checkpoint.smiles}
    ranked = sorted(new.values(), reverse=True)
    top_ten_new = ranked[:10]
    above_seed = (
        [v for v in new.values() if seed_score is not None and v > seed_score]
        if seed_score is not None
        else []
    )
    return {
        "task": task_name,
        "checkpoint_label": checkpoint.label,
        "checkpoint_step_index": checkpoint.step_index,
        "checkpoint_fraction": round(checkpoint.fraction, 4),
        "remaining_teacher_steps": checkpoint.remaining_steps,
        "seed_smiles": checkpoint.smiles,
        "seed_heavy_atoms": checkpoint.heavy_atoms,
        "seed_score": seed_score,
        "charged_calls": len(rows),
        "budget": budget,
        "distinct_new_molecules": len(new),
        "best_overall": max(scores.values()) if scores else None,
        "best_new": ranked[0] if ranked else None,
        "top_ten_new_mean": (sum(top_ten_new) / len(top_ten_new)) if top_ten_new else None,
        "top_ten_new_count": len(top_ten_new),
        "distinct_new_above_seed": len(above_seed),
        "lift_best_new_minus_seed": (
            ranked[0] - seed_score if ranked and seed_score is not None else None
        ),
        "seconds": round(elapsed, 2),
        "oracle_ledger": oracle.ledger(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--queries-per-round", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output", default="diagnostics/pmo_atlas_v1/test_b_local_lift.json")
    parser.add_argument("--reference-panel", default="diagnostics/pmo_atlas_v1/reference_panel.json")
    args = parser.parse_args(argv)

    assert_not_scored_consumable("atlas_local_lift_diagnostic")
    repo_root = Path(args.repo_root).resolve()
    work = Path(args.work_dir) if args.work_dir else repo_root / ".pmo_atlas_runs"
    work.mkdir(parents=True, exist_ok=True)

    dossier = load_atlas(repo_root)
    spines = {route.task: route for route in dossier.spines()}
    tasks = args.tasks or sorted(spines)

    panel_document = json.loads((repo_root / args.reference_panel).read_text())
    panel_payload = panel_document.get("payload", panel_document)

    started = time.time()
    runs: list[dict[str, Any]] = []
    controls: dict[str, Any] = {}
    def _write(partial: bool) -> dict[str, Any]:
        payload = _payload(partial)
        document = {"payload": payload, "payload_sha256": payload_sha256(payload)}
        output = repo_root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=1, sort_keys=True))
        return payload

    def _payload(partial: bool) -> dict[str, Any]:
        return {
            "schema_version": "pmo_atlas_test_b_local_lift_v1",
            "partial": partial,
            "information_regime": DEVELOPMENT_INFORMED_LABEL,
            "information_regime_statement": REGIME_STATEMENT,
            "question": (
                "Started in a teacher region with a fixed small budget, does the frozen "
                "local controller fill a strong top ten, or only preserve one good seed?"
        ),
            "controller": {
                "class": "PmoPopulationController",
                "enable_online_memory": True,
                "arm": "B_memory",
                "jump_checkpoint": CHECKPOINTS,
                "jump_checkpoint_sha256": file_sha256(repo_root / CHECKPOINTS),
                "destination_or_teacher_suffix_supplied_after_initialization": False,
            },
            "budget_per_run": args.budget,
            "queries_per_round": args.queries_per_round,
            "seed": args.seed,
            "probe_positions": [
                {"label": label, "fraction": fraction} for label, fraction in PROBE_POSITIONS
            ],
            "tasks_probed": tasks,
            "diagnostic_oracle_calls": sum(r["charged_calls"] for r in runs),
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
            "runs": list(runs),
    }


    for task_name in tasks:
        route = spines[task_name]
        oracle = build_oracle(task_name)
        panel = tuple(
            ReferenceExpectation(item["smiles"], item["value"], item["label"])
            for item in panel_payload["panels"][task_name]
        )
        # The positive control runs BEFORE any diagnostic call, so a silently
        # failing oracle cannot manufacture a plausible ledger.
        controls[task_name] = assert_reference_panel(oracle, panel)

        for checkpoint in route_checkpoints(route, positions=PROBE_POSITIONS):
            folder = work / f"{task_name}__{checkpoint.label}"
            record = _run_one(
                repo_root,
                folder,
                task_name,
                checkpoint,
                oracle,
                args.budget,
                args.queries_per_round,
                args.seed,
            )
            runs.append(record)
            _write(partial=True)
            print(
                f"{task_name:24s} {checkpoint.label:12s} seed={record['seed_score']} "
                f"best_new={record['best_new']} top10new={record['top_ten_new_mean']} "
                f"above_seed={record['distinct_new_above_seed']}/"
                f"{record['distinct_new_molecules']} calls={record['charged_calls']} "
                f"{record['seconds']:.0f}s",
                flush=True,
            )

    payload = _write(partial=False)
    print(f"\ndiagnostic oracle calls: {payload['diagnostic_oracle_calls']}")
    print(f"wrote {repo_root / args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
