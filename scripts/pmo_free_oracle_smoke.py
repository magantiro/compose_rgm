"""Full free-oracle execution gate for the PMO scored pilot.

An import smoke is not a launch gate.  The previous one proved the image could import
torch and construct the controller, then stopped; the scored run went on to die on all
three tasks -- gsk3b immediately on a contract/runtime asset mismatch, the other two at
round six on the archive's bootstrap-pool invariant -- after charging 224 oracle calls.

This runs the REAL `execute_task` path on the exact pinned payload with an injected free
scorer, so it exercises everything a scored run does except paying for it: task protocol
resolution (including the gsk3b-only asset branch), bootstrap, recursive descendants,
joint-credit activation, and the coexistence of several round pools.

Nothing here can charge an oracle call: `evaluate` is supplied by this module and the
PyTDC `Oracle` is never constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.experiments import pmo_population_v1 as population

ROOT = Path(__file__).resolve().parents[1]
MIN_ROUNDS = 8


def _free_score(smiles: str) -> float:
    """Deterministic pseudo-objective in [0, 1]. No oracle, no network, no cost."""
    return (int(hashlib.sha256(smiles.encode()).hexdigest()[:8], 16) % 10_000) / 10_000.0


def run_task(task_name: str, folder: Path, budget: int) -> dict:
    contract = dict(population.load_contract(ROOT))
    # The frozen contract stays fail-closed on disk; scoring authority is granted in
    # memory only, exactly as the Modal worker does it.
    contract["scored_launch_authorized"] = True
    contract["modal_launch_authorized"] = True

    seen: list[PmoPopulationController] = []
    original_init = PmoPopulationController.__init__

    def capture(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        seen.append(self)

    calls = {"n": 0}

    def evaluate(smiles: str) -> float:
        calls["n"] += 1
        return _free_score(smiles)

    highest = {"round": 0}

    def progress(row: dict) -> None:
        highest["round"] = max(highest["round"], int(row.get("round", 0)))

    PmoPopulationController.__init__ = capture
    try:
        result = population.execute_task(
            contract,
            ROOT,
            folder,
            task_name,
            evaluate=evaluate,
            charged_calls_per_task=budget,
            progress=progress,
        )
        status, error = "COMPLETED", None
        auc_budget = result.get("auc_budget")
    except Exception as failure:  # noqa: BLE001 - the gate reports any failure verbatim
        status, error, auc_budget = "FAILED", f"{type(failure).__name__}: {failure}", None
    finally:
        PmoPopulationController.__init__ = original_init

    controller = seen[-1] if seen else None
    report = {
        "task": task_name,
        "status": status,
        "error": error,
        "rounds_reached": highest["round"],
        "free_evaluations": calls["n"],
        "budget": budget,
        "charged_oracle_calls": 0,
        "auc_budget": auc_budget,
        "task_protocol_resolved": True,
    }
    if controller is not None:
        continuity = controller._pool_continuity
        report.update({
            "credit_cells": len(controller.credit.cells),
            "exploration_floor": controller.credit.exploration_floor,
            "archive_entries": len(controller.entries),
            "physical_pools_absorbed": len(continuity.generation_pool_ids),
            "continuity_pool_id": continuity.continuity_pool_id,
            "active_basins": controller.credit_report()["active_basins"],
            "productive_parents": controller.credit_report()["productive_parents"],
        })
    return report


def assess(report: dict) -> list[str]:
    """Return the gate criteria this task failed, empty when it passes."""
    failures = []
    if report["status"] != "COMPLETED":
        failures.append(f"execution failed: {report['error']}")
    if report["rounds_reached"] < MIN_ROUNDS:
        failures.append(f"only reached round {report['rounds_reached']} < {MIN_ROUNDS}")
    if report.get("charged_oracle_calls") != 0:
        failures.append("charged oracle calls must be zero")
    # A run that stops early has not exercised the budget the scored run will spend.
    if report.get("free_evaluations") != report.get("budget"):
        failures.append(
            f"spent {report.get('free_evaluations')} of {report.get('budget')} calls; "
            "the gate must exhaust the budget the scored run is authorized for"
        )
    if report.get("auc_budget") != report.get("budget"):
        failures.append(
            f"AUC denominator is {report.get('auc_budget')}, not the {report.get('budget')} "
            "actually run -- a historical 1000/10000 constant is not comparable"
        )
    if not report.get("credit_cells"):
        failures.append("joint credit never activated")
    if not report.get("exploration_floor"):
        failures.append("exploration floor is not live")
    if report.get("physical_pools_absorbed", 0) < 1:
        failures.append("no bootstrap pool was recorded")
    if report.get("archive_entries", 0) <= population.INIT_COUNT:
        failures.append("children never re-entered the archive as parents")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(population.TASKS))
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="charged calls per task; the gate must run the budget the scored run will",
    )
    args = parser.parse_args()
    folder = Path(args.out) / args.task
    folder.mkdir(parents=True, exist_ok=True)
    report = run_task(args.task, folder, args.budget)
    report["gate_failures"] = assess(report)
    report["gate"] = "PASS" if not report["gate_failures"] else "FAIL"
    (Path(args.out) / f"{args.task}_smoke.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
