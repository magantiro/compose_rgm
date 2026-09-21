"""Capture real COMPOSE candidate pools and the production allocation, free.

Runs the REAL `pmo_population_v1.execute_task` path with an injected free
scorer that is an EXACT re-derivation of the benchmark objective (verified to
0.0 absolute delta against 250 charged rows), so every candidate the controller
proposes can be scored offline at no cost.

The point is the per-round artifact `campaign/round_XXXX/pending.json`, which
records BOTH the full eligible pool the controller had to choose from AND the
subset its production `_allocate` actually charged.  That pair is exactly the
matched-budget counterfactual an allocation study needs: hold the pool fixed,
vary only the choice of which members to buy.

CHARGES NOTHING.  `evaluate` is supplied here and the PyTDC Oracle is never
constructed, which is the same structural guarantee `scripts/pmo_free_oracle_smoke.py`
relies on.  The emitted artifact records `charged_oracle_calls: 0`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from compose_v4.experiments import pmo_population_v1 as population
from compose_v4.experiments.pmo_free_objectives import (
    FREE_ORACLE_PROTOCOL,
    build_free_scorer,
)

ROOT = Path(__file__).resolve().parents[1]


def capture(task: str, folder: Path, budget: int, wall_seconds: float | None) -> dict:
    scorer = build_free_scorer(task)
    contract = dict(population.load_contract(ROOT))
    # In memory only; the frozen contract on disk stays fail-closed.
    contract["scored_launch_authorized"] = True
    contract["modal_launch_authorized"] = True

    calls = {"n": 0}

    def evaluate(smiles: str) -> float:
        calls["n"] += 1
        return float(scorer(smiles))

    highest = {"round": 0}

    def progress(row: dict) -> None:
        highest["round"] = max(highest["round"], int(row.get("round", 0)))

    # ProgramSearchConfig is a frozen dataclass, so shorten the PROPOSAL wall
    # budget by replacing the factory rather than mutating the instance.  This
    # changes how many candidates each round YIELDS, never how the allocator
    # chooses among the ones it has -- which is the quantity under study.
    original_configuration = population.configuration
    if wall_seconds is not None:
        def patched_configuration(*args, **kwargs):
            return dataclasses.replace(
                original_configuration(*args, **kwargs), wall_seconds=wall_seconds
            )
        population.configuration = patched_configuration

    # Record the FULL pool `_allocate` sees.  The production method still makes
    # every decision -- this wraps it and delegates, it does not reimplement it,
    # so the captured selection is the real one by construction.
    from compose_v4.control.pmo_population_controller import PmoPopulationController

    pools: list[dict] = []
    original_allocate = PmoPopulationController._allocate

    def recording_allocate(self, merged):
        chosen, receipt = original_allocate(self, merged)
        pools.append({
            "batch_index": int(getattr(self, "batches", len(pools))),
            "pool": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "endpoint": row.get("endpoint"),
                    "channel": (row.get("provenance") or {}).get("planner_channel"),
                    "parent_measured_score": (row.get("provenance") or {}).get(
                        "parent_measured_score"
                    ),
                    "basin_id": (row.get("provenance") or {}).get("basin_id"),
                }
                for row in merged
            ],
            "selected_candidate_ids": [row.get("candidate_id") for row in chosen],
            "allocation_role": receipt.get("allocation_role"),
            "credit_detail_mode": (receipt.get("credit_detail") or {}).get("mode"),
        })
        return chosen, receipt

    PmoPopulationController._allocate = recording_allocate

    started = time.time()
    try:
        result = population.execute_task(
            contract, ROOT, folder, task,
            evaluate=evaluate, charged_calls_per_task=budget, progress=progress,
        )
        status, error = "COMPLETED", None
    except Exception as failure:  # noqa: BLE001 - report any failure verbatim
        result, status = {}, "FAILED"
        error = f"{type(failure).__name__}: {failure}"
    finally:
        population.configuration = original_configuration
        PmoPopulationController._allocate = original_allocate

    return {
        "schema_version": "pmo_surrogate_pool_capture_v1",
        "pools": pools,
        "pool_sizes": [len(p["pool"]) for p in pools],
        "task": task,
        "status": status,
        "error": error,
        "budget": budget,
        "free_evaluations": calls["n"],
        "charged_oracle_calls": 0,
        "oracle_protocol": FREE_ORACLE_PROTOCOL,
        "rounds_reached": highest["round"],
        "proposal_wall_seconds": wall_seconds,
        "elapsed_seconds": round(time.time() - started, 2),
        "result_best": result.get("best"),
        "result_auc": result.get("auc"),
        "auc_budget": result.get("auc_budget"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="celecoxib_rediscovery")
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--wall-seconds", type=float, default=None)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    folder = Path(args.run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    report = capture(args.task, folder, args.budget, args.wall_seconds)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "pools"}, indent=1))


if __name__ == "__main__":
    main()
