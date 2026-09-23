"""ZERO-ORACLE proof that the pool actually widened in the real PMO production path.

A previous attempt raised `candidates_per_batch` and changed nothing: PMO overrides the base
`propose_batch`, and the true cap is the module constant `CHANNEL_CANDIDATE_LIMIT = 16` applied
PER CHANNEL inside `dynamic_program_synthesis_v21._generate_channel_pool`.  The run completed,
produced plausible numbers, and tested nothing -- caught only because two arms came back
byte-identical.

So this gate runs BEFORE any oracle budget is committed and refuses to pass on an argument.  The
scorer is a stub: no PMO oracle is called and nothing is attributable to a task budget.

PASS REQUIRES, measured from the controller's own telemetry:
    median eligible pool  >= 100
    query fraction        <= 0.16

The constant is overridden at RUNTIME, never edited: the module is hash-pinned by three contracts,
and patching a module global is only valid if the reference resolves at call time -- which this
gate verifies by observation rather than by reading the source.
"""
from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path

import numpy as np

OUT = "diagnostics/pmo_pool_width_v1/pool_width_dryrun_v1.json"
POOL_TARGET = 128
QUERIES = 16


def main():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.control import dynamic_program_synthesis_v21 as v21
    from compose_v4.control.docking_value import identity
    from compose_v4.control.pmo_population_controller import PmoPopulationController
    from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
    from compose_v4.control.program_task import ProgramTask
    from compose_v4.experiments import pmo_population_v1 as production
    from compose_v4.experiments.pmo_dynamic_v21 import initial_dynamic_program_batch_v21

    before = v21.CHANNEL_CANDIDATE_LIMIT
    v21.CHANNEL_CANDIDATE_LIMIT = POOL_TARGET
    print(f"CHANNEL_CANDIDATE_LIMIT {before} -> {v21.CHANNEL_CANDIDATE_LIMIT}")

    root = Path(".")
    contract = production.load_contract(root)
    initialized = production._load_initialization(
        root, {"initialization": contract["initialization"]}
    )
    checkpoint = production._load_checkpoint(root, contract)
    config = production.configuration(contract["controller"]["seed"])
    task = ProgramTask(
        "celecoxib_rediscovery",
        identity(production._runtime_protocol(contract, "celecoxib_rediscovery")),
        "pmo",
    )
    folder = pathlib.Path("diagnostics/pmo_pool_width_v1/dryrun")
    folder.mkdir(parents=True, exist_ok=True)

    # A STUB scorer. No PMO oracle is constructed or called, so nothing here is chargeable.
    calls = {"n": 0}

    def stub(_smiles: str) -> float:
        calls["n"] += 1
        return 0.1

    telemetry = []

    class Observed(PmoPopulationController):
        def _allocate(self, candidates):
            room = min(QUERIES, len(candidates))
            selected, detail = super()._allocate(candidates)
            selected = selected[:room]
            telemetry.append(
                {
                    "batch": self.batches,
                    "pool": len(candidates),
                    "selected": len(selected),
                    "query_fraction": len(selected) / max(len(candidates), 1),
                }
            )
            return selected, detail

    ledger = ProgramQueryLedger(folder / "oracle", task, stub, budget=96)
    run_program_campaign(
        output=folder / "campaign", task=task, config=config,
        initialization=initialized, library=(), ledger=ledger,
        rounds=5, queries_per_round=QUERIES, hierarchy=None, fit_model=None,
        stagnation_rounds=None, bootstrap_rounds=1,
        initialization_mode="all_scored_pool", initial_parent_fraction=0.2,
        progress=lambda row: None,
        optimizer_type=Observed, optimizer_kwargs={"jump_checkpoint": checkpoint},
        initial_batch_fn=initial_dynamic_program_batch_v21,
    )
    v21.CHANNEL_CANDIDATE_LIMIT = before

    pools = [t["pool"] for t in telemetry]
    fractions = [t["query_fraction"] for t in telemetry]
    report = {
        "schema_version": "pmo_pool_width_dryrun_v1",
        "evidence_role": "zero_oracle_configuration_gate",
        "new_charged_oracle_calls": 0,
        "stub_scorer_calls": calls["n"],
        "constant_overridden": {"from": before, "to": POOL_TARGET},
        "rounds_observed": len(telemetry),
        "median_pool": float(np.median(pools)) if pools else 0.0,
        "min_pool": min(pools) if pools else 0,
        "max_pool": max(pools) if pools else 0,
        "median_query_fraction": float(np.median(fractions)) if fractions else 1.0,
        "max_query_fraction": max(fractions) if fractions else 1.0,
        "telemetry": telemetry,
    }
    report["pass"] = bool(
        pools and report["median_pool"] >= 100 and report["max_query_fraction"] <= 0.16
    )
    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print(
        f"rounds {len(telemetry)} | pool median {report['median_pool']:.0f} "
        f"(min {report['min_pool']} max {report['max_pool']}) | "
        f"query fraction median {report['median_query_fraction']:.3f} "
        f"max {report['max_query_fraction']:.3f}"
    )
    print(f"VERDICT: {'PASS' if report['pass'] else 'FAIL'}  (required pool>=100, fraction<=0.16)")
    print("WROTE", OUT)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
