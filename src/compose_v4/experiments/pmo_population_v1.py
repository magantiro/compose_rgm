"""PMO-v1 population-controller contract and scored-run adapter.

This module keeps PMO's population policy separate from T4's frozen controller while
sharing the COMPOSE program geometry, exact executor, FiberControl allocation, and
recursive archive.  It intentionally does not launch an oracle.  The command-line
tool must verify a separately sealed scored contract before calling ``execute_task``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v21 import (
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask, pmo_top_ten_auc
from compose_v4.experiments.continuation_profile import verify_file
from compose_v4.experiments.pmo_dynamic_v21 import (
    INITIALIZATION,
    ORACLE_ENVIRONMENT,
    ORACLE_SOURCE_SHA256,
    _load_initialization,
    _runtime_protocol,
    _score_curve,
)

SCHEMA = "pmo_population_controller_v1"
CONTRACT = "configs/pmo_population_controller_v1.json"
CHECKPOINTS = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"
INIT_COUNT = 16
QUERY_BUDGET = 1000
QUERIES_PER_ROUND = 16
MAX_ROUNDS = 64
TASKS = ("gsk3b", "perindopril_mpo", "celecoxib_rediscovery")
TASK_ROLES = {
    "gsk3b": "highest-priority basin-discovery black-box test",
    "perindopril_mpo": "jump then multi-generation descendant-refinement stress",
    "celecoxib_rediscovery": "long-program reachability and debugging control",
}
IVG_TARGETS = {"celecoxib_rediscovery": 0.798, "gsk3b": 0.952, "perindopril_mpo": 0.645}


def configuration(seed: int = 20260920) -> ProgramSearchConfig:
    """Shared PMO-v1 search geometry; proposal policy is population-level."""

    return replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        parent_allocation="niche_score",
        attempts_per_batch=128,
        candidates_per_batch=QUERIES_PER_ROUND,
        wall_seconds=45.0,
        proposal_cache_entries=128,
    )


def controller_payload(seed: int = 20260920) -> dict:
    return {
        "schema_version": SCHEMA,
        "tasks": list(TASKS),
        "task_roles": TASK_ROLES,
        "controller": asdict(configuration(seed)),
        "proposal_channels": [
            "shallow_program_channel",
            "structured_program_channel",
            "joint_dependency_region_jump",
        ],
        "modes": ["global_explore", "jump_from_elite", "refine_elite"],
        "continuous_retention": {"min": 0.0, "max": 1.0, "runtime_task_conditioning": False},
        "recursive_archive": True,
        "archive_policy": "structural_niche_plus_observed_upside",
        "early_quotas": {
            "shallow_program_channel": 2,
            "structured_program_channel": 2,
            "joint_dependency_region_jump": 2,
        },
        "plateau_escape": {"after_rounds": 3, "escape_rounds": 2},
        "initialization": {"path": INITIALIZATION, "count": INIT_COUNT, "all_scores_charged": True},
        "joint_checkpoint": {"path": CHECKPOINTS, "fit_scope": "shared_all_routes"},
        "oracle": {
            "environment": ORACLE_ENVIRONMENT,
            "source_files": ORACLE_SOURCE_SHA256,
            "direction": "maximize",
            "prescreen": False,
            "calls_include_initialization": True,
        },
        "development_budget": {
            "calls_per_task": QUERY_BUDGET,
            "calls_total": QUERY_BUDGET * len(TASKS),
        },
        "ivg_targets_postrun_only": IVG_TARGETS,
        "scored_launch_authorized": False,
        "modal_launch_authorized": False,
    }


def _load_checkpoint(root: Path, contract: dict) -> dict:
    path = root / contract["joint_checkpoint"]["path"]
    verify_file(path, contract["joint_checkpoint"]["sha256"])
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]["checkpoints"]["shared_all_routes"]
    if payload.get("fit_scope") != "shared_all_routes":
        raise ValueError("PMO runtime checkpoint must be shared and task blind")
    if identity(payload) != contract["joint_checkpoint"]["payload_sha256"]:
        raise ValueError("PMO joint checkpoint payload changed")
    return payload


def load_contract(root: Path) -> dict:
    envelope = json.loads((root / CONTRACT).read_text())
    contract = envelope["payload"]
    if envelope.get("payload_sha256") != identity(contract):
        raise ValueError("PMO-v1 contract envelope hash changed")
    if contract.get("schema_version") != SCHEMA:
        raise ValueError("unexpected PMO-v1 contract schema")
    if contract.get("scored_launch_authorized") is not False:
        raise ValueError("PMO-v1 preflight contract cannot authorize scoring")
    init = _load_initialization(root, {"initialization": contract["initialization"]})
    if init.get("count") != INIT_COUNT:
        raise ValueError("PMO-v1 initialization count changed")
    _load_checkpoint(root, contract)
    for path, digest in contract["implementation_sha256"].items():
        verify_file(root / path, digest)
    return contract


def execute_task(
    contract: dict,
    root: Path,
    folder: Path,
    task_name: str,
    *,
    evaluate: Callable[[str], float],
    progress=None,
) -> dict:
    """Run one PMO-v1 task after a separately authorized scored launch."""

    if task_name not in TASKS:
        raise ValueError("task is outside the PMO-v1 three-task lock")
    if not contract.get("scored_launch_authorized"):
        raise ValueError("PMO-v1 scored launch is not authorized by this contract")
    initialized = _load_initialization(root, {"initialization": contract["initialization"]})
    task = ProgramTask(task_name, identity(_runtime_protocol(contract, task_name)), "pmo")
    ledger = ProgramQueryLedger(folder / "oracle", task, evaluate, budget=QUERY_BUDGET)
    checkpoint = _load_checkpoint(root, contract)
    config = configuration(contract["controller"]["seed"])
    report = progress or (lambda row: None)
    campaign = run_program_campaign(
        output=folder / "campaign",
        task=task,
        config=config,
        initialization=initialized,
        library=(),
        ledger=ledger,
        rounds=MAX_ROUNDS,
        queries_per_round=QUERIES_PER_ROUND,
        hierarchy=None,
        fit_model=None,
        stagnation_rounds=None,
        bootstrap_rounds=1,
        initialization_mode="all_scored_pool",
        initial_parent_fraction=0.2,
        progress=report,
        optimizer_type=PmoPopulationController,
        optimizer_kwargs={"jump_checkpoint": checkpoint},
        initial_batch_fn=initial_dynamic_program_batch_v21,
    )
    curve = _score_curve(ledger.rows)
    values = [row["score"] for row in ledger.rows]
    return {
        "schema_version": "pmo_population_task_result_v1",
        "task": task_name,
        "role": TASK_ROLES[task_name],
        "charged_oracle_calls": len(ledger.rows),
        "best_score": curve[-1]["best_score"] if curve else None,
        "score_curve": curve,
        "auc_top10_development_1000": pmo_top_ten_auc(values, budget=QUERY_BUDGET, finish=True),
        "campaign": campaign,
    }


__all__ = [
    "CHECKPOINTS",
    "CONTRACT",
    "IVG_TARGETS",
    "QUERY_BUDGET",
    "TASKS",
    "configuration",
    "controller_payload",
    "execute_task",
    "load_contract",
]
