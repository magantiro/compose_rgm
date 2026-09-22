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
from compose_v4.control.pmo_donor_channel import DEFAULT_CUT_LAW
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

SCHEMA = "pmo_population_controller_v1"  # cosmetic: behaviour unchanged
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
        parent_allocation="niche_evidence",
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



def resolve_extension_prior(folder, *, result_name="result.json", started_name="started.json",
                            failure_name="failure.json"):
    """Locate the COMPLETED prior run an extension resumes, or refuse with the reason.

    An extension archives the prior result and removes ``result.json`` BEFORE the run
    itself starts, so an extension that fails partway leaves neither a result nor a
    clean slate.  Requiring ``result.json`` would make the extension un-resumable and
    make its own half-finished state read as a forbidden retry of the original run.

    Returns ``(prior_path, prior_budget)``.  Raises with the specific reason otherwise.
    """
    folder = Path(folder)
    result_path, started = folder / result_name, folder / started_name
    failure_path = folder / failure_name
    archives = sorted(
        folder.glob("result_at_*_calls.json"),
        key=lambda path: int(path.name.split("_")[2]),
    )
    if result_path.exists():
        prior_path = result_path
    elif archives:
        # The archive is strictly STRONGER evidence than ``result.json``: it proves the
        # prior run completed AND that this extension has run before.  It is only safe
        # to resume once that attempt has TERMINATED -- a container still running left
        # ``started.json`` and no ``failure.json``, and a second container against the
        # same durable state could double-charge.
        if started.exists() and not failure_path.exists():
            raise RuntimeError(
                "a prior extension attempt is still in flight (started, no failure); "
                "refusing a second container against the same durable state"
            )
        prior_path = archives[-1]
    else:
        raise RuntimeError(
            "extension requires a COMPLETED prior run; a mid-flight or absent run "
            "would be a retry, which stays forbidden"
        )
    prior = json.loads(prior_path.read_text())
    prior_budget = int(prior.get("charged_calls") or prior.get("auc_budget") or 0)
    if prior_path != result_path:
        # Two independent paths to one number: the name the archive was written under,
        # and the charged-call count recorded inside it.
        named_budget = int(prior_path.name.split("_")[2])
        if named_budget != prior_budget:
            raise RuntimeError(
                f"archive name says {named_budget} charged calls, its content says "
                f"{prior_budget}; refusing to extend from an ambiguous prior run"
            )
    return prior_path, prior_budget

def execute_task(
    contract: dict,
    root: Path,
    folder: Path,
    task_name: str,
    *,
    evaluate: Callable[[str], float],
    charged_calls_per_task: int,
    progress=None,
    enable_online_memory: bool = False,
    enable_discovery: bool = False,
    enable_donor_channel: bool = False,
    donor_cut_law: str = DEFAULT_CUT_LAW,
) -> dict:
    """Run one PMO-v1 task after a separately authorized scored launch.

    ``charged_calls_per_task`` is REQUIRED and has no default on purpose.  The ledger
    budget used to come from the module constant ``QUERY_BUDGET`` while the authorizing
    contract declared its own, smaller, figure, so the contract's budget was decorative:
    a run authorized for 250 calls per task would have charged up to 1000.  A required
    argument makes it impossible for a caller to inherit a budget it never stated.
    """
    if not isinstance(charged_calls_per_task, int) or charged_calls_per_task <= 0:
        raise ValueError("charged_calls_per_task must be a positive integer")

    if enable_donor_channel and not enable_online_memory:
        # Refused HERE, before the ledger is constructed and before any charged call,
        # rather than deeper in the controller. The controller raises for the same
        # reason, but by then a misconfigured launch has already built its ledger; the
        # discipline this repository paid for is that a positive check runs before the
        # first charged call, so a failure leaves the task re-runnable.
        raise ValueError(
            "arm D requires the online memory: the donor lane recombines the run's own "
            "stratified scored bank, which the memory holds"
        )
    if task_name not in TASKS:
        raise ValueError("task is outside the PMO-v1 three-task lock")
    if not contract.get("scored_launch_authorized"):
        raise ValueError("PMO-v1 scored launch is not authorized by this contract")
    initialized = _load_initialization(root, {"initialization": contract["initialization"]})
    task = ProgramTask(task_name, identity(_runtime_protocol(contract, task_name)), "pmo")
    ledger = ProgramQueryLedger(
        folder / "oracle", task, evaluate, budget=charged_calls_per_task
    )
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
        optimizer_kwargs={
            "jump_checkpoint": checkpoint,
            # Arm selector for the matched comparison. It rides in optimizer_kwargs, so
            # run_program_campaign folds it into `optimizer_kwargs_sha256` and the two
            # arms cannot share a run identity even if every other input matches.
            "enable_online_memory": bool(enable_online_memory),
            # Arm C = arm B + the discovery allocator. It rides here for the same
            # reason the memory flag does: `run_program_campaign` folds
            # optimizer_kwargs into `optimizer_kwargs_sha256`, so B and C cannot
            # share a run identity even when every other input matches.
            "enable_discovery": bool(enable_discovery),
            # Arm D = donor recombination over molecules THIS RUN has scored. The two
            # donor keys are present ONLY when the lane is on, and that is deliberate
            # rather than tidy: `run_program_campaign` hashes this dict into
            # `optimizer_kwargs_sha256`, writes it into the campaign manifest, and
            # REFUSES a resume whose recipe moved ("campaign recipe changed during
            # resume"). Adding a key unconditionally would therefore break every
            # in-flight arm A/B/C run at its next preemption retry, while changing
            # nothing about what they compute. Absent is the only byte-identical off,
            # here exactly as it is at the channel set.
            **(
                {
                    "enable_donor_channel": True,
                    "donor_cut_law": str(donor_cut_law),
                }
                if enable_donor_channel
                else {}
            ),
        },
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
        # The denominator is the budget this run was authorized for, not a fixed 1000:
        # an AUC over a 1000-call denominator computed from a 250-call run understates it,
        # and neither is comparable to a published 10000-call figure.
        "auc_top10_at_budget": pmo_top_ten_auc(
            values, budget=charged_calls_per_task, finish=True
        ),
        "auc_budget": charged_calls_per_task,
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
