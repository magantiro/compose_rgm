"""Four-task PMO adapter for the pooled Dynamic COMPOSE v2.1 controller.

The task runtime sees a task-independent initialization lock, an empty route
library and its own charged observations.  Prior PMO outcomes and IVG values
are consumed only by :func:`aggregate_offline` after scored tasks finish.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter, process_time

from rdkit import rdBase

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import (
    DynamicProgramOptimizer,
    initial_dynamic_program_batch,
)
from compose_v4.control.dynamic_program_synthesis_v2 import (
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
)
from compose_v4.control.dynamic_program_synthesis_v21 import (
    CHANNEL_CANDIDATE_LIMIT,
    EXPLORATION_FLOOR,
    UCB_EXPLORATION,
    DynamicV21ProgramOptimizer,
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask, pmo_top_ten_auc
from compose_v4.experiments.continuation_profile import (
    publish_json,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.pmo_ivg_oracle_parity import (
    verify_environment,
    verify_pytdc_sources,
)
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal

KIND = "pmo_dynamic_v21"
CONTRACT = "configs/pmo_dynamic_v21_development_v1.json"
OFFLINE_COMPARATORS = "configs/pmo_dynamic_v21_offline_comparators_v1.json"
PREFLIGHT = "diagnostics/pmo_dynamic_v21/preflight.json"
LAUNCH = "diagnostics/pmo_dynamic_v21/launch.json"
RESULT = "diagnostics/pmo_dynamic_v21/result.json"
INITIALIZATION = "diagnostics/parent_edit_cycles/prepared/init_20260921.json"
EMPTY_LIBRARY = "diagnostics/pmo_dynamic_v21/empty_library.json"
PARITY_RESULT = "diagnostics/pmo_ivg_oracle_parity/result.json"
ORACLE_ASSET_ROOT = "diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets"

TASKS = (
    "celecoxib_rediscovery",
    "perindopril_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
)
ARMS = ("dynamic_v0", "dynamic_v21")
SEARCH_SEED = 20260914
TASK_SEEDS = {task: SEARCH_SEED for task in TASKS}
QUERY_BUDGET = 1000
TOTAL_QUERY_CEILING = 8000
INITIALIZATION_COUNT = 16
QUERIES_PER_ROUND = 16
MAX_ROUNDS = 64
MAX_WORKERS = 4
OFFICIAL_METRIC_BUDGET = 10000
OFFICIAL_METRIC_FREQUENCY = 100

ORACLE_ENVIRONMENT = {
    "pytdc": "1.1.15",
    "rdkit": "2023.09.6",
    "numpy": "1.26.4",
    "scikit_learn": "1.2.2",
    "pandas": "2.1.4",
    "scipy": "1.15.0",
    "seaborn": "0.13.2",
    "requests": "2.32.4",
    "setuptools": "75.6.0",
}
ORACLE_SOURCE_SHA256 = {
    "pytdc_metadata": "0925dc34498dd5502429c6d278be70758f1d536417af9cc95392c28e014fe6e6",
    "tdc_oracles_py": "d2b2194e3573eb620759998e83d8803eacfd5cbd00531e862ebe3283c301cabe",
    "tdc_oracle_py": "211be401ec5f3c06ddca37e72064bbeb4de52437b218bb3f4a4a381eec1de1f5",
}
GSK3B_ASSET = {
    "path": f"{ORACLE_ASSET_ROOT}/oracle/gsk3b_current.pkl",
    "sha256": "d3a20701b80e5179c88c3ad4dc3483dd7ab35c50dc055c6773a7f5b63e89b6d5",
    "bytes": 27791877,
    "pytdc_dataverse_file_id": 6413412,
}


def configuration(seed: int = SEARCH_SEED) -> ProgramSearchConfig:
    """The unchanged v2.1 program-only recipe with PMO score orientation."""
    return replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        proposal_cache_entries=128,
        candidates_per_batch=QUERIES_PER_ROUND,
        attempts_per_batch=128,
        wall_seconds=45.0,
    )


def policy_payload() -> dict:
    return {
        "schema_version": "pmo_dynamic_v21_matched_development_v1",
        "arms": list(ARMS),
        "tasks": list(TASKS),
        "task_seeds": TASK_SEEDS,
        "charged_queries_per_task": QUERY_BUDGET,
        "total_charged_query_ceiling": TOTAL_QUERY_CEILING,
        "charged_units": len(ARMS) * len(TASKS),
        "initialization_queries_per_task": INITIALIZATION_COUNT,
        "queries_per_round": QUERIES_PER_ROUND,
        "max_rounds": MAX_ROUNDS,
        "max_concurrent_workers": MAX_WORKERS,
        "submission_order": "task_major_paired_arms",
        "cpu_per_worker": 1,
        "gpu": False,
        "automatic_retries": 0,
        "initial_route_archive": [],
        "initial_route_archive_by_arm": {arm: [] for arm in ARMS},
        "task_informed_program_curriculum": False,
        "source_library_rows_loaded": 0,
        "score_direction": "maximize",
        "validity": "RDKit molecule validity after exact executor support; no T4 endpoint gates",
        "planner_channels": [SHALLOW_CHANNEL, STRUCTURED_CHANNEL],
        "arm_proposal_paths": {
            "dynamic_v0": [SHALLOW_CHANNEL],
            "dynamic_v21": [SHALLOW_CHANNEL, STRUCTURED_CHANNEL],
        },
        "channel_candidate_limit": CHANNEL_CANDIDATE_LIMIT,
        "independent_rng_streams": ["shallow", "structured", "arbitration"],
        "post_filter_pooled_arbitration": True,
        "common_run_local_route_archive": True,
        "ucb_exploration": UCB_EXPLORATION,
        "eligible_exploration_floor": EXPLORATION_FLOOR,
        "official_metric": {
            "name": "PMO top-ten trapezoid AUC with flat tail",
            "budget": OFFICIAL_METRIC_BUDGET,
            "frequency": OFFICIAL_METRIC_FREQUENCY,
            "finish": True,
        },
        "development_metric": {
            "name": "PMO top-ten trapezoid AUC at charged development budget",
            "budget": QUERY_BUDGET,
            "frequency": OFFICIAL_METRIC_FREQUENCY,
            "finish": True,
        },
        "comparison_outcomes_available_to_runtime": False,
        "fifth_task_authorized": False,
        "held_out_or_general_pmo_claim": False,
    }


def _runtime_protocol(contract: dict, task: str) -> dict:
    return {
        "task": task,
        "implementation": "native PyTDC Oracle",
        "environment": contract["oracle"]["environment"],
        "source_sha256": contract["oracle"]["source_files"],
        "asset_sha256": (
            {"gsk3b_current.pkl": contract["oracle"]["gsk3b_asset"]["sha256"]}
            if task == "gsk3b"
            else {}
        ),
        "direction": "maximize",
        "range": [0, 1],
        "prescreen": False,
        "calls_include_initialization": True,
    }


def _load_initialization(root: Path, contract: dict) -> dict:
    path = root / contract["initialization"]["path"]
    verify_file(path, contract["initialization"]["sha256"])
    initialized = json.loads(path.read_text())
    body = {key: value for key, value in initialized.items() if key != "lock_sha256"}
    if identity(body) != initialized.get("lock_sha256"):
        raise ValueError("PMO Dynamic-v2.1 initialization lock changed")
    if (
        initialized.get("count") != INITIALIZATION_COUNT
        or len(initialized.get("candidates", ())) != INITIALIZATION_COUNT
        or initialized.get("source_sha256")
        != contract["initialization"]["source_sha256"]
        or initialized.get("accounting")
        != "all initialization scores count against each run's oracle budget"
    ):
        raise ValueError("PMO Dynamic-v2.1 initialization policy changed")
    if any("score" in row or "task" in row for row in initialized["candidates"]):
        raise ValueError("PMO Dynamic-v2.1 initialization contains task information")
    return initialized


def load_contract(root: Path) -> dict:
    contract = unseal(root / CONTRACT)
    if contract.get("schema_version") != "pmo_dynamic_v21_matched_contract_v1":
        raise ValueError("unexpected PMO Dynamic-v2.1 contract schema")
    if contract.get("dynamic_v21_development") != policy_payload():
        raise ValueError("PMO Dynamic-v2.1 development policy changed")
    if contract.get("controller") != json.loads(json.dumps(asdict(configuration()))):
        raise ValueError("PMO Dynamic-v2.1 controller recipe changed")
    if contract.get("initialization", {}).get("path") != INITIALIZATION:
        raise ValueError("PMO Dynamic-v2.1 initialization source changed")
    if (
        contract.get("library", {}).get("path") != EMPTY_LIBRARY
        or contract.get("library", {}).get("programs") != 0
        or json.loads((root / EMPTY_LIBRARY).read_text()) != []
    ):
        raise ValueError("PMO Dynamic-v2.1 must start with an empty route library")
    if contract.get("oracle", {}).get("environment") != ORACLE_ENVIRONMENT:
        raise ValueError("PMO Dynamic-v2.1 oracle environment changed")
    if contract["oracle"].get("source_files") != ORACLE_SOURCE_SHA256:
        raise ValueError("PMO Dynamic-v2.1 PyTDC source identities changed")
    if contract["oracle"].get("gsk3b_asset") != GSK3B_ASSET:
        raise ValueError("PMO Dynamic-v2.1 GSK3B asset changed")
    if set(contract.get("oracle", {}).get("tasks", ())) != set(TASKS):
        raise ValueError("PMO Dynamic-v2.1 native-oracle task set changed")
    forbidden_fragments = (
        "result.json",
        "query_lock",
        "winner",
        "panel",
        "invirtuogen",
        "genmol_pmo_targets",
    )
    if any(
        fragment in path.lower()
        for path in contract.get("inputs", {})
        for fragment in forbidden_fragments
    ):
        raise ValueError(
            "PMO comparator or task-informed artifact entered runtime inputs"
        )
    postrun = contract.get("postrun_only_comparator", {})
    if (
        postrun.get("path") != OFFLINE_COMPARATORS
        or postrun.get("available_to_runtime") is not False
        or postrun["path"] in contract.get("inputs", {})
    ):
        raise ValueError("PMO comparators are not isolated to post-run analysis")
    for path, digest in contract.get("inputs", {}).items():
        verify_file(root / path, digest)
    _load_initialization(root, contract)
    return contract


def verify_runtime_environment(root: Path, contract: dict) -> dict:
    environment = verify_environment(contract)
    sources = verify_pytdc_sources(contract)
    asset = root / contract["oracle"]["gsk3b_asset"]["path"]
    expected = contract["oracle"]["gsk3b_asset"]
    if asset.stat().st_size != expected["bytes"]:
        raise ValueError("PMO Dynamic-v2.1 GSK3B asset byte count changed")
    verify_file(asset, expected["sha256"])
    return {
        "environment": environment,
        "pytdc_source_sha256": sources,
        "gsk3b_asset_sha256": expected["sha256"],
    }


@contextmanager
def _campaign_namespace(arm: str):
    """Inject one unchanged dynamic core into the shared campaign boundary."""
    if arm not in ARMS:
        raise ValueError("PMO campaign arm is outside the matched-arm lock")
    from compose_v4.control import program_campaign

    original = program_campaign.ProgramOptimizer, program_campaign.initial_program_batch
    if arm == "dynamic_v0":
        program_campaign.ProgramOptimizer = DynamicProgramOptimizer
        program_campaign.initial_program_batch = initial_dynamic_program_batch
    else:
        program_campaign.ProgramOptimizer = DynamicV21ProgramOptimizer
        program_campaign.initial_program_batch = initial_dynamic_program_batch_v21
    try:
        yield
    finally:
        program_campaign.ProgramOptimizer, program_campaign.initial_program_batch = (
            original
        )


def _score_curve(rows: list[dict]) -> list[dict]:
    values, curve = [], []
    for index, row in enumerate(rows, start=1):
        if row.get("status") != "complete":
            raise ValueError("PMO score curve cannot impute a failed query")
        value = float(row["score"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("PMO score curve received an invalid reward")
        values.append(value)
        top = sorted(values, reverse=True)[:10]
        curve.append(
            {
                "charged_queries": index,
                "best_score": max(values),
                "top10_score": sum(top) / len(top),
            }
        )
    return curve


def execute_task(
    contract: dict,
    root: Path,
    folder: Path,
    arm: str,
    task_name: str,
    *,
    evaluate,
    progress=None,
) -> dict:
    """Run one bounded task using a caller-supplied, separately qualified oracle."""
    if arm not in ARMS:
        raise ValueError("PMO arm is outside the matched-arm development lock")
    if (
        task_name not in TASKS
        or task_name not in contract["dynamic_v21_development"]["tasks"]
    ):
        raise ValueError("PMO task is outside the four-task development lock")
    initialized = _load_initialization(root, contract)
    protocol = _runtime_protocol(contract, task_name)
    task = ProgramTask(task_name, identity(protocol), "pmo")
    ledger = ProgramQueryLedger(
        folder / "oracle",
        task,
        evaluate,
        budget=QUERY_BUDGET,
    )
    report = progress or (lambda row: None)
    publish_json(
        folder / "unit.json",
        {
            "schema_version": "pmo_dynamic_v21_matched_unit_v1",
            "arm": arm,
            "task": task_name,
            "search_seed": TASK_SEEDS[task_name],
            "configuration": asdict(configuration(TASK_SEEDS[task_name])),
            "contract_payload_id": identity(contract),
        },
    )
    with _campaign_namespace(arm):
        campaign = run_program_campaign(
            output=folder / "campaign",
            task=task,
            config=configuration(TASK_SEEDS[task_name]),
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
            max_seconds=None,
            progress=report,
        )
    curve = _score_curve(ledger.rows)
    values = [row["score"] for row in ledger.rows]
    shortfalls = []
    for row in campaign["history"]:
        available_budget = max(
            0,
            QUERY_BUDGET
            - (
                row["oracle_calls_including_initialization"] - row["queried_candidates"]
            ),
        )
        requested = min(QUERIES_PER_ROUND, available_budget)
        shortfalls.append(
            {
                "round": row["round"],
                "requested": requested,
                "queried": row["queried_candidates"],
                "shortfall": requested - row["queried_candidates"],
                "eligible_pool": row["pool_size"],
                "proposal_attempts": row["proposal_attempts"],
            }
        )
    attempt_counts = Counter()
    for pending in sorted((folder / "campaign").glob("round_*/pending.json")):
        batch = json.loads(pending.read_text())["batch"]
        attempt_counts.update(row["status"] for row in batch["attempts"])
    return {
        "schema_version": "pmo_dynamic_v21_matched_task_result_v1",
        "arm": arm,
        "task": task_name,
        "search_seed": TASK_SEEDS[task_name],
        "oracle_protocol": protocol,
        "oracle_protocol_id": task.oracle_protocol,
        "status": "complete",
        "termination": (
            "query_budget_exhausted"
            if not ledger.remaining
            else "proposal_yield_or_round_limit"
        ),
        "charged_oracle_calls": len(ledger.rows),
        "successful_oracle_calls": len(ledger.rows),
        "failed_oracle_calls": 0,
        "initialization_calls": sum(
            row["role"] == "initialization" for row in ledger.rows
        ),
        "candidate_calls": sum(row["role"] == "candidate" for row in ledger.rows),
        "best_score": curve[-1]["best_score"],
        "final_top10": curve[-1]["top10_score"],
        "score_curve": curve,
        "auc_top10_development_1000": pmo_top_ten_auc(
            values,
            budget=QUERY_BUDGET,
            frequency=OFFICIAL_METRIC_FREQUENCY,
            finish=True,
        ),
        "auc_top10_official_10k": pmo_top_ten_auc(
            values,
            budget=OFFICIAL_METRIC_BUDGET,
            frequency=OFFICIAL_METRIC_FREQUENCY,
            finish=True,
        ),
        "candidate_shortfalls": shortfalls,
        "total_candidate_shortfall": sum(row["shortfall"] for row in shortfalls),
        "proposal_attempt_status": dict(sorted(attempt_counts.items())),
        "campaign": campaign,
        "query_ledger_path": str((folder / "oracle").relative_to(root)),
    }


@contextmanager
def native_oracle(root: Path, contract: dict, task_name: str):
    """Construct exactly one native task adapter without observing a score."""
    if task_name not in TASKS:
        raise ValueError("native PMO adapter task is outside the four-task lock")
    from tdc import Oracle

    asset_root = root / ORACLE_ASSET_ROOT
    previous = Path.cwd()
    os.chdir(asset_root)
    try:
        adapter = Oracle(name=task_name)
        yield lambda smiles: float(adapter(smiles))
    finally:
        os.chdir(previous)


def _charged_reservations(folder: Path) -> int:
    return sum(1 for _ in (folder / "oracle").glob("query_*/started.json"))


def run_scored_task(root: Path, launch: dict, arm: str, task_name: str) -> dict:
    """Durable single-task boundary. Oracle failures remain charged and terminal."""
    contract = load_contract(root)
    base = {key: value for key, value in launch.items() if key != "run_id"}
    if identity(base) != launch.get("run_id"):
        raise ValueError("PMO Dynamic-v2.1 launch identity changed")
    if launch.get("contract_sha256") != sha256_file(root / CONTRACT):
        raise ValueError("PMO Dynamic-v2.1 launch contract changed")
    if (
        tuple(launch.get("tasks", ())) != TASKS
        or launch.get("charged_query_ceiling") != TOTAL_QUERY_CEILING
        or launch.get("automatic_retries") != 0
        or launch.get("configuration") != contract["controller"]
        or launch.get("configuration_id") != identity(contract["controller"])
        or tuple(launch.get("arms", ())) != ARMS
        or task_name not in launch["tasks"]
        or arm not in launch["arms"]
    ):
        raise ValueError("PMO Dynamic-v2.1 task is absent from the launch lock")
    verify_file(root / PREFLIGHT, launch["preflight_sha256"])
    folder = (
        root / "diagnostics/pmo_dynamic_v21/runs" / launch["run_id"] / arm / task_name
    )
    result_path, failure_path = folder / "result.json", folder / "failure.json"
    if result_path.exists():
        return unseal(result_path)
    if (folder / "started.json").exists():
        raise RuntimeError(
            "PMO Dynamic-v2.1 task was already started; no automatic retry"
        )
    seal(
        folder / "started.json",
        {
            "launch": launch,
            "arm": arm,
            "task": task_name,
            "started_at_utc": _stamp(),
        },
    )
    began, cpu_began = perf_counter(), process_time()

    def report(row):
        publish_json(
            folder / "progress.json",
            {**row, "arm": arm, "task": task_name, "at": _stamp()},
        )

    try:
        qualified = verify_runtime_environment(root, contract)
        report({"phase": "oracle_qualified", "charged_oracle_calls": 0})
        with native_oracle(root, contract, task_name) as evaluate:
            result = execute_task(
                contract,
                root,
                folder,
                arm,
                task_name,
                evaluate=evaluate,
                progress=report,
            )
        result.update(
            contract_sha256=launch["contract_sha256"],
            launch_id=launch["run_id"],
            oracle_qualification=qualified,
            wall_seconds=perf_counter() - began,
            cpu_seconds=process_time() - cpu_began,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            software={
                "python": platform.python_version(),
                "machine": platform.machine(),
                "rdkit": rdBase.rdkitVersion,
                "pytdc": importlib.metadata.version("PyTDC"),
                "numpy": importlib.metadata.version("numpy"),
            },
            completed_at_utc=_stamp(),
        )
        seal(result_path, result)
        return result
    # This outer task boundary records every ordinary failure and never retries it.
    except Exception as error:  # noqa: BLE001
        failure = {
            "schema_version": "pmo_dynamic_v21_matched_task_failure_v1",
            "arm": arm,
            "task": task_name,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "charged_oracle_calls": _charged_reservations(folder),
            "automatic_retry": False,
            "wall_seconds": perf_counter() - began,
            "cpu_seconds": process_time() - cpu_began,
            "failed_at_utc": _stamp(),
        }
        seal(failure_path, failure)
        return failure


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def validate_launch_ready(root: Path, *, workers: int) -> tuple[dict, dict]:
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError("PMO Dynamic-v2.1 launch permits one to four workers")
    if _git(root, "status", "--porcelain"):
        raise ValueError("PMO Dynamic-v2.1 scoring requires clean committed source")
    tracked = set(_git(root, "ls-files").splitlines())
    for path in (CONTRACT, OFFLINE_COMPARATORS, PREFLIGHT):
        if path not in tracked:
            raise ValueError(f"PMO Dynamic-v2.1 launch input is not committed: {path}")
    contract = load_contract(root)
    preflight = unseal(root / PREFLIGHT)
    if (
        preflight.get("passed") is not True
        or preflight.get("new_oracle_calls") != 0
        or preflight.get("contract_sha256") != sha256_file(root / CONTRACT)
        or preflight.get("implementation_sha256") != contract["inputs"]
    ):
        raise ValueError("PMO Dynamic-v2.1 zero-oracle preflight is stale or failed")
    verify_runtime_environment(root, contract)
    return contract, preflight


def aggregate_offline(root: Path, launch: dict, task_results: list[dict]) -> dict:
    """Compare only after all runtime selection has ended."""
    contract = load_contract(root)
    comparator_path = root / contract["postrun_only_comparator"]["path"]
    verify_file(comparator_path, contract["postrun_only_comparator"]["sha256"])
    comparators = unseal(comparator_path)
    by_unit = {}
    for result in task_results:
        key = result.get("arm"), result.get("task")
        if key in by_unit or key[0] not in ARMS or key[1] not in TASKS:
            raise ValueError("PMO matched aggregate has an invalid or duplicate unit")
        if result.get("charged_oracle_calls", 0) > QUERY_BUDGET:
            raise ValueError("PMO matched unit exceeds its charged-call ceiling")
        by_unit[key] = result
    rows, matched = {arm: {} for arm in ARMS}, {}
    for arm in ARMS:
        for task in TASKS:
            result = by_unit.get((arm, task))
            if result is None or result.get("status") != "complete":
                rows[arm][task] = {
                    "status": "failed_or_missing",
                    "charged_oracle_calls": (
                        0 if result is None else result["charged_oracle_calls"]
                    ),
                }
                continue
            old = comparators["tasks"][task]
            auc = result["auc_top10_official_10k"]
            rows[arm][task] = {
                "status": "complete",
                "charged_oracle_calls": result["charged_oracle_calls"],
                "best_score": result["best_score"],
                "final_top10": result["final_top10"],
                "auc_top10_development_1000": result["auc_top10_development_1000"],
                "auc_top10_official_10k": auc,
                "locked_compose_auc_top10_official_10k": old[
                    "locked_compose_auc_top10_official_10k"
                ],
                "ivg_no_prescreen_auc_top10": old["ivg_no_prescreen_auc_top10"],
                "ivg_prescreen_auc_top10": old["ivg_prescreen_auc_top10"],
                "margin_over_locked_compose": auc
                - old["locked_compose_auc_top10_official_10k"],
                "margin_over_ivg_no_prescreen": auc - old["ivg_no_prescreen_auc_top10"],
                "margin_over_ivg_prescreen": auc - old["ivg_prescreen_auc_top10"],
            }
    for task in TASKS:
        v0, v21 = rows["dynamic_v0"][task], rows["dynamic_v21"][task]
        if v0["status"] != "complete" or v21["status"] != "complete":
            matched[task] = {"status": "failed_or_missing_arm"}
            continue
        matched[task] = {
            "status": "complete",
            "delta_v21_minus_v0_best_score": v21["best_score"] - v0["best_score"],
            "delta_v21_minus_v0_final_top10": v21["final_top10"] - v0["final_top10"],
            "delta_v21_minus_v0_auc_top10_development_1000": v21[
                "auc_top10_development_1000"
            ]
            - v0["auc_top10_development_1000"],
            "delta_v21_minus_v0_auc_top10_official_10k": v21["auc_top10_official_10k"]
            - v0["auc_top10_official_10k"],
        }
    total = sum(row.get("charged_oracle_calls", 0) for row in task_results)
    if total > TOTAL_QUERY_CEILING:
        raise ValueError("PMO Dynamic-v2.1 aggregate exceeds its charged-call ceiling")
    return {
        "schema_version": "pmo_dynamic_v21_matched_result_v1",
        "status": (
            "complete"
            if all(row.get("status") == "complete" for row in task_results)
            and set(by_unit) == {(arm, task) for arm in ARMS for task in TASKS}
            else "incomplete"
        ),
        "launch": launch,
        "contract_sha256": sha256_file(root / CONTRACT),
        "preflight_sha256": sha256_file(root / PREFLIGHT),
        "offline_comparator_sha256": sha256_file(comparator_path),
        "charged_oracle_calls": total,
        "charged_oracle_call_ceiling": TOTAL_QUERY_CEILING,
        "automatic_retries": 0,
        "arms": rows,
        "matched_v21_minus_v0": matched,
        "information_regime": (
            "answer-known matched Dynamic-v0 versus Dynamic-v2.1 four-task PMO "
            "development diagnostic; all comparison arithmetic was unavailable to "
            "runtime selection"
        ),
        "completed_at_utc": _stamp(),
    }
