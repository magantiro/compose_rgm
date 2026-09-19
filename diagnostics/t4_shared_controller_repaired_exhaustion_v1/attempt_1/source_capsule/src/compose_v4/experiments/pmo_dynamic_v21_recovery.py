"""One-time recovery adapter for the failed PMO Dynamic-v2.1 units.

The original Dynamic-v2.1 core treats ``dynamic_v21_bootstrap.pool_id`` as a
single archive-continuity identity.  The PMO all-scored-parent campaign created
a new physical cold-start pool when it revisited an initialization parent.  A
small adapter retains every physical pool identity while presenting the first
pool's stable continuity identity to the unchanged core invariant.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import resource
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from rdkit import Chem, rdBase

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v21 import (
    CHANNELS,
    DynamicV21ProgramOptimizer,
    initial_dynamic_program_batch_v21,
    validate_allocator_state,
)
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask, archive_top_k, pmo_top_ten_auc
from compose_v4.experiments import pmo_dynamic_v21 as original
from compose_v4.experiments.continuation_profile import (
    publish_json,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal

RECOVERY_CONTRACT = "configs/pmo_dynamic_v21_recovery_v1.json"
FAILURE_CENSUS = "diagnostics/pmo_dynamic_v21/recovery_failure_census_v1.json"
RECOVERY_PREFLIGHT = "diagnostics/pmo_dynamic_v21/recovery_preflight_v1.json"
LAUNCH_V3 = "diagnostics/pmo_dynamic_v21/launch_v3.json"
RESULT_V3 = "diagnostics/pmo_dynamic_v21/result_v3.json"
RECOVERY_DOC = "docs/PMO_DYNAMIC_V21_RECOVERY.md"
SOURCE_RUN_ID = "97bbbf9a3e3d8ac36965e66ec5c9341b3441622b93f446ee68e414d03cfd0901"
SOURCE_CHARGED_CALLS = 4132
FAILED_V21_CALLS_PER_TASK = 33
MAX_NEW_CALLS_PER_TASK = original.QUERY_BUDGET - FAILED_V21_CALLS_PER_TASK
MAX_NEW_CALLS = len(original.TASKS) * MAX_NEW_CALLS_PER_TASK

_OUTCOME_FIELDS = {
    "charged_outcomes",
    "scored_outcomes",
    "parent_improvements",
    "global_best_improvements",
    "positive_improvement_sum",
}


def _canonical_endpoint(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("recovery encountered an invalid locked endpoint")
    return Chem.MolToSmiles(molecule)


def _merge_bootstrap_proposal_state(current: dict, incoming: dict) -> None:
    """Merge only score-blind proposal accounting from a fresh physical pool."""
    validate_allocator_state(current)
    validate_allocator_state(incoming)
    if current["score_direction"] != incoming["score_direction"]:
        raise ValueError("bootstrap continuity changed score direction")
    if incoming["contexts"] or incoming["global_best"] is not None:
        raise ValueError("fresh bootstrap unexpectedly contains outcome context")
    for channel in CHANNELS:
        target, source = current["channels"][channel], incoming["channels"][channel]
        if set(target) != set(source):
            raise ValueError("bootstrap allocator counter schema changed")
        for key, value in source.items():
            if key in _OUTCOME_FIELDS:
                if value != 0:
                    raise ValueError("fresh bootstrap contains scored outcome state")
                continue
            target[key] += value
    current["allocation_decisions"] += incoming["allocation_decisions"]
    validate_allocator_state(current)


class PMODynamicV21ContinuityAdapter(DynamicV21ProgramOptimizer):
    """PMO-only archive adapter that leaves the core pool check unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pmo_generation_pool_ids: list[str] = []

    def _adapt_bootstrap(self, record: dict) -> dict:
        bootstrap = record.get("provenance", {}).get("dynamic_v21_bootstrap")
        if bootstrap is None:
            return record
        generation_pool_id = bootstrap.get("generation_pool_id", bootstrap["pool_id"])
        if not isinstance(generation_pool_id, str) or len(generation_pool_id) != 64:
            raise ValueError("PMO bootstrap lacks a physical pool identity")
        if self._bootstrap_pool_id is None:
            if generation_pool_id not in self._pmo_generation_pool_ids:
                self._pmo_generation_pool_ids.append(generation_pool_id)
            return record
        if generation_pool_id not in self._pmo_generation_pool_ids:
            _merge_bootstrap_proposal_state(
                self.allocator_state, bootstrap["allocator_state"]
            )
            self.shallow_rng.bit_generator.state = bootstrap["shallow_rng"]
            self.structured_rng.bit_generator.state = bootstrap["structured_rng"]
            self.rng.bit_generator.state = bootstrap["arbitration_rng"]
            self._pmo_generation_pool_ids.append(generation_pool_id)
        if generation_pool_id == self._bootstrap_pool_id:
            return record
        adapted = json.loads(json.dumps(record))
        adapted_bootstrap = adapted["provenance"]["dynamic_v21_bootstrap"]
        adapted_bootstrap["generation_pool_id"] = generation_pool_id
        adapted_bootstrap["pool_id"] = self._bootstrap_pool_id
        adapted["provenance"]["pmo_bootstrap_continuity"] = {
            "schema_version": "pmo_dynamic_v21_bootstrap_continuity_v1",
            "generation_pool_id": generation_pool_id,
            "continuity_pool_id": self._bootstrap_pool_id,
            "boundary": "campaign_admission",
        }
        return adapted

    def add_measured_program(self, record, *, receipt_id, score, static_score=None):
        return super().add_measured_program(
            self._adapt_bootstrap(record),
            receipt_id=receipt_id,
            score=score,
            static_score=static_score,
        )

    def snapshot(self, *, include_history=True):
        snapshot = super().snapshot(include_history=include_history)
        body = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
        body["pmo_bootstrap_continuity"] = {
            "schema_version": "pmo_dynamic_v21_bootstrap_continuity_state_v1",
            "continuity_pool_id": self._bootstrap_pool_id,
            "generation_pool_ids": list(self._pmo_generation_pool_ids),
        }
        return json.loads(json.dumps({**body, "snapshot_id": identity(body)}))

    @classmethod
    def restore(cls, snapshot, *, hierarchy=None):
        result = super().restore(snapshot, hierarchy=hierarchy)
        state = snapshot.get("pmo_bootstrap_continuity")
        if state is None:
            result._pmo_generation_pool_ids = (
                [] if result._bootstrap_pool_id is None else [result._bootstrap_pool_id]
            )
            return result
        if (
            state.get("schema_version")
            != "pmo_dynamic_v21_bootstrap_continuity_state_v1"
            or state.get("continuity_pool_id") != result._bootstrap_pool_id
            or len(state.get("generation_pool_ids", ()))
            != len(set(state.get("generation_pool_ids", ())))
            or any(len(value) != 64 for value in state.get("generation_pool_ids", ()))
        ):
            raise ValueError("corrupt PMO bootstrap-continuity snapshot")
        result._pmo_generation_pool_ids = list(state["generation_pool_ids"])
        return result


@contextmanager
def recovery_campaign_namespace():
    """Inject only the PMO continuity adapter into the reusable campaign."""
    from compose_v4.control import program_campaign

    saved = program_campaign.ProgramOptimizer, program_campaign.initial_program_batch
    program_campaign.ProgramOptimizer = PMODynamicV21ContinuityAdapter
    program_campaign.initial_program_batch = initial_dynamic_program_batch_v21
    try:
        yield
    finally:
        program_campaign.ProgramOptimizer, program_campaign.initial_program_batch = (
            saved
        )


def _read_complete_query(path: Path, *, expected_index: int, protocol: str) -> dict:
    started_path, result_path = path / "started.json", path / "result.json"
    if not started_path.exists() or not result_path.exists():
        raise ValueError(f"recovery ledger reservation is unresolved: {path}")
    started, result = json.loads(started_path.read_text()), json.loads(
        result_path.read_text()
    )
    body = {key: value for key, value in result.items() if key != "receipt_id"}
    if (
        result.get("status") != "complete"
        or result.get("index") != expected_index
        or result.get("oracle_protocol") != protocol
        or identity(body) != result.get("receipt_id")
        or any(result.get(key) != started.get(key) for key in started)
    ):
        raise ValueError(f"recovery ledger receipt changed: {result_path}")
    return result


def _batch_generation_pool(batch: dict) -> str:
    candidates = batch.get("candidates", ())
    pools = {
        row.get("provenance", {}).get("dynamic_v21_bootstrap", {}).get("pool_id")
        for row in candidates
    }
    if len(candidates) != original.QUERIES_PER_ROUND or len(pools) != 1:
        raise ValueError("recovery pending batch is not one complete physical pool")
    pool_id = pools.pop()
    if not isinstance(pool_id, str) or len(pool_id) != 64:
        raise ValueError("recovery pending batch pool identity changed")
    return pool_id


def _unit_folder(root: Path, task_name: str) -> Path:
    return (
        root
        / "diagnostics/pmo_dynamic_v21/runs"
        / SOURCE_RUN_ID
        / "dynamic_v21"
        / task_name
    )


def audit_recovery_unit(root: Path, task_name: str) -> dict:
    """Read and simulate one failed unit without constructing or calling an oracle."""
    if task_name not in original.TASKS:
        raise ValueError("recovery task is outside the four-task lock")
    contract = original.load_contract(root)
    folder = _unit_folder(root, task_name)
    failure = unseal(folder / "failure.json")
    if (
        failure.get("status") != "failed"
        or failure.get("error") != "v2.1 bootstrap candidates came from different pools"
        or failure.get("charged_oracle_calls") != FAILED_V21_CALLS_PER_TASK
        or (folder / "result.json").exists()
    ):
        raise ValueError("unit does not match the authorized v2.1 failure census")
    protocol = original._runtime_protocol(contract, task_name)
    task = ProgramTask(task_name, identity(protocol), "pmo")
    oracle = folder / "oracle"
    manifest = json.loads((oracle / "manifest.json").read_text())
    if manifest != {
        "task": asdict(task),
        "budget": original.QUERY_BUDGET,
        "schema_version": "program_query_ledger_v1",
    }:
        raise ValueError("recovery query-ledger manifest changed")
    queries = [
        _read_complete_query(
            oracle / f"query_{index:06d}",
            expected_index=index,
            protocol=task.oracle_protocol,
        )
        for index in range(FAILED_V21_CALLS_PER_TASK)
    ]
    if (
        list(oracle.glob("query_*/started.json"))
        and len(list(oracle.glob("query_*/started.json"))) != FAILED_V21_CALLS_PER_TASK
    ):
        raise ValueError(
            "recovery query ledger is not exactly contiguous through index 32"
        )
    if (
        sum(row["role"] == "initialization" for row in queries)
        != original.INITIALIZATION_COUNT
    ):
        raise ValueError("recovery initialization accounting changed")
    round0 = json.loads((folder / "campaign/round_0000/complete.json").read_text())
    pending_path = folder / "campaign/round_0001/pending.json"
    if not pending_path.exists() or pending_path.with_name("complete.json").exists():
        raise ValueError("recovery requires round 0 complete and round 1 pending")
    pending = json.loads(pending_path.read_text())
    if pending.get("recipe_sha256") != round0.get("recipe_sha256"):
        raise ValueError("recovery campaign recipe changed between rounds")
    batch = pending["batch"]
    first, observed = batch["candidates"][0], queries[-1]
    if (
        observed["lock_id"] != batch.get("batch_id")
        or observed["role"] != "candidate"
        or _canonical_endpoint(first["endpoint"]) != observed["endpoint"]
    ):
        raise ValueError("query 32 is not the first locked round-1 candidate")
    base = DynamicV21ProgramOptimizer.restore(round0["snapshot"])
    old_pool = base._bootstrap_pool_id
    new_pool = _batch_generation_pool(batch)
    if old_pool == new_pool:
        raise ValueError("recorded failure does not contain distinct physical pools")
    try:
        base.add_measured_program(
            first, receipt_id=observed["receipt_id"], score=observed["score"]
        )
    except ValueError as error:
        if str(error) != "v2.1 bootstrap candidates came from different pools":
            raise
    else:
        raise ValueError("unchanged core no longer reproduces the pool mismatch")
    recovered = PMODynamicV21ContinuityAdapter.restore(round0["snapshot"])
    before = len(recovered.observations)
    recovered.add_measured_program(
        first, receipt_id=observed["receipt_id"], score=observed["score"]
    )
    if (
        len(recovered.observations) != before + 1
        or recovered.observations[observed["receipt_id"]]["score"] != observed["score"]
        or recovered._pmo_generation_pool_ids != [old_pool, new_pool]
    ):
        raise ValueError("continuity adapter did not consume query 32 exactly once")
    return {
        "task": task_name,
        "source_run_id": SOURCE_RUN_ID,
        "charged_completed_queries": len(queries),
        "unresolved_queries": 0,
        "initialization_queries": original.INITIALIZATION_COUNT,
        "candidate_queries": len(queries) - original.INITIALIZATION_COUNT,
        "remaining_query_ceiling": original.QUERY_BUDGET - len(queries),
        "round0_complete": True,
        "round1_pending": True,
        "round1_candidate_count": len(batch["candidates"]),
        "round1_consumed_prefix": 1,
        "existing_query_index": observed["index"],
        "existing_query_receipt_id": observed["receipt_id"],
        "existing_query_score": observed["score"],
        "existing_query_endpoint": observed["endpoint"],
        "round0_continuity_pool_id": old_pool,
        "round1_generation_pool_id": new_pool,
        "simulated_snapshot_id": recovered.snapshot()["snapshot_id"],
        "new_oracle_calls": 0,
    }


def consume_existing_round1_observation(
    search: PMODynamicV21ContinuityAdapter, batch: dict, ledger: ProgramQueryLedger
) -> dict:
    """Consume the one completed pending score before a recovery can call its oracle."""
    if len(ledger.rows) != FAILED_V21_CALLS_PER_TASK:
        raise ValueError("recovery must begin from exactly 33 charged observations")
    candidate = batch["candidates"][0]
    observed = ledger.rows[-1]
    if (
        observed.get("index") != FAILED_V21_CALLS_PER_TASK - 1
        or observed.get("status") != "complete"
        or observed.get("lock_id") != batch["batch_id"]
        or observed.get("endpoint") != _canonical_endpoint(candidate["endpoint"])
    ):
        raise ValueError("existing round-1 score does not match its locked candidate")
    before = len(ledger.rows)
    cached = ledger.query(
        candidate["endpoint"], lock_id=batch["batch_id"], role="candidate"
    )
    if len(ledger.rows) != before or cached != observed:
        raise ValueError(
            "existing round-1 query was not consumed as an uncharged cache hit"
        )
    search.add_measured_program(
        candidate, receipt_id=cached["receipt_id"], score=cached["score"]
    )
    return {
        "candidate_index": 0,
        "candidate_id": candidate["candidate_id"],
        "query_index": cached["index"],
        "receipt_id": cached["receipt_id"],
        "score": cached["score"],
        "charged_queries_before": before,
        "charged_queries_after": len(ledger.rows),
        "new_oracle_calls": 0,
        "snapshot_id_after_consumption": search.snapshot()["snapshot_id"],
    }


def _initialization_parent(
    initialized: dict, config, ledger: ProgramQueryLedger
) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1, 31]))
    if rng.random() >= 0.2:
        raise ValueError("round-1 initial-parent selection no longer reproduces")
    at = int(rng.integers(len(initialized["candidates"])))
    start = initialized["candidates"][at]
    return {
        "endpoint": start["endpoint"],
        "index": at,
        "available_scored_parents": len(initialized["candidates"]),
        "selection": "uniform_all_scored",
        "observation_receipt": ledger.cache[start["endpoint"]]["receipt_id"],
    }


def recover_pending_round(
    root: Path,
    folder: Path,
    task: ProgramTask,
    initialized: dict,
    ledger: ProgramQueryLedger,
) -> dict:
    """Finish the immutable round-1 lock, reusing its first completed result."""
    round0 = json.loads((folder / "campaign/round_0000/complete.json").read_text())
    pending_path = folder / "campaign/round_0001/pending.json"
    complete_path = pending_path.with_name("complete.json")
    if complete_path.exists():
        raise ValueError("round-1 recovery was already completed; no retry")
    pending = json.loads(pending_path.read_text())
    batch = pending["batch"]
    search = PMODynamicV21ContinuityAdapter.restore(round0["snapshot"])
    consumed = consume_existing_round1_observation(search, batch, ledger)
    recovery_folder = folder / "recovery_v1"
    seal(
        recovery_folder / "consumed_existing_round1.json",
        {
            "schema_version": "pmo_dynamic_v21_existing_observation_consumption_v1",
            "task": task.name,
            "source_run_id": SOURCE_RUN_ID,
            **consumed,
        },
    )
    began = perf_counter()
    for candidate in batch["candidates"][1:]:
        row = ledger.query(
            candidate["endpoint"], lock_id=batch["batch_id"], role="candidate"
        )
        search.add_measured_program(
            candidate, receipt_id=row["receipt_id"], score=row["score"]
        )
    values = [row["score"] for row in ledger.rows]
    summary = {
        "round": 1,
        "queried_candidates": len(batch["candidates"]),
        "oracle_calls_including_initialization": len(ledger.rows),
        "top_k_utility": archive_top_k(
            [(row["endpoint"], task.utility(row["score"])) for row in ledger.rows],
            k=task.top_k,
        ),
        "top_k": task.top_k,
        "model_updated": False,
        "proposal_seconds": batch.get("proposal_seconds", 0.0),
        "proposal_attempts": len(batch["attempts"]),
        "pool_size": len(batch.get("proposal_pool", batch)["candidates"]),
        "seconds": perf_counter() - began,
        "best_new_utility": max(task.utility(value) for value in values),
        "initialization_parent": _initialization_parent(
            initialized, search.config, ledger
        ),
        "work_cache": batch.get("work_cache"),
        "pmo_top10_auc_so_far": pmo_top_ten_auc(values, budget=ledger.budget),
        "recovery": {
            "schema_version": "pmo_dynamic_v21_pending_round_recovery_v1",
            "existing_observation_consumed_before_new_calls": True,
            "existing_query_index": consumed["query_index"],
            "new_oracle_calls_in_round": len(ledger.rows)
            - consumed["charged_queries_before"],
        },
    }
    publish_json(
        complete_path,
        {
            "snapshot": search.snapshot(),
            "summary": summary,
            "recipe_sha256": pending["recipe_sha256"],
        },
    )
    return summary


def _task_result(
    root: Path,
    folder: Path,
    task_name: str,
    protocol: dict,
    campaign: dict,
    ledger: ProgramQueryLedger,
    *,
    source_calls: int,
) -> dict:
    curve = original._score_curve(ledger.rows)
    values = [row["score"] for row in ledger.rows]
    shortfalls = []
    for row in campaign["history"]:
        available = max(
            0,
            original.QUERY_BUDGET
            - (
                row["oracle_calls_including_initialization"] - row["queried_candidates"]
            ),
        )
        requested = min(original.QUERIES_PER_ROUND, available)
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
    attempts = Counter()
    for pending in sorted((folder / "campaign").glob("round_*/pending.json")):
        attempts.update(
            row["status"]
            for row in json.loads(pending.read_text())["batch"]["attempts"]
        )
    new_calls = len(ledger.rows) - source_calls
    if not 0 <= new_calls <= MAX_NEW_CALLS_PER_TASK:
        raise ValueError("recovery exceeded its per-task new-call ceiling")
    return {
        "schema_version": "pmo_dynamic_v21_matched_task_result_v1",
        "arm": "dynamic_v21",
        "task": task_name,
        "search_seed": original.TASK_SEEDS[task_name],
        "oracle_protocol": protocol,
        "oracle_protocol_id": identity(protocol),
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
            budget=original.QUERY_BUDGET,
            frequency=original.OFFICIAL_METRIC_FREQUENCY,
            finish=True,
        ),
        "auc_top10_official_10k": pmo_top_ten_auc(
            values,
            budget=original.OFFICIAL_METRIC_BUDGET,
            frequency=original.OFFICIAL_METRIC_FREQUENCY,
            finish=True,
        ),
        "candidate_shortfalls": shortfalls,
        "total_candidate_shortfall": sum(row["shortfall"] for row in shortfalls),
        "proposal_attempt_status": dict(sorted(attempts.items())),
        "campaign": campaign,
        "query_ledger_path": str((folder / "oracle").relative_to(root)),
        "recovery": {
            "schema_version": "pmo_dynamic_v21_task_recovery_v1",
            "source_run_id": SOURCE_RUN_ID,
            "preserved_charged_calls": source_calls,
            "new_charged_calls": new_calls,
            "new_charged_call_ceiling": MAX_NEW_CALLS_PER_TASK,
            "automatic_retries": 0,
        },
    }


def execute_recovery_task(
    contract: dict,
    root: Path,
    folder: Path,
    task_name: str,
    *,
    evaluate,
    progress=None,
) -> dict:
    """Resume one authorized unit from its 33-call durable boundary."""
    audit_recovery_unit(root, task_name)
    original_contract = original.load_contract(root)
    initialized = original._load_initialization(root, original_contract)
    protocol = original._runtime_protocol(original_contract, task_name)
    task = ProgramTask(task_name, identity(protocol), "pmo")
    ledger = ProgramQueryLedger(
        folder / "oracle", task, evaluate, budget=original.QUERY_BUDGET
    )
    source_calls = len(ledger.rows)
    if source_calls != FAILED_V21_CALLS_PER_TASK:
        raise ValueError("recovery source ledger changed after audit")
    recover_pending_round(root, folder, task, initialized, ledger)
    report = progress or (lambda row: None)
    with recovery_campaign_namespace():
        campaign = run_program_campaign(
            output=folder / "campaign",
            task=task,
            config=original.configuration(original.TASK_SEEDS[task_name]),
            initialization=initialized,
            library=(),
            ledger=ledger,
            rounds=original.MAX_ROUNDS,
            queries_per_round=original.QUERIES_PER_ROUND,
            hierarchy=None,
            fit_model=None,
            stagnation_rounds=None,
            bootstrap_rounds=1,
            initialization_mode="all_scored_pool",
            initial_parent_fraction=0.2,
            max_seconds=None,
            progress=report,
        )
    return _task_result(
        root,
        folder,
        task_name,
        protocol,
        campaign,
        ledger,
        source_calls=source_calls,
    )


def run_recovery_task(root: Path, launch: dict, task_name: str) -> dict:
    """Durable one-time worker boundary; the original failure stays immutable."""
    contract = load_recovery_contract(root)
    body = {key: value for key, value in launch.items() if key != "recovery_id"}
    if identity(body) != launch.get("recovery_id"):
        raise ValueError("PMO recovery launch identity changed")
    if (
        launch.get("recovery_contract_sha256") != sha256_file(root / RECOVERY_CONTRACT)
        or launch.get("source_run_id") != SOURCE_RUN_ID
        or tuple(launch.get("tasks", ())) != original.TASKS
        or task_name not in launch["tasks"]
        or launch.get("max_new_calls_per_task") != MAX_NEW_CALLS_PER_TASK
        or launch.get("automatic_retries") != 0
        or launch.get("configuration") != contract["controller"]
    ):
        raise ValueError("PMO v2.1 recovery task is absent from the launch lock")
    verify_file(root / RECOVERY_PREFLIGHT, launch["recovery_preflight_sha256"])
    folder = _unit_folder(root, task_name)
    recovery_folder = folder / "recovery_v1"
    result_path, failure_path = folder / "result.json", recovery_folder / "failure.json"
    if result_path.exists() or (recovery_folder / "started.json").exists():
        raise RuntimeError("PMO v2.1 recovery was already started; no automatic retry")
    seal(
        recovery_folder / "started.json",
        {
            "schema_version": "pmo_dynamic_v21_task_recovery_start_v1",
            "launch": launch,
            "task": task_name,
            "source_failure_sha256": sha256_file(folder / "failure.json"),
            "started_at_utc": _stamp(),
        },
    )
    began, cpu_began = perf_counter(), process_time()

    def report(row):
        publish_json(
            recovery_folder / "progress.json",
            {**row, "arm": "dynamic_v21", "task": task_name, "at": _stamp()},
        )

    try:
        original_contract = original.load_contract(root)
        qualified = original.verify_runtime_environment(root, original_contract)
        report(
            {
                "phase": "recovery_oracle_qualified",
                "preserved_charged_oracle_calls": FAILED_V21_CALLS_PER_TASK,
            }
        )
        with original.native_oracle(root, original_contract, task_name) as evaluate:
            result = execute_recovery_task(
                contract,
                root,
                folder,
                task_name,
                evaluate=evaluate,
                progress=report,
            )
        result.update(
            contract_sha256=launch["source_contract_sha256"],
            launch_id=launch["source_run_id"],
            recovery_contract_sha256=launch["recovery_contract_sha256"],
            recovery_id=launch["recovery_id"],
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
    except Exception as error:  # noqa: BLE001
        failure = {
            "schema_version": "pmo_dynamic_v21_task_recovery_failure_v1",
            "arm": "dynamic_v21",
            "task": task_name,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "charged_oracle_calls": original._charged_reservations(folder),
            "source_charged_oracle_calls": FAILED_V21_CALLS_PER_TASK,
            "automatic_retry": False,
            "wall_seconds": perf_counter() - began,
            "cpu_seconds": process_time() - cpu_began,
            "failed_at_utc": _stamp(),
        }
        seal(failure_path, failure)
        return failure


def _verify_census_files(root: Path, census: dict) -> None:
    for relative, digest in census["source_artifact_sha256"].items():
        verify_file(root / relative, digest)


def load_recovery_contract(root: Path) -> dict:
    contract = unseal(root / RECOVERY_CONTRACT)
    if (
        contract.get("schema_version") != "pmo_dynamic_v21_recovery_contract_v1"
        or contract.get("tasks") != list(original.TASKS)
        or contract.get("source_run_id") != SOURCE_RUN_ID
        or contract.get("source_calls_per_task") != FAILED_V21_CALLS_PER_TASK
        or contract.get("max_new_calls_per_task") != MAX_NEW_CALLS_PER_TASK
        or contract.get("max_new_calls") != MAX_NEW_CALLS
        or contract.get("automatic_retries") != 0
        or contract.get("controller") != original.serialized_configuration()
        or contract.get("initial_task_specific_complete_routes") != 0
        or contract.get("t4_affected") is not False
    ):
        raise ValueError("PMO recovery contract policy changed")
    for relative, digest in contract["inputs"].items():
        verify_file(root / relative, digest)
    census = unseal(root / FAILURE_CENSUS)
    if sha256_file(root / FAILURE_CENSUS) != contract["failure_census_sha256"]:
        raise ValueError("PMO recovery failure census changed")
    _verify_census_files(root, census)
    return contract


def recovery_unit_files(root: Path) -> list[Path]:
    base = root / "diagnostics/pmo_dynamic_v21/runs" / SOURCE_RUN_ID
    return sorted(path for path in base.rglob("*") if path.is_file())


def build_failure_census(root: Path) -> dict:
    launch_v2 = unseal(root / original.LAUNCH_V2)
    result = unseal(root / original.RESULT)
    if (
        launch_v2.get("run_id") != SOURCE_RUN_ID
        or result.get("schema_version") != "pmo_dynamic_v21_matched_result_v1"
        or result.get("status") != "incomplete"
        or result.get("charged_oracle_calls") != SOURCE_CHARGED_CALLS
    ):
        raise ValueError("PMO source launch or incomplete aggregate changed")
    audits = [audit_recovery_unit(root, task) for task in original.TASKS]
    for task in original.TASKS:
        v0 = unseal(
            root
            / "diagnostics/pmo_dynamic_v21/runs"
            / SOURCE_RUN_ID
            / "dynamic_v0"
            / task
            / "result.json"
        )
        if v0.get("status") != "complete" or v0.get("charged_oracle_calls") != 1000:
            raise ValueError("completed Dynamic-v0 comparator outcome changed")
    files = recovery_unit_files(root)
    return {
        "schema_version": "pmo_dynamic_v21_recovery_failure_census_v1",
        "source_launch_v2": original.LAUNCH_V2,
        "source_launch_v2_sha256": sha256_file(root / original.LAUNCH_V2),
        "source_run_id": SOURCE_RUN_ID,
        "source_incomplete_result": original.RESULT,
        "source_incomplete_result_sha256": sha256_file(root / original.RESULT),
        "source_charged_oracle_calls": SOURCE_CHARGED_CALLS,
        "completed_dynamic_v0_tasks": list(original.TASKS),
        "completed_dynamic_v0_calls": 4000,
        "failed_dynamic_v21_tasks": list(original.TASKS),
        "preserved_dynamic_v21_calls": len(original.TASKS) * FAILED_V21_CALLS_PER_TASK,
        "failure": "v2.1 bootstrap candidates came from different pools",
        "automatic_retry": False,
        "unit_audits": audits,
        "source_artifact_count": len(files),
        "source_artifact_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in files
        },
        "new_oracle_calls": 0,
    }


def recovery_preflight(root: Path) -> dict:
    contract = load_recovery_contract(root)
    qualified = original.verify_runtime_environment(root, original.load_contract(root))
    audits = [audit_recovery_unit(root, task) for task in original.TASKS]
    return {
        "schema_version": "pmo_dynamic_v21_recovery_preflight_v1",
        "passed": True,
        "contract_sha256": sha256_file(root / RECOVERY_CONTRACT),
        "failure_census_sha256": contract["failure_census_sha256"],
        "source_run_id": SOURCE_RUN_ID,
        "preserved_completed_calls": sum(
            row["charged_completed_queries"] for row in audits
        ),
        "remaining_query_ceiling": sum(
            row["remaining_query_ceiling"] for row in audits
        ),
        "all_ledgers_contiguous": True,
        "all_unresolved_reservations": 0,
        "existing_round1_scores_consumed_before_new_calls": True,
        "core_mismatch_invariant_preserved": True,
        "continuity_boundary": "PMO campaign admission",
        "implementation_sha256": contract["inputs"],
        "oracle_qualification": qualified,
        "software": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": importlib.metadata.version("numpy"),
        },
        "task_audits": audits,
        "new_oracle_calls": 0,
    }
