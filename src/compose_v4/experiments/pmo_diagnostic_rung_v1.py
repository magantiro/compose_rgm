"""All-23-task PMO diagnostic rung: contract, runtime and per-task readouts.

WHY THIS IS A SEPARATE MODULE
-----------------------------
``compose_v4.experiments.pmo_population_v1`` holds a deliberate three-task lock --
``execute_task`` raises on any task outside ``("gsk3b", "perindopril_mpo",
"celecoxib_rediscovery")`` -- and its physical hash is pinned into the corrected
three-task scored contract's ``implementation_sha256``.  Widening that tuple in place
would invalidate the chain that produced the only VALID PMO readings we have, so the
rung gets its own module, its own contract and its own identity.  The *controller* is
unchanged and imported, not copied: ``configuration()`` and ``PmoPopulationController``
come from ``pmo_population_v1``, so the rung measures the same search that produced
gsk3b 0.21 / perindopril_mpo 0.486 / celecoxib_rediscovery 0.196.

WHAT THIS RUNG ANSWERS
----------------------
We hold valid 250-call readings on three tasks only.  That cannot distinguish "COMPOSE
is weak everywhere" from "COMPOSE is badly matched to those three landscapes".  A cheap
pass over all 23 tasks separates the two, and the per-task instrumentation below is
chosen so a weak score is attributable: a task where nothing is ever realized is a
different failure from one where realization succeeds and never improves.

BUDGET SEMANTICS -- THE THING THAT ALREADY BIT US
-------------------------------------------------
``charged_calls_per_task`` is REQUIRED and has no default, exactly as in
``pmo_population_v1``.  The prior defect was not a wrong number, it was a contract field
nothing read: the contract declared 250 while the ledger was built from a module
constant ``QUERY_BUDGET = 1000``.  Here the rung additionally derives its ROUND COUNT
from the budget (``rounds_for_budget``) rather than from a module constant, so a short
rung cannot silently inherit a 64-round campaign shape sized for 1,000 calls.

AUC AT A SHORT BUDGET IS NOT COMPARABLE TO A PUBLISHED ONE
----------------------------------------------------------
``pmo_top_ten_auc`` trapezoids up from (0, 0), so a run holding a constant top-10 level
``c`` scores ``c * (1 - frequency / (2 * budget))``.  At the official 10,000-call budget
that removes 0.5% of the level; at this rung's budget it removes far more.  The AUC key
therefore carries its own budget and the readout names ``best_at_k`` as primary.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v21 import (
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask, pmo_top_ten_auc
from compose_v4.control.docking_value import identity as _identity  # noqa: F401  (re-export clarity)
from compose_v4.experiments.continuation_profile import verify_file
from compose_v4.experiments.pmo_dynamic_v21 import (
    ORACLE_ENVIRONMENT,
    ORACLE_SOURCE_SHA256,
    _load_initialization,
)
from compose_v4.experiments.pmo_oracle_assets import (
    ASSET_BACKED_PMO_TASKS,
    POSITIVE_CONTROL_STATUS,
)
from compose_v4.experiments.pmo_population_v1 import (
    CHECKPOINTS,
    INIT_COUNT,
    QUERIES_PER_ROUND,
    _load_checkpoint,
    configuration,
)

SCHEMA = "pmo_diagnostic_rung_23task_v1"
CONTRACT = "configs/pmo_diagnostic_rung_23task_v1.json"
REGISTRY_VERIFICATION = "diagnostics/pmo_task_registry_verification_v1.json"
CONTROLLER_SOURCE = "configs/pmo_population_controller_v1.json"

# The 23-task PMO suite.  PyTDC does NOT define this suite and ships no membership
# list; it ships the oracles.  Membership is taken from two independent published
# transcriptions that agree exactly, recorded in REGISTRY_VERIFICATION, and
# `verify_suite_against_registry` re-derives it rather than trusting this tuple.
SUITE = (
    "albuterol_similarity",
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "deco_hop",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "jnk3",
    "median1",
    "median2",
    "mestranol_similarity",
    "osimertinib_mpo",
    "perindopril_mpo",
    "qed",
    "ranolazine_mpo",
    "scaffold_hop",
    "sitagliptin_mpo",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
    "valsartan_smarts",
    "zaleplon_mpo",
)

# The rung's readout checkpoints.  `best_at_k` is the PRIMARY readout; AUC is reported
# only within-rung.  The largest checkpoint must be <= the charged budget or it cannot
# be populated, which `validate_budget` enforces rather than silently reporting None.
READOUT_CHECKPOINTS = (32, 64, 128)

# Measured spin-up, from the campaign loop rather than from a docstring:
# `run_program_campaign` charges every initialization candidate BEFORE the round loop
# (`for row in starts: ledger.query(..., role="initialization")`), then round 0 is a
# bootstrap round because `round_index < bootstrap_rounds` with `bootstrap_rounds=1`.
# So adaptive (archive-informed) selection begins only after INIT_COUNT + one round.
SPIN_UP_CALLS = INIT_COUNT + QUERIES_PER_ROUND


class RungBudgetError(ValueError):
    """The requested budget cannot support the rung's declared readouts."""


def validate_budget(charged_calls_per_task: int) -> dict:
    """Refuse a budget that cannot populate what the contract promises to report.

    A budget below the largest readout checkpoint would leave `best_at_128` empty while
    the contract still advertised it; a budget at or below `SPIN_UP_CALLS` would buy no
    adaptive round at all and would measure the initialization bank alone.
    """
    if not isinstance(charged_calls_per_task, int) or isinstance(charged_calls_per_task, bool):
        raise RungBudgetError("charged_calls_per_task must be an int")
    if charged_calls_per_task <= 0:
        raise RungBudgetError("charged_calls_per_task must be positive")
    if charged_calls_per_task < INIT_COUNT:
        raise RungBudgetError(
            f"budget {charged_calls_per_task} cannot pay for the {INIT_COUNT}-molecule "
            "initialization block, which is charged before any round"
        )
    largest = max(READOUT_CHECKPOINTS)
    if charged_calls_per_task < largest:
        raise RungBudgetError(
            f"budget {charged_calls_per_task} cannot populate readout checkpoint {largest}"
        )
    if charged_calls_per_task <= SPIN_UP_CALLS:
        raise RungBudgetError(
            f"budget {charged_calls_per_task} buys no adaptive round: initialization "
            f"({INIT_COUNT}) plus the bootstrap round ({QUERIES_PER_ROUND}) already "
            f"consumes {SPIN_UP_CALLS} charged calls"
        )
    return {
        "charged_calls_per_task": charged_calls_per_task,
        "initialization_calls": INIT_COUNT,
        "bootstrap_round_calls": QUERIES_PER_ROUND,
        "spin_up_calls": SPIN_UP_CALLS,
        "adaptive_calls": charged_calls_per_task - SPIN_UP_CALLS,
        "adaptive_call_fraction": (charged_calls_per_task - SPIN_UP_CALLS)
        / charged_calls_per_task,
        "rounds": rounds_for_budget(charged_calls_per_task),
        "adaptive_rounds": rounds_for_budget(charged_calls_per_task) - 1,
        "readout_checkpoints": list(READOUT_CHECKPOINTS),
    }


def rounds_for_budget(charged_calls_per_task: int) -> int:
    """Rounds needed for the ledger -- not a module constant -- to bind the budget.

    `run_program_campaign` truncates its last round with `min(queries_per_round,
    ledger.remaining)` and breaks on `not ledger.remaining`, so supplying at least this
    many rounds makes the BUDGET the binding constraint.  Supplying a fixed 64 would
    also work numerically, but it would hide which constraint is binding, and a future
    budget above 64 rounds' worth of calls would silently stop early.
    """
    return max(1, math.ceil((charged_calls_per_task - INIT_COUNT) / QUERIES_PER_ROUND))


def verify_suite_against_registry(root: Path) -> dict:
    """Re-derive the 23 task names from the offline registry verification artifact.

    This never constructs a `tdc.Oracle`.  `Oracle.__init__` fuzzy-matches at threshold
    0.8, so a near-miss name resolves SILENTLY to a different oracle -- and the registry
    really does contain `sitagliptin_mpo_prev` and `zaleplon_mpo_prev`, one token from
    two real tasks.  Exact membership in `tdc.metadata.oracle_names` is the check, and
    the verification artifact records it together with its PyTDC wheel provenance.
    """
    payload = json.loads((root / REGISTRY_VERIFICATION).read_text())
    suite = tuple(payload["suite"]["tasks"])
    verification = payload["verification"]
    if payload.get("schema_version") != "pmo_task_registry_verification_v1":
        raise ValueError("unexpected PMO task registry verification schema")
    if len(suite) != 23:
        raise ValueError(f"PMO suite must hold 23 tasks, found {len(suite)}")
    if tuple(sorted(suite)) != tuple(sorted(SUITE)):
        raise ValueError("PMO suite membership disagrees with the verified registry")
    if not verification.get("all_resolve"):
        raise ValueError("verified registry does not report exact resolution for every task")
    per_task = verification["tasks"]
    missing = [task for task in SUITE if task not in per_task]
    if missing:
        raise ValueError(f"registry verification is missing tasks: {missing}")
    return {
        "n_tasks": len(suite),
        "all_resolve_exactly": True,
        "oracle_constructed": bool(payload.get("oracle_constructed")),
        "registry_provenance": payload["registry_provenance"],
        "fuzzy_fallback_hazard": verification["fuzzy_fallback_hazard"],
        "suite_provenance": payload["suite"]["provenance"],
    }


def asset_backed_plan(root: Path, asset_root: str) -> dict:
    """Which suite tasks need a pinned asset, and whether that asset is present.

    Fail-closed input to the launch decision.  A missing asset is NOT benign: `drd2`
    and `gsk3b` load `oracle/<name>.pkl` LAZILY at CALL time and PyTDC's
    `Oracle.__call__` swallows the resulting `FileNotFoundError` into a constant 0.0,
    which is how a complete 250-call gsk3b ledger of exact zeros was produced.  `jnk3`
    loads EAGERLY and raises at construction instead.
    """
    directory = root / asset_root / "oracle"
    present = {}
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.is_file():
                present[path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
    rows = {}
    for task in sorted(ASSET_BACKED_PMO_TASKS & set(SUITE)):
        filename = f"{task}_current.pkl"
        rows[task] = {
            "asset_filename": filename,
            "asset_present_in_capsule": filename in present,
            "asset_sha256": present.get(filename, {}).get("sha256"),
            "positive_control_status": POSITIVE_CONTROL_STATUS.get(task, "UNKNOWN"),
            "positive_control_required_before_first_charged_call": True,
        }
    blocked = sorted(t for t, row in rows.items() if not row["asset_present_in_capsule"])
    return {
        "asset_root": asset_root,
        "capsule_contents": sorted(present),
        "tasks": rows,
        "blocked_pending_asset_pin": blocked,
        "runnable_tasks": [t for t in SUITE if t not in blocked],
        "pure_rdkit_tasks": [t for t in SUITE if t not in ASSET_BACKED_PMO_TASKS],
    }


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(root: Path) -> dict:
    """Open the sealed rung contract and re-verify everything it pins."""
    envelope = json.loads((root / CONTRACT).read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError("rung contract must be a sealed {payload, payload_sha256} envelope")
    contract = envelope["payload"]
    if envelope["payload_sha256"] != identity(contract):
        raise ValueError("rung contract envelope hash changed")
    if contract.get("schema_version") != SCHEMA:
        raise ValueError("unexpected PMO diagnostic rung contract schema")
    if tuple(contract.get("tasks", ())) != SUITE:
        raise ValueError("rung contract task list is not the verified 23-task suite")
    verify_suite_against_registry(root)
    budget = contract["budget"]["charged_calls_per_task"]
    declared = validate_budget(budget)
    for key in ("rounds", "spin_up_calls", "adaptive_calls"):
        if contract["budget"].get(key) != declared[key]:
            raise ValueError(f"rung contract budget field {key!r} disagrees with the runtime")
    if contract["budget"].get("charged_calls_ceiling") != budget * len(SUITE):
        raise ValueError("rung contract ceiling is not budget x task count")
    _load_initialization(root, {"initialization": contract["initialization"]})
    _load_checkpoint(root, contract)
    for path, digest in contract["implementation_sha256"].items():
        verify_file(root / path, digest)
    return contract


def runtime_protocol(contract: dict, task: str) -> dict:
    """The oracle protocol string the ledger binds every charged receipt to.

    Unlike the three-task version this carries the asset digest for EVERY asset-backed
    task rather than for gsk3b alone, so a receipt produced against a different drd2 or
    jnk3 pickle is a different protocol and cannot be reconciled with this one.
    """
    assets = contract["oracle"]["assets"]
    return {
        "task": task,
        "implementation": "native PyTDC Oracle",
        "environment": contract["oracle"]["environment"],
        "source_sha256": contract["oracle"]["source_files"],
        "asset_sha256": (
            {f"{task}_current.pkl": assets[task]["asset_sha256"]}
            if task in assets and assets[task].get("asset_sha256")
            else {}
        ),
        "direction": "maximize",
        "range": [0, 1],
        "prescreen": False,
        "calls_include_initialization": True,
    }


def execute_task(
    contract: dict,
    root: Path,
    folder: Path,
    task_name: str,
    *,
    evaluate: Callable[[str], float],
    charged_calls_per_task: int,
    progress=None,
) -> dict:
    """Run one rung task once.

    ``charged_calls_per_task`` is REQUIRED and has no default, and the round count is
    derived from it.  Both exist so that no caller can inherit a budget or a campaign
    shape it never declared.
    """
    validate_budget(charged_calls_per_task)
    if task_name not in SUITE:
        raise ValueError(f"{task_name!r} is outside the verified 23-task PMO suite")
    if not contract.get("scored_launch_authorized"):
        raise ValueError("PMO diagnostic rung scored launch is not authorized by this contract")
    if contract["budget"]["charged_calls_per_task"] != charged_calls_per_task:
        raise ValueError("runtime budget does not match the authorizing contract")

    initialized = _load_initialization(root, {"initialization": contract["initialization"]})
    task = ProgramTask(task_name, identity(runtime_protocol(contract, task_name)), "pmo")
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
        rounds=rounds_for_budget(charged_calls_per_task),
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
    values = [row["score"] for row in ledger.rows if row.get("status") == "complete"]
    return {
        "schema_version": "pmo_diagnostic_rung_task_result_v1",
        "task": task_name,
        "charged_oracle_calls": len(ledger.rows),
        "authorized_charged_calls": charged_calls_per_task,
        "best_score": max(values) if values else None,
        # Primary readout.  AUC is reported beside it and is within-rung only.
        "best_at_k": best_at_checkpoints(ledger.rows),
        "auc_top10_at_budget": pmo_top_ten_auc(
            values, budget=charged_calls_per_task, finish=True
        ),
        "auc_budget": charged_calls_per_task,
        "auc_comparability": (
            "WITHIN-RUNG ONLY. pmo_top_ten_auc trapezoids up from (0,0), so a constant "
            "top-10 level c scores c*(1 - frequency/(2*budget)); at this budget that is a "
            "large structural discount. Never compare against a published 10000-call AUC."
        ),
        "diagnostics": task_diagnostics(folder / "campaign", ledger.rows),
        "campaign": campaign,
    }


# ---- per-task readouts -----------------------------------------------------


def best_at_checkpoints(rows: list[dict]) -> dict:
    """Best score after the first k charged calls, for each declared checkpoint.

    Charged-call index is the x-axis, not round index: the initialization block is
    charged before any round, and the contract declares calls including initialization.
    """
    out: dict[str, float | None] = {}
    for k in READOUT_CHECKPOINTS:
        seen = [
            row["score"]
            for row in rows[:k]
            if row.get("status") == "complete"
        ]
        out[f"best_at_{k}"] = max(seen) if seen else None
    return out


def task_diagnostics(campaign_folder: Path, rows: list[dict]) -> dict:
    """Attribution readouts, so a weak score says WHICH stage failed.

    Every field records its own coverage.  A provenance key that the batch did not
    emit yields an explicit `coverage` below 1.0 rather than a confident zero -- the
    difference between "no improvements by this family" and "this family was never
    labelled" is exactly what makes a weak score attributable.
    """
    complete = [row for row in rows if row.get("status") == "complete"]
    scores = [row["score"] for row in complete]
    provenance = _candidate_provenance(campaign_folder)

    improvements = _improvement_events(complete)
    improved_endpoints = {event["endpoint"] for event in improvements}

    by_family: dict[str, dict] = {}
    by_scale: dict[str, dict] = {}
    labelled = 0
    for row in complete:
        record = provenance.get(row["endpoint"])
        if record is None:
            continue
        labelled += 1
        family = record.get("planner_channel") or "unlabelled"
        bucket = by_family.setdefault(family, {"queried": 0, "improved": 0})
        bucket["queried"] += 1
        if row["endpoint"] in improved_endpoints:
            bucket["improved"] += 1
        scale = _edit_scale_bucket(record)
        cell = by_scale.setdefault(scale, {"queried": 0, "improved": 0})
        cell["queried"] += 1
        if row["endpoint"] in improved_endpoints:
            cell["improved"] += 1

    for table in (by_family, by_scale):
        for bucket in table.values():
            bucket["improvement_probability"] = (
                bucket["improved"] / bucket["queried"] if bucket["queried"] else None
            )

    return {
        "schema_version": "pmo_diagnostic_rung_task_diagnostics_v1",
        "charged_calls": len(rows),
        "completed_calls": len(complete),
        "nonzero_score_rate": (
            sum(1 for value in scores if value > 0.0) / len(scores) if scores else None
        ),
        "distinct_scores": len({round(value, 12) for value in scores}),
        "improvement_probability_by_program_family": by_family,
        "improvement_probability_by_edit_scale": by_scale,
        "provenance_coverage": (labelled / len(complete)) if complete else None,
        "productive_lineage_depth": _lineage_depth(improvements, provenance),
        "realization_refusal_causes": _refusal_causes(campaign_folder),
        "archive_churn": _archive_churn(campaign_folder),
        "successful_basin_diversity": _basin_diversity(complete, provenance, improvements),
    }


def _candidate_provenance(campaign_folder: Path) -> dict[str, dict]:
    """Map charged endpoint -> the provenance record of the candidate that proposed it."""
    out: dict[str, dict] = {}
    if not campaign_folder.is_dir():
        return out
    for pending in sorted(campaign_folder.glob("round_*/pending.json")):
        try:
            batch = json.loads(pending.read_text())["batch"]
        except (KeyError, json.JSONDecodeError):
            continue
        for candidate in batch.get("candidates", ()):
            record = dict(candidate.get("provenance") or {})
            endpoint = candidate.get("endpoint")
            # The realized primitive count lives on the trace, not on the provenance
            # record, and it is the only figure that reflects what the executor actually
            # applied rather than what the planner requested.
            trace = candidate.get("trace") or {}
            edits = trace.get("primitive_edits")
            if isinstance(edits, (list, tuple)):
                record["realized_primitive_edits"] = len(edits)
            if endpoint and endpoint not in out:
                out[endpoint] = record
    return out


def edit_scale_of(record: dict) -> int | None:
    """Realized structural edit scale of one candidate, in atoms touched.

    MEASURED field shapes, not assumed ones: `actual_changes` and `program_size` are
    both DICTS on real batches, so treating either as an int or a list silently buckets
    every candidate as "unlabelled" -- which is what a first version of this function
    did.  The scale used is the realized atom-level delta

        |changed_original_slots| + surviving_new_atoms + deleted_original_atoms

    i.e. what the executor actually applied.  `program_size.delta_heavy_atoms` is only a
    fallback: it is a NET heavy-atom change, so an edit that adds three atoms and
    deletes three reads as zero.
    """
    changes = record.get("actual_changes")
    if isinstance(changes, dict):
        slots = changes.get("changed_original_slots")
        touched = len(slots) if isinstance(slots, (list, tuple)) else 0
        new_atoms = changes.get("surviving_new_atoms")
        deleted = changes.get("deleted_original_atoms")
        total = touched
        total += new_atoms if isinstance(new_atoms, int) else 0
        total += deleted if isinstance(deleted, int) else 0
        return total
    if isinstance(changes, int):
        return changes
    if isinstance(changes, (list, tuple)):
        return len(changes)
    size = record.get("program_size")
    if isinstance(size, dict) and isinstance(size.get("delta_heavy_atoms"), int):
        return abs(size["delta_heavy_atoms"])
    if isinstance(size, int):
        return size
    return None


def _edit_scale_bucket(record: dict) -> str:
    """Bucket a candidate by realized edit scale."""
    value = edit_scale_of(record)
    if value is None:
        return "unlabelled"
    if value <= 2:
        return "micro_1_2"
    if value <= 5:
        return "small_3_5"
    if value <= 10:
        return "medium_6_10"
    return "large_11_plus"


def _improvement_events(complete: list[dict]) -> list[dict]:
    """Charged calls that strictly raised the running best score."""
    events, best = [], None
    for position, row in enumerate(complete, start=1):
        score = row["score"]
        if best is None or score > best:
            best = score
            events.append(
                {
                    "charged_call": position,
                    "endpoint": row["endpoint"],
                    "score": score,
                    "role": row.get("role"),
                }
            )
    return events


def _lineage_depth(improvements: list[dict], provenance: dict[str, dict]) -> dict:
    """How deep a productive chain the controller actually built.

    Depth is counted over IMPROVEMENT events that came from a proposed candidate, i.e.
    excluding the initialization block, because an initialization molecule has no
    program ancestry and counting it would report depth where no search happened.
    """
    from_search = [
        event for event in improvements
        if event.get("role") != "initialization" and event["endpoint"] in provenance
    ]
    depths = []
    for event in from_search:
        record = provenance.get(event["endpoint"], {})
        ancestry = record.get("metadata", {}).get("construction_ancestry")
        if isinstance(ancestry, (list, tuple)):
            depths.append(len(ancestry))
    return {
        "improvement_events_total": len(improvements),
        "improvement_events_from_initialization": sum(
            1 for event in improvements if event.get("role") == "initialization"
        ),
        "improvement_events_from_search": len(from_search),
        "first_search_improvement_at_charged_call": (
            from_search[0]["charged_call"] if from_search else None
        ),
        "max_construction_ancestry_depth": max(depths) if depths else None,
        "ancestry_coverage_of_search_improvements": (
            len(depths) / len(from_search) if from_search else None
        ),
        "improvement_events": improvements,
    }


def _refusal_causes(campaign_folder: Path) -> dict:
    """Why proposed programs failed to become charged candidates.

    A task that never realizes anything and a task that realizes freely but never
    improves are different failures with different fixes; this is the field that
    separates them.
    """
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    attempts_total = 0
    if campaign_folder.is_dir():
        for pending in sorted(campaign_folder.glob("round_*/pending.json")):
            try:
                batch = json.loads(pending.read_text())["batch"]
            except (KeyError, json.JSONDecodeError):
                continue
            for attempt in batch.get("attempts", ()):
                attempts_total += 1
                statuses[str(attempt.get("status", "unknown"))] += 1
                reason = attempt.get("reason")
                if reason:
                    # Executor messages carry molecule-specific tails; keep the head so
                    # the histogram groups causes instead of listing every instance.
                    reasons[str(reason).split(":")[0][:120]] += 1
    eligible = statuses.get("eligible", 0)
    return {
        "attempts_total": attempts_total,
        "eligible": eligible,
        "realization_yield": (eligible / attempts_total) if attempts_total else None,
        "status_histogram": dict(statuses.most_common()),
        "rejection_reason_histogram": dict(reasons.most_common(20)),
    }


def _archive_churn(campaign_folder: Path) -> dict:
    """How much the archive turned over per round, from the committed round summaries."""
    sizes, utilities, rounds = [], [], 0
    if campaign_folder.is_dir():
        for complete_path in sorted(campaign_folder.glob("round_*/complete.json")):
            try:
                saved = json.loads(complete_path.read_text())
            except json.JSONDecodeError:
                continue
            rounds += 1
            summary = saved.get("summary", {})
            snapshot = saved.get("snapshot", {})
            entries = snapshot.get("entries")
            if isinstance(entries, (list, dict)):
                sizes.append(len(entries))
            if summary.get("top_k_utility") is not None:
                utilities.append(summary["top_k_utility"])
    deltas = [b - a for a, b in zip(sizes, sizes[1:])]
    improved = [b - a for a, b in zip(utilities, utilities[1:])]
    return {
        "rounds_completed": rounds,
        "archive_size_by_round": sizes,
        "archive_growth_by_round": deltas,
        "mean_archive_growth_per_round": (sum(deltas) / len(deltas)) if deltas else None,
        "top_k_utility_by_round": utilities,
        "rounds_with_top_k_gain": sum(1 for value in improved if value > 0),
        "rounds_measured_for_gain": len(improved),
    }


def _basin_diversity(
    complete: list[dict], provenance: dict[str, dict], improvements: list[dict]
) -> dict:
    """Structural spread of the molecules that actually scored well.

    Restricted to the top decile by score so it describes SUCCESSFUL basins rather than
    the whole charged set; Murcko scaffolds are the structural key.  Scaffold counting
    is done with RDKit at read time so nothing depends on a field the batch may omit.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    if not complete:
        return {"top_decile_count": 0, "distinct_scaffolds": 0, "scaffold_entropy_bits": None}
    ordered = sorted(complete, key=lambda row: row["score"], reverse=True)
    cut = max(1, len(ordered) // 10)
    top = ordered[:cut]
    scaffolds: Counter[str] = Counter()
    for row in top:
        molecule = Chem.MolFromSmiles(row["endpoint"])
        if molecule is None:
            continue
        try:
            scaffolds[MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)] += 1
        except Exception:  # noqa: BLE001 - a scaffold failure is a data point, not a crash
            scaffolds["unparseable_scaffold"] += 1
    total = sum(scaffolds.values())
    entropy = None
    if total:
        entropy = -sum(
            (count / total) * math.log2(count / total) for count in scaffolds.values()
        )
    improved_scaffolds = set()
    for event in improvements:
        molecule = Chem.MolFromSmiles(event["endpoint"])
        if molecule is not None:
            try:
                improved_scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=molecule))
            except Exception:  # noqa: BLE001
                pass
    families = {
        (provenance.get(row["endpoint"]) or {}).get("planner_channel", "unlabelled")
        for row in top
    }
    return {
        "top_decile_count": len(top),
        "top_decile_score_floor": top[-1]["score"] if top else None,
        "distinct_scaffolds": len(scaffolds),
        "scaffold_entropy_bits": entropy,
        "scaffold_histogram": dict(scaffolds.most_common(10)),
        "distinct_scaffolds_among_improvements": len(improved_scaffolds),
        "program_families_in_top_decile": sorted(families),
    }


__all__ = [
    "CONTRACT",
    "READOUT_CHECKPOINTS",
    "SCHEMA",
    "SPIN_UP_CALLS",
    "SUITE",
    "RungBudgetError",
    "asset_backed_plan",
    "best_at_checkpoints",
    "edit_scale_of",
    "execute_task",
    "load_contract",
    "rounds_for_budget",
    "runtime_protocol",
    "task_diagnostics",
    "validate_budget",
    "verify_suite_against_registry",
]
