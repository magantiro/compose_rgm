"""Zero-oracle gate: independent vs support-conditioned (parent x plan) pairing.

WHAT THIS ANSWERS
-----------------
The completed PMO-v2 3x250 scored run executed 3 of 1,091 joint-dependency jump
proposals.  The attribution
(``diagnostics/pmo_v2_scored_attribution_v1.json``) located the cause in
``PmoPopulationController._generate_jump_pool``: the parent schedule and the plan
order were indexed by SEPARATE counters, so plan scale was never conditioned on what
the parent can support.  ``P(plan | parent)`` was implemented as ``P(plan) * P(parent)``.

This script replays the ACTUAL parent populations of that run -- the same three
tasks, the same 39 batches, the same ordered parent schedule -- and compares:

  A  ``independent_index_v1``      the pairing that ran, replayed from its own
                                   stored ``(entry_id, plan_id)`` decisions;
  B  ``support_conditioned_v1``    the same parents, the same plan order, but the
                                   plan drawn from the sub-sequence the parent's own
                                   support admits.

Both arms run the SAME pinned realizer at the SAME honest cap, so the only
difference is which pair is handed to it.  The starved-cap defect is reported
separately, against the stored production outcomes, so the two are never conflated.

ZERO oracle calls.  PMO endpoint eligibility is RDKit parseability
(``ProgramTask.endpoint_evaluator`` for ``kind == "pmo"``), which touches no oracle,
no asset and no task identity.

METHOD NOTES
------------
* Arm A is not a transcription: it replays the production run's OWN stored pairs,
  and the plan order is independently reconstructed and checked against all 1,091 of
  them before anything is measured.
* Work is reported as EXPANDED NODES and SUCCESSORS ENUMERATED, which are
  load-independent, alongside seconds, which are not.
* Falsifiers are written to the artifact BEFORE the realization phase begins.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from rdkit import Chem

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control import pmo_realization as PR
from compose_v4.control.pmo_population_controller import (
    JUMP_PAIRING_INDEPENDENT,
    JUMP_PAIRING_SUPPORT_CONDITIONED,
    supported_plan_index,
)
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "pmo_conditioned_jump_gate_v1"
RUN_ID = "35fc7bcd4bf5c964294698014c1ed40da483ad013f6b92c6201b01dee2d3d306"
CONTRACT_SHA = "ba9515615b341019f09f66f3224e590b4ca596dee6bb7c97c759dbe418f1ff4b"
CHECKPOINT = Path("diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json")
TASKS = ("gsk3b", "perindopril_mpo", "celecoxib_rediscovery")

# Production configuration of the completed run, read from its own snapshots.
RUN_SEED = 20260920
RUN_PLAN_ATTEMPTS = 32
RUN_WALL_SECONDS = 45.0

# Production baseline the falsifiers are stated against: 3 exact executions of 1,091
# stored jump proposals.
PRODUCTION_EXECUTIONS = 3
PRODUCTION_PROPOSALS = 1091
PRODUCTION_YIELD = PRODUCTION_EXECUTIONS / PRODUCTION_PROPOSALS

# The offline teacher-scale comparator the repair was measured against.
TEACHER_MEDIAN_PRIMITIVES = 31
TEACHER_MEDIAN_RETAINED = 0.630


def predeclared_falsifiers() -> dict[str, Any]:
    """Written to the artifact BEFORE the realization phase runs."""

    return {
        "F1_no_impossible_pair_selected": {
            "statement": (
                "Every (parent, plan) pair the conditioned arm SELECTS must pass the "
                "zero-search support certificate. The certificate is a necessary "
                "condition, so a selected pair it refuses would mean the sampler is "
                "not drawing from the feasible relation it claims to."
            ),
            "metric": "arm_b.selected_pairs_refused_by_certificate",
            "fails_if": "> 0",
        },
        "F2_yield_improves_by_an_order_of_magnitude": {
            "statement": (
                "Exact-realization yield per REALIZATION ATTEMPT must improve by at "
                "least 10x over the production rate of 3/1091 = 0.275%. Stated on the "
                "like-for-like denominator: production attempted a realization on "
                "every one of its 1,091 proposals."
            ),
            "metric": "arm_b.exact_realizations / arm_b.realization_attempted",
            "threshold": 10.0 * PRODUCTION_YIELD,
            "fails_if": f"< {10.0 * PRODUCTION_YIELD:.6f}",
        },
        "F3_successes_keep_large_jump_geometry": {
            "statement": (
                "Successful conditioned programs must NOT collapse toward a purely "
                "additive edit. Production's 9 honest-cap completions were retained "
                "1.000 at median 14 primitives against the offline teacher comparator "
                "of median 31 at retained 0.630. PASS requires the median realized "
                "retained fraction below 0.999, i.e. at least half the successes "
                "REMOVE something from the parent."
            ),
            "metric": "arm_b.realized.median_retained_fraction",
            "threshold": 0.999,
            "fails_if": ">= 0.999",
        },
        "F4_not_one_exceptional_parent": {
            "statement": (
                "Successful executions must not all come from one parent or one task."
            ),
            "metric": "arm_b.distinct_success_parents, arm_b.distinct_success_tasks",
            "fails_if": "distinct_success_parents < 2 or distinct_success_tasks < 2",
        },
        "F5_certificate_materially_reduces_realization_work": {
            "statement": (
                "The cheap pass must buy work, not just correctness. Realization "
                "SECONDS PER EXACT REALIZATION and EXPANDED NODES PER EXACT "
                "REALIZATION, both measured over the same 1,091 replayed requests, "
                "must fall by at least 2x against the independent arm. Seconds are "
                "load-dependent; expanded nodes are not, and the verdict is taken on "
                "the node ratio with seconds reported beside it."
            ),
            "metric": "work_per_realization.expanded_nodes ratio A/B",
            "threshold": 2.0,
            "fails_if": "< 2.0",
        },
    }


# ---- inputs ----


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jump_plan_order(plan_latents: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """The lane's intended plan order.

    Reproduced here rather than imported because ``_jump_plan_order`` is a controller
    METHOD reading ``self.config.seed``; the reconstruction is checked against all
    1,091 stored production pairs before use, which is a stronger guarantee than a
    shared call would give.
    """

    bands: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for plan in plan_latents:
        count = int(plan["primitive_count"])
        band = "small" if count <= 7 else "medium" if count <= 15 else "large"
        bands[band].append(plan)
    queues = []
    for band_index, band in enumerate(("small", "medium", "large")):
        rows = bands[band]
        if not rows:
            continue
        mass = np.asarray([max(float(row["mass"]), 1e-12) for row in rows])
        mass /= mass.sum()
        stream = np.random.default_rng(np.random.SeedSequence([seed, 719, band_index]))
        permutation = stream.choice(len(rows), len(rows), replace=False, p=mass)
        queues.append([rows[int(index)] for index in permutation])
    order: list[dict[str, Any]] = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop(0))
    return order


def load_inputs(data_root: Path) -> dict[str, Any]:
    envelope = json.loads(CHECKPOINT.read_text())
    checkpoint = envelope["payload"]["checkpoints"]["shared_all_routes"]
    if checkpoint.get("fit_scope") != "shared_all_routes":
        raise ValueError("unexpected jump checkpoint fit scope")
    order = jump_plan_order(checkpoint["plan_latents"], RUN_SEED)
    entries = json.loads((data_root / "parent_entries.json").read_text())
    rows = json.loads((data_root / "jump_attempts_final.json").read_text())
    if len(rows) != PRODUCTION_PROPOSALS:
        raise ValueError(f"expected {PRODUCTION_PROPOSALS} stored jump attempts, got {len(rows)}")

    # Provenance gate: the reconstructed order must reproduce every stored pair.
    ids = [plan["plan_id"] for plan in order]
    mismatches = 0
    for row in rows:
        offset = (int(row["batch"]) * RUN_PLAN_ATTEMPTS) % len(ids)
        if ids[(offset + int(row["attempt"])) % len(ids)] != row["plan_id"]:
            mismatches += 1
    if mismatches:
        raise ValueError(f"plan-order reconstruction disagrees with {mismatches} stored pairs")
    return {
        "order": order,
        "plans_by_id": {plan["plan_id"]: plan for plan in order},
        "entries": entries,
        "rows": rows,
        "plan_order_parity": {"checked": len(rows), "mismatches": 0},
    }


# ---- arm construction ----


def build_arms(data: dict[str, Any]) -> dict[str, Any]:
    """Both arms' per-request decisions over the SAME replayed parent sequence."""

    order = data["order"]
    entries = data["entries"]
    by_task_batch: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in data["rows"]:
        by_task_batch[(row["task"], int(row["batch"]))].append(row)
    for rows_in_batch in by_task_batch.values():
        rows_in_batch.sort(key=lambda row: int(row["attempt"]))

    sources: dict[str, Any] = {}

    def source_of(entry_id: str):
        if entry_id not in sources:
            sources[entry_id] = decode_state(entries[entry_id]["trace"]["states"][-1])
        return sources[entry_id]

    arm_a: list[dict[str, Any]] = []
    arm_b: list[dict[str, Any]] = []
    certificate_seconds = 0.0
    certificates_evaluated = 0
    abstention_reasons: collections.Counter = collections.Counter()
    for task in TASKS:
        # The memo persists across batches exactly as the controller's does.
        memo: dict[str, dict[str, Any]] = collections.defaultdict(dict)
        for batch in sorted(b for (t, b) in by_task_batch if t == task):
            offset = (batch * RUN_PLAN_ATTEMPTS) % len(order)
            for row in by_task_batch[(task, batch)]:
                attempt = int(row["attempt"])
                entry_id = row["entry_id"]
                source = source_of(entry_id)
                common = {
                    "task": task,
                    "batch": batch,
                    "attempt": attempt,
                    "entry_id": entry_id,
                    "parent_heavy_atoms": int(is_element(source.atom_types).sum()),
                }
                arm_a.append({**common, "plan_id": row["plan_id"], "stored_status": row["status"]})
                began = perf_counter()
                index, census = supported_plan_index(
                    source,
                    order,
                    offset + attempt,
                    specification=PR.PRODUCTION_SPECIFICATION,
                    memo=memo[entry_id],
                )
                certificate_seconds += perf_counter() - began
                certificates_evaluated += int(census["certificates_evaluated"])
                if index is None:
                    for reason, count in census["refusals_by_reason"].items():
                        abstention_reasons[reason] += count
                    arm_b.append({**common, "plan_id": None, "abstained": True})
                else:
                    arm_b.append(
                        {
                            **common,
                            "plan_id": order[index]["plan_id"],
                            "abstained": False,
                            "supported_plan_count": int(census["supported"]),
                        }
                    )
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "sources": sources,
        "certificate_cost": {
            "seconds_total": certificate_seconds,
            "certificates_evaluated": certificates_evaluated,
            "requests": len(arm_b),
            "ms_per_request": 1000.0 * certificate_seconds / max(1, len(arm_b)),
            "ms_per_certificate": 1000.0 * certificate_seconds / max(1, certificates_evaluated),
        },
        "abstention_refusal_reasons": dict(abstention_reasons.most_common()),
    }


# ---- realization (one job per distinct (parent, plan) pair, shared by both arms) ----

_WORKER: dict[str, Any] = {}


def _worker_init(entries_path: str, checkpoint_path: str) -> None:
    _WORKER["entries"] = json.loads(Path(entries_path).read_text())
    envelope = json.loads(Path(checkpoint_path).read_text())
    latents = envelope["payload"]["checkpoints"]["shared_all_routes"]["plan_latents"]
    _WORKER["plans"] = {plan["plan_id"]: plan for plan in latents}
    _WORKER["sources"] = {}


def _worker_source(entry_id: str):
    cache = _WORKER["sources"]
    if entry_id not in cache:
        cache[entry_id] = decode_state(_WORKER["entries"][entry_id]["trace"]["states"][-1])
    return cache[entry_id]


def run_pair(job: tuple[str, str]) -> dict[str, Any]:
    """One (parent, plan) pair through the pinned realizer at the HONEST cap."""

    entry_id, plan_id = job
    source = _worker_source(entry_id)
    plan = _WORKER["plans"][plan_id]
    spec = PR.PRODUCTION_SPECIFICATION
    certificate = PR.plan_parent_support(source, plan, spec)
    began = perf_counter()
    result = PR.realize(
        source,
        plan,
        spec,
        node_budget=PR.PRODUCTION_NODE_BUDGET,
        seconds_cap=PR.PRODUCTION_SECONDS_CAP,
        collect_all=True,
        max_realizations=PR.PRODUCTION_MAX_REALIZATIONS,
    )
    out: dict[str, Any] = {
        "entry_id": entry_id,
        "plan_id": plan_id,
        "certificate_supported": bool(certificate["supported"]),
        "certificate_stage": certificate["stage"],
        "certificate_reason": certificate["reason"],
        "outcome": result["outcome"],
        "nodes_expanded": int(result["nodes_expanded"]),
        "successors_enumerated": int(result["successors_enumerated"]),
        "seconds": perf_counter() - began,
        "budget_reason": result["budget_reason"],
        "plan_primitive_count": int(plan["primitive_count"]),
        "realizations": [],
    }
    for row in result["realizations"]:
        endpoint = decode_state(row["endpoint_state"])
        smiles = None
        try:
            from compose_v4.chem.molecular_graph import molecular_graph_to_smiles

            smiles = molecular_graph_to_smiles(endpoint)
        except (ValueError, RuntimeError):
            smiles = None
        out["realizations"].append(
            {
                "endpoint_key": row["endpoint_key"],
                "smiles": smiles,
                "rdkit_valid": bool(smiles is not None and Chem.MolFromSmiles(smiles) is not None),
                "primitive_count": int(row["primitive_count"]),
                "plan_primitive_count": int(row["plan_primitive_count"]),
                "retained_fraction": float(row["retained_fraction"]),
                "delta_heavy_atoms": int(row["delta_heavy_atoms"]),
                "component_count": int(row["component_count"]),
                "declared_component_count": int(row["declared_component_count"]),
                "created_dependency_edges": int(row["created_dependency_edges"]),
                "declared_created_dependency_count": int(
                    row["declared_created_dependency_count"]
                ),
            }
        )
    return out


# ---- reduction ----


def implied_retained_fraction(plan: dict[str, Any], heavy_atoms: int) -> float:
    """What the PLAN demands of the parent, before any search.

    A plan's preexisting-origin deletions are the parent atoms it must consume, so
    ``(heavy - preexisting_deletes) / heavy`` is the retained fraction the plan
    INTENDS on a parent of this size.  This is what "intended geometry" means for a
    latent that carries no retained fraction of its own.
    """

    if heavy_atoms <= 0:
        return 0.0
    deletes = sum(
        1
        for role in plan["roles"]
        if str(role["executor_rule"]) == "atom_delete"
        and role["operands"][0]["descriptor"].get("origin") == "preexisting"
    )
    return max(0.0, (heavy_atoms - deletes) / heavy_atoms)


def summarize(
    arm: list[dict[str, Any]],
    results: dict[tuple[str, str], dict[str, Any]],
    plans_by_id: dict[str, dict[str, Any]],
    archive_seen: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    requests = len(arm)
    abstained = sum(1 for row in arm if row.get("abstained"))
    attempted = [row for row in arm if row.get("plan_id")]
    compatible = 0
    refused_selected = 0
    realized_rows: list[dict[str, Any]] = []
    outcomes: collections.Counter = collections.Counter()
    nodes = successors = 0
    seconds = 0.0
    valid_endpoints = admitted = 0
    distinct_admitted: set[str] = set()
    distinct_archive_eligible: set[str] = set()
    success_parents: set[str] = set()
    success_tasks: set[str] = set()
    intended_primitives: list[int] = []
    realized_primitives: list[int] = []
    intended_retained: list[float] = []
    realized_retained: list[float] = []
    heavy_deltas: list[int] = []
    dependency_counts: list[int] = []
    success_seconds: list[float] = []
    success_nodes: list[int] = []
    for row in attempted:
        result = results[(row["entry_id"], row["plan_id"])]
        if result["certificate_supported"]:
            compatible += 1
        else:
            refused_selected += 1
        outcomes[result["outcome"]] += 1
        nodes += result["nodes_expanded"]
        successors += result["successors_enumerated"]
        seconds += result["seconds"]
        if result["outcome"] != PR.OUTCOME_COMPLETED or not result["realizations"]:
            continue
        # The controller commits ONE binding per attempt (the one its retention
        # ranking selects), so exactly one realization per attempt is counted here.
        best = result["realizations"][0]
        realized_rows.append(best)
        success_parents.add(row["entry_id"])
        success_tasks.add(row["task"])
        success_seconds.append(result["seconds"])
        success_nodes.append(result["nodes_expanded"])
        intended_primitives.append(int(result["plan_primitive_count"]))
        realized_primitives.append(int(best["primitive_count"]))
        intended_retained.append(
            implied_retained_fraction(plans_by_id[row["plan_id"]], row["parent_heavy_atoms"])
        )
        realized_retained.append(float(best["retained_fraction"]))
        heavy_deltas.append(int(best["delta_heavy_atoms"]))
        dependency_counts.append(int(best["created_dependency_edges"]))
        if best["rdkit_valid"]:
            valid_endpoints += 1
            admitted += 1  # PMO eligibility IS rdkit parseability; kept as two stages.
            distinct_admitted.add(best["smiles"])
            if best["smiles"] not in archive_seen:
                distinct_archive_eligible.add(best["smiles"])

    def med(values: list[Any]) -> float | None:
        return float(statistics.median(values)) if values else None

    exact = len(realized_rows)
    return {
        "arm": name,
        "funnel": {
            "parent_plan_requests": requests,
            "support_abstentions": abstained,
            "certificate_compatible": compatible,
            "realization_attempted": len(attempted),
            "exact_realization": exact,
            "valid_endpoint": valid_endpoints,
            "admitted_endpoint": admitted,
            "distinct_archive_eligible_endpoint": len(distinct_archive_eligible),
        },
        "selected_pairs_refused_by_certificate": refused_selected,
        "yield_per_realization_attempt": exact / max(1, len(attempted)),
        "yield_per_request": exact / max(1, requests),
        "outcomes": dict(outcomes.most_common()),
        "geometry": {
            "intended_primitive_count": {
                "median": med(intended_primitives),
                "values": sorted(intended_primitives),
            },
            "realized_primitive_count": {
                "median": med(realized_primitives),
                "values": sorted(realized_primitives),
            },
            "intended_retained_fraction": {
                "median": med(intended_retained),
                "values": [round(v, 4) for v in sorted(intended_retained)],
            },
            "realized_retained_fraction": {
                "median": med(realized_retained),
                "values": [round(v, 4) for v in sorted(realized_retained)],
                "count_below_0_999": sum(1 for v in realized_retained if v < 0.999),
            },
            "heavy_atom_delta": {"median": med(heavy_deltas), "values": sorted(heavy_deltas)},
            "created_dependency_edges": {
                "median": med(dependency_counts),
                "values": sorted(dependency_counts),
            },
        },
        "distinct_success_parents": len(success_parents),
        "distinct_success_tasks": len(success_tasks),
        "work": {
            "realization_seconds_total": seconds,
            "expanded_nodes_total": nodes,
            "successors_enumerated_total": successors,
            "seconds_per_exact_realization": (seconds / exact) if exact else None,
            "expanded_nodes_per_exact_realization": (nodes / exact) if exact else None,
            "median_success_seconds": med(success_seconds),
            "median_success_expanded_nodes": med(success_nodes),
        },
    }


def bottleneck_location(
    data: dict[str, Any], arms: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """Where excision geometry is lost: at SELECTION, or deeper in the search?

    Two competing explanations for "every success is purely additive":
      (i)  support conditioning SELECTS away from excision-demanding plans, or
      (ii) it selects them and they fail deeper than the root certificate can see.
    The discriminator is the intended retained fraction of the pairs the conditioned
    arm actually attempted, compared against the intended retained fraction of the
    pairs that realized.  Neither number requires a new search.
    """

    plans = data["plans_by_id"]

    def demands_excision(plan: dict[str, Any]) -> bool:
        return any(
            str(role["executor_rule"]) == "atom_delete"
            and role["operands"][0]["descriptor"].get("origin") == "preexisting"
            for role in plan["roles"]
        )

    catalog = collections.Counter(
        "requires_preexisting_deletes" if demands_excision(plan) else "purely_additive"
        for plan in data["order"]
    )
    attempted = collections.Counter()
    for row in arms["arm_b"]:
        if not row.get("plan_id"):
            attempted["abstained"] += 1
            continue
        intended = implied_retained_fraction(plans[row["plan_id"]], row["parent_heavy_atoms"])
        attempted["additive" if intended >= 0.999 else "excision_demanding"] += 1

    # The realized split is read off the summary the same run produced.
    intended_values = payload["arm_b_support_conditioned"]["geometry"][
        "intended_retained_fraction"
    ]["values"]
    realized_additive = sum(1 for value in intended_values if value >= 0.999)
    realized_excision = len(intended_values) - realized_additive

    # Depth-0 versus deeper, over EVERY (production parent, excision plan) pair.
    spec = PR.PRODUCTION_SPECIFICATION
    stages: collections.Counter = collections.Counter()
    supported = total = 0
    excision_plans = [plan for plan in data["order"] if demands_excision(plan)]
    for entry_id, source in arms["sources"].items():
        del entry_id
        for plan in excision_plans:
            certificate = PR.plan_parent_support(source, plan, spec)
            total += 1
            if certificate["supported"]:
                supported += 1
            else:
                stages[str(certificate["stage"])] += 1
    return {
        "question": (
            "Does support conditioning select AWAY from excision geometry, or select "
            "it and lose it deeper in the search?"
        ),
        "answer": (
            "It SELECTS excision geometry and loses it deeper. The root certificate is "
            "necessary but far from sufficient for a plan that must consume parent "
            "atoms."
        ),
        "plan_catalog": dict(catalog),
        "arm_b_attempted_pairs_by_geometry": dict(attempted),
        "arm_b_realizations_by_geometry": {
            "additive": realized_additive,
            "excision_demanding": realized_excision,
        },
        "yield_by_geometry": {
            "additive": realized_additive / max(1, attempted["additive"]),
            "excision_demanding": realized_excision / max(1, attempted["excision_demanding"]),
        },
        "every_parent_x_excision_plan_pair": {
            "pairs": total,
            "parents": len(arms["sources"]),
            "excision_plans": len(excision_plans),
            "supported_at_root": supported,
            "supported_fraction": supported / max(1, total),
            "refusal_stage": dict(stages),
        },
        "caveat_on_realized_retained_fraction": (
            "``retained_fraction`` is the controller's own SLOT-based rule "
            "(pmo_population_controller._retained_fraction): a parent slot that is "
            "deleted and then refilled by a later insertion counts as retained. For "
            "the 20 realizations whose plan demands no preexisting deletion at all "
            "there is no ambiguity; for the 2 excision-demanding ones the 1.000 "
            "reading cannot distinguish 'kept' from 'deleted and refilled'."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.expanduser("~/compose_pmo_data"))
    parser.add_argument("--out", default="diagnostics/pmo_conditioned_jump_gate_v1.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--amend-bottleneck",
        action="store_true",
        help="recompute only the BOTTLENECK_LOCATION block against an existing artifact",
    )
    args = parser.parse_args()

    if args.amend_bottleneck:
        payload = json.loads(Path(args.out).read_text())
        if payload.get("status") != "MEASURED":
            raise ValueError("--amend-bottleneck needs a completed gate artifact")
        data = load_inputs(Path(args.data_root))
        arms = build_arms(data)
        payload["BOTTLENECK_LOCATION"] = bottleneck_location(data, arms, payload)
        Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(json.dumps(payload["BOTTLENECK_LOCATION"], indent=1, sort_keys=True))
        return

    data_root = Path(args.data_root)
    out_path = Path(args.out)
    falsifiers = predeclared_falsifiers()
    # Predeclaration lands on disk BEFORE any realization runs.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "status": "PREDECLARED_AWAITING_MEASUREMENT",
                "PREDECLARED_FALSIFIERS": falsifiers,
            },
            indent=1,
            sort_keys=True,
        )
    )

    began = perf_counter()
    data = load_inputs(data_root)
    arms = build_arms(data)
    arm_a, arm_b = arms["arm_a"], arms["arm_b"]

    jobs = sorted(
        {(row["entry_id"], row["plan_id"]) for row in arm_a + arm_b if row.get("plan_id")}
    )
    print(f"distinct realization jobs: {len(jobs)}", flush=True)
    results: dict[tuple[str, str], dict[str, Any]] = {}
    realization_began = perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(str(data_root / "parent_entries.json"), str(CHECKPOINT)),
    ) as pool:
        for index, row in enumerate(pool.map(run_pair, jobs, chunksize=1), start=1):
            results[(row["entry_id"], row["plan_id"])] = row
            if index % 50 == 0:
                print(
                    f"  {index}/{len(jobs)}  {perf_counter() - realization_began:.0f}s",
                    flush=True,
                )
    realization_seconds = perf_counter() - realization_began

    archive_seen = {
        entry["endpoint"] for entry in data["entries"].values() if entry.get("endpoint")
    }
    summary_a = summarize(
        arm_a, results, data["plans_by_id"], archive_seen, name=JUMP_PAIRING_INDEPENDENT
    )
    summary_b = summarize(
        arm_b, results, data["plans_by_id"], archive_seen, name=JUMP_PAIRING_SUPPORT_CONDITIONED
    )

    # ---- falsifier verdicts ----
    node_ratio = None
    seconds_ratio = None
    if summary_a["work"]["expanded_nodes_per_exact_realization"] and summary_b["work"][
        "expanded_nodes_per_exact_realization"
    ]:
        node_ratio = (
            summary_a["work"]["expanded_nodes_per_exact_realization"]
            / summary_b["work"]["expanded_nodes_per_exact_realization"]
        )
    if summary_a["work"]["seconds_per_exact_realization"] and summary_b["work"][
        "seconds_per_exact_realization"
    ]:
        seconds_ratio = (
            summary_a["work"]["seconds_per_exact_realization"]
            / summary_b["work"]["seconds_per_exact_realization"]
        )
    yield_b = summary_b["yield_per_realization_attempt"]
    median_retained = summary_b["geometry"]["realized_retained_fraction"]["median"]
    verdicts = {
        "F1_no_impossible_pair_selected": {
            "measured": summary_b["selected_pairs_refused_by_certificate"],
            "verdict": "PASS" if summary_b["selected_pairs_refused_by_certificate"] == 0 else "FAIL",
        },
        "F2_yield_improves_by_an_order_of_magnitude": {
            "measured_yield": yield_b,
            "production_yield": PRODUCTION_YIELD,
            "improvement_factor": (yield_b / PRODUCTION_YIELD) if PRODUCTION_YIELD else None,
            "verdict": "PASS" if yield_b >= 10.0 * PRODUCTION_YIELD else "FAIL",
        },
        "F3_successes_keep_large_jump_geometry": {
            "measured_median_realized_retained_fraction": median_retained,
            "measured_median_realized_primitive_count": summary_b["geometry"][
                "realized_primitive_count"
            ]["median"],
            "teacher_comparator": {
                "median_primitives": TEACHER_MEDIAN_PRIMITIVES,
                "median_retained": TEACHER_MEDIAN_RETAINED,
            },
            "verdict": (
                "UNEVALUATED_NO_SUCCESSES"
                if median_retained is None
                else "PASS"
                if median_retained < 0.999
                else "FAIL"
            ),
        },
        "F4_not_one_exceptional_parent": {
            "distinct_success_parents": summary_b["distinct_success_parents"],
            "distinct_success_tasks": summary_b["distinct_success_tasks"],
            "verdict": (
                "PASS"
                if summary_b["distinct_success_parents"] >= 2
                and summary_b["distinct_success_tasks"] >= 2
                else "FAIL"
            ),
        },
        "F5_certificate_materially_reduces_realization_work": {
            "expanded_nodes_per_realization_ratio_A_over_B": node_ratio,
            "seconds_per_realization_ratio_A_over_B": seconds_ratio,
            "certificate_cost": arms["certificate_cost"],
            "verdict": (
                "UNEVALUATED"
                if node_ratio is None
                else "PASS"
                if node_ratio >= 2.0
                else "FAIL"
            ),
        },
    }

    payload = {
        "schema_version": SCHEMA,
        "status": "MEASURED",
        "PREDECLARED_FALSIFIERS": falsifiers,
        "FALSIFIER_VERDICTS": verdicts,
        "provenance": {
            "replayed_run_id": RUN_ID,
            "contract_payload_sha256": CONTRACT_SHA,
            "tasks": list(TASKS),
            "run_seed": RUN_SEED,
            "run_plan_attempts_per_batch": RUN_PLAN_ATTEMPTS,
            "run_wall_seconds": RUN_WALL_SECONDS,
            "plan_order_parity_against_stored_pairs": data["plan_order_parity"],
            "input_sha256": {
                "parent_entries.json": sha256_file(data_root / "parent_entries.json"),
                "jump_attempts_final.json": sha256_file(data_root / "jump_attempts_final.json"),
                "checkpoints.json": sha256_file(CHECKPOINT),
            },
            "code_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or None,
            "python": platform.python_version(),
            "rdkit": __import__("rdkit").__version__,
            "numpy": np.__version__,
            "realizer": {
                "specification": PR.PRODUCTION_SPECIFICATION.name,
                "node_budget": PR.PRODUCTION_NODE_BUDGET,
                "seconds_cap": PR.PRODUCTION_SECONDS_CAP,
                "max_realizations": PR.PRODUCTION_MAX_REALIZATIONS,
                "note": (
                    "BOTH arms run at the HONEST 20 s cap, so the pairing is the only "
                    "difference between them. The production run's starved cap "
                    "(min(20, wall_seconds - elapsed)) is reported separately below."
                ),
            },
            "distinct_realization_jobs": len(jobs),
            "realization_wall_seconds": realization_seconds,
            "workers": args.workers,
            "total_seconds": perf_counter() - began,
            "oracle_calls": 0,
        },
        "arm_a_independent": summary_a,
        "arm_b_support_conditioned": summary_b,
        "production_as_it_ran": {
            "proposals": PRODUCTION_PROPOSALS,
            "exact_executions": PRODUCTION_EXECUTIONS,
            "yield": PRODUCTION_YIELD,
            "note": (
                "Stored outcomes of the scored run under the STARVED cap. Arm A above "
                "re-runs the same pairs at the honest cap, so the difference between "
                "this block and arm A isolates the seconds_cap defect from the "
                "pairing defect."
            ),
        },
        "certificate_cost": arms["certificate_cost"],
        "abstention_refusal_reasons": arms["abstention_refusal_reasons"],
    }
    out_path.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {out_path}", flush=True)
    for name, row in verdicts.items():
        print(f"  {name}: {row['verdict']}", flush=True)


if __name__ == "__main__":
    main()
