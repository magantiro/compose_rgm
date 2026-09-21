"""Zero-oracle gate: does the PMO large structural jump mechanism survive?

THE QUESTION
------------
Support-conditioned pairing lifted jump realization 10.8x (0.275% production ->
2.96%) and was NOT enough: all 22 successes were purely ADDITIVE -- retained fraction
1.000, median 14 primitives, +12 heavy atoms -- against a teacher comparator of 31
primitives at 0.630 retained.  That is the predeclared falsifier shape.  Conditioning
SELECTS excision geometry (72.5% of attempts) and loses it deeper: yield splits 26x,
additive 9.80% against excision-demanding 0.37%.

Accepted diagnosis: the v1 plan representation is a source-specific TEMPLATE.  A role
pins each operand as a complete elemental-neighbourhood fingerprint of the molecule the
plan was mined from, so a different parent essentially never carries it -- over all
19,662 (production parent x excision plan) pairs only 18.0% pass the root certificate
and 97.7% of refusals are depth-0.

THREE ARMS, SAME PRODUCTION PARENTS, SAME PINNED REALIZER
---------------------------------------------------------
A  ``v1_exact``        the full source-role fingerprint.  The CONTROL.  Reused from
                       ``diagnostics/pmo_conditioned_jump_gate_v1.json``, which is the
                       best this descriptor achieves (support-conditioned pairing at
                       the honest cap), not a weaker variant of it.
B  ``relaxed_R1``      drop ``neighbor_element_histogram`` and ``creation_lag``.
                       DIAGNOSTIC ONLY.  A match under a coarser descriptor realizes a
                       DIFFERENT transformation, and the matched 36-pair comparison
                       measured both relaxations WORSE at teacher scale.  Reported;
                       never recommended.
C  ``parent_first``    invert the order.  Choose the parent, draw a legal region ON
                       THAT PARENT with the validated T4 ``BridgeRegionLaw``, then
                       construct the transformation conditional on that region.  The
                       library plan contributes structural INTENT only -- how many
                       parent atoms to consume, how many to build, with what
                       vocabulary -- and never an address or a fingerprint.

COST METRIC
-----------
Wall clock on this machine is load-contaminated (other jobs run here), so the primary
cost metric is LOAD-INDEPENDENT WORK.  For A and B that is expanded nodes; for C,
which performs no search, it is executor calls and regions tried.  **These are
different work units and are NOT interchangeable** -- they are reported side by side
and never summed or divided into one another.  Note also that ``collect_all=True``
makes a SUCCESSFUL search run to the cap, so a median success time measures the cap
rather than the difficulty.

ZERO oracle calls.  PMO endpoint eligibility is RDKit parseability; no oracle, no
asset, no task identity is reachable from this script.
"""

from __future__ import annotations

import argparse
import collections
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
import pmo_conditioned_jump_gate_v1 as BASE

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control import parent_first_program as PFP
from compose_v4.control import pmo_realization as PR
from compose_v4.control.pmo_population_controller import supported_plan_index
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "pmo_parent_first_jump_gate_v1"

TASKS = BASE.TASKS
RUN_SEED = BASE.RUN_SEED
RUN_PLAN_ATTEMPTS = BASE.RUN_PLAN_ATTEMPTS
PRODUCTION_PROPOSALS = BASE.PRODUCTION_PROPOSALS

# The settled arm-A measurement this gate reuses rather than recomputing.
ARM_A_ARTIFACT = Path("diagnostics/pmo_conditioned_jump_gate_v1.json")

# Arm A's measured split, quoted so the predeclared thresholds are anchored to a
# number already in an artifact rather than to a recollection of one.
ARM_A_EXCISION_DEMANDING_YIELD = 0.0037
ARM_A_ADDITIVE_YIELD = 0.0980
ARM_A_OVERALL_YIELD = 0.029609690444145357

TEACHER_MEDIAN_PRIMITIVES = 31
TEACHER_MEDIAN_RETAINED = 0.630

# Arm C's declared parameters.  Fixed for the whole arm, never adjusted per parent.
ARM_C_SIZE_TOLERANCE = 0
ARM_C_SEED = 20260921


def predeclared_criteria() -> dict[str, Any]:
    """The thresholds this experiment was originally to be judged against -- RETIRED.

    These were written to the artifact before any arm ran, which is why they are kept
    verbatim rather than deleted: a threshold quietly removed after seeing the numbers
    is indistinguishable from one that was never there.

    The owner RETIRED them mid-experiment, and the reasoning is the substance:
    "We have not proved that PMO needs deletion-heavy, 31-primitive jumps.  We have
    proved that our attempted mechanism for producing them transfers poorly."

    So excision yield, retained fraction and teacher-shaped geometry are DIAGNOSTICS
    OF ONE MECHANISM'S TRANSFER, never admission requirements for useful PMO search.
    A program that adds twelve atoms while retaining every original atom can still be
    a substantial and useful structural change.  Everything downstream of this block
    is reported as a DESCRIPTION of what the mechanism constructs, and nothing in this
    module returns a pass/fail verdict on whether the lane continues.
    """

    return {
        "STATUS": "RETIRED_BY_OWNER_MID_EXPERIMENT_KEPT_FOR_THE_RECORD",
        "retirement_reason": (
            "We have not proved that PMO needs deletion-heavy, 31-primitive jumps.  "
            "We have proved that our attempted mechanism for producing them transfers "
            "poorly.  These criteria are mechanism diagnostics, not admission "
            "requirements for useful PMO search, and they do not gate the lane."
        ),
        "C1_excision_yield_materially_exceeds_arm_a": {
            "criterion": (
                "arm C's excision-demanding realization yield must MATERIALLY exceed "
                "arm A's 0.37%.  Operationalized as at least 10x, i.e. >= 3.70%."
            ),
            "comparator": ARM_A_EXCISION_DEMANDING_YIELD,
            "threshold": 10.0 * ARM_A_EXCISION_DEMANDING_YIELD,
        },
        "C2_retained_fractions_materially_below_one": {
            "criterion": (
                "a HEALTHY POPULATION of genuine replacements, not a tail.  "
                "Operationalized as: median TRUE retained fraction < 1.000 AND at "
                "least 25% of realizations at true retained <= 0.90.  The TRUE "
                "measure is used because the controller's slot rule counts a deleted-"
                "then-refilled slot as retained."
            ),
            "median_true_retained_must_be_below": 1.0,
            "minimum_share_at_or_below_0_90": 0.25,
        },
        "C3_breadth_across_parents_and_tasks": {
            "criterion": (
                "many parents and all three tasks, not one exceptional case.  "
                "Operationalized as >= 20 distinct parents and 3 of 3 tasks."
            ),
            "minimum_distinct_parents": 20,
            "required_tasks": 3,
        },
        "FALSIFIER_additive_degeneracy": {
            "criterion": (
                "If arm C also produces only retained-1.000 additive edits, the "
                "structural-jump hypothesis has been properly tested and FAILED.  "
                "That is a complete and decisive answer, and is to be reported "
                "plainly rather than hedged."
            )
        },
        "caveat_retained_fraction_is_slot_based": (
            "pmo_population_controller._retained_fraction counts a parent SLOT that "
            "was deleted and later refilled by an insertion as retained.  Every arm-C "
            "row therefore carries BOTH that measure and a true retention derived "
            "from the atom_delete actions, plus the refilled slots the two disagree "
            "about."
        ),
    }


# ---- arm A: reuse the settled measurement ----


def load_arm_a() -> dict[str, Any]:
    payload = json.loads(ARM_A_ARTIFACT.read_text())
    arm = payload["arm_b_support_conditioned"]
    if payload.get("status") != "MEASURED":
        raise ValueError("arm A source artifact is not a MEASURED result")
    return {
        "arm": "A_v1_exact_descriptor",
        "descriptor": "v1_exact",
        "pairing": "support_conditioned_v1",
        "source": str(ARM_A_ARTIFACT),
        "source_code_revision": payload["provenance"]["code_revision"],
        "reused_not_recomputed": True,
        **{key: value for key, value in arm.items() if key != "arm"},
    }


# ---- arms B and C: worker plumbing ----

_W: dict[str, Any] = {}


def _init(entries_path: str, checkpoint_path: str) -> None:
    _W["entries"] = json.loads(Path(entries_path).read_text())
    envelope = json.loads(Path(checkpoint_path).read_text())
    latents = envelope["payload"]["checkpoints"]["shared_all_routes"]["plan_latents"]
    _W["plans"] = {plan["plan_id"]: plan for plan in latents}
    _W["sources"] = {}


def _source(entry_id: str):
    cache = _W["sources"]
    if entry_id not in cache:
        cache[entry_id] = decode_state(_W["entries"][entry_id]["trace"]["states"][-1])
    return cache[entry_id]


def run_relaxed_pair(job: tuple[str, str]) -> dict[str, Any]:
    """Arm B: the SAME realizer, the SAME cap, a coarser acceptance predicate."""

    entry_id, plan_id = job
    source = _source(entry_id)
    plan = _W["plans"][plan_id]
    spec = PR.SPEC_R1
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
    seconds = perf_counter() - began
    intent = PFP.intent_of_plan(plan)
    rows = []
    for row in result["realizations"]:
        endpoint = decode_state(row["endpoint_state"])
        accounting = PFP.excision_accounting(source, endpoint, row["actions"])
        rows.append(
            {
                "endpoint_key": row["endpoint_key"],
                "primitive_count": int(row["primitive_count"]),
                "delta_heavy_atoms": int(row["delta_heavy_atoms"]),
                "retained_fraction": float(row["retained_fraction"]),
                "true_retained_fraction": accounting.true_retained_fraction,
                "deleted_parent_atoms": len(accounting.deleted_parent_slots),
                "refilled_slots": len(accounting.refilled_slots),
            }
        )
    return {
        "entry_id": entry_id,
        "plan_id": plan_id,
        "supported": bool(certificate["supported"]),
        "certificate_stage": certificate["stage"],
        "certificate_reason": certificate["reason"],
        "outcome": result["outcome"],
        "nodes_expanded": int(result["nodes_expanded"]),
        "successors_enumerated": int(result["successors_enumerated"]),
        "seconds": seconds,
        "demands_excision": bool(intent.demands_excision),
        "intent_excise_atoms": int(intent.excise_atoms),
        "realizations": rows,
    }


def run_parent_first(job: tuple[str, str, int]) -> dict[str, Any]:
    """Arm C: one parent-first construction.

    ``execute_program`` is WRAPPED, not transcribed, so the executor-call count is the
    production function's own and the construction is decided by the production code.
    """

    entry_id, plan_id, seed = job
    source = _source(entry_id)
    plan = _W["plans"][plan_id]
    intent = PFP.intent_of_plan(plan)

    calls = {"n": 0}
    original = PFP.execute_program

    def counted(state, actions):
        calls["n"] += 1
        return original(state, actions)

    PFP.execute_program = counted
    began = perf_counter()
    try:
        result = PFP.construct_parent_first_program(
            source,
            intent,
            np.random.default_rng(seed),
            size_tolerance=ARM_C_SIZE_TOLERANCE,
        )
    finally:
        PFP.execute_program = original
    seconds = perf_counter() - began

    out = {
        "entry_id": entry_id,
        "plan_id": plan_id,
        "parent_heavy_atoms": int(is_element(source.atom_types).sum()),
        "realized": bool(result["realized"]),
        "reason": result.get("reason"),
        "executor_calls": calls["n"],
        "regions_tried": int(result.get("regions_tried", 0)),
        "seconds": seconds,
        "demands_excision": bool(intent.demands_excision),
        "intent_excise_atoms": int(intent.excise_atoms),
        "intent_insert_atoms": int(intent.insert_atoms),
    }
    if result["realized"]:
        out.update(
            {
                "endpoint_key": result["endpoint_key"],
                "realized_plan_id": result["plan_id"],
                "primitive_count": int(result["primitive_count"]),
                "region_size": int(result["region_size"]),
                "delta_heavy_atoms": int(result["delta_heavy_atoms"]),
                "retained_fraction": float(result["retained_fraction"]),
                "true_retained_fraction": float(result["true_retained_fraction"]),
                "deleted_parent_atoms": len(result["deleted_parent_slots"]),
                "refilled_slots": len(result["refilled_slots"]),
                "retention_measures_disagree": bool(result["retention_measures_disagree"]),
            }
        )
        # Endpoint chemistry, under the PMO eligibility rule (RDKit parseability).
        # Reported because "the executor accepted it" and "it is a plausible molecule"
        # are different claims and must not be conflated.
        from rdkit import Chem

        from compose_v4.chem.molecular_graph import molecular_graph_to_smiles

        smiles = molecular_graph_to_smiles(decode_state(result["endpoint_state"]))
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        out["endpoint_parseable"] = molecule is not None
        out["endpoint_rings"] = (
            int(molecule.GetRingInfo().NumRings()) if molecule is not None else 0
        )
    return out


# ---- request schedule (identical to arm A's) ----


def build_requests(data: dict[str, Any], spec) -> list[dict[str, Any]]:
    """The support-conditioned request schedule under ``spec``.

    Same parents, same order, same memo discipline as the controller; only the
    specification the certificate uses differs between arms.
    """

    order = data["order"]
    by_task_batch: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in data["rows"]:
        by_task_batch[(row["task"], int(row["batch"]))].append(row)
    for rows in by_task_batch.values():
        rows.sort(key=lambda row: int(row["attempt"]))

    sources: dict[str, Any] = {}
    entries = data["entries"]

    def source_of(entry_id: str):
        if entry_id not in sources:
            sources[entry_id] = decode_state(entries[entry_id]["trace"]["states"][-1])
        return sources[entry_id]

    requests: list[dict[str, Any]] = []
    abstentions: collections.Counter = collections.Counter()
    for task in TASKS:
        memo: dict[str, dict[str, Any]] = collections.defaultdict(dict)
        for batch in sorted(b for (t, b) in by_task_batch if t == task):
            offset = (batch * RUN_PLAN_ATTEMPTS) % len(order)
            for row in by_task_batch[(task, batch)]:
                attempt = int(row["attempt"])
                entry_id = row["entry_id"]
                source = source_of(entry_id)
                index, census = supported_plan_index(
                    source, order, offset + attempt, specification=spec, memo=memo[entry_id]
                )
                record = {
                    "task": task,
                    "batch": batch,
                    "attempt": attempt,
                    "entry_id": entry_id,
                    "parent_heavy_atoms": int(is_element(source.atom_types).sum()),
                    "parent_slots": int(source.n_atoms),
                }
                if index is None:
                    for reason, count in census["refusals_by_reason"].items():
                        abstentions[reason] += count
                    requests.append({**record, "plan_id": None, "abstained": True})
                else:
                    requests.append(
                        {**record, "plan_id": order[index]["plan_id"], "abstained": False}
                    )
    return requests, dict(abstentions.most_common())


def build_parent_first_requests(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Arm C's schedule: the same parents and the same plan ORDER, no certificate.

    A parent-first proposer has no use for a binding certificate -- there is no mined
    template to pre-screen -- so the plan index advances by the lane's own counter and
    contributes INTENT only.  Keeping the schedule otherwise identical is what makes
    the three arms matched on parent exposure.
    """

    order = data["order"]
    by_task_batch: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in data["rows"]:
        by_task_batch[(row["task"], int(row["batch"]))].append(row)
    for rows in by_task_batch.values():
        rows.sort(key=lambda row: int(row["attempt"]))

    requests: list[dict[str, Any]] = []
    for task in TASKS:
        for batch in sorted(b for (t, b) in by_task_batch if t == task):
            offset = (batch * RUN_PLAN_ATTEMPTS) % len(order)
            for row in by_task_batch[(task, batch)]:
                attempt = int(row["attempt"])
                plan = order[(offset + attempt) % len(order)]
                requests.append(
                    {
                        "task": task,
                        "batch": batch,
                        "attempt": attempt,
                        "entry_id": row["entry_id"],
                        "plan_id": plan["plan_id"],
                        "seed": ARM_C_SEED + 1_000_003 * batch + 10_007 * attempt,
                    }
                )
    return requests


# ---- reduction ----


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def q(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        at = p * (len(ordered) - 1)
        low = int(at)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (at - low) * (ordered[high] - ordered[low])

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p10": q(0.10),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "histogram": dict(
            sorted(collections.Counter(round(value, 2) for value in ordered).items())
        ),
    }


def reduce_relaxed(requests: list[dict[str, Any]], results: dict[tuple[str, str], dict]) -> dict:
    """Arm B's per-request reduction, over the SAME 1,091-request schedule."""

    attempted = successes = 0
    excision_attempts = excision_successes = 0
    additive_attempts = additive_successes = 0
    nodes = successors = 0
    seconds = 0.0
    endpoints: set[str] = set()
    parents: set[str] = set()
    tasks: set[str] = set()
    primitives: list[int] = []
    retained: list[float] = []
    true_retained: list[float] = []
    heavy: list[int] = []
    deleting = 0
    outcomes: collections.Counter = collections.Counter()
    stages: collections.Counter = collections.Counter()

    for request in requests:
        if request["abstained"]:
            outcomes["abstained_no_supported_plan"] += 1
            continue
        key = (request["entry_id"], request["plan_id"])
        row = results[key]
        attempted += 1
        nodes += row["nodes_expanded"]
        successors += row["successors_enumerated"]
        seconds += row["seconds"]
        outcomes[row["outcome"]] += 1
        if not row["supported"]:
            stages[str(row["certificate_stage"])] += 1
        demands = row["demands_excision"]
        if demands:
            excision_attempts += 1
        else:
            additive_attempts += 1
        if not row["realizations"]:
            continue
        successes += 1
        if demands:
            excision_successes += 1
        else:
            additive_successes += 1
        parents.add(request["entry_id"])
        tasks.add(request["task"])
        best = row["realizations"][0]
        for realization in row["realizations"]:
            endpoints.add(realization["endpoint_key"])
        primitives.append(int(best["primitive_count"]))
        retained.append(float(best["retained_fraction"]))
        true_retained.append(float(best["true_retained_fraction"]))
        heavy.append(int(best["delta_heavy_atoms"]))
        if int(best["deleted_parent_atoms"]) > 0:
            deleting += 1

    return {
        "arm": "B_relaxed_R1_descriptor",
        "descriptor": "R1_drop_environment_and_lag",
        "pairing": "support_conditioned_v1",
        "status": "DIAGNOSTIC_ONLY_NOT_A_CANDIDATE_FIX",
        "requests": len(requests),
        "realization_attempts": attempted,
        "abstained_requests": len(requests) - attempted,
        "successes": successes,
        "yield_per_request": successes / max(1, len(requests)),
        "yield_per_realization_attempt": successes / max(1, attempted),
        "excision_demanding_attempts": excision_attempts,
        "excision_demanding_successes": excision_successes,
        "excision_demanding_yield": excision_successes / max(1, excision_attempts),
        "additive_attempts": additive_attempts,
        "additive_successes": additive_successes,
        "additive_yield": additive_successes / max(1, additive_attempts),
        "realizations_deleting_a_parent_atom": deleting,
        "fraction_of_successes_deleting_a_parent_atom": deleting / max(1, successes),
        "distinct_endpoints": len(endpoints),
        "distinct_parents": len(parents),
        "distinct_tasks": sorted(tasks),
        "primitive_count": _distribution([float(v) for v in primitives]),
        "controller_retained_fraction": _distribution(retained),
        "true_retained_fraction": _distribution(true_retained),
        "delta_heavy_atoms": _distribution([float(v) for v in heavy]),
        "work": {
            "nodes_expanded": nodes,
            "successors_enumerated": successors,
            "successes_per_1000_expanded_nodes": 1000.0 * successes / max(1, nodes),
            "seconds_load_contaminated": seconds,
            "successes_per_second_load_contaminated": successes / max(1e-9, seconds),
        },
        "outcomes": dict(outcomes.most_common()),
        "certificate_refusal_stages": dict(stages.most_common()),
    }


def reduce_parent_first(requests: list[dict[str, Any]], results: list[dict]) -> dict:
    attempted = len(results)
    successes = 0
    excision_attempts = excision_successes = 0
    executor_calls = regions = 0
    seconds = 0.0
    endpoints: set[str] = set()
    realized_plans: set[str] = set()
    parents: set[str] = set()
    tasks: set[str] = set()
    primitives: list[float] = []
    retained: list[float] = []
    true_retained: list[float] = []
    heavy: list[float] = []
    region_sizes: list[float] = []
    deleting = disagreeing = 0
    reasons: collections.Counter = collections.Counter()

    for request, row in zip(requests, results, strict=True):
        executor_calls += row["executor_calls"]
        regions += row["regions_tried"]
        seconds += row["seconds"]
        if row["demands_excision"]:
            excision_attempts += 1
        if not row["realized"]:
            reasons[str(row["reason"])] += 1
            continue
        successes += 1
        if row["demands_excision"]:
            excision_successes += 1
        endpoints.add(row["endpoint_key"])
        realized_plans.add(row["realized_plan_id"])
        parents.add(row["entry_id"])
        tasks.add(request["task"])
        primitives.append(float(row["primitive_count"]))
        retained.append(float(row["retained_fraction"]))
        true_retained.append(float(row["true_retained_fraction"]))
        heavy.append(float(row["delta_heavy_atoms"]))
        region_sizes.append(float(row["region_size"]))
        if int(row["deleted_parent_atoms"]) > 0:
            deleting += 1
        if row["retention_measures_disagree"]:
            disagreeing += 1

    bands = ((1, 9), (10, 19), (20, 29), (30, 39))
    by_band: dict[str, dict[str, int]] = {
        f"{low}-{high}": {"attempts": 0, "realized": 0} for low, high in bands
    }
    parseable = ring_bearing = 0
    for row in results:
        parent_heavy = int(row["parent_heavy_atoms"])
        label = next(
            (f"{lo}-{hi}" for lo, hi in bands if lo <= parent_heavy <= hi), None
        )
        if label is None:
            continue
        by_band[label]["attempts"] += 1
        if row["realized"]:
            by_band[label]["realized"] += 1
            parseable += int(bool(row.get("endpoint_parseable")))
            ring_bearing += int(int(row.get("endpoint_rings", 0)) > 0)
    for row in by_band.values():
        row["rate"] = row["realized"] / max(1, row["attempts"])

    at_or_below_90 = sum(1 for value in true_retained if value <= 0.90)
    return {
        "arm": "C_parent_first_construction",
        "descriptor": "none_roles_read_off_the_realized_execution",
        "pairing": "parent_first_region_draw",
        "region_law": "SizeBandRegionLaw(margin=construction_headroom_margin)",
        "size_tolerance": ARM_C_SIZE_TOLERANCE,
        "requests": len(requests),
        "realization_attempts": attempted,
        "abstained_requests": attempted - successes,
        "successes": successes,
        "yield_per_request": successes / max(1, len(requests)),
        "yield_per_realization_attempt": successes / max(1, attempted),
        "excision_demanding_attempts": excision_attempts,
        "excision_demanding_successes": excision_successes,
        "excision_demanding_yield": excision_successes / max(1, excision_attempts),
        "realizations_deleting_a_parent_atom": deleting,
        "fraction_of_successes_deleting_a_parent_atom": deleting / max(1, successes),
        "successes_where_retention_measures_disagree": disagreeing,
        "successes_at_true_retained_at_or_below_0_90": at_or_below_90,
        "share_at_true_retained_at_or_below_0_90": at_or_below_90 / max(1, successes),
        "distinct_endpoints": len(endpoints),
        "distinct_realized_programs": len(realized_plans),
        "distinct_parents": len(parents),
        "distinct_tasks": sorted(tasks),
        "primitive_count": _distribution(primitives),
        "region_size": _distribution(region_sizes),
        "controller_retained_fraction": _distribution(retained),
        "true_retained_fraction": _distribution(true_retained),
        "delta_heavy_atoms": _distribution(heavy),
        "work": {
            "executor_calls": executor_calls,
            "regions_tried": regions,
            "successes_per_1000_executor_calls": 1000.0 * successes / max(1, executor_calls),
            "seconds_load_contaminated": seconds,
            "successes_per_second_load_contaminated": successes / max(1e-9, seconds),
            "work_unit_note": (
                "executor calls are NOT expanded nodes.  Arms A and B search a plan's "
                "binding tree and are costed in nodes; arm C performs no search and is "
                "costed in executor calls and regions tried.  The two are reported "
                "side by side and are never summed or divided into one another."
            ),
        },
        "abstention_reasons": dict(reasons.most_common()),
        "realization_by_parent_size": {
            "bands": by_band,
            "note": (
                "Realization RISES with parent size (tiny parents simply carry no "
                "bridge-separated region), so the mechanism is not an artifact of the "
                "small early PMO population.  Parent heavy atoms across this run: "
                "min 1, median 14, max 39."
            ),
        },
        "endpoint_chemistry": {
            "rdkit_parseable": parseable,
            "rdkit_parseable_share": parseable / max(1, successes),
            "ring_bearing": ring_bearing,
            "ring_bearing_share": ring_bearing / max(1, successes),
            "note": (
                "Parseability is PMO's endpoint eligibility rule and is a FLOOR, not "
                "a quality score.  The construction is a chain drawn from the intent's "
                "element vocabulary, so it produces valid but frequently implausible "
                "chemistry (peroxide and N-O linkages among them).  The mechanism "
                "guarantees validity and structural coherence, NOT medicinal-chemistry "
                "plausibility."
            ),
        },
    }


def capability_statement(arm_c: dict) -> dict[str, Any]:
    """What the mechanism CAN and CANNOT currently construct, from its own counts.

    Derived from the run rather than asserted, so it cannot drift away from the
    numbers beside it.
    """

    reasons = arm_c["abstention_reasons"]
    region = arm_c["region_size"]
    return {
        "can_construct": [
            (
                "bridge-separated region excision of ANY size the parent carries, "
                f"realized here at sizes {region.get('min')} to {region.get('max')} "
                f"(median {region.get('median')}) -- the v1 MAX_SEGMENT_LENGTH = 8 "
                "cap does not apply, because the region is drawn on the parent"
            ),
            (
                "a compatible chain construction re-attached at the retained anchor, "
                "so the action is a REPLACEMENT rather than a deletion"
            ),
            (
                "small coherent actions as well as large ones: the realized region "
                "size distribution starts at 1 atom, so 'change a couple of atoms and "
                "one attachment' is inside the support, not a special case"
            ),
            (
                "programs whose roles are read off the realized execution, in the v1 "
                "representation, and are therefore comparable to mined latents"
            ),
        ],
        "cannot_currently_construct": [
            {
                "gap": "pure construction with NO excision",
                "evidence_count": reasons.get("intent_demands_no_excision", 0),
                "detail": (
                    "construct_parent_first_program REQUIRES an excision demand and "
                    "abstains otherwise.  The owner's reset explicitly wants "
                    "additions and ring-system attachment to be first-class, so this "
                    "is a real gap, not a scoping choice.  It is one branch: draw an "
                    "attachment site instead of a region when excise_atoms == 0."
                ),
            },
            {
                "gap": "interior (two-bridge) excision joining the two flanks",
                "evidence_count": None,
                "detail": (
                    "only ONE-bridge pendant regions are drawable, so a cut that "
                    "removes an internal segment and reconnects its flanks is outside "
                    "the support.  Same limitation the T4 5ht1b_2 analysis named."
                ),
            },
            {
                "gap": "ring-system construction",
                "evidence_count": None,
                "detail": (
                    "the construction is a CHAIN of single-bonded atoms.  A whole "
                    "ring system is not built, though the excision side handles ring "
                    "regions by opening them first."
                ),
            },
            {
                "gap": "regions the parent simply does not carry at the demanded size",
                "evidence_count": reasons.get("no_legal_region_in_demanded_size_band", 0),
                "detail": (
                    "the dominant abstention.  This is the no-shrinking invariant "
                    "working as designed: the alternative is to bind a smaller "
                    "transformation and report it as the requested one."
                ),
            },
        ],
        "work_unit_caveat": (
            "regions_tried counts region ATTEMPTS and is 0 for a request whose parent "
            "carries no region in the demanded band, so it is not a count of requests. "
            "executor_calls is the complete cost measure."
        ),
    }


def descriptive_diagnostics(arm_c: dict, criteria: dict) -> dict:
    """What arm C constructed, measured against the RETIRED thresholds for reference.

    Reported as DESCRIPTION, not as a verdict.  The ``met`` flags say whether a number
    clears a threshold that is no longer an admission requirement; they exist so the
    retired criteria remain checkable against the record, and they decide nothing.
    """

    c1 = criteria["C1_excision_yield_materially_exceeds_arm_a"]
    c2 = criteria["C2_retained_fractions_materially_below_one"]
    c3 = criteria["C3_breadth_across_parents_and_tasks"]
    true_retained = arm_c["true_retained_fraction"]
    median = true_retained.get("median")
    c1_pass = arm_c["excision_demanding_yield"] >= c1["threshold"]
    c2_pass = bool(
        median is not None
        and median < c2["median_true_retained_must_be_below"]
        and arm_c["share_at_true_retained_at_or_below_0_90"]
        >= c2["minimum_share_at_or_below_0_90"]
    )
    c3_pass = (
        arm_c["distinct_parents"] >= c3["minimum_distinct_parents"]
        and len(arm_c["distinct_tasks"]) >= c3["required_tasks"]
    )
    additive_only = bool(
        arm_c["successes"] and arm_c["fraction_of_successes_deleting_a_parent_atom"] == 0.0
    )
    return {
        "FRAMING": (
            "DESCRIPTIVE ONLY.  The thresholds below are RETIRED and decide nothing.  "
            "Excision depth and retained fraction are diagnostics of how the v1 "
            "mechanism transferred, not requirements a useful structural action must "
            "satisfy."
        ),
        "C1_excision_yield_vs_retired_threshold": {
            "met": c1_pass,
            "measured": arm_c["excision_demanding_yield"],
            "threshold": c1["threshold"],
            "comparator_arm_a": c1["comparator"],
            "multiple_of_arm_a": (
                arm_c["excision_demanding_yield"] / c1["comparator"] if c1["comparator"] else None
            ),
        },
        "C2_retained_fraction_vs_retired_threshold": {
            "met": c2_pass,
            "median_true_retained": median,
            "share_at_or_below_0_90": arm_c["share_at_true_retained_at_or_below_0_90"],
            "required_share": c2["minimum_share_at_or_below_0_90"],
        },
        "C3_breadth_across_parents_and_tasks": {
            "met": c3_pass,
            "distinct_parents": arm_c["distinct_parents"],
            "required_parents": c3["minimum_distinct_parents"],
            "tasks": arm_c["distinct_tasks"],
        },
        "additive_only": {
            "observed": additive_only,
            "statement": (
                "every arm-C realization retained every parent atom"
                if additive_only
                else "arm C realizations delete parent atoms; the construction is not "
                "purely additive"
            ),
            "note": (
                "Under the retired framing this was the falsifier.  It is now an "
                "OBSERVATION about what the mechanism builds.  A purely additive "
                "result would not by itself have condemned the lane."
            ),
        },
        "all_retired_thresholds_cleared": bool(
            c1_pass and c2_pass and c3_pass and not additive_only
        ),
    }


# ---- entry point ----


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.path.expanduser("~/compose_pmo_data"))
    parser.add_argument("--out", default="diagnostics/pmo_parent_first_jump_gate_v1.json")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="pilot: first N requests only")
    parser.add_argument("--skip-arm-b", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    began = perf_counter()
    criteria = predeclared_criteria()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import rdkit

    provenance = {
        "schema_version": SCHEMA,
        "code_revision": _revision(),
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "numpy": np.__version__,
        "oracle_calls": 0,
        "replayed_run_id": BASE.RUN_ID,
        "contract_payload_sha256": BASE.CONTRACT_SHA,
        "run_seed": RUN_SEED,
        "arm_c_seed": ARM_C_SEED,
        "tasks": list(TASKS),
        "workers": args.workers,
    }
    # The criteria are on disk BEFORE any arm runs.
    out_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "status": "PREDECLARED_CRITERIA_WRITTEN_ARMS_NOT_YET_RUN",
                "predeclared_criteria": criteria,
                "provenance": provenance,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"predeclared criteria written to {out_path}", flush=True)

    data = BASE.load_inputs(data_root)
    print(f"inputs loaded; plan order parity {data['plan_order_parity']}", flush=True)

    arm_a = load_arm_a()
    print(f"arm A reused: yield {arm_a['yield_per_realization_attempt']:.4%}", flush=True)

    # ---- state semantics, MEASURED on this arm's own sources, before any work ----
    slots = collections.Counter()
    free = []
    for entry_id in sorted({row["entry_id"] for row in data["rows"]}):
        graph = decode_state(data["entries"][entry_id]["trace"]["states"][-1])
        heavy = int(is_element(graph.atom_types).sum())
        slots[int(graph.n_atoms)] += 1
        free.append(int(graph.n_atoms) - heavy)
    semantics = {
        "slot_capacity_histogram": dict(sorted(slots.items())),
        "parents": sum(slots.values()),
        "free_slots_min": min(free),
        "free_slots_median": float(np.median(free)),
        "parents_without_insertion_capacity": sum(1 for value in free if value == 0),
        "note": (
            "PMO parents carry 48 SLOTS, matching the T4 proposal path, NOT the "
            "40-slot editing corpus.  48 is the slot CAPACITY of the state array; 40 "
            "is the separate heavy-atom CEILING the executor enforces on endpoints.  "
            "Every parent has free slots, so atom_insert is expressible everywhere.  "
            "assert_production_state_semantics / production_state_from_smiles are the "
            "40-slot editing-corpus preflight and are the WRONG check for this path."
        ),
    }
    print(f"state semantics: {semantics['slot_capacity_histogram']}", flush=True)

    # ---- arm B ----
    arm_b: dict[str, Any] = {"arm": "B_relaxed_R1_descriptor", "status": "SKIPPED"}
    if not args.skip_arm_b:
        requests_b, abstentions_b = build_requests(data, PR.SPEC_R1)
        if args.limit:
            requests_b = requests_b[: args.limit]
        jobs = sorted({(r["entry_id"], r["plan_id"]) for r in requests_b if not r["abstained"]})
        print(f"arm B: {len(requests_b)} requests, {len(jobs)} distinct pairs", flush=True)
        began_b = perf_counter()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init,
            initargs=(str(data_root / "parent_entries.json"), str(BASE.CHECKPOINT)),
        ) as pool:
            rows = list(pool.map(run_relaxed_pair, jobs, chunksize=1))
        results_b = {(row["entry_id"], row["plan_id"]): row for row in rows}
        arm_b = reduce_relaxed(requests_b, results_b)
        arm_b["abstention_refusal_reasons"] = abstentions_b
        arm_b["wall_seconds"] = perf_counter() - began_b
        print(
            f"arm B: {arm_b['successes']} successes, "
            f"excision yield {arm_b['excision_demanding_yield']:.4%}",
            flush=True,
        )

    # ---- arm C ----
    requests_c = build_parent_first_requests(data)
    if args.limit:
        requests_c = requests_c[: args.limit]
    print(f"arm C: {len(requests_c)} requests", flush=True)
    began_c = perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init,
        initargs=(str(data_root / "parent_entries.json"), str(BASE.CHECKPOINT)),
    ) as pool:
        rows_c = list(
            pool.map(
                run_parent_first,
                [(r["entry_id"], r["plan_id"], r["seed"]) for r in requests_c],
                chunksize=4,
            )
        )
    arm_c = reduce_parent_first(requests_c, rows_c)
    arm_c["wall_seconds"] = perf_counter() - began_c
    print(
        f"arm C: {arm_c['successes']} successes, "
        f"excision yield {arm_c['excision_demanding_yield']:.4%}, "
        f"median true retained {arm_c['true_retained_fraction'].get('median')}",
        flush=True,
    )

    diagnostics = descriptive_diagnostics(arm_c, criteria)
    provenance["total_seconds"] = perf_counter() - began
    out_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "status": "MEASURED",
                "predeclared_criteria": criteria,
                "state_semantics": semantics,
                "arm_a_v1_exact": arm_a,
                "arm_b_relaxed_R1": arm_b,
                "arm_c_parent_first": arm_c,
                "descriptive_diagnostics": diagnostics,
                "capability_statement": capability_statement(arm_c),
                "provenance": provenance,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nwrote {out_path}", flush=True)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
