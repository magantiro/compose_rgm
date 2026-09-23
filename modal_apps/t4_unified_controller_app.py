"""The unified T4 controller: ONE engine, state routing, and the frozen ladder.

WHAT IS NEW HERE, AND WHY IT EXISTS
-----------------------------------
`compose_v4.control.t4_unified_routing` and
`compose_v4.control.frozen_proposal_escalation` were built, measured and tested
with NO production caller.  A mechanism with no production caller is inert, and
this repository has paid for that repeatedly -- the bridge region law behind an
opt-in keyword, the `exact_early_ring` scheduler, `allocation_priority` with zero
call sites, `donor_program` absent from the scored entry point's import closure,
and a completion law silently dropped by a drop-in replacement adapter.

This app is the production caller.  It differs from
`modal_apps/t4_integrated_route_fiber_parp1_app` in exactly three places:

1. `proposal_worker` runs ONE shared proposal unit
   (`compose_v4.experiments.t4_unified_proposal.proposal_unit`) for all five
   lanes, instead of inlining them.  The zero-oracle gate drives the same
   function, so the gate measures the controller the campaign executes rather
   than a transcription of it.  Two copies of a proposal law are two proposal
   laws.
2. The empty-pool branch calls
   `compose_v4.experiments.t4_unified_controller.routed_support_expansion`,
   which routes each parent to a rung-0 kernel FROM THE MOLECULAR STATE and then
   climbs the frozen, target-agnostic ladder.  The whole branch body is the
   module-level function `expand_support`, so the consumption probe drives the
   production path rather than a copy of it.
3. `_validate_task` resolves the whole unified contract policy, so a contract
   missing the routing, the frozen ladder or the declared
   `proposal.shallow.region_law` fails on the FIRST worker that reads it rather
   than on whichever one happens to need it first.

THREE GUARDED HOPS
------------------
  hop 1  contract -> objects   `resolve_unified_controller` in `_validate_task`.
  hop 2  objects  -> draw site `assert_state_routing_is_consumed`, run in
                               `run_cell` BEFORE the root docking call, against
                               this app's OWN `expand_support` with a fan-out
                               that raises if reached.  Costs nothing: the probe
                               router raises from `decision`, which happens
                               before any dispatch.
  hop 3  draw site -> terminal `assert_unified_expansion_is_consumed`, run
                               immediately before a cell publishes
                               `candidate_exhaustion`.  A round loop that
                               quietly returns to the bare hard stop therefore
                               cannot end a cell.

WHAT IT SPENDS
--------------
The expansion spends CPU, never oracle calls.  It returns proposal records that
are locked and docked through the UNCHANGED round path, so charged-call
accounting, the per-query receipts, the no-retry/no-replacement/no-backfill
policy and the budget ceiling are untouched.  Every round still publishes an
immutable candidate and query lock before any docking.

A ladder step is realised as PARALLEL REPLICATE workers at each lane's own
contract base draw count, never as one deeper worker: a proposal worker is
bounded by wall clock and 480 draws on a drug-like parent already costs minutes,
so deepening a worker would trade a bounded expansion for a timeout.

LAUNCH AUTHORITY
----------------
`_local_task` REFUSES unless the contract carries both
`scored_launch_authorized` and `modal_launch_authorized`.  The unified panel is a
22,500-call decision the owner takes; "not authorized" is executable here rather
than prose in a status string.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/compose")
APP_MODULE = "modal_apps/t4_unified_controller_app.py"
CONTRACT = os.environ.get(
    "COMPOSE_HELD_CONTRACT", "configs/t4_unified_controller_parp1_d06_v1.json"
)
# The per-protein leave-one-out route expert, read by the `route_complete_region`
# lane.
CHECKPOINT = os.environ.get(
    "COMPOSE_HELD_CHECKPOINT",
    "diagnostics/t4_held_target_distillation_quality_v1/parp1_checkpoint.json",
)
# The ONE shared retained-rewrite expert, read by the state-aware rung-0 lane. It
# is deliberately not per-protein: the same expert serves every cell, which is
# what makes the state-aware kernel a general capability rather than a rescue.
SHARED_RETAINED_CHECKPOINT = "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json"
VOLUME_NAME = os.environ.get(
    "COMPOSE_HELD_VOLUME", "compose-t4-unified-controller-parp1-v1"
)
OUTPUT = Path(os.environ.get("COMPOSE_HELD_OUTPUT", "/unified_parp1"))
MOOD = "https://raw.githubusercontent.com/SeulLee05/MOOD/main/scorer"
RECEPTOR_NAME = os.environ.get("COMPOSE_HELD_RECEPTOR_NAME", "parp1")
RECEPTOR_PATH = f"/opt/dock/receptors/{RECEPTOR_NAME}.pdbqt"
# The arm wrapper that set the COMPOSE_HELD_* environment above. It is PINNED in
# `runtime_inputs_sha256` and `_validate_task` re-hashes every pinned entry inside
# the container, so the wrapper has to be baked into the image too.
WRAPPER = os.environ.get(
    "COMPOSE_HELD_WRAPPER", "modal_apps/t4_unified_controller_parp1_app.py"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .apt_install("openbabel", "curl", "ca-certificates")
    .run_commands(
        "mkdir -p /opt/dock/receptors",
        f"curl --fail -sSL -o /opt/dock/qvina02 {MOOD}/qvina02",
        "chmod +x /opt/dock/qvina02",
        *[
            f"curl --fail -sSL -o /opt/dock/receptors/{name}.pdbqt {MOOD}/receptors/{name}.pdbqt"
            for name in ("parp1", "jak2", "braf", "5ht1b", "fa7")
        ],
    )
    .add_local_dir(
        ROOT / "src", str(REMOTE / "src"), copy=True, ignore=["**/__pycache__/**"]
    )
    .add_local_file(ROOT / CONTRACT, str(REMOTE / CONTRACT), copy=True)
    .add_local_file(ROOT / CHECKPOINT, str(REMOTE / CHECKPOINT), copy=True)
    .add_local_file(
        ROOT / SHARED_RETAINED_CHECKPOINT,
        str(REMOTE / SHARED_RETAINED_CHECKPOINT),
        copy=True,
    )
    .add_local_file(
        ROOT / "docs/GENMOL_T4_SEEDS.json",
        str(REMOTE / "docs/GENMOL_T4_SEEDS.json"),
        copy=True,
    )
    .add_local_file(ROOT / APP_MODULE, str(REMOTE / APP_MODULE), copy=True)
    .add_local_file(ROOT / WRAPPER, str(REMOTE / WRAPPER), copy=True)
    .env(
        {
            "PYTHONPATH": str(REMOTE / "src"),
            "COMPOSE_HELD_CONTRACT": CONTRACT,
            "COMPOSE_HELD_CHECKPOINT": CHECKPOINT,
            "COMPOSE_HELD_VOLUME": VOLUME_NAME,
            "COMPOSE_HELD_OUTPUT": str(OUTPUT),
            "COMPOSE_HELD_RECEPTOR_NAME": RECEPTOR_NAME,
            "COMPOSE_HELD_WRAPPER": WRAPPER,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(os.environ.get("COMPOSE_HELD_APP", "compose-t4-unified-controller-parp1-v1"))
common = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 4096,
    "retries": 0,
    "volumes": {str(OUTPUT): volume},
    "scaledown_window": 20,
}


def _load_contract() -> dict:
    from compose_v4.experiments.t4_matched_pilot import unseal

    return unseal(REMOTE / CONTRACT)


def _validate_task(task: dict, *, role: str) -> dict:
    """Verify the contract identity, every pinned runtime input, and hop 1.

    `resolve_unified_controller` is called on EVERY role, not only the cell
    driver, so a contract that declares routing the runtime cannot honour fails
    on the first worker that reads it rather than on whichever one happens to
    need it first. It resolves the router, the frozen ladder and the declared
    region configuration together, and raises naming whichever is wrong.
    """

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import sha256_file
    from compose_v4.experiments.t4_unified_controller import resolve_unified_controller

    contract = _load_contract()
    if task.get("contract_payload_sha256") != identity(contract):
        raise ValueError("unified controller contract identity mismatch")
    if role not in {"proposal", "dock", "cell", "driver", "status"}:
        raise ValueError(f"unknown unified runtime role {role!r}")
    for relative, expected in contract["runtime_inputs_sha256"].items():
        actual = sha256_file(REMOTE / relative)
        if actual != expected:
            raise ValueError(f"runtime input mismatch for {relative}: {actual}")
    resolve_unified_controller(contract)
    return contract


def _publish(path: Path, payload: dict) -> str:
    from compose_v4.control.docking_value import identity

    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    import json

    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)
    volume.commit()
    return envelope["payload_sha256"]


def _load_expert(relative: str):
    """A fitted route expert from a sealed checkpoint, hash verified first."""

    import json

    from compose_v4.control.docking_value import identity
    from compose_v4.control.route_distilled_goal_expert import RouteDistilledGoalExpert

    envelope = json.loads((REMOTE / relative).read_text())
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"route expert checkpoint payload hash mismatch: {relative}")
    return RouteDistilledGoalExpert.from_checkpoint(envelope["payload"]["expert"])


def _resume_state(folder: Path, task: dict, contract: dict) -> dict | None:
    """Restore a preempted cell from its last completed round, or None to start fresh.

    The checkpoint is published only after a round finishes, while that round's lock is
    published before any of its queries are docked. A lock above the last checkpointed
    round is therefore an interrupted round whose charged-call count is unknowable: its
    queries are debited in full and never re-docked, and the search resumes past it, so
    budget is forfeited rather than an ambiguous scored retry being issued.
    """
    import json

    from compose_v4.control.docking_value import identity

    checkpoint_path = folder / "checkpoint.json"
    locks = sorted(folder.glob("round_*_lock.json")) if folder.exists() else []
    if not checkpoint_path.exists():
        if locks:
            raise RuntimeError(
                "an unfinished query lock exists with no recoverable checkpoint; "
                "automatic or ambiguous scored retry is forbidden"
            )
        return None

    envelope = json.loads(checkpoint_path.read_text())
    payload = envelope["payload"]
    if identity(payload) != envelope["payload_sha256"]:
        raise RuntimeError(
            "refusing to resume a checkpoint whose payload hash does not verify"
        )
    accepted = {task["contract_payload_sha256"]}
    predecessor = (contract.get("resume_predecessor") or {}).get(
        "contract_payload_sha256"
    )
    if predecessor:
        accepted.add(predecessor)
    if payload["contract_payload_sha256"] not in accepted:
        raise RuntimeError(
            "refusing to resume a checkpoint sealed under a different contract"
        )

    completed = max((row["round"] for row in payload["rounds"]), default=0)
    charged = payload["charged_calls"]
    budget = payload["budget_remaining"]
    history = list(payload["history"])

    forfeited = 0
    resume_after = completed
    for lock_path in locks:
        index = int(lock_path.name.split("_")[1])
        if index <= completed:
            continue
        lock = json.loads(lock_path.read_text())["payload"]
        forfeited += len(lock.get("queries") or [])
        resume_after = max(resume_after, index)
        history.append({"round": index, "improved": False, "forfeited": True})
    charged += forfeited
    budget -= forfeited

    # A round lock records each query's parent and that parent's observed score, so a
    # checkpoint that a restart overwrote can have its lost archive entries rebuilt from
    # the locks alone. Where two locks disagree about one molecule the worse score is
    # kept, so recovery can never flatter the archive.
    archive = dict(payload["archive"])
    recovered = 0
    conflicts = []
    for lock_path in locks:
        lock = json.loads(lock_path.read_text())["payload"]
        for query in lock.get("queries") or []:
            parent, score = query.get("parent"), query.get("parent_score")
            if not isinstance(parent, str) or not isinstance(score, (int, float)):
                continue
            score = float(score)
            if parent not in archive:
                archive[parent] = score
                recovered += 1
            elif abs(archive[parent] - score) > 1e-9:
                conflicts.append(
                    {"smiles": parent, "kept": max(archive[parent], score)}
                )
                archive[parent] = max(archive[parent], score)

    return {
        "archive": archive,
        "recovered_archive_entries": recovered,
        "archive_score_conflicts": conflicts,
        "budget_remaining": max(budget, 0),
        "rounds_completed": resume_after,
        "history": history,
        "features": payload["features"],
        "improvements": payload["improvements"],
        "rounds": payload["rounds"],
        "charged_calls": charged,
        "rng_state": payload["rng_state"],
        "forfeited_calls": forfeited,
    }


def _jsonable(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# 1800 s, matching both production siblings (`t4_integrated_route_fiber_parp1_app`
# and `t4_5ht1b2_protonation_rescue_d06_app`). It is also the figure the frozen
# ladder's replicate rationale cites: a step is spent on MORE workers at each
# lane's base draw count rather than on a deeper one, so no worker's cost grows
# with the ladder.
@app.function(**common, max_containers=36, timeout=1800)
def proposal_worker(task: dict) -> dict:
    """One lane on one parent; no task-oracle access.

    The chemistry lives in `t4_unified_proposal.proposal_unit`, which the
    zero-oracle gate drives too. This worker supplies only what a remote fan-out
    knows that a serial local driver does not: a validated contract, a baked
    checkpoint path, and the region law's contract resolution.
    """

    import numpy as np

    from compose_v4.control.region_law_contract import region_law_for_proposal_lane
    from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
    from compose_v4.experiments.t4_unified_proposal import (
        STATE_AWARE_LANE,
        proposal_unit,
    )

    contract = _validate_task(task, role="proposal")
    expert = task["expert"]
    fiber = Fiber(task["original_seed"], contract["delta"], support=contract["support"])

    def _probe_draw(law, attempt):
        """One production proposal draw, used only to prove the law is consumed.

        It calls `expand` itself -- same function, same lane, same horizon -- so
        what the check verifies is the path this worker is about to run, not a
        transcription of it. Its RNG is a separate stream, so probing cannot
        perturb the scored draw below.
        """

        return expand(
            task["parent"],
            task["parent_score"],
            fiber,
            np.random.default_rng(task["proposal_seed"] + 1_000_003 * (attempt + 1)),
            draws=1,
            multi_region=False,
            horizon=contract["proposal"]["shallow"]["horizon"],
            proposal_lane="shallow",
            region_law=law,
        )

    # Resolve the contract's region-draw law and PROVE the runtime consumes it
    # before anything is charged. Called on every lane, because a field parked on
    # a lane that cannot honour it must fail on the worker that would otherwise
    # have ignored it. Absent field -> (None, {}) -> byte-identical.
    region_law, region_law_telemetry = region_law_for_proposal_lane(
        contract,
        lane=expert,
        delta=contract["delta"],
        reference_smiles=task["original_seed"],
        draw=_probe_draw,
    )

    # Two DIFFERENT fitted experts, and which one a lane gets is not a detail:
    # `route_complete_region` runs the per-protein leave-one-out expert, while the
    # state-aware rung-0 lane runs the ONE shared retained-rewrite expert that
    # serves every cell.
    route_expert = None
    if expert == "route_complete_region":
        route_expert = _load_expert(CHECKPOINT)
    elif expert == STATE_AWARE_LANE:
        route_expert = _load_expert(SHARED_RETAINED_CHECKPOINT)

    answer = proposal_unit(
        contract,
        task,
        fiber=fiber,
        route_expert=route_expert,
        region_law=region_law,
    )
    telemetry = dict(answer["telemetry"])
    telemetry.update(region_law_telemetry)
    telemetry["expansion_replicate"] = task.get("expansion_replicate")
    telemetry["routed_kernel"] = task.get("routed_kernel")
    return {
        **answer,
        "parent_score": task["parent_score"],
        "telemetry": telemetry,
        "eligible_unique": answer["eligible_records_returned"],
    }


@app.function(**common, max_containers=24, timeout=720)
def dock_worker(task: dict) -> dict:
    """One locked docking request, charged once, with no retry."""

    import time

    from compose_v4.experiments.continuation_profile import sha256_file
    from compose_v4.experiments.t4_docking_adapter import dock_t4

    contract = _validate_task(task, role="dock")
    physical = {
        "qvina02": sha256_file(Path("/opt/dock/qvina02")),
        "receptor": sha256_file(Path(RECEPTOR_PATH)),
    }
    if physical != contract["evaluator_sha256"]:
        raise ValueError(f"docking evaluator identity mismatch: {physical}")
    started = time.time()
    score = dock_t4(
        task["smiles"],
        task["query_id"],
        contract["docking_seed"],
        box={
            "coordinates": contract["docking_box"],
            "receptor": RECEPTOR_PATH,
        },
    )
    return {
        "query_id": task["query_id"],
        "smiles": task["smiles"],
        "score": score,
        "failure": None if score is not None else "oracle_no_score",
        "elapsed_seconds": time.time() - started,
        "evaluator_sha256": physical,
    }


def _remote_proposals(requests: list[dict]) -> list:
    """The production fan-out: one container per request, exceptions returned."""

    return list(proposal_worker.map(requests, order_outputs=True, return_exceptions=True))


def expand_support(
    task: dict,
    contract: dict,
    cell: dict,
    *,
    parents,
    archive,
    round_index: int,
    router=None,
    ladder=None,
    fan_out=None,
):
    """Returns ``(outcome, record)``. THE one empty-pool path.

    `select_batch` returns empty exactly when `candidates` is empty, so "this
    round proposed no eligible endpoint" and "this cell terminates at
    candidate_exhaustion" are THE SAME EVENT in the shipped loop. This function
    is what that event runs instead.

    It builds the two production fan-outs as closures and hands them to
    `routed_support_expansion`, which owns the routing and the frozen stopping
    rule. Nothing about WHICH rung runs, at what draw count, or when to stop is
    decided here.

    `router` and `ladder` exist only so the consumption probe can install a
    stand-in; `routed_support_expansion` records an injected object as such and
    `assert_unified_expansion_is_consumed` refuses a terminal result built from
    one, so a probe run can never be published as a campaign result. `fan_out` is
    the dispatcher, defaulting to the real `proposal_worker.map`, so the probe
    drives THIS function with a dispatcher that raises rather than a copy of it.

    Charges nothing.
    """

    from compose_v4.control.frozen_proposal_escalation import resolve_frozen_escalation
    from compose_v4.control.zero_support_fallback import fallback_seed
    from compose_v4.experiments.t4_unified_controller import (
        ROUTED_RUNG_ZERO_LANE,
        routed_support_expansion,
    )

    dispatch = fan_out if fan_out is not None else _remote_proposals
    lanes = (ladder if ladder is not None else resolve_frozen_escalation(contract)).lanes
    telemetry: list[dict] = []

    def _collect(requests):
        """Dispatch, log every answer, and return the (request, answer) survivors."""

        answers = dispatch(requests)
        survivors = []
        for request, answer in zip(requests, answers, strict=True):
            if isinstance(answer, Exception):
                telemetry.append(
                    {
                        "expert": request["expert"],
                        "parent": request["parent"],
                        "routed_kernel": request.get("routed_kernel"),
                        "expansion_replicate": request.get("expansion_replicate"),
                        "status": "failed",
                        "error": repr(answer),
                    }
                )
                continue
            telemetry.append({**answer, "records": None, "status": "complete"})
            survivors.append((request, answer))
        return survivors

    def rung_zero(assignments):
        """Rung 0: ONE request per routed assignment, at that parent's own lane.

        This is the whole point of the routing -- a neutral parent's compute goes
        to the region kernel and a charged parent's to the state-aware one, from
        the molecular state and never from a target identity.
        """

        requests = []
        for item in assignments:
            lane = item["lane"]
            parent_index = int(item["parent_index"])
            parent = item["parent"]
            if lane == ROUTED_RUNG_ZERO_LANE["region"]:
                seed = fallback_seed(
                    cell["controller_seed"],
                    round_index=round_index,
                    parent_index=parent_index,
                )
            else:
                # The state-aware lane's own seed, matching
                # `scripts/t4_support_restoration_gate.py::state_aware_kernel`.
                # It coincides with `fallback_seed(..., replicate=0)` by
                # construction, which cannot collide in practice because the
                # router sends each parent to exactly ONE rung-0 lane.
                seed = int(
                    cell["controller_seed"]
                    + 900_007
                    + 1_000_003 * round_index
                    + 10_007 * parent_index
                )
            requests.append(
                {
                    **task,
                    "expert": lane,
                    "parent": parent,
                    "parent_score": archive[parent],
                    "original_seed": cell["smiles"],
                    "routed_kernel": item["kernel"],
                    "proposal_seed": seed,
                }
            )
        produced: list[dict] = []
        work: dict = {}
        for _request, answer in _collect(requests):
            produced.extend(answer["records"])
            counters = answer.get("telemetry") or {}
            for block in ("zero_support_fallback_work", "protonation_aware_work"):
                for key, count in (counters.get(block) or {}).items():
                    if isinstance(count, bool) or not isinstance(count, (int, float)):
                        continue
                    name = f"{block}.{key}"
                    work[name] = work.get(name, 0) + count
        return produced, work

    def escalate(draws, attempt):
        """One ladder step, as PARALLEL REPLICATES at each lane's base draw count.

        Never one deeper worker: a proposal worker is bounded by a wall-clock
        timeout and 480 draws on a drug-like parent already costs minutes, so a
        step is spent on more workers rather than on a longer one. The
        700_000_003-per-attempt offset cannot collide with a normal round seed
        (`controller_seed + 1_000_003*round + 10_007*parent + 101*expert`) for
        any round count this budget admits.
        """

        requests = []
        for lane_index, lane in enumerate(lanes):
            base = int(contract["proposal"][lane]["draws"])
            replicates = max(1, -(-int(draws) // base))
            for replicate in range(replicates):
                for parent_index, parent in enumerate(parents):
                    requests.append(
                        {
                            **task,
                            "expert": lane,
                            "parent": parent,
                            "parent_score": archive[parent],
                            "original_seed": cell["smiles"],
                            "expansion_replicate": replicate,
                            "proposal_seed": int(
                                cell["controller_seed"]
                                + 700_000_003 * (attempt + 1)
                                + 13_000_003 * replicate
                                + 1_000_003 * round_index
                                + 10_007 * parent_index
                                + 101 * lane_index
                            ),
                        }
                    )
        produced: list[dict] = []
        for _request, answer in _collect(requests):
            produced.extend(answer["records"])
        return produced

    outcome, record = routed_support_expansion(
        contract=contract,
        parents=list(parents),
        rounds_completed=max(int(round_index) - 1, 0),
        rung_zero=rung_zero,
        escalate=escalate,
        already_seen=sorted(archive),
        router=router,
        ladder=ladder,
    )
    return outcome, {**record, "worker_telemetry": _jsonable(telemetry)}


@app.function(**common, max_containers=3, timeout=20 * 3600)
def run_cell(task: dict) -> dict:
    """Run one cell with round locks, a shared online value model, and hops 2 and 3."""

    import json
    import time

    import numpy as np

    from compose_v4.control.fiber_control import ProgramValue, SearchState
    from compose_v4.experiments.t4_fiber_campaign import Fiber
    from compose_v4.experiments.t4_integrated_route_fiber import (
        EXPERTS,
        attach_features,
        expert_census,
        merge_expert_pools,
        select_batch,
    )
    from compose_v4.experiments.t4_support_expansion import normalize_expansion_records
    from compose_v4.experiments.t4_unified_controller import (
        assert_state_routing_is_consumed,
        assert_unified_expansion_is_consumed,
    )

    contract = _validate_task(task, role="cell")
    cell = next(row for row in contract["cells"] if row["cell"] == task["cell"])
    folder = OUTPUT / task["run_id"] / task["cell"]
    final_path = folder / "result.json"
    if final_path.exists():
        return json.loads(final_path.read_text())["payload"]

    # ---- Hop 2: prove the declared routing is CONSUMED, before anything is charged ----
    # The probe router raises from `decision`, which `routed_support_expansion`
    # calls in its parents loop before any RNG is drawn and before any dispatch,
    # so a healthy wiring returns having spent no proposal work at all. The
    # dispatcher handed to `expand_support` RAISES if it is reached, so a broken
    # wiring cannot quietly fan out real containers instead of failing.
    #
    # It drives this app's OWN `expand_support`, not a transcription of it, which
    # is the only form of this check that cannot drift with the code.
    def _probe_route(probe_router, attempt):
        def _refuse(_requests):
            raise AssertionError(
                "the routing consumption probe reached a proposal fan-out; the "
                "router must be consulted before any rung-0 or ladder work is "
                "dispatched"
            )

        return expand_support(
            task,
            contract,
            cell,
            parents=[cell["smiles"]],
            archive={cell["smiles"]: 0.0},
            round_index=1,
            router=probe_router,
            fan_out=_refuse,
        )

    routing_probe_attempts = assert_state_routing_is_consumed(_probe_route)
    print(
        f"[{cell['cell']}] ROUTING CONSUMED attempts={routing_probe_attempts}",
        flush=True,
    )

    # ---- Resume ----
    # The checkpoint written at the end of every round carries the whole search state,
    # so a preempted cell continues instead of re-charging the oracle. A round lock is
    # published BEFORE its queries are docked, so a lock above the last checkpointed
    # round is an interrupted round whose charge count is unknowable: its queries are
    # debited in full and never re-docked, spending budget rather than risking an
    # ambiguous double charge.
    resumed = _resume_state(folder, task, contract)

    rng = np.random.default_rng(cell["controller_seed"])
    fiber = Fiber(cell["smiles"], contract["delta"], support=contract["support"])
    value = ProgramValue(penalty=contract["value_penalty"])
    if resumed is not None:
        state = SearchState(
            archive=dict(resumed["archive"]),
            budget=resumed["budget_remaining"],
            rounds=resumed["rounds_completed"],
            history=list(resumed["history"]),
        )
        features = [list(row) for row in resumed["features"]]
        improvements = list(resumed["improvements"])
        rounds = list(resumed["rounds"])
        charged = resumed["charged_calls"]
        rng.bit_generator.state = resumed["rng_state"]
        value.fit(features, improvements)
        previous = state.incumbent
        # The root docking answer belongs to the first attempt and is not carried in
        # the checkpoint, so a resumed cell has none. It is bound explicitly here
        # because every terminal result records `root_result`: without this a
        # resumed cell that later exhausts raises NameError instead of publishing,
        # which is precisely the path this arm is most likely to take.
        root_answer = None
        print(
            f"[{cell['cell']}] RESUME round={state.rounds} calls={charged} "
            f"budget={state.budget} best={previous:.2f} "
            f"forfeited_interrupted_calls={resumed['forfeited_calls']} "
            f"recovered_archive={resumed['recovered_archive_entries']}",
            flush=True,
        )
    else:
        state = SearchState(
            archive={}, budget=contract["charged_calls_per_cell"], rounds=0
        )
        features: list[list[float]] = []
        improvements: list[float] = []
        rounds: list[dict] = []
        charged = 0

        root_query = {
            "schema_version": "t4_unified_round_lock_v1",
            "cell": cell["cell"],
            "round": 0,
            "kind": "root",
            "queries": [
                {
                    "query_id": f"{task['run_id']}_{cell['cell']}_root",
                    "smiles": cell["smiles"],
                }
            ],
        }
        root_lock = _publish(folder / "round_000_lock.json", root_query)
        root_answer = dock_worker.remote(
            {
                **task,
                "query_id": root_query["queries"][0]["query_id"],
                "smiles": cell["smiles"],
            }
        )
        charged += 1
        state.budget -= 1
        if root_answer["score"] is None:
            result = {
                "schema_version": "t4_unified_controller_result_v1",
                "status": "root_oracle_failure",
                "cell": cell["cell"],
                "charged_calls": charged,
                "root_lock_sha256": root_lock,
                "root_result": root_answer,
                "state_routing_probe_attempts": routing_probe_attempts,
            }
            _publish(final_path, result)
            return result
        state.archive[cell["smiles"]] = float(root_answer["score"])
        previous = state.incumbent
        print(
            f"[{cell['cell']}] root call=1 score={previous:.2f} budget={state.budget}",
            flush=True,
        )
        # ---- Checkpoint the completed root ----
        # MEASURED FAILURE, run 4321b8b1: the root lock is published BEFORE the root
        # docking, and the first checkpoint was only written at the END of round one.
        # A preemption anywhere in between therefore left a lock with no checkpoint,
        # which `_resume_state` correctly refuses as an ambiguous scored retry -- and
        # Modal DOES restart a preempted container with the same input even at
        # `retries=0` ("Container terminated due to preemption. Your Function will be
        # restarted with the same input"), so the restart met its own root lock and
        # the cell died having charged one call and produced nothing.
        #
        # The support expansion widens that window from minutes to tens of minutes,
        # so this controller makes a pre-existing hazard into the likely outcome.
        #
        # Round 0 IS a completed round: its one call is charged, scored and recorded,
        # so checkpointing it admits no ambiguity -- nothing is re-docked and no
        # uncertain call becomes free. `_resume_state` skips locks whose index is at
        # or below the last checkpointed round, so the root lock is then correctly
        # read as complete rather than forfeited.
        _publish(
            folder / "checkpoint.json",
            {
                "schema_version": "t4_unified_controller_checkpoint_v1",
                "status": "running",
                "cell": cell["cell"],
                "contract_payload_sha256": task["contract_payload_sha256"],
                "charged_calls": charged,
                "budget_remaining": state.budget,
                "archive": dict(sorted(state.archive.items())),
                "features": features,
                "improvements": improvements,
                "history": state.history,
                "rounds": rounds,
                "rng_state": _jsonable(rng.bit_generator.state),
                "state_routing_probe_attempts": routing_probe_attempts,
            },
        )

    while state.budget > 0:
        round_index = state.rounds + 1
        started = time.time()
        parents = state.parents(
            limit=contract["parents"], rng=rng, explore=contract["parent_explore"]
        )
        requests = []
        for parent_index, parent in enumerate(parents):
            for expert_index, expert in enumerate(EXPERTS):
                requests.append(
                    {
                        **task,
                        "expert": expert,
                        "parent": parent,
                        "parent_score": state.archive[parent],
                        "original_seed": cell["smiles"],
                        "proposal_seed": int(
                            cell["controller_seed"]
                            + 1_000_003 * round_index
                            + 10_007 * parent_index
                            + 101 * expert_index
                        ),
                    }
                )
        worker_results = _remote_proposals(requests)
        pools = {expert: [] for expert in EXPERTS}
        worker_telemetry = []
        for request, answer in zip(requests, worker_results, strict=True):
            if isinstance(answer, Exception):
                worker_telemetry.append(
                    {
                        "expert": request["expert"],
                        "parent": request["parent"],
                        "status": "failed",
                        "error": repr(answer),
                    }
                )
                continue
            pools[answer["expert"]].extend(answer["records"])
            worker_telemetry.append({**answer, "records": None, "status": "complete"})

        merged = merge_expert_pools(pools)
        fresh = [row for row in merged if row["smiles"] not in state.archive]
        candidates = attach_features(fresh, state, fiber)
        room = min(contract["batch"], state.budget)
        selected = select_batch(
            candidates,
            value,
            state,
            rng,
            round_index=round_index,
            batch=room,
            exploration=min(contract["exploration"], room),
            expert_floor_rounds=contract["expert_floor_rounds"],
        )
        pool_census = expert_census(candidates)
        selected_census = expert_census(selected)

        # ---- The routed, frozen support expansion ----
        # This branch IS the cell's terminal condition in the shipped loop, so a
        # cell whose eligible yield is low but positive died on one unlucky draw
        # with its budget unspent. Here it first routes each parent to a rung-0
        # kernel from the MOLECULAR STATE and climbs the frozen ladder, charging
        # nothing; only an expansion that ran to its declared end without
        # producing an eligible endpoint publishes candidate exhaustion.
        #
        # Structural consequence worth stating: every round that wrote a lock had
        # at least one eligible candidate and so never entered this branch, which
        # is why the expansion cannot perturb a healthy round. No RNG is drawn and
        # no state is mutated before the check.
        expansion = None
        expansion_record = None
        if not selected:
            expansion_started = time.time()
            expansion, expansion_record = expand_support(
                task,
                contract,
                cell,
                parents=parents,
                archive=state.archive,
                round_index=round_index,
            )
            print(
                f"[{cell['cell']}] SUPPORT EXPANSION round={round_index} "
                f"stop={expansion.stop_reason} "
                f"eligible={expansion.distinct_eligible} "
                f"fallback={expansion.fallback_eligible} "
                f"attempts={expansion.attempts} draws={expansion.draws_spent} "
                f"kernels={expansion_record['routed_kernels']} "
                f"lanes={expansion_record['routed_lanes']} "
                f"{time.time() - expansion_started:.0f}s",
                flush=True,
            )
            # The normal path gets `proposal_experts` from `merge_expert_pools`,
            # which the expansion path does not use because it deduplicates by
            # endpoint itself. Without this the round dies building its query rows
            # AFTER the expansion has already found its endpoints -- MEASURED, run
            # 727d9db5, which reached its target with four eligible and discarded
            # all four with `KeyError('proposal_experts')`.
            fresh = normalize_expansion_records(
                row for row in expansion.records if row["smiles"] not in state.archive
            )
            candidates = attach_features(fresh, state, fiber)
            selected = select_batch(
                candidates,
                value,
                state,
                rng,
                round_index=round_index,
                batch=room,
                exploration=min(contract["exploration"], room),
                expert_floor_rounds=contract["expert_floor_rounds"],
            )
            pool_census = expert_census(candidates)
            selected_census = expert_census(selected)

        if not selected:
            # ---- Hop 3 ----
            # Candidate exhaustion can only be published when THIS module's routed
            # expansion produced the record, the record names the frozen ladder and
            # the frozen router, the routing was resolved from the CONTRACT rather
            # than injected by a probe, and every parent carries a routing decision
            # naming a registered kernel. A round loop that quietly returned to the
            # bare hard stop therefore cannot end a cell.
            assert_unified_expansion_is_consumed(expansion_record, expansion)
            result = {
                "schema_version": "t4_unified_controller_result_v1",
                "status": "candidate_exhaustion",
                "contract_payload_sha256": task["contract_payload_sha256"],
                "cell": cell["cell"],
                "charged_calls": charged,
                "root_result": root_answer,
                "rounds": rounds,
                "archive": dict(sorted(state.archive.items())),
                "final_best": state.incumbent,
                "state_routing_probe_attempts": routing_probe_attempts,
                "unified_expansion_record": _jsonable(expansion_record),
                "support_expansion": expansion.as_record(),
                "claim_boundary": contract["claim_boundary"],
            }
            _publish(final_path, result)
            return result

        queries = []
        for index, row in enumerate(selected):
            queries.append(
                {
                    "query_id": (
                        f"{task['run_id']}_{cell['cell']}_r{round_index:03d}_q{index:02d}"
                    ),
                    "smiles": row["smiles"],
                    "selection_kind": row["selection_kind"],
                    "proposal_experts": row["proposal_experts"],
                    "parent": row["parent"],
                    "parent_score": row["parent_score"],
                }
            )
        lock_payload = {
            "schema_version": "t4_unified_round_lock_v1",
            "contract_payload_sha256": task["contract_payload_sha256"],
            "cell": cell["cell"],
            "round": round_index,
            "charged_before": charged,
            "parents": parents,
            "pool_census": pool_census,
            "selected_census": selected_census,
            "worker_telemetry": _jsonable(worker_telemetry),
            "unified_expansion_record": _jsonable(expansion_record),
            "support_expansion": (
                expansion.as_record() if expansion is not None else None
            ),
            "candidate_pool": _jsonable(candidates),
            "queries": queries,
        }
        lock_hash = _publish(folder / f"round_{round_index:03d}_lock.json", lock_payload)
        dock_tasks = [
            {**task, "query_id": query["query_id"], "smiles": query["smiles"]}
            for query in queries
        ]
        answers = list(
            dock_worker.map(dock_tasks, order_outputs=True, return_exceptions=True)
        )
        observed = []
        for row, query, answer in zip(selected, queries, answers, strict=True):
            if isinstance(answer, Exception):
                answer = {
                    "query_id": query["query_id"],
                    "smiles": query["smiles"],
                    "score": None,
                    "failure": repr(answer),
                }
            charged += 1
            state.budget -= 1
            scored = {**row, **answer}
            observed.append(scored)
            if answer["score"] is None:
                continue
            features.append(np.asarray(row["features"], dtype=float).tolist())
            improvements.append(float(row["parent_score"] - answer["score"]))
            state.archive[row["smiles"]] = float(answer["score"])
        value.fit(features, improvements)
        improved = state.incumbent < previous
        state.rounds = round_index
        state.history.append({"round": round_index, "improved": improved})
        previous = state.incumbent
        round_result = {
            "round": round_index,
            "charged_calls": charged,
            "charged_this_round": len(observed),
            "candidate_lock_payload_sha256": lock_hash,
            "pool_census": pool_census,
            "selected_census": selected_census,
            "selection_kind_census": {
                kind: sum(row["selection_kind"] == kind for row in selected)
                for kind in ("expert_floor", "model", "exploration")
            },
            "docked": _jsonable(observed),
            "round_best": min(
                (row["score"] for row in observed if row.get("score") is not None),
                default=None,
            ),
            "best_so_far": state.incumbent,
            "improved": improved,
            "value_training_rows": len(improvements),
            "support_expansion": (
                expansion.as_record() if expansion is not None else None
            ),
            "timing_seconds": {
                "round_total": time.time() - started,
                "proposal_max": max(
                    (row.get("elapsed_seconds", 0.0) for row in worker_telemetry),
                    default=0.0,
                ),
            },
            "rng_state_after": _jsonable(rng.bit_generator.state),
        }
        rounds.append(round_result)
        checkpoint = {
            "schema_version": "t4_unified_controller_checkpoint_v1",
            "status": "running",
            "cell": cell["cell"],
            "contract_payload_sha256": task["contract_payload_sha256"],
            "charged_calls": charged,
            "budget_remaining": state.budget,
            "archive": dict(sorted(state.archive.items())),
            "features": features,
            "improvements": improvements,
            "history": state.history,
            "rounds": rounds,
            "rng_state": _jsonable(rng.bit_generator.state),
            "state_routing_probe_attempts": routing_probe_attempts,
        }
        _publish(folder / "checkpoint.json", checkpoint)
        print(
            f"[{cell['cell']}] round={round_index} calls={charged} "
            f"best={state.incumbent:.2f} round_best={round_result['round_best']} "
            f"eligible={pool_census} selected={selected_census} "
            f"kinds={round_result['selection_kind_census']} "
            f"seconds={round_result['timing_seconds']['round_total']:.1f}",
            flush=True,
        )

    result = {
        "schema_version": "t4_unified_controller_result_v1",
        "status": "complete_budget",
        "contract_payload_sha256": task["contract_payload_sha256"],
        "cell": cell["cell"],
        "root_result": root_answer,
        "charged_calls": charged,
        "rounds": rounds,
        "archive": dict(sorted(state.archive.items())),
        "final_best": state.incumbent,
        "best_smiles": min(state.archive, key=state.archive.get),
        "training_rows": len(improvements),
        "new_oracle_calls": charged,
        "state_routing_probe_attempts": routing_probe_attempts,
        "claim_boundary": contract["claim_boundary"],
    }
    _publish(final_path, result)
    return result


@app.function(**common, max_containers=1, timeout=21 * 3600)
def drive(task: dict) -> dict:
    import json

    contract = _validate_task(task, role="driver")
    launch_path = OUTPUT / task["run_id"] / "launch.json"
    _publish(launch_path, task)
    tasks = [{**task, "cell": row["cell"]} for row in contract["cells"]]
    answers = list(run_cell.map(tasks, order_outputs=True, return_exceptions=True))
    records = []
    for row, answer in zip(contract["cells"], answers, strict=True):
        if isinstance(answer, Exception):
            records.append(
                {"cell": row["cell"], "status": "failed", "error": repr(answer)}
            )
        else:
            records.append(
                {
                    "cell": row["cell"],
                    "status": answer["status"],
                    "charged_calls": answer["charged_calls"],
                    "final_best": answer.get("final_best"),
                }
            )
    summary = {
        "schema_version": "t4_unified_controller_summary_v1",
        "run_id": task["run_id"],
        "contract_payload_sha256": task["contract_payload_sha256"],
        "records": records,
        "finished": True,
        "new_oracle_calls": sum(row.get("charged_calls", 0) for row in records),
        "claim_boundary": contract["claim_boundary"],
    }
    _publish(OUTPUT / task["run_id"] / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(**common, max_containers=1, timeout=120)
def remote_status(task: dict) -> dict:
    import json

    _validate_task(task, role="status")
    folder = OUTPUT / task["run_id"]
    volume.reload()
    cells = []
    contract = _load_contract()
    for cell in [row["cell"] for row in contract["cells"]]:
        result_path = folder / cell / "result.json"
        checkpoint_path = folder / cell / "checkpoint.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text())["payload"]
            cells.append(
                {
                    "cell": cell,
                    "status": payload["status"],
                    "calls": payload["charged_calls"],
                    "best": payload.get("final_best"),
                    "rounds": len(payload.get("rounds", [])),
                }
            )
        elif checkpoint_path.exists():
            payload = json.loads(checkpoint_path.read_text())["payload"]
            cells.append(
                {
                    "cell": cell,
                    "status": payload["status"],
                    "calls": payload["charged_calls"],
                    "best": min(payload["archive"].values()),
                    "rounds": len(payload["rounds"]),
                }
            )
        else:
            cells.append({"cell": cell, "status": "not_started", "calls": 0})
    return {"run_id": task["run_id"], "cells": cells}


def _local_task() -> dict:
    """The launch guard. Refuses an unauthorized, drifted or dirty launch.

    The authorization check is FIRST and is executable rather than prose: the
    unified panel is a 22,500-call decision the owner takes, so a contract whose
    `scored_launch_authorized` / `modal_launch_authorized` are false cannot reach
    `drive.spawn` at all.
    """

    import subprocess

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import sha256_file
    from compose_v4.experiments.t4_matched_pilot import unseal
    from compose_v4.experiments.t4_unified_controller import resolve_unified_controller

    contract = unseal(ROOT / CONTRACT)
    for field in ("scored_launch_authorized", "modal_launch_authorized"):
        if contract.get(field) is not True:
            raise ValueError(
                f"{CONTRACT} carries {field}={contract.get(field)!r} and "
                f"status={contract.get('status')!r}; the unified controller panel "
                "is not launchable without an explicit owner authorization that "
                "names this payload hash"
            )
    resolve_unified_controller(contract)
    for relative, expected in contract["runtime_inputs_sha256"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"local runtime input mismatch for {relative}: {actual}")
    runtime_paths = sorted({*contract["runtime_inputs_sha256"], CONTRACT})
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *runtime_paths], cwd=ROOT, check=True
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *runtime_paths],
        cwd=ROOT,
        text=True,
    )
    if untracked.strip():
        raise ValueError(f"untracked unified runtime inputs: {untracked}")
    body = {
        "schema_version": "t4_unified_controller_launch_v1",
        "contract_payload_sha256": identity(contract),
        "contract_file_sha256": sha256_file(ROOT / CONTRACT),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "automatic_retries": 0,
        "charged_call_ceiling": contract["total_charged_call_ceiling"],
    }
    return {**body, "run_id": identity(body)}


@app.local_entrypoint()
def main(mode: str = "launch", run_id: str = "") -> None:
    import json

    from compose_v4.experiments.t4_matched_pilot import seal

    if mode in ("launch", "resume"):
        task = _local_task()
        if mode == "resume":
            # run_id is content-addressed over code_revision, so a fixed launcher can
            # never recompute a prior run's id. Resuming therefore names it explicitly;
            # the contract hash is still verified against every checkpoint before any
            # state is restored.
            if not run_id:
                raise ValueError("mode=resume requires the prior run_id")
            task["run_id"] = run_id
        call = drive.spawn(task)
        receipt = {
            "schema_version": "t4_unified_controller_launch_receipt_v1",
            "mode": mode,
            "task": task,
            "function_call_id": call.object_id,
            "volume": VOLUME_NAME,
            "output_prefix": task["run_id"],
        }
        destination = (
            ROOT
            / "diagnostics/t4_unified_controller_v1/launches"
            / (task["run_id"] + ".json")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        seal(destination, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return
    if mode != "status" or not run_id:
        raise ValueError(
            "use mode=launch, mode=resume with run_id, or mode=status with run_id"
        )
    task = _local_task()
    task["run_id"] = run_id
    print(json.dumps(remote_status.remote(task), indent=2))
