"""fa7_0 at delta=0.6: the integrated route/FiberControl policy with support expansion.

This is `t4_integrated_route_fiber_parp1_app` with ONE behavioural change, and the
change is the reason the arm exists.

The shipped round loop ends a cell the first time selection returns nothing::

    if not selected:
        result = {..., "status": "candidate_exhaustion", ...}
        return result

`select_batch` returns empty exactly when `candidates` is empty, so "this round
proposed no eligible endpoint" and "this cell is over" are THE SAME EVENT.  That
is what the blank `fa7_0` cell is: it terminated at `charged_calls=1` -- the seed's
own docking call -- with 247 authorized calls never issued.  Re-running that loop
with a different proposal law is a coin flip on round one; if the coin lands the
same way the whole budget is spent reproducing the termination.

Here that branch runs a BOUNDED SUPPORT EXPANSION first
(`compose_v4.experiments.t4_support_expansion`): the zero-support fallback, then an
escalating fan-out of the primary draw lanes.  Only an expansion that runs to its
declared end without producing an eligible endpoint publishes candidate exhaustion,
and the terminal record carries which bound stopped it.

Expansion spends CPU, never oracle calls.  It returns proposal records that are
locked and docked through the UNCHANGED round path, so charged-call accounting, the
per-query receipts, the no-retry/no-replacement/no-backfill policy and the budget
ceiling are all untouched.  Nothing here retries a query or replaces a receipt.

A ladder step is realised as PARALLEL replicate workers at the contract's own base
draw count rather than as one deeper worker, because a proposal worker is capped at
a 1800 s timeout and 480 draws on a drug-like parent already costs minutes.

Every round still publishes an immutable candidate and query lock before any docking.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/compose")
CONTRACT = os.environ.get(
    "COMPOSE_HELD_CONTRACT", "configs/t4_integrated_route_fiber_parp1_v1.json"
)
CHECKPOINT = os.environ.get(
    "COMPOSE_HELD_CHECKPOINT",
    "diagnostics/t4_integrated_route_fiber_parp1_v1/route_expert_checkpoint.json",
)
VOLUME_NAME = os.environ.get(
    "COMPOSE_HELD_VOLUME", "compose-t4-integrated-route-fiber-parp1-v1"
)
OUTPUT = Path(os.environ.get("COMPOSE_HELD_OUTPUT", "/integrated_parp1"))
MOOD = "https://raw.githubusercontent.com/SeulLee05/MOOD/main/scorer"
RECEPTOR_NAME = os.environ.get("COMPOSE_HELD_RECEPTOR_NAME", "parp1")
RECEPTOR_PATH = f"/opt/dock/receptors/{RECEPTOR_NAME}.pdbqt"
# The arm wrapper that set the COMPOSE_HELD_* environment above. It is PINNED in
# `runtime_inputs_sha256` -- it carries the launch guard that refuses an arm whose
# contract has lost a mechanism -- and `_validate_task` re-hashes every pinned entry
# inside the container, so the wrapper has to be in the image too. The parent arm
# left its wrapper unpinned and therefore unbaked; pinning it is the stronger choice
# and baking it is what makes that choice runnable.
WRAPPER = os.environ.get(
    "COMPOSE_HELD_WRAPPER", "modal_apps/t4_fa7_0_support_expansion_base_app.py"
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
    .add_local_dir(ROOT / "src", str(REMOTE / "src"), copy=True, ignore=["**/__pycache__/**"])
    .add_local_file(ROOT / CONTRACT, str(REMOTE / CONTRACT), copy=True)
    .add_local_file(ROOT / CHECKPOINT, str(REMOTE / CHECKPOINT), copy=True)
    .add_local_file(
        ROOT / "docs/GENMOL_T4_SEEDS.json",
        str(REMOTE / "docs/GENMOL_T4_SEEDS.json"),
        copy=True,
    )
    .add_local_file(
        ROOT / "modal_apps/t4_fa7_0_support_expansion_base_app.py",
        str(REMOTE / "modal_apps/t4_fa7_0_support_expansion_base_app.py"),
        copy=True,
    )
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
app = modal.App(
    os.environ.get("COMPOSE_HELD_APP", "compose-t4-integrated-route-fiber-parp1-v1")
)
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
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import sha256_file

    contract = _load_contract()
    if task.get("contract_payload_sha256") != identity(contract):
        raise ValueError("integrated controller contract identity mismatch")
    if role not in {"proposal", "dock", "cell", "driver", "status"}:
        raise ValueError(f"unknown integrated runtime role {role!r}")
    for relative, expected in contract["runtime_inputs_sha256"].items():
        actual = sha256_file(REMOTE / relative)
        if actual != expected:
            raise ValueError(f"runtime input mismatch for {relative}: {actual}")
    return contract


def _publish(path: Path, payload: dict) -> str:
    from compose_v4.control.docking_value import identity

    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    import json

    temporary.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)
    volume.commit()
    return envelope["payload_sha256"]


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
        raise RuntimeError("refusing to resume a checkpoint whose payload hash does not verify")
    # A checkpoint may carry the current contract, or the single predecessor the current
    # contract declares. The predecessor is admitted only because the re-seal tool proved
    # offline that the app's own hash is the one field that moved; every scientific field
    # is byte-identical. Any other ancestor is refused.
    accepted = {task["contract_payload_sha256"]}
    predecessor = (contract.get("resume_predecessor") or {}).get("contract_payload_sha256")
    if predecessor:
        accepted.add(predecessor)
    if payload["contract_payload_sha256"] not in accepted:
        raise RuntimeError("refusing to resume a checkpoint sealed under a different contract")

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
                conflicts.append({"smiles": parent, "kept": max(archive[parent], score)})
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


# 3600 s, not the parent arm's 1800. MEASURED under the pinned kernel: a shallow
# draw on this parent costs ~1.2 s with the completion law resolving 16 candidate
# completions per `segment_replace`, so a 480-draw worker is ~10 min on a fast
# core and a slower single-CPU container has no margin at 1800 s. The worker is
# bounded by its draw count, never open-ended.
@app.function(**common, max_containers=36, timeout=3600)
def proposal_worker(task: dict) -> dict:
    """One expert on one measured parent; no task-oracle access."""

    import json
    import time

    import numpy as np

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.completion_law_contract import completion_law_for_proposal_lane
    from compose_v4.control.docking_value import identity
    from compose_v4.control.region_law_contract import region_law_for_proposal_lane
    from compose_v4.control.route_distilled_goal_expert import (
        RouteDistilledGoalExpert,
        propose_route_expert_candidates,
    )
    from compose_v4.control.zero_support_fallback import fallback_candidates
    from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
    from compose_v4.experiments.t4_support_expansion import (
        ESCALATABLE_LANES,
        resolve_support_expansion,
    )

    contract = _validate_task(task, role="proposal")
    expert = task["expert"]
    started = time.time()
    fiber = Fiber(task["original_seed"], contract["delta"], support=contract["support"])
    telemetry = {}

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

    def _probe_completion_draw(law, attempt):
        """The same proof for the COMPLETION half of `segment_replace`.

        A separate RNG offset from `_probe_draw`, so the two consumption proofs
        cannot interfere with each other or with the scored draw below.
        """

        return expand(
            task["parent"],
            task["parent_score"],
            fiber,
            np.random.default_rng(task["proposal_seed"] + 2_000_003 * (attempt + 1)),
            draws=1,
            multi_region=False,
            horizon=contract["proposal"]["shallow"]["horizon"],
            proposal_lane="shallow",
            completion_law=law,
        )

    # Resolve the contract's region-draw law and PROVE the runtime consumes it
    # before anything is charged. Called on every expert, because a field parked
    # on a lane that cannot honour it must fail on the worker that would
    # otherwise have ignored it. Absent field -> (None, {}) -> byte-identical.
    region_law, region_law_telemetry = region_law_for_proposal_lane(
        contract,
        lane=expert,
        delta=contract["delta"],
        reference_smiles=task["original_seed"],
        draw=_probe_draw,
    )
    # The completion half of `segment_replace` drew blind -- a length uniform on
    # 1..8 and a uniform C/N/O chain -- while every measured eligible completion
    # inserts one or two atoms. Resolved and PROVEN CONSUMED exactly as the
    # region law is, on every expert, so a field parked on a lane that cannot
    # honour it fails on the worker that would otherwise have ignored it.
    completion_law, completion_law_telemetry = completion_law_for_proposal_lane(
        contract,
        lane=expert,
        delta=contract["delta"],
        reference_smiles=task["original_seed"],
        draw=_probe_completion_draw,
    )
    if expert == "zero_support_fallback":
        # The support-expansion stage that does NOT route through the
        # goal-abstraction layer: it excises each bridge-separated region with
        # the production executor and gates the EXECUTED endpoint directly. It
        # is reachable only from a support-expansion event, never from the
        # normal round, and the contract must authorize the stage.
        policy = resolve_support_expansion(contract)
        if policy is None or not policy.zero_support_fallback:
            raise ValueError(
                "the zero-support fallback ran without a contract authorizing it"
            )
        proposed, work = fallback_candidates(
            task["parent"],
            np.random.default_rng(task["proposal_seed"]),
            check=fiber.check,
            reference_smiles=task["original_seed"],
            delta=contract["delta"],
        )
        # `proposal_lane` is cleared and the true stage moved to its own key.
        # `t4_integrated_route_fiber._experts` validates a record's lane against the
        # frozen expert vocabulary and RAISES on an unknown one, so a record carrying
        # "zero_support_fallback" would kill `attach_features` at precisely the moment
        # the fallback first succeeded. Widening the shared vocabulary instead would
        # move `t4_integrated_route_fiber.py`, which other live arms pin.
        records = [
            {
                **row,
                "proposal_lane": None,
                "proposal_experts": [],
                "support_expansion_stage": "zero_support_fallback",
                "parent_score": task["parent_score"],
                "families": ("atom_delete",),
                "program_families": ("atom_delete",),
                "regions": 1,
                "created": int(row.get("inserted_atoms", 0)),
                "deleted": int(row.get("deleted_atoms", 0)),
            }
            for row in proposed
            if row["smiles"] != task["parent"]
        ]
        telemetry = {"zero_support_fallback_work": work.as_dict()}
    elif expert in ESCALATABLE_LANES:
        # A support-expansion replicate runs the SAME lane at the SAME base draw
        # count under a different seed. Escalation is realised as more replicate
        # workers, never as a deeper single worker: a proposal worker is capped
        # at 1800 s and 480 draws on a drug-like parent already costs minutes,
        # so deepening one worker would trade a bounded expansion for a timeout.
        draws = int(contract["proposal"][expert]["draws"])
        records = expand(
            task["parent"],
            task["parent_score"],
            fiber,
            np.random.default_rng(task["proposal_seed"]),
            draws=draws,
            multi_region=True,
            horizon=contract["proposal"][expert]["horizon"],
            proposal_lane=expert,
            region_law=region_law,
            completion_law=completion_law,
        )
        telemetry = {
            "raw_draws": draws,
            "expansion_replicate": task.get("expansion_replicate"),
        }
        telemetry.update(region_law_telemetry)
        telemetry.update(completion_law_telemetry)
    elif expert == "route_complete_region":
        envelope = json.loads((REMOTE / CHECKPOINT).read_text())
        if identity(envelope["payload"]) != envelope["payload_sha256"]:
            raise ValueError("route expert checkpoint payload hash mismatch")
        route = contract["proposal"][expert]
        model = RouteDistilledGoalExpert.from_checkpoint(envelope["payload"]["expert"])
        source = pad_molecular_graph(smiles_to_molecular_graph(task["parent"]), 48)
        proposed, telemetry = propose_route_expert_candidates(
            source,
            model,
            pool_size=route["pool_size"],
            realization_limit=route["realization_limit"],
            beam_width=route["beam_width"],
            expansion_width=route["expansion_width"],
            max_bindings_per_template=route["max_bindings_per_template"],
            maximum_expansions=route["maximum_expansions"],
        )
        records = []
        for row in proposed:
            properties = fiber.check(row["smiles"])
            if properties is None or properties["smiles"] == task["parent"]:
                continue
            records.append(
                {
                    **row,
                    **properties,
                    "parent": task["parent"],
                    "parent_score": task["parent_score"],
                    "delta": contract["delta"],
                }
            )
    else:
        raise ValueError(f"unknown proposal expert {expert!r}")
    return {
        "expert": expert,
        "parent": task["parent"],
        "parent_score": task["parent_score"],
        "proposal_seed": task["proposal_seed"],
        "records": records,
        "telemetry": telemetry,
        "eligible_unique": len(records),
        "elapsed_seconds": time.time() - started,
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


@app.function(**common, max_containers=3, timeout=20 * 3600)
def run_cell(task: dict) -> dict:
    """Run one PARP1 seed with round locks and a shared online value model."""

    import json
    import time

    import numpy as np

    from compose_v4.control.fiber_control import ProgramValue, SearchState
    from compose_v4.control.zero_support_fallback import fallback_seed
    from compose_v4.experiments.t4_fiber_campaign import Fiber
    from compose_v4.experiments.t4_integrated_route_fiber import (
        EXPERTS,
        attach_features,
        expert_census,
        merge_expert_pools,
        select_batch,
    )
    from compose_v4.experiments.t4_support_expansion import (
        assert_support_expansion_is_consumed,
        resolve_support_expansion,
        run_support_expansion,
    )

    contract = _validate_task(task, role="cell")
    cell = next(row for row in contract["cells"] if row["cell"] == task["cell"])
    folder = OUTPUT / task["run_id"] / task["cell"]
    final_path = folder / "result.json"
    if final_path.exists():
        return json.loads(final_path.read_text())["payload"]
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
        state = SearchState(archive={}, budget=contract["charged_calls_per_cell"], rounds=0)
        features: list[list[float]] = []
        improvements: list[float] = []
        rounds: list[dict] = []
        charged = 0

        root_query = {
            "schema_version": "t4_integrated_round_lock_v1",
            "cell": cell["cell"],
            "round": 0,
            "kind": "root",
            "queries": [
                {"query_id": f"{task['run_id']}_{cell['cell']}_root", "smiles": cell["smiles"]}
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
                "schema_version": "t4_integrated_route_fiber_result_v1",
                "status": "root_oracle_failure",
                "cell": cell["cell"],
                "charged_calls": charged,
                "root_lock_sha256": root_lock,
                "root_result": root_answer,
            }
            _publish(final_path, result)
            return result
        state.archive[cell["smiles"]] = float(root_answer["score"])
        previous = state.incumbent
        print(
            f"[{cell['cell']}] root call=1 score={previous:.2f} budget={state.budget}",
            flush=True,
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
        worker_results = list(
            proposal_worker.map(requests, order_outputs=True, return_exceptions=True)
        )
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

        # ---- Support expansion ----
        # `select_batch` returns empty exactly when `candidates` is empty, so this
        # branch IS the cell's terminal condition in the shipped loop. A cell whose
        # eligible yield is low but positive therefore died on one unlucky draw with
        # its budget unspent. Here the branch first runs a BOUNDED expansion that
        # charges nothing, and only an expansion which ran to its declared end
        # without producing an eligible endpoint publishes candidate exhaustion.
        #
        # Structural consequence worth stating: every round that wrote a lock had at
        # least one eligible candidate and so never entered this branch, which is why
        # the expansion cannot perturb a healthy round. No RNG is drawn and no state
        # is mutated before the check.
        expansion = None
        expansion_telemetry: list[dict] = []
        if not selected:
            policy = resolve_support_expansion(contract)
            if policy is None:
                raise ValueError(
                    "the candidate pool is empty and the contract declares no "
                    f"support_expansion block; this arm exists to expand support "
                    f"for {cell['cell']} rather than terminate on the first empty draw"
                )
            expansion_started = time.time()

            def _record(request, answer, *, log=expansion_telemetry):
                if isinstance(answer, Exception):
                    log.append(
                        {
                            "expert": request["expert"],
                            "parent": request["parent"],
                            "status": "failed",
                            "error": repr(answer),
                        }
                    )
                    return []
                log.append({**answer, "records": None, "status": "complete"})
                return answer["records"]

            def _fallback(*, parents=parents, round_index=round_index):
                """The structurally different stage: excise, execute, gate directly.

                It does not enter the goal-abstraction layer where `expand`
                abstracts a program to a structural goal and gates whatever
                `instantiate_goal` rebuilds -- the layer at which this cell's
                programs were measured to reach the gate with frequency 0.
                """

                requests = [
                    {
                        **task,
                        "expert": "zero_support_fallback",
                        "parent": parent,
                        "parent_score": state.archive[parent],
                        "original_seed": cell["smiles"],
                        "proposal_seed": fallback_seed(
                            cell["controller_seed"],
                            round_index=round_index,
                            parent_index=parent_index,
                        ),
                    }
                    for parent_index, parent in enumerate(parents)
                ]
                answers = list(
                    proposal_worker.map(requests, order_outputs=True, return_exceptions=True)
                )
                produced, work = [], {}
                for request, answer in zip(requests, answers, strict=True):
                    produced.extend(_record(request, answer))
                    if isinstance(answer, Exception):
                        continue
                    counters = (answer.get("telemetry") or {}).get(
                        "zero_support_fallback_work"
                    ) or {}
                    for key, count in counters.items():
                        work[key] = work.get(key, 0) + int(count)
                return produced, work

            def _escalate(draws, attempt, *, policy=policy, parents=parents,
                          round_index=round_index):
                """One ladder step, as PARALLEL replicates at the base draw count.

                A proposal worker is capped at 1800 s and 480 draws on a drug-like
                parent already costs minutes, so a step is spent on more workers
                rather than on a deeper one. Seeds carry a 700_000_003 offset per
                attempt, which cannot collide with a normal round seed
                (`controller_seed + 1_000_003*round + 10_007*parent + 101*expert`)
                for any round count this budget admits.
                """

                requests = []
                for lane_index, lane in enumerate(policy.lanes):
                    base = int(contract["proposal"][lane]["draws"])
                    replicates = max(1, -(-int(draws) // base))
                    for replicate in range(replicates):
                        for parent_index, parent in enumerate(parents):
                            requests.append(
                                {
                                    **task,
                                    "expert": lane,
                                    "parent": parent,
                                    "parent_score": state.archive[parent],
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
                answers = list(
                    proposal_worker.map(requests, order_outputs=True, return_exceptions=True)
                )
                produced = []
                for request, answer in zip(requests, answers, strict=True):
                    produced.extend(_record(request, answer))
                return produced

            expansion = run_support_expansion(
                policy,
                escalate=_escalate,
                fallback=_fallback,
                already_seen=sorted(state.archive),
            )
            print(
                f"[{cell['cell']}] SUPPORT EXPANSION round={round_index} "
                f"stop={expansion.stop_reason} eligible={expansion.distinct_eligible} "
                f"fallback={expansion.fallback_eligible} attempts={expansion.attempts} "
                f"draws={expansion.draws_spent} "
                f"{time.time() - expansion_started:.0f}s",
                flush=True,
            )
            fresh = [
                row for row in expansion.records if row["smiles"] not in state.archive
            ]
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
            # Publishing exhaustion is gated on the expansion having actually run,
            # so a contract that declares one and a runtime that quietly does not
            # perform it cannot end the cell.
            assert_support_expansion_is_consumed(expansion)
            result = {
                "schema_version": "t4_integrated_route_fiber_result_v1",
                "status": "candidate_exhaustion",
                "cell": cell["cell"],
                "charged_calls": charged,
                "root_result": root_answer,
                "rounds": rounds,
                "archive": dict(sorted(state.archive.items())),
                "final_best": state.incumbent,
                "support_expansion": expansion.as_record(),
                "support_expansion_policy": policy.as_record(),
            }
            _publish(final_path, result)
            return result

        queries = []
        for index, row in enumerate(selected):
            queries.append(
                {
                    "query_id": f"{task['run_id']}_{cell['cell']}_r{round_index:03d}_q{index:02d}",
                    "smiles": row["smiles"],
                    "selection_kind": row["selection_kind"],
                    "proposal_experts": row["proposal_experts"],
                    "parent": row["parent"],
                    "parent_score": row["parent_score"],
                }
            )
        lock_payload = {
            "schema_version": "t4_integrated_round_lock_v1",
            "contract_payload_sha256": task["contract_payload_sha256"],
            "cell": cell["cell"],
            "round": round_index,
            "charged_before": charged,
            "parents": parents,
            "pool_census": pool_census,
            "selected_census": selected_census,
            "worker_telemetry": _jsonable(worker_telemetry),
            "support_expansion": (
                expansion.as_record() if expansion is not None else None
            ),
            "support_expansion_telemetry": _jsonable(expansion_telemetry),
            "candidate_pool": _jsonable(candidates),
            "queries": queries,
        }
        lock_hash = _publish(folder / f"round_{round_index:03d}_lock.json", lock_payload)
        dock_tasks = [
            {**task, "query_id": query["query_id"], "smiles": query["smiles"]} for query in queries
        ]
        answers = list(dock_worker.map(dock_tasks, order_outputs=True, return_exceptions=True))
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
            "timing_seconds": {
                "round_total": time.time() - started,
                "proposal_max": max(
                    (row.get("elapsed_seconds", 0.0) for row in worker_telemetry), default=0.0
                ),
            },
            "rng_state_after": _jsonable(rng.bit_generator.state),
        }
        rounds.append(round_result)
        checkpoint = {
            "schema_version": "t4_integrated_route_fiber_checkpoint_v1",
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
        "schema_version": "t4_integrated_route_fiber_result_v1",
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
            records.append({"cell": row["cell"], "status": "failed", "error": repr(answer)})
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
        "schema_version": "t4_integrated_route_fiber_summary_v1",
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
    import subprocess

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import sha256_file
    from compose_v4.experiments.t4_matched_pilot import unseal

    contract = unseal(ROOT / CONTRACT)
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
        raise ValueError(f"untracked integrated runtime inputs: {untracked}")
    body = {
        "schema_version": "t4_integrated_route_fiber_launch_v1",
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
            "schema_version": "t4_integrated_route_fiber_launch_receipt_v1",
            "mode": mode,
            "task": task,
            "function_call_id": call.object_id,
            "volume": VOLUME_NAME,
            "output_prefix": task["run_id"],
        }
        destination = (
            ROOT / "diagnostics/t4_integrated_route_fiber_parp1_v1/launches" / (task["run_id"] + ".json")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        seal(destination, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return
    if mode != "status" or not run_id:
        raise ValueError("use mode=launch, mode=resume with run_id, or mode=status with run_id")
    task = _local_task()
    task["run_id"] = run_id
    print(json.dumps(remote_status.remote(task), indent=2))
