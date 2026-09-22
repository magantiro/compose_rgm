"""The canonical T4 experiment: ONE frozen controller, three arms, thirty cells.

One Modal app, one volume, three sealed arm contracts baked into one image. A cell
task carries only ``arm`` and ``cell``; everything that decides how the controller
searches is a module constant of
``compose_v4.experiments.t4_canonical_controller`` or a field of the shared
controller block, which all three contracts must agree on exactly
(``controller_identity``).

    A  local        shallow/local executable programs only          15 cells, delta 0.6
    B  coordinated  A + the existing coordinated structural         15 cells, delta 0.6
                    program families, fixed shared allocation
    C  adaptive     B + the frozen generic adaptive support-        30 cells, both deltas
                    expansion rule                <- the canonical controller

A vs B isolates coordinated structural programs. B vs C isolates adaptive use of
structural support. FiberControl, the archive, the executor, initialization, budgets
and endpoint constraints are identical in all three.

Per-cell independence
---------------------
Cells are spawned as SEPARATE function calls against a DEPLOYED app, not as one
``.map`` inside a driver. A ``.map`` that raises has destroyed 59 healthy shards in
this repository before, and a driver container is a single point of failure for the
whole panel. Here a cell can fail, be preempted, or be relaunched without any other
cell noticing, and ``tools/launch_t4_canonical.py --mode resume`` re-spawns only the
cells that have no ``result.json``.

Preemption
----------
``run_cell`` publishes ``checkpoint.json`` after EVERY round including the root, and
a round lock before any of that round's docking. A preempted container restarts with
the same input -- Modal does this even at ``retries=0`` -- meets its own checkpoint,
and resumes. A lock above the last checkpointed round is an interrupted round whose
charge count is unknowable: it is debited in full and never re-docked. ``dock_worker``
keeps ``retries=0`` because a retry there would issue a second real oracle call
against one query id and break the ledger; ``run_cell`` and ``proposal_worker`` carry
retries because neither can charge the oracle on its own.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/compose")
APP_NAME = "compose-t4-canonical-shared-controller"
VOLUME_NAME = "compose-t4-canonical-shared-controller"
OUTPUT = Path("/t4_canonical")
MOOD = "https://raw.githubusercontent.com/SeulLee05/MOOD/main/scorer"
RECEPTORS = ("parp1", "jak2", "braf", "5ht1b", "fa7")
SELF = "modal_apps/t4_canonical_shared_controller_app.py"

ARM_CONTRACTS = {
    "A": "configs/t4_canonical_shared_controller_a_v1.json",
    "B": "configs/t4_canonical_shared_controller_b_v1.json",
    "C": "configs/t4_canonical_shared_controller_c_v1.json",
}

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
            for name in RECEPTORS
        ],
    )
    .add_local_dir(ROOT / "src", str(REMOTE / "src"), copy=True, ignore=["**/__pycache__/**"])
    .add_local_file(ROOT / SELF, str(REMOTE / SELF), copy=True)
    .add_local_file(
        ROOT / "docs/GENMOL_T4_SEEDS.json", str(REMOTE / "docs/GENMOL_T4_SEEDS.json"), copy=True
    )
    .add_local_file(ROOT / ARM_CONTRACTS["A"], str(REMOTE / ARM_CONTRACTS["A"]), copy=True)
    .add_local_file(ROOT / ARM_CONTRACTS["B"], str(REMOTE / ARM_CONTRACTS["B"]), copy=True)
    .add_local_file(ROOT / ARM_CONTRACTS["C"], str(REMOTE / ARM_CONTRACTS["C"]), copy=True)
    .env({"PYTHONPATH": str(REMOTE / "src"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)

common = {
    "image": image,
    "memory": 4096,
    "volumes": {str(OUTPUT): volume},
    "scaledown_window": 30,
}


# ---- Contract plumbing --------------------------------------------------------------
def _contract_path(arm: str) -> str:
    try:
        return ARM_CONTRACTS[str(arm)]
    except KeyError:
        raise ValueError(f"unknown arm {arm!r}; arms are {sorted(ARM_CONTRACTS)}") from None


def _load_contract(arm: str, *, root: Path = REMOTE) -> dict:
    from compose_v4.experiments.t4_matched_pilot import unseal

    return unseal(root / _contract_path(arm))


def _validate_task(task: dict, *, role: str, root: Path = REMOTE) -> dict:
    """Re-prove the whole runtime identity inside the container, before any work.

    Three independent checks: the arm contract's payload hash matches what the task
    was launched against, the role is a declared one, and EVERY file the contract
    pins re-hashes to its pinned value in this image.
    """

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import sha256_file

    contract = _load_contract(task["arm"], root=root)
    if task.get("contract_payload_sha256") != identity(contract):
        raise ValueError(f"canonical contract identity mismatch for arm {task['arm']!r}")
    if role not in {"proposal", "dock", "cell", "status", "preflight"}:
        raise ValueError(f"unknown canonical runtime role {role!r}")
    for relative, expected in contract["runtime_inputs_sha256"].items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"runtime input mismatch for {relative}: {actual}")
    return contract


def _cell_record(contract: dict, name: str) -> dict:
    for row in contract["cells"]:
        if row["cell"] == name:
            return row
    raise ValueError(f"cell {name!r} is not declared by this arm contract")


def _publish(path: Path, payload: dict) -> str:
    import json

    from compose_v4.control.docking_value import identity

    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)
    volume.commit()
    return envelope["payload_sha256"]


def _jsonable(value):
    """The round-lock serializer, defined once in the controller module so a test can
    drive the SAME function without importing `modal`."""

    from compose_v4.experiments.t4_canonical_controller import jsonable

    return jsonable(value)


def _resume_state(folder: Path, task: dict, contract: dict) -> dict | None:
    """Restore a preempted cell from its last completed round, or None to start fresh.

    The checkpoint is published after every completed round INCLUDING the root, while
    a round lock is published before any of that round's queries are docked. A lock
    above the last checkpointed round is therefore an interrupted round whose charged
    count is unknowable: its queries are debited in full and never re-docked, so
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
    if payload["contract_payload_sha256"] != task["contract_payload_sha256"]:
        raise RuntimeError("refusing to resume a checkpoint sealed under a different contract")
    if payload["cell"] != task["cell"] or payload["arm"] != task["arm"]:
        raise RuntimeError("refusing to resume a checkpoint belonging to another cell or arm")

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
    # checkpoint an overwriting restart lost can have its archive rebuilt from locks
    # alone. Where two locks disagree about one molecule the WORSE score is kept, so
    # recovery can never flatter the archive.
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
        "expansion_events": payload.get("expansion_events", []),
    }


# ---- Proposal -----------------------------------------------------------------------
# Container budget, MEASURED: this workspace tops out near 100 concurrent CPUs
# (300 one-CPU tasks x 25 busy seconds ran at implied parallelism 72.8 over 100
# distinct containers; the same probe at cpu=4 gave 26.7 x 4 = ~107 CPUs, so the
# ceiling is CPUs, not containers). A `run_cell` container idles while its maps run
# but still reserves a CPU, so the three limits are chosen to SUM to that ceiling:
# 20 cells in flight + 56 proposal workers + 24 dockers = 100. Raising `run_cell`
# instead would starve the workers those very cells are waiting on.
@app.function(
    **common,
    cpu=(1.0, 1.0),
    max_containers=56,
    timeout=3600,
    retries=modal.Retries(max_retries=3, initial_delay=5.0),
)
def proposal_worker(task: dict) -> dict:
    """One expert on one measured parent. No task-oracle access of any kind."""

    import time

    import numpy as np

    from compose_v4.control.completion_law_contract import completion_law_for_proposal_lane
    from compose_v4.control.region_law_contract import region_law_for_proposal_lane
    from compose_v4.control.zero_support_fallback import fallback_candidates
    from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
    from compose_v4.experiments.t4_support_expansion import ESCALATABLE_LANES

    contract = _validate_task(task, role="proposal")
    cell = _cell_record(contract, task["cell"])
    delta = float(cell["delta"])
    expert = task["expert"]
    started = time.time()
    fiber = Fiber(cell["smiles"], delta, support=contract["support"])
    telemetry: dict = {}

    def _probe_draw(law, attempt):
        """One production draw, used only to prove the region law is consumed."""

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
        """The same proof for the COMPLETION half of `segment_replace`."""

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

    # Resolved and PROVEN CONSUMED on every expert, so a law parked on a lane that
    # cannot honour it fails on the worker that would otherwise have ignored it.
    region_law, region_law_telemetry = region_law_for_proposal_lane(
        contract, lane=expert, delta=delta, reference_smiles=cell["smiles"], draw=_probe_draw
    )
    completion_law, completion_law_telemetry = completion_law_for_proposal_lane(
        contract,
        lane=expert,
        delta=delta,
        reference_smiles=cell["smiles"],
        draw=_probe_completion_draw,
    )

    if expert == "zero_support_fallback":
        # The support-expansion stage that does NOT route through the
        # goal-abstraction layer: it excises each bridge-separated region with the
        # production executor and gates the EXECUTED endpoint directly.
        if not contract.get("adaptive_support_expansion"):
            raise ValueError("the zero-support fallback ran under an arm that forbids it")
        proposed, work = fallback_candidates(
            task["parent"],
            np.random.default_rng(task["proposal_seed"]),
            check=fiber.check,
            reference_smiles=cell["smiles"],
            delta=delta,
        )
        # `proposal_lane` is cleared and the true stage moved to its own key:
        # `_experts` validates a record's lane against the frozen expert vocabulary
        # and RAISES on an unknown one, so a record carrying "zero_support_fallback"
        # would kill `attach_features` at precisely the moment the fallback first
        # succeeded.
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
    elif expert in contract["experts"] or expert in ESCALATABLE_LANES:
        lane = contract["proposal"][expert]
        draws = int(lane["draws"])
        records = expand(
            task["parent"],
            task["parent_score"],
            fiber,
            np.random.default_rng(task["proposal_seed"]),
            draws=draws,
            multi_region=True,
            horizon=int(lane["horizon"]),
            proposal_lane=expert,
            region_law=region_law,
            completion_law=completion_law,
        )
        telemetry = {"raw_draws": draws, "expansion_replicate": task.get("expansion_replicate")}
        telemetry.update(region_law_telemetry)
        telemetry.update(completion_law_telemetry)
    else:
        raise ValueError(f"expert {expert!r} is not declared by arm {task['arm']!r}")

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


# ---- Oracle -------------------------------------------------------------------------
# retries=0 is LOAD-BEARING: a retry here issues a second real oracle call against one
# query id, which the ledger would still count once.
@app.function(**common, cpu=(1.0, 1.0), max_containers=24, timeout=900, retries=0)
def dock_worker(task: dict) -> dict:
    """One locked docking request, charged once, with no retry."""

    import time

    from compose_v4.experiments.continuation_profile import sha256_file
    from compose_v4.experiments.t4_docking_adapter import dock_t4

    contract = _validate_task(task, role="dock")
    cell = _cell_record(contract, task["cell"])
    receptor_path = Path(f"/opt/dock/receptors/{cell['receptor']}.pdbqt")
    physical = {
        "qvina02": sha256_file(Path("/opt/dock/qvina02")),
        "receptor": sha256_file(receptor_path),
    }
    expected = {
        "qvina02": contract["evaluator_sha256"]["qvina02"],
        "receptor": cell["receptor_sha256"],
    }
    if physical != expected:
        raise ValueError(f"docking evaluator identity mismatch: {physical}")
    started = time.time()
    score = dock_t4(
        task["smiles"],
        task["query_id"],
        contract["docking_seed"],
        box={"coordinates": cell["docking_box"], "receptor": str(receptor_path)},
    )
    return {
        "query_id": task["query_id"],
        "smiles": task["smiles"],
        "score": score,
        "failure": None if score is not None else "oracle_no_score",
        "elapsed_seconds": time.time() - started,
        "evaluator_sha256": physical,
    }


# ---- The cell loop ------------------------------------------------------------------
@app.function(
    **common,
    cpu=(1.0, 1.0),
    max_containers=20,
    timeout=23 * 3600,
    retries=modal.Retries(max_retries=8, initial_delay=10.0),
)
def run_cell(task: dict) -> dict:
    """One (arm, cell) campaign: round locks, checkpoints, and a per-cell value model."""

    import json
    import time

    import numpy as np

    from compose_v4.control.fiber_control import ProgramValue, SearchState
    from compose_v4.control.zero_support_fallback import fallback_seed
    from compose_v4.experiments.t4_canonical_controller import should_expand_support
    from compose_v4.experiments.t4_fiber_campaign import Fiber
    from compose_v4.experiments.t4_integrated_route_fiber import (
        attach_features,
        expert_census,
        merge_expert_pools,
        select_batch,
    )
    from compose_v4.experiments.t4_support_expansion import (
        assert_support_expansion_is_consumed,
        normalize_expansion_records,
        resolve_support_expansion,
        run_support_expansion,
    )

    contract = _validate_task(task, role="cell")
    cell = _cell_record(contract, task["cell"])
    experts = tuple(contract["experts"])
    delta = float(cell["delta"])
    adaptive = bool(contract["adaptive_support_expansion"])
    trigger_minimum = int(contract["expansion_trigger_min_eligible"]) if adaptive else 0
    folder = OUTPUT / task["run_id"] / task["arm"] / task["cell"]
    final_path = folder / "result.json"
    if final_path.exists():
        return json.loads(final_path.read_text())["payload"]

    resumed = _resume_state(folder, task, contract)
    rng = np.random.default_rng(cell["controller_seed"])
    fiber = Fiber(cell["smiles"], delta, support=contract["support"])
    value = ProgramValue(penalty=contract["value_penalty"])
    tag = f"{task['arm']}/{cell['cell']}"

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
        expansion_events = list(resumed["expansion_events"])
        charged = resumed["charged_calls"]
        rng.bit_generator.state = resumed["rng_state"]
        value.fit(features, improvements)
        previous = state.incumbent
        # The root docking answer belongs to the first attempt and is not carried in
        # the checkpoint, so a resumed cell has none. Binding it explicitly is what
        # stops a resumed cell that later exhausts from raising NameError instead of
        # publishing its terminal record.
        root_answer = None
        print(
            f"[{tag}] RESUME round={state.rounds} calls={charged} budget={state.budget} "
            f"best={previous:.2f} forfeited={resumed['forfeited_calls']} "
            f"recovered_archive={resumed['recovered_archive_entries']}",
            flush=True,
        )
    else:
        state = SearchState(archive={}, budget=contract["charged_calls_per_cell"], rounds=0)
        features: list[list[float]] = []
        improvements: list[float] = []
        rounds: list[dict] = []
        expansion_events: list[dict] = []
        charged = 0

        root_query = {
            "schema_version": "t4_canonical_round_lock_v1",
            "arm": task["arm"],
            "cell": cell["cell"],
            "round": 0,
            "kind": "root",
            "queries": [
                {"query_id": f"{task['run_id']}_{task['arm']}_{cell['cell']}_root",
                 "smiles": cell["smiles"]}
            ],
        }
        root_lock = _publish(folder / "round_000_lock.json", root_query)
        root_answer = dock_worker.remote(
            {**task, "query_id": root_query["queries"][0]["query_id"], "smiles": cell["smiles"]}
        )
        charged += 1
        state.budget -= 1
        if root_answer["score"] is None:
            result = {
                "schema_version": "t4_canonical_result_v1",
                "status": "root_oracle_failure",
                "arm": task["arm"],
                "cell": cell["cell"],
                "charged_calls": charged,
                "root_lock_sha256": root_lock,
                "root_result": root_answer,
            }
            _publish(final_path, result)
            return result
        state.archive[cell["smiles"]] = float(root_answer["score"])
        previous = state.incumbent
        print(f"[{tag}] root call=1 score={previous:.2f} budget={state.budget}", flush=True)
        # Round 0 IS a completed round: its one call is charged, scored and recorded,
        # so checkpointing it admits no ambiguity. Without this a preemption between
        # the root lock and the end of round one leaves a lock with no checkpoint,
        # which `_resume_state` correctly refuses -- and Modal restarts a preempted
        # container with the same input even at retries=0, so the restart meets its
        # own root lock and the cell dies having charged one call for nothing.
        _publish(
            folder / "checkpoint.json",
            {
                "schema_version": "t4_canonical_checkpoint_v1",
                "status": "running",
                "arm": task["arm"],
                "cell": cell["cell"],
                "contract_payload_sha256": task["contract_payload_sha256"],
                "charged_calls": charged,
                "budget_remaining": state.budget,
                "archive": dict(sorted(state.archive.items())),
                "features": features,
                "improvements": improvements,
                "history": state.history,
                "rounds": rounds,
                "expansion_events": expansion_events,
                "rng_state": _jsonable(rng.bit_generator.state),
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
            for expert_index, expert in enumerate(experts):
                requests.append(
                    {
                        **task,
                        "expert": expert,
                        "parent": parent,
                        "parent_score": state.archive[parent],
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
        pools = {expert: [] for expert in experts}
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

        merged = merge_expert_pools(pools, experts=experts)
        fresh = [row for row in merged if row["smiles"] not in state.archive]
        ordinary_eligible = len(fresh)
        candidates = attach_features(fresh, state, fiber, experts=experts)
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
            experts=experts,
        )

        # ---- Adaptive support expansion (arm C only) ----
        # The trigger is a GENERIC search statistic -- this round's own distinct
        # eligible-endpoint count -- evaluated identically on every cell of every
        # target. Nothing about the cell, the target, the round index or the
        # incumbent score enters it. Arms A and B carry no expansion at all, so an
        # empty pool ends the cell exactly as the shipped loop does; that is the
        # ablation, not a defect.
        expansion = None
        expansion_telemetry: list[dict] = []
        policy = None
        if adaptive and should_expand_support(
            distinct_eligible=ordinary_eligible, minimum=trigger_minimum
        ):
            policy = resolve_support_expansion(contract)
            if policy is None:
                raise ValueError(
                    "the arm declares adaptive support expansion and the contract "
                    "carries no support_expansion block"
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

                It does not enter the goal-abstraction layer in which `expand`
                abstracts a program to a structural goal and gates whatever
                `instantiate_goal` rebuilds.
                """

                inner = [
                    {
                        **task,
                        "expert": "zero_support_fallback",
                        "parent": parent,
                        "parent_score": state.archive[parent],
                        "proposal_seed": fallback_seed(
                            cell["controller_seed"],
                            round_index=round_index,
                            parent_index=parent_index,
                        ),
                    }
                    for parent_index, parent in enumerate(parents)
                ]
                answers = list(
                    proposal_worker.map(inner, order_outputs=True, return_exceptions=True)
                )
                produced, work = [], {}
                for request, answer in zip(inner, answers, strict=True):
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

                A proposal worker is bounded by its draw count, and 480 draws on a
                drug-like parent already costs minutes, so a step buys more workers
                rather than a deeper one. The 700_000_003 offset per attempt cannot
                collide with a normal round seed for any round count this budget
                admits.
                """

                inner = []
                for lane_index, lane in enumerate(policy.lanes):
                    base = int(contract["proposal"][lane]["draws"])
                    replicates = max(1, -(-int(draws) // base))
                    for replicate in range(replicates):
                        for parent_index, parent in enumerate(parents):
                            inner.append(
                                {
                                    **task,
                                    "expert": lane,
                                    "parent": parent,
                                    "parent_score": state.archive[parent],
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
                    proposal_worker.map(inner, order_outputs=True, return_exceptions=True)
                )
                produced = []
                for request, answer in zip(inner, answers, strict=True):
                    produced.extend(_record(request, answer))
                return produced

            known = set(state.archive) | {row["smiles"] for row in fresh}
            expansion = run_support_expansion(
                policy, escalate=_escalate, fallback=_fallback, already_seen=sorted(known)
            )
            print(
                f"[{tag}] EXPANSION round={round_index} ordinary={ordinary_eligible} "
                f"stop={expansion.stop_reason} new_eligible={expansion.distinct_eligible} "
                f"fallback={expansion.fallback_eligible} attempts={expansion.attempts} "
                f"draws={expansion.draws_spent} {time.time() - expansion_started:.0f}s",
                flush=True,
            )
            # The normal path gets `proposal_experts` from `merge_expert_pools`, which
            # the expansion path never calls because it deduplicates by endpoint
            # itself. Without this the round dies building its query rows AFTER the
            # expansion has already found its endpoints.
            extra = normalize_expansion_records(
                row for row in expansion.records if row["smiles"] not in known
            )
            # UNION, not replacement: arm C's pool is a strict superset of the pool
            # arm B would have had on the same round, which is what makes B vs C an
            # ablation of the expansion rather than of the ordinary draw.
            fresh = list(fresh) + list(extra)
            candidates = attach_features(fresh, state, fiber, experts=experts)
            selected = select_batch(
                candidates,
                value,
                state,
                rng,
                round_index=round_index,
                batch=room,
                exploration=min(contract["exploration"], room),
                expert_floor_rounds=contract["expert_floor_rounds"],
                experts=experts,
            )
            expansion_events.append(
                {
                    "round": round_index,
                    "ordinary_eligible": ordinary_eligible,
                    "trigger_minimum": trigger_minimum,
                    "expansion": expansion.as_record(),
                    "endpoints_added": len(extra),
                }
            )

        pool_census = expert_census(candidates, experts=experts)
        selected_census = expert_census(selected, experts=experts)

        if not selected:
            if adaptive:
                # Publishing exhaustion is gated on the expansion having actually run,
                # so a contract that declares one and a runtime that quietly does not
                # perform it cannot end the cell.
                assert_support_expansion_is_consumed(expansion)
            result = {
                "schema_version": "t4_canonical_result_v1",
                "status": "candidate_exhaustion",
                "arm": task["arm"],
                "cell": cell["cell"],
                "contract_payload_sha256": task["contract_payload_sha256"],
                "charged_calls": charged,
                "root_result": root_answer,
                "rounds": rounds,
                "expansion_events": expansion_events,
                "archive": dict(sorted(state.archive.items())),
                "final_best": state.incumbent,
                "best_smiles": min(state.archive, key=state.archive.get),
                "support_expansion": expansion.as_record() if expansion is not None else None,
                "support_expansion_policy": policy.as_record() if policy is not None else None,
            }
            _publish(final_path, result)
            return result

        queries = []
        for index, row in enumerate(selected):
            queries.append(
                {
                    "query_id": (
                        f"{task['run_id']}_{task['arm']}_{cell['cell']}"
                        f"_r{round_index:03d}_q{index:02d}"
                    ),
                    "smiles": row["smiles"],
                    "selection_kind": row["selection_kind"],
                    "proposal_experts": row["proposal_experts"],
                    "parent": row["parent"],
                    "parent_score": row["parent_score"],
                }
            )
        lock_payload = {
            "schema_version": "t4_canonical_round_lock_v1",
            "contract_payload_sha256": task["contract_payload_sha256"],
            "arm": task["arm"],
            "cell": cell["cell"],
            "round": round_index,
            "charged_before": charged,
            "parents": parents,
            "ordinary_eligible": ordinary_eligible,
            "pool_census": pool_census,
            "selected_census": selected_census,
            "worker_telemetry": _jsonable(worker_telemetry),
            "support_expansion": expansion.as_record() if expansion is not None else None,
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
            observed.append({**row, **answer})
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
            "ordinary_eligible": ordinary_eligible,
            "expansion_ran": expansion is not None,
            "pool_census": pool_census,
            "selected_census": selected_census,
            "selection_kind_census": {
                kind: sum(row["selection_kind"] == kind for row in selected)
                for kind in ("route_scale_floor", "generic_scale_floor", "expert_floor",
                             "model", "exploration")
            },
            "docked": _jsonable(observed),
            "round_best": min(
                (row["score"] for row in observed if row.get("score") is not None), default=None
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
        _publish(
            folder / "checkpoint.json",
            {
                "schema_version": "t4_canonical_checkpoint_v1",
                "status": "running",
                "arm": task["arm"],
                "cell": cell["cell"],
                "contract_payload_sha256": task["contract_payload_sha256"],
                "charged_calls": charged,
                "budget_remaining": state.budget,
                "archive": dict(sorted(state.archive.items())),
                "features": features,
                "improvements": improvements,
                "history": state.history,
                "rounds": rounds,
                "expansion_events": expansion_events,
                "rng_state": _jsonable(rng.bit_generator.state),
            },
        )
        print(
            f"[{tag}] round={round_index} calls={charged} best={state.incumbent:.2f} "
            f"round_best={round_result['round_best']} eligible={ordinary_eligible} "
            f"pool={pool_census} selected={selected_census} "
            f"seconds={round_result['timing_seconds']['round_total']:.0f}",
            flush=True,
        )

    result = {
        "schema_version": "t4_canonical_result_v1",
        "status": "complete_budget",
        "contract_payload_sha256": task["contract_payload_sha256"],
        "arm": task["arm"],
        "cell": cell["cell"],
        "root_result": root_answer,
        "charged_calls": charged,
        "rounds": rounds,
        "expansion_events": expansion_events,
        "archive": dict(sorted(state.archive.items())),
        "final_best": state.incumbent,
        "best_smiles": min(state.archive, key=state.archive.get),
        "training_rows": len(improvements),
        "new_oracle_calls": charged,
        "claim_boundary": contract["claim_boundary"],
    }
    _publish(final_path, result)
    return result


# ---- Status -------------------------------------------------------------------------
@app.function(**common, cpu=(1.0, 1.0), max_containers=1, timeout=600, retries=0)
def remote_status(task: dict) -> dict:
    """Every declared cell of every arm, read off the volume. Zero oracle calls."""

    import json

    volume.reload()
    rows = []
    for arm in sorted(ARM_CONTRACTS):
        contract = _load_contract(arm)
        for cell in contract["cells"]:
            folder = OUTPUT / task["run_id"] / arm / cell["cell"]
            result_path = folder / "result.json"
            checkpoint_path = folder / "checkpoint.json"
            locks = sorted(folder.glob("round_*_lock.json")) if folder.exists() else []
            entry = {"arm": arm, "cell": cell["cell"], "delta": cell["delta"],
                     "locks": len(locks)}
            if result_path.exists():
                payload = json.loads(result_path.read_text())["payload"]
                entry.update(state="final", status=payload["status"],
                             calls=payload["charged_calls"], best=payload.get("final_best"),
                             rounds=len(payload.get("rounds", [])),
                             expansions=len(payload.get("expansion_events", [])))
            elif checkpoint_path.exists():
                payload = json.loads(checkpoint_path.read_text())["payload"]
                entry.update(state="running", status=payload["status"],
                             calls=payload["charged_calls"],
                             best=min(payload["archive"].values()) if payload["archive"] else None,
                             rounds=len(payload["rounds"]),
                             expansions=len(payload.get("expansion_events", [])))
            else:
                entry.update(state="not_started", status=None, calls=0, best=None,
                             rounds=0, expansions=0)
            rows.append(entry)
    return {
        "run_id": task["run_id"],
        "cells": rows,
        "charged_calls_total": sum(row["calls"] for row in rows),
        "final": sum(row["state"] == "final" for row in rows),
        "running": sum(row["state"] == "running" for row in rows),
        "not_started": sum(row["state"] == "not_started" for row in rows),
    }


# ---- Zero-oracle preflight ----------------------------------------------------------
@app.function(**common, cpu=(1.0, 1.0), max_containers=56, timeout=5400, retries=0)
def preflight_lane(task: dict) -> dict:
    """ONE lane of ONE (arm, cell) from the ROOT parent, at the contract's own draw
    count. ZERO oracle calls.

    It runs the production proposal worker unchanged -- same expert, same draws, same
    laws, same `Fiber.check` -- so what it measures is the path `run_cell` will run,
    not a transcription of it. A lane that returns nothing is a MEASUREMENT, not a
    failure: the complete controller is what has to work, not every channel from every
    state.
    """

    import time

    from compose_v4.experiments.t4_canonical_controller import controller_identity
    from compose_v4.experiments.t4_fiber_campaign import Fiber

    contract = _validate_task(task, role="preflight")
    cell = _cell_record(contract, task["cell"])
    expert = task["expert"]
    if expert not in tuple(contract["experts"]):
        raise ValueError(f"lane {expert!r} is not declared by arm {task['arm']!r}")
    fiber = Fiber(cell["smiles"], float(cell["delta"]), support=contract["support"])
    root_gate = fiber.check(cell["smiles"])

    started = time.time()
    answer = proposal_worker.local(
        {
            **task,
            "parent": cell["smiles"],
            "parent_score": 0.0,
            "proposal_seed": int(
                cell["controller_seed"] + 101 * tuple(contract["experts"]).index(expert)
            ),
        }
    )
    elapsed = time.time() - started
    draws = int(contract["proposal"][expert]["draws"])
    records = answer["records"]
    return {
        "arm": task["arm"],
        "cell": cell["cell"],
        "target": cell["target"],
        "delta": float(cell["delta"]),
        "expert": expert,
        "controller_identity_sha256": controller_identity(contract),
        "root_heavy_atoms": root_gate["heavy"] if root_gate else None,
        "root_passes_own_gate": root_gate is not None,
        "draws": draws,
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_draw": round(elapsed / max(1, draws), 4),
        "eligible_unique": len(records),
        "eligible_smiles": sorted({row["smiles"] for row in records}),
        "telemetry": {
            key: value
            for key, value in (answer.get("telemetry") or {}).items()
            if key != "zero_support_fallback_work"
        },
    }
