"""Zero-oracle Modal fan-out for the T4 support-restoration gate.

WHAT THIS APP DOES, AND WHAT IT CANNOT DO
------------------------------------------
It asks whether the automatically routed unified controller recovers the useful
proposal SUPPORT that five hand-chosen per-cell rescue arms produced.  It generates
proposals and gates them with the unmodified production `Fiber.check`, which is pure
RDKit arithmetic over similarity, QED and SA.

It charges NO oracle call and performs NO docking, and that is structural rather than
promised:

* the image installs NO `qvina02`, NO receptor `.pdbqt`, and NO openbabel, so there is
  nothing in the container to dock with;
* no function here imports `compose_v4.experiments.t4_docking_adapter`;
* every unit calls `assert_no_docking_reachable()`, which REFUSES to run if any
  docking-capable module is resident.

Compare the rescue apps this one is modelled on: they `curl` a qvina binary and a
receptor into `/opt/dock` at image build.  The absence of those three lines is the
whole safety argument, and it is visible in the image definition.

THE UNIT OF WORK, AND WHY IT IS SHAPED THIS WAY
------------------------------------------------
`proposal_unit` runs exactly one bounded proposal: one round-one expert, one rung-0
kernel call, or ONE escalation replicate at its lane's own contract base draw count.
A ladder step of 960 draws over a 480-draw lane is therefore TWO concurrent units, never
one unit at 960 -- a proposal worker is capped at 1800 s and 480 draws on a drug-like
parent already costs minutes, so deepening a unit would trade a bounded expansion for a
timeout.

`gate_event` is the per-(cell, seed) driver.  It must be a Modal function rather than
local code because `run_support_expansion` checks `wall_seconds` against a MONOTONIC
clock: a rung's replicates have to be genuinely concurrent, or the measurement records
queueing as an algorithmic stop.  Events are launched in small batches for the same
reason -- 40 containers shared between too many simultaneous events would manufacture
`wall_clock` stops caused by contention rather than by the algorithm.  Whether any such
stop occurred is reported.

DURABILITY
----------
Each event commits its own shard to a DEDICATED volume before returning, so one unit
exhausting its retries cannot take healthy siblings with it.  Nothing here writes to any
campaign volume, and nothing is ever deleted.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REMOTE = Path("/compose")
OUTPUT = Path("/gate")
VOLUME_NAME = "compose-t4-support-restoration-gate"
GATE_SCRIPT = "scripts/t4_support_restoration_gate.py"

CONTRACT_FILES = (
    "configs/t4_held_target_distilled_braf_d06_250.json",
    "configs/t4_held_target_distilled_fa7_d06_250.json",
    "configs/t4_held_target_distilled_5ht1b_d06_250.json",
    "configs/t4_held_target_distilled_parp1_d06_250.json",
    "configs/t4_held_target_distilled_jak2_true_d06_250.json",
    "configs/t4_5ht1b2_protonation_rescue_d06_v1.json",
)
CHECKPOINT_FILES = (
    "diagnostics/t4_held_target_distillation_quality_v1/braf_checkpoint.json",
    "diagnostics/t4_held_target_distillation_quality_v1/fa7_checkpoint.json",
    "diagnostics/t4_held_target_distillation_quality_v1/5ht1b_checkpoint.json",
    "diagnostics/t4_held_target_distillation_quality_v1/parp1_checkpoint.json",
    "diagnostics/t4_held_target_distillation_quality_v1/jak2_checkpoint.json",
    "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json",
)

#: No qvina, no receptor, no openbabel. The gate cannot dock because the container
#: holds nothing to dock with.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .add_local_dir(
        ROOT / "src", str(REMOTE / "src"), copy=True, ignore=["**/__pycache__/**"]
    )
    .add_local_file(ROOT / GATE_SCRIPT, str(REMOTE / GATE_SCRIPT), copy=True)
)
for _relative in CONTRACT_FILES + CHECKPOINT_FILES:
    image = image.add_local_file(ROOT / _relative, str(REMOTE / _relative), copy=True)

app = modal.App("compose-t4-support-restoration-gate")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
common = {"image": image, "volumes": {str(OUTPUT): volume}, "cpu": 1.0}


def _gate_module():
    """Import the gate library from the image, and prove nothing can dock."""

    sys.path.insert(0, str(REMOTE / "src"))
    sys.path.insert(0, str(REMOTE / "scripts"))
    import t4_support_restoration_gate as gate

    gate.assert_no_docking_reachable()
    return gate


@app.function(**common, max_containers=40, timeout=1800, retries=2)
def proposal_unit(task: dict) -> dict:
    """One bounded proposal unit. Charges nothing; docks nothing."""

    gate = _gate_module()
    started = time.time()
    payload, row = gate.load_cell(task["cell"], root=REMOTE)
    fiber = gate.Fiber(row["smiles"], payload["delta"], support=payload["support"])
    kind = task["kind"]
    if kind == "round_one_expert":
        answer = gate.round_one_expert(
            payload,
            row,
            task["expert"],
            task["expert_index"],
            task["seed_index"],
            fiber=fiber,
            root=REMOTE,
        )
    elif kind == "escalate":
        answer = gate.run_escalate_request(task["request"], row, payload, fiber=fiber)
    else:
        raise ValueError(f"unknown proposal unit kind {kind!r}")
    return {
        "kind": kind,
        "cell": task["cell"],
        "seed_index": task["seed_index"],
        "unit_seconds": round(time.time() - started, 2),
        **answer,
    }


@app.function(**common, max_containers=8, timeout=5400, retries=2)
def kernel_unit(task: dict) -> dict:
    """Rung 0: the ROUTED kernel, on its own cap.

    Deliberately a separate function from `proposal_unit`. A ladder replicate is held
    at the production 1800 s cap because it runs the production lane at the production
    base draw count, and a replicate that needed longer would mean the rung was being
    deepened rather than widened. Rung 0 is a different shape -- one pass over every
    bridge-separated region, or the whole protonation fiber -- so it gets its own bound
    instead of silently relaxing the one that keeps escalation honest.
    """

    gate = _gate_module()
    started = time.time()
    payload, row = gate.load_cell(task["cell"], root=REMOTE)
    fiber = gate.Fiber(row["smiles"], payload["delta"], support=payload["support"])
    answer = gate.rung_zero(
        row, payload, task["seed_index"], task["kernel"], fiber=fiber, root=REMOTE
    )
    return {
        "kind": "rung_zero",
        "cell": task["cell"],
        "seed_index": task["seed_index"],
        "unit_seconds": round(time.time() - started, 2),
        **answer,
    }


def _run_units(tasks: list[dict]) -> list[dict]:
    """Run units concurrently, surfacing a failed unit instead of losing the batch."""

    answers = list(
        proposal_unit.map(tasks, order_outputs=False, return_exceptions=True)
    )
    failures = [repr(a) for a in answers if isinstance(a, Exception)]
    if failures:
        raise RuntimeError(f"{len(failures)} proposal units failed: {failures[:3]}")
    return answers


@app.function(**common, max_containers=8, timeout=10800, retries=1)
def gate_event(task: dict) -> dict:
    """One (cell, seed) trigger event, with every rung fanned out in parallel."""

    gate = _gate_module()
    cell, seed_index = task["cell"], task["seed_index"]

    def _round_one(requests):
        return _run_units(
            [
                {
                    "kind": "round_one_expert",
                    "cell": cell,
                    "seed_index": seed_index,
                    "expert": request["expert"],
                    "expert_index": request["expert_index"],
                }
                for request in requests
            ]
        )

    def _escalator(requests):
        return _run_units(
            [
                {
                    "kind": "escalate",
                    "cell": cell,
                    "seed_index": seed_index,
                    "request": request,
                }
                for request in requests
            ]
        )

    def _rung_zero(kernel):
        return kernel_unit.remote(
            {"cell": cell, "seed_index": seed_index, "kernel": kernel}
        )

    shard = gate.run_trigger_event(
        cell,
        seed_index,
        root=REMOTE,
        escalator=_escalator,
        round_one_runner=_round_one,
        rung_zero_runner=_rung_zero,
    )
    shard["docking_interlock"] = gate.assert_no_docking_reachable()
    shard["execution"] = {"mode": "modal_fanout", "app": app.name}
    _commit(f"{cell}_seed{seed_index}.json", shard)
    return _summary(shard)


@app.function(**common, max_containers=10, timeout=10800, retries=1)
def control_event(task: dict) -> dict:
    """One non-trigger control: round one only, ladder unreachable."""

    gate = _gate_module()
    cell, seed_index = task["cell"], task["seed_index"]

    def _round_one(requests):
        return _run_units(
            [
                {
                    "kind": "round_one_expert",
                    "cell": cell,
                    "seed_index": seed_index,
                    "expert": request["expert"],
                    "expert_index": request["expert_index"],
                }
                for request in requests
            ]
        )

    shard = gate.run_control_event(
        cell, seed_index, root=REMOTE, round_one_runner=_round_one
    )
    shard["docking_interlock"] = gate.assert_no_docking_reachable()
    shard["execution"] = {"mode": "modal_fanout", "app": app.name}
    _commit(f"control_{cell}_seed{seed_index}.json", shard)
    return _summary(shard)


def _commit(name: str, shard: dict) -> None:
    destination = OUTPUT / "shards" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(shard, indent=1, sort_keys=True))
    volume.commit()


def _summary(shard: dict) -> dict:
    """The compact row the driver prints. The shard itself is the artifact."""

    expansion = shard.get("expansion") or {}
    return {
        "cell": shard["cell"],
        "role": shard["role"],
        "seed_index": shard["seed_index"],
        "selected": shard["round_one"]["decision"]["selected"],
        "agrees_with_committed_audit": shard["round_one"]["agrees_with_committed_audit"],
        "routed_kernel": (shard.get("routing") or {}).get("routed_kernel")
        or (shard.get("routing") or {}).get("routed_kernel_if_it_had_exhausted"),
        "distinct_eligible": shard.get("distinct_eligible"),
        "rung_zero_fresh_distinct": shard.get("rung_zero_fresh_distinct"),
        "stop_reason": expansion.get("stop_reason"),
        "draws_spent": expansion.get("draws_spent"),
        "attempts": expansion.get("attempts"),
        "drug_like_eligible": shard.get("drug_like_eligible"),
        "verdict": (shard.get("verdict") or {}).get("verdict"),
        "failure_kind": (shard.get("verdict") or {}).get("failure_kind"),
        "elapsed_seconds": shard.get("elapsed_seconds"),
    }


def _gate_constants() -> dict:
    """The gate module's cell lists, read WITHOUT importing it.

    `modal run` executes this entrypoint in the local CLI interpreter, which has no
    rdkit, while the gate module imports rdkit at module scope. Parsing the literals out
    of the source keeps ONE source of truth for which cells are triggers and which are
    controls, instead of a second copy here that could silently disagree with the module
    the remote units actually run.
    """

    import ast

    wanted = {"TRIGGER_CELLS", "CONTROL_CELLS", "SEED_INDICES"}
    tree = ast.parse((ROOT / GATE_SCRIPT).read_text())
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"{GATE_SCRIPT} no longer defines {missing}")
    return found


@app.local_entrypoint()
def main(
    mode: str = "pilot",
    cell: str = "fa7_0",
    seed_index: int = 0,
    batch: str = "",
    seeds: str = "",
) -> None:
    """`pilot` times ONE event; `triggers` and `controls` run a named batch.

    A batch is deliberately small: `wall_seconds` is checked against a monotonic clock
    inside the stopping rule, so too many simultaneous events against 40 containers
    would record contention as an algorithmic stop.
    """

    if mode == "pull":
        # `modal volume get` collapses a directory onto one path when the destination
        # does not already exist as a directory, and exits 0 while doing it, so the
        # shards are read through the Python API and written one file at a time.
        destination = ROOT / "diagnostics/t4_support_restoration_gate/shards"
        destination.mkdir(parents=True, exist_ok=True)
        pulled = 0
        for entry in volume.listdir("shards"):
            name = entry.path.rsplit("/", 1)[-1]
            if not name.endswith(".json"):
                continue
            body = b"".join(volume.read_file(entry.path))
            json.loads(body)  # refuse to land a truncated shard
            (destination / name).write_bytes(body)
            pulled += 1
            print(f"pulled {name} ({len(body)} bytes)", flush=True)
        print(f"PULLED {pulled} shards -> {destination}", flush=True)
        return

    if mode == "pilot":
        started = time.time()
        row = gate_event.remote({"cell": cell, "seed_index": seed_index})
        print(json.dumps(row, indent=1), flush=True)
        print(f"PILOT WALL {time.time() - started:.0f}s", flush=True)
        return

    if mode == "triggers":
        names = [name for name in batch.split(",") if name] or list(
            _gate_constants()["TRIGGER_CELLS"]
        )
        wanted = (
            [int(part) for part in seeds.split(",") if part]
            or list(_gate_constants()["SEED_INDICES"])
        )
        tasks = [
            {"cell": name, "seed_index": index} for name in names for index in wanted
        ]
        runner = gate_event
    elif mode == "controls":
        names = [name for name in batch.split(",") if name] or list(
            _gate_constants()["CONTROL_CELLS"]
        )
        tasks = [{"cell": name, "seed_index": 0} for name in names]
        runner = control_event
    else:
        raise ValueError(f"unknown mode {mode!r}")

    started = time.time()
    answers = list(runner.map(tasks, order_outputs=False, return_exceptions=True))
    ok = [a for a in answers if not isinstance(a, Exception)]
    bad = [repr(a) for a in answers if isinstance(a, Exception)]
    for row in sorted(ok, key=lambda r: (r["cell"], r["seed_index"])):
        print(json.dumps(row), flush=True)
    for failure in bad:
        print(f"FAILED {failure}", flush=True)
    print(
        f"BATCH mode={mode} complete={len(ok)}/{len(tasks)} "
        f"wall={time.time() - started:.0f}s",
        flush=True,
    )
