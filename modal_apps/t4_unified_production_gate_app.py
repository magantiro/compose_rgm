"""Zero-oracle Modal fan-out for the support gate, run through the PRODUCTION path.

WHAT THIS APP DOES, AND WHAT IT CANNOT DO
------------------------------------------
It re-asks the support-restoration question -- does the automatically routed
unified controller recover the proposal SUPPORT the five hand-chosen per-cell
rescue arms produced -- with the proposals coming from the functions the Modal
campaign app itself calls (`t4_unified_proposal.proposal_unit` and
`t4_unified_controller.routed_support_expansion`) rather than from a harness that
reproduced their call shape by hand.

It charges NO oracle call and performs NO docking, and that is STRUCTURAL rather
than promised:

* the image installs NO `qvina02`, NO receptor `.pdbqt` and NO openbabel, so there
  is nothing in the container to dock with;
* no function here imports `compose_v4.experiments.t4_docking_adapter`;
* every unit calls `assert_no_docking_reachable()`, which REFUSES to run if any
  docking-capable module is resident.

Compare the campaign apps this one is modelled on: they `curl` a qvina binary and
a receptor into `/opt/dock` at image build. The absence of those three lines is
the whole safety argument, and it is visible in the image definition above.

THE UNIT OF WORK
----------------
`proposal_unit` runs exactly one bounded proposal: one round-one expert, or ONE
escalation replicate at its lane's own contract base draw count. A ladder step of
960 draws over a 480-draw lane is therefore TWO concurrent units, never one unit
at 960 -- a proposal worker is capped at 1800 s and 480 draws on a drug-like
parent already costs minutes, so deepening a unit would trade a bounded expansion
for a timeout.

`kernel_unit` runs a rung-0 lane and is deliberately a SEPARATE function on its
own cap. A ladder replicate is held at the production 1800 s bound because it runs
the production lane at the production base draw count, and a replicate needing
longer would mean the rung was being deepened rather than widened. Rung 0 is a
different shape -- one pass over every bridge-separated region, or the whole
protonation fiber -- so it gets its own bound instead of silently relaxing the one
that keeps escalation honest.

`gate_event` is the per-(cell, seed, arm) driver, and it must be a Modal function
rather than local code because `run_support_expansion` checks `wall_seconds`
against a MONOTONIC clock: a rung's replicates have to be genuinely concurrent, or
the measurement records queueing as an algorithmic stop. Events are launched in
small batches for the same reason. Whether any such stop occurred is reported.

THE TWO ARMS
------------
`declared` is the production configuration, with the contract's
`proposal.shallow.region_law` resolved and proven consumed. `law_off` omits the
field, which is the only byte-identical OFF state, and exists to attribute a
difference between this table and the standalone gate's -- never to re-open the
declared configuration.

DURABILITY
----------
Each event commits its own shard to a DEDICATED volume before returning, so one
unit exhausting its retries cannot take healthy siblings with it. Nothing here
writes to any campaign volume, and nothing is ever deleted.
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
VOLUME_NAME = "compose-t4-unified-production-gate"

GATE_SCRIPT = "scripts/t4_unified_production_gate.py"
#: The scoring machinery the production gate imports verbatim, so the two tables
#: are comparable. It must be in the image or the driver cannot import.
STANDALONE_SCRIPT = "scripts/t4_support_restoration_gate.py"

PROTEINS = ("braf", "fa7", "5ht1b", "parp1", "jak2")

CONTRACT_FILES = (
    *(f"configs/t4_unified_controller_{protein}_d06_v1.json" for protein in PROTEINS),
    # The standalone module reads these at import time for its cell lists and its
    # state-aware settings, so they travel with it.
    "configs/t4_held_target_distilled_braf_d06_250.json",
    "configs/t4_held_target_distilled_fa7_d06_250.json",
    "configs/t4_held_target_distilled_5ht1b_d06_250.json",
    "configs/t4_held_target_distilled_parp1_d06_250.json",
    "configs/t4_held_target_distilled_jak2_true_d06_250.json",
    "configs/t4_5ht1b2_protonation_rescue_d06_v1.json",
)
CHECKPOINT_FILES = (
    *(
        f"diagnostics/t4_held_target_distillation_quality_v1/{protein}_checkpoint.json"
        for protein in PROTEINS
    ),
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
    .add_local_file(ROOT / STANDALONE_SCRIPT, str(REMOTE / STANDALONE_SCRIPT), copy=True)
)
for _relative in CONTRACT_FILES + CHECKPOINT_FILES:
    image = image.add_local_file(ROOT / _relative, str(REMOTE / _relative), copy=True)

app = modal.App("compose-t4-unified-production-gate")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
common = {"image": image, "volumes": {str(OUTPUT): volume}, "cpu": 1.0}

def _rung_zero_lanes() -> tuple[str, ...]:
    """The rung-0 lanes, derived from the routing's own kernel-to-lane mapping.

    Read rather than restated: a second copy of this tuple could silently
    disagree with the mapping the router actually uses, and the consequence
    would be a rung-0 unit given the ladder replicate's 1800 s bound. This runs
    remotely, where rdkit is available; the local `modal run` interpreter never
    reaches it.
    """

    from compose_v4.experiments.t4_unified_controller import ROUTED_RUNG_ZERO_LANE

    return tuple(sorted(ROUTED_RUNG_ZERO_LANE.values()))


def _gate_module():
    """Import the production gate driver from the image, and prove nothing can dock."""

    sys.path.insert(0, str(REMOTE / "src"))
    sys.path.insert(0, str(REMOTE / "scripts"))
    import t4_unified_production_gate as gate

    gate.standalone.assert_no_docking_reachable()
    return gate


@app.function(**common, max_containers=40, timeout=1800, retries=2)
def proposal_unit(task: dict) -> dict:
    """One bounded production proposal unit. Charges nothing; docks nothing."""

    gate = _gate_module()
    started = time.time()
    answer = gate.production_unit(task, root=REMOTE)
    answer["unit_seconds"] = round(time.time() - started, 2)
    return answer


@app.function(**common, max_containers=12, timeout=5400, retries=2)
def kernel_unit(task: dict) -> dict:
    """One rung-0 production unit, on its own cap. See the module docstring."""

    gate = _gate_module()
    started = time.time()
    answer = gate.production_unit(task, root=REMOTE)
    answer["unit_seconds"] = round(time.time() - started, 2)
    return answer


def _run_units(tasks: list[dict]) -> list[dict]:
    """Run units concurrently, surfacing a failed unit instead of losing the batch.

    Rung-0 requests are routed to `kernel_unit` and everything else to
    `proposal_unit`, by LANE rather than by a flag the caller sets, so a request
    cannot be given the wrong bound by mislabelling itself.
    """

    lanes = _rung_zero_lanes()
    rung_zero = [task for task in tasks if task["expert"] in lanes]
    ordinary = [task for task in tasks if task["expert"] not in lanes]
    answers: list = []
    if ordinary:
        answers += list(
            proposal_unit.map(ordinary, order_outputs=False, return_exceptions=True)
        )
    if rung_zero:
        answers += list(
            kernel_unit.map(rung_zero, order_outputs=False, return_exceptions=True)
        )
    failures = [repr(a) for a in answers if isinstance(a, Exception)]
    if failures:
        raise RuntimeError(f"{len(failures)} proposal units failed: {failures[:3]}")
    return answers


@app.function(**common, max_containers=8, timeout=14400, retries=1)
def gate_event(task: dict) -> dict:
    """One (cell, seed, arm) trigger event, with every rung fanned out in parallel."""

    gate = _gate_module()
    cell, seed_index, arm = task["cell"], task["seed_index"], task["arm"]
    shard = gate.run_trigger_event(
        cell,
        seed_index,
        arm=arm,
        root=REMOTE,
        round_one_runner=_run_units,
        unit_runner=_run_units,
    )
    shard["docking_interlock"] = gate.standalone.assert_no_docking_reachable()
    shard["execution"] = {"mode": "modal_fanout", "app": app.name}
    _commit(f"{cell}_seed{seed_index}_{arm}.json", shard)
    return _summary(shard)


@app.function(**common, max_containers=10, timeout=10800, retries=1)
def control_event(task: dict) -> dict:
    """One non-trigger control: round one only. The ladder must be UNREACHABLE."""

    gate = _gate_module()
    cell, seed_index, arm = task["cell"], task["seed_index"], task["arm"]
    shard = gate.run_control_event(
        cell, seed_index, arm=arm, root=REMOTE, round_one_runner=_run_units
    )
    shard["docking_interlock"] = gate.standalone.assert_no_docking_reachable()
    shard["execution"] = {"mode": "modal_fanout", "app": app.name}
    _commit(f"control_{cell}_seed{seed_index}_{arm}.json", shard)
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
        "arm": shard["region_law_arm"],
        "selected": shard["round_one"]["decision"]["selected"],
        "agrees_with_committed_audit": shard["round_one"]["agrees_with_committed_audit"],
        "routed_kernel": (shard.get("routing") or {}).get("routed_kernel"),
        "routing_agrees": (shard.get("routing") or {}).get("agrees"),
        "state_routing_attempts": (shard.get("consumption") or {}).get(
            "state_routing_attempts"
        ),
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
    """The gate's cell lists, read WITHOUT importing it.

    `modal run` executes this entrypoint in the local CLI interpreter, which has
    no rdkit, while the gate module imports rdkit at module scope. Parsing the
    literals out of the STANDALONE module's source keeps ONE source of truth for
    which cells are triggers and which are controls, instead of a second copy here
    that could silently disagree with the module the remote units actually run.
    """

    import ast

    wanted = {"TRIGGER_CELLS", "CONTROL_CELLS", "SEED_INDICES"}
    tree = ast.parse((ROOT / STANDALONE_SCRIPT).read_text())
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"{STANDALONE_SCRIPT} no longer defines {missing}")
    return found


@app.local_entrypoint()
def main(
    mode: str = "pilot",
    cell: str = "fa7_0",
    seed_index: int = 0,
    arm: str = "declared",
    batch: str = "",
    seeds: str = "",
) -> None:
    """`pilot` times ONE event; `triggers` and `controls` run a named batch.

    A batch is deliberately small: `wall_seconds` is checked against a monotonic
    clock inside the stopping rule, so too many simultaneous events against a
    shared container pool would record contention as an algorithmic stop.
    """

    if mode == "pull":
        # `modal volume get` collapses a directory onto one path when the
        # destination does not already exist as a directory, and exits 0 while
        # doing it, so shards are read through the Python API one file at a time.
        destination = ROOT / "diagnostics/t4_unified_production_gate/shards"
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
        row = gate_event.remote(
            {"cell": cell, "seed_index": seed_index, "arm": arm}
        )
        print(json.dumps(row, indent=1), flush=True)
        print(f"PILOT WALL {time.time() - started:.0f}s", flush=True)
        return

    if mode == "triggers":
        names = [name for name in batch.split(",") if name] or list(
            _gate_constants()["TRIGGER_CELLS"]
        )
        wanted = [int(part) for part in seeds.split(",") if part] or list(
            _gate_constants()["SEED_INDICES"]
        )
        tasks = [
            {"cell": name, "seed_index": index, "arm": arm}
            for name in names
            for index in wanted
        ]
        runner = gate_event
    elif mode == "controls":
        names = [name for name in batch.split(",") if name] or list(
            _gate_constants()["CONTROL_CELLS"]
        )
        tasks = [{"cell": name, "seed_index": 0, "arm": arm} for name in names]
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
        f"BATCH mode={mode} arm={arm} complete={len(ok)}/{len(tasks)} "
        f"wall={time.time() - started:.0f}s",
        flush=True,
    )
