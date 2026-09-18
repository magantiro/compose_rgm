"""Wide free expansion of the feasible program tree, and the depth-3 branch value.

Two measured facts set this module's shape.

THE POOL WAS THE BINDING CONSTRAINT, NOT THE SELECTION RULE. Accessibility at
delta=0.6 is about 1.1% of proposed endpoints and a proposal yields roughly 11 distinct
molecules, so 90 draws buys on the order of ten feasible candidates -- which is what the
first autonomous run saw (pools of 9 to 77). Choosing well among ten candidates cannot
recover what a thin pool never contained. Expansion is free of oracle calls and embarrassingly
parallel, so width is bought with cores rather than with budget.

EXPANSION RUNS IN SUBPROCESSES, NOT A FORK POOL. The multiprocessing rollout pool
deadlocks at 0% CPU on this machine at two or more workers (see learnings, 2026-07-21).
Each shard is therefore a fresh interpreter invoked as `python -m`, which also gives every
shard its own RDKit and its own chemistry cache. The parent merges and deduplicates by
canonical SMILES, so a shard boundary cannot change the pool it produces -- only how long
it takes to produce it.

The branch value is a receding-horizon rollout rather than a learned long-horizon critic:
the historical credit audit put most useful delayed credit within about three program
decisions, so a beam of free children evaluated under the current reward model is enough,
and it costs no docking call.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "t4_fiber_expansion_v1"

#: Each shard pays one interpreter start plus one RDKit import, so very small shards are
#: dominated by startup. Measured startup is about 2 s, and a 90-draw shard runs tens of
#: seconds, so this floor keeps the overhead under roughly a tenth of the work.
MINIMUM_DRAWS_PER_SHARD = 45


def _shard_plan(draws: int, shards: int) -> list[int]:
    """Split `draws` into at most `shards` pieces, none below the startup floor."""
    if draws < 1:
        raise ValueError("expansion needs at least one draw")
    usable = max(1, min(shards, draws // MINIMUM_DRAWS_PER_SHARD or 1))
    base, extra = divmod(draws, usable)
    return [base + (1 if i < extra else 0) for i in range(usable)]


def expand_wide(
    parent: str,
    parent_score: float,
    seed_smiles: str,
    delta: float,
    *,
    draws: int,
    shards: int = 8,
    horizon: int = 3,
    multi_region: bool = True,
    seed: int = 0,
    timeout: float = 3600.0,
) -> list[dict]:
    """Free feasible endpoints from one parent, expanded across `shards` subprocesses.

    Every shard draws from an independent stream, so the union is a larger sample of the
    same proposal law rather than a different law. Endpoints are deduplicated by canonical
    SMILES; a molecule reachable by several programs keeps the first program recorded for
    it, which is the one whose features the reward model will see.
    """
    plan = _shard_plan(draws, shards)
    environment = dict(os.environ)
    root = Path(__file__).resolve().parents[3]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    found: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as workspace, contextlib.ExitStack() as open_logs:
        running = []
        for index, count in enumerate(plan):
            output = Path(workspace) / f"shard_{index}.json"
            # Diagnostics go to a FILE, never a pipe. All shards are launched before any
            # is waited on, so a shard that fills a 64 KiB stderr pipe blocks forever
            # while the parent is still inside communicate() on an earlier one. That is
            # a genuine deadlock and it happened on the first wide run: two shards sat at
            # 6:25 elapsed, one spinning and one at 0.0% CPU with the parent waiting.
            log = open_logs.enter_context(
                (Path(workspace) / f"shard_{index}.err").open("w+")
            )
            request = {
                "parent": parent,
                "parent_score": parent_score,
                "seed_smiles": seed_smiles,
                "delta": delta,
                "draws": count,
                "horizon": horizon,
                "multi_region": multi_region,
                "seed": int(seed) + 1000 * index + 1,
                "output": str(output),
            }
            process = subprocess.Popen(
                [sys.executable, "-m", "compose_v4.experiments.t4_fiber_expansion"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=log,
                env=environment,
                text=True,
            )
            process.stdin.write(json.dumps(request))
            process.stdin.close()
            running.append((output, process, log))
        for output, process, log in running:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if process.returncode != 0 or not output.exists():
                # A dead shard costs width, never correctness: the pool is a union.
                log.seek(0)
                print(f"[expand] shard failed: {log.read().strip()[:200]}", flush=True)
                continue
            for record in json.loads(output.read_text()):
                found.setdefault(record["smiles"], record)
    return list(found.values())


def expand_frontier(
    frontier,
    seed_smiles: str,
    delta: float,
    *,
    draws: int,
    workers: int = 8,
    horizon: int = 3,
    multi_region: bool = True,
    seed: int = 0,
    support: str = "compose_valid",
    timeout: float = 3600.0,
) -> dict[str, dict]:
    """Expand every parent on the frontier at once, under one global worker cap.

    Expanding parents one after another leaves most cores idle: a single parent at 720
    draws measured 176 feasible endpoints in 70.5 s against 29 in 62.2 s serially, so the
    per-parent call is already saturating its own shards and eight parents in sequence
    costs eight times that. Here every (parent, shard) pair is one unit of work and at
    most `workers` run at a time, which keeps the machine busy regardless of how the
    frontier happens to be sized.

    Returns the merged pool keyed by canonical SMILES. `parent_score` on each record is
    the score of the parent it came from, which is what the reward model contrasts against.
    """
    units = []
    for parent, parent_score in frontier:
        for index, count in enumerate(_shard_plan(draws, workers)):
            units.append((parent, parent_score, count, index))
    if not units:
        return {}

    environment = dict(os.environ)
    root = Path(__file__).resolve().parents[3]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    pool: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as workspace, contextlib.ExitStack() as open_logs:
        pending, running = list(enumerate(units)), []

        def launch(slot, unit):
            parent, parent_score, count, index = unit
            output = Path(workspace) / f"unit_{slot}.json"
            log = open_logs.enter_context((Path(workspace) / f"unit_{slot}.err").open("w+"))
            request = {
                "parent": parent, "parent_score": parent_score, "seed_smiles": seed_smiles,
                "delta": delta, "draws": count, "horizon": horizon,
                "multi_region": multi_region, "support": support,
                "seed": int(seed) + 1000 * slot + 7 * index + 1, "output": str(output),
            }
            process = subprocess.Popen(
                [sys.executable, "-m", "compose_v4.experiments.t4_fiber_expansion"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log,
                env=environment, text=True,
            )
            process.stdin.write(json.dumps(request))
            process.stdin.close()
            return output, process, log

        def harvest(finished):
            output, process, log = finished
            if process.returncode != 0 or not output.exists():
                log.seek(0)
                print(f"[expand] unit failed: {log.read().strip()[:200]}", flush=True)
                return
            for record in json.loads(output.read_text()):
                pool.setdefault(record["smiles"], record)

        deadline = time.monotonic() + timeout
        while pending or running:
            while pending and len(running) < max(1, workers):
                slot, unit = pending.pop(0)
                running.append(launch(slot, unit))
            still = []
            for entry in running:
                if entry[1].poll() is None:
                    still.append(entry)
                else:
                    harvest(entry)
            running = still
            if running and pending is not None:
                if time.monotonic() > deadline:
                    for _, process, _ in running:
                        process.kill()
                        process.wait()
                    print("[expand] frontier expansion hit its timeout", flush=True)
                    break
                time.sleep(0.25)
    return pool


def branch_value(
    candidates,
    value,
    state,
    seed_smiles: str,
    delta: float,
    *,
    beam: int,
    draws: int,
    shards: int = 8,
    levels: int = 2,
    seed: int = 0,
) -> dict[str, dict]:
    """Best predicted endpoint score reachable from each candidate in <= `levels` more steps.

    This is the `Q_3` the plan asks for, computed by free rollout rather than learned: a
    beam of the most promising depth-one endpoints is expanded again, and a candidate
    inherits the best predicted endpoint score found anywhere in its own subtree. A
    candidate that is itself mediocre but opens a strong continuation therefore outranks
    one that is slightly better and dead -- which is the whole reason the horizon exists.

    Only the beam is expanded, so this is a greedy rollout and not an exhaustive tree. It
    costs no docking call, and candidates outside the beam keep their own predicted score.
    """
    from compose_v4.control.fiber_control import program_features

    if beam < 1 or draws < 1:
        raise ValueError("branch valuation needs a positive beam and draw count")
    own = {c["smiles"]: -(c["parent_score"] - float(p)) for c, p in
           zip(candidates, value.predict(np.asarray([c["features"] for c in candidates])))}
    reachable = {smiles: {"branch": utility, "own": utility, "explored": 0, "depth": 0}
                 for smiles, utility in own.items()}
    if levels < 1:
        return reachable

    frontier = sorted(own, key=own.get, reverse=True)[:beam]
    lineage = {smiles: smiles for smiles in frontier}
    for depth in range(1, levels + 1):
        following = []
        for node in frontier:
            ancestor = lineage[node]
            children = expand_wide(
                node, -own[node] if node in own else 0.0, seed_smiles, delta,
                draws=draws, shards=shards, seed=seed + 7919 * depth + len(following),
            )
            if not children:
                continue
            # Features must be built against the SAME state the model was fitted under,
            # or the rollout scores a different function than the one being trusted.
            scored = []
            for child in children:
                child["features"] = program_features(child, state)
                scored.append(child)
            predicted = value.predict(np.asarray([c["features"] for c in scored]))
            for child, gain in zip(scored, predicted):
                utility = -(child["parent_score"] - float(gain))
                own[child["smiles"]] = utility
                lineage[child["smiles"]] = ancestor
                if utility > reachable[ancestor]["branch"]:
                    reachable[ancestor].update(branch=utility, depth=depth)
                following.append(child["smiles"])
            reachable[ancestor]["explored"] += len(scored)
        if not following:
            break
        frontier = sorted(set(following), key=own.get, reverse=True)[:beam]
    return reachable


def _shard_main() -> None:
    """One expansion shard. Reads a request on stdin and writes endpoints to a file."""
    from compose_v4.experiments.t4_fiber_campaign import Fiber, expand

    request = json.loads(sys.stdin.read())
    fiber = Fiber(
        request["seed_smiles"], request["delta"],
        support=request.get("support", "compose_valid"),
    )
    rng = np.random.default_rng(request["seed"])
    records = expand(
        request["parent"],
        request["parent_score"],
        fiber,
        rng,
        draws=request["draws"],
        multi_region=request.get("multi_region", True),
        horizon=request.get("horizon", 3),
    )
    Path(request["output"]).write_text(json.dumps(records))


if __name__ == "__main__":
    _shard_main()
