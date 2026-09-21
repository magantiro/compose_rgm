"""A deterministic fake-oracle campaign worker that can be SIGKILLed at named points.

Not a test module (no ``test_`` prefix, so pytest does not collect it): this is the
subject of ``tests/test_pmo_resumability.py``.  It is the smallest program that has
every part of a real run whose loss would matter -- an RNG stream that drives the
proposals, an archive, a learned memory, an allocator, credit statistics, a pending
candidate batch, an append-only charged ledger and a durable snapshot -- and none of
the chemistry, so a kill/resume matrix costs seconds and cannot be confounded by the
molecular kernel.

The oracle is a pure function of the molecule string.  It charges nothing and is
never a PMO oracle.

The worker dies by ``SIGKILL`` on itself rather than by raising, because an exception
unwinds: it would run ``finally`` blocks, flush buffers and close files, and a resume
protocol that only survives a graceful exception has not been tested at all.

It spawns no child processes, which the driver asserts: a resume that leaves orphans
behind has not resumed cleanly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

from compose_v4.control.durable_resume import (
    CallLedger,
    DurableSnapshotStore,
    LeaseHeld,
    WorkerLease,
    atomic_write_json,
)

KILL_POINTS = (
    "proposal",
    "before_reserve",
    "after_reserve",
    "after_observe",
    "before_commit",
)

REVISION = {"contract": "fake_oracle_resumability_v1", "controller": "kill_worker_v1"}


def fake_oracle(molecule: str) -> float:
    """Deterministic, free, and a pure function of its argument."""
    digest = hashlib.sha256(molecule.encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _die(point: str) -> None:
    sys.stderr.write(f"KILL {point}\n")
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGKILL)


def _initial_state(seed: int) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 17]))
    return {
        "archive": [],
        "memory": {"donor_regions": {}, "edit_outcomes": []},
        "allocator": {"parent_mass": {}, "draws": 0},
        "credit": {"cells": {}},
        "rng": rng.bit_generator.state,
        "pending_candidates": [],
    }


def _propose(state: dict, round_index: int, count: int) -> tuple[list[str], dict]:
    """Proposals depend on the RNG stream AND on learned state, so losing either diverges."""
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng"]
    memory_bias = len(state["memory"]["edit_outcomes"])
    archive_bias = len(state["archive"])
    molecules = []
    for index in range(count):
        token = int(rng.integers(0, 1_000_000))
        molecules.append(f"MOL-{round_index}-{index}-{token}-{memory_bias}-{archive_bias}")
    state["rng"] = rng.bit_generator.state
    state["pending_candidates"] = list(molecules)
    return molecules, state


def _incorporate(state: dict, molecule: str, score: float) -> dict:
    state["archive"].append({"molecule": molecule, "score": score})
    state["archive"].sort(key=lambda row: (-row["score"], row["molecule"]))
    del state["archive"][16:]
    state["memory"]["edit_outcomes"].append({"molecule": molecule, "delta": score})
    del state["memory"]["edit_outcomes"][32:]
    region = molecule.split("-")[1]
    state["memory"]["donor_regions"][region] = (
        state["memory"]["donor_regions"].get(region, 0.0) + score
    )
    state["allocator"]["parent_mass"][molecule] = score
    state["allocator"]["draws"] += 1
    cell = str(len(state["archive"]) % 4)
    cell_state = state["credit"]["cells"].setdefault(cell, {"trials": 0, "total": 0.0})
    cell_state["trials"] += 1
    cell_state["total"] += score
    return state


def _acquire_with_wait(lease_path: Path, run_id: str, worker_id: str, lease_seconds: float):
    """A resumer waits out a dead holder's lease rather than forcing it."""
    deadline = time.time() + 30.0
    while True:
        try:
            return WorkerLease.acquire(
                lease_path,
                run_id=run_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            )
        except LeaseHeld:
            if time.time() > deadline:
                raise
            time.sleep(0.05)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--calls-per-round", type=int, default=4)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--lease-seconds", type=float, default=0.25)
    parser.add_argument("--kill-at", choices=KILL_POINTS, default=None)
    parser.add_argument("--kill-round", type=int, default=2)
    parser.add_argument("--kill-ordinal", type=int, default=1)
    args = parser.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "result.json"
    if result_path.exists():
        return 0

    lease = _acquire_with_wait(
        root / "lease.json", args.run_id, args.worker_id, args.lease_seconds
    )
    ledger = CallLedger(root / "oracle", run_id=args.run_id, budget=args.budget)
    store = DurableSnapshotStore(
        root / "snapshots", run_id=args.run_id, revision_identity=REVISION
    )

    # Repair before reading state: a reservation whose worker died mid-call is
    # already charged, so its score is filled in exactly once and never re-spent.
    ledger.repair_pending(fake_oracle)

    latest = store.load_latest()
    if latest is None:
        state = _initial_state(args.seed)
        start_round = 0
    else:
        state = json.loads(json.dumps(latest["components"]))
        start_round = int(latest["round_index"]) + 1

    for round_index in range(start_round, args.rounds):
        lease.renew()
        if args.kill_at == "proposal" and round_index == args.kill_round:
            _die("proposal")
        molecules, state = _propose(state, round_index, args.calls_per_round)
        if args.kill_at == "before_reserve" and round_index == args.kill_round:
            _die("before_reserve")
        for ordinal, molecule in enumerate(molecules):
            reservation = ledger.reserve(
                round_index=round_index, ordinal=ordinal, molecule=molecule
            )
            if (
                args.kill_at == "after_reserve"
                and round_index == args.kill_round
                and ordinal == args.kill_ordinal
            ):
                _die("after_reserve")
            score = fake_oracle(molecule)
            ledger.physical_calls += 1
            ledger.observe(reservation, score)
            if (
                args.kill_at == "after_observe"
                and round_index == args.kill_round
                and ordinal == args.kill_ordinal
            ):
                _die("after_observe")
            state = _incorporate(state, molecule, score)
        state["pending_candidates"] = []
        if args.kill_at == "before_commit" and round_index == args.kill_round:
            _die("before_commit")
        store.commit(
            round_index=round_index,
            charged_calls=ledger.charged_calls,
            scored_call_ordinal=len(ledger.reconcile().complete),
            components=state,
            lease=lease,
        )

    reconciliation = ledger.reconcile()
    atomic_write_json(
        result_path,
        {
            "schema_version": "resumability_kill_worker_result_v1",
            "run_id": args.run_id,
            "charged_calls": reconciliation.charged,
            "complete_calls": len(reconciliation.complete),
            "pending_calls": len(reconciliation.pending),
            "repaired_calls": ledger.repaired_calls,
            "rounds": args.rounds,
        },
    )
    lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
