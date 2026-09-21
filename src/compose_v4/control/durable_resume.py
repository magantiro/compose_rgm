"""Preemption-invisible run state: append-only call ledger, fenced lease, atomic snapshot.

A cloud worker dies without warning.  The scientific requirement is not that it
stops dying, it is that a resumed run is INDISTINGUISHABLE from one that was
never interrupted -- same charged calls, same call ids, same molecules, same
scores, same order, and the same controller state including every random stream.

The three durable objects here are deliberately separate, because they fail
differently:

``CallLedger``
    Append-only.  Spend is the number of RESERVATIONS on disk and is never
    derived from a mutable checkpoint, so a rolled-back snapshot cannot refund a
    call.  Each reservation is written BEFORE the oracle runs and carries an
    immutable ``call_id`` fixed at reservation time.  A reservation addresses a
    SLOT ``(round_index, ordinal)``, so re-reserving the same slot with the same
    molecule is idempotent -- that is what makes a resumed round free -- while
    re-reserving it with a DIFFERENT molecule raises ``ResumeDivergence``.  That
    second case is the whole point: it is the signature of controller state (in
    practice an RNG stream) that failed to restore, and it must be loud rather
    than quietly producing a different trajectory.

``WorkerLease``
    Replaces "a started marker means a worker may still be alive, forever".  A
    lease is renewable and EXPIRES.  A live holder keeps renewing and blocks a
    second worker; a dead holder's lease lapses and another worker may take the
    run over.  Takeover bumps a monotone ``epoch`` which acts as a FENCING
    TOKEN: a previous incarnation that wakes up late cannot renew or commit,
    because the epoch on disk has moved past its own.  Without the fence a lease
    is only a hint, since "expired" and "dead" are not the same statement.

``DurableSnapshotStore``
    Atomic (write-temp-then-rename) and, critically, REQUIRED-COMPONENT CHECKED
    in both directions.  A snapshot that omits a declared component is refused at
    commit, and a stored snapshot missing one is refused at load.  The failure
    this prevents is specific and has already happened here: a restore path that
    substitutes an empty object for an absent key turns a lost learned memory
    into a silent cold start that still produces plausible numbers.

Deliberately chemistry-free and dependency-light: this module imports nothing
from the rewrite kernel, so the kill/resume test runs in seconds and cannot be
confounded by the chemistry environment.  ``atomic_write_json`` is local for the
same reason -- the repository's other atomic writer lives behind the executor
import graph.

Durability boundary: ``atomic_write_json`` renames into place, which is atomic
against process death (preemption, SIGKILL).  It does not fsync, so it is NOT a
claim about surviving host power loss; on a network volume the caller's commit
hook is the durability point, which is why ``CallLedger`` takes a ``flush``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compose_v4.data.durable_path import require_durable_path

# ---- Errors -----------------------------------------------------------------


class ResumeError(RuntimeError):
    """Base for every refusal in this module."""


class LeaseHeld(ResumeError):
    """Another worker holds an unexpired lease, or this worker has been fenced out."""


class ResumeDivergence(ResumeError):
    """A resumed run proposed different work for a slot that was already reserved."""


class SnapshotIncomplete(ResumeError):
    """A snapshot is missing a component the run declared as required."""


class BudgetExhausted(ResumeError):
    """The append-only ledger already holds its authorized number of reservations."""


class LedgerCorrupt(ResumeError):
    """A durable record failed its own content hash or ordering invariant."""


# ---- Atomic JSON ------------------------------------------------------------


def canonical_json(value: Any) -> bytes:
    """Stable bytes for hashing and comparison; rejects NaN so ids stay total."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def atomic_write_json(path: Path, value: Any) -> str:
    """Write-temp-then-rename.  A reader sees the old file or the new one, never half.

    The temporary name carries the writing pid so two processes racing on the same
    path cannot truncate each other's partial file and produce a torn rename.
    """
    path = Path(path)
    payload = canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


# ---- Worker lease -----------------------------------------------------------

LEASE_SCHEMA = "compose_worker_lease_v1"


@dataclass
class WorkerLease:
    """A renewable, expiring, fenced claim on one run directory."""

    path: Path
    run_id: str
    worker_id: str
    epoch: int
    lease_seconds: float
    acquired_at: float
    heartbeat_at: float
    clock: Callable[[], float] = time.time

    # -- construction --

    @staticmethod
    def _record(lease: WorkerLease) -> dict:
        return {
            "schema_version": LEASE_SCHEMA,
            "run_id": lease.run_id,
            "worker_id": lease.worker_id,
            "epoch": lease.epoch,
            "lease_seconds": lease.lease_seconds,
            "acquired_at": lease.acquired_at,
            "heartbeat_at": lease.heartbeat_at,
        }

    @classmethod
    def acquire(
        cls,
        path: Path | str,
        *,
        run_id: str,
        worker_id: str,
        lease_seconds: float,
        clock: Callable[[], float] = time.time,
    ) -> WorkerLease:
        """Take the lease, or refuse if a live worker holds it.

        Every acquisition bumps ``epoch``, including a re-acquisition by the same
        worker id, so a previous incarnation is always fenced out.  Worker ids are
        not assumed unique across container restarts.
        """
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        path = Path(path)
        now = float(clock())
        prior = read_json(path) if path.exists() else None
        epoch = 0
        if prior is not None:
            if prior.get("schema_version") != LEASE_SCHEMA:
                raise LedgerCorrupt(f"unknown lease schema at {path}")
            if prior["run_id"] != run_id:
                raise LeaseHeld(
                    f"lease at {path} belongs to run {prior['run_id']!r}, not {run_id!r}"
                )
            age = now - float(prior["heartbeat_at"])
            if age <= float(prior["lease_seconds"]):
                raise LeaseHeld(
                    "a live worker holds this run: "
                    f"worker={prior['worker_id']!r} epoch={prior['epoch']} "
                    f"heartbeat_age={age:.3f}s lease={prior['lease_seconds']}s"
                )
            epoch = int(prior["epoch"])
        lease = cls(
            path=path,
            run_id=run_id,
            worker_id=worker_id,
            epoch=epoch + 1,
            lease_seconds=float(lease_seconds),
            acquired_at=now,
            heartbeat_at=now,
            clock=clock,
        )
        atomic_write_json(path, cls._record(lease))
        return lease

    # -- maintenance --

    def renew(self) -> None:
        """Extend the lease, refusing if this worker has been fenced out."""
        self.assert_held()
        self.heartbeat_at = float(self.clock())
        atomic_write_json(self.path, self._record(self))

    def assert_held(self) -> None:
        """Re-read the lease and require that this worker still owns the epoch.

        Called before any durable commit.  A worker whose lease lapsed while it was
        merely slow -- not dead -- is refused here rather than writing over the
        state of the worker that legitimately took the run over.
        """
        if not self.path.exists():
            raise LeaseHeld("lease record vanished; this worker no longer owns the run")
        current = read_json(self.path)
        if int(current["epoch"]) != self.epoch or current["worker_id"] != self.worker_id:
            raise LeaseHeld(
                f"fenced out: lease moved to worker={current['worker_id']!r} "
                f"epoch={current['epoch']} (this worker holds epoch {self.epoch})"
            )

    def expired(self) -> bool:
        if not self.path.exists():
            return True
        current = read_json(self.path)
        return (float(self.clock()) - float(current["heartbeat_at"])) > float(
            current["lease_seconds"]
        )

    def release(self) -> None:
        """Release voluntarily so a successor need not wait out the lease."""
        self.assert_held()
        record = {**self._record(self), "heartbeat_at": 0.0, "released": True}
        atomic_write_json(self.path, record)


# ---- Append-only oracle call ledger -----------------------------------------

RESERVED = "reserved"
OBSERVED = "observed"
CALL_SCHEMA = "compose_call_reservation_v1"


@dataclass(frozen=True)
class CallReservation:
    call_id: str
    run_id: str
    round_index: int
    ordinal: int
    molecule: str
    slot: str

    @property
    def sort_key(self) -> tuple[int, int]:
        return (self.round_index, self.ordinal)


@dataclass
class LedgerReconciliation:
    """Every reservation on disk, classified.  There is no unclassifiable state."""

    complete: list[CallReservation] = field(default_factory=list)
    pending: list[CallReservation] = field(default_factory=list)

    @property
    def charged(self) -> int:
        return len(self.complete) + len(self.pending)


class CallLedger:
    """Append-only reservations; spend is counted on disk, never from a snapshot.

    Two-phase commit per call::

        reserve(round, ordinal, molecule)  ->  reserved.json   (call_id fixed here)
        oracle runs
        observe(reservation, score)        ->  observed.json

    On resume every reservation is either OBSERVED (complete) or RESERVED-only
    (pending).  A pending reservation stays CHARGED -- the oracle may well have
    run -- and is repaired by re-evaluating that exact molecule once.  The repair
    is an extra PHYSICAL call and is reported as such; it never adds a charge,
    and it can never be skipped silently because ``reserve`` on that slot returns
    the existing reservation rather than allocating a new one.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        run_id: str,
        budget: int,
        flush: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("a positive explicitly authorized call budget is required")
        self.root = require_durable_path(root, role="oracle call ledger")
        self.run_id = run_id
        self.budget = budget
        self.flush = flush or (lambda: None)
        self.physical_calls = 0
        self.repaired_calls = 0
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": CALL_SCHEMA, "run_id": run_id, "budget": budget}
        manifest_path = self.root / "ledger_manifest.json"
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise LedgerCorrupt("call ledger run identity or budget changed across resume")
        atomic_write_json(manifest_path, manifest)

    # -- durable layout --

    def _slot(self, round_index: int, ordinal: int) -> str:
        return f"round_{round_index:06d}_call_{ordinal:04d}"

    def _folder(self, slot: str) -> Path:
        return self.root / "calls" / slot

    @staticmethod
    def call_id_for(run_id: str, round_index: int, ordinal: int, molecule: str) -> str:
        """Immutable, content-derived, and fixed before the oracle is invoked."""
        return identity(
            {
                "run_id": run_id,
                "round_index": round_index,
                "ordinal": ordinal,
                "molecule": molecule,
            }
        )

    # -- reconciliation --

    def reconcile(self) -> LedgerReconciliation:
        result = LedgerReconciliation()
        calls_root = self.root / "calls"
        folders = sorted(calls_root.glob("round_*_call_*")) if calls_root.exists() else []
        for folder in folders:
            reserved_path = folder / "reserved.json"
            if not reserved_path.exists():
                # A reservation folder with no record is a torn create, not a charge.
                continue
            record = read_json(reserved_path)
            body = {k: v for k, v in record.items() if k != "call_id"}
            expected = self.call_id_for(
                record["run_id"], record["round_index"], record["ordinal"], record["molecule"]
            )
            if record["call_id"] != expected or body.get("run_id") != self.run_id:
                raise LedgerCorrupt(f"reservation identity failed verification: {reserved_path}")
            reservation = CallReservation(
                call_id=record["call_id"],
                run_id=record["run_id"],
                round_index=int(record["round_index"]),
                ordinal=int(record["ordinal"]),
                molecule=record["molecule"],
                slot=folder.name,
            )
            observed_path = folder / "observed.json"
            if observed_path.exists():
                observation = read_json(observed_path)
                if observation["call_id"] != reservation.call_id:
                    raise LedgerCorrupt(
                        f"observation does not match its reservation: {observed_path}"
                    )
                result.complete.append(reservation)
            else:
                result.pending.append(reservation)
        result.complete.sort(key=lambda r: r.sort_key)
        result.pending.sort(key=lambda r: r.sort_key)
        if len(result.complete) + len(result.pending) > self.budget:
            raise BudgetExhausted("restored ledger already exceeds its authorized budget")
        return result

    @property
    def charged_calls(self) -> int:
        """Counted from durable reservations.  Never read from controller state."""
        return self.reconcile().charged

    @property
    def remaining(self) -> int:
        return self.budget - self.charged_calls

    # -- two-phase commit --

    def reserve(self, *, round_index: int, ordinal: int, molecule: str) -> CallReservation:
        """Claim a call slot durably before spending it.  Idempotent per slot.

        Double-spend is prevented by SLOT ADDRESSING, not by the existence check
        below: a reservation lives at a path derived from ``(round, ordinal)``, so a
        replayed round rewrites its own folders and the charge count -- which is the
        number of folders -- cannot grow.  A sequence-addressed ledger, where each
        reservation appends a new record, would double-charge every replayed round.
        The existence branch is the DIVERGENCE detector: it is what notices that the
        molecule proposed for an already-reserved slot has changed.
        """
        slot = self._slot(round_index, ordinal)
        folder = self._folder(slot)
        reserved_path = folder / "reserved.json"
        if reserved_path.exists():
            existing = read_json(reserved_path)
            if existing["molecule"] != molecule:
                raise ResumeDivergence(
                    f"slot {slot} was reserved for {existing['molecule']!r} but this run "
                    f"proposed {molecule!r}; controller state did not restore exactly"
                )
            return CallReservation(
                call_id=existing["call_id"],
                run_id=existing["run_id"],
                round_index=int(existing["round_index"]),
                ordinal=int(existing["ordinal"]),
                molecule=existing["molecule"],
                slot=slot,
            )
        if self.remaining <= 0:
            raise BudgetExhausted("no authorized calls remain")
        call_id = self.call_id_for(self.run_id, round_index, ordinal, molecule)
        record = {
            "schema_version": CALL_SCHEMA,
            "run_id": self.run_id,
            "round_index": round_index,
            "ordinal": ordinal,
            "molecule": molecule,
            "call_id": call_id,
        }
        atomic_write_json(reserved_path, record)
        # The reservation must outlive this worker before the oracle is invoked;
        # otherwise a death during evaluation loses the charge entirely.
        self.flush()
        return CallReservation(
            call_id=call_id,
            run_id=self.run_id,
            round_index=round_index,
            ordinal=ordinal,
            molecule=molecule,
            slot=slot,
        )

    def observe(self, reservation: CallReservation, score: float, *, repaired: bool = False) -> dict:
        """Record the oracle's answer for a reservation already on disk."""
        folder = self._folder(reservation.slot)
        if not (folder / "reserved.json").exists():
            raise LedgerCorrupt("observation without a durable reservation")
        observed_path = folder / "observed.json"
        if observed_path.exists():
            return read_json(observed_path)
        record = {
            "schema_version": CALL_SCHEMA,
            "call_id": reservation.call_id,
            "run_id": reservation.run_id,
            "round_index": reservation.round_index,
            "ordinal": reservation.ordinal,
            "molecule": reservation.molecule,
            "score": float(score),
            "repaired": bool(repaired),
        }
        atomic_write_json(observed_path, record)
        self.flush()
        return record

    def score_of(self, reservation: CallReservation) -> float:
        return float(read_json(self._folder(reservation.slot) / "observed.json")["score"])

    def rows(self) -> list[dict]:
        """Every complete call in canonical (round, ordinal) order."""
        return [
            read_json(self._folder(r.slot) / "observed.json")
            for r in self.reconcile().complete
        ]

    def repair_pending(self, evaluate: Callable[[str], float]) -> list[dict]:
        """Fill in scores for reservations whose worker died before observing.

        Each repair is one extra PHYSICAL oracle call and zero extra CHARGED calls:
        the reservation was already counted, because a reservation written before
        the oracle ran cannot be proven not to have reached it.  Conservative in the
        direction that matters -- a preemption never makes calls cheaper.
        """
        repaired = []
        for reservation in self.reconcile().pending:
            score = float(evaluate(reservation.molecule))
            self.physical_calls += 1
            self.repaired_calls += 1
            repaired.append(self.observe(reservation, score, repaired=True))
        return repaired


# ---- Durable controller snapshot --------------------------------------------

SNAPSHOT_SCHEMA = "compose_durable_snapshot_v1"

#: The scientific state a PMO run must carry across a preemption.  Every entry is
#: something whose absence measurably changes the trajectory, so the store refuses
#: a snapshot that omits one rather than restoring an empty stand-in.
PMO_REQUIRED_COMPONENTS = frozenset(
    {
        "archive",
        "memory",
        "allocator",
        "credit",
        "rng",
        "pending_candidates",
    }
)


class DurableSnapshotStore:
    """Atomic, round-addressed controller snapshots with a required-component check."""

    def __init__(
        self,
        root: Path | str,
        *,
        run_id: str,
        revision_identity: dict,
        required_components: frozenset[str] = PMO_REQUIRED_COMPONENTS,
    ) -> None:
        if not revision_identity:
            raise ValueError("a snapshot must carry the contract/revision identity it ran under")
        if not required_components:
            raise ValueError("declare the components whose absence would change the trajectory")
        self.root = require_durable_path(root, role="controller snapshot store")
        self.run_id = run_id
        self.revision_identity = json.loads(json.dumps(revision_identity))
        self.required_components = frozenset(required_components)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, round_index: int) -> Path:
        return self.root / f"snapshot_{round_index:06d}.json"

    def commit(
        self,
        *,
        round_index: int,
        charged_calls: int,
        scored_call_ordinal: int,
        components: dict,
        lease: WorkerLease | None = None,
    ) -> dict:
        """Publish a snapshot atomically, after proving this worker still holds the run."""
        missing = sorted(self.required_components - set(components))
        if missing:
            raise SnapshotIncomplete(
                f"snapshot for round {round_index} omits required components: {missing}"
            )
        empty = sorted(name for name in self.required_components if components[name] is None)
        if empty:
            raise SnapshotIncomplete(
                f"snapshot for round {round_index} carries null required components: {empty}"
            )
        if lease is not None:
            # Fence before writing: a worker that lost its lease must not overwrite
            # the state of the worker that legitimately took the run over.
            lease.assert_held()
        body = {
            "schema_version": SNAPSHOT_SCHEMA,
            "run_id": self.run_id,
            "round_index": int(round_index),
            "charged_calls": int(charged_calls),
            "scored_call_ordinal": int(scored_call_ordinal),
            "revision_identity": self.revision_identity,
            "required_components": sorted(self.required_components),
            "epoch": None if lease is None else lease.epoch,
            "components": json.loads(json.dumps(components)),
        }
        record = {**body, "snapshot_sha256": identity(body)}
        atomic_write_json(self._path(round_index), record)
        return record

    def committed_rounds(self) -> list[int]:
        rounds = []
        for path in self.root.glob("snapshot_*.json"):
            try:
                rounds.append(int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        return sorted(rounds)

    def load_latest(self) -> dict | None:
        """Return the highest committed snapshot, or None for a fresh run.

        Verifies the content hash, the run identity, the revision identity and the
        presence of every required component.  A snapshot that fails any of these is
        an error, never a reason to start from nothing.
        """
        rounds = self.committed_rounds()
        if not rounds:
            return None
        path = self._path(rounds[-1])
        record = read_json(path)
        body = {k: v for k, v in record.items() if k != "snapshot_sha256"}
        if identity(body) != record.get("snapshot_sha256"):
            raise LedgerCorrupt(f"snapshot failed its content hash: {path}")
        if record["run_id"] != self.run_id:
            raise LedgerCorrupt(f"snapshot belongs to a different run: {path}")
        if record["revision_identity"] != self.revision_identity:
            raise LedgerCorrupt(
                "snapshot was produced under a different contract/revision identity; "
                "resuming would splice two runtimes into one trajectory"
            )
        missing = sorted(self.required_components - set(record.get("components", {})))
        if missing:
            raise SnapshotIncomplete(
                f"stored snapshot is missing required components {missing}; refusing to "
                "resume with empty stand-ins"
            )
        return record


# ---- Run start classification -----------------------------------------------

FRESH = "fresh"
RESUME_IN_PLACE = "resume_in_place"
COMPLETE = "complete"


def classify_run_start(
    *,
    result_path: Path | str,
    lease_path: Path | str,
    run_id: str,
    worker_id: str,
    lease_seconds: float,
    clock: Callable[[], float] = time.time,
) -> tuple[str, WorkerLease | None]:
    """Decide what this worker is allowed to do, and take the lease if allowed.

    Exactly three outcomes, and ``resume_in_place`` is never spelled as a retry or
    a backfill: a retry re-runs from scratch and could double-charge, a backfill
    invents history, while a resume continues one run monotonically at its own
    unchanged budget.  A live lease raises ``LeaseHeld`` instead of returning, so a
    caller cannot mistake "blocked" for "start fresh".
    """
    if Path(result_path).exists():
        return COMPLETE, None
    lease_exists = Path(lease_path).exists()
    lease = WorkerLease.acquire(
        lease_path,
        run_id=run_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        clock=clock,
    )
    return (RESUME_IN_PLACE if lease_exists else FRESH), lease
