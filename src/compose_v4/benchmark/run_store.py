"""Durable, resumable Task 3 runs. A crash at call 8,700 must lose ~nothing.

WHY THIS IS PART OF THE BENCHMARK AND NOT PART OF A POLICY
----------------------------------------------------------
Two runs have already been destroyed on this project by holding expensive work
in memory until the end of a job that was then stopped -- 93.5 core-hours, twice.
The lesson was not "be careful"; it was that durability has to be a property of
the harness, so that no policy can be written that lacks it.  A policy here
cannot accidentally hold 10,000 oracle evaluations in RAM: every charged
molecule is on disk before the next one is asked for.

THE THREE THINGS THAT MUST SURVIVE A KILL -9
--------------------------------------------
1. THE LEDGER -- every charged (canonical molecule -> five objectives), append
   only, one JSON object per line, with a contiguous sequence number.  This IS
   the budget record: `spent` on resume is derived from it, not trusted from a
   counter someone remembered to save.
2. THE CHECKPOINT -- RNG state, archive, step, and whatever the policy needs to
   continue.  Written to a temporary file, fsynced, then renamed over the
   previous one, so an interrupted write leaves the OLD checkpoint intact rather
   than a half-written new one.
3. THE LINK BETWEEN THEM -- a checkpoint records the ledger's byte length and
   the sha256 of that prefix.  On resume the prefix is verified, so a ledger
   that was rolled back, truncated by a filesystem, or edited cannot be silently
   replayed into a run that then reports a budget it never spent.

WHAT A CRASH ACTUALLY COSTS
---------------------------
Everything up to the last fsync survives.  Beyond that the loss is bounded by
`fsync_every` evaluations -- and even those are cheap to redo, because the
oracle is a deterministic function of the molecule.  What is NOT recoverable is
the policy's own progress since the last checkpoint, which is why
`checkpoint()` is cheap enough to call every generation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

LEDGER_NAME = "evaluations.jsonl"
CHECKPOINT_NAME = "checkpoint.json"
PREVIOUS_CHECKPOINT_NAME = "checkpoint.previous.json"
IDENTITY_NAME = "run.json"


def atomic_write(path: Path, payload: bytes) -> None:
    """Write-then-rename, with both file and directory fsynced.

    Without the directory fsync the rename itself can be lost on power failure,
    which would leave the old file in place -- survivable -- or, on some
    filesystems, an empty one -- not.
    """

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_prefix(path: Path, length: int) -> str:
    """sha256 of the first `length` bytes of a file."""

    h = hashlib.sha256()
    remaining = length
    with open(path, "rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(1 << 20, remaining))
            if not chunk:
                raise ValueError(
                    f"{path.name} is {path.stat().st_size} bytes but a checkpoint "
                    f"expects at least {length}: the ledger has been truncated")
            remaining -= len(chunk)
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ResumeState:
    """What a resumed run gets back. Everything else must be re-derived."""

    evaluations: dict[str, tuple[float, ...]]
    step: int
    archive: list[str]
    rng_state: dict | None
    policy_state: dict
    ledger_bytes: int
    #: Evaluations found in the ledger BEYOND the last checkpoint. They are
    #: real, paid-for oracle calls and are honoured -- but the policy that
    #: requested them did not survive to use them, so a policy that wants to
    #: know it is in that situation can ask.
    uncheckpointed: int = 0

    @property
    def spent(self) -> int:
        return len(self.evaluations)


@dataclass
class RunStore:
    """One seed's durable state. Open it once; it owns the files until closed."""

    root: Path
    seed: int
    budget: int
    policy: str
    #: How often the ledger is forced to stable storage. Every record is the
    #: safest and costs an fsync per evaluation (~1 ms); 25 bounds the loss at
    #: 25 re-derivable evaluations and makes the cost invisible.
    fsync_every: int = 25
    metadata: dict = field(default_factory=dict)

    _handle: Any = field(default=None, init=False, repr=False)
    _since_fsync: int = field(default=0, init=False, repr=False)
    _sequence: int = field(default=0, init=False, repr=False)

    # ---- lifecycle -------------------------------------------------------

    @classmethod
    def open(cls, root: Path, *, seed: int, budget: int, policy: str,
             fsync_every: int = 25, metadata: dict | None = None,
             resume: bool = True) -> tuple[RunStore, ResumeState | None]:
        """Open a run directory, resuming from disk if there is anything there."""

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        store = cls(root=root, seed=seed, budget=budget, policy=policy,
                    fsync_every=fsync_every, metadata=metadata or {})
        state = store._load() if resume else None
        store._write_identity(resumed=state is not None)
        store._handle = open(store.ledger_path, "a", buffering=1)
        store._sequence = 0 if state is None else state.spent
        return store, state

    @property
    def ledger_path(self) -> Path:
        return self.root / LEDGER_NAME

    @property
    def checkpoint_path(self) -> Path:
        return self.root / CHECKPOINT_NAME

    def _write_identity(self, *, resumed: bool) -> None:
        identity = {
            "seed": self.seed, "budget": self.budget, "policy": self.policy,
            "opened": time.time(), "resumed": resumed, **self.metadata,
        }
        history = []
        path = self.root / IDENTITY_NAME
        if path.exists():
            history = json.loads(path.read_text()).get("history", [])
        identity["history"] = history + [{"opened": identity["opened"],
                                          "resumed": resumed}]
        atomic_write(path, json.dumps(identity, indent=1).encode())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- the ledger ------------------------------------------------------

    def record(self, smiles: str, values: tuple[float, ...]) -> None:
        """Append one charged evaluation. Wire this to `OracleMeter.on_evaluated`.

        Repr-round-trip floats (`repr` is exact for float64 in Python 3) so a
        resumed run's cached values are bit-identical to the ones the crashed
        run computed. An approximate ledger would make resume a different
        experiment.
        """

        line = json.dumps({"n": self._sequence, "smiles": smiles,
                           "v": [float(v) for v in values]},
                          separators=(",", ":"))
        self._handle.write(line + "\n")
        self._sequence += 1
        self._since_fsync += 1
        if self._since_fsync >= self.fsync_every:
            self.flush()

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._since_fsync = 0

    def _read_ledger(self) -> tuple[dict[str, tuple[float, ...]], int]:
        """Parse the ledger, tolerating exactly one truncated final line.

        A process killed mid-write leaves a partial last line. That is the ONE
        corruption that is expected and forgivable, because the record it
        represents was never acknowledged. Anything else -- a bad line in the
        middle, a gap in the sequence -- means the file is not what it claims
        and raises.
        """

        path = self.ledger_path
        if not path.exists():
            return {}, 0
        evaluations: dict[str, tuple[float, ...]] = {}
        good_bytes = 0
        expected = 0
        with open(path, "rb") as handle:
            raw = handle.read()
        lines = raw.split(b"\n")
        # A well-formed file ends with a newline, so the final split element is
        # empty; anything else is a partial write.
        trailing_partial = lines[-1] != b""
        for index, line in enumerate(lines[:-1]):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name} line {index + 1} is corrupt and is not the "
                    f"final line, so it was acknowledged: {error}") from error
            if record["n"] != expected:
                raise ValueError(
                    f"{path.name} jumps from sequence {expected - 1} to "
                    f"{record['n']} at line {index + 1}: records are missing")
            if record["smiles"] in evaluations:
                # The meter charges once per canonical molecule and only
                # records what it charged, so a repeat means the ledger and the
                # budget no longer describe the same run.
                raise ValueError(
                    f"{path.name} records {record['smiles']!r} twice at line "
                    f"{index + 1}; the budget was charged once")
            evaluations[record["smiles"]] = tuple(record["v"])
            expected += 1
            good_bytes += len(line) + 1
        if trailing_partial:
            # Truncate the unacknowledged remainder so the append handle starts
            # at a record boundary. Leaving it would corrupt the NEXT line.
            with open(path, "r+b") as handle:
                handle.truncate(good_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        return evaluations, good_bytes

    # ---- checkpoints -----------------------------------------------------

    def checkpoint(self, *, step: int, archive: list[str],
                   rng: np.random.Generator | None = None,
                   policy_state: dict | None = None,
                   arrays: dict[str, np.ndarray] | None = None) -> None:
        """Snapshot the policy atomically, pinned to a ledger prefix.

        Cheap by design: call it every generation. The ledger is flushed first,
        so a checkpoint never claims a prefix that is not yet on disk.
        """

        self.flush()
        length = self.ledger_path.stat().st_size if self.ledger_path.exists() else 0
        if arrays:
            # Serialised in memory and then written atomically, so an
            # interrupted checkpoint cannot leave a half-written npz that the
            # next resume would load as if it were complete.
            buffer = io.BytesIO()
            np.savez(buffer, **arrays)
            atomic_write(self.root / "checkpoint_arrays.npz", buffer.getvalue())

        payload = {
            "step": step,
            "archive": list(archive),
            "rng_state": _rng_state(rng),
            "policy_state": policy_state or {},
            "ledger_bytes": length,
            "ledger_sha256": sha256_prefix(self.ledger_path, length) if length else "",
            "n_evaluated": self._sequence,
            "arrays": bool(arrays),
            "written": time.time(),
        }
        body = json.dumps(payload, separators=(",", ":")).encode()
        document = json.dumps({"sha256": sha256_bytes(body),
                               "payload": payload}, indent=1).encode()
        # Keep the previous checkpoint: if the machine dies during the rename
        # AND the new file is unreadable, there is still a good one to resume
        # from, one generation older.
        if self.checkpoint_path.exists():
            atomic_write(self.root / PREVIOUS_CHECKPOINT_NAME,
                         self.checkpoint_path.read_bytes())
        atomic_write(self.checkpoint_path, document)

    def _load_checkpoint(self) -> dict | None:
        for name in (CHECKPOINT_NAME, PREVIOUS_CHECKPOINT_NAME):
            path = self.root / name
            if not path.exists():
                continue
            try:
                document = json.loads(path.read_text())
                body = json.dumps(document["payload"],
                                  separators=(",", ":")).encode()
                if sha256_bytes(body) != document["sha256"]:
                    raise ValueError("checkpoint payload does not match its digest")
                return document["payload"]
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                if name == PREVIOUS_CHECKPOINT_NAME:
                    raise ValueError(
                        f"both checkpoints in {self.root} are unreadable") from error
                continue
        return None

    def _load(self) -> ResumeState | None:
        evaluations, ledger_bytes = self._read_ledger()
        checkpoint = self._load_checkpoint()
        if checkpoint is None:
            if not evaluations:
                return None
            # A ledger with no checkpoint is still worth every call it paid for.
            return ResumeState(evaluations=evaluations, step=0, archive=[],
                               rng_state=None, policy_state={},
                               ledger_bytes=ledger_bytes,
                               uncheckpointed=len(evaluations))
        claimed = checkpoint["ledger_bytes"]
        if claimed > ledger_bytes:
            raise ValueError(
                f"the checkpoint in {self.root} was written against "
                f"{claimed} ledger bytes but only {ledger_bytes} survive. The "
                f"ledger has gone backwards; this run cannot be resumed honestly.")
        if claimed and sha256_prefix(self.ledger_path, claimed) != checkpoint["ledger_sha256"]:
            raise ValueError(
                f"the ledger prefix in {self.root} does not hash to what the "
                f"checkpoint recorded: it has been modified, not merely extended.")
        return ResumeState(
            evaluations=evaluations,
            step=int(checkpoint["step"]),
            archive=list(checkpoint["archive"]),
            rng_state=checkpoint["rng_state"],
            policy_state=checkpoint["policy_state"],
            ledger_bytes=ledger_bytes,
            uncheckpointed=len(evaluations) - int(checkpoint["n_evaluated"]),
        )

    def load_arrays(self) -> dict[str, np.ndarray]:
        path = self.root / "checkpoint_arrays.npz"
        if not path.exists():
            return {}
        with np.load(path) as data:
            return {key: data[key] for key in data.files}


def _rng_state(rng: np.random.Generator | None) -> dict | None:
    """numpy's bit-generator state, JSON-safe.

    The state contains 128-bit integers; JSON handles them exactly, whereas a
    float round-trip would not. Restoring it is what makes a resumed run the
    SAME run rather than a similar one.
    """

    if rng is None:
        return None
    state = rng.bit_generator.state
    return json.loads(json.dumps(state, default=int))


def restore_rng(state: dict | None) -> np.random.Generator | None:
    """Rebuild the exact generator the checkpoint came from, not a fresh one."""

    if state is None:
        return None
    name = state.get("bit_generator", "PCG64")
    bit_generator = getattr(np.random, name, None)
    if bit_generator is None:
        raise ValueError(f"unknown bit generator {name!r} in checkpoint")
    generator = np.random.Generator(bit_generator())
    generator.bit_generator.state = state
    return generator


def iter_ledger(path: Path) -> Iterator[dict]:
    """Stream a finished run's evaluations without loading it all."""

    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
