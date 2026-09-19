"""Resumable best-witness planning on full executable reference rows.

This is an optimization heuristic, not an estimator of reference expectations.
Scheduling quanta pause work; they neither terminate a program nor create a
failed return. The legacy importance-weighted TaskSearch is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Callable, Hashable
from dataclasses import asdict, dataclass
from typing import Generic, TypeVar

import numpy as np

from compose_v4.control.continuation import ContinuationBudgetExceeded, _tilted
from compose_v4.control.task_search import SearchRow

State = TypeVar("State")
SCHEMA = "resumable_best_witness_v1"


def payload_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass
class FrontierWork:
    row_attempts: int = 0
    row_hits: int = 0
    endpoint_evaluations: int = 0
    planning_transitions: int = 0
    paths_started: int = 0
    paths_completed: int = 0
    dead_ends: int = 0
    paused_quanta: int = 0
    executor_interruptions: int = 0
    snapshot_invalidations: int = 0


class FrontierSearch(Generic[State]):
    """Sparse planning with monotone max backups of evaluated endpoint witnesses.

    State keys must include every transition-relevant variable. Reference rows
    depend only on reference_id; endpoint scores depend also on snapshot_id.
    Cached rows contain the FULL support, never sampled/truncated renormalizations.
    The caller retains a separate RNG and exact frontier for committed decisions.
    """

    def __init__(
        self,
        reference: Callable[[State], SearchRow[State]],
        endpoint: Callable[[State], float | None],
        terminal: Callable[[State], bool],
        state_key: Callable[[State], Hashable],
        *,
        reference_id: str,
        snapshot_id: str,
        seed: int,
    ):
        if not reference_id or not snapshot_id or type(seed) is not int or seed < 0:
            raise ValueError(
                "frontier requires reference/snapshot identities and a nonnegative seed"
            )
        self.reference, self.endpoint, self.terminal = reference, endpoint, terminal
        self.state_key = state_key
        self.reference_id, self.snapshot_id = reference_id, snapshot_id
        self.rng = np.random.default_rng(seed)
        self.states: list[State] = []
        self.ids: dict[Hashable, int] = {}
        self.rows: dict[int, SearchRow[State]] = {}
        self.parents: dict[int, set[int]] = {}
        self.leaves: dict[int, float | None] = {}
        self.best: dict[int, float] = {}
        self.cursor: int | None = None
        self.work = FrontierWork()

    def register(self, state: State) -> int:
        key = self.state_key(state)
        if key not in self.ids:
            self.ids[key] = len(self.states)
            self.states.append(state)
        return self.ids[key]

    def _backup(self, node: int, value: float):
        pending = deque([(node, value)])
        while pending:
            current, candidate = pending.popleft()
            if candidate <= self.best.get(current, 0.0):
                continue
            self.best[current] = candidate
            pending.extend((p, candidate) for p in sorted(self.parents.get(current, ())))

    def _evaluate(self, node: int):
        if node in self.leaves:
            return
        value = self.endpoint(self.states[node])
        if value is not None:
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("endpoint desirability must be finite in [0, 1], or None")
            value = float(value)
            self.work.endpoint_evaluations += 1
            self._backup(node, value)
        self.leaves[node] = value

    def row(self, state: State) -> SearchRow[State]:
        node = self.register(state)
        if node not in self.rows:
            self.work.row_attempts += 1
            row = self.reference(state)
            if not isinstance(row, SearchRow):
                raise TypeError("frontier reference must return a full SearchRow")
            self.rows[node] = row
            for child, probability in zip(row.successors, row.reference, strict=True):
                child_id = self.register(child)
                if probability > 0:
                    self.parents.setdefault(child_id, set()).add(node)
                    self._backup(node, self.best.get(child_id, 0.0))
        else:
            self.work.row_hits += 1
        return self.rows[node]

    def decision(self, state: State) -> dict:
        row = self.row(state)
        if not row.successors:
            return {"status": "no_admissible_action", "probabilities": [], "kl": 0.0}
        children = [self.register(s) for s in row.successors]
        # Exact row products are valid witnessed successors. Scoring a completed
        # option is cheap and does not require finishing a remaining full rollout.
        for child in children:
            self._evaluate(child)
        witness = np.asarray([self.best.get(child, 0.0) for child in children])
        values = (1.0 + witness) / 2.0
        controlled, eta, _ = _tilted(row.core, values, kappa=1.0, exploration=0.0)
        q = (1 - row.epsilon) * controlled + row.epsilon * np.asarray(row.floor)
        p = row.reference
        live = q > 0
        kl = float(np.sum(q[live] * np.log(q[live] / p[live])))
        if (
            not np.isclose(q.sum(), 1, atol=1e-12)
            or kl > 1 + 1e-10
            or np.any(q < row.epsilon * np.asarray(row.floor) - 1e-12)
            or np.any(live & (p == 0))
        ):
            raise RuntimeError("frontier full-support, exploration-floor or KL invariant violated")
        return {
            "status": "task_guided" if np.ptp(witness) > 1e-12 else "reference_no_contrast",
            "probabilities": q.tolist(),
            "reference": p.tolist(),
            "core": list(row.core),
            "floor": list(row.floor),
            "epsilon": row.epsilon,
            "values": values.tolist(),
            "best_witness": witness.tolist(),
            "labels": list(row.labels),
            "kl": kl,
            "eta": eta,
            "snapshot_id": self.snapshot_id,
            "value_semantics": "best_evaluated_endpoint_witness_not_expected_return",
        }

    def advance(self, root: State, transitions: int) -> dict:
        if type(transitions) is not int or transitions < 0:
            raise ValueError("planning quantum must be a nonnegative transition count")
        used, status = 0, "quantum_complete"
        try:
            while used < transitions:
                if self.cursor is None:
                    if self.terminal(root):
                        status = "terminal_root"
                        break
                    self.cursor = self.register(root)
                    self.work.paths_started += 1
                current = self.states[self.cursor]
                self._evaluate(self.cursor)
                if self.terminal(current):
                    self.cursor = None
                    self.work.paths_completed += 1
                    continue
                row = self.row(current)
                if not row.successors:
                    self.cursor = None
                    self.work.dead_ends += 1
                    status = "no_admissible_action"
                    break
                decision = self.decision(current)
                index = int(self.rng.choice(len(row.successors), p=decision["probabilities"]))
                self.cursor = self.register(row.successors[index])
                self.work.planning_transitions += 1
                used += 1
                self._evaluate(self.cursor)
                if self.terminal(self.states[self.cursor]):
                    self.cursor = None
                    self.work.paths_completed += 1
        except ContinuationBudgetExceeded:
            self.work.executor_interruptions += 1
            status = "executor_paused"
        if self.cursor is not None:
            self.work.paused_quanta += 1
        return {"status": status, "transitions": used, **self.receipt()}

    def receipt(self) -> dict:
        return {
            "schema_version": SCHEMA,
            "reference_id": self.reference_id,
            "snapshot_id": self.snapshot_id,
            **asdict(self.work),
            "cached_rows": len(self.rows),
            "exact_states": len(self.states),
            "cursor_pending": self.cursor is not None,
            "positive_witness_states": len(self.best),
            "exact_doob": False,
            "uncertainty_calibrated": False,
        }

    def checkpoint(self, encode: Callable[[State], dict]) -> dict:
        payload = {
            "schema_version": SCHEMA,
            "reference_id": self.reference_id,
            "snapshot_id": self.snapshot_id,
            "states": [encode(s) for s in self.states],
            "rows": [
                {
                    "node": node,
                    "children": [self.ids[self.state_key(s)] for s in row.successors],
                    "labels": list(row.labels),
                    "core": list(row.core),
                    "floor": list(row.floor),
                    "epsilon": row.epsilon,
                }
                for node, row in sorted(self.rows.items())
            ],
            "leaves": [[node, value] for node, value in sorted(self.leaves.items())],
            "cursor": self.cursor,
            "rng_state": self.rng.bit_generator.state,
            "work": asdict(self.work),
        }
        return {**payload, "checkpoint_sha256": payload_hash(payload)}

    def restore(self, checkpoint: dict, decode: Callable[[dict], State]):
        """Load before use. New task snapshots discard scores but keep exact work."""
        payload = {k: v for k, v in checkpoint.items() if k != "checkpoint_sha256"}
        if (
            self.states
            or payload.get("schema_version") != SCHEMA
            or payload.get("reference_id") != self.reference_id
            or payload_hash(payload) != checkpoint.get("checkpoint_sha256")
        ):
            raise ValueError("frontier checkpoint schema, reference, hash or empty-target mismatch")
        for record in payload["states"]:
            expected = len(self.states)
            if self.register(decode(record)) != expected:
                raise ValueError("checkpoint has duplicate exact augmented states")

        def node_id(value):
            if type(value) is not int or not 0 <= value < len(self.states):
                raise ValueError(f"checkpoint has invalid state index {value!r}")
            return value

        for record in payload["rows"]:
            node = node_id(record["node"])
            if node in self.rows:
                raise ValueError("checkpoint contains a duplicate cached row")
            child_ids = [node_id(i) for i in record["children"]]
            self.rows[node] = SearchRow(
                tuple(self.states[i] for i in child_ids),
                tuple(record["labels"]),
                tuple(record["core"]),
                tuple(record["floor"]),
                record["epsilon"],
            )
            for child, probability in zip(child_ids, self.rows[node].reference, strict=True):
                if probability > 0:
                    self.parents.setdefault(child, set()).add(node)
        self.cursor = None if payload["cursor"] is None else node_id(payload["cursor"])
        self.rng.bit_generator.state = payload["rng_state"]
        if any(type(v) is not int or v < 0 for v in payload["work"].values()):
            raise ValueError("checkpoint contains malformed work counters")
        self.work = FrontierWork(**payload["work"])
        for node, value in payload["leaves"]:
            node_id(node)
            if node in self.leaves or (
                value is not None
                and (isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1)
            ):
                raise ValueError("checkpoint contains malformed/duplicate endpoint scores")
            self.leaves[node] = value
        if payload["snapshot_id"] != self.snapshot_id:
            evaluated = [node for node, value in self.leaves.items() if value is not None]
            self.leaves.clear()
            self.work.snapshot_invalidations += 1
            # Refresh the already discovered endpoints, without re-enumerating
            # their paths. Otherwise a model update would erase useful memory.
            for node in evaluated:
                self._evaluate(node)
        else:
            for node, value in self.leaves.items():
                if value is not None:
                    self._backup(node, value)
