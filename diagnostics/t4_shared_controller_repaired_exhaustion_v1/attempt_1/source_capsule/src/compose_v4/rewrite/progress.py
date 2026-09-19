"""Endpoint-conditioned CTMCs on validity-closed rewrite traces."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, exp, log1p
from collections.abc import Iterator

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.rewrite.kernel import RewriteSystem, default_rewrite_system
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace


@dataclass(frozen=True)
class PowerSurvivalScheduler:
    """Monotone scheduler alpha(t) = 1 - (1 - t)^power."""

    power: float = 1.0

    def __post_init__(self) -> None:
        if self.power <= 0:
            raise ValueError("scheduler power must be positive")

    @staticmethod
    def _check_time(t: float) -> float:
        t = float(t)
        if not 0.0 <= t <= 1.0:
            raise ValueError("time must lie in [0, 1]")
        return t

    def alpha(self, t: float) -> float:
        t = self._check_time(t)
        return 1.0 - (1.0 - t) ** self.power

    def derivative(self, t: float) -> float:
        t = self._check_time(t)
        if t == 1.0 and self.power < 1.0:
            return float("inf")
        return self.power * (1.0 - t) ** (self.power - 1.0)

    def event_hazard(self, t: float) -> float:
        """Hazard for each not-yet-executed instruction."""

        t = self._check_time(t)
        if t == 1.0:
            return float("inf")
        return self.power / (1.0 - t)

    def operational_time(self, t: float) -> float:
        """Map physical time to cumulative per-instruction hazard."""

        t = self._check_time(t)
        if t == 1.0:
            return float("inf")
        return -self.power * log1p(-t)

    def time_from_operational(self, operational_time: float) -> float:
        """Inverse of :meth:`operational_time` on ``[0, infinity)``."""

        operational_time = float(operational_time)
        if operational_time < 0:
            raise ValueError("operational time must be non-negative")
        return 1.0 - exp(-operational_time / self.power)


@dataclass(frozen=True)
class ConditionalRewriteSample:
    """One conditional Generator Matching training sample."""

    time: float
    progress: int
    state: MolecularGraph
    next_step: RewriteStep | None
    next_successor: MolecularGraph | None
    teacher_rate: float
    path_length: int


@dataclass(frozen=True)
class PackedMolecularGraph:
    """Byte-packed checkpoint for a padded molecular graph.

    Molecular graph categories fit in signed bytes for the current chemistry
    vocabulary.  Packing avoids pickling four separate int32 NumPy arrays for
    every retained path state, while unpacking restores the exact public
    ``MolecularGraph`` representation.
    """

    n_slots: int
    payload: bytes

    @classmethod
    def from_graph(cls, state: MolecularGraph) -> "PackedMolecularGraph":
        n_slots = int(state.n_atoms)
        arrays = (
            state.atom_types,
            state.formal_charges,
            state.implicit_h_counts,
            state.bonds,
        )
        if any(
            int(array.min(initial=0)) < -128 or int(array.max(initial=0)) > 127
            for array in arrays
        ):
            raise ValueError("molecular graph category exceeds int8 packing range")
        payload = b"".join(
            np.asarray(array, dtype=np.int8).tobytes(order="C")
            for array in arrays
        )
        expected = n_slots * n_slots + 3 * n_slots
        if len(payload) != expected:
            raise RuntimeError("packed molecular graph has unexpected byte length")
        return cls(n_slots=n_slots, payload=payload)

    def unpack(self) -> MolecularGraph:
        n_slots = int(self.n_slots)
        expected = n_slots * n_slots + 3 * n_slots
        if len(self.payload) != expected:
            raise ValueError("packed molecular graph payload is truncated")
        values = np.frombuffer(self.payload, dtype=np.int8)
        atom_types = values[:n_slots].astype(np.int32)
        formal_charges = values[n_slots : 2 * n_slots].astype(np.int32)
        implicit_h_counts = values[2 * n_slots : 3 * n_slots].astype(np.int32)
        bonds = values[3 * n_slots :].reshape(n_slots, n_slots).astype(np.int32)
        return MolecularGraph(atom_types, formal_charges, implicit_h_counts, bonds)


class TraceProgressCTMC:
    """Pure-birth CTMC that reveals a legal rewrite trace by time one.

    If a trace has K instructions, the progress marginal is
    N_t ~ Binomial(K, alpha(t)). At progress k < K, the only conditional jump
    is the next trace instruction, with rate

        (K - k) alpha'(t) / (1 - alpha(t)).

    The trace is privileged teacher information. It is never part of the
    model-facing state and is not available to the production sampler.
    """

    def __init__(
        self,
        trace: RewriteTrace,
        *,
        scheduler: PowerSurvivalScheduler | None = None,
        system: RewriteSystem | None = None,
        checkpoint_interval: int | None = None,
    ) -> None:
        self.trace = trace
        self.scheduler = scheduler or PowerSurvivalScheduler()
        self.system = system or default_rewrite_system()
        if checkpoint_interval is not None and int(checkpoint_interval) <= 0:
            raise ValueError("checkpoint_interval must be positive or None")
        self.checkpoint_interval = (
            None if checkpoint_interval is None else int(checkpoint_interval)
        )
        state = trace.source
        states = [state] if self.checkpoint_interval is None else None
        checkpoints: list[tuple[int, PackedMolecularGraph]] = []
        for progress, step in enumerate(trace.steps, start=1):
            state = self.system.apply(state, step.rule_name, step.action)
            if states is not None:
                states.append(state)
            elif (
                progress < len(trace.steps)
                and progress % int(self.checkpoint_interval) == 0
            ):
                checkpoints.append((progress, PackedMolecularGraph.from_graph(state)))
        endpoint = state
        if not _same_state(endpoint, trace.target):
            raise ValueError("trace endpoint does not equal its declared target")
        self._states = None if states is None else tuple(states)
        self._checkpoints = tuple(checkpoints)

    @property
    def path_length(self) -> int:
        return len(self.trace.steps)

    @property
    def states(self) -> tuple[MolecularGraph, ...]:
        if self._states is not None:
            return self._states
        return tuple(self.iter_states())

    def state_at(self, progress: int) -> MolecularGraph:
        self._check_progress(progress)
        if self._states is not None:
            return self._states[progress]
        if progress == 0:
            return self.trace.source
        if progress == self.path_length:
            return self.trace.target
        checkpoint_progress = 0
        state = self.trace.source
        for candidate_progress, packed in self._checkpoints:
            if candidate_progress > progress:
                break
            checkpoint_progress = int(candidate_progress)
            state = packed.unpack()
        for step in self.trace.steps[checkpoint_progress:progress]:
            state = self.system.apply(state, step.rule_name, step.action)
        return state

    def iter_states(self) -> Iterator[MolecularGraph]:
        """Yield path states in one streaming replay without retaining them."""

        if self._states is not None:
            yield from self._states
            return
        state = self.trace.source
        yield state
        for step in self.trace.steps:
            state = self.system.apply(state, step.rule_name, step.action)
            yield state

    @property
    def retained_checkpoint_count(self) -> int:
        return len(self._checkpoints)

    @property
    def packed_checkpoint_bytes(self) -> int:
        return sum(len(packed.payload) for _, packed in self._checkpoints)

    def marginal(self, t: float) -> np.ndarray:
        """Closed-form probability of every latent progress state."""

        alpha = self.scheduler.alpha(t)
        length = self.path_length
        if length == 0:
            return np.ones(1, dtype=np.float64)
        if alpha == 0.0:
            result = np.zeros(length + 1, dtype=np.float64)
            result[0] = 1.0
            return result
        if alpha == 1.0:
            result = np.zeros(length + 1, dtype=np.float64)
            result[-1] = 1.0
            return result
        return np.asarray(
            [
                comb(length, k)
                * alpha**k
                * (1.0 - alpha) ** (length - k)
                for k in range(length + 1)
            ],
            dtype=np.float64,
        )

    def jump_rate(self, progress: int, t: float) -> float:
        """Conditional rate of the single next instruction."""

        self._check_progress(progress)
        remaining = self.path_length - progress
        if remaining == 0:
            return 0.0
        return remaining * self.scheduler.event_hazard(t)

    def sample_progress(self, t: float, rng: np.random.Generator) -> int:
        alpha = self.scheduler.alpha(t)
        return int(rng.binomial(self.path_length, alpha))

    def sample(self, t: float, rng: np.random.Generator) -> ConditionalRewriteSample:
        progress = self.sample_progress(t, rng)
        state = self.state_at(progress)
        if progress == self.path_length:
            return ConditionalRewriteSample(
                time=float(t),
                progress=progress,
                state=state,
                next_step=None,
                next_successor=None,
                teacher_rate=0.0,
                path_length=self.path_length,
            )
        return ConditionalRewriteSample(
            time=float(t),
            progress=progress,
            state=state,
            next_step=self.trace.steps[progress],
            next_successor=self.state_at(progress + 1),
            teacher_rate=self.jump_rate(progress, t),
            path_length=self.path_length,
        )

    def operational_jump_rate(self, progress: int) -> float:
        """Next-instruction rate after cumulative-hazard time change."""

        self._check_progress(progress)
        return float(self.path_length - progress)

    def sample_operational(
        self,
        operational_time: float,
        rng: np.random.Generator,
    ) -> ConditionalRewriteSample:
        """Sample in the non-singular cumulative-hazard clock.

        In this clock the next-instruction teacher rate is simply the number of
        remaining trace steps. ``sample.time`` remains the normalized physical
        time supplied to the neural model.
        """

        time = self.scheduler.time_from_operational(operational_time)
        progress = self.sample_progress(time, rng)
        state = self.state_at(progress)
        if progress == self.path_length:
            return ConditionalRewriteSample(
                time=time,
                progress=progress,
                state=state,
                next_step=None,
                next_successor=None,
                teacher_rate=0.0,
                path_length=self.path_length,
            )
        return ConditionalRewriteSample(
            time=time,
            progress=progress,
            state=state,
            next_step=self.trace.steps[progress],
            next_successor=self.state_at(progress + 1),
            teacher_rate=self.operational_jump_rate(progress),
            path_length=self.path_length,
        )

    def _check_progress(self, progress: int) -> None:
        if not isinstance(progress, (int, np.integer)):
            raise TypeError("progress must be an integer")
        if not 0 <= int(progress) <= self.path_length:
            raise ValueError("progress is outside the trace")


def _same_state(left: MolecularGraph, right: MolecularGraph) -> bool:
    return bool(
        np.array_equal(left.atom_types, right.atom_types)
        and np.array_equal(left.formal_charges, right.formal_charges)
        and np.array_equal(left.implicit_h_counts, right.implicit_h_counts)
        and np.array_equal(left.bonds, right.bonds)
    )
