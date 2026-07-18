"""Partial-order conditional CTMCs for monotone molecular construction traces.

The sequential compiler emits one valid linearization.  For typed construction
traces, the actual causal dependencies are simpler: an event must wait for the
events that create its anchors, while independent branch events may occur in
either order.  This module constructs that dependency poset and defines a pure-
birth CTMC on its ideals.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.rewrite.kernel import RewriteSystem, default_rewrite_system
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.progress import PowerSurvivalScheduler
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.tracelets import CycleAttach, CycleInsert, RingEarInsert


class CausalTraceError(ValueError):
    """Raised when a trace is not a supported monotone construction program."""


@dataclass(frozen=True)
class CausalTraceSample:
    time: float
    completed: frozenset[int]
    state: MolecularGraph
    frontier_indices: tuple[int, ...]
    frontier_steps: tuple[RewriteStep, ...]
    frontier_successors: tuple[MolecularGraph, ...]
    teacher_rates: tuple[float, ...]
    path_length: int


class CausalTraceCTMC:
    """Pure-birth CTMC over ideals of a construction dependency poset.

    At an ideal containing ``k`` of ``K`` events, total operational hazard is
    ``K-k`` and is divided uniformly across the currently enabled causal
    frontier.  Therefore the event-count marginal remains Binomial(K, alpha(t))
    exactly, while arbitrary linearization choices are removed from the
    conditional teacher.
    """

    def __init__(
        self,
        trace: RewriteTrace,
        *,
        scheduler: PowerSurvivalScheduler | None = None,
        system: RewriteSystem | None = None,
    ) -> None:
        self.trace = trace
        self.scheduler = scheduler or PowerSurvivalScheduler()
        self.system = system or default_rewrite_system()
        self.dependencies = construction_dependencies(trace)

    @property
    def path_length(self) -> int:
        return len(self.trace.steps)

    def frontier(self, completed: frozenset[int]) -> tuple[int, ...]:
        self._check_ideal(completed)
        return tuple(
            index
            for index, dependencies in enumerate(self.dependencies)
            if index not in completed and dependencies.issubset(completed)
        )

    def state_for_order(self, order: tuple[int, ...]) -> MolecularGraph:
        completed: set[int] = set()
        state = self.trace.source
        for index in order:
            if index in completed or not 0 <= int(index) < self.path_length:
                raise CausalTraceError("causal order repeats or misindexes an event")
            if not self.dependencies[index].issubset(completed):
                raise CausalTraceError("causal order violates an event dependency")
            step = self.trace.steps[index]
            state = self.system.apply(state, step.rule_name, step.action)
            completed.add(int(index))
        return state

    def sample_ideal(
        self,
        progress: int,
        rng: np.random.Generator,
    ) -> tuple[frozenset[int], tuple[int, ...], MolecularGraph]:
        completed, order = self.sample_ideal_indices(progress, rng)
        return completed, order, self.state_for_order(order)

    def sample_ideal_indices(
        self,
        progress: int,
        rng: np.random.Generator,
    ) -> tuple[frozenset[int], tuple[int, ...]]:
        """Sample a causal ideal without executing its chemistry."""

        if not 0 <= int(progress) <= self.path_length:
            raise ValueError("progress is outside the causal trace")
        completed: set[int] = set()
        order: list[int] = []
        for _ in range(int(progress)):
            frontier = self.frontier(frozenset(completed))
            if not frontier:
                raise CausalTraceError("nonterminal causal ideal has an empty frontier")
            index = int(frontier[int(rng.integers(0, len(frontier)))])
            completed.add(index)
            order.append(index)
        return frozenset(completed), tuple(order)

    def sample_progress(self, t: float, rng: np.random.Generator) -> int:
        return int(rng.binomial(self.path_length, self.scheduler.alpha(t)))

    def sample(self, t: float, rng: np.random.Generator) -> CausalTraceSample:
        progress = self.sample_progress(t, rng)
        return self.sample_at_progress(
            progress,
            time=float(t),
            rng=rng,
            operational=False,
        )

    def sample_at_progress(
        self,
        progress: int,
        *,
        time: float,
        rng: np.random.Generator,
        operational: bool,
    ) -> CausalTraceSample:
        completed, _, state = self.sample_ideal(progress, rng)
        return self._sample_from_ideal(
            float(time),
            completed,
            state,
            operational=operational,
        )

    def sample_operational(
        self,
        operational_time: float,
        rng: np.random.Generator,
    ) -> CausalTraceSample:
        operational_time = float(operational_time)
        if operational_time < 0:
            raise ValueError("operational time must be non-negative")
        time = 1.0 - exp(-operational_time / self.scheduler.power)
        progress = self.sample_progress(time, rng)
        return self.sample_at_progress(
            progress,
            time=time,
            rng=rng,
            operational=True,
        )

    def _sample_from_ideal(
        self,
        time: float,
        completed: frozenset[int],
        state: MolecularGraph,
        *,
        operational: bool,
    ) -> CausalTraceSample:
        frontier = self.frontier(completed)
        steps = tuple(self.trace.steps[index] for index in frontier)
        successors = tuple(
            self.system.apply(state, step.rule_name, step.action) for step in steps
        )
        remaining = self.path_length - len(completed)
        if not frontier:
            rates: tuple[float, ...] = ()
        else:
            total = float(remaining)
            if not operational:
                total *= self.scheduler.event_hazard(time)
            rates = (total / len(frontier),) * len(frontier)
        return CausalTraceSample(
            time=time,
            completed=completed,
            state=state,
            frontier_indices=frontier,
            frontier_steps=steps,
            frontier_successors=successors,
            teacher_rates=rates,
            path_length=self.path_length,
        )

    def _check_ideal(self, completed: frozenset[int]) -> None:
        if any(not 0 <= int(index) < self.path_length for index in completed):
            raise CausalTraceError("causal ideal contains an invalid event index")
        for index in completed:
            if not self.dependencies[index].issubset(completed):
                raise CausalTraceError("completed events are not a causal ideal")


def construction_dependencies(
    trace: RewriteTrace,
) -> tuple[frozenset[int], ...]:
    """Infer creator-anchor dependencies for a monotone construction trace."""

    creator: dict[int, int] = {}
    source_slots = {
        int(v) for v in np.flatnonzero(is_element(trace.source.atom_types))
    }
    dependencies: list[frozenset[int]] = []
    for index, step in enumerate(trace.steps):
        required, created = _required_and_created_slots(step)
        missing = required - source_slots - creator.keys()
        if missing:
            raise CausalTraceError(
                f"event {index} requires atoms with no creator: {sorted(missing)}"
            )
        event_dependencies = frozenset(
            creator[slot] for slot in required if slot in creator
        )
        if any(slot in source_slots or slot in creator for slot in created):
            raise CausalTraceError("construction trace creates an existing atom twice")
        dependencies.append(event_dependencies)
        for slot in created:
            creator[slot] = index
    return tuple(dependencies)


def _required_and_created_slots(
    step: RewriteStep,
) -> tuple[set[int], set[int]]:
    action = step.action
    if isinstance(action, AtomInsert):
        return (
            {int(neighbor) for neighbor, _ in action.neighbors},
            {int(action.slot)},
        )
    if isinstance(action, CycleInsert):
        return set(), {int(atom.slot) for atom in action.atoms}
    if isinstance(action, CycleAttach):
        return (
            {int(action.anchor)},
            {int(atom.slot) for atom in action.atoms},
        )
    if isinstance(action, RingEarInsert):
        return (
            {int(action.a), int(action.b)},
            {int(atom.slot) for atom in action.atoms},
        )
    raise CausalTraceError(
        f"unsupported non-monotone construction event: {step.rule_name!r}"
    )


__all__ = [
    "CausalTraceCTMC",
    "CausalTraceError",
    "CausalTraceSample",
    "construction_dependencies",
]
