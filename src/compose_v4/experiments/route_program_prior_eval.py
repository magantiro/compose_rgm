"""Held-out comparison: teacher-route structural prior versus a generic control.

THE QUESTION, AND WHY IT IS NOT NEXT-ACTION ACCURACY
----------------------------------------------------
A previous cycle fitted a route actor that put the teacher's next decision first
91.6% of the time on fitted states and replayed all 77 teacher routes exactly,
while autonomous route recovery on held-out sources stayed near zero.
Recognition was far better than generation.  So this harness does not score
one-step imitation.  It asks whether a parent-conditioned prior GENERATES
complete coherent programs on molecules it has never seen, and the headline is
COMPLETE-PROGRAM YIELD.

TWO EXPERIMENTS, BECAUSE THE SEAM AND THE CLAIM ARE DIFFERENT SIZES
-------------------------------------------------------------------
``production`` -- ``synthesize_dynamic_program`` with ``region_law=None``
against the same call with the learned law.  This is maximum production
fidelity: the only difference between the arms is the region draw, and
``completed_module_count == requested_module_count`` is the synthesizer's own
completeness verdict.  Its treatment is deliberately narrow, because the region
law reaches only two of the thirteen module families
(``substituent_delete`` and ``segment_replace``).

``declared`` -- ``synthesize_named_module_sequence``, where the arm declares the
whole program shape UP FRONT (how many modules, which families) and the
synthesizer either executes all of it or refuses.  This is the experiment that
matches the claim: a prior over complete program structures is worth having only
if the structures it declares can actually be built on an unseen molecule.

MATCHING
--------
``K`` is drawn ONCE per (parent, draw) from the production module-count table
and handed to every ``declared`` arm, so no arm can win by declaring shorter and
therefore easier programs.  Every arm gets a fresh generator seeded from the
same (parent, draw) seed, so the arms see the same randomness budget.  The
learned ``K`` is reported as a coordinate but is NOT used in the matched
comparison; a separate arm exercises it.

WHAT IS MEASURED PER DRAW
-------------------------
legal execution (does the first declared module compile at all), complete
program (does the whole declared program compile, extract and replay), the
endpoint, the released region, the realized program length, and a
load-independent work counter -- module compile attempts -- reported beside wall
seconds because other jobs share this machine and seconds alone are not a cost.

ZERO ORACLE CALLS.  Nothing here evaluates docking, a PMO oracle, or any task
objective.  The T4 free gates are not applied either: this measures structural
generation, and folding an eligibility screen into it would confuse "can it
build a program" with "does the program pass a benchmark filter".
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.bridge_region_law import bridge_separated_regions
from compose_v4.control.dynamic_program_synthesis import (
    CAPACITY_AWARE_THRESHOLD,
    GENERIC_MODULES,
    MODULE_COUNT_PROBABILITIES,
    NEAR_CAPACITY_MODULE_COUNT_PROBABILITIES,
    module_count_distribution,
    synthesize_dynamic_program,
    synthesize_named_module_sequence,
)

SCHEMA_VERSION = "route_program_prior_eval_v1"

#: Slot capacity of the T4 proposal path -- `t4_fiber_campaign` pads every
#: parent to 48. This is NOT the 40 of the editing corpus, and it is NOT the
#: 40-heavy-atom ENDPOINT ceiling the fiber applies. Three distinct numbers
#: that have been conflated before; a TIGHT graph silently deletes the whole
#: `atom_insert` family from the legal support.
PROPOSAL_SLOTS = 48

#: Declared program length is capped by the production entry point, which
#: accepts one to three named modules.
MAX_DECLARED_MODULES = 3

#: The five exception types `t4_fiber_campaign.expand` swallows per draw. A
#: refusal here is a real refusal of the proposal, not a harness bug, so the
#: harness counts them by type instead of hiding them.
_PROPOSAL_REFUSALS = (ValueError, RuntimeError, KeyError, IndexError, TypeError)


@dataclass
class ArmTally:
    """Accumulated outcomes of one arm on one parent."""

    draws: int = 0
    legal: int = 0
    complete: int = 0
    compile_attempts: int = 0
    wall_seconds: float = 0.0
    endpoints: set = field(default_factory=set)
    family_multisets: set = field(default_factory=set)
    released_sizes: list = field(default_factory=list)
    changed_fractions: list = field(default_factory=list)
    primitive_counts: list = field(default_factory=list)
    module_counts: list = field(default_factory=list)
    teacher_region_hits: int = 0
    teacher_endpoint_hits: int = 0
    refusals: Counter = field(default_factory=Counter)

    def summary(self) -> dict:
        def stats(values):
            if not values:
                return {"n": 0}
            ordered = sorted(values)
            return {
                "n": len(ordered),
                "mean": float(np.mean(ordered)),
                "median": float(np.median(ordered)),
                "p90": float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]),
                "max": float(ordered[-1]),
            }

        return {
            "draws": self.draws,
            "legal_execution_rate": self.legal / self.draws if self.draws else 0.0,
            "complete_program_yield": self.complete / self.draws if self.draws else 0.0,
            "complete_programs": self.complete,
            "distinct_endpoints": len(self.endpoints),
            "distinct_endpoints_per_complete": (
                len(self.endpoints) / self.complete if self.complete else 0.0
            ),
            "distinct_family_multisets": len(self.family_multisets),
            "teacher_region_recall": (
                self.teacher_region_hits / self.complete if self.complete else 0.0
            ),
            "teacher_endpoint_hits": self.teacher_endpoint_hits,
            "endpoint_sample": sorted(self.endpoints)[:64],
            "released_region_size": stats(self.released_sizes),
            "changed_fraction": stats(self.changed_fractions),
            "primitive_count": stats(self.primitive_counts),
            "realized_module_count": stats(self.module_counts),
            "compile_attempts": self.compile_attempts,
            "wall_seconds": round(self.wall_seconds, 2),
            "refusals": dict(self.refusals.most_common(8)),
        }


def parent_state(smiles: str) -> MolecularGraph:
    """The proposal-path state of a parent: padded to 48 slots.

    A tight graph carries exactly `n_real_atoms` slots, which removes atom
    birth from the legal support entirely -- measured at 35-100% understatement
    of the mark count, and the cause of three earlier invalid measurements.
    """

    return pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)


def _real_slots(graph: MolecularGraph) -> set[int]:
    return {int(i) for i in np.flatnonzero(is_element(graph.atom_types))}


def _released(source: MolecularGraph, endpoint: MolecularGraph) -> set[int]:
    return _real_slots(source) - _real_slots(endpoint)


def _changed_fraction(trace: dict, source: MolecularGraph) -> float:
    """Fraction of source atoms the program actually touched.

    Computed from the executed trace's own action footprints, so it reports the
    realized transformation rather than the declared intent.
    """

    heavy = len(_real_slots(source))
    if heavy == 0:
        return 0.0
    touched: set[int] = set()
    for action in trace["actions"]:
        payload = action.get("payload") or {}
        for value in payload.values():
            if isinstance(value, int):
                touched.add(int(value))
            elif isinstance(value, (list, tuple)):
                touched.update(int(v) for v in value if isinstance(v, int))
    return len(touched & _real_slots(source)) / heavy


def _module_count(rng, source: MolecularGraph, *, maximum: int) -> int:
    """K from the PRODUCTION table -- the matched control for program length."""

    near_capacity = source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD
    table = (
        NEAR_CAPACITY_MODULE_COUNT_PROBABILITIES
        if near_capacity
        else MODULE_COUNT_PROBABILITIES
    )
    return int(
        rng.choice(
            np.arange(1, maximum + 1),
            p=module_count_distribution(table, maximum),
        )
    )


class _CountingLaw:
    """Wraps a law so the harness can count what the draw site consulted.

    It delegates rather than reimplements: a transcribed law could drift from
    the one under test and the comparison would not notice.
    """

    def __init__(self, law) -> None:
        self._law = law
        self.consulted = 0

    def order(self, graph, rng):
        self.consulted += 1
        return self._law.order(graph, rng)


def _record(
    tally: ArmTally,
    source: MolecularGraph,
    program,
    trace: dict,
    metadata: dict,
    *,
    teacher_regions: frozenset | None,
    teacher_endpoints: frozenset | None = None,
) -> None:
    """Record one complete program.

    The endpoint SMILES are KEPT, not just counted. A previous fragment
    evaluation stored aggregates plus five examples per shard, and when a
    metric turned out to be wrong the corrected quantities could be rebuilt
    only where counters happened to carry them -- uniqueness, quality and
    diversity were unrecoverable because the molecules were gone.
    """
    endpoint_state = trace["states"][-1]
    endpoint = (
        endpoint_state
        if isinstance(endpoint_state, MolecularGraph)
        else _decode(endpoint_state)
    )
    smiles = molecular_graph_to_smiles(endpoint)
    if smiles is not None:
        tally.endpoints.add(smiles)
    families = tuple(sorted(m["family"] for m in metadata.get("modules", ())))
    tally.family_multisets.add(families)
    released = _released(source, endpoint)
    tally.released_sizes.append(len(released))
    tally.changed_fractions.append(_changed_fraction(trace, source))
    tally.primitive_counts.append(len(program.marks))
    tally.module_counts.append(int(metadata.get("completed_module_count", 0)))
    if teacher_endpoints is not None and smiles is not None and smiles in teacher_endpoints:
        tally.teacher_endpoint_hits += 1
    if teacher_regions is not None and released:
        for region in teacher_regions:
            if released <= set(region):
                tally.teacher_region_hits += 1
                break


def _decode(state):
    from compose_v4.rewrite.trace_shard import decode_state

    return decode_state(state)


def teacher_region_targets(
    source: MolecularGraph, released_sets: list[set[int]]
) -> frozenset:
    """Bridge regions of this parent that contain a teacher's released set.

    This is the region law's own supervision target, so recovering it is the
    coordinate-level question the law is answerable for -- as distinct from
    recovering the teacher's endpoint, which no arm here is expected to do.
    """

    support = bridge_separated_regions(source, maximum=None)
    targets = set()
    for released in released_sets:
        best = None
        for region in support:
            fragment = set(region.fragment)
            if released <= fragment and (best is None or len(fragment) < len(best)):
                best = fragment
        if best is not None:
            targets.add(tuple(sorted(best)))
    return frozenset(targets)


def run_production_arm(
    source: MolecularGraph,
    *,
    draws: int,
    seed: int,
    region_law,
    teacher_regions: frozenset | None = None,
    teacher_endpoints: frozenset | None = None,
    max_modules: int = MAX_DECLARED_MODULES,
) -> ArmTally:
    """`synthesize_dynamic_program`, the shipped path, law on or off."""

    tally = ArmTally()
    law = _CountingLaw(region_law) if region_law is not None else None
    for draw in range(draws):
        rng = np.random.default_rng(seed + 101 * draw)
        started = time.monotonic()
        tally.draws += 1
        try:
            _src, program, _assign, trace, metadata = synthesize_dynamic_program(
                source, rng, max_modules=max_modules, region_law=law
            )
        except _PROPOSAL_REFUSALS as error:
            tally.refusals[f"{type(error).__name__}:{str(error)[:60]}"] += 1
            tally.wall_seconds += time.monotonic() - started
            continue
        tally.wall_seconds += time.monotonic() - started
        tally.legal += 1
        if int(metadata["completed_module_count"]) == int(
            metadata["requested_module_count"]
        ):
            tally.complete += 1
        _record(
            tally, source, program, trace, metadata,
            teacher_regions=teacher_regions, teacher_endpoints=teacher_endpoints,
        )
    tally.compile_attempts = law.consulted if law is not None else 0
    return tally


def run_declared_arm(
    source: MolecularGraph,
    *,
    draws: int,
    seed: int,
    families_for: callable,
    region_law,
    module_counts: list[int],
    teacher_regions: frozenset | None = None,
    teacher_endpoints: frozenset | None = None,
) -> ArmTally:
    """`synthesize_named_module_sequence` on a program shape declared up front.

    ``families_for(graph, rng, k)`` returns the declared family sequence. The
    arm is charged for a LEGAL execution when the one-module prefix of its own
    declaration compiles, and for a COMPLETE program when the whole declaration
    compiles, extracts and replays.
    """

    tally = ArmTally()
    for draw, module_count in zip(range(draws), module_counts):
        seeded = seed + 101 * draw
        families = families_for(source, np.random.default_rng(seeded), module_count)
        started = time.monotonic()
        tally.draws += 1
        tally.compile_attempts += len(families) + 1
        try:
            _src, program, _assign, trace, metadata = synthesize_named_module_sequence(
                source,
                np.random.default_rng(seeded + 7),
                families,
                region_law=region_law,
            )
        except _PROPOSAL_REFUSALS as error:
            tally.refusals[f"{type(error).__name__}:{str(error)[:60]}"] += 1
            tally.wall_seconds += time.monotonic() - started
            # Legality is the one-module prefix of this arm's OWN declaration,
            # so a failure at module 2 is still a legal execution at module 1.
            try:
                synthesize_named_module_sequence(
                    source,
                    np.random.default_rng(seeded + 7),
                    families[:1],
                    region_law=region_law,
                )
            except _PROPOSAL_REFUSALS:
                pass
            else:
                tally.legal += 1
            continue
        tally.wall_seconds += time.monotonic() - started
        tally.legal += 1
        tally.complete += 1
        _record(
            tally, source, program, trace, metadata,
            teacher_regions=teacher_regions, teacher_endpoints=teacher_endpoints,
        )
    return tally


def uniform_families(graph, rng, module_count: int) -> tuple[str, ...]:
    """The generic control: the thirteen production families, equiprobable."""

    names = np.asarray(sorted(GENERIC_MODULES))
    return tuple(str(name) for name in rng.choice(names, size=module_count))


def prior_families(prior):
    """A declaration drawn from the fitted prior's family projection."""

    def draw(graph, rng, module_count: int) -> tuple[str, ...]:
        return prior.sample_family_sequence(graph, rng, module_count=module_count)

    return draw
