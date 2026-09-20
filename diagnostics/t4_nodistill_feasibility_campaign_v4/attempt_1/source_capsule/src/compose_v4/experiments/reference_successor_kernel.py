"""Deliberately slow, obviously-correct canonical-successor aggregation. TEST ORACLE ONLY.

This exists to answer "is the production aggregation right?" with an implementation whose correctness can
be checked by reading it: explicit Python loops, an ordinary dictionary, probability mass summed directly.
No tensors, no segmented reductions, no vectorization.

WHY IT MUST NOT PRODUCE REPORTED NUMBERS
----------------------------------------
The trained stochastic process is defined by the production primitives -- the executor, the canonical state
key, the legal-action enumerators, the capability resolution. An independent aggregation is valuable as a
CHECK on the production path, but if it silently replaced that path in a reported result, the paper would
describe a process that was never trained. A reviewer could then reasonably ask whether the evaluated
kernel is the optimized kernel, and the answer would be "not exactly."

So the hierarchy is fixed:

    main paper results   ->  the production segmented-successor path
    independent oracle   ->  THIS module
    optional cross-check ->  canonical_successor_distillation

A disagreement between any two of them is a finding to resolve, never a licence to report whichever number
looks better. ``tests/test_reference_successor_kernel.py`` enforces the "never in results" rule by scanning
for imports of this module outside tests.

WHAT IS REUSED VERSUS REIMPLEMENTED
-----------------------------------
Reused, because these DEFINE the process: the production executor (``RewriteSystem.apply``) and the
production canonical key (``canonical_state_key``). Reimplemented from scratch: the grouping and the mass
summation -- which is precisely the step being verified.
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Iterable, Sequence

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.experiments.successor_kernel import (
    CanonicalSuccessor,
    KernelIdentity,
    SuccessorBatch,
    SupportSignature,
)
from compose_v4.rewrite.kernel import RewriteSystem, canonical_state_key

# One marked rewrite and its unnormalized probability/rate mass: (rule_name, action, mass).
MarkedMass = tuple[str, Any, float]


def reference_successor_batch(
    state: MolecularGraph,
    marked_masses: Iterable[MarkedMass],
    *,
    system: RewriteSystem,
    previous_state_key: str | None = None,
    identity: KernelIdentity | None = None,
) -> SuccessorBatch:
    """Aggregate marked mass into distinct canonical successors with explicit loops.

    ``system`` is required rather than defaulted: the de-novo and editing regimes have different legal
    supports, and defaulting would silently pick one. With ``previous_state_key=None`` this is the
    state-only base molecular kernel. Supplying ``previous_state_key`` explicitly applies a history-aware
    no-immediate-return wrapper and routes its rejected mass into ``virtual_mass``; that wrapper is not part
    of the base Markov kernel.
    """
    source_key = canonical_state_key(state)
    productive: dict[str, float] = {}
    aliases: dict[str, int] = {}
    successor_states: dict[str, MolecularGraph] = {}
    virtual = 0.0

    for rule_name, action, raw_mass in marked_masses:
        mass = float(raw_mass)
        if not isfinite(mass) or mass < 0.0:
            raise ValueError(f"marked mass must be finite and nonnegative, got {raw_mass!r}")
        if mass == 0.0:
            continue
        successor = system.apply(state, rule_name, action)
        key = canonical_state_key(successor)
        if key == source_key or (previous_state_key is not None and key == previous_state_key):
            virtual += mass
            continue
        productive[key] = productive.get(key, 0.0) + mass
        aliases[key] = aliases.get(key, 0) + 1
        successor_states.setdefault(key, successor)

    total = sum(productive.values())
    if total <= 0.0:
        return SuccessorBatch(
            source_key=source_key,
            successors=(),
            identity=identity or _oracle_identity(),
            virtual_mass=virtual,
        )
    return SuccessorBatch(
        source_key=source_key,
        successors=tuple(
            CanonicalSuccessor(
                key=key,
                state=successor_states[key],
                # Conditioned on taking a jump: normalized over PRODUCTIVE mass only, never including the
                # virtual mass, which is reported separately.
                probability=productive[key] / total,
                alias_count=aliases[key],
            )
            for key in sorted(productive)
        ),
        identity=identity or _oracle_identity(),
        virtual_mass=virtual,
    )


def reference_successor_probabilities(
    state: MolecularGraph,
    marked_masses: Iterable[MarkedMass],
    *,
    system: RewriteSystem,
    previous_state_key: str | None = None,
) -> dict[str, float]:
    """Just the canonical-key -> probability map, for direct comparison against a vectorized path."""
    batch = reference_successor_batch(
        state, marked_masses, system=system, previous_state_key=previous_state_key
    )
    return {successor.key: successor.probability for successor in batch.successors}


def compare_against_reference(
    produced: dict[str, float], reference: dict[str, float], *, tolerance: float = 1e-9
) -> list[str]:
    """Return human-readable disagreements between a production result and the oracle.

    Returns an empty list on agreement. Reports missing keys, extra keys and per-key deviations separately,
    because a scalar aggregate can coincide while the GROUPING is wrong -- which is the failure this oracle
    exists to catch.
    """
    problems: list[str] = []
    for key in sorted(set(reference) - set(produced)):
        problems.append(f"missing successor {key!r}: reference has {reference[key]:.12g}")
    for key in sorted(set(produced) - set(reference)):
        problems.append(f"extra successor {key!r}: produced {produced[key]:.12g}, reference has none")
    for key in sorted(set(produced) & set(reference)):
        delta = abs(produced[key] - reference[key])
        if delta > tolerance:
            problems.append(
                f"successor {key!r} differs by {delta:.3g}: produced {produced[key]:.12g}, "
                f"reference {reference[key]:.12g}"
            )
    return problems


def _oracle_identity() -> KernelIdentity:
    return KernelIdentity(
        implementation="reference_dictionary_oracle",
        support_signature=SupportSignature(canonicalizer_version="canonical_state_key"),
    )


def enumerate_uniform_marked_masses(marks: Sequence[tuple[str, Any]]) -> list[MarkedMass]:
    """Give every supplied mark unit mass, for tests that care about grouping rather than the law."""
    return [(rule_name, action, 1.0) for rule_name, action in marks]
