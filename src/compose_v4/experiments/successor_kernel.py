"""The single definition of the molecular process every experiment consumes: the PUSHFORWARD kernel.

What is actually learned, and what is derived
--------------------------------------------
COMPOSE learns a stochastic process over executable rewrite MARKS: ``q_theta(a | x)``. Pushing that process
through execution and molecular canonicalization induces a well-defined kernel over molecular successors,

    y = pi_x(a),        P_theta(y | x) = sum_{a : pi_x(a) = y} q_theta(a | x)

and all control and state-level comparisons operate on this quotient kernel. Both statements matter and
they are NOT the same statement:

* the trained objective is selected-mark likelihood, ``-log q_theta(a* | x)``;
* it is NOT canonical-successor likelihood, ``-log P_theta(y* | x)``.

Several syntactic marks routinely execute to the same molecule, so a mark-level law is not a molecular law.
Mark-level controls (power tilts, top-k, nucleus, family gating) are provably NOT invariant to how a fiber
is encoded, while successor-level control is -- verified numerically: a successor reachable by one mark
versus four aliases keeps aggregate mass 0.500 under a beta=1 tilt but moves to 0.200 at beta=2 and 0.667 at
beta=0.5. That asymmetry is a result the paper reports, and it is only meaningful if every experiment agrees
on one definition of the successor law.

What E5 does and does not license
---------------------------------
E5 establishes that the DERIVED successor kernel and successor-level controllers are invariant to aliasing
and within-fiber refinement. It does **not** establish -- and cannot retroactively make -- the original
training objective successor-level. Saying E5 "licenses the derivation" overstates it; E5 licenses the
quotient-level interpretation and control of the trained mark process after pushforward.

Hence this module defines a PROTOCOL rather than an implementation. Its purpose is to let the exact-control
work (E6) proceed in parallel with the production evaluator without a second definition of the process
appearing. Consumers depend on ``CanonicalSuccessorKernel``; they must never independently implement

    legal mark enumeration | canonicalization | mark-to-successor aggregation
    capability resolution  | slot masking

When the production evaluator lands it supplies the implementation, and parallel-developed consumers switch
to it without changing their own code.

Invariants maintained (checked by ``validate_successor_batch``)
--------------------------------------------------------------
* successor keys within a batch are DISTINCT -- aliases are already merged;
* probabilities are finite, nonnegative, and sum to 1 for a non-terminal state;
* ``alias_count >= 1`` for every successor, and records the pre-quotient mark multiplicity;
* no successor carries the source key -- a self-transition is not a molecular jump;
* canonical self-event mass is reported EXPLICITLY rather than silently discarded;
* an immediate backtrack is a legal successor of the state-only base kernel.  It
  can be suppressed only by a declared history-aware wrapper whose state also
  includes the previous molecular key.

Relationship to existing code -- established by tracing, not assumption
-----------------------------------------------------------------------
The trained objective is MARK-LEVEL. ``factorized_mark_bregman_loss`` is

    loss = total_hazard - teacher_rate * (log_hazard + selected_mark_log_probability)

a Poisson-KL Generator Matching loss "in normalized marked-rate form" that scores the selected teacher
MARK. Consequently the completed run did **not** train a general canonical-successor objective. The
production evaluator now derives the complete pushforward in ``production_successor_kernel``, using
``model.segmented_successor`` for aggregation. A bounded differentiable replacement-training bridge now
exists in ``factorized_successor_training``; it is not yet wired into the packed full trainer and therefore
does not change the completed run's semantics. The single completed-run exception is the graft family,
which already receives a within-fiber quotient at training time via ``graft_successor_groups`` -- so the
completed trained law is mark-level EXCEPT for graft.

Consequences, all load-bearing:

1. The scientific object is defined MATHEMATICALLY, not by a source file. The production molecular kernel
   is the pushforward of the trained mark law through (a) the production executor and (b) the production
   canonical molecular key, summed over each successor fiber. ``model.segmented_successor`` *computes* that
   pushforward; the fresh dictionary-based reference implementation *verifies* it on bounded fixtures. No
   single module is the definition. The distillation aggregator is at most a cross-check, never a second
   production definition.
2. The "one kernel" discipline is therefore about implementing that DERIVATION exactly once -- which is the
   unified evaluator's job.
3. An audit is owed, not assumed: on the full validation set, measure how often aliasing actually occurs
   (fraction of states with an aliased successor, fraction of teacher transitions whose successor has alias
   count > 1, the alias-count distribution) together with selected-mark NLL, canonical-successor NLL and
   their gap ``Delta(x,a*) = -log q_theta(a*|x) + log P_theta(pi_x(a*)|x)``. If aliasing is rare and the gap
   tiny, mark-level training is mostly an implementation distinction; if common and substantial, it is a
   real ablation or stated limitation. This audit must never interrupt the active run.
4. The graft family is already quotiented at training time via ``graft_successor_groups``, so the trained
   law is mark-level EXCEPT for graft. E5 must handle that asymmetry explicitly rather than average over it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields as dataclass_fields
from math import isfinite
from typing import Iterable, Protocol, runtime_checkable

from compose_v4.chem.molecular_graph import MolecularGraph

# A committed state: a complete, valid, connected molecule.
MolecularState = MolecularGraph

# Absolute tolerance for the normalization invariant. Loose enough for float32 accumulation over a few
# thousand candidates, tight enough that a genuinely unnormalized law fails.
NORMALIZATION_TOLERANCE = 1e-6


class SuccessorKernelViolation(ValueError):
    """A successor batch broke an invariant of the molecular kernel."""


# ---- identity -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SupportSignature:
    """Every field that determines which molecular successors are LEGAL.

    Capability flags and the operator registry alone are not the complete support signature: element and
    charge vocabularies, the atom size bound, the valence policy, the canonicalizer and executor versions,
    the persistent-slot schema, the supported AtomInsert arity, the jump-chain policy and the RingCore
    configuration all change the legal successor set. Two arms may differ in weights, in the name of the
    probability law, and in controller -- they may NOT differ here, or the comparison between them is not a
    comparison of laws over a common support.
    """

    operator_registry_hash: str | None = None
    capability_flags: tuple[tuple[str, bool], ...] = ()
    # The FULL ORDERED (element, valence) class table, not a name or a count. Index order determines
    # classifier-head semantics, the same count can hide different classes, and one element can occupy
    # several valence classes -- so only the exact ordered table identifies the vocabulary.
    element_vocabulary: tuple[str, ...] = ()
    charge_vocabulary: tuple[int, ...] = ()
    bond_vocabulary: tuple[str, ...] = ()
    aromaticity_policy: str | None = None
    max_atoms: int | None = None
    valence_policy: str | None = None
    canonicalizer_version: str | None = None
    executor_version: str | None = None
    persistent_slot_schema: str | None = None
    # Production supports AtomInsert with 0 neighbours (grow_root) or exactly 1 (grow_connected); >=2
    # (vertex subdivision) has no head. Recorded because a change here silently alters legal support.
    atom_insert_arity_support: tuple[int, ...] = (0, 1)
    embedded_jump_chain_policy: str | None = None
    ringcore_configuration: str | None = None

    def vocabulary_hash(self) -> str:
        """Stable digest of the ordered class table, for compact provenance next to the full table."""
        return hashlib.sha256("|".join(self.element_vocabulary).encode()).hexdigest()[:16]

    def differing_fields(self, other: SupportSignature) -> tuple[str, ...]:
        """Names of support fields that differ, reporting individual CAPABILITY FLAGS by name.

        Flag-level granularity is what makes a preregistered support ablation expressible: E4 needs to say
        "only ``enable_cycle_ops`` may differ", which is impossible if the whole ``capability_flags``
        container is reported as one opaque difference.
        """
        differences: list[str] = []
        for descriptor in dataclass_fields(self):
            name = descriptor.name
            if name == "capability_flags":
                mine, theirs = dict(self.capability_flags), dict(other.capability_flags)
                for flag in sorted(set(mine) | set(theirs)):
                    if mine.get(flag) != theirs.get(flag):
                        differences.append(flag)
                continue
            if getattr(self, name) != getattr(other, name):
                differences.append(name)
        return tuple(differences)


@dataclass(frozen=True)
class KernelIdentity:
    """What kernel produced a batch, so two consumers can prove they used the same process.

    ``support_signature`` is the load-bearing field: it must be EQUAL across arms. ``implementation`` and
    ``checkpoint_sha256`` are free to differ -- a baseline legitimately has no checkpoint, and differing in
    the probability law is the entire point of the comparison.
    """

    implementation: str
    support_signature: SupportSignature = SupportSignature()
    checkpoint_sha256: str | None = None

    @property
    def capability_flags(self) -> tuple[tuple[str, bool], ...]:
        return self.support_signature.capability_flags

    @property
    def operator_registry_hash(self) -> str | None:
        return self.support_signature.operator_registry_hash

    def differs_only_in_law(self, other: KernelIdentity) -> bool:
        """True when the two kernels share the ENTIRE support signature."""
        return not self.support_signature.differing_fields(other.support_signature)


# ---- successors ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalSuccessor:
    """One distinct molecular successor and its aggregated probability."""

    key: str
    state: MolecularState
    probability: float
    alias_count: int


@dataclass(frozen=True)
class SuccessorBatch:
    """The molecular jump law out of one state, over distinct canonical successors.

    Normalization convention: probabilities are over PRODUCTIVE canonical successors CONDITIONED ON
    TAKING A JUMP. Canonical self-event mass is recorded separately in ``virtual_mass`` and is NOT folded
    into the productive law. A declared history-aware wrapper may additionally reject a return to its
    previous molecular key, but that rejection is not a property of this state-only base kernel.
    """

    source_key: str
    successors: tuple[CanonicalSuccessor, ...]
    identity: KernelIdentity
    # Proposal mass that is not a molecular jump. In the state-only base kernel
    # this means canonical self-events. A history-aware wrapper may add its own
    # rejected mass (for example, an immediate return to its previous key).
    virtual_mass: float = 0.0

    @property
    def support_size(self) -> int:
        """|N(x)|: the number of distinct canonical molecular successors."""
        return len(self.successors)

    @property
    def is_terminal(self) -> bool:
        return not self.successors

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(successor.key for successor in self.successors)

    def probability_of(self, key: str) -> float:
        for successor in self.successors:
            if successor.key == key:
                return successor.probability
        return 0.0

    def total_probability(self) -> float:
        return float(sum(successor.probability for successor in self.successors))


@runtime_checkable
class CanonicalSuccessorKernel(Protocol):
    """The one interface every experiment depends on."""

    def successors(self, state: MolecularState) -> SuccessorBatch:
        """Distinct canonical molecular successors of ``state`` with their aggregated probabilities."""
        ...

    def identity(self) -> KernelIdentity:
        """Stable identity of this kernel, for cross-arm fairness assertions."""
        ...


# ---- invariant checking -------------------------------------------------------------------------------


def validate_successor_batch(
    batch: SuccessorBatch, *, tolerance: float = NORMALIZATION_TOLERANCE
) -> None:
    """Raise ``SuccessorKernelViolation`` unless every documented invariant holds.

    Called by consumers rather than assumed: a silently unnormalized or alias-duplicated batch would
    corrupt every downstream number while still looking like a probability distribution.
    """
    keys = batch.keys
    if len(set(keys)) != len(keys):
        duplicated = sorted({key for key in keys if keys.count(key) > 1})
        raise SuccessorKernelViolation(
            f"successor keys are not distinct -- aliases were not merged: {duplicated}"
        )
    if batch.source_key in keys:
        raise SuccessorKernelViolation(
            f"a successor carries the source key {batch.source_key!r}; a self-transition is not a "
            "molecular jump and belongs in virtual_mass"
        )
    for successor in batch.successors:
        if not isfinite(successor.probability) or successor.probability < 0.0:
            raise SuccessorKernelViolation(
                f"successor {successor.key!r} has non-finite or negative probability "
                f"{successor.probability!r}"
            )
        if successor.alias_count < 1:
            raise SuccessorKernelViolation(
                f"successor {successor.key!r} has alias_count {successor.alias_count}; a merged "
                "successor was produced by at least one mark"
            )
    if not isfinite(batch.virtual_mass) or batch.virtual_mass < 0.0:
        raise SuccessorKernelViolation(f"virtual_mass must be finite and nonnegative: {batch.virtual_mass!r}")
    if batch.is_terminal:
        return
    total = batch.total_probability()
    if abs(total - 1.0) > tolerance:
        raise SuccessorKernelViolation(
            f"successor probabilities sum to {total!r}, not 1 (tolerance {tolerance}); the molecular jump "
            "law must be normalized over distinct successors"
        )


# How two arms are meant to relate. Comparing arms without declaring this is how an unfair comparison, or
# a wrongly-blocked legitimate ablation, sneaks in.
LAW_ONLY = "law_only"                    # identical support; only the probability assignment differs
SUPPORT_ABLATION = "support_ablation"    # support differs ONLY in preregistered fields
EXTERNAL_BASELINE = "external_baseline"  # a different system entirely; support is NOT claimed equal
COMPARISON_TYPES = frozenset({LAW_ONLY, SUPPORT_ABLATION, EXTERNAL_BASELINE})


def assert_arms_comparable(
    left: CanonicalSuccessorKernel,
    right: CanonicalSuccessorKernel,
    *,
    comparison: str = LAW_ONLY,
    allowed_support_differences: tuple[str, ...] = (),
) -> None:
    """Check that two arms relate the way the experiment claims they do.

    A single blanket "supports must be identical" rule is wrong in both directions. It is necessary for a
    learned-versus-uniform comparison, where only the probability assignment may change. But it would
    WRONGLY BLOCK a legitimate support ablation -- E4's ``no_cycle_operations`` arm exists precisely to
    remove a capability, and an earlier version of this function rejected it. So the intended relationship
    is declared, and only undeclared differences fail:

    * ``law_only`` -- every support field must match. Learned vs uniform.
    * ``support_ablation`` -- support may differ ONLY in ``allowed_support_differences``, which must be
      preregistered; any other difference fails. E4 RingCore vs no-cycle-ops.
    * ``external_baseline`` -- no support claim is made. Do NOT pretend the supports match; compare
      endpoints and compute transparently instead.
    """
    if comparison not in COMPARISON_TYPES:
        raise SuccessorKernelViolation(
            f"unknown comparison type {comparison!r}; expected one of {sorted(COMPARISON_TYPES)}"
        )
    if comparison == EXTERNAL_BASELINE:
        return
    if comparison == LAW_ONLY and allowed_support_differences:
        raise SuccessorKernelViolation(
            "law_only comparisons may not declare allowed_support_differences; if the support genuinely "
            "differs the comparison is a support_ablation, and saying so is the point"
        )
    if comparison == SUPPORT_ABLATION and not allowed_support_differences:
        raise SuccessorKernelViolation(
            "a support_ablation must preregister which support fields may differ; with none declared, use "
            "law_only"
        )

    differences = left.identity().support_signature.differing_fields(
        right.identity().support_signature
    )
    undeclared = tuple(name for name in differences if name not in set(allowed_support_differences))
    if undeclared:
        raise SuccessorKernelViolation(
            f"kernel arms differ in undeclared support-determining fields {list(undeclared)} under a "
            f"{comparison} comparison. Declared differences: {list(allowed_support_differences)}.\n"
            f"  left : {left.identity().implementation}\n  right: {right.identity().implementation}"
        )


# ---- the uniform legal-successor law ------------------------------------------------------------------


class UniformSuccessorKernel:
    """Uniform over DISTINCT canonical molecular successors: ``P(y|x) = 1/|N(x)|``.

    This is E2's control arm and the stage-1 kernel for the exact-control benchmark. It is a PROBABILITY
    WRAPPER over another kernel, never a second enumeration -- it inherits support, executor, capability
    flags and alias counts from the wrapped kernel and replaces only the law.

    The distinction from uniform-over-marks is the entire point of the comparison: a molecule reachable by
    five marks receives the SAME mass as one reachable by a single mark. Uniform-over-marks would instead
    reward whichever successor happens to have the most syntactic encodings, which is a property of the
    encoding rather than of chemistry.
    """

    def __init__(self, base: CanonicalSuccessorKernel) -> None:
        self._base = base

    def successors(self, state: MolecularState) -> SuccessorBatch:
        batch = self._base.successors(state)
        if batch.is_terminal:
            return SuccessorBatch(
                source_key=batch.source_key,
                successors=(),
                identity=self.identity(),
                virtual_mass=batch.virtual_mass,
            )
        share = 1.0 / float(batch.support_size)
        return SuccessorBatch(
            source_key=batch.source_key,
            successors=tuple(
                CanonicalSuccessor(
                    key=successor.key,
                    state=successor.state,
                    probability=share,
                    alias_count=successor.alias_count,
                )
                for successor in batch.successors
            ),
            identity=self.identity(),
            virtual_mass=batch.virtual_mass,
        )

    def identity(self) -> KernelIdentity:
        base = self._base.identity()
        return KernelIdentity(
            implementation=f"uniform_over_canonical_successors({base.implementation})",
            # The support signature is INHERITED verbatim: the wrapper changes the law, never the support.
            support_signature=base.support_signature,
            checkpoint_sha256=base.checkpoint_sha256,
        )


# ---- explicit fixture kernel (for parallel development of exact control) ------------------------------


class ExplicitGraphKernel:
    """A kernel defined by an explicit transition table, for developing consumers before A1.2 lands.

    Exists so the exact-control solver can be built and verified against the FROZEN protocol without
    reimplementing enumeration, and without waiting for the production evaluator. It is a test/development
    fixture: it never appears in a reported result.
    """

    def __init__(
        self,
        transitions: dict[str, Iterable[tuple[str, float, int]]],
        states: dict[str, MolecularState] | None = None,
        *,
        implementation: str = "explicit_graph_fixture",
    ) -> None:
        self._transitions = {
            source: tuple(targets) for source, targets in transitions.items()
        }
        self._states = dict(states or {})
        self._implementation = implementation

    def successors(self, state: MolecularState | str) -> SuccessorBatch:
        source_key = state if isinstance(state, str) else _fixture_key(state, self._states)
        rows = self._transitions.get(source_key, ())
        return SuccessorBatch(
            source_key=source_key,
            successors=tuple(
                CanonicalSuccessor(
                    key=key,
                    state=self._states.get(key, key),  # type: ignore[arg-type]
                    probability=float(probability),
                    alias_count=int(alias_count),
                )
                for key, probability, alias_count in rows
            ),
            identity=self.identity(),
        )

    def identity(self) -> KernelIdentity:
        return KernelIdentity(implementation=self._implementation)


def _fixture_key(state: MolecularState, states: dict[str, MolecularState]) -> str:
    for key, candidate in states.items():
        if candidate is state:
            return key
    raise KeyError("state is not registered in the fixture kernel")
