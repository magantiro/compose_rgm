"""The single definition of the molecular process every experiment consumes.

The paper's claims live on MOLECULES, not on action encodings: the kernel is

    P(y | x) = sum_{a : T(x,a) ~= y} p(a | x)

aggregated over the distinct canonical successors ``y``. Several syntactic marks routinely execute to the
same molecule, so a mark-level law is not a molecular law -- and mark-level controls (power tilts, top-k,
nucleus, family gating) are provably NOT invariant to how a fiber is encoded, while successor-level control
is. That asymmetry is a result the paper reports, and it is only meaningful if every experiment agrees on
one definition of the successor law.

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
* virtual (self / immediate-backtrack) mass is reported EXPLICITLY rather than silently renormalized away,
  because deleting it and renormalizing changes every productive rate.

Relationship to existing code -- established by tracing, not assumption
-----------------------------------------------------------------------
The trained objective is MARK-LEVEL. ``factorized_mark_bregman_loss`` is

    loss = total_hazard - teacher_rate * (log_hazard + selected_mark_log_probability)

a Poisson-KL Generator Matching loss "in normalized marked-rate form" that scores the selected teacher
MARK. Consequently there is **no production canonical-successor aggregator wired into training or
rollout**: both ``model.segmented_successor`` and
``experiments.canonical_successor_distillation.aggregate_canonical_successor_rates`` have test callers
only. The single exception is the graft family, which already receives a within-fiber quotient at training
time via ``graft_successor_groups`` -- so the trained law is mark-level EXCEPT for graft.

Two consequences, both load-bearing:

1. The molecular kernel is a DERIVED object: production mark log-probabilities -> execute each mark through
   the production executor -> group by the production ``canonical_state_key`` -> aggregate. The "one
   kernel" discipline is therefore about implementing that DERIVATION exactly once, which is what the
   unified evaluator is for. ``model.segmented_successor`` is the authoritative aggregation implementation
   for that step; the distillation aggregator is a CROSS-CHECK only, never a second production definition.
2. E5 is not a nicety. The paper states the process lives on molecules while the objective is trained on
   marks; quotient invariance is precisely what licenses the derivation. The graft asymmetry must be
   handled explicitly there rather than averaged over.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    element_vocabulary: tuple[str, ...] = ()
    charge_vocabulary: tuple[int, ...] = ()
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
        return self.support_signature == other.support_signature


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
    TAKING A JUMP. Virtual mass (self-transitions, immediate backtracks) and any terminal/no-jump mass are
    recorded separately in ``virtual_mass`` and are NOT folded into the productive law -- deleting them and
    renormalizing would change every productive rate. A wrapper inherits ``virtual_mass`` unchanged.
    """

    source_key: str
    successors: tuple[CanonicalSuccessor, ...]
    identity: KernelIdentity
    # Proposal mass that is not a molecular jump (self-transitions, immediate backtracks). Reported so a
    # consumer can account for it; never folded into the productive law by this module.
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


def assert_arms_comparable(left: CanonicalSuccessorKernel, right: CanonicalSuccessorKernel) -> None:
    """Refuse to compare two arms whose legal support could differ.

    The learned-versus-uniform comparison is only causal if the executor, capability flags and enumeration
    are identical and ONLY the probability assignment changes. A capability-flag mismatch silently
    invalidates it, so this is checked rather than documented.
    """
    if not left.identity().differs_only_in_law(right.identity()):
        raise SuccessorKernelViolation(
            "kernel arms differ in support-determining configuration, not only in their probability law:\n"
            f"  left : {left.identity()}\n  right: {right.identity()}"
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
