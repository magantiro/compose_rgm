"""The shared molecular-kernel contract: invariants, the uniform law, and the anti-duplication guard.

These tests defend the property the whole experimental program rests on -- that there is ONE definition of
the molecular successor law. They check three separable things:

  1. the invariants of a successor batch actually fail when violated (not merely documented);
  2. the uniform control arm is uniform over DISTINCT SUCCESSORS, not over marks -- the difference that makes
     E2 a causal comparison rather than a measure of encoding redundancy;
  3. the fairness assertion refuses arms whose legal support could differ.
"""
from __future__ import annotations

import pytest

from compose_v4.experiments.successor_kernel import (
    CanonicalSuccessor,
    CanonicalSuccessorKernel,
    ExplicitGraphKernel,
    KernelIdentity,
    SuccessorBatch,
    SuccessorKernelViolation,
    SupportSignature,
    UniformSuccessorKernel,
    assert_arms_comparable,
    validate_successor_batch,
)

_SIGNATURE = SupportSignature(
    operator_registry_hash="abc123", capability_flags=(("enable_cycle_ops", True),)
)
_IDENTITY = KernelIdentity(implementation="test", support_signature=_SIGNATURE)


def _batch(rows, *, source="x0", identity=_IDENTITY, virtual_mass=0.0):
    return SuccessorBatch(
        source_key=source,
        successors=tuple(
            CanonicalSuccessor(key=k, state=k, probability=p, alias_count=a) for k, p, a in rows
        ),
        identity=identity,
        virtual_mass=virtual_mass,
    )


# ---- invariants ---------------------------------------------------------------------------------------


def test_a_well_formed_batch_validates():
    validate_successor_batch(_batch([("y1", 0.6, 1), ("y2", 0.4, 3)]))


def test_terminal_batch_is_legal_and_needs_no_normalization():
    batch = _batch([])
    assert batch.is_terminal and batch.support_size == 0
    validate_successor_batch(batch)


def test_duplicate_successor_keys_are_rejected():
    """An unmerged alias would double-count one molecule as two."""
    with pytest.raises(SuccessorKernelViolation, match="not distinct"):
        validate_successor_batch(_batch([("y1", 0.5, 1), ("y1", 0.5, 1)]))


def test_unnormalized_law_is_rejected():
    with pytest.raises(SuccessorKernelViolation, match="sum to"):
        validate_successor_batch(_batch([("y1", 0.6, 1), ("y2", 0.6, 1)]))


def test_negative_probability_is_rejected():
    with pytest.raises(SuccessorKernelViolation, match="negative probability"):
        validate_successor_batch(_batch([("y1", 1.4, 1), ("y2", -0.4, 1)]))


def test_self_transition_must_not_appear_as_a_successor():
    """A move that returns the same molecule is proposal mass, not a molecular jump."""
    with pytest.raises(SuccessorKernelViolation, match="source key"):
        validate_successor_batch(_batch([("x0", 0.5, 1), ("y1", 0.5, 1)]))


def test_zero_alias_count_is_rejected():
    with pytest.raises(SuccessorKernelViolation, match="alias_count"):
        validate_successor_batch(_batch([("y1", 1.0, 0)]))


def test_negative_virtual_mass_is_rejected():
    with pytest.raises(SuccessorKernelViolation, match="virtual_mass"):
        validate_successor_batch(_batch([("y1", 1.0, 1)], virtual_mass=-0.1))


# ---- the uniform law: uniform over SUCCESSORS, not marks ----------------------------------------------


def test_uniform_ignores_mark_multiplicity():
    """The load-bearing property of E2's control arm.

    y1 is reachable by a single mark and y2 by five. Uniform-over-marks would give y2 five times the mass;
    uniform over distinct canonical successors gives them the same mass, because the number of syntactic
    encodings is a property of the action representation and not of chemistry.
    """
    base = ExplicitGraphKernel({"x0": [("y1", 0.9, 1), ("y2", 0.1, 5)]})
    batch = UniformSuccessorKernel(base).successors("x0")
    validate_successor_batch(batch)
    assert batch.probability_of("y1") == pytest.approx(0.5)
    assert batch.probability_of("y2") == pytest.approx(0.5)


def test_uniform_is_one_over_support_size():
    base = ExplicitGraphKernel({"x0": [("y1", 0.7, 1), ("y2", 0.2, 1), ("y3", 0.1, 2)]})
    batch = UniformSuccessorKernel(base).successors("x0")
    assert batch.support_size == 3
    for successor in batch.successors:
        assert successor.probability == pytest.approx(1.0 / 3.0)


def test_uniform_preserves_support_and_alias_counts():
    """It replaces the law only; support and multiplicity are inherited, never recomputed."""
    base = ExplicitGraphKernel({"x0": [("y1", 0.9, 1), ("y2", 0.1, 5)]})
    learned = base.successors("x0")
    uniform = UniformSuccessorKernel(base).successors("x0")
    assert set(uniform.keys) == set(learned.keys)
    assert {s.key: s.alias_count for s in uniform.successors} == {
        s.key: s.alias_count for s in learned.successors
    }


def test_uniform_of_a_terminal_state_stays_terminal():
    uniform = UniformSuccessorKernel(ExplicitGraphKernel({"x0": []}))
    assert uniform.successors("x0").is_terminal


# ---- fairness between arms ----------------------------------------------------------------------------


def test_arms_with_identical_support_configuration_are_comparable():
    base = ExplicitGraphKernel({"x0": [("y1", 1.0, 1)]})
    assert_arms_comparable(base, UniformSuccessorKernel(base))


def test_arms_with_different_capability_flags_are_refused():
    """A capability difference changes the LEGAL SUPPORT, so the comparison would not be causal."""

    class _OtherFlags:
        def successors(self, state):  # pragma: no cover - identity is what matters here
            raise NotImplementedError

        def identity(self):
            return KernelIdentity(
                implementation="other",
                support_signature=SupportSignature(capability_flags=(("enable_cycle_ops", False),)),
            )

    class _Ours:
        def successors(self, state):  # pragma: no cover
            raise NotImplementedError

        def identity(self):
            return KernelIdentity(
                implementation="ours",
                support_signature=SupportSignature(capability_flags=(("enable_cycle_ops", True),)),
            )

    with pytest.raises(SuccessorKernelViolation, match="support-determining"):
        assert_arms_comparable(_Ours(), _OtherFlags())


def test_differing_checkpoint_alone_remains_comparable():
    """A baseline legitimately has no checkpoint; only support-determining fields must agree."""
    shared = SupportSignature(capability_flags=(("a", True),))
    left = KernelIdentity(implementation="learned", support_signature=shared, checkpoint_sha256="ff")
    right = KernelIdentity(implementation="uniform", support_signature=shared, checkpoint_sha256=None)
    assert left.differs_only_in_law(right)


# ---- protocol conformance -----------------------------------------------------------------------------


def test_fixture_and_uniform_both_satisfy_the_protocol():
    base = ExplicitGraphKernel({"x0": [("y1", 1.0, 1)]})
    assert isinstance(base, CanonicalSuccessorKernel)
    assert isinstance(UniformSuccessorKernel(base), CanonicalSuccessorKernel)


def test_protocol_rejects_an_object_missing_identity():
    class _NoIdentity:
        def successors(self, state):  # pragma: no cover
            raise NotImplementedError

    assert not isinstance(_NoIdentity(), CanonicalSuccessorKernel)
