"""Guards for the zero-training de-novo ring-schedule probe.

Every guard here is written to FAIL against a specific production mutation.  The
probe's whole purpose is to decide whether a retrain is worth running, so a guard
that cannot fail would let an inert arm read as a working repair -- or, worse, let
a relaxed arm silently keep the barrier it was built to remove.

The near-miss this file is shaped around: the scheduler reports its own
``priority_positions_before``/``after``, and those fields are produced by the very
code whose movement is in question.  Positions are therefore recomputed from the
returned trace, and one test mutates the metadata to prove the recomputation does
not consult it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.eval import denovo_schedule_probe as probe
from compose_v4.rewrite.commuting_schedule import DEFAULT_PHASE_BARRIERS
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.trace import RewriteTrace, execute_trace
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target

_SMILES = ("c1ccncc1CCO", "C1CCC2CCCCC2C1O", "C1CCC2(CC1)CCCC2CO")


def _compiled_trace(smiles: str, *, seed: int = 7, n_slots: int = 24) -> RewriteTrace:
    """Compile one real de-novo transport trace in the production flexible mode."""

    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), n_slots)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms - 2,)).sample(
        np.random.default_rng(seed),
        n_slots=target.n_atoms,
    )
    return compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        flexible_size=True,
    )


@pytest.mark.parametrize("smiles", _SMILES)
def test_ring_event_indices_read_the_steps_not_the_scheduler_report(smiles: str) -> None:
    """Positions must come from the trace's steps, never from its metadata.

    Mutating the recorded positions to a lie must not move the recomputed answer.
    """

    trace = _compiled_trace(smiles)
    honest = probe.ring_event_indices(trace)
    lying = RewriteTrace(
        source=trace.source,
        target=trace.target,
        steps=trace.steps,
        metadata={
            **trace.metadata,
            "schedule_priority_positions_before": (999,),
            "schedule_priority_positions_after": (0,),
        },
    )

    assert probe.ring_event_indices(lying) == honest
    assert all(
        trace.steps[index].rule_name in {"ring_system_grow", "ring_system_delete"}
        for index in honest
    )


@pytest.mark.parametrize("smiles", _SMILES)
def test_free_slots_are_measured_at_the_state_the_ring_event_fires_from(
    smiles: str,
) -> None:
    """The mechanistic variable is the PRE-event state, not the post-event state.

    A ring grow consumes free slots, so measuring after it understates the slots
    the decision actually had.  This drives the real executor rather than
    recomputing a reference, so it fails if the prefix alignment slips by one.
    """

    system = de_novo_rewrite_system()
    trace = _compiled_trace(smiles)
    observations = probe.observe_ring_events(trace, system=system)
    assert observations, "the flexible transport trace must contain a ring event"

    _, states = execute_trace(
        trace.source,
        trace.steps,
        system=system,
        return_states=True,
    )
    for observation in observations:
        step = trace.steps[observation.index]
        # MEASURED on these traces: ring_system_grow cyclizes atoms that already
        # exist, so free slots are IDENTICAL before and after it.  A guard that
        # only compared the two counts therefore could not detect an off-by-one
        # at all -- it survived exactly that mutation.  The alignment is pinned
        # instead by driving the executor: the state an observation carries must
        # be the state this step consumes, and applying the step to it must
        # reproduce the next prefix state.
        assert probe.states_are_array_exact(observation.state, states[observation.index])
        successor = system.apply(observation.state, step.rule_name, step.action)
        assert probe.states_are_array_exact(successor, states[observation.index + 1])
        # The reported number must be a function of the state that was stored,
        # so it cannot be read off a different index than the state it claims.
        assert observation.free_slots == probe.free_slots(observation.state)
        assert observation.occupied_free_slots == probe.occupied_free_slots(
            observation.state
        )


@pytest.mark.parametrize("smiles", _SMILES)
def test_every_arm_preserves_the_endpoint_and_the_step_multiset(smiles: str) -> None:
    """A rescheduled trace is a PERMUTATION that lands on the same target.

    This is what makes the arms comparable: they differ in conditioning order and
    in nothing else.  It fails if an arm drops, adds, or rewrites a step.
    """

    system = de_novo_rewrite_system()
    trace = _compiled_trace(smiles)
    baseline = Counter(step.rule_name for step in trace.steps)

    for arm, barriers in probe.ARM_BARRIERS.items():
        result = probe.apply_schedule_arm(
            trace,
            arm=arm,
            phase_barrier_rule_names=barriers,
            system=system,
        )
        assert not result.dropped, result.drop_reason
        assert result.trace is not None
        assert Counter(step.rule_name for step in result.trace.steps) == baseline
        endpoint, _ = execute_trace(
            result.trace.source,
            result.trace.steps,
            system=system,
            return_states=True,
        )
        assert np.array_equal(endpoint.atom_types, trace.target.atom_types)
        assert np.array_equal(endpoint.bonds, trace.target.bonds)
        assert np.array_equal(endpoint.formal_charges, trace.target.formal_charges)
        assert np.array_equal(
            endpoint.implicit_h_counts, trace.target.implicit_h_counts
        )


def test_arm_c_passes_an_empty_barrier_set_into_the_production_scheduler() -> None:
    """Arm C must actually reach the scheduler with the barrier removed.

    The relaxation is the whole point of arm C, and it is one keyword deep -- the
    exact failure mode that made the region-law repair inert.  This captures the
    barrier set the PRODUCTION function receives rather than asserting on a
    constant, so pointing arm C back at the default barriers turns it red.
    """

    system = de_novo_rewrite_system()
    trace = _compiled_trace(_SMILES[0])
    seen: list[frozenset[str]] = []
    original = probe.schedule_priority_events_earliest

    def recording(traced, **kwargs):
        seen.append(frozenset(kwargs["phase_barrier_rule_names"]))
        return original(traced, **kwargs)

    probe.schedule_priority_events_earliest = recording  # type: ignore[assignment]
    try:
        probe.apply_schedule_arm(
            trace,
            arm="C_exact_early_ring_barrier_relaxed",
            phase_barrier_rule_names=probe.ARM_BARRIERS[
                "C_exact_early_ring_barrier_relaxed"
            ],
            system=system,
        )
        probe.apply_schedule_arm(
            trace,
            arm="B_exact_early_ring_default_barriers",
            phase_barrier_rule_names=probe.ARM_BARRIERS[
                "B_exact_early_ring_default_barriers"
            ],
            system=system,
        )
    finally:
        probe.schedule_priority_events_earliest = original  # type: ignore[assignment]

    assert seen == [frozenset(), frozenset(DEFAULT_PHASE_BARRIERS)]
    assert frozenset(DEFAULT_PHASE_BARRIERS) == frozenset(
        {"atom_insert", "atom_delete"}
    )


def test_arm_a_is_not_a_scheduler_configuration() -> None:
    """Arm A must be the untouched compiled trace, never a scheduler round-trip.

    Running the scheduler with an 'identity' barrier set would still consume the
    exactness path and could reorder; arm A has to be the trace Lineage B trained
    on, byte for byte.
    """

    assert probe.ARM_A not in probe.ARM_BARRIERS
    assert set(probe.ARM_BARRIERS) == {
        "B_exact_early_ring_default_barriers",
        "C_exact_early_ring_barrier_relaxed",
    }


def test_a_non_exact_endpoint_is_dropped_and_counted_never_returned() -> None:
    """An arm that cannot stay array-exact must yield no trace at all.

    Mutating ``apply_schedule_arm`` to return the trace regardless turns this red,
    which is what stops a silently-wrong reordering from entering a distribution.
    """

    system = de_novo_rewrite_system()
    trace = _compiled_trace(_SMILES[0])
    corrupted = RewriteTrace(
        source=trace.source,
        target=trace.source,  # a target the steps provably do not reach
        steps=trace.steps,
        metadata=trace.metadata,
    )
    result = probe.apply_schedule_arm(
        corrupted,
        arm="B_exact_early_ring_default_barriers",
        phase_barrier_rule_names=DEFAULT_PHASE_BARRIERS,
        system=system,
    )
    assert result.dropped
    assert result.trace is None
    assert result.drop_reason


def test_a_scheduler_returning_a_drifted_trace_is_dropped_by_the_arm_itself() -> None:
    """The arm's own re-execution must catch drift the scheduler did not raise on.

    The scheduler already refuses a trace whose endpoint moved, so feeding it a
    corrupted trace exercises ITS guard, not this module's -- that is why the
    redundant check here survived being disabled.  This test replaces the
    scheduler with one that returns a genuinely drifted trace, which is the only
    way to reach the arm's own array-exact certificate.
    """

    system = de_novo_rewrite_system()
    trace = _compiled_trace(_SMILES[0])
    original = probe.schedule_priority_events_earliest

    class _Report:
        attempted_swaps = 0
        accepted_swaps = 0
        priority_positions_before: tuple[int, ...] = ()
        priority_positions_after: tuple[int, ...] = ()

    def drifting(traced, **kwargs):
        # Drop the final step: still a legal executable prefix, but it lands on a
        # state that is NOT the recorded target.
        truncated = RewriteTrace(
            source=traced.source,
            target=traced.target,
            steps=traced.steps[:-1],
            metadata=traced.metadata,
        )
        return truncated, _Report()

    probe.schedule_priority_events_earliest = drifting  # type: ignore[assignment]
    try:
        result = probe.apply_schedule_arm(
            trace,
            arm="B_exact_early_ring_default_barriers",
            phase_barrier_rule_names=DEFAULT_PHASE_BARRIERS,
            system=system,
        )
    finally:
        probe.schedule_priority_events_earliest = original  # type: ignore[assignment]

    assert result.dropped
    assert result.trace is None
    assert result.drop_reason == "rescheduled endpoint is not array-exact"


def test_empty_distributions_report_none_rather_than_zero() -> None:
    """An arm with no ring events must not read as a measured zero."""

    summary = probe.summarize_arm(
        arm="empty",
        observations=(),
        support_masses=(),
        legal_counts=(),
        legal_small_counts=(),
        attempted_swaps=0,
        accepted_swaps=0,
        traces_scheduled=0,
        traces_dropped=0,
        drop_reasons=(),
        position_report_disagreements=0,
    )
    assert summary["m2_free_slots"]["n"] == 0
    assert summary["m2_free_slots"]["mean"] is None
    assert summary["m3_small_ring_support_mass"]["mean"] is None


def test_small_ring_support_mass_matches_the_production_helper() -> None:
    """The reported mass must be the production statistic, not a local formula."""

    import torch

    from compose_v4.eval.ring_calibration import uniform_category_mass

    support = (True, True, False, True)
    category = torch.tensor((True, False, True, False), dtype=torch.bool)
    mass, legal, legal_small = probe.small_ring_support_mass(support, category)

    assert legal == 3
    assert legal_small == 1
    assert mass == pytest.approx(
        uniform_category_mass(
            torch.tensor(support, dtype=torch.bool),
            category,
        )
    )


def test_prefix_alignment_is_pinned_by_an_operator_that_changes_slot_count() -> None:
    """Pin the off-by-one with ``atom_insert``, the only slot-changing event here.

    A ring grow leaves the free-slot count untouched, so measuring it one step
    late is numerically invisible on ring events -- a mutation that did exactly
    that survived every other guard in this file.  Atom insertion changes the
    count by construction, so pointing the same production observer at it makes
    the alignment observable in the reported NUMBER rather than only in the
    stored state.
    """

    system = de_novo_rewrite_system()
    trace = _compiled_trace(_SMILES[0])
    insertions = probe.observe_ring_events(
        trace,
        priority_rule_names=frozenset({"atom_insert"}),
        system=system,
    )
    assert insertions, "the flexible transport trace must insert atoms"

    _, states = execute_trace(
        trace.source,
        trace.steps,
        system=system,
        return_states=True,
    )
    for observation in insertions:
        pre = probe.free_slots(states[observation.index])
        post = probe.free_slots(states[observation.index + 1])
        assert post == pre - 1, "atom insertion must consume exactly one slot"
        assert observation.free_slots == pre


def test_priority_default_covers_both_ring_transactions() -> None:
    """Narrowing the default priority set would hide ring deletions silently."""

    from compose_v4.rewrite.commuting_schedule import DEFAULT_PRIORITY_RULES

    assert frozenset(DEFAULT_PRIORITY_RULES) == frozenset(
        {"ring_system_grow", "ring_system_delete"}
    )
    signature_default = probe.ring_event_indices.__defaults__
    assert signature_default is None  # keyword-only, so read the annotation path
    trace = _compiled_trace(_SMILES[0])
    assert probe.ring_event_indices(trace) == probe.ring_event_indices(
        trace,
        priority_rule_names=DEFAULT_PRIORITY_RULES,
    )
