"""The three Claim-2 transport laws, tested without a model or a checkpoint.

The point of the ``SuccessorRow`` boundary is that everything below it is
arithmetic over one enumeration, so the arithmetic can be pinned locally and a
container-hour is never spent discovering that a baseline was mis-specified.
"""

from __future__ import annotations

import math

import pytest

from compose_v4.experiments.claim2_transport_laws import (
    ARMS,
    ARM_EMPIRICAL_FAMILY,
    ARM_REFERENCE,
    ARM_UNIFORM,
    SuccessorRow,
    TransportLawError,
    arm_divergence,
    empirical_family_law,
    law_for_arm,
    sample_index,
    total_variation,
    uniform_canonical_law,
)

#: The frozen realized family law, as committed in
#: diagnostics/editing_v2_sampling_law_v2.json.
FROZEN_Q = {
    "atom_delete": 0.1730,
    "atom_insert": 0.1608,
    "atom_restate": 0.2042,
    "bond_reorder": 0.0500,
    "bond_reroute": 0.1902,
    "cycle_attach": 0.0500,
    "cycle_insert": 0.0644,
    "ring_system_restate": 0.1074,
}


def make_row(**overrides):
    """A four-successor row with a deliberately uneven family structure.

    ``b`` is reachable by two families, which is the case that separates a
    correct cross-family SUM from a wrong "pick one family" shortcut.
    """
    payload = {
        "source_key": "CCO",
        "successor_keys": ("a", "b", "c", "d"),
        "reference_probabilities": (0.5, 0.25, 0.15, 0.10),
        "alias_counts": (3, 2, 1, 1),
        "families": (
            ("atom_delete",),
            ("atom_delete", "bond_reorder"),
            ("bond_reorder",),
            ("atom_insert",),
        ),
    }
    payload.update(overrides)
    return SuccessorRow(**payload)


# ---- the row invariants ---------------------------------------------------


def test_row_rejects_duplicate_keys():
    with pytest.raises(TransportLawError, match="not distinct"):
        make_row(successor_keys=("a", "a", "c", "d"))


def test_row_rejects_unsorted_keys():
    with pytest.raises(TransportLawError, match="sorted order"):
        make_row(successor_keys=("b", "a", "c", "d"))


def test_row_rejects_self_transition():
    with pytest.raises(TransportLawError, match="self-transition"):
        make_row(source_key="a")


def test_row_rejects_unnormalized_reference_law():
    with pytest.raises(TransportLawError, match="not 1"):
        make_row(reference_probabilities=(0.5, 0.25, 0.15, 0.5))


def test_row_rejects_successor_with_no_reaching_family():
    with pytest.raises(TransportLawError, match="no reaching family"):
        make_row(families=(("atom_delete",), (), ("bond_reorder",), ("atom_insert",)))


def test_family_successor_counts_count_molecules_not_marks():
    """|N_k(x)| counts distinct successors; alias multiplicity must not leak in."""
    row = make_row()
    assert row.family_successor_counts == {
        "atom_delete": 2,
        "bond_reorder": 2,
        "atom_insert": 1,
    }
    assert row.alias_counts[0] == 3  # three marks, still one successor


# ---- uniform canonical ----------------------------------------------------


def test_uniform_is_uniform_over_distinct_successors_not_marks():
    """The successor reachable by three marks gets no more mass than the others."""
    row = make_row()
    law = uniform_canonical_law(row)
    assert law == pytest.approx((0.25, 0.25, 0.25, 0.25))
    assert math.isclose(sum(law), 1.0)


def test_uniform_of_terminal_row_is_empty():
    row = SuccessorRow(
        source_key="CCO",
        successor_keys=(),
        reference_probabilities=(),
        alias_counts=(),
        families=(),
    )
    assert row.is_terminal
    assert uniform_canonical_law(row) == ()


# ---- empirical family -----------------------------------------------------


def test_empirical_family_matches_the_frozen_definition_by_hand():
    """Reproduce the committed five-step definition arithmetically.

    Legal families at this state are atom_delete, bond_reorder, atom_insert.
    qhat renormalizes q over exactly those three, then each family spreads its
    mass uniformly over the distinct successors it reaches, and a successor
    reached by two families receives the SUM.
    """
    row = make_row()
    mass = FROZEN_Q["atom_delete"] + FROZEN_Q["bond_reorder"] + FROZEN_Q["atom_insert"]
    q_delete = FROZEN_Q["atom_delete"] / mass
    q_reorder = FROZEN_Q["bond_reorder"] / mass
    q_insert = FROZEN_Q["atom_insert"] / mass

    expected = (
        q_delete / 2,                    # a: atom_delete only, 2 successors
        q_delete / 2 + q_reorder / 2,    # b: BOTH families, summed
        q_reorder / 2,                   # c: bond_reorder only
        q_insert / 1,                    # d: atom_insert only, 1 successor
    )
    assert empirical_family_law(row, FROZEN_Q) == pytest.approx(expected)


def test_empirical_family_is_normalized_because_of_the_cross_family_sum():
    """Dropping the sum would silently produce a sub-probability."""
    row = make_row()
    assert math.isclose(sum(empirical_family_law(row, FROZEN_Q)), 1.0, abs_tol=1e-12)


def test_empirical_family_ignores_families_with_no_legal_successor_here():
    """A family absent from the state must not steal mass through renormalization."""
    row = make_row(
        families=(("atom_delete",), ("atom_delete",), ("atom_delete",), ("atom_delete",))
    )
    law = empirical_family_law(row, FROZEN_Q)
    assert law == pytest.approx((0.25, 0.25, 0.25, 0.25))


def test_empirical_family_refuses_to_silently_become_uniform():
    """A state whose legal families carry no frozen mass must fail loudly."""
    row = make_row(
        families=(("ring_system_grow",),) * 4,
    )
    with pytest.raises(TransportLawError, match="frozen empirical mass"):
        empirical_family_law(row, FROZEN_Q)


def test_empirical_family_differs_from_uniform_when_family_sizes_differ():
    """If these two coincided, the experiment would have two arms, not three."""
    row = make_row()
    assert total_variation(
        empirical_family_law(row, FROZEN_Q), uniform_canonical_law(row)
    ) > 0.01


# ---- dispatch and divergence ----------------------------------------------


def test_law_for_arm_covers_every_declared_arm():
    row = make_row()
    for arm in ARMS:
        law = law_for_arm(arm, row, FROZEN_Q)
        assert len(law) == row.support_size
        assert math.isclose(sum(law), 1.0, abs_tol=1e-9)


def test_law_for_arm_refuses_an_undeclared_arm():
    with pytest.raises(TransportLawError, match="unknown arm"):
        law_for_arm("greedy", make_row(), FROZEN_Q)


def test_arm_divergence_flags_a_single_successor_state_as_degenerate():
    """With one legal successor all three laws are the delta. Nothing is measured."""
    row = SuccessorRow(
        source_key="CCO",
        successor_keys=("a",),
        reference_probabilities=(1.0,),
        alias_counts=(1,),
        families=(("atom_delete",),),
    )
    divergence = arm_divergence(row, FROZEN_Q)
    assert divergence.degenerate
    assert divergence.minimum == pytest.approx(0.0)


def test_arm_divergence_reports_three_distinct_laws_on_a_real_row():
    divergence = arm_divergence(make_row(), FROZEN_Q)
    assert set(divergence.pairwise) == {
        f"{ARM_REFERENCE}|{ARM_UNIFORM}",
        f"{ARM_REFERENCE}|{ARM_EMPIRICAL_FAMILY}",
        f"{ARM_UNIFORM}|{ARM_EMPIRICAL_FAMILY}",
    }
    assert not divergence.degenerate
    assert divergence.minimum > 0.0


# ---- sampling -------------------------------------------------------------


def test_sample_index_is_inverse_cdf():
    law = (0.5, 0.25, 0.15, 0.10)
    assert sample_index(law, 0.0) == 0
    assert sample_index(law, 0.49) == 0
    assert sample_index(law, 0.51) == 1
    assert sample_index(law, 0.76) == 2
    assert sample_index(law, 0.91) == 3
    assert sample_index(law, 0.9999999) == 3


def test_sample_index_never_leaves_the_support_on_rounding():
    """Float accumulation must not index past the end."""
    law = tuple([1.0 / 3.0] * 3)
    assert sample_index(law, 0.999999999999) == 2


def test_common_random_numbers_couple_the_arms_where_the_laws_agree():
    """Identical laws plus identical variates must give identical actions.

    This is the property that makes a trajectory difference attributable to the
    law rather than to the sampler's seed.
    """
    row = make_row(reference_probabilities=(0.25, 0.25, 0.25, 0.25))
    variates = [0.05, 0.3, 0.6, 0.95]
    reference = [sample_index(law_for_arm(ARM_REFERENCE, row, FROZEN_Q), u) for u in variates]
    uniform = [sample_index(law_for_arm(ARM_UNIFORM, row, FROZEN_Q), u) for u in variates]
    assert reference == uniform


def test_sample_index_rejects_an_out_of_range_variate():
    with pytest.raises(TransportLawError, match="outside"):
        sample_index((0.5, 0.5), 1.0)


def test_sample_index_refuses_an_empty_support():
    with pytest.raises(TransportLawError, match="empty support"):
        sample_index((), 0.5)


# ---- round trip -----------------------------------------------------------


def test_row_round_trips_through_json():
    row = make_row(cells=(("atom_delete:atom_delete",),) * 4)
    assert SuccessorRow.from_json(row.to_json()) == row


def test_total_variation_needs_one_shared_support():
    with pytest.raises(TransportLawError, match="one support"):
        total_variation((0.5, 0.5), (0.3, 0.3, 0.4))
