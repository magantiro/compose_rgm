"""Scientific invariants for Q(o | x, M, z) and option conditioning."""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.control.macro_engine import BUILD_RING_SYSTEM, MACRO_FAMILIES
from compose_v4.control.option_selector import (
    BUILD_RING_SYSTEM_OPTION,
    GENERIC_OPTION,
    MACRO_OPTIONS,
    OPTION_GROUP_BY_NAME,
    OPTIONS,
    applicable_options,
    balanced_option_prior,
    bundle_identity,
    conditioned_action_distribution,
    option_horizon,
    primitive_option_at_step,
    retain_product_applicable_options,
    sample_option,
)


def test_option_vocabulary_covers_the_complete_macro_inventory() -> None:
    assert set(MACRO_OPTIONS) == set(MACRO_FAMILIES)
    assert OPTIONS[0] == GENERIC_OPTION
    assert OPTIONS[-1] == BUILD_RING_SYSTEM_OPTION
    assert set(OPTIONS) == set(OPTION_GROUP_BY_NAME)


def test_generic_is_permanently_applicable_even_without_macro_support() -> None:
    assert applicable_options([], []) == (GENERIC_OPTION,)


def test_applicability_uses_only_region_admissible_families() -> None:
    fams = ["atom_insert", "cycle_close", "bond_insert", "atom_delete"]
    options = applicable_options(fams, [0, 3], n_free_slots=8)
    assert GENERIC_OPTION in options
    assert "grow" in options and "shrink" in options
    assert "cyclize" not in options, "closure is outside this region's support"
    assert BUILD_RING_SYSTEM_OPTION in options


def test_compound_option_requires_its_eight_growth_slots() -> None:
    fams = ["atom_insert"]
    assert BUILD_RING_SYSTEM_OPTION not in applicable_options(fams, [0], n_free_slots=7)
    assert BUILD_RING_SYSTEM_OPTION in applicable_options(fams, [0], n_free_slots=8)


def test_product_applicability_removes_unsatisfiable_macros_but_never_generic() -> None:
    options = (GENERIC_OPTION, "cyclize", "annulate", BUILD_RING_SYSTEM_OPTION)
    retained = retain_product_applicable_options(
        options, lambda option: option in {"cyclize", BUILD_RING_SYSTEM_OPTION}
    )
    assert retained == (GENERIC_OPTION, "cyclize", BUILD_RING_SYSTEM_OPTION)


def test_bundle_identity_preserves_distinct_parent_lineages() -> None:
    region_key = ((0, 1), ((0, 2),))
    first = bundle_identity("CC", 0, region_key, "cyclize")
    repeated = bundle_identity("CC", 0, region_key, "cyclize")
    other_lineage = bundle_identity("CC", 1, region_key, "cyclize")
    assert first == repeated
    assert first != other_lineage


def test_prior_balances_groups_before_variants() -> None:
    a = (GENERIC_OPTION, "grow", "cyclize", "open")
    qa = balanced_option_prior(a, exploration=0.0)
    assert qa == pytest.approx(np.full(4, 0.25))

    # Adding ring variants divides the ring-purpose mass; it does not increase it.
    b = a + ("annulate", "small_ring", BUILD_RING_SYSTEM_OPTION)
    qb = balanced_option_prior(b, exploration=0.0)
    ring = [b.index(x) for x in ("cyclize", "annulate", "small_ring", BUILD_RING_SYSTEM_OPTION)]
    assert float(qb[ring].sum()) == pytest.approx(0.25)
    assert qb[b.index(GENERIC_OPTION)] == pytest.approx(0.25)


def test_option_exploration_floor_is_positive_and_auditable() -> None:
    opts = tuple(OPTIONS)
    eps = 0.1
    q = balanced_option_prior(opts, exploration=eps)
    assert q.sum() == pytest.approx(1.0)
    assert np.all(q >= eps / len(opts) - 1e-15)
    choice = sample_option(opts, np.random.default_rng(7), exploration=eps)
    assert choice.selected in opts
    assert choice.selected_probability == pytest.approx(
        choice.probabilities[choice.applicable.index(choice.selected)]
    )


def test_build_ring_system_has_the_declared_eleven_step_schedule() -> None:
    expected = [macro for macro, length in BUILD_RING_SYSTEM for _ in range(length)]
    assert len(expected) == 11
    observed = [
        primitive_option_at_step(BUILD_RING_SYSTEM_OPTION, step) for step in range(len(expected))
    ]
    assert observed == expected
    assert option_horizon(BUILD_RING_SYSTEM_OPTION, 99) == 11
    assert primitive_option_at_step(BUILD_RING_SYSTEM_OPTION, 11) is None


def test_only_generic_retains_the_qualified_multistep_region_horizon() -> None:
    assert option_horizon(GENERIC_OPTION, 16) == 16
    assert option_horizon("cyclize", 16) == 1
    assert option_horizon("scaffold_extend", 16) == 1


def test_generic_conditioning_preserves_the_frozen_region_law() -> None:
    fams = ["atom_insert", "cycle_close", "atom_delete", "bond_reorder"]
    probs = np.array([0.55, 0.05, 0.30, 0.10])
    out = conditioned_action_distribution(fams, probs, [0, 2, 3], GENERIC_OPTION)
    assert out.active_macro is None
    assert out.indices.tolist() == [0, 2, 3]
    assert out.probabilities == pytest.approx(probs[[0, 2, 3]] / 0.95)


def test_macro_conditioning_restricts_support_without_touching_rtheta() -> None:
    fams = ["atom_insert", "cycle_close", "bond_insert", "atom_delete"]
    probs = np.array([0.90, 0.0001, 0.0099, 0.09])
    clean = np.array([True, True, False, True])
    out = conditioned_action_distribution(fams, probs, range(4), "cyclize", clean=clean)
    assert out.active_macro == "cyclize"
    assert out.indices.tolist() == [1]
    assert out.probabilities.tolist() == pytest.approx([1.0])
    assert probs.tolist() == [0.90, 0.0001, 0.0099, 0.09]


def test_compound_phase_conditioning_uses_the_existing_macro_machinery() -> None:
    fams = ["atom_insert", "cycle_close", "bond_insert", "bond_reorder"]
    probs = np.full(4, 0.25)
    grow = conditioned_action_distribution(fams, probs, range(4), BUILD_RING_SYSTEM_OPTION, step=0)
    close = conditioned_action_distribution(fams, probs, range(4), BUILD_RING_SYSTEM_OPTION, step=8)
    restate = conditioned_action_distribution(
        fams, probs, range(4), BUILD_RING_SYSTEM_OPTION, step=9
    )
    assert grow.active_macro == "scaffold_extend" and grow.indices.tolist() == [0]
    assert close.active_macro == "append_system" and close.indices.tolist() == [1, 2]
    assert restate.active_macro == "restate" and restate.indices.tolist() == [3]


def test_invalid_prior_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        balanced_option_prior((GENERIC_OPTION, GENERIC_OPTION))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        balanced_option_prior((GENERIC_OPTION,), exploration=1.1)
    with pytest.raises(KeyError, match="unknown"):
        primitive_option_at_step("teleport", 0)
    with pytest.raises(ValueError, match="aligned"):
        conditioned_action_distribution(["atom_insert"], [0.2, 0.8], [0], GENERIC_OPTION)
    with pytest.raises(IndexError, match="outside"):
        applicable_options(["atom_insert"], [1])
