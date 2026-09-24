"""Attachment-aware fragment control: declared interfaces, staging, redirection.

These tests fix the three properties the controller has to have for its numbers
to mean anything:

* the declared interfaces it steers toward are the ones the BENCHMARK declared,
  recovered through the canonical slot reordering the region lock also uses;
* switching it off leaves the sampler's decision sequence untouched, so the
  frozen-sampler rows stay reproducible; and
* switching it on actually deletes the undeclared-interface events from the
  fiber, rather than filtering them at the endpoint.

The negative controls matter as much as the positives here: a prompt that
declares no interface (``superstructure_generation``) must come out of the
controller bit-identical, and a mutation that removes the restriction must make
a test fail.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from compose_v4.benchmark.fragment_attachment_control import (
    REDIRECT_COMPLETED_INTERFACES,
    AttachmentControlConfig,
    AttachmentController,
    AttachmentSpec,
    external_neighbour_count,
)
from compose_v4.benchmark.fragment_conditioned_sampler import (
    FragmentConditioningError,
    RegionLock,
    SamplerConfig,
    _declared_sites,
    build_prompt_context,
    retained_core,
)
from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    _fragment_spec,
    load_genmol_prompts,
)
from compose_v4.chem.molecular_graph import NULL_IDX

MANIFEST = "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"

_ON = AttachmentControlConfig(enabled=True)
_OFF = AttachmentControlConfig(enabled=False)


def _prompts():
    return load_genmol_prompts(MANIFEST)


def _prompt(task: FragmentTask, drug: str):
    return next(p for p in _prompts() if p.task is task and p.drug_name == drug)


def _first_null_slot(state) -> int:
    nulls = np.flatnonzero(state.atom_types == NULL_IDX)
    assert nulls.size, "the padded proposal state must expose a free slot"
    return int(nulls[0])


def _attach_new_atom(state, anchor: int, *, order: int = 1):
    """A successor that bonds a fresh carbon to ``anchor``; array-level, no executor."""
    successor = copy.deepcopy(state)
    slot = _first_null_slot(state)
    successor.atom_types[slot] = state.atom_types[anchor]
    successor.formal_charges[slot] = 0
    successor.implicit_h_counts[slot] = 3
    successor.bonds[slot][anchor] = order
    successor.bonds[anchor][slot] = order
    return successor


# ---- The interfaces are the benchmark's own ----


def test_two_core_construction_routes_agree_on_every_released_fragment():
    """``retained_core`` and ``_fragment_spec`` must index the same core.

    The controller reads attachment COUNTS from ``_fragment_spec`` and maps
    them through the slot order ``retained_core`` produced.  The two build the
    dummy-stripped core by different routes -- capping the dummy with hydrogen
    and then removing it, versus deleting the dummy outright -- so their
    agreement is a real assumption and is asserted here rather than trusted.
    """
    checked = 0
    for prompt in _prompts():
        for fragment in prompt.fragments:
            core, sites = retained_core(fragment)
            spec = _fragment_spec(fragment)
            assert core.GetNumAtoms() == spec.core.GetNumAtoms(), fragment
            assert [a.GetSymbol() for a in core.GetAtoms()] == [
                a.GetSymbol() for a in spec.core.GetAtoms()
            ], fragment
            assert sites == tuple(
                sorted(site for site, _count in spec.attachment_requirements)
            ), fragment
            checked += 1
    assert checked == 70


def test_declared_interfaces_land_on_locked_slots_carrying_free_valence():
    """Every declared interface is a locked slot, and can still take a bond."""
    config = SamplerConfig()
    for prompt in _prompts():
        if prompt.task is FragmentTask.SCAFFOLD_MORPHING:
            continue  # same prompt strings as linker_design
        context = build_prompt_context(prompt, config=config, control=_ON)
        spec = context.attachment
        assert spec is not None
        locked = set(context.locked_slots)
        assert set(spec.interfaces) <= locked, prompt.drug_name
        assert len(spec.requirements) == len(spec.interfaces)
        for slot in spec.interfaces:
            assert spec.requirement_of(slot) >= 1
        if len(spec.lock_groups) == 1:
            # The dummy was replaced by a hydrogen, so a single-core prompt's
            # site retains a free valence and the interface can be covered.
            for slot in spec.interfaces:
                assert int(context.start_state.implicit_h_counts[slot]) >= 1, (
                    prompt.drug_name,
                    prompt.task.value,
                    slot,
                )


def test_the_constructed_linker_join_consumes_the_declared_valences():
    """MEASURED, and the reason linker design stays withheld.

    ``build_prompt_context`` has to hand the executor a CONNECTED state, so it
    joins the two retained cores with a direct bond -- and that bond is spent
    out of the very valence each fragment declared open.  Twelve of the twenty
    declared linker interfaces are left with no free valence at all, and for
    four of the ten drugs BOTH sites are saturated, so no atom can be attached
    at a declared site without first removing the join.  Removing it in
    isolation disconnects the state, which the executor refuses.  Attachment
    control therefore cannot open this task on its own: the START STATE, not
    the proposal distribution, is what forecloses it.
    """
    saturated = 0
    total = 0
    both_saturated = []
    for prompt in _prompts():
        if prompt.task is not FragmentTask.LINKER_DESIGN:
            continue
        context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
        free = [
            int(context.start_state.implicit_h_counts[slot])
            for slot in context.attachment.interfaces
        ]
        assert len(free) == 2
        total += len(free)
        saturated += sum(1 for h in free if h == 0)
        if all(h == 0 for h in free):
            both_saturated.append(prompt.drug_name)
    assert total == 20
    assert saturated == 12
    assert sorted(both_saturated) == [
        "BARICITINIB",
        "ELIGLUSTAT",
        "ERLOTINIB",
        "SPIRAPRIL",
    ]


def test_declared_interface_count_matches_the_prompt_dummy_count():
    for prompt in _prompts():
        if prompt.task is FragmentTask.SCAFFOLD_MORPHING:
            continue
        context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
        dummies = sum(
            1
            for fragment in prompt.fragments
            for atom in Chem.MolFromSmiles(fragment).GetAtoms()
            if atom.GetAtomicNum() == 0
        )
        declared = sum(
            count
            for fragment in prompt.fragments
            for _site, count in _declared_sites(fragment)
        )
        assert declared == dummies, prompt.drug_name
        assert (
            sum(count for _slot, count in context.attachment.requirements) == dummies
        )


def test_superstructure_declares_no_interface_and_deactivates_the_controller():
    """The vacuous case is a negative control, not a special case for one task."""
    for prompt in _prompts():
        if prompt.task is not FragmentTask.SUPERSTRUCTURE_GENERATION:
            continue
        context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
        assert context.attachment.interfaces == ()
        controller = AttachmentController(
            context.attachment, context.locked_slots, _ON
        )
        assert controller.active is False
        state = context.start_state
        # With no declared interface, every locked atom stays open: growth
        # anywhere on the core is admitted exactly as before.
        for anchor in context.locked_slots[:5]:
            admitted, reason = controller.permits(
                _attach_new_atom(state, anchor), state
            )
            assert admitted and reason == ""


# ---- Pathwise admission ----


def test_interface_restriction_refuses_growth_off_an_undeclared_atom():
    prompt = _prompt(FragmentTask.MOTIF_EXTENSION, "BARICITINIB")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
    controller = AttachmentController(context.attachment, context.locked_slots, _ON)
    assert controller.active
    state = context.start_state
    interface = context.attachment.interfaces[0]
    undeclared = [s for s in context.locked_slots if s != interface]
    assert undeclared

    admitted, _ = controller.permits(_attach_new_atom(state, interface), state)
    assert admitted, "growth at the declared site must be admitted"

    for anchor in undeclared:
        admitted, reason = controller.permits(_attach_new_atom(state, anchor), state)
        assert not admitted and reason == "undeclared_interface", anchor


def test_the_same_undeclared_growth_is_admitted_with_the_controller_off():
    """The restriction is the controller's, not an accident of the state."""
    prompt = _prompt(FragmentTask.MOTIF_EXTENSION, "BARICITINIB")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_OFF)
    controller = AttachmentController(context.attachment, context.locked_slots, _OFF)
    state = context.start_state
    undeclared = [s for s in context.locked_slots if s not in context.attachment.interfaces]
    for anchor in undeclared:
        admitted, reason = controller.permits(_attach_new_atom(state, anchor), state)
        assert admitted and reason == ""


def test_staging_refuses_an_event_that_makes_no_coverage_progress():
    """A decoration prompt with several open sites must cover them first."""
    prompt = _prompt(FragmentTask.SCAFFOLD_DECORATION, "ERLOTINIB")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
    controller = AttachmentController(context.attachment, context.locked_slots, _ON)
    state = context.start_state
    interfaces = context.attachment.interfaces
    assert len(interfaces) >= 2

    assert set(controller.unsatisfied(state)) == set(interfaces)
    covered_once = _attach_new_atom(state, interfaces[0])
    admitted, _ = controller.permits(covered_once, state)
    assert admitted
    assert set(controller.unsatisfied(covered_once)) == set(interfaces[1:])

    # Elaborating the decoration we just placed leaves the other sites open,
    # so while staging is engaged it is refused.
    grown = _attach_new_atom(covered_once, _first_null_slot(state))
    admitted, reason = controller.permits(grown, covered_once)
    assert not admitted and reason == "no_coverage_progress"


def test_staging_releases_once_every_interface_is_covered():
    prompt = _prompt(FragmentTask.SCAFFOLD_DECORATION, "ERLOTINIB")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
    controller = AttachmentController(context.attachment, context.locked_slots, _ON)
    state = context.start_state
    for slot in context.attachment.interfaces:
        state = _attach_new_atom(state, slot)
    assert controller.unsatisfied(state) == ()
    assert controller.all_interfaces_covered(state)

    decorated = _attach_new_atom(state, _first_null_slot(context.start_state))
    admitted, reason = controller.permits(decorated, state)
    assert admitted and reason == ""


def test_interface_release_preserves_first_growth_then_allows_legal_core_growth():
    """Only the permanent post-coverage ban differs between the two arms."""
    from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
    from compose_v4.rewrite.kernel import de_novo_rewrite_system
    from compose_v4.rewrite.operators import AtomInsert

    prompt = _prompt(FragmentTask.MOTIF_EXTENSION, "BARICITINIB")
    strict_config = AttachmentControlConfig(enabled=True)
    release_config = dataclasses.replace(strict_config, restrict_interfaces=False)
    context = build_prompt_context(
        prompt, config=SamplerConfig(), control=strict_config
    )
    strict = AttachmentController(
        context.attachment, context.locked_slots, strict_config
    )
    release = AttachmentController(
        context.attachment, context.locked_slots, release_config
    )
    lock = RegionLock(
        context.start_state,
        context.locked_slots,
        released_pairs=context.attachment.released_pairs,
    )
    system = de_novo_rewrite_system()
    start = context.start_state
    undeclared = next(
        slot
        for slot in context.locked_slots
        if slot not in context.attachment.interfaces
        and int(start.implicit_h_counts[slot]) > 0
    )

    # Before coverage, redirection stays active in BOTH arms. The released
    # arm's attachment-first gate also refuses a non-progressing insertion.
    first = AtomInsert(_first_null_slot(start), 2, 0, 3, ((undeclared, 1),))
    assert strict.redirect("atom_insert", first, start) == release.redirect(
        "atom_insert", first, start
    )
    redirected = release.redirect("atom_insert", first, start)
    assert redirected.neighbors[0][0] in context.attachment.interfaces
    uncovered = system.apply(start, "atom_insert", first)
    assert release.permits(uncovered, start) == (False, "no_coverage_progress")

    covered = system.apply(start, "atom_insert", redirected)
    assert strict.permits(covered, start) == (True, "")
    assert release.permits(covered, start) == (True, "")
    assert strict.all_interfaces_covered(covered)
    assert release.all_interfaces_covered(covered)
    assert lock.permits(covered)

    second = AtomInsert(_first_null_slot(covered), 2, 0, 3, ((undeclared, 1),))
    grown = system.apply(covered, "atom_insert", second)
    assert strict.permits(grown, covered) == (False, "undeclared_interface")
    assert release.permits(grown, covered) == (True, "")
    assert lock.permits(grown)

    # Exact locked atom/bond identity survives both accepted steps, and the
    # production executor produces chemically valid connected endpoints.
    for state in (covered, grown):
        for slot in context.locked_slots:
            assert state.atom_types[slot] == start.atom_types[slot]
            for other in context.locked_slots:
                assert state.bonds[slot, other] == start.bonds[slot, other]
        mol = Chem.MolFromSmiles(molecular_graph_to_smiles(state))
        assert mol is not None and len(Chem.GetMolFrags(mol)) == 1


def test_single_interface_policy_routes_by_declared_count_and_coverage_only():
    """One-interface release, multi-interface restriction, zero-interface no-op."""
    from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
    from compose_v4.rewrite.kernel import de_novo_rewrite_system
    from compose_v4.rewrite.operators import AtomInsert

    policy = AttachmentControlConfig(
        enabled=True, restrict_interfaces="single_interface_after_coverage"
    )
    strict = AttachmentControlConfig(enabled=True)
    released = AttachmentControlConfig(enabled=True, restrict_interfaces=False)
    system = de_novo_rewrite_system()
    for task, drug, expected_interfaces in (
        (FragmentTask.MOTIF_EXTENSION, "BARICITINIB", 1),
        (FragmentTask.SCAFFOLD_DECORATION, "ERLOTINIB", 3),
        (FragmentTask.SUPERSTRUCTURE_GENERATION, "BARICITINIB", 0),
    ):
        prompt = _prompt(task, drug)
        context = build_prompt_context(prompt, config=SamplerConfig(), control=policy)
        assert len(context.attachment.interfaces) == expected_interfaces
        controllers = [
            AttachmentController(context.attachment, context.locked_slots, config)
            for config in (strict, released, policy)
        ]
        start = context.start_state
        undeclared = next(
            slot
            for slot in context.locked_slots
            if slot not in context.attachment.interfaces
            and int(start.implicit_h_counts[slot]) > 0
        )
        lock = RegionLock(
            start, context.locked_slots,
            released_pairs=context.attachment.released_pairs,
        )
        state = start
        if expected_interfaces:
            for interface in context.attachment.interfaces:
                action = AtomInsert(
                    _first_null_slot(state), 2, 0, 3, ((interface, 1),)
                )
                successor = system.apply(state, "atom_insert", action)
                assert lock.permits(successor)
                assert controllers[2].permits(successor, state) == (True, "")
                state = successor
            assert controllers[2].all_interfaces_covered(state)

        outside = AtomInsert(
            _first_null_slot(state), 2, 0, 3, ((undeclared, 1),)
        )
        successor = system.apply(state, "atom_insert", outside)
        assert lock.permits(successor)
        decisions = [controller.permits(successor, state) for controller in controllers]
        if expected_interfaces == 1:
            assert decisions[0] == (False, "undeclared_interface")
            assert decisions[1:] == [(True, ""), (True, "")]
        elif expected_interfaces > 1:
            assert decisions[0] == decisions[2] == (
                False, "undeclared_interface"
            )
            assert decisions[1] == (True, "")
        else:
            assert decisions == [(True, "")] * 3
        for slot in context.locked_slots:
            assert successor.atom_types[slot] == start.atom_types[slot]
            for other in context.locked_slots:
                assert successor.bonds[slot, other] == start.bonds[slot, other]
        mol = Chem.MolFromSmiles(molecular_graph_to_smiles(successor))
        assert mol is not None and len(Chem.GetMolFrags(mol)) == 1


def test_interface_policy_rejects_unrecognized_mode():
    with pytest.raises(ValueError, match="unknown interface restriction mode"):
        AttachmentControlConfig(enabled=True, restrict_interfaces="other")


def test_coverage_ignores_neighbours_inside_the_retained_region():
    """A locked neighbour is not an EXTERNAL neighbour; the linker join is the case."""
    prompt = _prompt(FragmentTask.LINKER_DESIGN, "ELIGLUSTAT")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
    state = context.start_state
    for slot in context.attachment.interfaces:
        assert external_neighbour_count(state, slot, context.locked_slots) == 0
    controller = AttachmentController(context.attachment, context.locked_slots, _ON)
    assert set(controller.unsatisfied(state)) == set(context.attachment.interfaces)


# ---- Attachment redirection ----


def test_redirection_moves_the_anchor_and_keeps_the_prior_payload():
    from compose_v4.rewrite.operators import AtomInsert

    prompt = _prompt(FragmentTask.MOTIF_EXTENSION, "BARICITINIB")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
    controller = AttachmentController(context.attachment, context.locked_slots, _ON)
    state = context.start_state
    interface = context.attachment.interfaces[0]
    undeclared = next(s for s in context.locked_slots if s != interface)

    free = int(state.implicit_h_counts[interface])
    assert free == 1, "this fixture site declares exactly one open valence"

    # An order the declared site can take is carried across untouched.
    proposed = AtomInsert(_first_null_slot(state), 2, 0, 3, ((undeclared, 1),))
    redirected = controller.redirect("atom_insert", proposed, state)
    assert redirected is not proposed
    assert redirected.neighbors == ((interface, 1),)
    # Payload is the PRIOR's: element, charge, derived hydrogens, bond order.
    assert redirected.slot == proposed.slot
    assert redirected.atom_type == proposed.atom_type
    assert redirected.formal_charge == proposed.formal_charge
    assert redirected.implicit_h_count == proposed.implicit_h_count


def test_redirection_realizes_an_order_the_declared_site_cannot_take():
    """Constrained realization: the element survives, the order is lowered."""
    from compose_v4.rewrite.operators import AtomInsert

    prompt = _prompt(FragmentTask.MOTIF_EXTENSION, "BARICITINIB")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
    controller = AttachmentController(context.attachment, context.locked_slots, _ON)
    state = context.start_state
    interface = context.attachment.interfaces[0]
    undeclared = next(s for s in context.locked_slots if s != interface)
    free = int(state.implicit_h_counts[interface])
    assert free == 1

    proposed = AtomInsert(_first_null_slot(state), 2, 0, 2, ((undeclared, 2),))
    redirected = controller.redirect("atom_insert", proposed, state)
    assert redirected.neighbors == ((interface, free),)
    assert redirected.atom_type == proposed.atom_type
    # The hydrogen count is re-derived for the reduced order, so the element's
    # total valence is unchanged: 2 bonds + 2 H becomes 1 bond + 3 H.
    assert redirected.implicit_h_count == proposed.implicit_h_count + 2 - free
    assert (
        redirected.implicit_h_count + redirected.neighbors[0][1]
        == proposed.implicit_h_count + proposed.neighbors[0][1]
    )


def test_redirection_refuses_to_manufacture_a_bond_at_a_saturated_site():
    """A linker interface has no valence left; the action must pass through."""
    from compose_v4.rewrite.operators import AtomInsert

    prompt = _prompt(FragmentTask.LINKER_DESIGN, "ELIGLUSTAT")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
    controller = AttachmentController(context.attachment, context.locked_slots, _ON)
    state = context.start_state
    assert all(
        int(state.implicit_h_counts[slot]) == 0
        for slot in context.attachment.interfaces
    )
    undeclared = next(
        s for s in context.locked_slots if s not in context.attachment.interfaces
    )
    proposed = AtomInsert(_first_null_slot(state), 2, 0, 3, ((undeclared, 1),))
    assert controller.redirect("atom_insert", proposed, state) is proposed


def test_redirection_leaves_an_already_admissible_anchor_alone():
    from compose_v4.rewrite.operators import AtomInsert

    prompt = _prompt(FragmentTask.MOTIF_EXTENSION, "BARICITINIB")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
    controller = AttachmentController(context.attachment, context.locked_slots, _ON)
    state = context.start_state
    interface = context.attachment.interfaces[0]
    proposed = AtomInsert(_first_null_slot(state), 2, 0, 3, ((interface, 1),))
    assert controller.redirect("atom_insert", proposed, state) is proposed


def test_redirection_is_the_identity_when_the_controller_is_off():
    from compose_v4.rewrite.operators import AtomInsert

    prompt = _prompt(FragmentTask.MOTIF_EXTENSION, "BARICITINIB")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_OFF)
    controller = AttachmentController(context.attachment, context.locked_slots, _OFF)
    state = context.start_state
    undeclared = next(
        s for s in context.locked_slots if s not in context.attachment.interfaces
    )
    proposed = AtomInsert(_first_null_slot(state), 2, 0, 3, ((undeclared, 1),))
    assert controller.redirect("atom_insert", proposed, state) is proposed


def test_redirection_never_touches_a_family_without_an_anchor():
    from compose_v4.rewrite.operators import AtomDelete, BondReorder

    prompt = _prompt(FragmentTask.SCAFFOLD_DECORATION, "ERLOTINIB")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
    controller = AttachmentController(context.attachment, context.locked_slots, _ON)
    state = context.start_state
    for rule, action in (
        ("atom_delete", AtomDelete(3)),
        ("bond_reorder", BondReorder(1, 2, 2)),
    ):
        assert controller.redirect(rule, action, state) is action


def test_redirection_spreads_across_unsatisfied_interfaces():
    """The target is the LEAST covered site, so several open sites get filled."""
    from compose_v4.rewrite.operators import AtomInsert

    prompt = _prompt(FragmentTask.SCAFFOLD_DECORATION, "MARIBAVIR")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
    controller = AttachmentController(context.attachment, context.locked_slots, _ON)
    interfaces = context.attachment.interfaces
    assert len(interfaces) >= 5

    state = context.start_state
    undeclared = next(s for s in context.locked_slots if s not in interfaces)
    hit = []
    for _ in range(len(interfaces)):
        proposed = AtomInsert(_first_null_slot(state), 2, 0, 3, ((undeclared, 1),))
        target = controller.redirect("atom_insert", proposed, state).neighbors[0][0]
        hit.append(target)
        state = _attach_new_atom(state, target)
    assert sorted(hit) == sorted(interfaces)
    assert controller.unsatisfied(state) == ()


def test_redirection_moves_a_redundant_declared_anchor_to_an_open_interface():
    """A declared site stops being admissible during staging once it is covered."""
    from compose_v4.rewrite.operators import AtomInsert

    prompt = _prompt(FragmentTask.SCAFFOLD_DECORATION, "ERLOTINIB")
    repaired = dataclasses.replace(
        _ON, redirect_attachment=REDIRECT_COMPLETED_INTERFACES
    )
    context = build_prompt_context(prompt, config=SamplerConfig(), control=repaired)
    controller = AttachmentController(context.attachment, context.locked_slots, repaired)
    first, second = context.attachment.interfaces[:2]
    state = _attach_new_atom(context.start_state, first)
    assert first not in controller.unsatisfied(state)
    assert second in controller.unsatisfied(state)

    proposed = AtomInsert(_first_null_slot(state), 2, 0, 3, ((first, 1),))
    redirected = controller.redirect("atom_insert", proposed, state)
    assert redirected.neighbors == ((second, 1),)
    assert redirected.atom_type == proposed.atom_type
    assert redirected.slot == proposed.slot

    historical = AttachmentController(context.attachment, context.locked_slots, _ON)
    assert historical.redirect("atom_insert", proposed, state) is proposed

    already_open = dataclasses.replace(proposed, neighbors=((second, 1),))
    assert controller.redirect("atom_insert", already_open, state) is already_open


# ---- The linker join ----


def test_region_lock_pins_the_linker_join_when_the_controller_is_off():
    prompt = _prompt(FragmentTask.LINKER_DESIGN, "ELIGLUSTAT")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_OFF)
    assert context.attachment.released_pairs == frozenset()
    lock = RegionLock(
        context.start_state,
        context.locked_slots,
        released_pairs=context.attachment.released_pairs,
    )
    left, right = context.attachment.lock_groups
    joined = [
        (i, j)
        for i in left
        for j in right
        if int(context.start_state.bonds[i][j]) > 0
    ]
    assert len(joined) == 1, "the constructed start state joins the cores exactly once"
    i, j = joined[0]
    broken = copy.deepcopy(context.start_state)
    broken.bonds[i][j] = 0
    broken.bonds[j][i] = 0
    assert lock.permits(broken) is False


def test_region_lock_releases_only_the_join_pair_when_the_controller_is_on():
    prompt = _prompt(FragmentTask.LINKER_DESIGN, "ELIGLUSTAT")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
    released = context.attachment.released_pairs
    assert len(released) == 1
    (pair,) = released
    i, j = sorted(pair)
    lock = RegionLock(
        context.start_state, context.locked_slots, released_pairs=released
    )

    opened = copy.deepcopy(context.start_state)
    opened.bonds[i][j] = 0
    opened.bonds[j][i] = 0
    assert lock.permits(opened) is True, "the constructed join must be openable"

    # Every OTHER locked pair is still pinned, and both endpoints keep their
    # element and charge: releasing the join is not releasing the cores.
    others = [
        (a, b)
        for index, a in enumerate(context.locked_slots)
        for b in context.locked_slots[index + 1 :]
        if int(context.start_state.bonds[a][b]) > 0 and {a, b} != {i, j}
    ]
    assert others
    for a, b in others:
        mutated = copy.deepcopy(context.start_state)
        mutated.bonds[a][b] = 0
        mutated.bonds[b][a] = 0
        assert lock.permits(mutated) is False, (a, b)
    retyped = copy.deepcopy(context.start_state)
    retyped.atom_types[i] = int(retyped.atom_types[i]) + 1
    assert lock.permits(retyped) is False


def test_zero_atom_linker_is_refused_by_the_separation_check():
    prompt = _prompt(FragmentTask.LINKER_DESIGN, "ELIGLUSTAT")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
    controller = AttachmentController(context.attachment, context.locked_slots, _ON)
    # The start state IS the defect the invalidation report measured: the two
    # cores directly bonded, with zero linker atoms between them.
    assert controller.cores_are_separated(context.start_state) is False

    separated = copy.deepcopy(context.start_state)
    (pair,) = context.attachment.released_pairs
    i, j = sorted(pair)
    separated.bonds[i][j] = 0
    separated.bonds[j][i] = 0
    assert controller.cores_are_separated(separated) is True


def test_separation_check_is_vacuous_for_a_single_core_prompt():
    prompt = _prompt(FragmentTask.MOTIF_EXTENSION, "BARICITINIB")
    context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
    controller = AttachmentController(context.attachment, context.locked_slots, _ON)
    assert len(context.attachment.lock_groups) == 1
    assert controller.cores_are_separated(context.start_state) is True


# ---- Identity of the off path ----


def test_switching_the_controller_off_restores_every_decision():
    """OFF must be a no-op on admission, redirection and the lock alike."""
    for task in (
        FragmentTask.MOTIF_EXTENSION,
        FragmentTask.SCAFFOLD_DECORATION,
        FragmentTask.SUPERSTRUCTURE_GENERATION,
        FragmentTask.LINKER_DESIGN,
    ):
        for drug in ("BARICITINIB", "MARIBAVIR"):
            prompt = _prompt(task, drug)
            off = build_prompt_context(prompt, config=SamplerConfig(), control=_OFF)
            controller = AttachmentController(off.attachment, off.locked_slots, _OFF)
            assert controller.active is False
            assert off.attachment.released_pairs == frozenset()
            lock = RegionLock(
                off.start_state,
                off.locked_slots,
                released_pairs=off.attachment.released_pairs,
            )
            legacy = RegionLock(off.start_state, off.locked_slots)
            assert lock._bonds == legacy._bonds
            state = off.start_state
            for anchor in off.locked_slots:
                admitted, reason = controller.permits(
                    _attach_new_atom(state, anchor), state
                )
                assert admitted and reason == ""


def test_the_start_state_never_satisfies_a_declared_interface():
    """Coverage must start at zero, or the staging phase would be skipped."""
    for prompt in _prompts():
        if prompt.task is FragmentTask.SCAFFOLD_MORPHING:
            continue
        context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
        controller = AttachmentController(
            context.attachment, context.locked_slots, _ON
        )
        assert set(controller.unsatisfied(context.start_state)) == set(
            context.attachment.interfaces
        ), (prompt.drug_name, prompt.task.value)


@pytest.mark.parametrize(
    "task",
    [
        FragmentTask.MOTIF_EXTENSION,
        FragmentTask.SCAFFOLD_DECORATION,
        FragmentTask.SUPERSTRUCTURE_GENERATION,
    ],
)
def test_every_drug_builds_a_context_under_the_controller(task):
    for prompt in _prompts():
        if prompt.task is not task:
            continue
        context = build_prompt_context(prompt, config=SamplerConfig(), control=_ON)
        assert context.attachment is not None
        assert len(context.attachment.lock_groups) == 1


# ---- The aggregator's refusals ----
#
# A before/after table is only evidence if nothing per-instance can hide inside
# it.  These pin the refusals rather than the happy path: each one builds an
# arm that SHOULD be rejected and requires the rejection.

import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from build_fragment_attachment_table import (
    ArmError,
    check_row_guards,
    load_arm,
)


def _shard(
    tmp_path: Path,
    name: str,
    *,
    sampler: str = "sampler-a",
    control: str = "control-a",
    drug: str = "BARICITINIB",
    seed: int = 0,
    produced: int = 40,
    valid: int | None = None,
    contained: int = 30,
    success: int = 20,
) -> Path:
    import json

    payload = {
        "sampler": {"config_sha256": sampler},
        "attachment_control": {"config_sha256": control, "arm": "x", "config": {}},
        "kernel": {"rdkit": "2026.03.6"},
        "checkpoint": {"path": "/ckpt.pt"},
        "results": {
            "motif_extension": {
                "per_drug": {
                    drug: [
                        {
                            "seed": seed,
                            "attempts": 100,
                            "committed_endpoints": produced,
                            "committed_chemically_valid": (
                                produced if valid is None else valid
                            ),
                            "committed_fragment_preserving": contained,
                            "emitted_nonempty": success,
                            "committed_endpoint_smiles": ["C"] * produced,
                            "emitted_samples": ["C"] * success + [""] * (100 - success),
                        }
                    ]
                }
            }
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_aggregator_refuses_shards_that_disagree_on_the_controller(tmp_path):
    _shard(tmp_path, "a.json", control="control-a")
    _shard(tmp_path, "b.json", control="control-b", drug="ERLOTINIB")
    with pytest.raises(ArmError, match="attachment controller"):
        load_arm(tmp_path, label="arm")


def test_aggregator_refuses_shards_that_disagree_on_the_sampler(tmp_path):
    _shard(tmp_path, "a.json", sampler="sampler-a")
    _shard(tmp_path, "b.json", sampler="sampler-b", drug="ERLOTINIB")
    with pytest.raises(ArmError, match="sampler"):
        load_arm(tmp_path, label="arm")


def test_aggregator_names_a_pre_controller_shard_instead_of_reading_it_as_agreement(
    tmp_path,
):
    import json

    path = _shard(tmp_path, "a.json")
    payload = json.loads(path.read_text())
    del payload["attachment_control"]
    path.write_text(json.dumps(payload))
    arm = load_arm(tmp_path, label="arm")
    assert arm["attachment_sha256"] == "absent:pre_attachment_control"


def test_row_guard_refuses_a_committed_endpoint_that_is_not_chemically_valid():
    row = {
        "drug": "X", "seed": 0, "committed_endpoints": 40,
        "committed_chemically_valid": 39, "committed_fragment_preserving": 30,
        "emitted_nonempty": 20,
    }
    with pytest.raises(ArmError, match="BY CONSTRUCTION"):
        check_row_guards("motif_extension", row)


def test_row_guard_refuses_success_above_containment():
    row = {
        "drug": "X", "seed": 0, "committed_endpoints": 40,
        "committed_chemically_valid": 40, "committed_fragment_preserving": 10,
        "emitted_nonempty": 20,
    }
    with pytest.raises(ArmError, match="is violated"):
        check_row_guards("motif_extension", row)


def test_row_guard_refuses_containment_above_produced():
    row = {
        "drug": "X", "seed": 0, "committed_endpoints": 40,
        "committed_chemically_valid": 40, "committed_fragment_preserving": 41,
        "emitted_nonempty": 20,
    }
    with pytest.raises(ArmError, match="is violated"):
        check_row_guards("motif_extension", row)


def test_row_guard_accepts_the_admissible_ordering():
    check_row_guards(
        "motif_extension",
        {
            "drug": "X", "seed": 0, "committed_endpoints": 40,
            "committed_chemically_valid": 40, "committed_fragment_preserving": 30,
            "emitted_nonempty": 30,
        },
    )


# ---- The corrected linker start state ----
#
# The original construction bonds the two retained cores DIRECTLY, which makes a
# genuine linker unreachable: the join consumes the hydrogen each declared site
# needed, so on several released drugs both sites start saturated and no first
# event can increase coverage.  ``linker_bridge_atoms`` seeds unlocked carbons
# between the sites instead.  These tests pin the properties that make the
# corrected state a fair statement of the task, and they drive the production
# ``build_prompt_context`` rather than reconstructing a start state locally.


def _linker_prompts():
    return [p for p in _prompts() if p.task is FragmentTask.LINKER_DESIGN]


def test_direct_join_leaves_the_two_cores_bonded_on_every_released_drug():
    """The defect itself, pinned so a silent revert is visible."""
    for prompt in _linker_prompts():
        context = build_prompt_context(prompt, linker_bridge_atoms=0)
        controller = AttachmentController(
            context.attachment, context.locked_slots, _ON
        )
        assert not controller.cores_are_separated(context.start_state), (
            f"{prompt.drug_name}: the direct join is supposed to leave the cores "
            "bonded; if this passes, the zero-atom-linker defect is gone and the "
            "invalidation needs revisiting"
        )


def test_a_seeded_bridge_separates_the_cores_on_every_released_drug():
    for prompt in _linker_prompts():
        context = build_prompt_context(prompt, linker_bridge_atoms=1)
        controller = AttachmentController(
            context.attachment, context.locked_slots, _ON
        )
        assert controller.cores_are_separated(context.start_state), (
            f"{prompt.drug_name}: a seeded bridge must leave no direct core-core bond"
        )


def test_the_seeded_bridge_is_not_locked():
    """The linker is what the generator designs, so it must stay editable."""
    for prompt in _linker_prompts():
        direct = build_prompt_context(prompt, linker_bridge_atoms=0)
        seeded = build_prompt_context(prompt, linker_bridge_atoms=2)
        assert len(seeded.locked_slots) == len(direct.locked_slots), (
            f"{prompt.drug_name}: seeding a bridge must not enlarge the retained "
            "region -- the bridge belongs to neither core"
        )
        real = int(np.count_nonzero(seeded.start_state.atom_types != NULL_IDX))
        direct_real = int(np.count_nonzero(direct.start_state.atom_types != NULL_IDX))
        assert real == direct_real + 2, (
            f"{prompt.drug_name}: two seeded carbons must add exactly two atoms"
        )


def test_seeding_a_bridge_releases_nothing_from_the_lock():
    """A released pair only exists to undo a join this adapter constructed.

    With a bridge there is no core-core bond to undo, so releasing one would
    weaken the retained region for no reason.
    """
    for prompt in _linker_prompts():
        seeded = build_prompt_context(
            prompt, control=_ON, linker_bridge_atoms=1
        )
        assert seeded.attachment.released_pairs == frozenset(), (
            f"{prompt.drug_name}: a seeded-bridge start has no constructed join "
            "to release"
        )
        direct = build_prompt_context(prompt, control=_ON, linker_bridge_atoms=0)
        assert direct.attachment.released_pairs, (
            f"{prompt.drug_name}: the direct join must still be released when it "
            "is the construction in use"
        )


def test_bridge_seeding_is_inert_for_a_single_core_task():
    """Only linker-shaped prompts have anything to bridge."""
    for task in (FragmentTask.MOTIF_EXTENSION, FragmentTask.SCAFFOLD_DECORATION):
        prompt = _prompt(task, "BARICITINIB")
        base = build_prompt_context(prompt, linker_bridge_atoms=0)
        seeded = build_prompt_context(prompt, linker_bridge_atoms=3)
        assert seeded.start_smiles == base.start_smiles
        assert seeded.locked_slots == base.locked_slots


def test_a_negative_bridge_length_is_refused():
    prompt = _linker_prompts()[0]
    with pytest.raises(FragmentConditioningError, match="non-negative"):
        build_prompt_context(prompt, linker_bridge_atoms=-1)


# ---- Realized linker length ----
#
# The corrected start state hands the generator a seed bridge, so a linker row
# is only meaningful beside the distribution of realized lengths. These tests
# pin the measurement against hand-built states where the answer is known by
# inspection, and against the production start states where the seed length is
# the construction's own parameter.


def _controller_for(prompt, *, bridge: int):
    context = build_prompt_context(prompt, control=_ON, linker_bridge_atoms=bridge)
    return context, AttachmentController(
        context.attachment, context.locked_slots, _ON
    )


def test_realized_length_is_undefined_for_a_single_core_prompt():
    prompt = _prompt(FragmentTask.MOTIF_EXTENSION, "BARICITINIB")
    context, controller = _controller_for(prompt, bridge=0)
    assert controller.realized_linker_length(context.start_state) is None


def test_realized_length_is_zero_for_the_direct_join():
    """The zero-atom linker the invalidation named, measured as zero."""
    for prompt in _linker_prompts():
        context, controller = _controller_for(prompt, bridge=0)
        assert controller.realized_linker_length(context.start_state) == 0, (
            f"{prompt.drug_name}: a directly-bonded pair of cores is a zero-atom linker"
        )


def test_realized_length_reports_the_seed_the_construction_supplied():
    for bridge in (1, 2, 3):
        for prompt in _linker_prompts():
            context, controller = _controller_for(prompt, bridge=bridge)
            assert controller.realized_linker_length(context.start_state) == bridge, (
                f"{prompt.drug_name}: a {bridge}-atom seed must measure as {bridge}"
            )


def test_realized_length_grows_when_the_linker_grows():
    """Extend the seed by one atom and the measurement must follow."""
    prompt = _linker_prompts()[0]
    context, controller = _controller_for(prompt, bridge=1)
    state = context.start_state
    assert controller.realized_linker_length(state) == 1
    # Find the seed atom: the one non-core atom bonded to a core atom.
    cores = frozenset().union(*context.attachment.lock_groups)
    seed = next(
        j
        for i in cores
        for j in range(len(state.bonds[i]))
        if int(state.bonds[i][j]) > 0 and j not in cores
    )
    # Splice a fresh carbon between the seed and the core it reaches on one side.
    other = context.attachment.lock_groups[1]
    anchor = next(j for j in other if int(state.bonds[seed][j]) > 0)
    grown = copy.deepcopy(state)
    slot = _first_null_slot(state)
    grown.atom_types[slot] = state.atom_types[seed]
    grown.formal_charges[slot] = 0
    grown.implicit_h_counts[slot] = 2
    grown.bonds[seed][anchor] = 0
    grown.bonds[anchor][seed] = 0
    for a, b in ((seed, slot), (slot, anchor)):
        grown.bonds[a][b] = 1
        grown.bonds[b][a] = 1
    assert controller.realized_linker_length(grown) == 2


def test_realized_length_is_none_when_no_core_to_core_path_exists():
    prompt = _linker_prompts()[0]
    context, controller = _controller_for(prompt, bridge=1)
    severed = copy.deepcopy(context.start_state)
    severed.bonds[:, :] = 0
    assert controller.realized_linker_length(severed) is None


def _synthetic_spec(groups, interfaces=()):
    """An AttachmentSpec over hand-chosen slot groups, for guards the released
    two-fragment prompts cannot reach."""
    return AttachmentSpec(
        interfaces=tuple(interfaces),
        requirements=tuple((i, 1) for i in interfaces),
        lock_groups=tuple(frozenset(g) for g in groups),
        released_pairs=frozenset(),
    )


def _chain_state(n_slots, bonds, elements=6):
    state = build_prompt_context(
        _prompt(FragmentTask.MOTIF_EXTENSION, "BARICITINIB")
    ).start_state
    blank = copy.deepcopy(state)
    blank.atom_types[:] = NULL_IDX
    blank.formal_charges[:] = 0
    blank.implicit_h_counts[:] = 0
    blank.bonds[:, :] = 0
    for i in range(n_slots):
        blank.atom_types[i] = elements
    for a, b in bonds:
        blank.bonds[a][b] = 1
        blank.bonds[b][a] = 1
    return blank


def test_realized_length_never_routes_through_a_third_core():
    """With three retained cores, a path through the middle one is not a linker.

    The released prompts declare exactly two fragments, so this guard is
    unreachable from the benchmark; it is tested directly rather than left as an
    untested branch.  Slots 0 and 4 are the two cores being measured, slot 2 is
    a third core, and the ONLY short route from 0 to 4 runs through it.
    """
    # 0-1-2-3-4 chain, with 2 a third locked core; plus a long free detour
    # 0-5-6-7-8-4 that does not touch any core.
    state = _chain_state(9, [(0, 1), (1, 2), (2, 3), (3, 4),
                             (0, 5), (5, 6), (6, 7), (7, 8), (8, 4)])
    spec = _synthetic_spec([{0}, {4}, {2}])
    controller = AttachmentController(spec, (0, 2, 4), _ON)
    assert controller.realized_linker_length(state) == 4, (
        "the 3-atom route through the third core must not be counted; the "
        "4-atom core-free route is the linker"
    )


def test_realized_length_does_not_walk_padding_slots():
    """A null slot is not an atom, even if a stale bond row points at one.

    States are slot-stable, so a padded slot can carry a bond row left by an
    earlier occupant; walking one would invent a linker atom that does not
    exist.
    """
    state = _chain_state(3, [(0, 1), (1, 2)])
    state.atom_types[1] = NULL_IDX  # the intermediate is padding, not an atom
    spec = _synthetic_spec([{0}, {2}])
    controller = AttachmentController(spec, (0, 2), _ON)
    assert controller.realized_linker_length(state) is None


# ---- One controller, every instance ----
#
# Consuming the benchmark's DECLARED constraint -- motif, attachment sites,
# locked atoms -- is what a fragment-conditioned generator is supposed to do.
# Tuning on the drug or the task label is not, and it would invalidate the
# section rather than improve it.  These guards enforce the distinction on the
# source itself, because it is the kind of claim a reader has to take on trust
# otherwise.


def test_the_controller_names_no_benchmark_instance():
    """No drug name from the released manifest may appear in the controller.

    The drug list is read from the manifest rather than written out here, so a
    newly released drug is covered the day it lands.
    """
    import compose_v4.benchmark.fragment_attachment_control as module

    source = Path(module.__file__).read_text()
    lowered = source.lower()
    offenders = sorted(
        {
            prompt.drug_name
            for prompt in _prompts()
            if prompt.drug_name.lower() in lowered
        }
    )
    assert not offenders, (
        f"the attachment controller names benchmark instances {offenders}; a "
        "mechanism must activate from the declared constraint and structural "
        "state, never from an instance identity"
    )


def test_the_sampler_names_no_benchmark_instance_either():
    """The guard has to cover the module the MECHANISM lives in.

    The two genericity checks scanned the controller only, and v3's composite
    transaction lives in the sampler, so a drug-shaped branch there would have
    passed both. The mechanism module is exactly where a per-instance rule is
    most tempting and least visible.
    """
    import compose_v4.benchmark.fragment_conditioned_sampler as module

    source = Path(module.__file__).read_text()
    lowered = source.lower()
    offenders = sorted(
        {
            prompt.drug_name
            for prompt in _prompts()
            if prompt.drug_name.lower() in lowered
        }
    )
    assert not offenders, (
        f"the sampler names benchmark instances {offenders}; a mechanism must "
        "activate from the declared constraint and structural state, never from "
        "an instance identity"
    )

    assert "drug_name" not in source, (
        "the sampler reads a drug name; routing on the instance is benchmark "
        "engineering, not a general capability"
    )

    # ``prompt.task`` is legitimate in ONE place and illegitimate everywhere
    # else, so a blanket ban would be wrong and a blanket allowance useless.
    # ``build_prompt_context`` is the ADAPTER: it turns the benchmark's own
    # declared prompt into a start state, and a two-fragment prompt genuinely
    # needs a different construction from a one-fragment prompt. The MECHANISM
    # -- the sampler loop and the composite transaction -- must read structural
    # state only. This locates the branch instead of counting it.
    import ast

    # Matched on the ATTRIBUTE NAME and on the FragmentTask symbol, never on the
    # base expression. A first version required the base to be a bare ``prompt``
    # and a mutation reading ``context.prompt.task`` inside ``sample_completion``
    # walked straight past it -- the mechanism could route on the task label and
    # the guard stayed green. One spelling is not a guard.
    tree = ast.parse(source)
    readers = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            reads_task = isinstance(inner, ast.Attribute) and inner.attr == "task"
            names_enum = isinstance(inner, ast.Name) and inner.id == "FragmentTask"
            if reads_task or names_enum:
                readers.add(node.name)
    assert readers <= {"build_prompt_context"}, (
        f"these functions route on the task label: {sorted(readers)}. Only the "
        "prompt adapter may read it; the sampler loop and the composite "
        "transaction must activate from structural state alone"
    )


def test_the_controller_does_not_branch_on_the_task_label():
    """A task-shaped branch is per-task tuning wearing a general name.

    'No declared interface' is a property of the SPECIFICATION and is already
    how superstructure deactivates the controller; reading FragmentTask would
    be a different and illegitimate thing.
    """
    import compose_v4.benchmark.fragment_attachment_control as module

    source = Path(module.__file__).read_text()
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    _, _, body = code.partition('"""')
    _, _, body = body.partition('"""')  # drop the module docstring
    for forbidden in ("FragmentTask", "prompt.task", "drug_name"):
        assert forbidden not in body, (
            f"the controller reads {forbidden!r}; routing on the task label or "
            "the drug is benchmark engineering, not a general capability"
        )


def test_one_frozen_parameter_set_covers_every_prompt():
    """The same config object must serve every released prompt."""
    config = AttachmentControlConfig(enabled=True)
    seen = set()
    for prompt in _prompts():
        bridge = 1 if len(prompt.fragments) == 2 else 0
        context = build_prompt_context(
            prompt, control=config, linker_bridge_atoms=bridge
        )
        controller = AttachmentController(
            context.attachment, context.locked_slots, config
        )
        # The tunable parameter set must not vary with the instance; only the
        # DECLARED spec may differ from prompt to prompt.
        seen.add(json.dumps(dataclasses.asdict(controller.config), sort_keys=True))
    assert len(seen) == 1, (
        f"the controller held {len(seen)} distinct parameter sets across the "
        "panel; it must hold exactly one"
    )


# ---- Two-interface path program ----
#
# v1 of this predicate is FALSIFIED (see
# diagnostics/fragment_path_program_v1_falsification.json): a per-event monotone
# path predicate is unsatisfiable, because no single legal event lengthens the
# core-to-core path. It is retained OFF by default, and these guards hold the
# properties the next version must also satisfy.


def test_path_program_is_off_by_default():
    assert AttachmentControlConfig().path_program is False
    assert AttachmentControlConfig(enabled=True).path_program is False


def test_path_program_is_vacuous_for_a_single_core_prompt():
    """Activation must come from the SPECIFICATION: two retained regions."""
    config = AttachmentControlConfig(enabled=True, path_program=True)
    for task in (FragmentTask.MOTIF_EXTENSION, FragmentTask.SCAFFOLD_DECORATION):
        prompt = _prompt(task, "BARICITINIB")
        context = build_prompt_context(prompt, control=config)
        controller = AttachmentController(
            context.attachment, context.locked_slots, config
        )
        assert not controller.path_active
        assert not controller.path_unsatisfied(context.start_state, 3)
        ok, _ = controller.path_permits(
            context.start_state, context.start_state, 3
        )
        assert ok


def test_path_program_activates_on_a_two_core_prompt():
    config = AttachmentControlConfig(enabled=True, path_program=True)
    for prompt in _linker_prompts():
        context = build_prompt_context(
            prompt, control=config, linker_bridge_atoms=1
        )
        controller = AttachmentController(
            context.attachment, context.locked_slots, config
        )
        assert controller.path_active, prompt.drug_name
        # Seed length 1 against a target of 3 is unsatisfied -- which is exactly
        # what coverage staging could NOT express, since both interfaces are
        # already covered by the seed at event 0.
        assert controller.path_unsatisfied(context.start_state, 3)
        assert not controller.path_unsatisfied(context.start_state, 1)


def test_path_target_is_drawn_from_the_declared_band():
    config = AttachmentControlConfig(
        enabled=True, path_program=True, path_length_min=2, path_length_max=5
    )
    prompt = _linker_prompts()[0]
    context = build_prompt_context(prompt, control=config, linker_bridge_atoms=1)
    controller = AttachmentController(
        context.attachment, context.locked_slots, config
    )
    rng = np.random.default_rng(20260921)
    draws = {controller.path_target(rng) for _ in range(200)}
    assert draws <= {2, 3, 4, 5}
    # It must VARY -- a target pinned per instance is per-instance tuning.
    assert len(draws) > 1


# ---- The composite path transaction ----


def test_transaction_sites_are_none_for_a_single_core_prompt():
    """There is no second core to route a path to, so there is no transaction.

    Checked on a GROWN state, not only the start state: a single-core prompt
    starts with no atoms outside its core at all, so the start state cannot
    exercise the guard -- any implementation returns None there for the wrong
    reason.
    """
    config = AttachmentControlConfig(enabled=True, path_program=True)
    prompt = _prompt(FragmentTask.MOTIF_EXTENSION, "BARICITINIB")
    context = build_prompt_context(prompt, control=config)
    controller = AttachmentController(context.attachment, context.locked_slots, config)
    assert controller.path_transaction_sites(context.start_state) is None

    # Grow a pendant off a declared interface, so a non-core atom now exists
    # adjacent to the core -- exactly the shape the two-core branch looks for.
    anchor = context.attachment.interfaces[0]
    grown = _attach_new_atom(context.start_state, anchor)
    assert any(
        int(grown.bonds[anchor][j]) > 0
        and j not in frozenset().union(*context.attachment.lock_groups)
        for j in range(grown.n_atoms)
    ), "the fixture must actually place a non-core atom beside the core"
    assert controller.path_transaction_sites(grown) is None


def test_a_saturated_anchor_no_longer_blocks_the_transaction():
    """v3 deletes v2's free-valence precondition, and this pins why.

    v2 ring-closed to the far anchor BEFORE removing the old bond, so the
    anchor had to carry two external bonds at once and a declared site offering
    one free valence could not host it -- 4 of 10 drugs. bond_reroute exchanges
    the bridge atomically, so the anchor never holds more than one bond and a
    saturated anchor is no longer an obstacle. This test asserts the OPPOSITE
    of the v2 test it replaces, because the mechanism deliberately changed.
    """
    config = AttachmentControlConfig(enabled=True, path_program=True)
    available = 0
    for prompt in _linker_prompts():
        context = build_prompt_context(
            prompt, control=config, linker_bridge_atoms=1
        )
        controller = AttachmentController(
            context.attachment, context.locked_slots, config
        )
        sites = controller.path_transaction_sites(context.start_state)
        assert sites is not None, (
            f"{prompt.drug_name}: v3 must offer the transaction on every linker "
            "prompt, with no free-valence gate"
        )
        available += 1
        _path_atom, anchor, _free = sites
        # Saturating the anchor must NOT withdraw the sites under v3.
        blocked = copy.deepcopy(context.start_state)
        blocked.implicit_h_counts[anchor] = 0
        assert controller.path_transaction_sites(blocked) is not None, (
            f"{prompt.drug_name}: an atomic bridge exchange needs no free "
            "valence at the anchor"
        )
    assert available == 10, "all ten released linker prompts must host it"


def test_an_unsatisfiable_path_predicate_does_not_block():
    """A predicate with no way to make progress must release.

    Under v3 the transaction is available on every released prompt, so the
    release is exercised with a state that genuinely offers no sites: one whose
    padding is fully consumed, leaving nowhere to insert the new atom.
    """
    config = AttachmentControlConfig(enabled=True, path_program=True)
    prompt = _linker_prompts()[0]
    context = build_prompt_context(prompt, control=config, linker_bridge_atoms=1)
    controller = AttachmentController(context.attachment, context.locked_slots, config)
    full = copy.deepcopy(context.start_state)
    # Occupy every padding slot: no free slot means no transaction is possible.
    for i in range(full.n_atoms):
        if full.atom_types[i] == NULL_IDX:
            full.atom_types[i] = 6
    assert controller.path_transaction_sites(full) is None
    assert controller.path_unsatisfied(full, 4)
    ok, reason = controller.path_permits(full, full, 4)
    assert ok, f"an unsatisfiable predicate must release, got {reason!r}"


# ---- The composite transaction itself ----
#
# The mechanism that does the work had no direct test: every v3 test above
# checks a PRECONDITION of the transaction (are the sites offered, does the
# predicate release) and none of them runs it. A mutation that drops the bridge
# exchange, or reroutes the wrong atom, or invents the payload, survived all of
# them. These drive the production ``_attempt_path_transaction`` against the
# real executor with a stub model, so the production function is the only thing
# that decides.


class _StubMark:
    def __init__(self, rule_name, action):
        self.rule_name = rule_name
        self.action = action
        self.total_hazard = 1.0


class _StubModel:
    """Offers one fixed mark. The prior's role here is to supply a PAYLOAD."""

    def __init__(self, mark):
        self._mark = mark

    def sample_rewrite_mark(self, state, time, rng):
        return self._mark


def _monovalent_payload(state, slot_hint: int):
    """An ``AtomInsert`` copying a real monovalent atom of ``state``.

    Copying an existing atom's (type, charge, hydrogens) is what keeps this
    valid in whatever vocabulary the state uses: ``atom_type`` is a VOCABULARY
    INDEX, not an atomic number, and a probe that wrote 6 for carbon was
    writing phosphorus.
    """
    from compose_v4.chem.molecular_graph import is_element
    from compose_v4.rewrite.operators import AtomInsert

    real = is_element(state.atom_types)
    for i in range(state.n_atoms):
        if not bool(real[i]):
            continue
        degree = sum(
            1 for j in range(state.n_atoms)
            if j != i and bool(real[j]) and int(state.bonds[i][j]) > 0
        )
        if degree == 1:
            return AtomInsert(
                slot=slot_hint,
                atom_type=int(state.atom_types[i]),
                formal_charge=int(state.formal_charges[i]),
                implicit_h_count=int(state.implicit_h_counts[i]),
                neighbors=((slot_hint, 1),),
            )
    return None


def _transaction_fixture(prompt, *, config=None):
    from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context

    config = config or AttachmentControlConfig(enabled=True, path_program=True)
    context = build_prompt_context(prompt, control=config, linker_bridge_atoms=1)
    controller = AttachmentController(context.attachment, context.locked_slots, config)
    lock = RegionLock(
        context.start_state,
        context.locked_slots,
        released_pairs=context.attachment.released_pairs,
    )
    return context, controller, lock


def test_the_transaction_lengthens_the_core_to_core_path():
    """The property the whole program exists for, on every released prompt."""
    from compose_v4.benchmark.fragment_conditioned_sampler import (
        SamplingReceipt,
        _attempt_path_transaction,
    )
    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    system = de_novo_rewrite_system()
    exercised = 0
    for prompt in _linker_prompts():
        context, controller, lock = _transaction_fixture(prompt)
        sites = controller.path_transaction_sites(context.start_state)
        assert sites is not None
        path_atom, _anchor, _free = sites
        payload = _monovalent_payload(context.start_state, path_atom)
        if payload is None:
            continue
        before = controller.realized_linker_length(context.start_state)
        receipt = SamplingReceipt()
        out = _attempt_path_transaction(
            _StubModel(_StubMark("atom_insert", payload)),
            system,
            context.start_state,
            controller,
            lock,
            np.random.default_rng(0),
            receipt,
        )
        if out is None:
            continue
        assert receipt.path_transaction_offers == 1
        assert receipt.path_payload_found == 1
        assert receipt.path_transactions == 1
        assert receipt.path_transaction_refusals == 0
        exercised += 1
        after = controller.realized_linker_length(out)
        assert after is not None and after > before, (
            f"{prompt.drug_name}: the transaction must lengthen the path, "
            f"{before} -> {after}"
        )
        assert controller.cores_are_separated(out)
        assert lock.permits(out)
    assert exercised >= 5, (
        f"only {exercised} released prompts exercised the transaction; the test "
        "must actually run the mechanism, not skip past it"
    )


def test_the_transaction_payload_comes_from_the_prior():
    """The prior chooses WHAT to insert; the constraint chooses WHERE.

    Both directions: the proposed atom's identity survives into the endpoint,
    and a prior offering nothing usable makes the transaction ABANDON rather
    than invent a payload of its own.
    """
    from compose_v4.benchmark.fragment_conditioned_sampler import (
        SamplingReceipt,
        _attempt_path_transaction,
    )
    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    system = de_novo_rewrite_system()
    carried = 0
    for prompt in _linker_prompts():
        context, controller, lock = _transaction_fixture(prompt)
        sites = controller.path_transaction_sites(context.start_state)
        path_atom, _anchor, free_slot = sites
        payload = _monovalent_payload(context.start_state, path_atom)
        if payload is None:
            continue
        out = _attempt_path_transaction(
            _StubModel(_StubMark("atom_insert", payload)),
            system, context.start_state, controller, lock,
            np.random.default_rng(0), SamplingReceipt(),
        )
        if out is not None:
            assert int(out.atom_types[free_slot]) == payload.atom_type, (
                f"{prompt.drug_name}: the endpoint must carry the atom the prior "
                "proposed, not one the module chose"
            )
            carried += 1

        # A prior that never offers an atom_insert must not be overridden.
        no_payload_receipt = SamplingReceipt()
        nothing = _attempt_path_transaction(
            _StubModel(_StubMark("bond_reorder", None)),
            system, context.start_state, controller, lock,
            np.random.default_rng(0), no_payload_receipt,
        )
        assert nothing is None, (
            f"{prompt.drug_name}: with no payload from the prior the transaction "
            "must be abandoned, never invented"
        )
        assert no_payload_receipt.path_transaction_offers == 1
        assert no_payload_receipt.path_payload_draws == 32
        assert no_payload_receipt.path_payload_noninsert == 32
        assert no_payload_receipt.path_payload_absent == 1

        # The prompt specifies the path, so the prior may draw its payload at
        # another site while the constraint binds that payload to the path.
        elsewhere = dataclasses.replace(payload, neighbors=((_anchor, 1),))
        assert _anchor != path_atom
        rebound_receipt = SamplingReceipt()
        rebound = _attempt_path_transaction(
            _StubModel(_StubMark("atom_insert", elsewhere)),
            system, context.start_state, controller, lock,
            np.random.default_rng(0), rebound_receipt,
        )
        if out is not None:
            assert rebound is not None
            assert int(rebound.atom_types[free_slot]) == payload.atom_type
            assert rebound_receipt.path_payload_rebound == 1
            assert rebound_receipt.path_payload_absent == 0
    assert carried >= 5


def test_rebound_double_bond_payload_preserves_valence_class():
    from compose_v4.benchmark.fragment_conditioned_sampler import (
        SamplingReceipt,
        _attempt_path_transaction,
    )
    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    system = de_novo_rewrite_system()
    exercised = 0
    for prompt in _linker_prompts():
        context, controller, lock = _transaction_fixture(prompt)
        sites = controller.path_transaction_sites(context.start_state)
        assert sites is not None
        path_atom, far_anchor, free_slot = sites
        payload = _monovalent_payload(context.start_state, path_atom)
        if payload is None or payload.implicit_h_count < 1:
            continue
        original_valence = payload.implicit_h_count + 1 - payload.formal_charge
        doubled = dataclasses.replace(
            payload,
            implicit_h_count=payload.implicit_h_count - 1,
            neighbors=((far_anchor, 2),),
        )
        receipt = SamplingReceipt()
        out = _attempt_path_transaction(
            _StubModel(_StubMark("atom_insert", doubled)),
            system, context.start_state, controller, lock,
            np.random.default_rng(0), receipt,
        )
        if out is None:
            continue
        assert receipt.path_payload_rebound == 1
        assert int(out.implicit_h_counts[free_slot]) + 2 - int(
            out.formal_charges[free_slot]
        ) == original_valence
        exercised += 1
    assert exercised >= 5


def test_the_transaction_commits_nothing_when_the_exchange_cannot_execute():
    """ATOMIC: a refused constituent leaves the caller the state it had.

    The refusal is counted, so a transaction that never fires is visible in the
    artifact instead of looking like a state that was never reached.
    """
    from compose_v4.benchmark.fragment_conditioned_sampler import (
        SamplingReceipt,
        _attempt_path_transaction,
    )
    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    system = de_novo_rewrite_system()
    prompt = _linker_prompts()[0]
    context, controller, lock = _transaction_fixture(prompt)
    sites = controller.path_transaction_sites(context.start_state)
    path_atom, _anchor, _free = sites
    payload = _monovalent_payload(context.start_state, path_atom)
    assert payload is not None
    # An unsatisfiable payload: a bond order the site cannot take.
    impossible = dataclasses.replace(payload, implicit_h_count=99)
    receipt = SamplingReceipt()
    before = copy.deepcopy(context.start_state)
    out = _attempt_path_transaction(
        _StubModel(_StubMark("atom_insert", impossible)),
        system, context.start_state, controller, lock,
        np.random.default_rng(0), receipt,
    )
    assert out is None
    assert receipt.path_transactions == 0
    assert receipt.path_payload_found == 1
    assert receipt.path_insert_refusals == 1
    assert receipt.path_transaction_refusals == 1
    assert np.array_equal(before.atom_types, context.start_state.atom_types)
    assert np.array_equal(before.bonds, context.start_state.bonds)
