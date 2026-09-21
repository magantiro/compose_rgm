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

import numpy as np
import pytest
from rdkit import Chem

from compose_v4.benchmark.fragment_attachment_control import (
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
from pathlib import Path

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
