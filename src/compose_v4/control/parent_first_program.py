"""Parent-first structural program construction.

WHAT THIS INVERTS
-----------------
The v1 joint-dependency jump proposer samples a plan latent from a global library and
then searches for a parent that can bind it.  A v1 role pins each operand as the
complete elemental neighbourhood of the molecule the plan was MINED from, so a
different parent essentially never carries it: over all 19,662 production
(parent x excision plan) pairs only 18.0% pass even the root support certificate, and
97.7% of the refusals are depth-0 ``no_atom_carries_a_required_operand_descriptor``.
The representation is a source-specific TEMPLATE, not a reusable structural program.

This module draws the structural action on the parent's own legal fiber instead::

    parent -> legal region ON THAT PARENT -> transformation conditional on that region

The library plan supplies only the structural INTENT -- how many parent atoms the
transformation consumes, how many it builds, and with what construction vocabulary.
It never supplies an address, a neighbourhood fingerprint or a step schedule.  Roles
are read off the REALIZED execution by the same ``action_role_supervision`` the v1
latents were mined with, so the emitted object is a program in the identical
representation and is directly comparable to them.

SHARED MECHANISM, TASK-APPROPRIATE CONTROL
------------------------------------------
Region selection is :class:`~compose_v4.control.bridge_region_law.BridgeRegionLaw` --
the same validated law the T4 region repair uses -- with a size BAND supplied by the
intent and a task-appropriate feasibility margin supplied by the caller.  T4 tilts by
its free docking gate (similarity / QED / SA / heavy); PMO has no similarity reference
and no quality gate, so it tilts by construction headroom instead.  The generative
process and the region mechanism are shared; only the margin differs.

INVARIANTS MAINTAINED (and tested)
----------------------------------
* **No shrinking.**  A parent carrying no region in the demanded size band ABSTAINS.
  Shrinking a program until it binds is exactly the ``retained_fraction == 1.000``
  additive degeneracy this experiment exists to detect, so it is refused structurally
  rather than discouraged.
* **Executor-certified.**  Every emitted program is replayed by the production
  executor; nothing is accepted on a closed form.  ``excise_region``'s closed form is
  used only to WEIGHT candidates.
* **Excision is real.**  ``ExcisionAccounting`` reports deletion of parent atoms from
  the action list, which is why it can separate "kept" from "deleted and refilled" --
  a distinction the controller's slot-based ``_retained_fraction`` cannot make.

STATE SEMANTICS -- 48 SLOTS
---------------------------
Production PMO parents carry ``n_atoms == 48`` (measured, unanimous over all 226
parents of the completed 3x250 run), with 9-47 free slots.  That capacity is what
makes ``atom_insert`` expressible: a TIGHT graph, which is what
``smiles_to_molecular_graph`` returns, has no free slot and therefore no
``atom_insert`` support at all.  Build parents with ``pad_molecular_graph(..., 48)``
or ``decode_state`` of a real production state.  ``assert_production_state_semantics``
(40-slot, editing corpus) is a DIFFERENT contract and is the wrong preflight here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from compose_v4.chem.molecular_graph import (
    ALLOWED_VALENCES,
    ELEMENT_TO_IDX,
    IDX_TO_ELEMENT,
    MolecularGraph,
    is_element,
)
from compose_v4.control.bridge_region_law import (
    BridgeRegion,
    BridgeRegionLaw,
    RegionRealizationError,
    bridge_separated_regions,
)
from compose_v4.control.dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.pmo_action_roles import action_role_supervision
from compose_v4.experiments.whole_ring_plan import execute_program, fresh_slot
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import AtomDelete, AtomInsert, CycleOpenEdge
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "pmo_parent_first_program_v1"

# The plan schema v1 latents carry, reproduced so a constructed program is comparable
# to a mined one by identity rather than by resemblance.
PLAN_SCHEMA_VERSION = "pmo_joint_role_plan_v1"

# Heavy-atom ceiling the executor enforces on endpoints.  Distinct from the 48-slot
# capacity of the state array: 48 is how many slots exist, 40 is how many may be
# occupied.  These two quantities have been confused before; say which one you mean.
REPRESENTABLE_HEAVY_ATOMS = 40

# Construction vocabulary bound when an intent's own inserted elements are
# unavailable.  Neutral, unambiguous-valence organic elements only.
DEFAULT_CONSTRUCTION_ELEMENTS = ("C", "N", "O")

# ---- abstention reasons (a refusal must be attributable) ----

ABSTAIN_NO_REGION_IN_BAND = "no_legal_region_in_demanded_size_band"
ABSTAIN_NO_EXCISION_DEMAND = "intent_demands_no_excision"
ABSTAIN_NO_DELETION_SCHEDULE = "no_region_in_band_admits_a_deletion_schedule"
ABSTAIN_NO_INSERTION_CAPACITY = "contracted_state_has_no_insertion_capacity"
ABSTAIN_NO_ANCHOR = "contracted_state_has_no_hydrogen_bearing_anchor"
ABSTAIN_CONSTRUCTION_REFUSED = "executor_refused_the_construction"
ABSTAIN_ENDPOINT_EQUALS_SOURCE = "endpoint_equals_source"
ABSTAIN_ENDPOINT_UNSUPPORTED = "endpoint_representation_unsupported"

ABSTENTION_REASONS = (
    ABSTAIN_NO_REGION_IN_BAND,
    ABSTAIN_NO_EXCISION_DEMAND,
    ABSTAIN_NO_DELETION_SCHEDULE,
    ABSTAIN_NO_INSERTION_CAPACITY,
    ABSTAIN_NO_ANCHOR,
    ABSTAIN_CONSTRUCTION_REFUSED,
    ABSTAIN_ENDPOINT_EQUALS_SOURCE,
    ABSTAIN_ENDPOINT_UNSUPPORTED,
)


# ---- the structural intent a library plan contributes ----


@dataclass(frozen=True)
class ProgramIntent:
    """What a plan latent asks for, with every parent-specific address removed.

    ``excise_atoms``   parent atoms the transformation must consume
    ``insert_atoms``   atoms it must build
    ``elements``       construction vocabulary, as element symbols
    ``source_plan_id`` provenance only; never consulted when binding
    """

    excise_atoms: int
    insert_atoms: int
    elements: tuple[str, ...]
    source_plan_id: str
    primitive_count: int

    @property
    def demands_excision(self) -> bool:
        return self.excise_atoms > 0


def intent_of_plan(plan: dict[str, Any]) -> ProgramIntent:
    """Project a v1 plan latent onto its parent-independent structural demand.

    Everything that makes the latent a template -- operand descriptors, neighbourhood
    histograms, created ordinals, creation lags, the teacher's step order -- is
    DISCARDED here.  What survives is a count of parent atoms consumed, a count of
    atoms built, and the element vocabulary of the construction, none of which
    references the molecule the plan was mined from.
    """

    excise = 0
    elements: list[str] = []
    for role in plan["roles"]:
        rule = str(role["executor_rule"])
        if rule == "atom_delete":
            descriptor = role["operands"][0]["descriptor"]
            if descriptor.get("origin") == "preexisting":
                excise += 1
        elif rule == "atom_insert":
            symbol = IDX_TO_ELEMENT[int(role["parameters"]["atom_type"])]
            if int(role["parameters"]["formal_charge"]) == 0 and len(
                ALLOWED_VALENCES.get(symbol, ())
            ) == 1:
                elements.append(symbol)
    vocabulary = tuple(sorted(set(elements))) or DEFAULT_CONSTRUCTION_ELEMENTS
    return ProgramIntent(
        excise_atoms=excise,
        insert_atoms=sum(
            1 for role in plan["roles"] if str(role["executor_rule"]) == "atom_insert"
        ),
        elements=vocabulary,
        source_plan_id=str(plan["plan_id"]),
        primitive_count=int(plan["primitive_count"]),
    )


# ---- region selection inside a demanded size band ----


@dataclass(frozen=True)
class SizeBandRegionLaw(BridgeRegionLaw):
    """A :class:`BridgeRegionLaw` restricted to a closed size band.

    The band is the mechanism that refuses shrinking.  ``BridgeRegionLaw`` carries
    only an upper bound, because T4's defect was a cap that made large regions
    UNDRAWABLE; here the intent additionally demands a floor, so a parent whose
    largest region is smaller than the demand abstains instead of quietly binding a
    smaller transformation.  Weighting, the support floor and the draw order are
    inherited unchanged.
    """

    minimum: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.minimum < 1:
            raise ValueError("region size floor must be at least one atom")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("region size band is empty")

    def regions(self, graph: MolecularGraph) -> tuple[BridgeRegion, ...]:
        return tuple(
            region
            for region in bridge_separated_regions(graph, maximum=self.maximum)
            if region.size >= self.minimum
        )


def construction_headroom_margin(
    *, insert_atoms: int, ceiling: int = REPRESENTABLE_HEAVY_ATOMS
) -> Callable[[MolecularGraph], float]:
    """PMO's free feasibility margin: does the contracted state admit the build?

    PMO declares no similarity reference and no quality gate -- its endpoint
    eligibility is RDKit parseability -- so the only cost-free structural constraint
    is whether the contracted state has room for the construction the intent demands
    and an anchor to attach it to.  Normalized by the ceiling so the scale matches
    T4's gate units, and negative when the build cannot fit.
    """

    def margin(child: MolecularGraph) -> float:
        heavy = int(is_element(child.atom_types).sum())
        if heavy < 1:
            return -1.0
        free_slots = int(child.n_atoms) - heavy
        anchors = sum(
            1
            for slot in np.flatnonzero(is_element(child.atom_types))
            if int(child.implicit_h_counts[int(slot)]) >= 1
        )
        if anchors < 1:
            return -1.0
        room = min(ceiling - heavy, free_slots) - int(insert_atoms)
        return float(room) / float(ceiling)

    return margin


# ---- exact excision + construction on the drawn region ----


def _deletion_schedule(source: MolecularGraph, region: BridgeRegion, rng) -> tuple[list, MolecularGraph]:
    """Executor-verified action sequence deleting exactly ``region``.

    Ring bonds INSIDE the fragment are opened first, then atoms are removed in leaf
    order.  This mirrors ``dynamic_program_synthesis._delete_pendant_fragment``'s
    schedule, which the T4 region tests drive against the live function; the endpoint
    is independently cross-checked against ``excise_region``'s closed form by
    ``tests/test_parent_first_program.py``.
    """

    current = source
    actions: list = []
    remaining = {int(slot) for slot in region.fragment}
    while True:
        internal = [
            CycleOpenEdge(a, b)
            for a in sorted(remaining)
            for b in sorted(remaining)
            if a < b and current.bonds[a, b]
        ]
        opened = False
        for raw in rng.permutation(len(internal)):
            record = encode_action("cycle_open", internal[int(raw)])
            try:
                following, _ = execute_program(current, [record])
            except ValueError:
                continue
            current, opened = following, True
            actions.append(record)
            break
        if not opened:
            break
    while remaining:
        leaves = [
            slot for slot in sorted(remaining) if int(np.count_nonzero(current.bonds[slot])) == 1
        ]
        if not leaves:
            raise RegionRealizationError("region does not admit a leaf deletion order")
        slot = leaves[int(rng.integers(len(leaves)))]
        record = encode_action("atom_delete", AtomDelete(slot))
        current, _ = execute_program(current, [record])
        actions.append(record)
        remaining.remove(slot)
    return actions, current


def _construction_schedule(
    contracted: MolecularGraph,
    anchor: int,
    rng,
    *,
    length: int,
    elements: Sequence[str],
) -> tuple[list, MolecularGraph]:
    """Executor-verified chain construction of ``length`` atoms from ``anchor``."""

    current = contracted
    actions: list = []
    at = int(anchor)
    for _ in range(length):
        symbol = elements[int(rng.integers(len(elements)))]
        valences = ALLOWED_VALENCES[symbol]
        if len(valences) != 1:
            raise RegionRealizationError("construction requires an unambiguous valence")
        action = AtomInsert(
            fresh_slot(current), ELEMENT_TO_IDX[symbol], 0, valences[0] - 1, ((at, 1),)
        )
        record = encode_action("atom_insert", action)
        current, _ = execute_program(current, [record])
        actions.append(record)
        at = action.slot
    return actions, current


# ---- excision accounting: "kept" vs "deleted and refilled" ----


@dataclass(frozen=True)
class ExcisionAccounting:
    """Retention measured two ways, because the controller's rule cannot separate them.

    ``controller_retained_fraction`` is ``pmo_population_controller._retained_fraction``
    verbatim: the fraction of the parent's original SLOTS occupied in the endpoint.  A
    slot deleted and then refilled by a later insertion counts as retained under that
    rule, so a program that excised a whole substituent can still read 1.000.

    ``true_retained_fraction`` counts a parent atom as retained only if no
    ``atom_delete`` in the program consumed it.  ``refilled_slots`` is exactly the
    population the two measures disagree about, so quoting both is what makes an
    excision claim checkable.
    """

    parent_heavy_atoms: int
    deleted_parent_slots: tuple[int, ...]
    refilled_slots: tuple[int, ...]
    controller_retained_fraction: float
    true_retained_fraction: float

    @property
    def measures_disagree(self) -> bool:
        return bool(self.refilled_slots)


def excision_accounting(
    source: MolecularGraph, endpoint: MolecularGraph, actions: Sequence[dict[str, Any]]
) -> ExcisionAccounting:
    """Separate retained parent atoms from deleted-then-refilled slots."""

    original = {int(slot) for slot in np.flatnonzero(is_element(source.atom_types))}
    deleted: set[int] = set()
    for record in actions:
        if str(record["executor_rule"]) != "atom_delete":
            continue
        slot = int(record["payload"]["v"])
        if slot in original:
            deleted.add(slot)
    occupied = is_element(endpoint.atom_types)
    refilled = tuple(sorted(slot for slot in deleted if bool(occupied[slot])))
    controller = (
        float(sum(1 for slot in original if bool(occupied[slot]))) / len(original)
        if original
        else 0.0
    )
    true_retained = (
        float(len(original) - len(deleted)) / len(original) if original else 0.0
    )
    return ExcisionAccounting(
        parent_heavy_atoms=len(original),
        deleted_parent_slots=tuple(sorted(deleted)),
        refilled_slots=refilled,
        controller_retained_fraction=controller,
        true_retained_fraction=true_retained,
    )


# ---- role extraction: the realized execution, in the v1 plan representation ----


def roles_of_execution(
    source: MolecularGraph, actions: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Roles of a realized action sequence, by the PINNED supervision function.

    Identical to ``pmo_joint_dependency_jump._generic_role_sequence`` -- the function
    the v1 library itself was mined with -- so a constructed program and a mined latent
    are the same kind of object and may be compared without a shim.
    """

    _, receipt = execute_program(source, list(actions))
    states = list(receipt["states"])
    created: dict[int, tuple[int, int]] = {}
    next_ordinal = 0
    roles: list[dict[str, Any]] = []
    for step, (state, record) in enumerate(zip(states[:-1], actions, strict=True)):
        role, next_ordinal = action_role_supervision(
            decode_state(state), record, created, step, next_ordinal
        )
        roles.append(role)
    return roles


def plan_of_execution(
    source: MolecularGraph, actions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """The realized program as a v1-shaped plan latent, with declared conditions set.

    ``component_count`` and ``created_dependency_count`` are taken from the SAME
    ``dependency_region_program`` extraction ``pmo_realization.finalize`` performs, so
    the emitted plan's declared completion conditions are the ones its own realization
    satisfies rather than a guess that would be rejected downstream.
    """

    _, receipt = execute_program(source, list(actions))
    states = list(receipt["states"])
    roles = roles_of_execution(source, actions)
    program = dependency_region_program(
        states,
        list(actions),
        config=DependencyRegionConfig(
            runtime_maximum_primitives=max(len(actions), 1),
            runtime_maximum_components=max(len(actions), 1),
        ),
    )
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "roles": roles,
        "primitive_count": len(roles),
        "component_count": int(program["component_count"]),
        "created_dependency_count": sum(
            len(role["created_handle_dependencies"]) for role in roles
        ),
        "exact_replay": bool(program["exact_replay"]),
        "complete_representation_supported": bool(
            program["complete_representation_supported"]
        ),
    }
    plan["plan_id"] = identity({"schema_version": PLAN_SCHEMA_VERSION, "roles": roles})
    return plan


# ---- the constructor ----


def construct_parent_first_program(
    source: MolecularGraph,
    intent: ProgramIntent,
    rng,
    *,
    law: BridgeRegionLaw | None = None,
    size_tolerance: int = 0,
) -> dict[str, Any]:
    """One parent-first structural program, or an attributed abstention.

    The intent's excision demand fixes a size BAND; the law draws a region inside it;
    the executor performs the excision and the construction.  The program is NEVER
    reduced to fit: if no region in the band admits a deletion schedule, or the
    contracted state cannot carry the construction, the result is an abstention with
    a reason, not a smaller transformation.

    ``size_tolerance`` widens the band symmetrically; it is a declared parameter of the
    arm, reported alongside the result, never adjusted per parent.
    """

    if not intent.demands_excision:
        return {"realized": False, "reason": ABSTAIN_NO_EXCISION_DEMAND}
    low = max(1, intent.excise_atoms - int(size_tolerance))
    high = intent.excise_atoms + int(size_tolerance)
    if law is None:
        law = SizeBandRegionLaw(
            minimum=low,
            maximum=high,
            margin=construction_headroom_margin(insert_atoms=intent.insert_atoms),
        )
    elif isinstance(law, SizeBandRegionLaw):
        law = replace(law, minimum=low, maximum=high)
    else:
        # A plain BridgeRegionLaw carries no floor; adopt its weighting and impose the
        # band the intent demands, so a caller may supply an alternative margin
        # (including ``margin=None`` for the uniform control) without also having to
        # know about the band.
        law = SizeBandRegionLaw(
            minimum=low,
            maximum=high,
            margin=law.margin,
            floor=law.floor,
            temperature=law.temperature,
        )
    ordered = law.order(source, rng)
    if not ordered:
        return {
            "realized": False,
            "reason": ABSTAIN_NO_REGION_IN_BAND,
            "demanded_band": [low, high],
        }
    failures: Counter = Counter()
    for region in ordered:
        try:
            delete_actions, contracted = _deletion_schedule(source, region, rng)
        except (RegionRealizationError, ValueError) as error:
            failures[str(error)] += 1
            continue
        heavy = int(is_element(contracted.atom_types).sum())
        free_slots = int(contracted.n_atoms) - heavy
        room = min(REPRESENTABLE_HEAVY_ATOMS - heavy, free_slots)
        if intent.insert_atoms > room:
            failures[ABSTAIN_NO_INSERTION_CAPACITY] += 1
            continue
        anchor = int(region.anchor)
        if intent.insert_atoms and int(contracted.implicit_h_counts[anchor]) < 1:
            failures[ABSTAIN_NO_ANCHOR] += 1
            continue
        try:
            grow_actions, endpoint = _construction_schedule(
                contracted, anchor, rng, length=intent.insert_atoms, elements=intent.elements
            )
        except (RegionRealizationError, ValueError) as error:
            failures[str(error)] += 1
            continue
        actions = [*delete_actions, *grow_actions]
        if canonical_state_key(endpoint) == canonical_state_key(source):
            failures[ABSTAIN_ENDPOINT_EQUALS_SOURCE] += 1
            continue
        plan = plan_of_execution(source, actions)
        if not (plan["exact_replay"] and plan["complete_representation_supported"]):
            failures[ABSTAIN_ENDPOINT_UNSUPPORTED] += 1
            continue
        accounting = excision_accounting(source, endpoint, actions)
        return {
            "realized": True,
            "reason": None,
            "plan": plan,
            "plan_id": plan["plan_id"],
            "actions": actions,
            "states": list(execute_program(source, actions)[1]["states"]),
            "endpoint_state": encode_state(endpoint),
            "endpoint_key": canonical_state_key(endpoint),
            "primitive_count": len(actions),
            "region_size": int(region.size),
            "region_anchor": anchor,
            "demanded_band": [low, high],
            "intent": {
                "excise_atoms": intent.excise_atoms,
                "insert_atoms": intent.insert_atoms,
                "elements": list(intent.elements),
                "source_plan_id": intent.source_plan_id,
                "primitive_count": intent.primitive_count,
            },
            "delta_heavy_atoms": int(endpoint.n_real_atoms) - int(source.n_real_atoms),
            "retained_fraction": accounting.controller_retained_fraction,
            "true_retained_fraction": accounting.true_retained_fraction,
            "deleted_parent_slots": list(accounting.deleted_parent_slots),
            "refilled_slots": list(accounting.refilled_slots),
            "retention_measures_disagree": accounting.measures_disagree,
            "regions_tried": int(sum(failures.values())) + 1,
        }
    # Attribute the abstention to the failure that actually dominated, not to a
    # single undifferentiated "no schedule" string: a band whose regions all lacked
    # INSERTION CAPACITY is a different finding from one whose regions could not be
    # deleted, and collapsing them is the reporting defect this experiment exists to
    # avoid repeating.
    reason = failures.most_common(1)[0][0] if failures else ABSTAIN_NO_DELETION_SCHEDULE
    if reason not in ABSTENTION_REASONS:
        reason = ABSTAIN_NO_DELETION_SCHEDULE
    return {
        "realized": False,
        "reason": reason,
        "demanded_band": [low, high],
        "regions_tried": len(ordered),
        "failures": dict(failures.most_common()),
    }
