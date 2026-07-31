"""Prospective alternate-Kekule invariance diagnostic for primitive cycle opening.

This module is deliberately separate from the frozen E5 quotient contract.  E5
qualifies persistent-slot relabeling and within-successor-fiber mark refinement;
it does not claim that distinct exact Kekule states representing one aromatic
molecule induce the same cycle-opening law.

The diagnostic fails in the scientifically useful order:

1. construct one narrowly validated alternate Kekule encoding;
2. compare model-independent executable ``bond_delete`` successor support;
3. only when support agrees, optionally compare scored ``cycle_attach`` laws.

An executor-level support mismatch is therefore recorded as a non-authorizing
negative result.  It is never hidden by skipping a fixture or by tuning a neural
scorer against an impossible target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
    MolecularGraph,
)
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.experiments.production_successor_kernel import (
    enumerate_factorized_marked_law,
)
from compose_v4.experiments.quotient_invariance import permute_persistent_slots
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    RewriteSystem,
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.operators import BondDelete

SCHEMA = "compose.diagnostics.cycle_open_alternate_kekule_invariance"
SCHEMA_VERSION = 1
ALLOWED_SCORER_MODES = frozenset(
    {
        "pair_linear",
        "exact_bond_contextual_probe",
    }
)


class CycleOpenKekuleInvarianceError(ValueError):
    """The requested bounded diagnostic is malformed or outside its scope."""


@dataclass(frozen=True)
class AlternatingKekulePair:
    """Two exact states differing only by one isolated six-cycle matching."""

    original: MolecularGraph
    alternate: MolecularGraph
    component_slots: tuple[int, ...]
    component_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CycleOpenLaw:
    """A productive family-conditional law over canonical molecular successors."""

    probabilities: tuple[tuple[str, float], ...]
    alias_counts: tuple[tuple[str, int], ...]
    raw_mark_count: int
    productive_mark_count: int
    virtual_mark_count: int
    virtual_mass: float

    @property
    def support(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.probabilities)

    def probability_map(self) -> dict[str, float]:
        return dict(self.probabilities)

    def alias_map(self) -> dict[str, int]:
        return dict(self.alias_counts)

    def to_payload(self) -> dict[str, object]:
        return {
            "probabilities": dict(self.probabilities),
            "alias_counts": dict(self.alias_counts),
            "raw_mark_count": self.raw_mark_count,
            "productive_mark_count": self.productive_mark_count,
            "virtual_mark_count": self.virtual_mark_count,
            "virtual_mass": self.virtual_mass,
        }


def _aromatic_component(
    state: MolecularGraph,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    perceived = resonance_invariant_bond_classes(state)
    aromatic_edges = tuple(
        (int(left), int(right))
        for left in range(state.n_atoms)
        for right in range(left + 1, state.n_atoms)
        if int(perceived[left, right]) == BOND_AROMATIC
    )
    graph = nx.Graph()
    graph.add_edges_from(aromatic_edges)
    components = tuple(nx.connected_components(graph))
    if len(components) != 1:
        raise CycleOpenKekuleInvarianceError(
            "alternate-Kekule construction requires exactly one aromatic component"
        )
    members = tuple(sorted(int(slot) for slot in components[0]))
    member_set = set(members)
    edges = tuple(
        edge for edge in aromatic_edges if edge[0] in member_set and edge[1] in member_set
    )
    if len(members) != 6 or len(edges) != 6:
        raise CycleOpenKekuleInvarianceError(
            "alternate-Kekule construction is limited to one isolated six-membered cycle"
        )
    degree = {slot: 0 for slot in members}
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    if set(degree.values()) != {2}:
        raise CycleOpenKekuleInvarianceError(
            "aromatic component is fused, bridged, or otherwise not an isolated cycle"
        )
    return members, tuple(sorted(edges))


def build_alternate_kekule_pair(state: MolecularGraph) -> AlternatingKekulePair:
    """Flip one validated isolated alternating aromatic six-cycle in place.

    The persistent slots, atom states, charges, and hydrogen counts are held
    fixed.  This helper intentionally rejects fused systems and general
    resonance enumeration; those require a separately specified canonical
    matching algorithm rather than a convenient but unsafe blanket flip.
    """

    if not is_valid_state(state) or not is_connected_or_null(state):
        raise CycleOpenKekuleInvarianceError(
            "alternate-Kekule construction requires a valid connected source"
        )
    members, edges = _aromatic_component(state)
    incident_orders = {slot: [] for slot in members}
    for left, right in edges:
        order = int(state.bonds[left, right])
        if order not in {BOND_SINGLE, BOND_DOUBLE}:
            raise CycleOpenKekuleInvarianceError(
                "aromatic component is not stored as exact single/double Kekule bonds"
            )
        incident_orders[left].append(order)
        incident_orders[right].append(order)
    if any(sorted(orders) != [BOND_SINGLE, BOND_DOUBLE] for orders in incident_orders.values()):
        raise CycleOpenKekuleInvarianceError(
            "six-membered aromatic component is not an alternating Kekule matching"
        )

    bonds = state.bonds.copy()
    for left, right in edges:
        changed = BOND_DOUBLE if int(state.bonds[left, right]) == BOND_SINGLE else BOND_SINGLE
        bonds[left, right] = bonds[right, left] = changed
    alternate = MolecularGraph(
        atom_types=state.atom_types.copy(),
        formal_charges=state.formal_charges.copy(),
        implicit_h_counts=state.implicit_h_counts.copy(),
        bonds=bonds,
    )
    if not is_valid_state(alternate) or not is_connected_or_null(alternate):
        raise CycleOpenKekuleInvarianceError(
            "flipped exact encoding is not a valid connected molecular state"
        )
    if canonical_state_key(state) != canonical_state_key(alternate):
        raise CycleOpenKekuleInvarianceError(
            "flipped exact encoding changed canonical molecular identity"
        )
    if not np.array_equal(
        resonance_invariant_bond_classes(state),
        resonance_invariant_bond_classes(alternate),
    ):
        raise CycleOpenKekuleInvarianceError(
            "flipped exact encoding changed the resonance-invariant neural view"
        )
    if persistent_slot_state_sha256(state) == persistent_slot_state_sha256(alternate):
        raise CycleOpenKekuleInvarianceError(
            "alternate-Kekule construction did not change exact state identity"
        )
    return AlternatingKekulePair(
        original=state,
        alternate=alternate,
        component_slots=members,
        component_edges=edges,
    )


def _validated_permutation(permutation: Sequence[int], n_slots: int) -> np.ndarray:
    order = np.asarray(tuple(int(value) for value in permutation), dtype=np.int64)
    if order.shape != (n_slots,) or set(order.tolist()) != set(range(n_slots)):
        raise CycleOpenKekuleInvarianceError(
            "slot transform must be a bijection over every persistent slot"
        )
    return order


def transport_bond_delete(
    action: BondDelete,
    permutation: Sequence[int],
    *,
    n_slots: int,
) -> BondDelete:
    """Conjugate an undirected deletion under ``new[i] = old[permutation[i]]``."""

    order = _validated_permutation(permutation, n_slots)
    inverse = np.argsort(order)
    endpoints = sorted((int(inverse[int(action.a)]), int(inverse[int(action.b)])))
    return BondDelete(*endpoints)


def transport_component_slots(
    component_slots: Sequence[int],
    permutation: Sequence[int],
    *,
    n_slots: int,
) -> tuple[int, ...]:
    """Transport a source-slot subset under the repository permutation convention."""

    order = _validated_permutation(permutation, n_slots)
    inverse = np.argsort(order)
    return tuple(sorted(int(inverse[int(slot)]) for slot in component_slots))


def _component_cycle_edges(
    state: MolecularGraph,
    component_slots: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    members = {int(slot) for slot in component_slots}
    perceived = resonance_invariant_bond_classes(state)
    return tuple(
        (left, right)
        for left in sorted(members)
        for right in sorted(members)
        if left < right and int(perceived[left, right]) == BOND_AROMATIC
    )


def executable_cycle_open_actions(
    state: MolecularGraph,
    component_slots: Sequence[int],
    *,
    system: RewriteSystem | None = None,
) -> tuple[BondDelete, ...]:
    """Enumerate executable charge-preserving primitive opens on the selected cycle."""

    runtime = system or de_novo_rewrite_system()
    actions: list[BondDelete] = []
    for left, right in _component_cycle_edges(state, component_slots):
        # The production Active8 mask protects every charged bond endpoint.
        if int(state.formal_charges[left]) != 0 or int(state.formal_charges[right]) != 0:
            continue
        action = BondDelete(left, right)
        try:
            runtime.apply(state, "bond_delete", action)
        except InvalidRewrite:
            continue
        actions.append(action)
    if not actions:
        raise CycleOpenKekuleInvarianceError(
            "selected aromatic component has no executable primitive cycle opening"
        )
    return tuple(sorted(actions, key=lambda item: (int(item.a), int(item.b))))


def _law_from_marked_rates(
    state: MolecularGraph,
    marked_rates: Sequence[tuple[str, BondDelete, float]],
    *,
    system: RewriteSystem,
) -> CycleOpenLaw:
    source_key = canonical_state_key(state)
    aggregated = system.aggregate_successor_rates(state, marked_rates)
    productive = {key: value for key, value in aggregated.items() if key != source_key}
    virtual = aggregated.get(source_key)
    total = float(sum(value.rate for value in productive.values()))
    if not np.isfinite(total) or total <= 0.0:
        raise CycleOpenKekuleInvarianceError(
            "cycle-open marked rates contain no finite positive productive mass"
        )
    return CycleOpenLaw(
        probabilities=tuple(
            (key, float(productive[key].rate) / total) for key in sorted(productive)
        ),
        alias_counts=tuple((key, len(productive[key].provenance)) for key in sorted(productive)),
        raw_mark_count=len(marked_rates),
        productive_mark_count=sum(len(productive[key].provenance) for key in productive),
        virtual_mark_count=(0 if virtual is None else len(virtual.provenance)),
        virtual_mass=(0.0 if virtual is None else float(virtual.rate)),
    )


def unit_mass_cycle_open_law(
    state: MolecularGraph,
    component_slots: Sequence[int],
    *,
    system: RewriteSystem | None = None,
) -> CycleOpenLaw:
    """Push uniform executable marks through the production executor and key."""

    runtime = system or de_novo_rewrite_system()
    actions = executable_cycle_open_actions(state, component_slots, system=runtime)
    return _law_from_marked_rates(
        state,
        tuple(("bond_delete", action, 1.0) for action in actions),
        system=runtime,
    )


def family_conditional_cycle_open_law(
    model: Any,
    state: MolecularGraph,
    *,
    time: float,
    system: RewriteSystem | None = None,
) -> CycleOpenLaw:
    """Push the scored cycle-open family through the executor, conditional on that family."""

    mode = str(getattr(model, "cycle_open_scorer_mode", ""))
    if mode not in ALLOWED_SCORER_MODES:
        raise CycleOpenKekuleInvarianceError(
            f"diagnostic supports only scorer modes {sorted(ALLOWED_SCORER_MODES)}, got {mode!r}"
        )
    runtime = system or de_novo_rewrite_system()
    law = enumerate_factorized_marked_law(model, state, float(time))
    selected = tuple(mark for mark in law.marks if mark.family_name == "cycle_attach")
    if not selected:
        raise CycleOpenKekuleInvarianceError(
            "scored marked law contains no cycle-open family marks"
        )
    return _law_from_marked_rates(
        state,
        tuple((mark.executor_rule_name, mark.action, mark.probability) for mark in selected),
        system=runtime,
    )


def compare_cycle_open_laws(
    left: CycleOpenLaw,
    right: CycleOpenLaw,
    *,
    tolerance: float,
) -> dict[str, object]:
    """Compare support, alias structure, and pushed-forward probabilities."""

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise CycleOpenKekuleInvarianceError("law tolerance must be finite and nonnegative")
    left_probabilities = left.probability_map()
    right_probabilities = right.probability_map()
    keys = set(left_probabilities) | set(right_probabilities)
    maximum = max(
        (abs(left_probabilities.get(key, 0.0) - right_probabilities.get(key, 0.0)) for key in keys),
        default=0.0,
    )
    l1 = sum(
        abs(left_probabilities.get(key, 0.0) - right_probabilities.get(key, 0.0)) for key in keys
    )
    support_identical = set(left_probabilities) == set(right_probabilities)
    aliases_identical = left.alias_map() == right.alias_map()
    return {
        "support_identical": support_identical,
        "alias_counts_identical": aliases_identical,
        "maximum_probability_residual": float(maximum),
        "l1_probability_residual": float(l1),
        "passes_probability_tolerance": bool(support_identical and maximum <= tolerance),
        "passes_strict_fiber_check": bool(
            support_identical and aliases_identical and maximum <= tolerance
        ),
        "left_only_successors": sorted(set(left_probabilities) - set(right_probabilities)),
        "right_only_successors": sorted(set(right_probabilities) - set(left_probabilities)),
    }


def _action_transport_check(
    state: MolecularGraph,
    component_slots: Sequence[int],
    permutation: Sequence[int],
    *,
    system: RewriteSystem,
) -> dict[str, object]:
    relabeled = permute_persistent_slots(state, permutation)
    changed_component = transport_component_slots(
        component_slots,
        permutation,
        n_slots=state.n_atoms,
    )
    source_actions = executable_cycle_open_actions(state, component_slots, system=system)
    expected = {
        transport_bond_delete(action, permutation, n_slots=state.n_atoms)
        for action in source_actions
    }
    observed = set(executable_cycle_open_actions(relabeled, changed_component, system=system))
    successor_mismatches = []
    for action in source_actions:
        transported = transport_bond_delete(action, permutation, n_slots=state.n_atoms)
        left_key = canonical_state_key(system.apply(state, "bond_delete", action))
        right_key = canonical_state_key(system.apply(relabeled, "bond_delete", transported))
        if left_key != right_key:
            successor_mismatches.append(
                {
                    "source_action": [int(action.a), int(action.b)],
                    "transported_action": [int(transported.a), int(transported.b)],
                    "source_successor": left_key,
                    "transported_successor": right_key,
                }
            )
    return {
        "action_support_identical_under_transport": expected == observed,
        "source_action_count": len(source_actions),
        "transported_action_count": len(observed),
        "successor_mismatches": successor_mismatches,
        "passes": bool(expected == observed and not successor_mismatches),
    }


def _default_permutation(n_slots: int) -> tuple[int, ...]:
    if n_slots < 2:
        raise CycleOpenKekuleInvarianceError(
            "slot-equivariance diagnostic requires at least two persistent slots"
        )
    return tuple(int(value) for value in np.roll(np.arange(n_slots), 1))


def run_cycle_open_kekule_invariance_diagnostic(
    state: MolecularGraph,
    *,
    permutation: Sequence[int] | None = None,
    models: Mapping[str, Any] | None = None,
    time: float = 0.37,
    tolerance: float = 2e-6,
    system: RewriteSystem | None = None,
) -> dict[str, object]:
    """Run the fail-closed bounded diagnostic and return non-authorizing evidence."""

    if not np.isfinite(time):
        raise CycleOpenKekuleInvarianceError("scoring time must be finite")
    runtime = system or de_novo_rewrite_system()
    pair = build_alternate_kekule_pair(state)
    order = tuple(permutation if permutation is not None else _default_permutation(state.n_atoms))
    _validated_permutation(order, state.n_atoms)
    if tuple(order) == tuple(range(state.n_atoms)):
        raise CycleOpenKekuleInvarianceError("slot diagnostic requires a nonidentity permutation")

    relabeled_original = permute_persistent_slots(pair.original, order)
    relabeled_alternate = permute_persistent_slots(pair.alternate, order)
    relabeled_component = transport_component_slots(
        pair.component_slots,
        order,
        n_slots=state.n_atoms,
    )
    states = {
        "original": (pair.original, pair.component_slots),
        "alternate": (pair.alternate, pair.component_slots),
        "slot_original": (relabeled_original, relabeled_component),
        "slot_alternate": (relabeled_alternate, relabeled_component),
    }
    unit_laws = {
        name: unit_mass_cycle_open_law(item, component, system=runtime)
        for name, (item, component) in states.items()
    }
    comparisons = {
        "original_vs_alternate": compare_cycle_open_laws(
            unit_laws["original"], unit_laws["alternate"], tolerance=tolerance
        ),
        "original_vs_slot_original": compare_cycle_open_laws(
            unit_laws["original"], unit_laws["slot_original"], tolerance=tolerance
        ),
        "alternate_vs_slot_alternate": compare_cycle_open_laws(
            unit_laws["alternate"], unit_laws["slot_alternate"], tolerance=tolerance
        ),
        "original_vs_combined_alternate_slot": compare_cycle_open_laws(
            unit_laws["original"], unit_laws["slot_alternate"], tolerance=tolerance
        ),
    }
    slot_transport = {
        "original": _action_transport_check(
            pair.original,
            pair.component_slots,
            order,
            system=runtime,
        ),
        "alternate": _action_transport_check(
            pair.alternate,
            pair.component_slots,
            order,
            system=runtime,
        ),
    }
    slot_gate_passed = bool(
        comparisons["original_vs_slot_original"]["passes_strict_fiber_check"]
        and comparisons["alternate_vs_slot_alternate"]["passes_strict_fiber_check"]
        and slot_transport["original"]["passes"]
        and slot_transport["alternate"]["passes"]
    )
    kekule_support_passed = bool(
        comparisons["original_vs_alternate"]["support_identical"]
        and comparisons["original_vs_alternate"]["alias_counts_identical"]
    )
    unit_law_passed = bool(
        comparisons["original_vs_alternate"]["passes_strict_fiber_check"]
        and comparisons["original_vs_combined_alternate_slot"]["passes_strict_fiber_check"]
    )

    requested_models = dict(models or {})
    model_modes = {
        name: str(getattr(model, "cycle_open_scorer_mode", ""))
        for name, model in requested_models.items()
    }
    bad_modes = {
        name: mode for name, mode in model_modes.items() if mode not in ALLOWED_SCORER_MODES
    }
    if bad_modes:
        raise CycleOpenKekuleInvarianceError(f"unsupported scorer modes requested: {bad_modes}")

    scored_payload: dict[str, object]
    scored_gate_passed: bool | None
    if not requested_models:
        scored_payload = {
            "status": "NOT_REQUESTED",
            "was_run": False,
            "reason": "no prospective scorer models were supplied",
            "models": {},
        }
        scored_gate_passed = None
    elif not kekule_support_passed or not slot_gate_passed:
        scored_payload = {
            "status": "BLOCKED_BY_MODEL_INDEPENDENT_SUPPORT_OR_SLOT_MISMATCH",
            "was_run": False,
            "reason": (
                "neural scoring cannot repair unequal executable canonical-successor support"
            ),
            "requested_modes": model_modes,
            "models": {},
        }
        scored_gate_passed = False
    else:
        scored_models = {}
        score_passes = []
        for name, model in requested_models.items():
            laws = {
                state_name: family_conditional_cycle_open_law(
                    model,
                    item,
                    time=float(time),
                    system=runtime,
                )
                for state_name, (item, _component) in states.items()
            }
            model_comparisons = {
                "original_vs_alternate": compare_cycle_open_laws(
                    laws["original"], laws["alternate"], tolerance=tolerance
                ),
                "original_vs_slot_original": compare_cycle_open_laws(
                    laws["original"], laws["slot_original"], tolerance=tolerance
                ),
                "alternate_vs_slot_alternate": compare_cycle_open_laws(
                    laws["alternate"], laws["slot_alternate"], tolerance=tolerance
                ),
                "original_vs_combined_alternate_slot": compare_cycle_open_laws(
                    laws["original"], laws["slot_alternate"], tolerance=tolerance
                ),
            }
            model_passed = all(
                result["passes_probability_tolerance"] for result in model_comparisons.values()
            )
            score_passes.append(bool(model_passed))
            scored_models[name] = {
                "scorer_mode": model_modes[name],
                "laws": {key: value.to_payload() for key, value in laws.items()},
                "comparisons": model_comparisons,
                "passes": bool(model_passed),
            }
        scored_gate_passed = all(score_passes)
        scored_payload = {
            "status": "PASS" if scored_gate_passed else "NO_GO_SCORED_LAW_MISMATCH",
            "was_run": True,
            "models": scored_models,
        }

    if not kekule_support_passed:
        status = "NO_GO_ALTERNATE_KEKULE_SUPPORT_MISMATCH"
    elif not slot_gate_passed:
        status = "NO_GO_SLOT_EQUIVARIANCE_MISMATCH"
    elif not unit_law_passed:
        status = "NO_GO_ALTERNATE_KEKULE_UNIT_LAW_MISMATCH"
    elif scored_gate_passed is False:
        status = "NO_GO_SCORED_LAW_MISMATCH"
    elif scored_gate_passed is True:
        status = "BOUNDED_PASS_NON_AUTHORIZING"
    else:
        status = "SUPPORT_PREFLIGHT_PASS_SCORING_NOT_REQUESTED"

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "scope": "bounded_development_diagnostic_isolated_alternating_six_cycle",
        "paper_claim_authorized": False,
        "training_authorized": False,
        "scorer_promotion_authorized": False,
        "frozen_e5_contract_modified": False,
        "source_evidence": {
            "canonical_source_key": canonical_state_key(pair.original),
            "alternate_canonical_source_key": canonical_state_key(pair.alternate),
            "canonical_source_identity_equal": (
                canonical_state_key(pair.original) == canonical_state_key(pair.alternate)
            ),
            "neural_bond_view_equal": bool(
                np.array_equal(
                    resonance_invariant_bond_classes(pair.original),
                    resonance_invariant_bond_classes(pair.alternate),
                )
            ),
            "original_exact_state_sha256": persistent_slot_state_sha256(pair.original),
            "alternate_exact_state_sha256": persistent_slot_state_sha256(pair.alternate),
            "exact_state_identity_differs": (
                persistent_slot_state_sha256(pair.original)
                != persistent_slot_state_sha256(pair.alternate)
            ),
            "component_slots": list(pair.component_slots),
            "component_edges": [list(edge) for edge in pair.component_edges],
            "slot_permutation_new_to_old": list(order),
        },
        "unit_mass_preflight": {
            "laws": {name: law.to_payload() for name, law in unit_laws.items()},
            "comparisons": comparisons,
            "slot_action_transport": slot_transport,
            "kekule_support_passed": kekule_support_passed,
            "slot_gate_passed": slot_gate_passed,
            "unit_law_passed": unit_law_passed,
        },
        "scored_family_conditional": scored_payload,
        "limitations": [
            (
                "The alternate constructor covers one isolated alternating six-membered aromatic "
                "component, not fused or general resonance systems."
            ),
            (
                "A bounded pass would be regression evidence, not a proof for every molecule or "
                "parameter setting."
            ),
            (
                "A support mismatch is an executor or semantic-state finding and cannot be repaired "
                "by the tested neural scorer alone."
            ),
            (
                "This diagnostic is prospective and does not alter or retrospectively extend the "
                "frozen E5 quotient-invariance claim."
            ),
        ],
    }


__all__ = [
    "ALLOWED_SCORER_MODES",
    "AlternatingKekulePair",
    "CycleOpenKekuleInvarianceError",
    "CycleOpenLaw",
    "build_alternate_kekule_pair",
    "compare_cycle_open_laws",
    "executable_cycle_open_actions",
    "family_conditional_cycle_open_law",
    "run_cycle_open_kekule_invariance_diagnostic",
    "transport_bond_delete",
    "transport_component_slots",
    "unit_mass_cycle_open_law",
]
