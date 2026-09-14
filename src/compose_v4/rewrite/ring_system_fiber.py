"""Exact cached candidates for complete ring-system grow/delete templates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_CLASS_TO_H_CHANGE,
    ELEMENT_TO_IDX,
    MAX_H_COUNT,
    MolecularGraph,
    canonical_h_count,
    is_element,
    per_atom_valence_check,
)
from compose_v4.rewrite.factorized_fiber import CNOF_ATOM_TYPES, CNOF_VALENCE
from compose_v4.rewrite.scaffold_construction import ScaffoldContext
from compose_v4.rewrite.trace import RewriteTrace
from compose_v4.rewrite.tracelets import (
    AtomPayload,
    BondOrderChange,
    RingBond,
    RingSystemDelete,
    RingSystemGrow,
    is_valid_ring_system_grow,
)
from compose_v4.rewrite.typed_ring_catalog import (
    require_ring_system_electronic_aliases,
    ring_system_electronic_alias,
    ring_system_electronic_aliases_equivalent,
    RingSystemElectronicAlias,
    RingSystemTemplate,
    TypedRingCatalog,
)


@dataclass(frozen=True)
class RingSystemPlacement:
    """One topology-and-bond-pattern match with atom labels left factorized."""

    system_atoms: tuple[int, ...]
    scaffold_bonds: tuple[RingBond, ...]
    bond_reorders: tuple[BondOrderChange, ...]
    bond_insertions: tuple[RingBond, ...]
    aromatic_edges: tuple[tuple[int, int], ...]
    topology_class: str

    def instantiate(
        self,
        state: MolecularGraph,
        atom_types: tuple[int, ...],
    ) -> RingSystemGrow:
        if len(atom_types) != len(self.system_atoms):
            raise ValueError("ring placement received the wrong number of atom labels")
        target_orders = {
            frozenset((int(bond.a), int(bond.b))): int(bond.order) for bond in self.scaffold_bonds
        }
        target_orders.update(
            {
                frozenset((int(change.a), int(change.b))): int(change.new_order)
                for change in self.bond_reorders
            }
        )
        target_orders.update(
            {
                frozenset((int(bond.a), int(bond.b))): int(bond.order)
                for bond in self.bond_insertions
            }
        )
        members = set(self.system_atoms)
        payloads = []
        for slot, atom_type in zip(self.system_atoms, atom_types):
            external_valence = sum(
                int(BOND_CLASS_TO_H_CHANGE[int(state.bonds[slot, neighbor])])
                for neighbor in np.flatnonzero(state.bonds[slot] != 0)
                if int(neighbor) not in members
            )
            internal_valence = sum(
                int(BOND_CLASS_TO_H_CHANGE[order])
                for edge, order in target_orders.items()
                if slot in edge
            )
            final_h = int(CNOF_VALENCE[int(atom_type)]) - external_valence - internal_valence
            closure_h = sum(
                int(BOND_CLASS_TO_H_CHANGE[int(bond.order)])
                for bond in self.bond_insertions
                if slot in (int(bond.a), int(bond.b))
            )
            precursor_h = final_h + closure_h
            if not 0 <= final_h <= MAX_H_COUNT or not 0 <= precursor_h <= MAX_H_COUNT:
                raise ValueError("atom labels violate the installed ring valence")
            payloads.append(
                AtomPayload(
                    slot=int(slot),
                    atom_type=int(atom_type),
                    formal_charge=0,
                    implicit_h_count=int(precursor_h),
                )
            )
        return RingSystemGrow(
            system_atoms=self.system_atoms,
            interface_atoms=(),
            scaffold_bonds=self.scaffold_bonds,
            bond_reorders=self.bond_reorders,
            atom_payloads=tuple(payloads),
            atom_insertions=(),
            bond_insertions=self.bond_insertions,
            source_aromatic_edges=(),
            aromatic_edges=self.aromatic_edges,
            topology_class=self.topology_class,
        )


@dataclass(frozen=True)
class ExecutableRingGrowCandidate:
    """One executor-verified paired electronic ring action."""

    placement: RingSystemPlacement
    atom_types: tuple[int, ...]
    action: RingSystemGrow
    support_count: int

    def __post_init__(self) -> None:
        if len(self.atom_types) != len(self.placement.system_atoms):
            raise ValueError("ring candidate atom labels do not align with its placement")
        if int(self.support_count) <= 0:
            raise ValueError("ring candidate support count must be positive")


@dataclass(frozen=True, order=True)
class RingAtomElectronicState:
    """One semantic atom label in a complete ring-system successor."""

    atom_type: int
    final_h_count: int
    aromatic_demand: int
    pi_electrons: int


@dataclass(frozen=True)
class SemanticAromaticComponent:
    """One sparse aromatic matching problem in placement-local indices."""

    positions: tuple[int, ...]
    adjacency_masks: tuple[int, ...]
    enforce_huckel_parity: bool


@dataclass(frozen=True)
class SemanticRingSystemDecoder:
    """Finite-state electronic decoder for one semantic ring placement."""

    placement: RingSystemPlacement
    options_by_member: tuple[tuple[RingAtomElectronicState, ...], ...]
    aromatic_components: tuple[SemanticAromaticComponent, ...]
    aromatic_positions: frozenset[int]
    source_state_key: tuple[bytes, bytes, bytes, bytes]
    source_state: MolecularGraph = field(compare=False, hash=False, repr=False)
    # Optional source-only condition. It participates in equality/hash, so all
    # memoized prefix and leaf predicates are isolated across supplied cores.
    scaffold_context: ScaffoldContext | None = None

    @property
    def span(self) -> int:
        return len(self.options_by_member)


_ELECTRONIC_CATEGORY_COUNT = 2 * len(CNOF_ATOM_TYPES)
_CNOF_TO_LOCAL_INDEX = {
    int(atom_type): index for index, atom_type in enumerate(CNOF_ATOM_TYPES)
}


def ring_atom_electronic_category(state: RingAtomElectronicState) -> int:
    """Map a semantic atom state to a C/N/O/F-by-role model category."""

    try:
        atom_index = _CNOF_TO_LOCAL_INDEX[int(state.atom_type)]
    except KeyError as exc:
        raise ValueError("semantic ring atom is outside C/N/O/F") from exc
    demand = int(state.aromatic_demand)
    if demand not in (0, 1):
        raise ValueError("aromatic demand must be zero or one")
    return 2 * atom_index + demand


def _semantic_target_orders(
    placement: RingSystemPlacement,
) -> dict[frozenset[int], int]:
    target_orders = {
        frozenset((int(bond.a), int(bond.b))): int(bond.order)
        for bond in placement.scaffold_bonds
    }
    target_orders.update(
        {
            frozenset((int(change.a), int(change.b))): int(change.new_order)
            for change in placement.bond_reorders
        }
    )
    target_orders.update(
        {
            frozenset((int(bond.a), int(bond.b))): int(bond.order)
            for bond in placement.bond_insertions
        }
    )
    return target_orders


def build_semantic_ring_system_decoder(
    state: MolecularGraph,
    placement: RingSystemPlacement,
    *,
    scaffold_context: ScaffoldContext | None = None,
) -> SemanticRingSystemDecoder:
    """Compile neutral-CNOF joint electronic support for one placement.

    Aromatic integer-order lowerings are matchings. A normal aromatic carbon
    or pyridine-like nitrogen demands one matched edge; donor N/O demands
    none; an exocyclicly unsaturated carbon also demands none. Every connected
    monocyclic aromatic component additionally carries 4k+2 pi electrons.
    Polycyclic aromatic systems do not obey one global H\N{LATIN SMALL LETTER U WITH DIAERESIS}ckel count, so
    their sparse matching support is pruned here while the molecular executor
    remains the authoritative electronic predicate at complete leaves.
    """

    if scaffold_context is not None and not scaffold_context.accepts(state):
        raise ValueError("ring decoder state violates its supplied scaffold context")
    protected = (
        {} if scaffold_context is None else
        {slot: (element, charge, h) for slot, element, charge, h in scaffold_context.atoms}
    )
    minimum_h = {} if scaffold_context is None else dict(scaffold_context.minimum_h_counts)
    members = tuple(int(slot) for slot in placement.system_atoms)
    if len(set(members)) != len(members):
        raise ValueError("semantic ring placement repeats a member")
    member_set = set(members)
    position_by_slot = {slot: index for index, slot in enumerate(members)}
    aromatic_edges = {
        frozenset((int(a), int(b))) for a, b in placement.aromatic_edges
    }
    target_orders = _semantic_target_orders(placement)
    if not aromatic_edges.issubset(target_orders):
        raise ValueError("aromatic edge is absent from the semantic target")

    carbon = int(ELEMENT_TO_IDX["C"])
    nitrogen = int(ELEMENT_TO_IDX["N"])
    oxygen = int(ELEMENT_TO_IDX["O"])
    options_by_member: list[tuple[RingAtomElectronicState, ...]] = []
    for slot in members:
        external_orders = tuple(
            int(state.bonds[slot, neighbor])
            for neighbor in np.flatnonzero(state.bonds[slot] != 0)
            if int(neighbor) not in member_set
        )
        baseline_valence = sum(
            int(BOND_CLASS_TO_H_CHANGE[order]) for order in external_orders
        )
        baseline_valence += sum(
            1 if edge in aromatic_edges else int(BOND_CLASS_TO_H_CHANGE[order])
            for edge, order in target_orders.items()
            if slot in edge
        )
        fixed_pi_excess = sum(
            max(int(BOND_CLASS_TO_H_CHANGE[order]) - 1, 0)
            for order in external_orders
        )
        fixed_pi_excess += sum(
            max(int(BOND_CLASS_TO_H_CHANGE[order]) - 1, 0)
            for edge, order in target_orders.items()
            if slot in edge and edge not in aromatic_edges
        )
        aromatic = any(slot in edge for edge in aromatic_edges)
        options: set[RingAtomElectronicState] = set()
        if not aromatic:
            for atom_type in CNOF_ATOM_TYPES:
                hydrogens = int(CNOF_VALENCE[int(atom_type)]) - baseline_valence
                if 0 <= hydrogens <= MAX_H_COUNT:
                    options.add(
                        RingAtomElectronicState(
                            int(atom_type),
                            int(hydrogens),
                            0,
                            0,
                        )
                    )
        else:
            for atom_type_value in CNOF_ATOM_TYPES:
                atom_type = int(atom_type_value)
                for demand in (0, 1):
                    hydrogens = (
                        int(CNOF_VALENCE[atom_type])
                        - baseline_valence
                        - demand
                    )
                    if not 0 <= hydrogens <= MAX_H_COUNT:
                        continue
                    if atom_type == carbon:
                        allowed = demand + fixed_pi_excess == 1
                        pi_electrons = 1 if demand else 0
                    elif atom_type == nitrogen:
                        allowed = fixed_pi_excess == 0
                        pi_electrons = 1 if demand else 2
                    elif atom_type == oxygen:
                        allowed = fixed_pi_excess == 0 and demand == 0
                        pi_electrons = 2
                    else:
                        allowed = False
                        pi_electrons = 0
                    if allowed:
                        options.add(
                            RingAtomElectronicState(
                                atom_type,
                                int(hydrogens),
                                demand,
                                pi_electrons,
                            )
                        )
        if slot in protected:
            element, charge, source_h = protected[slot]
            options = {
                option for option in options
                if charge == 0 and option.atom_type == element
                and option.final_h_count >= minimum_h.get(slot, 0)
                and (slot in scaffold_context.attachment_slots
                     or option.final_h_count == source_h)
            }
        options_by_member.append(tuple(sorted(options)))

    aromatic_graph = nx.Graph()
    aromatic_positions = {
        position_by_slot[slot] for edge in aromatic_edges for slot in edge
    }
    aromatic_graph.add_nodes_from(sorted(aromatic_positions))
    aromatic_graph.add_edges_from(
        (position_by_slot[min(edge)], position_by_slot[max(edge)])
        for edge in aromatic_edges
    )
    components = []
    component_positions = tuple(
        sorted(
            (tuple(sorted(items)) for items in nx.connected_components(aromatic_graph)),
            key=lambda items: items,
        )
    )
    for positions in component_positions:
        local_index = {position: index for index, position in enumerate(positions)}
        adjacency_masks = []
        for position in positions:
            mask = 0
            for neighbor in aromatic_graph.neighbors(position):
                mask |= 1 << local_index[int(neighbor)]
            adjacency_masks.append(mask)
        edge_count = sum(int(mask).bit_count() for mask in adjacency_masks) // 2
        is_simple_cycle = (
            len(positions) >= 3
            and edge_count == len(positions)
            and all(int(mask).bit_count() == 2 for mask in adjacency_masks)
        )
        components.append(
            SemanticAromaticComponent(
                positions,
                tuple(adjacency_masks),
                enforce_huckel_parity=is_simple_cycle,
            )
        )
    return SemanticRingSystemDecoder(
        placement,
        tuple(options_by_member),
        tuple(components),
        frozenset(aromatic_positions),
        (
            state.atom_types.tobytes(),
            state.formal_charges.tobytes(),
            state.implicit_h_counts.tobytes(),
            state.bonds.tobytes(),
        ),
        state,
        scaffold_context,
    )


def _option_mode(option: RingAtomElectronicState) -> int:
    if int(option.aromatic_demand) == 1:
        return 0b100
    if int(option.pi_electrons) == 0:
        return 0b001
    if int(option.pi_electrons) == 2:
        return 0b010
    raise ValueError("unmatched aromatic atom must contribute zero or two pi electrons")


def _option_completion_symbol(
    decoder: SemanticRingSystemDecoder,
    position: int,
    option: RingAtomElectronicState,
) -> tuple[int, int]:
    """Return the smallest electronic class relevant to exact completion.

    Aromatic atom types with the same donation/matching role are equivalent
    for aromaticity. Nonaromatic labels remain category-specific: their
    element can affect exocyclic donation and therefore cannot be quotiented
    safely in the general case.
    """

    if int(position) in decoder.aromatic_positions:
        return (1, _option_mode(option))
    return (0, ring_atom_electronic_category(option))


@lru_cache(maxsize=131072)
def _aromatic_component_parities(
    adjacency_masks: tuple[int, ...],
    option_modes: tuple[int, ...],
) -> int:
    """Return a two-bit mask of achievable half-electron parities."""

    if len(adjacency_masks) != len(option_modes):
        raise ValueError("aromatic modes do not align with the component")

    @lru_cache(maxsize=None)
    def solve(remaining: int) -> int:
        if remaining == 0:
            return 0b01
        vertex_bit = remaining & -remaining
        vertex = vertex_bit.bit_length() - 1
        rest = remaining ^ vertex_bit
        mode = int(option_modes[vertex])
        parities = 0
        if mode & 0b001:
            parities |= solve(rest)
        if mode & 0b010:
            base = solve(rest)
            parities |= ((base & 0b01) << 1) | ((base & 0b10) >> 1)
        if mode & 0b100:
            candidates = int(adjacency_masks[vertex]) & rest
            while candidates:
                neighbor_bit = candidates & -candidates
                candidates ^= neighbor_bit
                neighbor = neighbor_bit.bit_length() - 1
                if not (int(option_modes[neighbor]) & 0b100):
                    continue
                base = solve(rest ^ neighbor_bit)
                parities |= ((base & 0b01) << 1) | ((base & 0b10) >> 1)
        return parities

    return solve((1 << len(option_modes)) - 1)


def _selected_semantic_option(
    decoder: SemanticRingSystemDecoder,
    position: int,
    category: int,
) -> RingAtomElectronicState | None:
    for option in decoder.options_by_member[position]:
        if ring_atom_electronic_category(option) == int(category):
            return option
    return None


@lru_cache(maxsize=131072)
def semantic_ring_prefix_is_completable(
    decoder: SemanticRingSystemDecoder,
    prefix: tuple[int, ...],
) -> bool:
    """Whether a prefix has an executor-valid semantic completion.

    Sparse matching and electron parity are inexpensive necessary conditions.
    The molecular executor remains the authoritative leaf predicate so fused
    and heteroaromatic cases are never accepted by a hand-written relaxation
    alone.  Memoization turns repeated teacher/sampler prefix queries into a
    compact exact decision diagram for this placed ring system.
    """

    if len(prefix) > decoder.span:
        return False
    symbols = []
    for position, category in enumerate(prefix):
        option = _selected_semantic_option(decoder, position, int(category))
        if option is None:
            return False
        symbols.append(_option_completion_symbol(decoder, position, option))
    return _semantic_ring_symbol_prefix_is_completable(
        decoder,
        tuple(symbols),
    )


def _selected_symbol_option(
    decoder: SemanticRingSystemDecoder,
    position: int,
    symbol: tuple[int, int],
) -> RingAtomElectronicState | None:
    return next(
        (
            option
            for option in decoder.options_by_member[position]
            if _option_completion_symbol(decoder, position, option) == symbol
        ),
        None,
    )


@lru_cache(maxsize=131072)
def _semantic_ring_symbol_prefix_is_completable(
    decoder: SemanticRingSystemDecoder,
    prefix: tuple[tuple[int, int], ...],
) -> bool:
    """Exact completion DP quotiented by electronic equivalence classes."""

    if len(prefix) > decoder.span:
        return False
    selected = []
    for position, symbol in enumerate(prefix):
        option = _selected_symbol_option(decoder, position, symbol)
        if option is None:
            return False
        selected.append(option)
    if any(not options for options in decoder.options_by_member[len(prefix) :]):
        return False
    for component in decoder.aromatic_components:
        modes = []
        for position in component.positions:
            if position < len(selected):
                modes.append(_option_mode(selected[position]))
            else:
                mode = 0
                for option in decoder.options_by_member[position]:
                    mode |= _option_mode(option)
                modes.append(mode)
        achievable_parities = _aromatic_component_parities(
            component.adjacency_masks,
            tuple(modes),
        )
        required_parities = 0b10 if component.enforce_huckel_parity else 0b11
        if not (achievable_parities & required_parities):
            return False
    if len(prefix) == decoder.span:
        categories = tuple(
            ring_atom_electronic_category(option)
            for option in selected
        )
        return _semantic_categories_are_executor_valid(decoder, categories)
    available_symbols = {
        _option_completion_symbol(decoder, len(prefix), option)
        for option in decoder.options_by_member[len(prefix)]
    }
    return any(
        _semantic_ring_symbol_prefix_is_completable(decoder, (*prefix, symbol))
        for symbol in sorted(available_symbols)
    )


@lru_cache(maxsize=131072)
def semantic_ring_next_category_mask(
    decoder: SemanticRingSystemDecoder,
    prefix: tuple[int, ...],
) -> tuple[bool, ...]:
    """Return the exact completion-aware mask for the next semantic label."""

    if len(prefix) >= decoder.span:
        return tuple(False for _ in range(_ELECTRONIC_CATEGORY_COUNT))
    available = {
        ring_atom_electronic_category(option)
        for option in decoder.options_by_member[len(prefix)]
    }
    prefix_symbols = tuple(
        _option_completion_symbol(
            decoder,
            position,
            _selected_semantic_option(decoder, position, category),
        )
        for position, category in enumerate(prefix)
        if _selected_semantic_option(decoder, position, category) is not None
    )
    if len(prefix_symbols) != len(prefix):
        return tuple(False for _ in range(_ELECTRONIC_CATEGORY_COUNT))
    return tuple(
        category in available
        and _semantic_ring_symbol_prefix_is_completable(
            decoder,
            (
                *prefix_symbols,
                _option_completion_symbol(
                    decoder,
                    len(prefix),
                    _selected_semantic_option(decoder, len(prefix), category),
                ),
            ),
        )
        for category in range(_ELECTRONIC_CATEGORY_COUNT)
    )


def clear_semantic_ring_state_caches() -> None:
    """Release decoder/state-specific semantic DP entries in data workers.

    These memo tables are exact accelerators rather than stochastic state.
    Clearing them periodically bounds long-run worker memory without changing
    any support decision. Generic topology and matching caches are retained.
    """

    semantic_ring_prefix_is_completable.cache_clear()
    _semantic_ring_symbol_prefix_is_completable.cache_clear()
    semantic_ring_next_category_mask.cache_clear()
    _semantic_action_for_categories.cache_clear()


@lru_cache(maxsize=131072)
def _canonical_component_matching(
    adjacency_masks: tuple[int, ...],
    required: int,
) -> tuple[tuple[int, int], ...] | None:
    if required == 0:
        return ()
    vertex_bit = required & -required
    vertex = vertex_bit.bit_length() - 1
    candidates = int(adjacency_masks[vertex]) & (required ^ vertex_bit)
    while candidates:
        neighbor_bit = candidates & -candidates
        candidates ^= neighbor_bit
        neighbor = neighbor_bit.bit_length() - 1
        remainder = _canonical_component_matching(
            adjacency_masks,
            required ^ vertex_bit ^ neighbor_bit,
        )
        if remainder is not None:
            return ((vertex, neighbor), *remainder)
    return None


def semantic_ring_canonical_matching(
    decoder: SemanticRingSystemDecoder,
    categories: tuple[int, ...],
) -> frozenset[frozenset[int]]:
    """Return one deterministic scaffold-compatible Kekule lowering.

    A semantic aromatic system can have several equivalent perfect matchings.
    When some of its edges belong to a supplied scaffold, their executable
    integer bond orders are exact conditions: protected single edges cannot be
    selected as double bonds and protected double edges must be selected.  The
    unconstrained lexicographic matching can otherwise reject a valid aromatic
    successor merely because it chooses a different resonance representative.
    """

    if len(categories) != decoder.span:
        raise ValueError("semantic category sequence is incomplete")
    selected = tuple(
        _selected_semantic_option(decoder, position, category)
        for position, category in enumerate(categories)
    )
    if any(option is None for option in selected):
        raise ValueError("semantic category is unavailable at its position")
    result: set[frozenset[int]] = set()
    for component in decoder.aromatic_components:
        slots = tuple(
            int(decoder.placement.system_atoms[position])
            for position in component.positions
        )
        local_by_slot = {slot: index for index, slot in enumerate(slots)}
        adjacency_masks = list(component.adjacency_masks)
        forced: list[tuple[int, int]] = []
        forced_vertices = 0
        if decoder.scaffold_context is not None:
            aromatic = {
                frozenset((int(a), int(b)))
                for a, b in decoder.placement.aromatic_edges
            }
            for left_slot, right_slot, order in decoder.scaffold_context.bonds:
                edge = frozenset((int(left_slot), int(right_slot)))
                if edge not in aromatic or not edge.issubset(local_by_slot):
                    continue
                left = local_by_slot[int(left_slot)]
                right = local_by_slot[int(right_slot)]
                if int(order) == 1:
                    adjacency_masks[left] &= ~(1 << right)
                    adjacency_masks[right] &= ~(1 << left)
                elif int(order) == 2:
                    bits = (1 << left) | (1 << right)
                    if forced_vertices & bits:
                        raise ValueError(
                            "protected aromatic double bonds do not form a matching"
                        )
                    forced_vertices |= bits
                    forced.append((left, right))
                else:
                    raise ValueError(
                        "protected aromatic edge lacks a Kekule-compatible bond order"
                    )
        required = 0
        for local_index, position in enumerate(component.positions):
            option = selected[position]
            if option is not None and int(option.aromatic_demand) == 1:
                required |= 1 << local_index
        if forced_vertices & ~required:
            raise ValueError(
                "protected aromatic double bond disagrees with semantic demand"
            )
        matching = _canonical_component_matching(
            tuple(adjacency_masks),
            required ^ forced_vertices,
        )
        if matching is None:
            raise ValueError("semantic aromatic demands lack a perfect matching")
        for left, right in (*forced, *matching):
            left_slot = slots[left]
            right_slot = slots[right]
            result.add(frozenset((int(left_slot), int(right_slot))))
    return frozenset(result)


def ring_system_placement_with_aromatic_matching(
    placement: RingSystemPlacement,
    double_edges: frozenset[frozenset[int]],
) -> RingSystemPlacement:
    """Lower a semantic placement through one canonical aromatic matching."""

    aromatic = {
        frozenset((int(a), int(b))) for a, b in placement.aromatic_edges
    }
    if not double_edges.issubset(aromatic):
        raise ValueError("aromatic matching leaves the placement")
    used: set[int] = set()
    for edge in double_edges:
        if len(edge) != 2 or used & set(edge):
            raise ValueError("aromatic double edges do not form a matching")
        used.update(int(slot) for slot in edge)
    source_orders = {
        frozenset((int(bond.a), int(bond.b))): int(bond.order)
        for bond in placement.scaffold_bonds
    }
    inserted_edges = {
        frozenset((int(bond.a), int(bond.b)))
        for bond in placement.bond_insertions
    }
    target_orders = _semantic_target_orders(placement)
    for edge in aromatic:
        target_orders[edge] = 2 if edge in double_edges else 1
    return RingSystemPlacement(
        system_atoms=placement.system_atoms,
        scaffold_bonds=placement.scaffold_bonds,
        bond_reorders=tuple(
            sorted(
                BondOrderChange(min(edge), max(edge), int(target_orders[edge]))
                for edge, source_order in source_orders.items()
                if int(target_orders[edge]) != int(source_order)
            )
        ),
        bond_insertions=tuple(
            sorted(
                RingBond(min(edge), max(edge), int(target_orders[edge]))
                for edge in inserted_edges
            )
        ),
        aromatic_edges=placement.aromatic_edges,
        topology_class=placement.topology_class,
    )


def ring_system_grow_final_atom_states(
    action: RingSystemGrow,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return slot, element, charge, final-H semantic labels for a grow."""

    closure_h = {
        int(slot): sum(
            int(BOND_CLASS_TO_H_CHANGE[int(bond.order)])
            for bond in action.bond_insertions
            if int(slot) in (int(bond.a), int(bond.b))
        )
        for slot in action.system_atoms
    }
    payload_by_slot = {
        int(payload.slot): payload for payload in action.atom_payloads
    }
    result = []
    for slot in sorted(int(value) for value in action.system_atoms):
        payload = payload_by_slot[slot]
        result.append(
            (
                slot,
                int(payload.atom_type),
                int(payload.formal_charge),
                int(payload.implicit_h_count) - closure_h[slot],
            )
        )
    return tuple(result)


@lru_cache(maxsize=131072)
def _semantic_action_for_categories(
    decoder: SemanticRingSystemDecoder,
    categories: tuple[int, ...],
) -> RingSystemGrow | None:
    """Materialize one canonical lowering, returning ``None`` if rejected."""

    if len(categories) != decoder.span:
        return None
    selected = tuple(
        _selected_semantic_option(decoder, position, category)
        for position, category in enumerate(categories)
    )
    if any(option is None for option in selected):
        return None
    try:
        lowering = ring_system_placement_with_aromatic_matching(
            decoder.placement,
            semantic_ring_canonical_matching(decoder, categories),
        )
        action = lowering.instantiate(
            decoder.source_state,
            tuple(
                int(option.atom_type)
                for option in selected
                if option is not None
            ),
        )
    except ValueError:
        return None
    expected = tuple(
        (
            int(slot),
            int(option.atom_type),
            0,
            int(option.final_h_count),
        )
        for slot, option in zip(decoder.placement.system_atoms, selected)
        if option is not None
    )
    if ring_system_grow_final_atom_states(action) != expected:
        return None
    if not _semantic_ring_final_state_is_valid(decoder, categories):
        return None
    return action


def _semantic_ring_final_state_is_valid(
    decoder: SemanticRingSystemDecoder,
    categories: tuple[int, ...],
) -> bool:
    """Validate one semantic lowering with a single final-state sanitize.

    A v1 placement matches a complete acyclic scaffold. Bond reorders preserve
    endpoint valence, atom payloads encode the exact pre-closure hydrogens, and
    every closure consumes those reserved hydrogens. Consequently transient
    micro-step validity reduces to local range checks already enforced by the
    placement plus validity of the committed successor. Building that
    successor directly avoids repeatedly sanitizing every micro-step while
    retaining RDKit as the authoritative aromaticity predicate.
    """

    if len(categories) != decoder.span:
        return False
    selected = tuple(
        _selected_semantic_option(decoder, position, category)
        for position, category in enumerate(categories)
    )
    if any(option is None for option in selected):
        return False
    try:
        lowering = ring_system_placement_with_aromatic_matching(
            decoder.placement,
            semantic_ring_canonical_matching(decoder, categories),
        )
    except ValueError:
        return False

    atom_types = decoder.source_state.atom_types.copy()
    formal_charges = decoder.source_state.formal_charges.copy()
    implicit_h_counts = decoder.source_state.implicit_h_counts.copy()
    bonds = decoder.source_state.bonds.copy()
    members = tuple(int(slot) for slot in lowering.system_atoms)
    for slot, option in zip(members, selected):
        if option is None:
            return False
        atom_types[slot] = int(option.atom_type)
        formal_charges[slot] = 0
        implicit_h_counts[slot] = int(option.final_h_count)
    target_orders = _semantic_target_orders(lowering)
    for left_index, left in enumerate(members):
        for right in members[left_index + 1 :]:
            order = int(target_orders.get(frozenset((left, right)), 0))
            bonds[left, right] = bonds[right, left] = order
    successor = MolecularGraph(
        atom_types,
        formal_charges,
        implicit_h_counts,
        bonds,
    )
    if decoder.scaffold_context is not None and not decoder.scaffold_context.accepts(successor):
        return False
    if not bool(per_atom_valence_check(successor).all()):
        return False
    try:
        perceived = resonance_invariant_bond_classes(successor)
    except Exception:
        return False

    graph = nx.Graph()
    real = tuple(int(slot) for slot in np.flatnonzero(is_element(atom_types)))
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (left, right)
        for left_index, left in enumerate(real)
        for right in real[left_index + 1 :]
        if int(bonds[left, right]) != 0
    )
    bridges = {frozenset((int(left), int(right))) for left, right in nx.bridges(graph)}
    cyclic_graph = nx.Graph()
    cyclic_graph.add_edges_from(
        (int(left), int(right))
        for left, right in graph.edges()
        if frozenset((int(left), int(right))) not in bridges
    )
    cyclic_components = {
        frozenset(int(slot) for slot in component)
        for component in nx.connected_components(cyclic_graph)
    }
    if frozenset(members) not in cyclic_components:
        return False
    actual_aromatic = {
        (left, right)
        for left_index, left in enumerate(sorted(members))
        for right in sorted(members)[left_index + 1 :]
        if int(perceived[left, right]) == BOND_AROMATIC
    }
    expected_aromatic = {
        (int(left), int(right))
        for left, right in lowering.aromatic_edges
    }
    return actual_aromatic == expected_aromatic


def _semantic_categories_are_executor_valid(
    decoder: SemanticRingSystemDecoder,
    categories: tuple[int, ...],
) -> bool:
    return _semantic_action_for_categories(decoder, categories) is not None


def semantic_ring_categories_for_action(
    decoder: SemanticRingSystemDecoder,
    action: RingSystemGrow,
) -> tuple[int, ...]:
    """Map any Kekule alias of a teacher to semantic type/H/role labels."""

    final_by_slot = {
        slot: (atom_type, charge, hydrogens)
        for slot, atom_type, charge, hydrogens
        in ring_system_grow_final_atom_states(action)
    }
    categories = []
    for position, slot in enumerate(decoder.placement.system_atoms):
        atom_type, charge, hydrogens = final_by_slot[int(slot)]
        if charge != 0:
            raise ValueError("v1 semantic ring decoder supports neutral labels")
        matching = tuple(
            option
            for option in decoder.options_by_member[position]
            if int(option.atom_type) == atom_type
            and int(option.final_h_count) == hydrogens
        )
        if len(matching) != 1:
            raise ValueError("teacher atom state has no unique semantic category")
        categories.append(ring_atom_electronic_category(matching[0]))
    return tuple(categories)


def instantiate_semantic_ring_system_grow(
    state: MolecularGraph,
    decoder: SemanticRingSystemDecoder,
    categories: tuple[int, ...],
) -> RingSystemGrow:
    """Instantiate the canonical executor-valid alias for semantic labels."""

    state_key = (
        state.atom_types.tobytes(),
        state.formal_charges.tobytes(),
        state.implicit_h_counts.tobytes(),
        state.bonds.tobytes(),
    )
    if state_key != decoder.source_state_key:
        raise ValueError("semantic ring decoder belongs to a different source state")
    if not semantic_ring_prefix_is_completable(decoder, categories):
        raise ValueError("semantic ring labels have no executable completion")
    action = _semantic_action_for_categories(decoder, categories)
    if action is None:
        raise RuntimeError("completion oracle accepted a non-executable ring action")
    return action


def ring_system_placement_local_atom_type_mask(
    state: MolecularGraph,
    placement: RingSystemPlacement,
) -> np.ndarray:
    """Return the locally executable C/N/O/F labels for one placement.

    The mask is an application condition shared by training and sampling.  It
    accounts for the complete installed bond pattern, bonds to atoms outside
    the ring system, and the restricted neutral aromatic atom semantics.  A
    nonempty row for every member is necessary for an executable joint label;
    the final rewrite validator remains the authoritative whole-action check.
    """

    members = set(placement.system_atoms)
    aromatic_edges = {frozenset((int(a), int(b))) for a, b in placement.aromatic_edges}
    target_orders = {
        frozenset((int(bond.a), int(bond.b))): int(bond.order)
        for bond in placement.scaffold_bonds
    }
    target_orders.update(
        {
            frozenset((int(change.a), int(change.b))): int(change.new_order)
            for change in placement.bond_reorders
        }
    )
    target_orders.update(
        {
            frozenset((int(bond.a), int(bond.b))): int(bond.order)
            for bond in placement.bond_insertions
        }
    )
    aromatic_slots = {int(slot) for edge in aromatic_edges for slot in edge}
    mask = np.zeros(
        (len(placement.system_atoms), len(CNOF_ATOM_TYPES)),
        dtype=np.bool_,
    )
    for member_index, slot in enumerate(placement.system_atoms):
        external_valence = sum(
            int(BOND_CLASS_TO_H_CHANGE[int(state.bonds[slot, neighbor])])
            for neighbor in np.flatnonzero(state.bonds[slot] != 0)
            if int(neighbor) not in members
        )
        internal_valence = sum(
            int(BOND_CLASS_TO_H_CHANGE[order])
            for edge, order in target_orders.items()
            if int(slot) in edge
        )
        bond_valence = external_valence + internal_valence
        for atom_index, candidate_atom_type in enumerate(CNOF_ATOM_TYPES):
            hydrogens = int(CNOF_VALENCE[int(candidate_atom_type)]) - bond_valence
            allowed = 0 <= hydrogens <= MAX_H_COUNT
            if int(slot) in aromatic_slots and allowed:
                atom_type = int(candidate_atom_type)
                if atom_type == int(CNOF_ATOM_TYPES[0]):
                    allowed = True
                elif atom_type == int(CNOF_ATOM_TYPES[1]):
                    pyridine_like = bond_valence == 3 and hydrogens == 0
                    pyrrole_like = (
                        internal_valence == 2
                        and external_valence == 0
                        and hydrogens == 1
                    )
                    allowed = pyridine_like or pyrrole_like
                elif atom_type == int(CNOF_ATOM_TYPES[2]):
                    allowed = (
                        internal_valence == 2
                        and external_valence == 0
                        and hydrogens == 0
                    )
                else:
                    allowed = False
            mask[member_index, atom_index] = allowed
    mask.setflags(write=False)
    return mask


def ring_system_placement_resonance_atom_type_mask(
    state: MolecularGraph,
    placement: RingSystemPlacement,
) -> np.ndarray:
    """Union the exact local label mask over all resonance matchings.

    A node in a Kekule lowering is either unmatched (all incident aromatic
    edges single) or matched once (one incident aromatic edge double).  The
    row-wise union therefore needs only these two valence cases, not an
    exponential scan of complete-ring matchings.  Joint feasibility is still
    checked by the atomic executor when labels and a resonance alias are paired.
    """

    members = set(placement.system_atoms)
    aromatic_edges = {
        frozenset((int(a), int(b))) for a, b in placement.aromatic_edges
    }
    aromatic_slots = {slot for edge in aromatic_edges for slot in edge}
    target_orders = {
        frozenset((int(bond.a), int(bond.b))): int(bond.order)
        for bond in placement.scaffold_bonds
    }
    target_orders.update(
        {
            frozenset((int(change.a), int(change.b))): int(change.new_order)
            for change in placement.bond_reorders
        }
    )
    target_orders.update(
        {
            frozenset((int(bond.a), int(bond.b))): int(bond.order)
            for bond in placement.bond_insertions
        }
    )
    for edge in aromatic_edges:
        target_orders[edge] = 1

    mask = np.zeros(
        (len(placement.system_atoms), len(CNOF_ATOM_TYPES)),
        dtype=np.bool_,
    )
    for member_index, slot in enumerate(placement.system_atoms):
        external_valence = sum(
            int(BOND_CLASS_TO_H_CHANGE[int(state.bonds[slot, neighbor])])
            for neighbor in np.flatnonzero(state.bonds[slot] != 0)
            if int(neighbor) not in members
        )
        baseline_internal_valence = sum(
            int(BOND_CLASS_TO_H_CHANGE[order])
            for edge, order in target_orders.items()
            if int(slot) in edge
        )
        matched_cases = (0, 1) if int(slot) in aromatic_slots else (0,)
        for matched_increment in matched_cases:
            internal_valence = baseline_internal_valence + matched_increment
            bond_valence = external_valence + internal_valence
            for atom_index, candidate_atom_type in enumerate(CNOF_ATOM_TYPES):
                hydrogens = int(CNOF_VALENCE[int(candidate_atom_type)]) - bond_valence
                allowed = 0 <= hydrogens <= MAX_H_COUNT
                if int(slot) in aromatic_slots and allowed:
                    atom_type = int(candidate_atom_type)
                    if atom_type == int(CNOF_ATOM_TYPES[0]):
                        allowed = True
                    elif atom_type == int(CNOF_ATOM_TYPES[1]):
                        pyridine_like = bond_valence == 3 and hydrogens == 0
                        pyrrole_like = (
                            internal_valence == 2
                            and external_valence == 0
                            and hydrogens == 1
                        )
                        allowed = pyridine_like or pyrrole_like
                    elif atom_type == int(CNOF_ATOM_TYPES[2]):
                        allowed = (
                            internal_valence == 2
                            and external_valence == 0
                            and hydrogens == 0
                        )
                    else:
                        allowed = False
                mask[member_index, atom_index] |= allowed
    mask.setflags(write=False)
    return mask


def ring_system_placement_has_local_atom_support(
    state: MolecularGraph,
    placement: RingSystemPlacement,
) -> bool:
    """Whether every installed member has at least one locally legal label."""

    mask = ring_system_placement_local_atom_type_mask(state, placement)
    return bool(mask.any(axis=1).all())


@lru_cache(maxsize=4096)
def _matching_edge_sets(
    edges: tuple[tuple[int, int], ...],
) -> tuple[frozenset[frozenset[int]], ...]:
    """Enumerate every single/double Kekule matching candidate."""

    normalized = tuple(sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in edges))
    if not normalized:
        return (frozenset(),)
    results: set[frozenset[frozenset[int]]] = set()

    def search(
        offset: int,
        used: frozenset[int],
        selected: tuple[frozenset[int], ...],
    ) -> None:
        if offset == len(normalized):
            results.add(frozenset(selected))
            return
        a, b = normalized[offset]
        search(offset + 1, used, selected)
        if a not in used and b not in used:
            search(
                offset + 1,
                used | frozenset((a, b)),
                (*selected, frozenset((a, b))),
            )

    search(0, frozenset(), ())
    return tuple(
        sorted(
            results,
            key=lambda matching: tuple(sorted(tuple(sorted(edge)) for edge in matching)),
        )
    )


def enumerate_ring_system_placement_resonance_aliases(
    placement: RingSystemPlacement,
) -> tuple[RingSystemPlacement, ...]:
    """Expand a semantic aromatic placement into all Kekule matching aliases.

    A structured template catalog may not observe every legal resonance
    lowering needed by a held-out scaffold.  Aromatic edges are semantic class
    four in the molecular state, but the atomic executor commits them through
    an explicit single/double-bond lowering.  Enumerating maximum matchings
    supplies every such lowering without treating resonance as a new chemical
    action or memorizing validation atom labels.
    """

    aromatic_edges = tuple(
        sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in placement.aromatic_edges)
    )
    if not aromatic_edges:
        return (placement,)
    aromatic = {frozenset(edge) for edge in aromatic_edges}
    source_orders = {
        frozenset((int(bond.a), int(bond.b))): int(bond.order)
        for bond in placement.scaffold_bonds
    }
    inserted_edges = {
        frozenset((int(bond.a), int(bond.b))) for bond in placement.bond_insertions
    }
    target_orders = dict(source_orders)
    target_orders.update(
        {
            frozenset((int(change.a), int(change.b))): int(change.new_order)
            for change in placement.bond_reorders
        }
    )
    target_orders.update(
        {
            frozenset((int(bond.a), int(bond.b))): int(bond.order)
            for bond in placement.bond_insertions
        }
    )
    if not aromatic.issubset(target_orders):
        raise ValueError("aromatic placement edges are missing from its target graph")

    aliases = set()
    for double_edges in _matching_edge_sets(aromatic_edges):
        orders = dict(target_orders)
        for edge in aromatic:
            orders[edge] = 2 if edge in double_edges else 1
        bond_reorders = tuple(
            sorted(
                BondOrderChange(min(edge), max(edge), orders[edge])
                for edge, source_order in source_orders.items()
                if orders[edge] != source_order
            )
        )
        bond_insertions = tuple(
            sorted(
                RingBond(min(edge), max(edge), orders[edge])
                for edge in inserted_edges
            )
        )
        aliases.add(
            RingSystemPlacement(
                system_atoms=placement.system_atoms,
                scaffold_bonds=placement.scaffold_bonds,
                bond_reorders=bond_reorders,
                bond_insertions=bond_insertions,
                aromatic_edges=placement.aromatic_edges,
                topology_class=placement.topology_class,
            )
        )
    return tuple(sorted(aliases, key=_placement_sort_key))


def ring_system_placement(action: RingSystemGrow) -> RingSystemPlacement:
    """Discard atom labels while retaining one complete coordinated ring mark."""

    if action.interface_atoms or action.atom_insertions:
        raise ValueError("v1 structured placements require an existing acyclic scaffold")
    return RingSystemPlacement(
        system_atoms=tuple(sorted(int(slot) for slot in action.system_atoms)),
        scaffold_bonds=tuple(sorted(action.scaffold_bonds)),
        bond_reorders=tuple(sorted(action.bond_reorders)),
        bond_insertions=tuple(sorted(action.bond_insertions)),
        aromatic_edges=tuple(sorted(action.aromatic_edges)),
        topology_class=str(action.topology_class),
    )


def ring_system_placement_key(placement: RingSystemPlacement) -> tuple:
    """Return a resonance-invariant key for one placed complete ring system."""

    aromatic = {frozenset((int(a), int(b))) for a, b in placement.aromatic_edges}
    target_orders = {
        frozenset((int(bond.a), int(bond.b))): int(bond.order) for bond in placement.scaffold_bonds
    }
    target_orders.update(
        {
            frozenset((int(change.a), int(change.b))): int(change.new_order)
            for change in placement.bond_reorders
        }
    )
    target_orders.update(
        {
            frozenset((int(bond.a), int(bond.b))): int(bond.order)
            for bond in placement.bond_insertions
        }
    )
    scaffold_edges = tuple(
        sorted(
            (min(int(bond.a), int(bond.b)), max(int(bond.a), int(bond.b)))
            for bond in placement.scaffold_bonds
        )
    )
    target_edges = tuple(
        sorted(
            (
                min(edge),
                max(edge),
                4 if edge in aromatic else int(order),
            )
            for edge, order in target_orders.items()
        )
    )
    return (
        placement.system_atoms,
        scaffold_edges,
        target_edges,
        placement.topology_class,
    )


def ring_system_grow_electronic_key(action: RingSystemGrow) -> tuple:
    """Exact resonance-invariant semantic key for one complete ring grow."""

    placement = ring_system_placement(action)
    final_states = tuple(
        (atom_type, charge, hydrogens)
        for _, atom_type, charge, hydrogens
        in ring_system_grow_final_atom_states(action)
    )
    return ring_system_placement_key(placement), final_states


def _action_pattern_graph(action: RingSystemGrow) -> nx.Graph:
    members = tuple(int(slot) for slot in action.system_atoms)
    graph = nx.Graph()
    graph.add_nodes_from(members)
    aromatic = {frozenset((int(a), int(b))) for a, b in action.aromatic_edges}
    target_orders = {
        frozenset((int(bond.a), int(bond.b))): int(bond.order) for bond in action.scaffold_bonds
    }
    target_orders.update(
        {
            frozenset((int(change.a), int(change.b))): int(change.new_order)
            for change in action.bond_reorders
        }
    )
    inserted = {frozenset((int(bond.a), int(bond.b))) for bond in action.bond_insertions}
    target_orders.update(
        {frozenset((int(bond.a), int(bond.b))): int(bond.order) for bond in action.bond_insertions}
    )
    for edge, order in target_orders.items():
        a, b = tuple(edge)
        semantic_order = BOND_AROMATIC if edge in aromatic else int(order)
        role = "inserted" if edge in inserted else "scaffold"
        graph.add_edge(a, b, color=f"{role}:{semantic_order}")
    return graph


def _template_pattern_graph(template: RingSystemTemplate) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(template.span))
    aromatic = {frozenset((int(a), int(b))) for a, b in template.target_aromatic_edges}
    inserted = {frozenset((int(a), int(b))) for a, b, _ in template.inserted_bonds}
    for a, b, order in template.target_bonds:
        edge = frozenset((int(a), int(b)))
        semantic_order = BOND_AROMATIC if edge in aromatic else int(order)
        role = "inserted" if edge in inserted else "scaffold"
        graph.add_edge(int(a), int(b), color=f"{role}:{semantic_order}")
    return graph


def _template_semantic_target_graph(template: RingSystemTemplate) -> nx.Graph:
    """Observable target topology, quotienting scaffold-versus-inserted roles."""

    graph = nx.Graph()
    graph.add_nodes_from(range(template.span))
    aromatic = {frozenset((int(a), int(b))) for a, b in template.target_aromatic_edges}
    for a, b, order in template.target_bonds:
        edge = frozenset((int(a), int(b)))
        semantic_order = BOND_AROMATIC if edge in aromatic else int(order)
        graph.add_edge(int(a), int(b), color=int(semantic_order))
    return graph


def _pattern_bucket_key(graph: nx.Graph, topology_class: str) -> tuple:
    return (
        graph.number_of_nodes(),
        str(topology_class),
        nx.weisfeiler_lehman_graph_hash(graph, edge_attr="color"),
    )


@lru_cache(maxsize=16)
def _structured_template_index(
    templates: tuple[RingSystemTemplate, ...],
) -> tuple[
    tuple[RingSystemTemplate, ...],
    dict[tuple, tuple[tuple[RingSystemTemplate, nx.Graph], ...]],
]:
    """Deduplicate typed/resonance aliases and bucket exact topology checks."""

    mutable: dict[tuple, list[tuple[RingSystemTemplate, nx.Graph]]] = {}
    representatives = []
    for template in templates:
        graph = _template_pattern_graph(template)
        key = _pattern_bucket_key(graph, template.topology_class)
        bucket = mutable.setdefault(key, [])
        if any(
            nx.is_isomorphic(
                graph,
                existing_graph,
                edge_match=lambda left, right: left["color"] == right["color"],
            )
            for _, existing_graph in bucket
        ):
            continue
        bucket.append((template, graph))
        representatives.append(template)
    return (
        tuple(representatives),
        {key: tuple(items) for key, items in mutable.items()},
    )


@lru_cache(maxsize=16)
def _structured_delete_index(
    templates: tuple[RingSystemTemplate, ...],
) -> dict[tuple[int, str], tuple[tuple[RingSystemTemplate, nx.Graph], ...]]:
    """Index inverse rules by the observable complete cyclic component."""

    mutable: dict[tuple[int, str], list[tuple[RingSystemTemplate, nx.Graph]]] = {}
    for template in templates:
        graph = _template_semantic_target_graph(template)
        key = (
            graph.number_of_nodes(),
            nx.weisfeiler_lehman_graph_hash(graph, edge_attr="color"),
        )
        mutable.setdefault(key, []).append((template, graph))
    return {key: tuple(items) for key, items in mutable.items()}


_STRUCTURED_TEMPLATE_INDEX_ID_CACHE: dict[
    int,
    tuple[
        tuple[RingSystemTemplate, ...],
        tuple[
            tuple[RingSystemTemplate, ...],
            dict[tuple, tuple[tuple[RingSystemTemplate, nx.Graph], ...]],
        ],
    ],
] = {}
_STRUCTURED_DELETE_INDEX_ID_CACHE: dict[
    int,
    tuple[
        tuple[RingSystemTemplate, ...],
        dict[tuple[int, str], tuple[tuple[RingSystemTemplate, nx.Graph], ...]],
    ],
] = {}


def _structured_template_index_by_identity(
    templates: tuple[RingSystemTemplate, ...],
) -> tuple[
    tuple[RingSystemTemplate, ...],
    dict[tuple, tuple[tuple[RingSystemTemplate, nx.Graph], ...]],
]:
    key = id(templates)
    cached = _STRUCTURED_TEMPLATE_INDEX_ID_CACHE.get(key)
    if cached is not None and cached[0] is templates:
        return cached[1]
    index = _structured_template_index(templates)
    if len(_STRUCTURED_TEMPLATE_INDEX_ID_CACHE) >= 16:
        _STRUCTURED_TEMPLATE_INDEX_ID_CACHE.pop(next(iter(_STRUCTURED_TEMPLATE_INDEX_ID_CACHE)))
    _STRUCTURED_TEMPLATE_INDEX_ID_CACHE[key] = (templates, index)
    return index


def _structured_delete_index_by_identity(
    templates: tuple[RingSystemTemplate, ...],
) -> dict[tuple[int, str], tuple[tuple[RingSystemTemplate, nx.Graph], ...]]:
    key = id(templates)
    cached = _STRUCTURED_DELETE_INDEX_ID_CACHE.get(key)
    if cached is not None and cached[0] is templates:
        return cached[1]
    index = _structured_delete_index(templates)
    if len(_STRUCTURED_DELETE_INDEX_ID_CACHE) >= 16:
        _STRUCTURED_DELETE_INDEX_ID_CACHE.pop(next(iter(_STRUCTURED_DELETE_INDEX_ID_CACHE)))
    _STRUCTURED_DELETE_INDEX_ID_CACHE[key] = (templates, index)
    return index


def structured_ring_system_templates(
    catalog: TypedRingCatalog,
) -> tuple[RingSystemTemplate, ...]:
    """Return resonance/label-quotiented production ring-system rules."""

    return _structured_catalog_views(catalog)[0]


@lru_cache(maxsize=16)
def _structured_template_alias_groups(
    templates: tuple[RingSystemTemplate, ...],
) -> tuple[tuple[RingSystemTemplate, ...], ...]:
    """Retain executable Kekule/lowering aliases under each semantic rule."""

    representatives, buckets = _structured_template_index_by_identity(templates)
    index_by_template = {template: index for index, template in enumerate(representatives)}
    groups: list[list[RingSystemTemplate]] = [[] for _ in representatives]
    for template in templates:
        graph = _template_pattern_graph(template)
        key = _pattern_bucket_key(graph, template.topology_class)
        for representative, representative_graph in buckets.get(key, ()):
            if nx.is_isomorphic(
                graph,
                representative_graph,
                edge_match=lambda left, right: left["color"] == right["color"],
            ):
                groups[index_by_template[representative]].append(template)
                break
        else:
            raise RuntimeError("structured ring alias has no semantic representative")
    return tuple(tuple(group) for group in groups)


def structured_ring_system_template_aliases(
    catalog: TypedRingCatalog,
) -> tuple[tuple[RingSystemTemplate, ...], ...]:
    """Return executable aliases aligned with structured semantic templates."""

    return _structured_catalog_views(catalog)[1]


_STRUCTURED_ELECTRONIC_ALIAS_VIEW_CACHE: dict[
    int,
    tuple[
        TypedRingCatalog,
        tuple[tuple[tuple[RingSystemElectronicAlias, int], ...], ...],
    ],
] = {}


def structured_ring_system_electronic_aliases(
    catalog: TypedRingCatalog,
) -> tuple[tuple[tuple[RingSystemElectronicAlias, int], ...], ...]:
    """Return paired electronic aliases aligned with semantic templates."""

    key = id(catalog)
    cached = _STRUCTURED_ELECTRONIC_ALIAS_VIEW_CACHE.get(key)
    if cached is not None and cached[0] is catalog:
        return cached[1]
    items = require_ring_system_electronic_aliases(catalog)
    representatives, buckets = _structured_template_index_by_identity(
        catalog.ring_system_templates
    )
    index_by_template = {template: index for index, template in enumerate(representatives)}
    groups: list[list[tuple[RingSystemElectronicAlias, int]]] = [
        [] for _ in representatives
    ]
    for alias, count in items:
        graph = _template_pattern_graph(alias.pattern)
        bucket_key = _pattern_bucket_key(graph, alias.pattern.topology_class)
        for representative, representative_graph in buckets.get(bucket_key, ()):
            if nx.is_isomorphic(
                graph,
                representative_graph,
                edge_match=lambda left, right: left["color"] == right["color"],
            ):
                groups[index_by_template[representative]].append((alias, int(count)))
                break
        else:
            raise RuntimeError("ring electronic alias has no selected semantic template")
    result = tuple(
        tuple(sorted(group, key=lambda item: (-item[1], item[0]))) for group in groups
    )
    if len(_STRUCTURED_ELECTRONIC_ALIAS_VIEW_CACHE) >= 16:
        _STRUCTURED_ELECTRONIC_ALIAS_VIEW_CACHE.pop(
            next(iter(_STRUCTURED_ELECTRONIC_ALIAS_VIEW_CACHE))
        )
    _STRUCTURED_ELECTRONIC_ALIAS_VIEW_CACHE[key] = (catalog, result)
    return result


_STRUCTURED_CATALOG_VIEW_CACHE: dict[
    int,
    tuple[
        TypedRingCatalog,
        tuple[RingSystemTemplate, ...],
        tuple[tuple[RingSystemTemplate, ...], ...],
    ],
] = {}


def _structured_catalog_views(
    catalog: TypedRingCatalog,
) -> tuple[
    tuple[RingSystemTemplate, ...],
    tuple[tuple[RingSystemTemplate, ...], ...],
]:
    """Identity-cache large catalog views without rehashing 4k templates."""

    key = id(catalog)
    cached = _STRUCTURED_CATALOG_VIEW_CACHE.get(key)
    if cached is not None and cached[0] is catalog:
        return cached[1], cached[2]
    templates, _ = _structured_template_index_by_identity(catalog.ring_system_templates)
    aliases = _structured_template_alias_groups(catalog.ring_system_templates)
    if len(_STRUCTURED_CATALOG_VIEW_CACHE) >= 16:
        _STRUCTURED_CATALOG_VIEW_CACHE.pop(next(iter(_STRUCTURED_CATALOG_VIEW_CACHE)))
    _STRUCTURED_CATALOG_VIEW_CACHE[key] = (catalog, templates, aliases)
    return templates, aliases


def _source_topology_groups(
    templates: tuple[RingSystemTemplate, ...],
) -> tuple[tuple[nx.Graph, tuple[int, ...], tuple[int, int, int, int, int]], ...]:
    """Group rules that share the same acyclic scaffold match problem."""

    cache_key = id(templates)
    cached = _SOURCE_TOPOLOGY_GROUP_CACHE.get(cache_key)
    if cached is not None and cached[0] is templates:
        return cached[1]

    mutable: dict[tuple[int, tuple[int, ...]], list[tuple[nx.Graph, list[int]]]] = {}
    for index, template in enumerate(templates):
        pattern = _template_graph(
            template,
            target=False,
            include_atom_labels=False,
        )
        if not nx.is_tree(pattern):
            raise ValueError("structured ring scaffolds must be trees")
        key = (
            pattern.number_of_nodes(),
            tuple(sorted(int(degree) for _, degree in pattern.degree())),
        )
        bucket = mutable.setdefault(key, [])
        for representative, indices in bucket:
            if nx.is_isomorphic(pattern, representative):
                indices.append(index)
                break
        else:
            bucket.append((pattern, [index]))
    groups = tuple(
        (
            pattern,
            tuple(indices),
            (
                pattern.number_of_nodes(),
                sum(int(degree >= 2) for _, degree in pattern.degree()),
                sum(int(degree >= 3) for _, degree in pattern.degree()),
                sum(int(degree >= 4) for _, degree in pattern.degree()),
                nx.diameter(pattern),
            ),
        )
        for bucket_key in sorted(mutable)
        for pattern, indices in mutable[bucket_key]
    )
    if len(_SOURCE_TOPOLOGY_GROUP_CACHE) >= 16:
        _SOURCE_TOPOLOGY_GROUP_CACHE.pop(next(iter(_SOURCE_TOPOLOGY_GROUP_CACHE)))
    _SOURCE_TOPOLOGY_GROUP_CACHE[cache_key] = (templates, groups)
    return groups


def _local_support_pattern(template: RingSystemTemplate) -> nx.Graph:
    """Decorate a source tree by a resonance-safe valence lower bound."""

    pattern = _template_graph(
        template,
        target=False,
        include_atom_labels=False,
    )
    if not nx.is_tree(pattern):
        raise ValueError("structured ring scaffolds must be trees")
    source_valence = {int(node): 0 for node in pattern.nodes()}
    target_valence = {int(node): 0 for node in pattern.nodes()}
    for a, b, order in template.source_bonds:
        contribution = int(BOND_CLASS_TO_H_CHANGE[int(order)])
        source_valence[int(a)] += contribution
        source_valence[int(b)] += contribution
    aromatic = {
        frozenset((int(a), int(b))) for a, b in template.target_aromatic_edges
    }
    for a, b, order in template.target_bonds:
        edge = frozenset((int(a), int(b)))
        # The observed template stores one Kekule lowering, but a held-out
        # scaffold may require a different resonance alias.  A single bond is
        # the exact lower bound for every aromatic edge; executor-verified
        # placement aliases enforce the eventual joint electronic state.
        semantic_order = 1 if edge in aromatic else int(order)
        contribution = int(BOND_CLASS_TO_H_CHANGE[semantic_order])
        target_valence[int(a)] += contribution
        target_valence[int(b)] += contribution
    for node in pattern.nodes():
        increment = target_valence[int(node)] - source_valence[int(node)]
        pattern.nodes[int(node)]["color"] = str(int(increment))
        pattern.nodes[int(node)]["valence_increment"] = int(increment)
    return pattern


def _source_local_support_groups(
    templates: tuple[RingSystemTemplate, ...],
    aliases: tuple[tuple[RingSystemTemplate, ...], ...],
) -> tuple[tuple[nx.Graph, tuple[int, ...], tuple[int, int, int, int, int]], ...]:
    """Group alias patterns with the same local-capacity match problem."""

    if len(templates) != len(aliases):
        raise ValueError("ring template aliases do not align with representatives")
    cache_key = (id(templates), id(aliases))
    cached = _SOURCE_LOCAL_SUPPORT_GROUP_CACHE.get(cache_key)
    if cached is not None and cached[0] is templates and cached[1] is aliases:
        return cached[2]

    mutable: dict[
        tuple[int, tuple[int, ...], tuple[str, ...], str],
        list[tuple[nx.Graph, list[int]]],
    ] = {}
    for index, template_aliases in enumerate(aliases):
        for template in template_aliases:
            pattern = _local_support_pattern(template)
            colors = tuple(sorted(str(pattern.nodes[node]["color"]) for node in pattern.nodes()))
            key = (
                pattern.number_of_nodes(),
                tuple(sorted(int(degree) for _, degree in pattern.degree())),
                colors,
                nx.weisfeiler_lehman_graph_hash(pattern, node_attr="color"),
            )
            bucket = mutable.setdefault(key, [])
            for representative, indices in bucket:
                if nx.is_isomorphic(
                    pattern,
                    representative,
                    node_match=lambda left, right: left["color"] == right["color"],
                ):
                    if index not in indices:
                        indices.append(index)
                    break
            else:
                bucket.append((pattern, [index]))
    groups = tuple(
        (
            pattern,
            tuple(sorted(indices)),
            (
                pattern.number_of_nodes(),
                sum(int(degree >= 2) for _, degree in pattern.degree()),
                sum(int(degree >= 3) for _, degree in pattern.degree()),
                sum(int(degree >= 4) for _, degree in pattern.degree()),
                nx.diameter(pattern),
            ),
        )
        for bucket_key in sorted(mutable)
        for pattern, indices in mutable[bucket_key]
    )
    if len(_SOURCE_LOCAL_SUPPORT_GROUP_CACHE) >= 16:
        _SOURCE_LOCAL_SUPPORT_GROUP_CACHE.pop(
            next(iter(_SOURCE_LOCAL_SUPPORT_GROUP_CACHE))
        )
    _SOURCE_LOCAL_SUPPORT_GROUP_CACHE[cache_key] = (templates, aliases, groups)
    return groups


def _local_rooted_support_index(
    templates: tuple[RingSystemTemplate, ...],
    aliases: tuple[tuple[RingSystemTemplate, ...], ...],
) -> _RootedTreeSupportIndex:
    cache_key = (id(templates), id(aliases))
    cached = _LOCAL_ROOTED_SUPPORT_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] is templates and cached[1] is aliases:
        return cached[2]
    index = _build_rooted_tree_support_index(
        _source_local_support_groups(templates, aliases),
        constraint_for_node=lambda pattern, node: (
            int(pattern.nodes[int(node)]["valence_increment"]),
        ),
    )
    if len(_LOCAL_ROOTED_SUPPORT_INDEX_CACHE) >= 16:
        _LOCAL_ROOTED_SUPPORT_INDEX_CACHE.pop(
            next(iter(_LOCAL_ROOTED_SUPPORT_INDEX_CACHE))
        )
    _LOCAL_ROOTED_SUPPORT_INDEX_CACHE[cache_key] = (templates, aliases, index)
    return index


def _electronic_support_pattern(alias: RingSystemElectronicAlias) -> nx.Graph:
    """Decorate a source tree with one fixed observed electronic assignment."""

    pattern = _template_graph(
        alias.pattern,
        target=False,
        include_atom_labels=False,
    )
    if not nx.is_tree(pattern):
        raise ValueError("structured ring electronic scaffolds must be trees")
    source_valence = {int(node): 0 for node in pattern.nodes()}
    target_valence = {int(node): 0 for node in pattern.nodes()}
    for a, b, order in alias.pattern.source_bonds:
        contribution = int(BOND_CLASS_TO_H_CHANGE[int(order)])
        source_valence[int(a)] += contribution
        source_valence[int(b)] += contribution
    for a, b, order in alias.pattern.target_bonds:
        contribution = int(BOND_CLASS_TO_H_CHANGE[int(order)])
        target_valence[int(a)] += contribution
        target_valence[int(b)] += contribution
    aromatic_nodes = {
        int(node)
        for edge in alias.pattern.target_aromatic_edges
        for node in edge
    }
    for node, atom_state in enumerate(alias.target_atoms):
        atom_type, charge, target_hydrogens = (int(value) for value in atom_state)
        attributes = (
            atom_type,
            charge,
            source_valence[node],
            target_valence[node],
            int(node in aromatic_nodes),
            target_hydrogens,
        )
        pattern.nodes[node]["color"] = repr(attributes)
        pattern.nodes[node]["atom_type"] = atom_type
        pattern.nodes[node]["charge"] = charge
        pattern.nodes[node]["source_valence"] = source_valence[node]
        pattern.nodes[node]["target_valence"] = target_valence[node]
        pattern.nodes[node]["aromatic"] = int(node in aromatic_nodes)
        pattern.nodes[node]["target_hydrogens"] = target_hydrogens
    return pattern


def _source_electronic_support_groups(
    templates: tuple[RingSystemTemplate, ...],
    electronic_aliases: tuple[
        tuple[tuple[RingSystemElectronicAlias, int], ...], ...
    ],
) -> tuple[tuple[nx.Graph, tuple[int, ...], tuple[int, int, int, int, int]], ...]:
    """Group fixed electronic aliases with identical decorated tree support."""

    if len(templates) != len(electronic_aliases):
        raise ValueError("ring electronic aliases do not align with representatives")
    cache_key = (id(templates), id(electronic_aliases))
    cached = _SOURCE_ELECTRONIC_SUPPORT_GROUP_CACHE.get(cache_key)
    if (
        cached is not None
        and cached[0] is templates
        and cached[1] is electronic_aliases
    ):
        return cached[2]
    mutable: dict[
        tuple[int, tuple[int, ...], tuple[str, ...], str],
        list[tuple[nx.Graph, list[int]]],
    ] = {}
    for index, aliases in enumerate(electronic_aliases):
        for alias, _ in aliases:
            pattern = _electronic_support_pattern(alias)
            colors = tuple(sorted(str(pattern.nodes[node]["color"]) for node in pattern.nodes()))
            key = (
                pattern.number_of_nodes(),
                tuple(sorted(int(degree) for _, degree in pattern.degree())),
                colors,
                nx.weisfeiler_lehman_graph_hash(pattern, node_attr="color"),
            )
            bucket = mutable.setdefault(key, [])
            for representative, indices in bucket:
                if nx.is_isomorphic(
                    pattern,
                    representative,
                    node_match=lambda left, right: left["color"] == right["color"],
                ):
                    if index not in indices:
                        indices.append(index)
                    break
            else:
                bucket.append((pattern, [index]))
    groups = tuple(
        (
            pattern,
            tuple(sorted(indices)),
            (
                pattern.number_of_nodes(),
                sum(int(degree >= 2) for _, degree in pattern.degree()),
                sum(int(degree >= 3) for _, degree in pattern.degree()),
                sum(int(degree >= 4) for _, degree in pattern.degree()),
                nx.diameter(pattern),
            ),
        )
        for bucket_key in sorted(mutable)
        for pattern, indices in mutable[bucket_key]
    )
    if len(_SOURCE_ELECTRONIC_SUPPORT_GROUP_CACHE) >= 16:
        _SOURCE_ELECTRONIC_SUPPORT_GROUP_CACHE.pop(
            next(iter(_SOURCE_ELECTRONIC_SUPPORT_GROUP_CACHE))
        )
    _SOURCE_ELECTRONIC_SUPPORT_GROUP_CACHE[cache_key] = (
        templates,
        electronic_aliases,
        groups,
    )
    return groups


def _electronic_rooted_support_index(
    templates: tuple[RingSystemTemplate, ...],
    electronic_aliases: tuple[
        tuple[tuple[RingSystemElectronicAlias, int], ...], ...
    ],
) -> _RootedTreeSupportIndex:
    cache_key = (id(templates), id(electronic_aliases))
    cached = _ELECTRONIC_ROOTED_SUPPORT_INDEX_CACHE.get(cache_key)
    if (
        cached is not None
        and cached[0] is templates
        and cached[1] is electronic_aliases
    ):
        return cached[2]

    def constraint(pattern: nx.Graph, node: int) -> tuple[int, ...]:
        attributes = pattern.nodes[int(node)]
        return (
            int(attributes["atom_type"]),
            int(attributes["charge"]),
            int(attributes["source_valence"]),
            int(attributes["target_valence"]),
            int(attributes["aromatic"]),
            int(attributes["target_hydrogens"]),
        )

    index = _build_rooted_tree_support_index(
        _source_electronic_support_groups(templates, electronic_aliases),
        constraint_for_node=constraint,
    )
    if len(_ELECTRONIC_ROOTED_SUPPORT_INDEX_CACHE) >= 16:
        _ELECTRONIC_ROOTED_SUPPORT_INDEX_CACHE.pop(
            next(iter(_ELECTRONIC_ROOTED_SUPPORT_INDEX_CACHE))
        )
    _ELECTRONIC_ROOTED_SUPPORT_INDEX_CACHE[cache_key] = (
        templates,
        electronic_aliases,
        index,
    )
    return index


_SOURCE_TOPOLOGY_GROUP_CACHE: dict[
    int,
    tuple[
        tuple[RingSystemTemplate, ...],
        tuple[tuple[nx.Graph, tuple[int, ...], tuple[int, int, int, int, int]], ...],
    ],
] = {}

_RING_SUPPORT_MASK_CACHE: dict[
    tuple[int, tuple[str, ...]],
    tuple[tuple[RingSystemTemplate, ...], np.ndarray],
] = {}

_RING_LOCAL_SUPPORT_MASK_CACHE: dict[
    tuple[int, int, tuple[str, ...]],
    tuple[
        tuple[RingSystemTemplate, ...],
        tuple[tuple[RingSystemTemplate, ...], ...],
        np.ndarray,
    ],
] = {}

_SOURCE_LOCAL_SUPPORT_GROUP_CACHE: dict[
    tuple[int, int],
    tuple[
        tuple[RingSystemTemplate, ...],
        tuple[tuple[RingSystemTemplate, ...], ...],
        tuple[tuple[nx.Graph, tuple[int, ...], tuple[int, int, int, int, int]], ...],
    ],
] = {}

_RING_ELECTRONIC_SUPPORT_MASK_CACHE: dict[
    tuple[int, int, tuple[str, ...]],
    tuple[
        tuple[RingSystemTemplate, ...],
        tuple[tuple[tuple[RingSystemElectronicAlias, int], ...], ...],
        np.ndarray,
    ],
] = {}

_SOURCE_ELECTRONIC_SUPPORT_GROUP_CACHE: dict[
    tuple[int, int],
    tuple[
        tuple[RingSystemTemplate, ...],
        tuple[tuple[tuple[RingSystemElectronicAlias, int], ...], ...],
        tuple[tuple[nx.Graph, tuple[int, ...], tuple[int, int, int, int, int]], ...],
    ],
] = {}


@dataclass(frozen=True)
class _RootedTreeSupportState:
    """One deduplicated rooted colored-pattern subproblem."""

    constraint: tuple[int, ...]
    children: tuple[int, ...]
    size: int


@dataclass(frozen=True)
class _RootedTreeSupportIndex:
    """Shared exact tree-pattern automaton for a complete template vocabulary."""

    states: tuple[_RootedTreeSupportState, ...]
    root_outputs: tuple[tuple[int, tuple[int, ...]], ...]


_LOCAL_ROOTED_SUPPORT_INDEX_CACHE: dict[
    tuple[int, int],
    tuple[
        tuple[RingSystemTemplate, ...],
        tuple[tuple[RingSystemTemplate, ...], ...],
        _RootedTreeSupportIndex,
    ],
] = {}

_ELECTRONIC_ROOTED_SUPPORT_INDEX_CACHE: dict[
    tuple[int, int],
    tuple[
        tuple[RingSystemTemplate, ...],
        tuple[tuple[tuple[RingSystemElectronicAlias, int], ...], ...],
        _RootedTreeSupportIndex,
    ],
] = {}


def _build_rooted_tree_support_index(
    groups: tuple[
        tuple[nx.Graph, tuple[int, ...], tuple[int, int, int, int, int]],
        ...,
    ],
    *,
    constraint_for_node: Callable[[nx.Graph, int], tuple[int, ...]],
) -> _RootedTreeSupportIndex:
    """Hash-cons every rooted pattern branch across all support groups."""

    state_ids: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    states: list[_RootedTreeSupportState] = []

    def encode(pattern: nx.Graph, node: int, parent: int) -> int:
        children = tuple(
            sorted(
                encode(pattern, int(child), int(node))
                for child in pattern.neighbors(node)
                if int(child) != parent
            )
        )
        constraint = tuple(int(value) for value in constraint_for_node(pattern, node))
        key = (constraint, children)
        identifier = state_ids.get(key)
        if identifier is None:
            identifier = len(states)
            state_ids[key] = identifier
            states.append(
                _RootedTreeSupportState(
                    constraint=constraint,
                    children=children,
                    size=1 + sum(states[child].size for child in children),
                )
            )
        return identifier

    outputs: dict[int, set[int]] = {}
    for pattern, indices, _ in groups:
        root = min(
            (int(node) for node in pattern.nodes()),
            key=lambda node: (-int(pattern.degree[node]), node),
        )
        root_id = encode(pattern, root, -1)
        outputs.setdefault(root_id, set()).update(int(index) for index in indices)
    return _RootedTreeSupportIndex(
        states=tuple(states),
        root_outputs=tuple(
            (root_id, tuple(sorted(indices)))
            for root_id, indices in sorted(outputs.items())
        ),
    )


def _rooted_tree_support_mask(
    host: nx.Graph,
    *,
    width: int,
    index: _RootedTreeSupportIndex,
    compatible: Callable[[tuple[int, ...], int], bool],
) -> np.ndarray:
    """Evaluate all deduplicated tree patterns together by exact rooted DP."""

    mask = np.zeros(int(width), dtype=np.bool_)
    if not len(index.root_outputs) or host.number_of_nodes() == 0:
        return mask
    host_neighbors = {
        int(node): tuple(sorted(int(neighbor) for neighbor in host.neighbors(node)))
        for node in host.nodes()
    }

    @lru_cache(maxsize=None)
    def matches(pattern_id: int, host_node: int, host_parent: int) -> bool:
        pattern = index.states[int(pattern_id)]
        if not compatible(pattern.constraint, int(host_node)):
            return False
        host_children = tuple(
            child for child in host_neighbors[int(host_node)] if child != int(host_parent)
        )
        if len(pattern.children) > len(host_children):
            return False
        candidate_masks = []
        for pattern_child in pattern.children:
            candidates = 0
            for offset, host_child in enumerate(host_children):
                if matches(pattern_child, host_child, int(host_node)):
                    candidates |= 1 << offset
            if candidates == 0:
                return False
            candidate_masks.append(candidates)
        candidate_masks.sort(key=int.bit_count)

        @lru_cache(maxsize=None)
        def assign(position: int, used: int) -> bool:
            if position == len(candidate_masks):
                return True
            available = candidate_masks[position] & ~used
            while available:
                selected = available & -available
                available ^= selected
                if assign(position + 1, used | selected):
                    return True
            return False

        return assign(0, 0)

    host_nodes = tuple(sorted(int(node) for node in host.nodes()))
    for root_id, outputs in index.root_outputs:
        if index.states[root_id].size > host.number_of_nodes():
            continue
        if any(matches(root_id, host_node, -1) for host_node in host_nodes):
            mask[np.asarray(outputs, dtype=np.int64)] = True
    return mask


def _canonical_tree_code(tree: nx.Graph) -> str:
    """Exact AHU-style isomorphism code for one unlabeled tree."""

    if tree.number_of_nodes() == 0:
        return ""

    def rooted(node: int, parent: int) -> str:
        children = sorted(
            rooted(int(child), node) for child in tree.neighbors(node) if int(child) != parent
        )
        return "(" + "".join(children) + ")"

    return min(rooted(int(center), -1) for center in nx.center(tree))


def _forest_topology_key(host: nx.Graph) -> tuple[str, ...]:
    return tuple(
        sorted(
            _canonical_tree_code(host.subgraph(nodes)) for nodes in nx.connected_components(host)
        )
    )


def _canonical_labeled_tree_code(
    tree: nx.Graph,
    labels: dict[int, int],
) -> str:
    """Exact AHU-style code for a node-labeled unrooted tree."""

    if tree.number_of_nodes() == 0:
        return ""

    def rooted(node: int, parent: int) -> str:
        children = sorted(
            rooted(int(child), node)
            for child in tree.neighbors(node)
            if int(child) != parent
        )
        return f"({int(labels[node])}|{''.join(children)})"

    return min(rooted(int(center), -1) for center in nx.center(tree))


def _forest_local_support_key(
    host: nx.Graph,
    state: MolecularGraph,
) -> tuple[str, ...]:
    valences = {
        int(node): sum(
            int(BOND_CLASS_TO_H_CHANGE[int(state.bonds[int(node), neighbor])])
            for neighbor in np.flatnonzero(state.bonds[int(node)] != 0)
        )
        for node in host.nodes()
    }
    return tuple(
        sorted(
            _canonical_labeled_tree_code(host.subgraph(nodes), valences)
            for nodes in nx.connected_components(host)
        )
    )


def warm_ring_system_candidate_indices(catalog: TypedRingCatalog) -> None:
    """Build immutable topology indices once before DataLoader workers fork."""

    templates, aliases = _structured_catalog_views(catalog)
    _source_topology_groups(templates)
    _source_local_support_groups(templates, aliases)
    _local_rooted_support_index(templates, aliases)
    if int(getattr(catalog, "ring_system_electronic_alias_version", 0)) >= 1:
        structured_ring_system_electronic_aliases(catalog)
    _structured_delete_index_by_identity(catalog.ring_system_templates)


def _eligible_grow_host_graph(state: MolecularGraph) -> nx.Graph:
    """Carbon, neutral, acyclic single-bond support for v1 ring installation."""

    full = _state_graph(state, include_atom_labels=False)
    cyclic_atoms = _cyclic_atoms(full)
    carbon = int(ELEMENT_TO_IDX["C"])
    eligible = {
        int(slot)
        for slot in full.nodes()
        if int(slot) not in cyclic_atoms
        and int(state.atom_types[int(slot)]) == carbon
        and int(state.formal_charges[int(slot)]) == 0
    }
    host = nx.Graph()
    host.add_nodes_from(sorted(eligible))
    host.add_edges_from(
        (int(a), int(b))
        for a, b in full.edges()
        if int(a) in eligible and int(b) in eligible and int(state.bonds[int(a), int(b)]) == 1
    )
    if host.number_of_nodes() and not nx.is_forest(host):
        raise RuntimeError("eligible ring-grow scaffold must be a forest")
    return host


def _forest_contains_tree(
    host: nx.Graph,
    pattern: nx.Graph,
    *,
    node_compatible: Callable[[int, int], bool] | None = None,
) -> bool:
    """Exact bounded-degree tree-subgraph decision by rooted dynamic programming."""

    if pattern.number_of_nodes() > host.number_of_nodes():
        return False
    if pattern.number_of_nodes() == 0:
        return True
    if not nx.is_tree(pattern) or not nx.is_forest(host):
        raise ValueError("tree support requires a tree pattern and forest host")
    pattern_neighbors = {
        int(node): tuple(
            sorted(
                (int(item) for item in pattern.neighbors(node)),
                key=lambda item: (-int(pattern.degree[item]), item),
            )
        )
        for node in pattern.nodes()
    }
    host_neighbors = {
        int(node): tuple(sorted(int(item) for item in host.neighbors(node)))
        for node in host.nodes()
    }

    @lru_cache(maxsize=None)
    def embeds(
        pattern_node: int,
        pattern_parent: int,
        host_node: int,
        host_parent: int,
    ) -> bool:
        if node_compatible is not None and not node_compatible(pattern_node, host_node):
            return False
        pattern_children = tuple(
            child for child in pattern_neighbors[pattern_node] if child != pattern_parent
        )
        host_children = tuple(child for child in host_neighbors[host_node] if child != host_parent)
        if len(pattern_children) > len(host_children):
            return False

        def assign(child_index: int, used: frozenset[int]) -> bool:
            if child_index == len(pattern_children):
                return True
            pattern_child = pattern_children[child_index]
            return any(
                host_child not in used
                and embeds(
                    pattern_child,
                    pattern_node,
                    host_child,
                    host_node,
                )
                and assign(child_index + 1, used | {host_child})
                for host_child in host_children
            )

        return assign(0, frozenset())

    root = min(
        (int(node) for node in pattern.nodes()),
        key=lambda node: (-int(pattern.degree[node]), node),
    )
    return any(embeds(root, -1, int(host_node), -1) for host_node in host.nodes())


def ring_system_template_support_mask(
    state: MolecularGraph,
    templates: tuple[RingSystemTemplate, ...],
) -> np.ndarray:
    """Return exact template embeddability without enumerating every match."""

    host = _eligible_grow_host_graph(state)
    cache_key = (id(templates), _forest_topology_key(host))
    cached = _RING_SUPPORT_MASK_CACHE.get(cache_key)
    if cached is not None and cached[0] is templates:
        return cached[1]
    mask = np.zeros(len(templates), dtype=np.bool_)
    components = []
    for nodes in nx.connected_components(host):
        component = host.subgraph(nodes)
        degrees = tuple(int(degree) for _, degree in component.degree())
        components.append(
            (
                component,
                (
                    component.number_of_nodes(),
                    sum(int(degree >= 2) for degree in degrees),
                    sum(int(degree >= 3) for degree in degrees),
                    sum(int(degree >= 4) for degree in degrees),
                    nx.diameter(component),
                ),
            )
        )
    for pattern, indices, requirements in _source_topology_groups(templates):
        if any(
            all(required <= available for required, available in zip(requirements, capacity))
            and _forest_contains_tree(component, pattern)
            for component, capacity in components
        ):
            mask[np.asarray(indices, dtype=np.int64)] = True
    if len(_RING_SUPPORT_MASK_CACHE) >= 8192:
        _RING_SUPPORT_MASK_CACHE.pop(next(iter(_RING_SUPPORT_MASK_CACHE)))
    mask.setflags(write=False)
    _RING_SUPPORT_MASK_CACHE[cache_key] = (templates, mask)
    return mask


def ring_system_template_local_support_mask(
    state: MolecularGraph,
    templates: tuple[RingSystemTemplate, ...],
    aliases: tuple[tuple[RingSystemTemplate, ...], ...],
) -> np.ndarray:
    """Return exact row-wise C/N/O/F feasibility via decorated tree DP.

    This strengthens topology support without enumerating placements.  It is
    exact for the local atom-label application condition because neutral carbon
    is a witness whenever the installed valence does not exceed four.  It is a
    necessary prefilter, not a claim of whole-action joint electronic support.
    """

    host = _eligible_grow_host_graph(state)
    decorated_key = _forest_local_support_key(host, state)
    cache_key = (id(templates), id(aliases), decorated_key)
    cached = _RING_LOCAL_SUPPORT_MASK_CACHE.get(cache_key)
    if (
        cached is not None
        and cached[0] is templates
        and cached[1] is aliases
    ):
        return cached[2]

    current_valence = {
        int(node): sum(
            int(BOND_CLASS_TO_H_CHANGE[int(state.bonds[int(node), neighbor])])
            for neighbor in np.flatnonzero(state.bonds[int(node)] != 0)
        )
        for node in host.nodes()
    }
    maximum_valence = max(int(value) for value in CNOF_VALENCE.values())
    mask = _rooted_tree_support_mask(
        host,
        width=len(templates),
        index=_local_rooted_support_index(templates, aliases),
        compatible=lambda constraint, host_node: (
            0
            <= current_valence[int(host_node)] + int(constraint[0])
            <= maximum_valence
        ),
    )
    if len(_RING_LOCAL_SUPPORT_MASK_CACHE) >= 8192:
        _RING_LOCAL_SUPPORT_MASK_CACHE.pop(next(iter(_RING_LOCAL_SUPPORT_MASK_CACHE)))
    mask.setflags(write=False)
    _RING_LOCAL_SUPPORT_MASK_CACHE[cache_key] = (templates, aliases, mask)
    return mask


def ring_system_electronic_template_support_mask(
    state: MolecularGraph,
    templates: tuple[RingSystemTemplate, ...],
    electronic_aliases: tuple[
        tuple[tuple[RingSystemElectronicAlias, int], ...], ...
    ],
) -> np.ndarray:
    """Return fixed-alias feasibility without explicit placement enumeration."""

    host = _eligible_grow_host_graph(state)
    decorated_key = _forest_local_support_key(host, state)
    cache_key = (id(templates), id(electronic_aliases), decorated_key)
    cached = _RING_ELECTRONIC_SUPPORT_MASK_CACHE.get(cache_key)
    if (
        cached is not None
        and cached[0] is templates
        and cached[1] is electronic_aliases
    ):
        return cached[2]
    current_valence = {
        int(node): sum(
            int(BOND_CLASS_TO_H_CHANGE[int(state.bonds[int(node), neighbor])])
            for neighbor in np.flatnonzero(state.bonds[int(node)] != 0)
        )
        for node in host.nodes()
    }

    def compatible(constraint: tuple[int, ...], host_node: int) -> bool:
        (
            atom_type,
            charge,
            source_valence,
            target_valence,
            aromatic,
            target_hydrogens,
        ) = constraint
        if int(charge) != 0 or int(atom_type) not in CNOF_ATOM_TYPES:
            return False
        external_valence = current_valence[int(host_node)] - int(source_valence)
        internal_valence = int(target_valence)
        bond_valence = external_valence + internal_valence
        hydrogens = int(CNOF_VALENCE[int(atom_type)]) - bond_valence
        allowed = 0 <= hydrogens <= MAX_H_COUNT
        if bool(aromatic) and allowed:
            if int(atom_type) == int(CNOF_ATOM_TYPES[0]):
                allowed = True
            elif int(atom_type) == int(CNOF_ATOM_TYPES[1]):
                pyridine_like = bond_valence == 3 and hydrogens == 0
                pyrrole_like = (
                    internal_valence == 2
                    and external_valence == 0
                    and hydrogens == 1
                )
                allowed = (pyridine_like or pyrrole_like) and hydrogens == int(
                    target_hydrogens
                )
            elif int(atom_type) == int(CNOF_ATOM_TYPES[2]):
                allowed = (
                    internal_valence == 2
                    and external_valence == 0
                    and hydrogens == 0
                    and int(target_hydrogens) == 0
                )
            else:
                allowed = False
        return bool(allowed)

    mask = _rooted_tree_support_mask(
        host,
        width=len(templates),
        index=_electronic_rooted_support_index(templates, electronic_aliases),
        compatible=compatible,
    )
    if len(_RING_ELECTRONIC_SUPPORT_MASK_CACHE) >= 8192:
        _RING_ELECTRONIC_SUPPORT_MASK_CACHE.pop(
            next(iter(_RING_ELECTRONIC_SUPPORT_MASK_CACHE))
        )
    mask.setflags(write=False)
    _RING_ELECTRONIC_SUPPORT_MASK_CACHE[cache_key] = (
        templates,
        electronic_aliases,
        mask,
    )
    return mask


def enumerate_ring_system_template_placements(
    state: MolecularGraph,
    template: RingSystemTemplate,
) -> tuple[RingSystemPlacement, ...]:
    """Enumerate matches for one selected rule rather than the whole catalog."""

    host = _eligible_grow_host_graph(state)
    if template.span > host.number_of_nodes():
        return ()
    pattern = _template_graph(template, target=False, include_atom_labels=False)
    matcher = nx.algorithms.isomorphism.GraphMatcher(host, pattern)
    actions: set[RingSystemPlacement] = set()
    for host_to_pattern in matcher.subgraph_isomorphisms_iter():
        mapping = {
            int(pattern_slot): int(host_slot) for host_slot, pattern_slot in host_to_pattern.items()
        }
        action = _instantiate_placement(template, mapping, state)
        actions.add(action)
    return tuple(sorted(actions, key=_placement_sort_key))


def _enumerate_mapped_ring_system_template_placements(
    state: MolecularGraph,
    template: RingSystemTemplate,
) -> tuple[tuple[RingSystemPlacement, tuple[tuple[int, int], ...]], ...]:
    """Enumerate placements together with pattern-node-to-host mappings."""

    host = _eligible_grow_host_graph(state)
    if template.span > host.number_of_nodes():
        return ()
    pattern = _template_graph(template, target=False, include_atom_labels=False)
    matcher = nx.algorithms.isomorphism.GraphMatcher(host, pattern)
    items = set()
    for host_to_pattern in matcher.subgraph_isomorphisms_iter():
        mapping = {
            int(pattern_slot): int(host_slot)
            for host_slot, pattern_slot in host_to_pattern.items()
        }
        placement = _instantiate_placement(template, mapping, state)
        items.add((placement, tuple(sorted(mapping.items()))))
    return tuple(
        sorted(
            items,
            key=lambda item: (
                _placement_sort_key(item[0]),
                item[1],
            ),
        )
    )


def enumerate_executable_ring_grow_candidates(
    state: MolecularGraph,
    electronic_aliases: tuple[tuple[RingSystemElectronicAlias, int], ...],
    *,
    stop_after_first: bool = False,
) -> tuple[ExecutableRingGrowCandidate, ...]:
    """Enumerate observed joint labels that execute in ``state``.

    ``stop_after_first`` is a positive-witness optimization for semantic
    template support.  It never proves absence; callers that need the complete
    catalog-exact distribution retain the exhaustive default.
    """

    candidates: dict[
        tuple,
        tuple[RingSystemPlacement, tuple[int, ...], RingSystemGrow, int],
    ] = {}
    for alias, count in electronic_aliases:
        if int(count) <= 0:
            raise ValueError("ring electronic alias counts must be positive")
        aromatic_nodes = {
            int(node)
            for edge in alias.pattern.target_aromatic_edges
            for node in edge
        }
        seen_keys = set()
        for placement, mapping_items in _enumerate_mapped_ring_system_template_placements(
            state,
            alias.pattern,
        ):
            mapping = dict(mapping_items)
            atom_state_by_slot = {
                int(mapping[pattern_slot]): alias.target_atoms[pattern_slot]
                for pattern_slot in range(alias.pattern.span)
            }
            atom_types = tuple(
                int(atom_state_by_slot[slot][0]) for slot in placement.system_atoms
            )
            if any(atom_type not in CNOF_ATOM_TYPES for atom_type in atom_types):
                continue
            if any(
                int(atom_state_by_slot[slot][1]) != 0 for slot in placement.system_atoms
            ):
                continue
            try:
                action = placement.instantiate(state, atom_types)
            except ValueError:
                continue
            payload_by_slot = {
                int(payload.slot): payload for payload in action.atom_payloads
            }
            closure_valence = {
                slot: sum(
                    int(BOND_CLASS_TO_H_CHANGE[int(bond.order)])
                    for bond in action.bond_insertions
                    if slot in (int(bond.a), int(bond.b))
                )
                for slot in placement.system_atoms
            }
            aromatic_role_matches = all(
                int(payload_by_slot[host_slot].atom_type)
                == int(alias.target_atoms[pattern_slot][0])
                and (
                    int(alias.target_atoms[pattern_slot][0])
                    not in (int(CNOF_ATOM_TYPES[1]), int(CNOF_ATOM_TYPES[2]))
                    or pattern_slot not in aromatic_nodes
                    or int(payload_by_slot[host_slot].implicit_h_count)
                    - closure_valence[host_slot]
                    == int(alias.target_atoms[pattern_slot][2])
                )
                for pattern_slot, host_slot in mapping_items
            )
            if not aromatic_role_matches or not is_valid_ring_system_grow(state, action):
                continue
            key = ring_system_grow_electronic_key(action)
            if key in seen_keys:
                continue
            if stop_after_first:
                return (
                    ExecutableRingGrowCandidate(
                        placement=placement,
                        atom_types=atom_types,
                        action=action,
                        support_count=int(count),
                    ),
                )
            seen_keys.add(key)
            incumbent = candidates.get(key)
            if incumbent is None:
                candidates[key] = (placement, atom_types, action, int(count))
            else:
                incumbent_action = incumbent[2]
                representative = min((incumbent_action, action), key=repr)
                candidates[key] = (
                    ring_system_placement(representative),
                    key[1],
                    representative,
                    incumbent[3] + int(count),
                )
    return tuple(
        ExecutableRingGrowCandidate(
            placement=placement,
            atom_types=atom_types,
            action=action,
            support_count=support_count,
        )
        for _, (placement, atom_types, action, support_count) in sorted(
            candidates.items(),
            key=lambda item: item[0],
        )
    )


def matching_ring_system_template_indices(
    action: RingSystemGrow,
    templates: tuple[RingSystemTemplate, ...],
) -> tuple[int, ...]:
    """Find production rules whose complete pattern realizes a teacher action."""

    representatives, buckets = _structured_template_index_by_identity(templates)
    if representatives != templates:
        raise ValueError("teacher matching requires structured representatives")
    teacher_graph = _action_pattern_graph(action)
    key = _pattern_bucket_key(teacher_graph, str(action.topology_class))
    index_by_template = {template: index for index, template in enumerate(templates)}
    return tuple(
        index_by_template[template]
        for template, candidate_graph in buckets.get(key, ())
        if nx.is_isomorphic(
            teacher_graph,
            candidate_graph,
            edge_match=lambda left, right: left["color"] == right["color"],
        )
    )


def enumerate_ring_system_placements(
    state: MolecularGraph,
    catalog: TypedRingCatalog,
) -> tuple[RingSystemPlacement, ...]:
    """Enumerate topology/pattern matches and factor atom identities separately."""

    host = _state_graph(state, include_atom_labels=False)
    cyclic_atoms = _cyclic_atoms(host)
    carbon = int(ELEMENT_TO_IDX["C"])
    actions: dict[tuple, RingSystemPlacement] = {}
    templates, _ = _structured_template_index_by_identity(catalog.ring_system_templates)
    for template in templates:
        if template.span > host.number_of_nodes():
            continue
        pattern = _template_graph(template, target=False, include_atom_labels=False)
        matcher = nx.algorithms.isomorphism.GraphMatcher(host, pattern)
        for host_to_pattern in matcher.subgraph_isomorphisms_iter():
            mapping = {
                int(pattern_slot): int(host_slot)
                for host_slot, pattern_slot in host_to_pattern.items()
            }
            members = set(mapping.values())
            if cyclic_atoms & members:
                continue
            if any(
                int(state.atom_types[slot]) != carbon or int(state.formal_charges[slot]) != 0
                for slot in members
            ):
                continue
            ordered_members = tuple(sorted(members))
            if any(
                int(state.bonds[a, b]) != 1
                for offset, a in enumerate(ordered_members)
                for b in ordered_members[offset + 1 :]
                if int(state.bonds[a, b]) != 0
            ):
                continue
            action = _instantiate_placement(template, mapping, state)
            key = ring_system_placement_key(action)
            incumbent = actions.get(key)
            if incumbent is None or _placement_sort_key(action) < _placement_sort_key(incumbent):
                actions[key] = action
    return tuple(sorted(actions.values(), key=_placement_sort_key))


def structured_ring_trace_supported(
    trace: RewriteTrace,
    catalog: TypedRingCatalog,
) -> bool:
    """Check teacher support without requiring corpus-memorized atom labels."""

    _, buckets = _structured_template_index_by_identity(catalog.ring_system_templates)
    for step in trace.steps:
        if isinstance(step.action, RingSystemGrow):
            teacher_graph = _action_pattern_graph(step.action)
            bucket_key = _pattern_bucket_key(
                teacher_graph,
                str(step.action.topology_class),
            )
            candidates = buckets.get(bucket_key, ())
            if not any(
                nx.is_isomorphic(
                    teacher_graph,
                    candidate_graph,
                    edge_match=lambda left, right: left["color"] == right["color"],
                )
                for _, candidate_graph in candidates
            ):
                return False
    return True


def structured_ring_electronic_trace_supported(
    trace: RewriteTrace,
    catalog: TypedRingCatalog,
) -> bool:
    """Whether every ring teacher has an observed paired electronic orbit."""

    templates = structured_ring_system_templates(catalog)
    electronic_groups = structured_ring_system_electronic_aliases(catalog)
    for step in trace.steps:
        if not isinstance(step.action, RingSystemGrow):
            continue
        teacher_alias = ring_system_electronic_alias(step.action)
        template_indices = matching_ring_system_template_indices(
            step.action,
            templates,
        )
        if not any(
            ring_system_electronic_aliases_equivalent(teacher_alias, alias)
            for template_index in template_indices
            for alias, _ in electronic_groups[template_index]
        ):
            return False
    return True


def enumerate_ring_system_grows(
    state: MolecularGraph,
    catalog: TypedRingCatalog,
) -> tuple[RingSystemGrow, ...]:
    """Enumerate complete acyclic-scaffold matches without building successors."""

    host = _state_graph(state)
    cyclic_atoms = _cyclic_atoms(host)
    actions: set[RingSystemGrow] = set()
    for template in catalog.ring_system_templates:
        if template.span > host.number_of_nodes():
            continue
        pattern = _template_graph(template, target=False)
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            host,
            pattern,
            node_match=lambda left, right: left["atom"] == right["atom"],
            edge_match=lambda left, right: left["order"] == right["order"],
        )
        for host_to_pattern in matcher.subgraph_isomorphisms_iter():
            pattern_to_host = {
                int(pattern_slot): int(host_slot)
                for host_slot, pattern_slot in host_to_pattern.items()
            }
            members = set(pattern_to_host.values())
            if cyclic_atoms & members:
                continue
            if not _external_signatures_match(
                state,
                pattern_to_host,
                template.source_external_bonds,
            ):
                continue
            actions.add(_instantiate_grow(template, pattern_to_host, state))
    return tuple(sorted(actions, key=_grow_sort_key))


def enumerate_ring_system_deletes(
    state: MolecularGraph,
    catalog: TypedRingCatalog,
) -> tuple[RingSystemDelete, ...]:
    """Enumerate whole-system inverses only for exact current cyclic blocks."""

    host = _state_graph(state)
    cyclic_components = _cyclic_components(host)
    actions: set[RingSystemDelete] = set()
    for template in catalog.ring_system_templates:
        if template.span > host.number_of_nodes():
            continue
        pattern = _template_graph(template, target=True)
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            host,
            pattern,
            node_match=lambda left, right: left["atom"] == right["atom"],
            edge_match=lambda left, right: left["order"] == right["order"],
        )
        for host_to_pattern in matcher.subgraph_isomorphisms_iter():
            pattern_to_host = {
                int(pattern_slot): int(host_slot)
                for host_slot, pattern_slot in host_to_pattern.items()
            }
            members = frozenset(pattern_to_host.values())
            if members not in cyclic_components:
                continue
            if not _external_signatures_match(
                state,
                pattern_to_host,
                template.source_external_bonds,
            ):
                continue
            actions.add(_instantiate_delete(template, pattern_to_host))
    return tuple(sorted(actions, key=_delete_sort_key))


def enumerate_structured_ring_system_deletes(
    state: MolecularGraph,
    catalog: TypedRingCatalog,
) -> tuple[RingSystemDelete, ...]:
    """Delete catalog-supported topologies regardless of their generated labels."""

    host = _state_graph(state, include_atom_labels=False)
    cyclic_components = _cyclic_components(host)
    if not cyclic_components:
        return ()
    perceived = resonance_invariant_bond_classes(state)
    actions: set[RingSystemDelete] = set()
    index = _structured_delete_index_by_identity(catalog.ring_system_templates)
    for members in cyclic_components:
        component = nx.Graph()
        component.add_nodes_from(sorted(members))
        component.add_edges_from(
            (int(a), int(b), {"color": int(perceived[int(a), int(b)])})
            for offset, a in enumerate(sorted(members))
            for b in sorted(members)[offset + 1 :]
            if int(perceived[int(a), int(b)]) != 0
        )
        key = (
            component.number_of_nodes(),
            nx.weisfeiler_lehman_graph_hash(component, edge_attr="color"),
        )
        selected: RingSystemDelete | None = None
        for template, pattern in index.get(key, ()):
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                component,
                pattern,
                edge_match=lambda left, right: left["color"] == right["color"],
            )
            for host_to_pattern in matcher.isomorphisms_iter():
                mapping = {
                    int(pattern_slot): int(host_slot)
                    for host_slot, pattern_slot in host_to_pattern.items()
                }
                action = _instantiate_structured_delete(
                    template,
                    mapping,
                    state,
                    perceived,
                )
                if action is not None:
                    selected = action
                    break
            if selected is not None:
                break
        if selected is not None:
            actions.add(selected)
    return tuple(sorted(actions, key=_delete_sort_key))


def enumerate_clean_ring_system_deletes(
    state: MolecularGraph,
    catalog: TypedRingCatalog,
) -> tuple[RingSystemDelete, ...]:
    """Like ``enumerate_structured_ring_system_deletes`` but DECORATION-PRESERVING (see
    ``_instantiate_clean_delete``): open a ring back to its acyclic scaffold while KEEPING each atom's
    element, so heteroatoms survive. This is the editing delete -- its inverse (``inverse_ring_system_
    delete``) re-cyclizes exactly, so a corruption can teach ring add/remove on real leads round-trip."""
    host = _state_graph(state, include_atom_labels=False)
    cyclic_components = _cyclic_components(host)
    if not cyclic_components:
        return ()
    perceived = resonance_invariant_bond_classes(state)
    actions: set[RingSystemDelete] = set()
    index = _structured_delete_index_by_identity(catalog.ring_system_templates)
    for members in cyclic_components:
        component = nx.Graph()
        component.add_nodes_from(sorted(members))
        component.add_edges_from(
            (int(a), int(b), {"color": int(perceived[int(a), int(b)])})
            for offset, a in enumerate(sorted(members))
            for b in sorted(members)[offset + 1 :]
            if int(perceived[int(a), int(b)]) != 0
        )
        key = (
            component.number_of_nodes(),
            nx.weisfeiler_lehman_graph_hash(component, edge_attr="color"),
        )
        selected: RingSystemDelete | None = None
        for template, pattern in index.get(key, ()):
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                component,
                pattern,
                edge_match=lambda left, right: left["color"] == right["color"],
            )
            for host_to_pattern in matcher.isomorphisms_iter():
                mapping = {
                    int(pattern_slot): int(host_slot)
                    for host_slot, pattern_slot in host_to_pattern.items()
                }
                action = _instantiate_clean_delete(template, mapping, state, perceived)
                if action is not None:
                    selected = action
                    break
            if selected is not None:
                break
        if selected is not None:
            actions.add(selected)
    return tuple(sorted(actions, key=_delete_sort_key))


def _state_graph(
    state: MolecularGraph,
    *,
    include_atom_labels: bool = True,
) -> nx.Graph:
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    for slot in real:
        attributes = {}
        if include_atom_labels:
            attributes["atom"] = (
                int(state.atom_types[slot]),
                int(state.formal_charges[slot]),
                int(state.implicit_h_counts[slot]),
            )
        graph.add_node(slot, **attributes)
    graph.add_edges_from(
        (a, b, {"order": int(state.bonds[a, b])})
        for offset, a in enumerate(real)
        for b in real[offset + 1 :]
        if int(state.bonds[a, b]) != 0
    )
    return graph


@lru_cache(maxsize=4096)
def _template_graph(
    template: RingSystemTemplate,
    *,
    target: bool,
    include_atom_labels: bool = True,
) -> nx.Graph:
    graph = nx.Graph()
    atoms = template.target_atoms if target else template.source_atoms
    bonds = template.target_bonds if target else template.source_bonds
    graph.add_nodes_from(
        (
            index,
            {"atom": atom} if include_atom_labels else {},
        )
        for index, atom in enumerate(atoms)
    )
    graph.add_edges_from((int(a), int(b), {"order": int(order)}) for a, b, order in bonds)
    return graph


def _cyclic_atoms(graph: nx.Graph) -> set[int]:
    bridges = {frozenset((int(a), int(b))) for a, b in nx.bridges(graph)}
    return {
        int(vertex)
        for a, b in graph.edges()
        if frozenset((int(a), int(b))) not in bridges
        for vertex in (a, b)
    }


def _cyclic_components(graph: nx.Graph) -> set[frozenset[int]]:
    bridges = {frozenset((int(a), int(b))) for a, b in nx.bridges(graph)}
    cyclic = nx.Graph()
    cyclic.add_edges_from(
        (int(a), int(b)) for a, b in graph.edges() if frozenset((int(a), int(b))) not in bridges
    )
    return {frozenset(int(v) for v in component) for component in nx.connected_components(cyclic)}


def _external_signatures_match(
    state: MolecularGraph,
    pattern_to_host: dict[int, int],
    expected: tuple[tuple[int, ...], ...],
) -> bool:
    members = set(pattern_to_host.values())
    for pattern_slot, host_slot in pattern_to_host.items():
        actual = tuple(
            sorted(
                int(state.bonds[host_slot, neighbor])
                for neighbor in np.flatnonzero(state.bonds[host_slot] != 0)
                if int(neighbor) not in members
            )
        )
        if actual != expected[pattern_slot]:
            return False
    return True


def _target_pattern_matches(
    state: MolecularGraph,
    perceived: np.ndarray,
    template: RingSystemTemplate,
    mapping: dict[int, int],
) -> bool:
    aromatic = {frozenset((int(a), int(b))) for a, b in template.target_aromatic_edges}
    for a, b, order in template.target_bonds:
        actual_a, actual_b = mapping[a], mapping[b]
        expected = BOND_AROMATIC if frozenset((a, b)) in aromatic else int(order)
        actual = int(perceived[actual_a, actual_b])
        if actual != expected:
            return False
    return True


def _instantiate_structured_delete(
    template: RingSystemTemplate,
    mapping: dict[int, int],
    state: MolecularGraph,
    perceived: np.ndarray,
) -> RingSystemDelete | None:
    members = tuple(sorted(mapping.values()))
    inserted_edges = tuple(
        (
            min(mapping[a], mapping[b]),
            max(mapping[a], mapping[b]),
        )
        for a, b, _ in template.inserted_bonds
    )
    source_edges = tuple(
        (
            min(mapping[a], mapping[b]),
            max(mapping[a], mapping[b]),
            int(order),
        )
        for a, b, order in template.source_bonds
    )

    post_delete_bonds = state.bonds.copy()
    for a, b in inserted_edges:
        post_delete_bonds[a, b] = post_delete_bonds[b, a] = 0
    precursor_bonds = post_delete_bonds.copy()
    for a, b, order in source_edges:
        precursor_bonds[a, b] = precursor_bonds[b, a] = int(order)
    post_delete_h: dict[int, int] = {}
    carbon = int(ELEMENT_TO_IDX["C"])
    for slot in members:
        bond_valence = sum(
            int(BOND_CLASS_TO_H_CHANGE[int(order)]) for order in precursor_bonds[slot]
        )
        hydrogens = int(CNOF_VALENCE[carbon]) - bond_valence
        if not 0 <= hydrogens <= MAX_H_COUNT:
            return None
        post_delete_valence = sum(
            int(BOND_CLASS_TO_H_CHANGE[int(order)]) for order in post_delete_bonds[slot]
        )
        intermediate_h = int(CNOF_VALENCE[carbon]) - post_delete_valence
        if not 0 <= intermediate_h <= MAX_H_COUNT:
            return None
        post_delete_h[slot] = intermediate_h
    bond_deletions = tuple(
        sorted(RingBond(a, b, int(state.bonds[a, b])) for a, b in inserted_edges)
    )
    atom_payloads = tuple(
        AtomPayload(
            slot=slot,
            atom_type=carbon,
            formal_charge=0,
            implicit_h_count=post_delete_h[slot],
        )
        for slot in members
    )
    bond_reorders = tuple(
        sorted(
            BondOrderChange(a, b, int(order))
            for a, b, order in source_edges
            if int(state.bonds[a, b]) != int(order)
        )
    )
    aromatic_edges = tuple(
        (a, b)
        for offset, a in enumerate(members)
        for b in members[offset + 1 :]
        if int(perceived[a, b]) == BOND_AROMATIC
    )
    return RingSystemDelete(
        system_atoms=members,
        retained_system_atoms=(),
        bond_deletions=bond_deletions,
        atom_deletions=(),
        atom_payloads=atom_payloads,
        bond_reorders=bond_reorders,
        source_aromatic_edges=(),
        aromatic_edges=aromatic_edges,
        topology_class=template.topology_class,
    )


def _instantiate_clean_delete(
    template: RingSystemTemplate,
    mapping: dict[int, int],
    state: MolecularGraph,
    perceived: np.ndarray,
) -> RingSystemDelete | None:
    """Decoration-PRESERVING ring delete: open the ring's closing bonds and KEEP each atom's element
    (unlike ``_instantiate_structured_delete``, which retypes every member to carbon). Each atom loses
    only the opened bonds, so its H is re-derived from its OWN valence -- making this the exact inverse of
    the grow that builds the ring, so ``inverse_ring_system_delete`` round-trips it. States are Kekule, so
    this generalizes across saturated/aromatic rings of any size and fused/spiro/bridged systems."""
    members = tuple(sorted(mapping.values()))
    inserted_edges = tuple(
        (min(mapping[a], mapping[b]), max(mapping[a], mapping[b]))
        for a, b, _ in template.inserted_bonds
    )
    post_delete_bonds = state.bonds.copy()
    for a, b in inserted_edges:
        post_delete_bonds[a, b] = post_delete_bonds[b, a] = 0
    payloads: list[AtomPayload] = []
    for slot in members:
        atom_type = int(state.atom_types[slot])
        post_delete_valence = sum(
            int(BOND_CLASS_TO_H_CHANGE[int(order)]) for order in post_delete_bonds[slot]
        )
        hydrogens = canonical_h_count(atom_type, post_delete_valence)
        if hydrogens is None:
            return None
        payloads.append(
            AtomPayload(
                slot=slot,
                atom_type=atom_type,
                formal_charge=int(state.formal_charges[slot]),
                implicit_h_count=hydrogens,
            )
        )
    bond_deletions = tuple(
        sorted(RingBond(a, b, int(state.bonds[a, b])) for a, b in inserted_edges)
    )
    aromatic_edges = tuple(
        (a, b)
        for offset, a in enumerate(members)
        for b in members[offset + 1 :]
        if int(perceived[a, b]) == BOND_AROMATIC
    )
    return RingSystemDelete(
        system_atoms=members,
        retained_system_atoms=(),
        bond_deletions=bond_deletions,
        atom_deletions=(),
        atom_payloads=tuple(payloads),
        bond_reorders=(),
        source_aromatic_edges=(),
        aromatic_edges=aromatic_edges,
        topology_class=template.topology_class,
    )


def _instantiate_grow(
    template: RingSystemTemplate,
    mapping: dict[int, int],
    state: MolecularGraph,
) -> RingSystemGrow:
    members = tuple(sorted(mapping.values()))
    return RingSystemGrow(
        system_atoms=members,
        interface_atoms=(),
        scaffold_bonds=tuple(
            sorted(
                RingBond(
                    min(mapping[a], mapping[b]),
                    max(mapping[a], mapping[b]),
                    int(order),
                )
                for a, b, order in template.source_bonds
            )
        ),
        bond_reorders=tuple(
            BondOrderChange(
                mapping[a],
                mapping[b],
                int(order),
            )
            for a, b, order in template.grow_bond_reorders
        ),
        atom_payloads=tuple(
            AtomPayload(mapping[index], atom_type, charge, hydrogens)
            for index, (atom_type, charge, hydrogens) in template.grow_atom_payloads
        ),
        atom_insertions=(),
        bond_insertions=tuple(
            RingBond(
                min(mapping[a], mapping[b]),
                max(mapping[a], mapping[b]),
                order,
            )
            for a, b, order in template.inserted_bonds
        ),
        source_aromatic_edges=(),
        aromatic_edges=tuple(
            sorted(
                (min(mapping[a], mapping[b]), max(mapping[a], mapping[b]))
                for a, b in template.target_aromatic_edges
            )
        ),
        topology_class=template.topology_class,
    )


def _instantiate_placement(
    template: RingSystemTemplate,
    mapping: dict[int, int],
    state: MolecularGraph,
) -> RingSystemPlacement:
    target_orders = {(a, b): order for a, b, order in template.target_bonds}
    inserted_edges = {(min(a, b), max(a, b)) for a, b, _ in template.inserted_bonds}
    members = tuple(sorted(mapping.values()))
    scaffold_bonds = tuple(
        sorted(
            RingBond(
                min(mapping[a], mapping[b]),
                max(mapping[a], mapping[b]),
                int(state.bonds[mapping[a], mapping[b]]),
            )
            for a, b, _ in template.source_bonds
        )
    )
    bond_reorders = tuple(
        sorted(
            BondOrderChange(
                min(mapping[a], mapping[b]),
                max(mapping[a], mapping[b]),
                int(target_orders[(a, b)]),
            )
            for a, b, _ in template.source_bonds
            if int(state.bonds[mapping[a], mapping[b]]) != int(target_orders[(a, b)])
        )
    )
    bond_insertions = tuple(
        sorted(
            RingBond(
                min(mapping[a], mapping[b]),
                max(mapping[a], mapping[b]),
                int(order),
            )
            for a, b, order in template.target_bonds
            if (min(a, b), max(a, b)) in inserted_edges
        )
    )
    aromatic_edges = tuple(
        sorted(
            (min(mapping[a], mapping[b]), max(mapping[a], mapping[b]))
            for a, b in template.target_aromatic_edges
        )
    )
    return RingSystemPlacement(
        system_atoms=members,
        scaffold_bonds=scaffold_bonds,
        bond_reorders=bond_reorders,
        bond_insertions=bond_insertions,
        aromatic_edges=aromatic_edges,
        topology_class=template.topology_class,
    )


def _instantiate_delete(
    template: RingSystemTemplate,
    mapping: dict[int, int],
) -> RingSystemDelete:
    return RingSystemDelete(
        system_atoms=tuple(sorted(mapping.values())),
        retained_system_atoms=(),
        bond_deletions=tuple(
            RingBond(
                min(mapping[a], mapping[b]),
                max(mapping[a], mapping[b]),
                order,
            )
            for a, b, order in template.deleted_bonds
        ),
        atom_deletions=(),
        atom_payloads=tuple(
            AtomPayload(mapping[index], atom_type, charge, hydrogens)
            for index, (atom_type, charge, hydrogens) in template.delete_atom_payloads
        ),
        bond_reorders=tuple(
            BondOrderChange(
                mapping[a],
                mapping[b],
                int(order),
            )
            for a, b, order in template.delete_bond_reorders
        ),
        source_aromatic_edges=(),
        aromatic_edges=tuple(
            sorted(
                (min(mapping[a], mapping[b]), max(mapping[a], mapping[b]))
                for a, b in template.target_aromatic_edges
            )
        ),
        topology_class=template.topology_class,
    )


def _grow_sort_key(action: RingSystemGrow) -> tuple:
    return (
        action.system_atoms,
        action.topology_class,
        action.bond_insertions,
        tuple(
            (
                item.slot,
                item.atom_type,
                item.formal_charge,
                item.implicit_h_count,
            )
            for item in action.atom_payloads
        ),
    )


def _delete_sort_key(action: RingSystemDelete) -> tuple:
    return (
        action.system_atoms,
        action.topology_class,
        action.bond_deletions,
        tuple(
            (
                item.slot,
                item.atom_type,
                item.formal_charge,
                item.implicit_h_count,
            )
            for item in action.atom_payloads
        ),
    )


def _placement_sort_key(action: RingSystemPlacement) -> tuple:
    return (
        action.system_atoms,
        action.topology_class,
        action.scaffold_bonds,
        action.bond_reorders,
        action.bond_insertions,
        action.aromatic_edges,
    )


__all__ = [
    "build_semantic_ring_system_decoder",
    "ExecutableRingGrowCandidate",
    "instantiate_semantic_ring_system_grow",
    "RingAtomElectronicState",
    "ring_atom_electronic_category",
    "ring_system_grow_final_atom_states",
    "ring_system_placement_with_aromatic_matching",
    "semantic_ring_canonical_matching",
    "semantic_ring_categories_for_action",
    "semantic_ring_next_category_mask",
    "semantic_ring_prefix_is_completable",
    "SemanticRingSystemDecoder",
    "enumerate_executable_ring_grow_candidates",
    "enumerate_ring_system_deletes",
    "enumerate_ring_system_grows",
    "enumerate_ring_system_placement_resonance_aliases",
    "enumerate_ring_system_placements",
    "enumerate_ring_system_template_placements",
    "enumerate_structured_ring_system_deletes",
    "matching_ring_system_template_indices",
    "ring_system_placement",
    "ring_system_placement_has_local_atom_support",
    "ring_system_placement_key",
    "ring_system_placement_local_atom_type_mask",
    "ring_system_placement_resonance_atom_type_mask",
    "ring_system_grow_electronic_key",
    "ring_system_electronic_template_support_mask",
    "ring_system_template_local_support_mask",
    "ring_system_template_support_mask",
    "RingSystemPlacement",
    "structured_ring_system_electronic_aliases",
    "structured_ring_system_template_aliases",
    "structured_ring_system_templates",
    "structured_ring_electronic_trace_supported",
    "structured_ring_trace_supported",
    "warm_ring_system_candidate_indices",
]
