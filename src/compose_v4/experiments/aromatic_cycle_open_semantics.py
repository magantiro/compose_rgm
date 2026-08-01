"""Non-authorizing prototypes for resonance-invariant aromatic cycle opening.

The production executor stores aromatic molecules in one integer-order Kekule
lowering.  Deleting the stored bond directly therefore makes the molecular
successor depend on an arbitrary resonance phase.  This module prototypes a
semantic alternative without changing the production implementation or any
existing artifact:

1. enumerate every charge/H/connectivity-preserving Kekule alias;
2. anchor the selected semantic aromatic edge as a single bond;
3. apply the current ``BondDelete`` arithmetic to every anchored alias;
4. require one nonempty canonical molecular-product group; and
5. select one exact persistent-slot representative deterministically with
   respect to supplier order.

The proposed law can reject aromatic edges that have no charge- and
hydrogen-preserving forced-single alias, so it may contract support. The
prototype cannot authorize Gate 0, T1, P50, corpus rematerialization, or a
production architecture. Promotion requires a frozen support decision and
full alternate-Kekule, slot, and multistep successor-law invariance gates.
The current lexicographic exact-representative choice is not claimed to be
slot-equivariant in symmetric fused systems. Only the canonical product is
qualified by this prototype.

This module also contains a second, non-authorizing component-factored
prototype. It replaces whole-molecule resonance enumeration with an exact
binary degree-constrained matching problem on each resonance-invariant
aromatic-edge component. Component assignments are enumerated completely,
never silently truncated, and their Cartesian-product cardinality is computed
without materializing the product. The original RDKit exhaustive resolver is
retained as the independent oracle for bounded equivalence tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from math import prod

import networkx as nx
import numpy as np
from rdkit import Chem

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_NULL,
    BOND_SINGLE,
    BOND_TRIPLE,
    IDX_TO_ELEMENT,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import (
    BondDelete,
    BondInsert,
    apply_bond_delete,
    is_valid_bond_delete,
)

PROTOTYPE_STATUS = "NON_AUTHORIZING_REPRESENTATION_INVARIANT_SEMANTICS_PROTOTYPE"
COMPONENT_FACTORED_PROTOTYPE_STATUS = (
    "NON_AUTHORIZING_COMPONENT_FACTORED_AROMATIC_CYCLE_OPEN_PROTOTYPE"
)


class AromaticCycleOpenRejectionCode(str, Enum):
    INVALID_SOURCE = "invalid_source"
    INVALID_EDGE = "invalid_edge"
    EDGE_ABSENT = "edge_absent"
    EDGE_IS_BRIDGE = "edge_is_bridge"
    NONAROMATIC_EXECUTOR_REJECTED = "nonaromatic_executor_rejected"
    NO_PRESERVING_KEKULE_ALIAS = "no_charge_h_connectivity_preserving_kekule_alias"
    NO_FORCED_SINGLE_ALIAS = "selected_aromatic_edge_has_no_forced_single_alias"
    FORCED_SINGLE_EXECUTION_DISAGREEMENT = "forced_single_alias_execution_disagreement"
    EMPTY_CANONICAL_PRODUCT_GROUP = "empty_canonical_product_group"
    AMBIGUOUS_CANONICAL_PRODUCT = "multiple_canonical_product_groups"
    COMPONENT_CONTAINS_NON_KEKULE_BOND = "aromatic_component_contains_non_single_double_source_bond"
    COMPONENT_FACTORIZATION_DISAGREEMENT = "component_factorization_composition_disagreement"


class AromaticCycleOpenAliasOverflow(RuntimeError):
    """Complete resonance enumeration exceeded the explicit prototype cap."""

    reason_code = "kekule_alias_enumeration_overflow"

    def __init__(self, *, maximum_aliases: int, observed_structures: int) -> None:
        self.maximum_aliases = int(maximum_aliases)
        self.observed_structures = int(observed_structures)
        super().__init__(
            f"{self.reason_code}: observed at least {observed_structures} structures "
            f"for cap {maximum_aliases}"
        )


@dataclass(frozen=True)
class KekuleAliasEnumeration:
    source_key: str
    aromatic_edges: tuple[tuple[int, int], ...]
    raw_structure_count: int
    aliases: tuple[MolecularGraph, ...]


@dataclass(frozen=True)
class AromaticCycleOpenResolution:
    """One semantic cycle-open decision with explicit evidence cardinalities.

    ``enumerated_alias_count`` and ``forced_single_alias_count`` are exact
    cardinalities of the represented global alias sets. In the component-
    factored prototype, ``executed_product_count`` is instead the number of
    selected-component representatives actually materialized and executed.
    """

    prototype_status: str
    admitted: bool
    rejection_code: AromaticCycleOpenRejectionCode | None
    source_key: str | None
    edge: tuple[int, int]
    semantic_aromatic_edge: bool
    enumerated_alias_count: int
    forced_single_alias_count: int
    executed_product_count: int
    canonical_product_keys: tuple[str, ...]
    successor: MolecularGraph | None
    inverse_bond_order: int | None


@dataclass(frozen=True)
class AromaticComponentAssignments:
    """Complete feasible Kekule assignments for one aromatic-edge component."""

    edges: tuple[tuple[int, int], ...]
    bond_orders: tuple[tuple[int, ...], ...]


ResonanceSupplierFactory = Callable[[Chem.Mol, int], Iterable[Chem.Mol]]


def rdkit_kekule_supplier(
    molecule: Chem.Mol,
    maximum_structures: int,
) -> Iterable[Chem.Mol]:
    """Return RDKit's Kekule-only resonance supplier with an explicit bound."""

    return Chem.ResonanceMolSupplier(
        molecule,
        flags=Chem.ResonanceFlags.KEKULE_ALL,
        maxStructs=int(maximum_structures),
    )


def _normalize_edge(action: BondDelete) -> tuple[int, int]:
    return tuple(sorted((int(action.a), int(action.b))))


def _exact_state_key(state: MolecularGraph) -> tuple:
    """Deterministic persistent-slot key, independent of supplier ordering."""

    return (
        tuple(int(value) for value in state.atom_types),
        tuple(int(value) for value in state.formal_charges),
        tuple(int(value) for value in state.implicit_h_counts),
        tuple(int(value) for value in state.bonds.reshape(-1)),
    )


def _index_preserving_rdkit_molecule(
    state: MolecularGraph,
) -> tuple[Chem.Mol, dict[int, int]]:
    real_slots = tuple(int(slot) for slot in np.flatnonzero(is_element(state.atom_types)))
    aromatic_atoms = (state.bonds == BOND_AROMATIC).any(axis=1)
    editable = Chem.RWMol()
    slot_to_rdkit: dict[int, int] = {}
    for slot in real_slots:
        atom = Chem.Atom(IDX_TO_ELEMENT[int(state.atom_types[slot])])
        atom.SetFormalCharge(int(state.formal_charges[slot]))
        atom.SetNumExplicitHs(int(state.implicit_h_counts[slot]))
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(bool(aromatic_atoms[slot]))
        slot_to_rdkit[slot] = int(editable.AddAtom(atom))

    bond_types = {
        BOND_SINGLE: Chem.BondType.SINGLE,
        BOND_DOUBLE: Chem.BondType.DOUBLE,
        BOND_TRIPLE: Chem.BondType.TRIPLE,
        BOND_AROMATIC: Chem.BondType.AROMATIC,
    }
    for left_offset, left in enumerate(real_slots):
        for right in real_slots[left_offset + 1 :]:
            bond_class = int(state.bonds[left, right])
            if bond_class == BOND_NULL:
                continue
            editable.AddBond(
                slot_to_rdkit[left],
                slot_to_rdkit[right],
                bond_types[bond_class],
            )
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule, slot_to_rdkit


def _supplier_preserves_atoms(
    resonance: Chem.Mol,
    state: MolecularGraph,
    slot_to_rdkit: dict[int, int],
) -> bool:
    if resonance.GetNumAtoms() != len(slot_to_rdkit):
        return False
    for slot, rdkit_index in slot_to_rdkit.items():
        atom = resonance.GetAtomWithIdx(rdkit_index)
        if atom.GetSymbol() != IDX_TO_ELEMENT[int(state.atom_types[slot])]:
            return False
        if int(atom.GetFormalCharge()) != int(state.formal_charges[slot]):
            return False
        if int(atom.GetTotalNumHs()) != int(state.implicit_h_counts[slot]):
            return False
    return True


def _alias_from_resonance(
    resonance: Chem.Mol,
    state: MolecularGraph,
    *,
    slot_to_rdkit: dict[int, int],
    aromatic_edges: tuple[tuple[int, int], ...],
    aromatic_edge_set: frozenset[frozenset[int]],
    source_key: str,
) -> MolecularGraph | None:
    if not _supplier_preserves_atoms(resonance, state, slot_to_rdkit):
        return None

    real_slots = tuple(sorted(slot_to_rdkit))
    resonance_edges: set[frozenset[int]] = set()
    for left_offset, left in enumerate(real_slots):
        for right in real_slots[left_offset + 1 :]:
            rdkit_bond = resonance.GetBondBetweenAtoms(
                slot_to_rdkit[left],
                slot_to_rdkit[right],
            )
            source_order = int(state.bonds[left, right])
            if rdkit_bond is None:
                if source_order != BOND_NULL:
                    return None
                continue
            resonance_edges.add(frozenset((left, right)))
            if source_order == BOND_NULL:
                return None
            edge = frozenset((left, right))
            if edge in aromatic_edge_set:
                if rdkit_bond.GetBondType() not in {
                    Chem.BondType.SINGLE,
                    Chem.BondType.DOUBLE,
                }:
                    return None
                continue
            expected = {
                BOND_SINGLE: Chem.BondType.SINGLE,
                BOND_DOUBLE: Chem.BondType.DOUBLE,
                BOND_TRIPLE: Chem.BondType.TRIPLE,
                BOND_AROMATIC: Chem.BondType.AROMATIC,
            }.get(source_order)
            if expected is None or rdkit_bond.GetBondType() != expected:
                return None
    source_edges = {
        frozenset((left, right))
        for left_offset, left in enumerate(real_slots)
        for right in real_slots[left_offset + 1 :]
        if int(state.bonds[left, right]) != BOND_NULL
    }
    if resonance_edges != source_edges:
        return None

    bonds = state.bonds.copy()
    for left, right in aromatic_edges:
        rdkit_bond = resonance.GetBondBetweenAtoms(
            slot_to_rdkit[left],
            slot_to_rdkit[right],
        )
        if rdkit_bond is None:
            return None
        order = BOND_SINGLE if rdkit_bond.GetBondType() == Chem.BondType.SINGLE else BOND_DOUBLE
        bonds[left, right] = bonds[right, left] = order
    alias = MolecularGraph(
        state.atom_types.copy(),
        state.formal_charges.copy(),
        state.implicit_h_counts.copy(),
        bonds,
    )
    if not is_valid_state(alias) or not is_connected_or_null(alias):
        return None
    if canonical_state_key(alias) != source_key:
        return None
    return alias


def enumerate_charge_h_preserving_kekule_aliases(
    state: MolecularGraph,
    *,
    maximum_aliases: int = 256,
    supplier_factory: ResonanceSupplierFactory = rdkit_kekule_supplier,
) -> KekuleAliasEnumeration:
    """Enumerate a complete, sorted set of exact Kekule aliases.

    The supplier is asked for ``cap + 1`` structures. Observing the extra
    structure is a typed failure rather than silent support truncation.
    """

    if maximum_aliases <= 0:
        raise ValueError("maximum_aliases must be positive")
    if not is_valid_state(state) or not is_connected_or_null(state):
        raise ValueError("Kekule alias enumeration requires a valid connected source")
    source_key = canonical_state_key(state)
    perceived = resonance_invariant_bond_classes(state)
    real_slots = tuple(int(slot) for slot in np.flatnonzero(is_element(state.atom_types)))
    aromatic_edges = tuple(
        (left, right)
        for left_offset, left in enumerate(real_slots)
        for right in real_slots[left_offset + 1 :]
        if int(perceived[left, right]) == BOND_AROMATIC
    )
    if not aromatic_edges:
        return KekuleAliasEnumeration(
            source_key=source_key,
            aromatic_edges=(),
            raw_structure_count=0,
            aliases=(state,),
        )

    molecule, slot_to_rdkit = _index_preserving_rdkit_molecule(state)
    raw_structures = tuple(supplier_factory(molecule, int(maximum_aliases) + 1))
    if len(raw_structures) > maximum_aliases:
        raise AromaticCycleOpenAliasOverflow(
            maximum_aliases=maximum_aliases,
            observed_structures=len(raw_structures),
        )
    aromatic_edge_set = frozenset(frozenset(edge) for edge in aromatic_edges)
    aliases_by_key: dict[tuple, MolecularGraph] = {}
    for resonance in raw_structures:
        alias = _alias_from_resonance(
            resonance,
            state,
            slot_to_rdkit=slot_to_rdkit,
            aromatic_edges=aromatic_edges,
            aromatic_edge_set=aromatic_edge_set,
            source_key=source_key,
        )
        if alias is not None:
            aliases_by_key.setdefault(_exact_state_key(alias), alias)
    aliases = tuple(aliases_by_key[key] for key in sorted(aliases_by_key))
    return KekuleAliasEnumeration(
        source_key=source_key,
        aromatic_edges=aromatic_edges,
        raw_structure_count=len(raw_structures),
        aliases=aliases,
    )


def _aromatic_edge_components(
    state: MolecularGraph,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return deterministic resonance-invariant aromatic-edge components."""

    perceived = resonance_invariant_bond_classes(state)
    real_slots = tuple(int(slot) for slot in np.flatnonzero(is_element(state.atom_types)))
    aromatic_edges = tuple(
        (left, right)
        for left_offset, left in enumerate(real_slots)
        for right in real_slots[left_offset + 1 :]
        if int(perceived[left, right]) == BOND_AROMATIC
    )
    graph = nx.Graph()
    graph.add_edges_from(aromatic_edges)
    components: list[tuple[tuple[int, int], ...]] = []
    for nodes in nx.connected_components(graph):
        node_set = frozenset(int(node) for node in nodes)
        edges = tuple(
            edge for edge in aromatic_edges if edge[0] in node_set and edge[1] in node_set
        )
        components.append(edges)
    return tuple(sorted(components))


def _state_with_component_orders(
    state: MolecularGraph,
    components: tuple[AromaticComponentAssignments, ...],
    selections: tuple[tuple[int, ...], ...],
) -> MolecularGraph:
    if len(components) != len(selections):
        raise ValueError("one bond-order selection is required per aromatic component")
    bonds = state.bonds.copy()
    for component, orders in zip(components, selections, strict=True):
        if len(component.edges) != len(orders):
            raise ValueError("component edge and bond-order lengths differ")
        for (left, right), order in zip(component.edges, orders, strict=True):
            bonds[left, right] = bonds[right, left] = int(order)
    return MolecularGraph(
        state.atom_types.copy(),
        state.formal_charges.copy(),
        state.implicit_h_counts.copy(),
        bonds,
    )


def _degree_constrained_component_orders(
    state: MolecularGraph,
    edges: tuple[tuple[int, int], ...],
    *,
    source_key: str,
) -> AromaticComponentAssignments:
    """Enumerate every fixed-charge/H Kekule assignment on one component.

    With atoms, formal charges, hydrogens, connectivity, and all non-aromatic
    bond orders fixed, a Kekule alias is a binary single/double assignment whose
    double-bond incidence at every aromatic atom matches the valid source
    lowering. This is a degree-constrained matching problem. Candidate
    solutions are still passed through the production validity and canonical
    identity predicates, so the matching construction never broadens support
    by algebra alone.
    """

    if not edges:
        raise ValueError("an aromatic component must contain at least one edge")
    source_orders = tuple(int(state.bonds[edge]) for edge in edges)
    if any(order not in {BOND_SINGLE, BOND_DOUBLE} for order in source_orders):
        return AromaticComponentAssignments(edges=edges, bond_orders=())

    vertices = tuple(sorted({vertex for edge in edges for vertex in edge}))
    vertex_offset = {vertex: offset for offset, vertex in enumerate(vertices)}
    demands = [0] * len(vertices)
    for (left, right), order in zip(edges, source_orders, strict=True):
        if order == BOND_DOUBLE:
            demands[vertex_offset[left]] += 1
            demands[vertex_offset[right]] += 1

    remaining_incidence = [[0] * len(vertices) for _ in range(len(edges) + 1)]
    for edge_index in range(len(edges) - 1, -1, -1):
        remaining_incidence[edge_index] = remaining_incidence[edge_index + 1].copy()
        left, right = edges[edge_index]
        remaining_incidence[edge_index][vertex_offset[left]] += 1
        remaining_incidence[edge_index][vertex_offset[right]] += 1

    feasible_orders: list[tuple[int, ...]] = []
    current: list[int] = []

    def visit(edge_index: int, residual: tuple[int, ...]) -> None:
        if any(
            demand < 0 or demand > remaining_incidence[edge_index][offset]
            for offset, demand in enumerate(residual)
        ):
            return
        if edge_index == len(edges):
            if any(residual):
                return
            orders = tuple(current)
            component = AromaticComponentAssignments(edges=edges, bond_orders=(orders,))
            alias = _state_with_component_orders(state, (component,), (orders,))
            if (
                is_valid_state(alias)
                and is_connected_or_null(alias)
                and canonical_state_key(alias) == source_key
            ):
                feasible_orders.append(orders)
            return

        left, right = edges[edge_index]
        left_offset = vertex_offset[left]
        right_offset = vertex_offset[right]
        for order in (BOND_SINGLE, BOND_DOUBLE):
            next_residual = list(residual)
            if order == BOND_DOUBLE:
                next_residual[left_offset] -= 1
                next_residual[right_offset] -= 1
            current.append(order)
            visit(edge_index + 1, tuple(next_residual))
            current.pop()

    visit(0, tuple(demands))
    return AromaticComponentAssignments(
        edges=edges,
        bond_orders=tuple(sorted(set(feasible_orders))),
    )


def enumerate_component_factored_kekule_assignments(
    state: MolecularGraph,
) -> tuple[AromaticComponentAssignments, ...]:
    """Enumerate complete local assignments without a global Cartesian product.

    This function has no assignment cap. It is an experimental feasibility
    construction, not a production enumerator. It raises on invalid sources and
    returns an empty tuple when no aromatic component is present.
    """

    if not is_valid_state(state) or not is_connected_or_null(state):
        raise ValueError("component assignment enumeration requires a valid connected source")
    source_key = canonical_state_key(state)
    return tuple(
        _degree_constrained_component_orders(state, edges, source_key=source_key)
        for edges in _aromatic_edge_components(state)
    )


def _edge_is_bridge(state: MolecularGraph, edge: tuple[int, int]) -> bool:
    real_slots = tuple(int(slot) for slot in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real_slots)
    graph.add_edges_from(
        (left, right)
        for left_offset, left in enumerate(real_slots)
        for right in real_slots[left_offset + 1 :]
        if int(state.bonds[left, right]) != BOND_NULL
    )
    return frozenset(edge) in {
        frozenset((int(left), int(right))) for left, right in nx.bridges(graph)
    }


def _rejected(
    *,
    code: AromaticCycleOpenRejectionCode,
    edge: tuple[int, int],
    source_key: str | None,
    semantic_aromatic_edge: bool,
    prototype_status: str = PROTOTYPE_STATUS,
    enumerated_alias_count: int = 0,
    forced_single_alias_count: int = 0,
    executed_product_count: int = 0,
    canonical_product_keys: tuple[str, ...] = (),
) -> AromaticCycleOpenResolution:
    return AromaticCycleOpenResolution(
        prototype_status=prototype_status,
        admitted=False,
        rejection_code=code,
        source_key=source_key,
        edge=edge,
        semantic_aromatic_edge=semantic_aromatic_edge,
        enumerated_alias_count=enumerated_alias_count,
        forced_single_alias_count=forced_single_alias_count,
        executed_product_count=executed_product_count,
        canonical_product_keys=canonical_product_keys,
        successor=None,
        inverse_bond_order=None,
    )


def resolve_edge_anchored_cycle_open(
    state: MolecularGraph,
    action: BondDelete,
    *,
    maximum_aliases: int = 256,
    supplier_factory: ResonanceSupplierFactory = rdkit_kekule_supplier,
    enumeration: KekuleAliasEnumeration | None = None,
) -> AromaticCycleOpenResolution:
    """Resolve one cycle-edge deletion under the prototype semantic law.

    A caller auditing several edges of the same exact source may provide one
    previously completed enumeration. Its canonical source identity and full
    resonance-invariant aromatic edge set are revalidated before reuse.
    """

    edge = _normalize_edge(action)
    if not is_valid_state(state) or not is_connected_or_null(state):
        return _rejected(
            code=AromaticCycleOpenRejectionCode.INVALID_SOURCE,
            edge=edge,
            source_key=None,
            semantic_aromatic_edge=False,
        )
    source_key = canonical_state_key(state)
    if (
        edge[0] == edge[1]
        or edge[0] < 0
        or edge[1] >= state.n_atoms
        or not bool(is_element(np.asarray(state.atom_types[edge[0]])))
        or not bool(is_element(np.asarray(state.atom_types[edge[1]])))
    ):
        return _rejected(
            code=AromaticCycleOpenRejectionCode.INVALID_EDGE,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=False,
        )
    if int(state.bonds[edge]) == BOND_NULL:
        return _rejected(
            code=AromaticCycleOpenRejectionCode.EDGE_ABSENT,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=False,
        )
    perceived = resonance_invariant_bond_classes(state)
    semantic_aromatic = int(perceived[edge]) == BOND_AROMATIC
    if _edge_is_bridge(state, edge):
        return _rejected(
            code=AromaticCycleOpenRejectionCode.EDGE_IS_BRIDGE,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=semantic_aromatic,
        )

    normalized_action = BondDelete(*edge)
    if not semantic_aromatic:
        if not is_valid_bond_delete(state, normalized_action):
            return _rejected(
                code=AromaticCycleOpenRejectionCode.NONAROMATIC_EXECUTOR_REJECTED,
                edge=edge,
                source_key=source_key,
                semantic_aromatic_edge=False,
            )
        successor = apply_bond_delete(state, normalized_action)
        if not is_connected_or_null(successor):
            return _rejected(
                code=AromaticCycleOpenRejectionCode.NONAROMATIC_EXECUTOR_REJECTED,
                edge=edge,
                source_key=source_key,
                semantic_aromatic_edge=False,
            )
        return AromaticCycleOpenResolution(
            prototype_status=PROTOTYPE_STATUS,
            admitted=True,
            rejection_code=None,
            source_key=source_key,
            edge=edge,
            semantic_aromatic_edge=False,
            enumerated_alias_count=0,
            forced_single_alias_count=0,
            executed_product_count=1,
            canonical_product_keys=(canonical_state_key(successor),),
            successor=successor,
            inverse_bond_order=int(state.bonds[edge]),
        )

    perceived_aromatic_edges = tuple(
        (left, right)
        for left in range(state.n_atoms)
        for right in range(left + 1, state.n_atoms)
        if int(perceived[left, right]) == BOND_AROMATIC
    )
    if enumeration is None:
        enumeration = enumerate_charge_h_preserving_kekule_aliases(
            state,
            maximum_aliases=maximum_aliases,
            supplier_factory=supplier_factory,
        )
    elif (
        enumeration.source_key != source_key
        or enumeration.aromatic_edges != perceived_aromatic_edges
    ):
        raise ValueError("precomputed Kekule enumeration belongs to another semantic source")
    if not enumeration.aliases:
        return _rejected(
            code=AromaticCycleOpenRejectionCode.NO_PRESERVING_KEKULE_ALIAS,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
        )
    forced_single = tuple(
        alias for alias in enumeration.aliases if int(alias.bonds[edge]) == BOND_SINGLE
    )
    if not forced_single:
        return _rejected(
            code=AromaticCycleOpenRejectionCode.NO_FORCED_SINGLE_ALIAS,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
            enumerated_alias_count=len(enumeration.aliases),
        )

    products: list[MolecularGraph] = []
    for alias in forced_single:
        if not is_valid_bond_delete(alias, normalized_action):
            return _rejected(
                code=AromaticCycleOpenRejectionCode.FORCED_SINGLE_EXECUTION_DISAGREEMENT,
                edge=edge,
                source_key=source_key,
                semantic_aromatic_edge=True,
                enumerated_alias_count=len(enumeration.aliases),
                forced_single_alias_count=len(forced_single),
                executed_product_count=len(products),
            )
        successor = apply_bond_delete(alias, normalized_action)
        if not is_connected_or_null(successor):
            return _rejected(
                code=AromaticCycleOpenRejectionCode.FORCED_SINGLE_EXECUTION_DISAGREEMENT,
                edge=edge,
                source_key=source_key,
                semantic_aromatic_edge=True,
                enumerated_alias_count=len(enumeration.aliases),
                forced_single_alias_count=len(forced_single),
                executed_product_count=len(products),
            )
        products.append(successor)
    if not products:
        return _rejected(
            code=AromaticCycleOpenRejectionCode.EMPTY_CANONICAL_PRODUCT_GROUP,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
            enumerated_alias_count=len(enumeration.aliases),
            forced_single_alias_count=len(forced_single),
        )

    product_groups: dict[str, dict[tuple, MolecularGraph]] = {}
    for product in products:
        product_groups.setdefault(canonical_state_key(product), {}).setdefault(
            _exact_state_key(product),
            product,
        )
    canonical_product_keys = tuple(sorted(product_groups))
    if len(canonical_product_keys) != 1:
        return _rejected(
            code=AromaticCycleOpenRejectionCode.AMBIGUOUS_CANONICAL_PRODUCT,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
            enumerated_alias_count=len(enumeration.aliases),
            forced_single_alias_count=len(forced_single),
            executed_product_count=len(products),
            canonical_product_keys=canonical_product_keys,
        )
    representatives = product_groups[canonical_product_keys[0]]
    successor = representatives[min(representatives)]
    return AromaticCycleOpenResolution(
        prototype_status=PROTOTYPE_STATUS,
        admitted=True,
        rejection_code=None,
        source_key=source_key,
        edge=edge,
        semantic_aromatic_edge=True,
        enumerated_alias_count=len(enumeration.aliases),
        forced_single_alias_count=len(forced_single),
        executed_product_count=len(products),
        canonical_product_keys=canonical_product_keys,
        successor=successor,
        inverse_bond_order=BOND_SINGLE,
    )


def resolve_component_factored_cycle_open(
    state: MolecularGraph,
    action: BondDelete,
) -> AromaticCycleOpenResolution:
    """Resolve a cycle-edge deletion without whole-molecule alias products.

    The semantic law matches :func:`resolve_edge_anchored_cycle_open`, but each
    resonance-invariant aromatic component is solved independently as a
    complete degree-constrained matching problem. Exact global alias counts are
    products of local counts; the products themselves are never materialized.

    This remains a non-authorizing experiment. The exhaustive RDKit resolver is
    the bounded oracle, and production integration requires a separate support
    decision plus corpus rematerialization.
    """

    edge = _normalize_edge(action)
    status = COMPONENT_FACTORED_PROTOTYPE_STATUS
    if not is_valid_state(state) or not is_connected_or_null(state):
        return _rejected(
            code=AromaticCycleOpenRejectionCode.INVALID_SOURCE,
            edge=edge,
            source_key=None,
            semantic_aromatic_edge=False,
            prototype_status=status,
        )
    source_key = canonical_state_key(state)
    if (
        edge[0] == edge[1]
        or edge[0] < 0
        or edge[1] >= state.n_atoms
        or not bool(is_element(np.asarray(state.atom_types[edge[0]])))
        or not bool(is_element(np.asarray(state.atom_types[edge[1]])))
    ):
        return _rejected(
            code=AromaticCycleOpenRejectionCode.INVALID_EDGE,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=False,
            prototype_status=status,
        )
    if int(state.bonds[edge]) == BOND_NULL:
        return _rejected(
            code=AromaticCycleOpenRejectionCode.EDGE_ABSENT,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=False,
            prototype_status=status,
        )
    perceived = resonance_invariant_bond_classes(state)
    semantic_aromatic = int(perceived[edge]) == BOND_AROMATIC
    if _edge_is_bridge(state, edge):
        return _rejected(
            code=AromaticCycleOpenRejectionCode.EDGE_IS_BRIDGE,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=semantic_aromatic,
            prototype_status=status,
        )

    normalized_action = BondDelete(*edge)
    if not semantic_aromatic:
        if not is_valid_bond_delete(state, normalized_action):
            return _rejected(
                code=AromaticCycleOpenRejectionCode.NONAROMATIC_EXECUTOR_REJECTED,
                edge=edge,
                source_key=source_key,
                semantic_aromatic_edge=False,
                prototype_status=status,
            )
        successor = apply_bond_delete(state, normalized_action)
        if not is_connected_or_null(successor):
            return _rejected(
                code=AromaticCycleOpenRejectionCode.NONAROMATIC_EXECUTOR_REJECTED,
                edge=edge,
                source_key=source_key,
                semantic_aromatic_edge=False,
                prototype_status=status,
            )
        return AromaticCycleOpenResolution(
            prototype_status=status,
            admitted=True,
            rejection_code=None,
            source_key=source_key,
            edge=edge,
            semantic_aromatic_edge=False,
            enumerated_alias_count=0,
            forced_single_alias_count=0,
            executed_product_count=1,
            canonical_product_keys=(canonical_state_key(successor),),
            successor=successor,
            inverse_bond_order=int(state.bonds[edge]),
        )

    components = enumerate_component_factored_kekule_assignments(state)
    selected_indices = tuple(
        index for index, component in enumerate(components) if edge in component.edges
    )
    if len(selected_indices) != 1:
        return _rejected(
            code=AromaticCycleOpenRejectionCode.COMPONENT_FACTORIZATION_DISAGREEMENT,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
            prototype_status=status,
        )
    if any(not component.bond_orders for component in components):
        code = (
            AromaticCycleOpenRejectionCode.COMPONENT_CONTAINS_NON_KEKULE_BOND
            if any(
                int(state.bonds[component_edge]) not in {BOND_SINGLE, BOND_DOUBLE}
                for component in components
                for component_edge in component.edges
            )
            else AromaticCycleOpenRejectionCode.NO_PRESERVING_KEKULE_ALIAS
        )
        return _rejected(
            code=code,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
            prototype_status=status,
        )

    selected_index = selected_indices[0]
    selected_component = components[selected_index]
    selected_edge_offset = selected_component.edges.index(edge)
    selected_forced_single = tuple(
        orders
        for orders in selected_component.bond_orders
        if int(orders[selected_edge_offset]) == BOND_SINGLE
    )
    alias_count = prod(len(component.bond_orders) for component in components)
    irrelevant_alias_multiplier = prod(
        len(component.bond_orders)
        for index, component in enumerate(components)
        if index != selected_index
    )
    forced_single_count = len(selected_forced_single) * irrelevant_alias_multiplier
    if not selected_forced_single:
        return _rejected(
            code=AromaticCycleOpenRejectionCode.NO_FORCED_SINGLE_ALIAS,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
            prototype_status=status,
            enumerated_alias_count=alias_count,
        )

    lexicographic_component_orders = tuple(min(component.bond_orders) for component in components)
    products: list[MolecularGraph] = []
    for selected_orders in selected_forced_single:
        selections = list(lexicographic_component_orders)
        selections[selected_index] = selected_orders
        alias = _state_with_component_orders(state, components, tuple(selections))
        if (
            not is_valid_state(alias)
            or not is_connected_or_null(alias)
            or canonical_state_key(alias) != source_key
            or not is_valid_bond_delete(alias, normalized_action)
        ):
            return _rejected(
                code=AromaticCycleOpenRejectionCode.COMPONENT_FACTORIZATION_DISAGREEMENT,
                edge=edge,
                source_key=source_key,
                semantic_aromatic_edge=True,
                prototype_status=status,
                enumerated_alias_count=alias_count,
                forced_single_alias_count=forced_single_count,
                executed_product_count=len(products),
            )
        successor = apply_bond_delete(alias, normalized_action)
        if not is_connected_or_null(successor):
            return _rejected(
                code=AromaticCycleOpenRejectionCode.FORCED_SINGLE_EXECUTION_DISAGREEMENT,
                edge=edge,
                source_key=source_key,
                semantic_aromatic_edge=True,
                prototype_status=status,
                enumerated_alias_count=alias_count,
                forced_single_alias_count=forced_single_count,
                executed_product_count=len(products),
            )
        products.append(successor)

    product_groups: dict[str, dict[tuple, MolecularGraph]] = {}
    for product in products:
        product_groups.setdefault(canonical_state_key(product), {}).setdefault(
            _exact_state_key(product),
            product,
        )
    canonical_product_keys = tuple(sorted(product_groups))
    if not canonical_product_keys:
        return _rejected(
            code=AromaticCycleOpenRejectionCode.EMPTY_CANONICAL_PRODUCT_GROUP,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
            prototype_status=status,
            enumerated_alias_count=alias_count,
            forced_single_alias_count=forced_single_count,
        )
    if len(canonical_product_keys) != 1:
        return _rejected(
            code=AromaticCycleOpenRejectionCode.AMBIGUOUS_CANONICAL_PRODUCT,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
            prototype_status=status,
            enumerated_alias_count=alias_count,
            forced_single_alias_count=forced_single_count,
            executed_product_count=len(products),
            canonical_product_keys=canonical_product_keys,
        )
    representatives = product_groups[canonical_product_keys[0]]
    successor = representatives[min(representatives)]
    return AromaticCycleOpenResolution(
        prototype_status=status,
        admitted=True,
        rejection_code=None,
        source_key=source_key,
        edge=edge,
        semantic_aromatic_edge=True,
        enumerated_alias_count=alias_count,
        forced_single_alias_count=forced_single_count,
        # Only selected-component representatives are executed. The complete
        # global forced-single cardinality is reported separately above; do
        # not promote an implied Cartesian product into executed evidence.
        executed_product_count=len(products),
        canonical_product_keys=canonical_product_keys,
        successor=successor,
        inverse_bond_order=BOND_SINGLE,
    )


def inverse_mark_for_resolution(
    resolution: AromaticCycleOpenResolution,
) -> BondInsert:
    """Return the exact bond-insertion inverse declared by one admission."""

    if not resolution.admitted or resolution.inverse_bond_order is None:
        raise ValueError("a rejected cycle-open resolution has no inverse mark")
    return BondInsert(
        resolution.edge[0],
        resolution.edge[1],
        resolution.inverse_bond_order,
    )


__all__ = [
    "COMPONENT_FACTORED_PROTOTYPE_STATUS",
    "PROTOTYPE_STATUS",
    "AromaticComponentAssignments",
    "AromaticCycleOpenAliasOverflow",
    "AromaticCycleOpenRejectionCode",
    "AromaticCycleOpenResolution",
    "KekuleAliasEnumeration",
    "enumerate_charge_h_preserving_kekule_aliases",
    "enumerate_component_factored_kekule_assignments",
    "inverse_mark_for_resolution",
    "rdkit_kekule_supplier",
    "resolve_component_factored_cycle_open",
    "resolve_edge_anchored_cycle_open",
]
