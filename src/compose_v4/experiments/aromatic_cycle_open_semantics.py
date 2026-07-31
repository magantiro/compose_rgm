"""Non-authorizing prototype for resonance-invariant aromatic cycle opening.

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
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

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
    enumerated_alias_count: int = 0,
    forced_single_alias_count: int = 0,
    executed_product_count: int = 0,
    canonical_product_keys: tuple[str, ...] = (),
) -> AromaticCycleOpenResolution:
    return AromaticCycleOpenResolution(
        prototype_status=PROTOTYPE_STATUS,
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
) -> AromaticCycleOpenResolution:
    """Resolve one cycle-edge deletion under the prototype semantic law."""

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

    enumeration = enumerate_charge_h_preserving_kekule_aliases(
        state,
        maximum_aliases=maximum_aliases,
        supplier_factory=supplier_factory,
    )
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
    "PROTOTYPE_STATUS",
    "AromaticCycleOpenAliasOverflow",
    "AromaticCycleOpenRejectionCode",
    "AromaticCycleOpenResolution",
    "KekuleAliasEnumeration",
    "enumerate_charge_h_preserving_kekule_aliases",
    "inverse_mark_for_resolution",
    "rdkit_kekule_supplier",
    "resolve_edge_anchored_cycle_open",
]
