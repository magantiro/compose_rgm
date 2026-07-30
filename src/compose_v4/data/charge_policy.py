"""Authoritative transition-level charge-preservation contract for editing data.

Charged molecules are part of the broad-organic state space, but formal-charge
design is not part of the current editing action space. A legal training path
therefore has to:

* preserve the complete persistent-slot formal-charge array; and
* protect every initially charged slot's element, implicit hydrogens, and full
  bond row.

The second condition is intentionally stronger than net-charge preservation.
It prevents a formally charged center from being deleted, moved, retyped, or
having its local bonding changed while a compensating charge elsewhere hides
the edit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph

CHARGE_POLICY_VERSION = "exact_charged_center_preservation_v1"

FORMAL_CHARGE_ARRAY_CHANGED = "formal_charge_array_changed"
CHARGED_CENTER_ELEMENT_CHANGED = "charged_center_element_changed"
CHARGED_CENTER_H_CHANGED = "charged_center_implicit_h_changed"
CHARGED_CENTER_BOND_ROW_CHANGED = "charged_center_bond_row_changed"

CHARGE_POLICY_VIOLATION_TYPES = (
    FORMAL_CHARGE_ARRAY_CHANGED,
    CHARGED_CENTER_ELEMENT_CHANGED,
    CHARGED_CENTER_H_CHANGED,
    CHARGED_CENTER_BOND_ROW_CHANGED,
)


@dataclass(frozen=True)
class ChargePolicyTransitionAudit:
    """Exact slot-level differences relevant to the editing charge policy."""

    formal_charge_changed_slots: tuple[int, ...]
    formal_charge_created_slots: tuple[int, ...]
    formal_charge_deleted_slots: tuple[int, ...]
    formal_charge_value_changed_slots: tuple[int, ...]
    charged_center_element_changed_slots: tuple[int, ...]
    charged_center_h_changed_slots: tuple[int, ...]
    charged_center_bond_row_changed_slots: tuple[int, ...]

    @property
    def violation_types(self) -> tuple[str, ...]:
        """Violation labels in one stable, versioned order."""

        out: list[str] = []
        if self.formal_charge_changed_slots:
            out.append(FORMAL_CHARGE_ARRAY_CHANGED)
        if self.charged_center_element_changed_slots:
            out.append(CHARGED_CENTER_ELEMENT_CHANGED)
        if self.charged_center_h_changed_slots:
            out.append(CHARGED_CENTER_H_CHANGED)
        if self.charged_center_bond_row_changed_slots:
            out.append(CHARGED_CENTER_BOND_ROW_CHANGED)
        return tuple(out)

    @property
    def preserved(self) -> bool:
        return not self.violation_types

    @property
    def formal_charge_coordinate_mutated(self) -> bool:
        return bool(self.formal_charge_changed_slots)

    @property
    def protected_charged_center_mutated(self) -> bool:
        return bool(
            self.charged_center_element_changed_slots
            or self.charged_center_h_changed_slots
            or self.charged_center_bond_row_changed_slots
        )

    def to_json(self) -> dict[str, object]:
        return {
            "violation_types": list(self.violation_types),
            "formal_charge_changed_slots": list(self.formal_charge_changed_slots),
            "formal_charge_created_slots": list(self.formal_charge_created_slots),
            "formal_charge_deleted_slots": list(self.formal_charge_deleted_slots),
            "formal_charge_value_changed_slots": list(
                self.formal_charge_value_changed_slots
            ),
            "charged_center_element_changed_slots": list(
                self.charged_center_element_changed_slots
            ),
            "charged_center_implicit_h_changed_slots": list(
                self.charged_center_h_changed_slots
            ),
            "charged_center_bond_row_changed_slots": list(
                self.charged_center_bond_row_changed_slots
            ),
        }


def audit_charge_policy_transition(
    source: MolecularGraph,
    successor: MolecularGraph,
) -> ChargePolicyTransitionAudit:
    """Audit one exact consecutive-state pair without canonicalization or replay."""

    if not isinstance(source, MolecularGraph) or not isinstance(
        successor, MolecularGraph
    ):
        raise TypeError("charge-policy audit requires MolecularGraph states")
    if source.n_atoms != successor.n_atoms:
        raise ValueError(
            "charge-policy audit requires equal persistent-slot capacity: "
            f"{source.n_atoms} != {successor.n_atoms}"
        )

    formal_charge_changed = tuple(
        int(slot)
        for slot in np.flatnonzero(source.formal_charges != successor.formal_charges)
    )
    charge_created = tuple(
        slot
        for slot in formal_charge_changed
        if int(source.formal_charges[slot]) == 0
        and int(successor.formal_charges[slot]) != 0
    )
    charge_deleted = tuple(
        slot
        for slot in formal_charge_changed
        if int(source.formal_charges[slot]) != 0
        and int(successor.formal_charges[slot]) == 0
    )
    charge_value_changed = tuple(
        slot
        for slot in formal_charge_changed
        if int(source.formal_charges[slot]) != 0
        and int(successor.formal_charges[slot]) != 0
    )
    charged_slots = np.flatnonzero(source.formal_charges != 0)
    element_changed = tuple(
        int(slot)
        for slot in charged_slots
        if int(source.atom_types[slot]) != int(successor.atom_types[slot])
    )
    h_changed = tuple(
        int(slot)
        for slot in charged_slots
        if int(source.implicit_h_counts[slot])
        != int(successor.implicit_h_counts[slot])
    )
    bond_row_changed = tuple(
        int(slot)
        for slot in charged_slots
        if not np.array_equal(source.bonds[slot], successor.bonds[slot])
    )
    return ChargePolicyTransitionAudit(
        formal_charge_changed_slots=formal_charge_changed,
        formal_charge_created_slots=charge_created,
        formal_charge_deleted_slots=charge_deleted,
        formal_charge_value_changed_slots=charge_value_changed,
        charged_center_element_changed_slots=element_changed,
        charged_center_h_changed_slots=h_changed,
        charged_center_bond_row_changed_slots=bond_row_changed,
    )


def charge_policy_preserved(
    source: MolecularGraph,
    successor: MolecularGraph,
) -> bool:
    """Return the exact predicate used to admit a compiled editing transition."""

    return audit_charge_policy_transition(source, successor).preserved


__all__ = [
    "CHARGE_POLICY_VERSION",
    "CHARGE_POLICY_VIOLATION_TYPES",
    "CHARGED_CENTER_BOND_ROW_CHANGED",
    "CHARGED_CENTER_ELEMENT_CHANGED",
    "CHARGED_CENTER_H_CHANGED",
    "FORMAL_CHARGE_ARRAY_CHANGED",
    "ChargePolicyTransitionAudit",
    "audit_charge_policy_transition",
    "charge_policy_preserved",
]
