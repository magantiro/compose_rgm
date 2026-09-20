"""Address-free primitive roles shared by PMO training and runtime binding.

The role object deliberately retains executor semantics and relative created-handle
dependencies while removing persistent slot addresses.  It is training/runtime
infrastructure, not a task or endpoint label.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from compose_v4.chem.molecular_graph import ELEMENTS, is_element

ACTION_ROLE_SCHEMA = "generic_primitive_action_role_v1"


def atom_role(
    graph, address: int, created: dict[int, tuple[int, int]], step: int
) -> dict[str, Any]:
    """Describe one operand without retaining its persistent address."""

    if not 0 <= address < graph.n_atoms or not bool(is_element(graph.atom_types)[address]):
        raise ValueError("action operand is not an active atom")
    bonds = np.asarray(graph.bonds[address], dtype=int)
    neighbors = np.flatnonzero(bonds)
    element_histogram = [0] * len(ELEMENTS)
    bond_histogram = [0] * 4
    for neighbor in neighbors:
        element_histogram[int(graph.atom_types[neighbor])] += 1
        order = int(bonds[neighbor])
        if not 1 <= order <= 4:
            raise ValueError("action operand has an invalid bond class")
        bond_histogram[order - 1] += 1
    result: dict[str, Any] = {
        "origin": "route_created" if address in created else "preexisting",
        "atom_type": int(graph.atom_types[address]),
        "formal_charge": int(graph.formal_charges[address]),
        "implicit_hydrogens": int(graph.implicit_h_counts[address]),
        "degree": len(neighbors),
        "bond_class_histogram": bond_histogram,
        "neighbor_element_histogram": element_histogram,
    }
    if address in created:
        ordinal, created_at = created[address]
        result.update(
            {
                "created_ordinal": ordinal,
                "creation_lag": step - created_at,
            }
        )
    return result


def action_role_supervision(
    graph,
    record: dict[str, Any],
    created: dict[int, tuple[int, int]],
    step: int,
    next_ordinal: int,
) -> tuple[dict[str, Any], int]:
    """Remove addresses while preserving action parameters and dependencies."""

    rule = str(record["executor_rule"])
    payload = record["payload"]

    def operand(name: str, address: int) -> dict[str, Any]:
        return {
            "role": name,
            "descriptor": atom_role(graph, int(address), created, step),
        }

    operands: list[dict[str, Any]] = []
    parameters: dict[str, Any] = {}
    created_output = None
    deleted = None
    if rule == "atom_insert":
        parameters = {
            "atom_type": int(payload["atom_type"]),
            "formal_charge": int(payload["formal_charge"]),
            "implicit_hydrogens": int(payload["implicit_h_count"]),
            "neighbor_bond_classes": [int(row[1]) for row in payload["neighbors"]],
        }
        operands = [
            operand(f"neighbor_{index}", address)
            for index, (address, _) in enumerate(payload["neighbors"])
        ]
        created_output = next_ordinal
        created[int(payload["slot"])] = (next_ordinal, step)
        next_ordinal += 1
    elif rule in ("atom_delete", "atom_restate_semantic"):
        address = int(payload["v"])
        operands = [operand("atom", address)]
        if rule == "atom_restate_semantic":
            parameters = {"target_class_index": int(payload["target_class_index"])}
        else:
            deleted = address
    elif rule in ("cycle_close", "cycle_open", "bond_reorder"):
        operands = [
            operand("endpoint_a", payload["a"]),
            operand("endpoint_b", payload["b"]),
        ]
        if rule == "cycle_close":
            parameters = {"bond_class": int(payload["order"])}
        elif rule == "bond_reorder":
            parameters = {"new_bond_class": int(payload["new_order"])}
    elif rule == "bond_reroute":
        operands = [operand(name, payload[name]) for name in ("a", "b", "u", "v")]
    elif rule == "ring_system_restate":
        parameters = {"new_bond_classes": [int(row["new_order"]) for row in payload["changes"]]}
        for index, change in enumerate(payload["changes"]):
            operands.extend(
                (
                    operand(f"change_{index}_a", change["a"]),
                    operand(f"change_{index}_b", change["b"]),
                )
            )
    else:
        raise ValueError(f"unsupported generic primitive rule: {rule}")
    dependencies = [
        {
            "operand": row["role"],
            "created_ordinal": row["descriptor"]["created_ordinal"],
            "creation_lag": row["descriptor"]["creation_lag"],
        }
        for row in operands
        if row["descriptor"]["origin"] == "route_created"
    ]
    result = {
        "schema_version": ACTION_ROLE_SCHEMA,
        "executor_rule": rule,
        "model_family": str(record["model_family"]),
        "parameters": parameters,
        "operands": operands,
        "created_handle_dependencies": dependencies,
        "created_output_ordinal": created_output,
    }
    if deleted is not None:
        created.pop(deleted, None)
    return result, next_ordinal


__all__ = ["ACTION_ROLE_SCHEMA", "action_role_supervision", "atom_role"]
