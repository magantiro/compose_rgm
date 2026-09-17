"""Minimal intervention closures induced by the complete-patch grammar.

COMPOSE does not have a factorised action space, and that is a structural property rather
than a defect. Measured over 147 teacher subgoals: element, charge, hydrogen count, degree
and bond order can be resampled alone (0.810 and 1.000), while `output_count`, edge
presence and dependency parent NEVER decode alone. The reason is in the grammar --
`output_count` determines how many later tokens exist, and `edge_presence` determines
whether a `bond_order` token follows it -- so those decisions carry dependents.

The object this module builds is the MINIMAL CLOSURE. To intervene on a semantic decision
`z_j`, find the smallest set `C(j)` of decisions whose existence, legal domain or value
must change for the program to remain legal, resample exactly those, and preserve the
maximal compatible complement. A free coordinate is then a singleton closure and a coupled
coordinate is a larger one; both come from the same rule rather than from two hand-written
mechanisms.

    P' ~ q_ref( P' | G, do(z_C = z'_C), preserve maximal compatible P_\\C )

Two programs are only matched siblings if that preservation is VERIFIED, which
`closure_report` does by naming every semantic decision that moved. An unverified pair is
not a contrast: the parent-cancellation argument that makes a matched bundle identify a
coordinate effect requires everything else to be genuinely equal.

Interventions act on `StructuralSubgoal`, which already carries the structured record the
token stream encodes -- per-input survival and target signature, output signatures, and
the full target bond matrix -- so a closure is computed on the object and then validated
by round-tripping through `encode_patch_stream` / `decode_patch_stream`.
"""

from __future__ import annotations

from dataclasses import replace

from compose_v4.control.complete_region_patch_policy import (
    SourceRegionContext,
    decode_patch_stream,
    encode_patch_stream,
)
from compose_v4.control.structural_subgoal import StructuralSubgoal

SCHEMA_VERSION = "intervention_closure_v1"

# Semantic coordinates, named by the factor the patch grammar already tags tokens with.
FREE_COORDINATES = ("output_element", "bond_order")
BLOCK_COORDINATES = ("scale", "edge_presence")
COORDINATES = (*FREE_COORDINATES, *BLOCK_COORDINATES)


def _matrix(rows) -> list[list[int]]:
    return [[int(v) for v in row] for row in rows]


def _symmetric_set(matrix, i: int, j: int, order: int) -> None:
    matrix[i][j] = int(order)
    matrix[j][i] = int(order)


def intervene_element(subgoal: StructuralSubgoal, *, output_index: int, element: int):
    """Free coordinate: one created atom's element. Closure is the singleton itself."""
    atoms = list(subgoal.output_atoms)
    if not 0 <= output_index < len(atoms):
        raise ValueError("output index outside the patch")
    old = tuple(int(v) for v in atoms[output_index])
    atoms[output_index] = (int(element), old[1], old[2], old[3])
    return replace(subgoal, output_atoms=tuple(atoms)), {"output_atoms": [output_index]}


def intervene_bond_order(subgoal: StructuralSubgoal, *, left: int, right: int, order: int):
    """Free coordinate: the order of an existing bond. Closure is the singleton itself."""
    bonds = _matrix(subgoal.target_bonds)
    if not bonds[left][right]:
        raise ValueError("bond order intervention requires an existing bond")
    if order < 1:
        raise ValueError("removing a bond is an edge_presence intervention, not bond_order")
    _symmetric_set(bonds, left, right, order)
    return replace(subgoal, target_bonds=tuple(tuple(r) for r in bonds)), {
        "target_bonds": [(left, right)]
    }


def intervene_edge(
    subgoal: StructuralSubgoal, *, left: int, right: int, present: bool, order: int = 1
):
    """Coupled block: edge presence drags its bond order in or out with it."""
    bonds = _matrix(subgoal.target_bonds)
    _symmetric_set(bonds, left, right, order if present else 0)
    closure = {"target_bonds": [(left, right)]}
    if present:
        closure["bond_order_created"] = [(left, right)]
    else:
        closure["bond_order_removed"] = [(left, right)]
    return replace(subgoal, target_bonds=tuple(tuple(r) for r in bonds)), closure


def intervene_scale(subgoal: StructuralSubgoal, *, output_count: int, donor: int = -1):
    """Coupled block: the number of created atoms, with everything it drags.

    Growing copies a donor atom's signature and attaches the new role wherever the donor
    was attached, which is the smallest coherent extension: it introduces no element or
    topology choice that the original patch had not already made. Shrinking drops trailing
    output roles and their incident bonds. Every surviving decision -- input survival,
    retained target signatures, all bonds among surviving roles -- is preserved exactly.
    """
    n_input = len(subgoal.input_atoms)
    outputs = list(subgoal.output_atoms)
    current = len(outputs)
    if output_count < 0 or output_count == current:
        raise ValueError("scale intervention must change the created-atom count")
    if not outputs:
        raise ValueError("scale intervention needs at least one existing output role")
    donor_index = donor if donor >= 0 else current - 1

    old = _matrix(subgoal.target_bonds)
    size = n_input + output_count
    bonds = [[0 for _ in range(size)] for _ in range(size)]
    kept = min(current, output_count)
    # preserve every bond among roles that survive the resize
    for i in range(n_input + kept):
        for j in range(n_input + kept):
            bonds[i][j] = old[i][j]

    closure: dict[str, list] = {"output_count": [current, output_count]}
    if output_count > current:
        signature = tuple(int(v) for v in outputs[donor_index])
        donor_row = n_input + donor_index
        added = []
        for k in range(current, output_count):
            outputs.append(signature)
            new_row = n_input + k
            # attach the new role exactly where the donor is attached
            for j in range(n_input + current):
                if old[donor_row][j]:
                    _symmetric_set(bonds, new_row, j, old[donor_row][j])
            added.append(k)
        closure["output_atoms_added"] = added
    else:
        dropped = list(range(output_count, current))
        outputs = outputs[:output_count]
        closure["output_atoms_removed"] = dropped
        closure["bonds_removed_with_roles"] = [n_input + k for k in dropped]

    return replace(
        subgoal,
        output_atoms=tuple(outputs),
        target_bonds=tuple(tuple(row) for row in bonds),
    ), closure


def validate(original: StructuralSubgoal, candidate: StructuralSubgoal) -> dict:
    """Is the counterfactual a legal patch, and what actually moved?

    Legality is the patch invariants plus a lossless round trip through the token stream,
    which is the same representation the runtime decodes. Locality is reported as the
    set of semantic decisions that differ, so a caller can refuse any pair whose
    complement did not stay fixed.
    """
    report = {"legal": False, "round_trips": False, "changed": {}}
    context = SourceRegionContext.from_subgoal(candidate)
    try:
        tokens = encode_patch_stream(candidate)
        restored = decode_patch_stream(context, tokens)
    except (ValueError, IndexError, KeyError) as error:
        report["refusal"] = f"{type(error).__name__}: {error}"
        return report
    report["legal"] = True
    report["round_trips"] = restored == candidate
    report["stream_length"] = len(tokens)

    changed = {}
    if original.output_atoms != candidate.output_atoms:
        changed["output_atoms"] = [
            i
            for i, (a, b) in enumerate(zip(original.output_atoms, candidate.output_atoms))
            if a != b
        ] or ["count"]
    if original.target_atoms != candidate.target_atoms:
        changed["target_atoms"] = [
            i
            for i, (a, b) in enumerate(zip(original.target_atoms, candidate.target_atoms))
            if a != b
        ]
    if original.target_bonds != candidate.target_bonds:
        shared = min(len(original.target_bonds), len(candidate.target_bonds))
        changed["target_bonds"] = [
            (i, j)
            for i in range(shared)
            for j in range(i + 1, shared)
            if original.target_bonds[i][j] != candidate.target_bonds[i][j]
        ]
    # the source context must never move: it is the parent, and the whole contrast
    # argument depends on it cancelling
    if (
        original.input_atoms != candidate.input_atoms
        or original.environments != candidate.environments
    ):
        changed["SOURCE_CONTEXT_MOVED"] = True
    report["changed"] = changed
    report["complement_preserved"] = "SOURCE_CONTEXT_MOVED" not in changed
    return report
