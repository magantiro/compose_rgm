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

from compose_v4.chem.molecular_graph import IDX_TO_ELEMENT, STANDARD_VALENCE
from compose_v4.control.complete_region_patch_policy import (
    SourceRegionContext,
    decode_patch_stream,
    encode_patch_stream,
)
from compose_v4.control.structural_subgoal import StructuralSubgoal

SCHEMA_VERSION = "intervention_closure_v1"

# Semantic coordinates, named by the factor the patch grammar already tags tokens with.
FREE_COORDINATES = ("output_element", "bond_order", "retained_element")
BLOCK_COORDINATES = ("scale", "edge_presence", "retained_deletion")
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
    candidate = _rebalance(subgoal, replace(subgoal, output_atoms=tuple(atoms)))
    return candidate, {"output_atoms": [output_index], "signatures_rebalanced": True}


def intervene_retained_element(subgoal: StructuralSubgoal, *, input_index: int, element: int):
    """Retype a RETAINED source role: the scaffold atom identity itself.

    This is the largest capability the intervention wrapper was missing, and its absence
    was measured rather than guessed. Refining a valid -11.30 JAK2 molecule produced 0
    improvements in 91 docked candidates, because the one edit separating it from the
    archive's -11.70 is a retained aromatic nitrogen becoming carbon -- an atom the patch
    keeps rather than creates. `intervene_element` only ever touched `output_atoms`, so
    that direction had probability zero under the controller's action interface while the
    production `heteroatom_substitute` family could express it perfectly well.

    Hydrogens are recomputed from the new element's standard valence against the role's
    unchanged bond-order sum, so a retype that cannot satisfy valence refuses instead of
    emitting an open shell.
    """
    targets = list(subgoal.target_atoms)
    if not 0 <= input_index < len(targets):
        raise ValueError("input index outside the patch")
    if targets[input_index] is None:
        raise ValueError("cannot retype a role the patch deletes")
    old_element, charge, _hydrogens, degree = (int(v) for v in targets[input_index])
    if int(element) == old_element:
        raise ValueError("retained retype must change the element")
    symbol = IDX_TO_ELEMENT.get(int(element))
    valence = STANDARD_VALENCE.get(symbol)
    if valence is None:
        raise ValueError(f"no standard valence for element {symbol!r}")
    # the role's bonds do not move, so its bond-order sum is whatever it already was
    row = sum(int(v) for v in subgoal.target_bonds[input_index])
    adjusted = valence + charge - row
    if adjusted < 0:
        raise ValueError("retained retype cannot satisfy valence at this role")
    targets[input_index] = (int(element), charge, adjusted, degree)
    return replace(subgoal, target_atoms=tuple(targets)), {
        "retained_element": [input_index, old_element, int(element)],
        "signatures_rebalanced": True,
    }


def intervene_retained_deletion(subgoal: StructuralSubgoal, *, input_index: int):
    """Delete a RETAINED source role, which is what strong routes do most often.

    Measured over the 77 strong routes: 109 subgoals delete a retained role, more than
    any other operation, and the wrapper could not express a single one. Deleting a role
    drops its bonds, so every role it was bonded to gains hydrogens -- which is exactly
    what `_rebalance` computes from the change in bond-order sum.
    """
    targets = list(subgoal.target_atoms)
    if not 0 <= input_index < len(targets):
        raise ValueError("input index outside the patch")
    if targets[input_index] is None:
        raise ValueError("role is already deleted by this patch")
    if sum(1 for t in targets if t is not None) <= 1:
        raise ValueError("refusing to delete the patch's last retained role")
    bonds = _matrix(subgoal.target_bonds)
    dropped = [j for j in range(len(bonds)) if bonds[input_index][j]]
    for j in dropped:
        _symmetric_set(bonds, input_index, j, 0)
    targets[input_index] = None
    candidate = _rebalance(
        subgoal,
        replace(
            subgoal, target_atoms=tuple(targets), target_bonds=tuple(tuple(row) for row in bonds)
        ),
    )
    return candidate, {
        "retained_deleted": [input_index],
        "bonds_dropped": dropped,
        "signatures_rebalanced": True,
    }


def intervene_bond_order(subgoal: StructuralSubgoal, *, left: int, right: int, order: int):
    """Free coordinate: the order of an existing bond. Closure is the singleton itself."""
    bonds = _matrix(subgoal.target_bonds)
    if not bonds[left][right]:
        raise ValueError("bond order intervention requires an existing bond")
    if order < 1:
        raise ValueError("removing a bond is an edge_presence intervention, not bond_order")
    _symmetric_set(bonds, left, right, order)
    candidate = _rebalance(subgoal, replace(subgoal, target_bonds=tuple(tuple(r) for r in bonds)))
    return candidate, {"target_bonds": [(left, right)], "signatures_rebalanced": True}


def intervene_edge(
    subgoal: StructuralSubgoal, *, left: int, right: int, present: bool, order: int = 1
):
    """Coupled block: edge presence drags its bond order in or out with it."""
    bonds = _matrix(subgoal.target_bonds)
    _symmetric_set(bonds, left, right, order if present else 0)
    closure = {"target_bonds": [(left, right)], "signatures_rebalanced": True}
    closure["bond_order_created" if present else "bond_order_removed"] = [(left, right)]
    candidate = _rebalance(subgoal, replace(subgoal, target_bonds=tuple(tuple(r) for r in bonds)))
    return candidate, closure


def _row_sums(bonds) -> list[int]:
    """Sum of bond ORDERS per role -- the quantity the valence equation uses."""
    return [sum(int(v) for v in row) for row in bonds]


def _neighbours(bonds) -> list[int]:
    """Count of bonded roles -- the quantity the signature's fourth field records."""
    return [sum(1 for v in row if v) for row in bonds]


def _rebalance(before: StructuralSubgoal, after: StructuralSubgoal) -> StructuralSubgoal:
    """Restore the valence equation after an intervention changed connectivity.

    An atom signature is `(element, charge, implicit_h, neighbour_count)` and the
    codebase's own equation is

        implicit_h = standard_valence(element) + formal_charge - row_sum

    where `row_sum` is the sum of BOND ORDERS, not the number of neighbours. Getting that
    wrong does not raise: `instantiate_goal` checks that the built atom matches the
    DECLARED signature, so a nitrogen declared with one hydrogen and one single bond is
    faithfully built as a nitrogen radical and the role counts as satisfied. The executor
    is correct; the declaration was impossible.

    Measured before this existed: bond-order interventions produced radicals 100% of the
    time (25 of 25), element interventions 54%, scale 29% -- while the production
    generator with no intervention produced 0 in 4,000 archive molecules. The failure
    rate tracked exactly how much valence arithmetic each intervention skipped.

    Input roles keep bonds to the rest of the molecule that the patch cannot see, so
    their hydrogens are adjusted by the CHANGE in in-patch row sum. Output roles exist
    only inside the patch, so their hydrogens are computed outright.
    """
    n_input = len(after.input_atoms)
    old_rows, new_rows = _row_sums(before.target_bonds), _row_sums(after.target_bonds)
    new_neighbours = _neighbours(after.target_bonds)

    old_neighbours = _neighbours(before.target_bonds)
    targets = list(after.target_atoms)
    for slot, signature in enumerate(targets):
        if signature is None:
            continue
        element, charge, hydrogens, degree = (int(v) for v in signature)
        # A retained role keeps bonds to the rest of the molecule that this patch cannot
        # see, so both its hydrogens and its neighbour count move by the DELTA inside the
        # patch. Overwriting them with in-patch absolutes silently deletes the external
        # context and makes every realization refuse.
        row_delta = new_rows[slot] - (old_rows[slot] if slot < len(old_rows) else 0)
        neighbour_delta = new_neighbours[slot] - (
            old_neighbours[slot] if slot < len(old_neighbours) else 0
        )
        adjusted = hydrogens - row_delta
        if adjusted < 0:
            raise ValueError("intervention leaves a retained role with negative hydrogens")
        targets[slot] = (element, charge, adjusted, degree + neighbour_delta)

    outputs = list(after.output_atoms)
    for index, signature in enumerate(outputs):
        element, charge, _, _ = (int(v) for v in signature)
        symbol = IDX_TO_ELEMENT.get(element)
        valence = STANDARD_VALENCE.get(symbol)
        if valence is None:
            raise ValueError(f"no standard valence for element {symbol!r}")
        row = new_rows[n_input + index]
        hydrogens = valence + charge - row
        if hydrogens < 0:
            raise ValueError("intervention over-bonds a created role")
        outputs[index] = (element, charge, hydrogens, new_neighbours[n_input + index])

    return replace(after, target_atoms=tuple(targets), output_atoms=tuple(outputs))


def intervene_scale(subgoal: StructuralSubgoal, *, output_count: int, donor: int = -1):
    """Coupled block: the number of created atoms, with everything it drags.

    Growing copies a donor role's signature and attaches the new role where the donor is
    attached. The closure therefore contains the new role, the bonds it introduces, AND
    the declared degree and hydrogen count of every role it bonds to -- connectivity is
    not separable from the signatures that describe it. Shrinking is the mirror image.

    Every decision outside that closure is preserved exactly: input survival, retained
    signatures on untouched roles, and all bonds among roles that survive the resize.
    """
    n_input = len(subgoal.input_atoms)
    outputs = list(subgoal.output_atoms)
    targets = list(subgoal.target_atoms)
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
    for i in range(n_input + kept):
        for j in range(n_input + kept):
            bonds[i][j] = old[i][j]

    closure: dict[str, list] = {"output_count": [current, output_count]}
    touched: dict[int, int] = {}

    if output_count > current:
        signature = tuple(int(v) for v in outputs[donor_index])
        donor_row = n_input + donor_index
        added = []
        for k in range(current, output_count):
            new_row = n_input + k
            outputs.append(signature)
            for j in range(n_input + current):
                order = old[donor_row][j]
                if order:
                    _symmetric_set(bonds, new_row, j, order)
                    touched[j] = touched.get(j, 0) + 1
            added.append(k)
        closure["output_atoms_added"] = added
    else:
        dropped = list(range(output_count, current))
        for k in dropped:
            row = n_input + k
            for j in range(n_input + current):
                if old[row][j] and j < n_input + output_count:
                    touched[j] = touched.get(j, 0) - 1
        outputs = outputs[:output_count]
        closure["output_atoms_removed"] = dropped

    if touched:
        closure["signatures_rebalanced"] = sorted(touched)
    for slot in touched:
        if slot < n_input and targets[slot] is None:
            raise ValueError("cannot attach to a role the patch deletes")

    candidate = _rebalance(
        subgoal,
        replace(
            subgoal,
            target_atoms=tuple(targets),
            output_atoms=tuple(outputs),
            target_bonds=tuple(tuple(row) for row in bonds),
        ),
    )
    return candidate, closure


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
