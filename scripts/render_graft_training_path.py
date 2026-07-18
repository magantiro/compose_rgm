"""Render an exact compiled carbon-tree-to-target Graft training path."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw
import torch

from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_NULL,
    BOND_SINGLE,
    BOND_TRIPLE,
    IDX_TO_ELEMENT,
    NULL_IDX,
)


_BOND_TYPES = {
    BOND_SINGLE: Chem.BondType.SINGLE,
    BOND_DOUBLE: Chem.BondType.DOUBLE,
    BOND_TRIPLE: Chem.BondType.TRIPLE,
    BOND_AROMATIC: Chem.BondType.AROMATIC,
}


def _to_labeled_molecule(state) -> tuple[Chem.Mol, dict[int, int]]:
    rw = Chem.RWMol()
    slot_to_atom = {}
    aromatic = (state.bonds == BOND_AROMATIC).any(axis=1)
    for slot in range(state.n_atoms):
        atom_type = int(state.atom_types[slot])
        if atom_type == NULL_IDX:
            continue
        atom = Chem.Atom(IDX_TO_ELEMENT[atom_type])
        atom.SetFormalCharge(int(state.formal_charges[slot]))
        atom.SetNumExplicitHs(int(state.implicit_h_counts[slot]))
        atom.SetNoImplicit(True)
        atom.SetProp("atomNote", str(slot))
        atom.SetIsAromatic(bool(aromatic[slot]))
        slot_to_atom[slot] = rw.AddAtom(atom)
    for left in range(state.n_atoms):
        if left not in slot_to_atom:
            continue
        for right in range(left + 1, state.n_atoms):
            bond_class = int(state.bonds[left, right])
            if bond_class == BOND_NULL:
                continue
            rw.AddBond(
                slot_to_atom[left],
                slot_to_atom[right],
                _BOND_TYPES[bond_class],
            )
    molecule = rw.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule, slot_to_atom


def _changed_slots(previous, current) -> tuple[set[int], set[tuple[int, int]]]:
    atom_changed = set(
        int(value)
        for value in np.flatnonzero(
            (previous.atom_types != current.atom_types)
            | (previous.formal_charges != current.formal_charges)
            | (previous.implicit_h_counts != current.implicit_h_counts)
        )
    )
    bond_changed = set()
    for left, right in zip(*np.where(previous.bonds != current.bonds)):
        if int(left) >= int(right):
            continue
        bond_changed.add((int(left), int(right)))
        atom_changed.update((int(left), int(right)))
    return atom_changed, bond_changed


def _action_label(step) -> str:
    action = step.action
    if step.rule_name == "bond_reroute":
        return (
            f"Graft: cut {action.a}-{action.b}; "
            f"add {action.u}-{action.v}"
        )
    if step.rule_name == "bond_reorder":
        return f"Bond order: {action.a}-{action.b} -> {action.new_order}"
    if step.rule_name == "atom_restate":
        return f"Retype slot {action.v} -> {IDX_TO_ELEMENT[action.atom_type]}"
    if step.rule_name == "ring_ear_insert":
        if action.atoms:
            return f"Coordinated ring grow: {action.a} -> {action.b}"
        return f"Ring closure: add {action.a}-{action.b}"
    return step.rule_name.replace("_", " ").title()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--partition", default="train")
    parser.add_argument("--record-index", type=int, default=1)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    payload = torch.load(args.path_cache, map_location="cpu", weights_only=False)
    records = tuple(payload[f"{args.partition}_records"])
    record = records[args.record_index]
    states = tuple(record.path.iter_states())

    molecules = []
    legends = []
    highlight_atoms = []
    highlight_bonds = []
    for progress, state in enumerate(states):
        molecule, slot_to_atom = _to_labeled_molecule(state)
        molecules.append(molecule)
        if progress == 0:
            legends.append("0. RANDOM CARBON TREE PRIOR")
            highlight_atoms.append([])
            highlight_bonds.append([])
            continue
        step = record.path.trace.steps[progress - 1]
        atoms, bonds = _changed_slots(states[progress - 1], state)
        highlight_atoms.append(
            [slot_to_atom[slot] for slot in sorted(atoms) if slot in slot_to_atom]
        )
        visible_bonds = []
        for left, right in sorted(bonds):
            if left not in slot_to_atom or right not in slot_to_atom:
                continue
            bond = molecule.GetBondBetweenAtoms(
                slot_to_atom[left],
                slot_to_atom[right],
            )
            if bond is not None:
                visible_bonds.append(bond.GetIdx())
        highlight_bonds.append(visible_bonds)
        suffix = "\nTARGET MOLECULE" if progress == len(states) - 1 else ""
        legends.append(f"{progress}. {_action_label(step)}{suffix}")

    grid = Draw.MolsToGridImage(
        molecules,
        molsPerRow=args.columns,
        subImgSize=(460, 310),
        legends=legends,
        highlightAtomLists=highlight_atoms,
        highlightBondLists=highlight_bonds,
        useSVG=False,
    )
    title_height = 90
    canvas = Image.new("RGB", (grid.width, grid.height + title_height), "white")
    canvas.paste(grid, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        note_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 19)
    except OSError:
        title_font = ImageFont.load_default()
        note_font = ImageFont.load_default()
    draw.text(
        (24, 14),
        "Actual compiled Graft training path",
        fill="black",
        font=title_font,
    )
    draw.text(
        (24, 53),
        "Slot labels track atoms; highlighted atoms/bonds changed in that step.",
        fill="black",
        font=note_font,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
